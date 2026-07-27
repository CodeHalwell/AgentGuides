---
title: "LangGraph Class Deep-Dives Vol. 43"
description: "Source-verified deep dives (langgraph==1.2.9) into 10 class groups: ToolNode full API (handle_tool_errors five forms, wrap_tool_call/awrap_tool_call middleware, messages_key custom state key), InjectedState/InjectedStore/_InjectedArgs (injection map built at __init__, field slicing vs full-state injection, invisible-to-model args), tools_condition (last-AIMessage inspection, messages_key param, list/dict/BaseModel dispatch), SyncAsyncFuture (concurrent.futures.Future subclass, __await__ generator yield, parallel @task fan-out), ChannelWrite/ChannelWriteEntry/ChannelWriteTupleEntry (PASSTHROUGH sentinel, skip_none, mapper transform, do_write static API, get_static_writes topology), StateSnapshot+PregelTask (tasks field task-level error/result/state, parent_config history walking, subgraph snapshot via PregelTask.state), Pregel.get_state_history()/update_state()/bulk_update_state() (newest-first iterator, filter/before/limit, StateUpdate for seeding), InMemoryStore+IndexConfig+SearchItem (cosine-similarity vector search, fields=[] embedding scope, SearchItem.score, filter operators), add_messages advanced patterns (format='langchain-openai' OpenAI-role conversion, ID dedup on human override, REMOVE_ALL_MESSAGES wipe, push_message dual write), and RunControl/GraphDrained/Runtime.drain_requested (cooperative drain signal, request_drain(reason), drain_requested property, GraphDrained bubble-up, heartbeat interaction)."
framework: langgraph
language: python
sidebar:
  label: "Class deep-dives Vol. 43"
  order: 74
---

Source-verified deep dives into **10 class groups**, each with **3 runnable examples**, verified against `langgraph==1.2.9` / `langgraph-checkpoint==4.1.1` / `langgraph-prebuilt==1.1.0`.

---

## 1 · `ToolNode` — full API

**Module:** `langgraph.prebuilt.tool_node`

`ToolNode` is the most widely used prebuilt node in LangGraph. It executes tool calls found in the last `AIMessage`, handles parallel execution, manages error recovery, and supports middleware interception via `wrap_tool_call`. It is a `RunnableCallable` subclass that builds its injection map once during `__init__` by calling `_get_all_injected_args` for every tool.

**Key source facts (`langgraph/prebuilt/tool_node.py`):**

- Constructor accepts `tools: Sequence[BaseTool | Callable]` — plain callables are converted via `create_tool()`.
- `handle_tool_errors` has **five** forms: `True` (default error template), `str` (fixed message), `type[Exception]` (filter single type), `tuple[type[Exception], ...]` (filter multiple), `Callable[..., str]` (custom formatter). `False` disables all catching.
- Default `handle_tool_errors` is `_default_handle_tool_errors`: catches `ToolInvocationError` (bad args) but re-raises tool-execution errors so they bubble to the error-handler node.
- `wrap_tool_call: ToolCallWrapper | None` — sync middleware receiving `(ToolCallRequest, execute_callable)`. `awrap_tool_call` is the async equivalent; falls back to sync wrapper when absent.
- `messages_key: str = "messages"` — key in the state dict that holds messages and receives `ToolMessage` outputs.
- `_injected_args` dict maps tool name → `_InjectedArgs` dataclass. Built once, reused per call.

### Example 1 — five `handle_tool_errors` forms

```python
from typing import Callable
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage

@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Division by zero")
    return a / b

# Form 1: True — default error template
node_true = ToolNode([divide], handle_tool_errors=True)

# Form 2: str — fixed message
node_str = ToolNode([divide], handle_tool_errors="Tool failed. Try different inputs.")

# Form 3: type[Exception] — only catch ValueError
node_type = ToolNode([divide], handle_tool_errors=ValueError)

# Form 4: tuple — catch multiple exception types
node_tuple = ToolNode([divide], handle_tool_errors=(ValueError, ZeroDivisionError))

# Form 5: Callable — custom formatter
def fmt(e: Exception) -> str:
    return f"[{type(e).__name__}] {e}"

node_callable = ToolNode([divide], handle_tool_errors=fmt)

bad_call = {"name": "divide", "args": {"a": 1.0, "b": 0.0}, "id": "tc1", "type": "tool_call"}
state = {"messages": [AIMessage("", tool_calls=[bad_call])]}

# Each node returns a ToolMessage with the error content set by its strategy
for label, node in [("True", node_true), ("str", node_str), ("Callable", node_callable)]:
    result = node.invoke(state)
    msg = result["messages"][0]
    print(f"[{label}] {msg.content[:60]}")
```

### Example 2 — `wrap_tool_call` middleware: logging + retry

```python
import time
from typing import Callable
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest

@tool
def fetch_price(ticker: str) -> str:
    """Fetch stock price for a ticker."""
    prices = {"AAPL": "193.42", "GOOG": "175.10"}
    if ticker not in prices:
        raise KeyError(f"Unknown ticker: {ticker}")
    return prices[ticker]

def log_and_time(
    request: ToolCallRequest,
    execute: Callable,
) -> ToolMessage:
    """Middleware that logs elapsed time and overrides the ticker."""
    # Override: normalise ticker to uppercase
    overridden = request.override(
        tool_call={"name": request.tool_call["name"],
                   "args": {"ticker": request.tool_call["args"]["ticker"].upper()},
                   "id": request.tool_call["id"],
                   "type": "tool_call"}
    )
    t0 = time.monotonic()
    result = execute(overridden)
    elapsed = time.monotonic() - t0
    print(f"Tool {request.tool_call['name']} took {elapsed*1000:.1f}ms")
    return result

node = ToolNode([fetch_price], wrap_tool_call=log_and_time, handle_tool_errors=True)

tc = {"name": "fetch_price", "args": {"ticker": "aapl"}, "id": "t1", "type": "tool_call"}
result = node.invoke({"messages": [AIMessage("", tool_calls=[tc])]})
print(result["messages"][0].content)  # 193.42
```

### Example 3 — `messages_key` for custom state schemas

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class ChatState(TypedDict):
    chat_history: Annotated[list[BaseMessage], add_messages]
    context: str

@tool
def summarize(text: str) -> str:
    """Summarize text."""
    return f"Summary: {text[:30]}..."

# Use messages_key="chat_history" to match custom state key
tool_node = ToolNode([summarize], messages_key="chat_history")

def call_model(state: ChatState) -> dict:
    tc = {"name": "summarize", "args": {"text": state["context"]}, "id": "s1", "type": "tool_call"}
    return {"chat_history": [AIMessage("Summarizing...", tool_calls=[tc])]}

builder = StateGraph(ChatState)
builder.add_node("model", call_model)
builder.add_node("tools", tool_node)
builder.add_edge(START, "model")
builder.add_conditional_edges(
    "model",
    lambda s: tools_condition(s, messages_key="chat_history"),
)
builder.add_edge("tools", END)
graph = builder.compile()

result = graph.invoke({"chat_history": [], "context": "LangGraph is a framework for building stateful agents."})
print(result["chat_history"][-1].content)
```

---

## 2 · `InjectedState` · `InjectedStore` · `_InjectedArgs`

**Module:** `langgraph.prebuilt.tool_node`

These three classes form LangGraph's tool-injection system. `InjectedState` and `InjectedStore` are annotation markers that are **invisible to the LLM** — they are stripped from the tool's JSON schema. `_InjectedArgs` is the internal dataclass that `ToolNode` builds once per tool during `__init__` via `_get_all_injected_args`, recording which parameters to fill from state (by field key or entire dict), which from the store, and which from the `ToolRuntime`.

**Key source facts:**

- `InjectedState(field=None)` — injects entire state dict. `InjectedState("messages")` injects `state["messages"]` only.
- `InjectedStore()` — injects the `BaseStore` instance compiled into the graph; `None` if no store was compiled.
- `_InjectedArgs.state: dict[str, str | None]` — maps parameter name to field key (or `None` for full state).
- `_InjectedArgs.store: str | None` — parameter name for the store.
- `_InjectedArgs.runtime: str | None` — parameter name for `ToolRuntime` injection.
- Injected parameters do not appear in `tool.args_schema` so the LLM never generates values for them.
- `_InjectedArgs` is built once at `ToolNode.__init__` by scanning `tool.get_input_schema()` annotations, not on every invocation.

### Example 1 — `InjectedState` full state vs field slice

```python
from typing import Annotated, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

class AppState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_name: str
    session_count: int

@tool
def greet(greeting: str, full_state: Annotated[dict, InjectedState()]) -> str:
    """Greet using full state context."""
    from langgraph.prebuilt.tool_node import InjectedState  # re-import for clarity
    return f"{greeting}, {full_state['user_name']}! Session #{full_state['session_count']}"

@tool
def count_messages(name: Annotated[str, InjectedState("user_name")]) -> str:
    """Return the user name (injected from state field)."""
    from langgraph.prebuilt.tool_node import InjectedState  # re-import for clarity
    return f"User: {name}"
```

```python
from langgraph.prebuilt.tool_node import InjectedState

class AppState2(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_name: str
    session_count: int

@tool
def greet_user(greeting: str, name: Annotated[str, InjectedState("user_name")]) -> str:
    """Greet the user with their name injected from state."""
    return f"{greeting}, {name}!"

@tool
def show_session(info: Annotated[dict, InjectedState()]) -> str:
    """Show full session info injected from entire state."""
    return f"User={info['user_name']}, session={info['session_count']}"

node = ToolNode([greet_user, show_session])

tc1 = {"name": "greet_user", "args": {"greeting": "Hello"}, "id": "g1", "type": "tool_call"}
tc2 = {"name": "show_session", "args": {}, "id": "g2", "type": "tool_call"}
state = {
    "messages": [AIMessage("", tool_calls=[tc1, tc2])],
    "user_name": "Alice",
    "session_count": 7,
}
result = node.invoke(state)
for msg in result["messages"]:
    print(msg.content)
# Hello, Alice!
# User=Alice, session=7
```

### Example 2 — `InjectedStore` for persistent memory across turns

```python
from typing import Annotated, Any
from langgraph.prebuilt.tool_node import InjectedStore
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import AIMessage

store = InMemoryStore()

@tool
def remember(fact: str, key: str, store: Annotated[Any, InjectedStore()]) -> str:
    """Store a fact in persistent memory."""
    store.put(("facts",), key, {"fact": fact})
    return f"Remembered: {key} = {fact}"

@tool
def recall(key: str, store: Annotated[Any, InjectedStore()]) -> str:
    """Recall a fact from persistent memory."""
    item = store.get(("facts",), key)
    return item.value["fact"] if item else "Not found"

node = ToolNode([remember, recall])

tc_put = {"name": "remember", "args": {"fact": "Python 3.14 released", "key": "py_news"}, "id": "r1", "type": "tool_call"}
state = {"messages": [AIMessage("", tool_calls=[tc_put])]}
# Inject store via configurable — ToolNode reads it from config["configurable"]["store"]
from langchain_core.runnables import RunnableConfig
config = RunnableConfig(configurable={"store": store})
result = node.invoke(state, config=config)
print(result["messages"][0].content)  # Remembered: py_news = Python 3.14 released

tc_get = {"name": "recall", "args": {"key": "py_news"}, "id": "r2", "type": "tool_call"}
state2 = {"messages": [AIMessage("", tool_calls=[tc_get])]}
result2 = node.invoke(state2, config=config)
print(result2["messages"][0].content)  # Python 3.14 released
```

### Example 3 — inspecting `_InjectedArgs` built by `ToolNode`

```python
from langgraph.prebuilt.tool_node import InjectedState, InjectedStore, ToolNode, _get_all_injected_args
from langchain_core.tools import tool
from typing import Annotated, Any

@tool
def complex_tool(
    query: str,
    messages: Annotated[list, InjectedState("messages")],
    full_state: Annotated[dict, InjectedState()],
    store: Annotated[Any, InjectedStore()],
) -> str:
    """Tool with multiple injection types."""
    return f"Got {len(messages)} messages"

# Inspect the _InjectedArgs structure built at ToolNode init time
injected = _get_all_injected_args(complex_tool)
print("State injections:", injected.state)
# {'messages': 'messages', 'full_state': None}

print("Store param:", injected.store)
# 'store'

print("All injected keys:", injected.all_injected_keys)
# {'messages', 'full_state', 'store'}

# The LLM only sees "query" in the tool schema — injected args are hidden:
schema = complex_tool.get_input_schema().model_json_schema()
print("LLM-visible params:", list(schema.get("properties", {}).keys()))
# ['query']
```

---

## 3 · `tools_condition`

**Module:** `langgraph.prebuilt.tool_node`

`tools_condition` is the standard conditional edge function for ReAct-style graphs. It inspects the **last message** in the state: if it is an `AIMessage` with non-empty `tool_calls`, it returns `"tools"`; otherwise `"__end__"`. It accepts three state shapes: `list[AnyMessage]`, `dict` (with configurable `messages_key`), or a `BaseModel` instance.

**Key source facts:**

- `isinstance(state, list)` path: `ai_message = state[-1]` directly.
- `dict` path: `messages = state.get(messages_key, [])` with `messages_key` defaulting to `"messages"`.
- `BaseModel` path: `messages = getattr(state, messages_key, [])`.
- Returns `Literal["tools", "__end__"]` — these match the canonical node names expected by `ToolNode`.
- Raises `ValueError` if `messages_key` is missing from the state.
- Can be wrapped to return custom names: `lambda s: {"tools": "my_tools", "__end__": END}[tools_condition(s)]`.

### Example 1 — standard ReAct loop

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

tool_node = ToolNode([add, multiply])

def call_model(state: State) -> dict:
    # Stub: In real usage, call an LLM here
    from langchain_core.messages import AIMessage
    if len(state["messages"]) == 1:
        return {"messages": [AIMessage("", tool_calls=[
            {"name": "add", "args": {"a": 3, "b": 4}, "id": "tc1", "type": "tool_call"}
        ])]}
    return {"messages": [AIMessage("The answer is 7.")]}

builder = StateGraph(State)
builder.add_node("model", call_model)
builder.add_node("tools", tool_node)
builder.add_edge(START, "model")
builder.add_conditional_edges("model", tools_condition)  # routes to "tools" or END
builder.add_edge("tools", "model")  # loop back after tool execution
graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage("What is 3 + 4?")]})
print(result["messages"][-1].content)  # The answer is 7.
```

### Example 2 — `tools_condition` with custom `messages_key`

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

class CustomState(TypedDict):
    chat: Annotated[list[BaseMessage], add_messages]
    turns: int

@tool
def echo(text: str) -> str:
    """Echo input text."""
    return text

tool_node = ToolNode([echo], messages_key="chat")

def router(state: CustomState) -> str:
    # Pass messages_key so tools_condition reads from "chat" instead of "messages"
    return tools_condition(state, messages_key="chat")

builder = StateGraph(CustomState)
builder.add_node("model", lambda s: {
    "chat": [AIMessage("", tool_calls=[
        {"name": "echo", "args": {"text": "hello"}, "id": "e1", "type": "tool_call"}
    ])],
    "turns": s["turns"] + 1,
})
builder.add_node("tools", tool_node)
builder.add_edge(START, "model")
builder.add_conditional_edges("model", router, {"tools": "tools", "__end__": END})
builder.add_edge("tools", END)
graph = builder.compile()

result = graph.invoke({"chat": [], "turns": 0})
print(result["chat"][-1].content)  # hello
```

### Example 3 — routing to multiple named tool nodes

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

@tool
def web_search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

@tool
def calculator(expr: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expr))  # noqa: S307

# Two separate tool nodes for different tool categories
search_node = ToolNode([web_search], name="search_tools")
calc_node = ToolNode([calculator], name="calc_tools")

TOOL_NAMES = {t.name: node_name
              for node_name, tnode in [("search_tools", search_node), ("calc_tools", calc_node)]
              for t in tnode._tools_by_name.values()}

def route_by_tool(state: State) -> str:
    """Route to the right tool node based on which tool was called."""
    last = state["messages"][-1]
    if not (hasattr(last, "tool_calls") and last.tool_calls):
        return END
    tool_name = last.tool_calls[0]["name"]
    return TOOL_NAMES.get(tool_name, END)

def call_model(state: State) -> dict:
    if len(state["messages"]) == 1:
        return {"messages": [AIMessage("", tool_calls=[
            {"name": "calculator", "args": {"expr": "2**10"}, "id": "c1", "type": "tool_call"}
        ])]}
    return {"messages": [AIMessage("Done.")]}

builder = StateGraph(State)
builder.add_node("model", call_model)
builder.add_node("search_tools", search_node)
builder.add_node("calc_tools", calc_node)
builder.add_edge(START, "model")
builder.add_conditional_edges("model", route_by_tool,
    {"search_tools": "search_tools", "calc_tools": "calc_tools", END: END})
builder.add_edge("search_tools", "model")
builder.add_edge("calc_tools", "model")
graph = builder.compile()

result = graph.invoke({"messages": [{"role": "user", "content": "What is 2^10?"}]})
print(result["messages"][-2].content)  # 1024
```

---

## 4 · `SyncAsyncFuture`

**Module:** `langgraph.pregel._call`

`SyncAsyncFuture[T]` is a `concurrent.futures.Future[T]` subclass that adds `__await__` support, making it usable both in synchronous contexts (`.result()`) and in async coroutines (`await`). It is the return type of every `@task` invocation inside a `@entrypoint` function, enabling parallel fan-out: call multiple `@task` functions, collect the returned `SyncAsyncFuture` objects, then await or `.result()` them all.

**Key source facts (`langgraph/pregel/_call.py`):**

- `class SyncAsyncFuture(Generic[T], concurrent.futures.Future[T])` — inherits the complete `concurrent.futures.Future` interface.
- `__await__(self) -> Generator[T, None, T]` — a one-step generator that `yield`s `cast(T, ...)`, allowing `await fut` in async code. The event loop polls the underlying future's result via the `__await__` generator protocol.
- Returned immediately when a `@task`-decorated function is called; the task is scheduled via `CONFIG_KEY_CALL` in the config.
- `.result()` blocks in sync context; `await fut` suspends in async context.
- Multiple `SyncAsyncFuture` objects can be gathered with `asyncio.gather` or resolved sequentially with `.result()`.

### Example 1 — parallel fan-out with `@task` in `@entrypoint`

```python
import asyncio
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

@task
def fetch_data(source: str) -> dict:
    """Simulate fetching data from a source."""
    import time
    time.sleep(0.01)  # simulate I/O
    return {"source": source, "records": len(source) * 10}

@entrypoint(checkpointer=InMemorySaver())
def pipeline(sources: list[str]) -> list[dict]:
    # Fan-out: launch all tasks concurrently
    futures = [fetch_data(src) for src in sources]
    # Fan-in: collect all results (blocks until all complete)
    return [f.result() for f in futures]

config = {"configurable": {"thread_id": "t1"}}
result = pipeline.invoke(["db", "api", "cache"], config=config)
for r in result:
    print(r)
# {'source': 'db', 'records': 20}
# {'source': 'api', 'records': 30}
# {'source': 'cache', 'records': 50}
```

### Example 2 — async fan-out with `await` on `SyncAsyncFuture`

```python
import asyncio
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

@task
async def async_fetch(url: str) -> str:
    """Simulate an async HTTP fetch."""
    await asyncio.sleep(0.01)
    return f"content from {url}"

@entrypoint(checkpointer=InMemorySaver())
async def async_pipeline(urls: list[str]) -> list[str]:
    # Each @task call returns a SyncAsyncFuture immediately
    futures = [async_fetch(url) for url in urls]
    # asyncio.gather works because SyncAsyncFuture is awaitable
    results = await asyncio.gather(*futures)
    return list(results)

async def main():
    config = {"configurable": {"thread_id": "async-t1"}}
    result = await async_pipeline.ainvoke(
        ["https://a.example.com", "https://b.example.com"],
        config=config,
    )
    for r in result:
        print(r)

asyncio.run(main())
# content from https://a.example.com
# content from https://b.example.com
```

### Example 3 — mixed parallel and sequential tasks

```python
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

@task
def score_document(doc: str) -> float:
    """Score document relevance."""
    return len(doc) / 100.0

@task
def summarize_doc(doc: str) -> str:
    """Summarize a document."""
    return doc[:50] + "..." if len(doc) > 50 else doc

@task
def rank_and_format(scores: list[float], summaries: list[str]) -> list[dict]:
    """Rank documents by score and format output."""
    pairs = sorted(zip(scores, summaries), reverse=True)
    return [{"summary": s, "score": round(sc, 2)} for sc, s in pairs]

@entrypoint(checkpointer=InMemorySaver())
def rag_pipeline(documents: list[str]) -> list[dict]:
    # Parallel: score and summarize all docs concurrently
    score_futs = [score_document(d) for d in documents]
    summary_futs = [summarize_doc(d) for d in documents]

    scores = [f.result() for f in score_futs]
    summaries = [f.result() for f in summary_futs]

    # Sequential: ranking depends on all scores and summaries
    ranked = rank_and_format(scores, summaries).result()
    return ranked

config = {"configurable": {"thread_id": "rag-1"}}
docs = [
    "Short doc.",
    "A much longer document with extensive content about LangGraph internals.",
    "Medium length document about agent frameworks.",
]
results = rag_pipeline.invoke(docs, config=config)
for r in results:
    print(r)
```

---

## 5 · `ChannelWrite` · `ChannelWriteEntry` · `ChannelWriteTupleEntry`

**Module:** `langgraph.pregel._write`

`ChannelWrite` is the `RunnableCallable` that routes node outputs into the Pregel write queue. `StateGraph` automatically inserts a `ChannelWrite` at the end of every node pipeline. `ChannelWriteEntry` and `ChannelWriteTupleEntry` are the `NamedTuple` write descriptors that `ChannelWrite` contains. Understanding these lets you control exactly which channels receive values and how they are transformed before being applied.

**Key source facts (`langgraph/pregel/_write.py`):**

- `ChannelWriteEntry(channel, value=PASSTHROUGH, skip_none=False, mapper=None)` — `PASSTHROUGH` means "use the node's output value"; a concrete `value` overrides. `skip_none=True` suppresses the write when the value is `None`. `mapper` is a `Callable` applied before writing.
- `ChannelWriteTupleEntry(mapper, value=PASSTHROUGH)` — the mapper receives the node output and returns a `Sequence[tuple[str, Any]]`, enabling dynamic channel → value pairs from a single write.
- `ChannelWrite.do_write(config, writes)` — static method that calls `config["configurable"][CONFIG_KEY_SEND](writes)` after validating each entry. Can be called imperatively from within a node.
- `ChannelWrite.get_static_writes(writes)` — static method used during graph compilation to infer topology (which channels a node writes to without running it).
- `register_writer(cls, runnable)` — class method marking a `Runnable` as a write-emitting node so the graph topology analysis recognises it.
- The `TASKS` reserved channel cannot be written to via `ChannelWriteEntry` — `do_write` raises `InvalidUpdateError`.

### Example 1 — direct `ChannelWrite.do_write` from a node

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.pregel._write import ChannelWrite, ChannelWriteEntry
from langchain_core.runnables import RunnableConfig
from langgraph.constants import PASSTHROUGH

class State(TypedDict):
    score: float
    label: str
    raw: str

def classify_node(state: State, config: RunnableConfig) -> None:
    """Node that uses do_write to imperatively push to multiple channels."""
    score = float(len(state["raw"])) / 100.0
    label = "high" if score > 0.5 else "low"

    # Imperatively write to specific channels
    ChannelWrite.do_write(
        config,
        [
            ChannelWriteEntry("score", score),
            ChannelWriteEntry("label", label),
            # skip_none: don't write if value is None
            ChannelWriteEntry("raw", None, skip_none=True),
        ],
    )

builder = StateGraph(State)
builder.add_node("classify", classify_node)
builder.add_edge(START, "classify")
builder.add_edge("classify", END)
graph = builder.compile()

result = graph.invoke({"score": 0.0, "label": "", "raw": "x" * 60})
print(result["score"])   # 0.6
print(result["label"])   # high
print(result["raw"])     # 'xxx...' — unchanged (skip_none prevented overwrite with None)
```

### Example 2 — `mapper` transform on write

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.pregel._write import ChannelWrite, ChannelWriteEntry

class State(TypedDict):
    items: list[str]
    count: int
    upper_items: list[str]

def process(state: State) -> dict:
    return {"items": state["items"]}

# Add a ChannelWrite with mapper: write uppercased items to upper_items channel
# and item count to count channel, both derived from the items list
write = ChannelWrite([
    ChannelWriteEntry("upper_items", mapper=lambda items: [i.upper() for i in items]),
    ChannelWriteEntry("count", mapper=lambda items: len(items)),
])

builder = StateGraph(State)
builder.add_node("process", process)
# The ChannelWrite is chained after the process node's return value
builder.add_node("write_extras", write)
builder.add_edge(START, "process")
# In practice, mappers are used when wiring nodes; here shown standalone for illustration
builder.add_edge("process", END)
graph = builder.compile()

result = graph.invoke({"items": ["apple", "banana", "cherry"], "count": 0, "upper_items": []})
print(result["items"])  # ['apple', 'banana', 'cherry']
```

### Example 3 — `ChannelWriteTupleEntry` for dynamic channel routing

```python
from langgraph.pregel._write import ChannelWriteTupleEntry
from langchain_core.runnables import RunnableConfig

# A ChannelWriteTupleEntry mapper returns (channel, value) pairs dynamically.
# This enables a single node output to fan out to different channels based on content.

def dispatch_by_type(output: dict) -> list[tuple[str, object]]:
    """Route node output to different channels based on 'type' field."""
    writes = []
    if "error" in output:
        writes.append(("errors", output["error"]))
    if "result" in output:
        writes.append(("results", output["result"]))
    if "metadata" in output:
        writes.append(("metadata", output["metadata"]))
    return writes

entry = ChannelWriteTupleEntry(mapper=dispatch_by_type)
# When ChannelWrite processes this entry:
# 1. Passes the node output (a dict) to dispatch_by_type
# 2. dispatch_by_type returns [(channel, value), ...]
# 3. Each pair is written to its channel
sample_output = {"result": 42, "metadata": {"source": "api"}}
pairs = dispatch_by_type(sample_output)
print(pairs)
# [('results', 42), ('metadata', {'source': 'api'})]
```

---

## 6 · `StateSnapshot` · `PregelTask`

**Module:** `langgraph.types` (canonical) / `langgraph.pregel.types` (deprecated alias)

`StateSnapshot` is the `NamedTuple` returned by `graph.get_state()` and each item in `graph.get_state_history()`. `PregelTask` is the per-task descriptor inside `StateSnapshot.tasks`, giving you name, path, id, any error raised, pending interrupts, and — for subgraph tasks — the child `StateSnapshot` in its `.state` field.

**Key source facts:**

- `StateSnapshot.values: dict[str, Any]` — current channel values.
- `StateSnapshot.next: tuple[str, ...]` — node names to execute in the next super-step.
- `StateSnapshot.tasks: tuple[PregelTask, ...]` — one entry per task scheduled for the next step.
- `StateSnapshot.interrupts: tuple[Interrupt, ...]` — pending `Interrupt` objects (from `interrupt()` calls) not yet resumed.
- `StateSnapshot.parent_config` — config of the immediately preceding checkpoint; use to walk history backwards.
- `PregelTask.error: Exception | None` — set if the task raised during the previous step.
- `PregelTask.result: Any | None` — set after the task completed (available in history snapshots).
- `PregelTask.state: None | RunnableConfig | StateSnapshot` — for subgraph nodes this is a `StateSnapshot` of the child graph's state at that step.
- `StateUpdate(values, as_node, task_id)` — the NamedTuple passed to `bulk_update_state`.

### Example 1 — inspecting `tasks` and `interrupts` in a suspended graph

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    approved: bool

def approval_gate(state: State) -> dict:
    """Gate that suspends for human approval."""
    answer = interrupt("Please approve this action.")
    return {"approved": answer == "yes"}

def finalize(state: State) -> dict:
    return {"messages": [{"role": "assistant", "content": "Action approved!" if state["approved"] else "Action denied."}]}

builder = StateGraph(State)
builder.add_node("gate", approval_gate)
builder.add_node("finalize", finalize)
builder.add_edge(START, "gate")
builder.add_edge("gate", "finalize")
builder.add_edge("finalize", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "approval-1"}}
graph.invoke({"messages": [HumanMessage("Do the action.")], "approved": False}, config=config)

# Inspect the suspended snapshot
snapshot = graph.get_state(config)
print("Next nodes:", snapshot.next)
# ('gate',)

print("Pending interrupts:", len(snapshot.interrupts))
# 1
print("Interrupt value:", snapshot.interrupts[0].value)
# 'Please approve this action.'

print("Tasks:", [(t.name, t.error) for t in snapshot.tasks])
# [('gate', None)]

# Resume with approval
graph.invoke({"messages": []}, config={**config, "input": {"resume": "yes"}})
```

### Example 2 — walking history with `parent_config`

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step: int

def node_a(state: State) -> dict:
    return {"messages": [AIMessage("Step A")], "step": state["step"] + 1}

def node_b(state: State) -> dict:
    return {"messages": [AIMessage("Step B")], "step": state["step"] + 1}

builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "history-walk"}}
graph.invoke({"messages": [HumanMessage("start")], "step": 0}, config=config)

# Walk history from latest to oldest via parent_config
snapshot = graph.get_state(config)
history = []
while snapshot is not None:
    history.append({
        "step": snapshot.values.get("step"),
        "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
    })
    if snapshot.parent_config:
        snapshot = graph.get_state(snapshot.parent_config)
    else:
        break

print(f"Collected {len(history)} checkpoints:")
for h in history:
    print(f"  step={h['step']} id={h['checkpoint_id'][:8]}...")
```

### Example 3 — using `get_state_history` with filters

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    counter: int

def increment(state: State) -> dict:
    return {"counter": state["counter"] + 1}

builder = StateGraph(State)
builder.add_node("inc", increment)
builder.add_edge(START, "inc")
builder.add_edge("inc", END)
graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "counter-thread"}}
# Run several times to accumulate history
for _ in range(5):
    graph.invoke({"counter": 0}, config=config)

# get_state_history returns newest-first iterator
snapshots = list(graph.get_state_history(config, limit=3))
print(f"Latest 3 checkpoints (newest first):")
for snap in snapshots:
    print(f"  counter={snap.values['counter']} next={snap.next}")

# Replay from an older checkpoint
old_snap = snapshots[-1]  # third-newest
replay_result = graph.invoke(None, config=old_snap.config)
print(f"Replayed from counter={old_snap.values['counter']}: got {replay_result['counter']}")
```

---

## 7 · `Pregel.get_state_history()` · `update_state()` · `bulk_update_state()`

**Module:** `langgraph.pregel.main`

These three methods form LangGraph's **time-travel and state injection** API. `get_state_history` streams `StateSnapshot` objects newest-first. `update_state` applies a single update dict (as if a named node wrote it), creating a new checkpoint branched from the target. `bulk_update_state` applies a batch of `StateUpdate` NamedTuples in sequence, useful for seeding multiple checkpoints atomically.

**Key source facts:**

- `get_state_history(config, *, filter=None, before=None, limit=None)` — all parameters are optional. `filter` is a metadata dict for equality filtering. `before` is a config pointing to a checkpoint; returns only history older than that. `limit` caps returned items.
- `update_state(config, values, as_node=None)` — `values` is a dict merged into the state via the channels' reducers. `as_node` identifies which node's perspective to apply (affects which input schema is used). Returns a new `RunnableConfig` with the new `checkpoint_id`.
- `bulk_update_state(config, supersteps)` — `supersteps` is a `Sequence[Sequence[StateUpdate]]`. Each inner sequence is one super-step; writes within a super-step are applied concurrently via reducers.
- `StateUpdate(values, as_node, task_id)` — NamedTuple. `as_node` can be `None` for a "no node" perspective.
- A branch is created by calling `update_state` on an old checkpoint; subsequent `invoke`/`stream` on the returned config runs from that branch point.

### Example 1 — branching execution from an older checkpoint

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    path: str

def route(state: State) -> dict:
    return {"path": state["path"] + "->route"}

def action_a(state: State) -> dict:
    return {"path": state["path"] + "->A", "messages": [AIMessage("Took path A")]}

def action_b(state: State) -> dict:
    return {"path": state["path"] + "->B", "messages": [AIMessage("Took path B")]}

builder = StateGraph(State)
builder.add_node("route", route)
builder.add_node("A", action_a)
builder.add_node("B", action_b)
builder.add_edge(START, "route")
builder.add_conditional_edges("route", lambda s: "A" if "A" in s["path"] else "B")
builder.add_edge("A", END)
builder.add_edge("B", END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "branch-demo"}}

# First run goes to A
result1 = graph.invoke({"messages": [HumanMessage("Go to A")], "path": ""}, config=config)
print("Run 1:", result1["path"])  # ->route->A

# Get the checkpoint just after the START node (before route)
history = list(graph.get_state_history(config))
after_start = history[-2]  # second-oldest

# Branch: update state on the old checkpoint and redirect to B
new_config = graph.update_state(
    after_start.config,
    {"path": "force"},
    as_node="route",
)
# Invoke from the branch point
result2 = graph.invoke(None, config=new_config)
print("Run 2 (branch):", result2["path"])
```

### Example 2 — `update_state` to correct a mistake mid-run

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    value: int
    approved: bool

def validate(state: State) -> dict:
    return {"approved": state["value"] > 0}

def process(state: State) -> dict:
    return {"value": state["value"] * 2}

builder = StateGraph(State)
builder.add_node("validate", validate)
builder.add_node("process", process)
builder.add_edge(START, "validate")
builder.add_edge("validate", "process")
builder.add_edge("process", END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "correction"}}

# Run with a value that passes validation
graph.invoke({"value": 5, "approved": False}, config=config)
current = graph.get_state(config)
print("After run:", current.values["value"])  # 10

# Imagine we want to go back and correct the initial value
history = list(graph.get_state_history(config))
initial_snap = history[-1]  # the starting checkpoint

# Inject a corrected initial value and re-run
corrected_config = graph.update_state(initial_snap.config, {"value": 3}, as_node="__input__")
result = graph.invoke(None, config=corrected_config)
print("After correction:", result["value"])  # 6
```

### Example 3 — `bulk_update_state` for test fixture seeding

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import StateUpdate

class State(TypedDict):
    messages: list[str]
    count: int

def process(state: State) -> dict:
    return {"messages": state["messages"] + ["processed"], "count": state["count"] + 1}

builder = StateGraph(State)
builder.add_node("process", process)
builder.add_edge(START, "process")
builder.add_edge("process", END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "bulk-seed"}}

# First invocation to establish the thread
graph.invoke({"messages": [], "count": 0}, config=config)

# Bulk update: two super-steps applied atomically
# Each inner list is one super-step; writes within a super-step merge via reducers
supersteps = [
    [StateUpdate(values={"messages": ["seed-step-1"], "count": 10}, as_node="process", task_id=None)],
    [StateUpdate(values={"messages": ["seed-step-2"], "count": 20}, as_node="process", task_id=None)],
]
graph.bulk_update_state(config, supersteps)

final = graph.get_state(config)
print("Messages:", final.values["messages"])
print("Count:", final.values["count"])
```

---

## 8 · `InMemoryStore` · `IndexConfig` · `SearchItem`

**Module:** `langgraph.store.memory` / `langgraph.store.base`

`InMemoryStore` is the built-in cross-thread key-value store with optional **cosine-similarity vector search**. `IndexConfig` is the TypedDict that activates embedding: set `dims` (embedding dimension), `embed` (a callable or provider string), and optionally `fields` (list of JSON paths to embed; default `["$"]` embeds the whole value). `SearchItem` extends `Item` with a `.score: float | None` field populated when searching with a `query`.

**Key source facts (`langgraph/store/memory.py`):**

- `InMemoryStore(index=None)` — without `index`, search by `filter` only (no semantic query). With `index`, both filter and semantic query work.
- `put(namespace, key, value, index=True)` — stores the item; if `index` is configured, embeds the `fields` paths and stores vectors in `self._vectors[namespace][key]`.
- `search(namespace_prefix, query=None, filter=None, limit=10, offset=0)` — `query` triggers embedding + cosine similarity ranking. `filter` is applied after ranking. Returns `list[SearchItem]`.
- `SearchItem.score: float | None` — cosine similarity score; `None` when no vector query was made.
- `IndexConfig.fields` — list of dot-path / array-index / multi-field selectors. The default `["$"]` embeds the serialised JSON string of the entire `value`.
- `get_text_at_path(value, path)` extracts text at the given path for embedding.
- The `embed` field accepts a callable `(list[str]) -> list[list[float]]` or a provider string like `"openai:text-embedding-3-small"`.

### Example 1 — basic `put` / `get` / `search` by filter

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# Store items in namespaced partitions
store.put(("users", "alice"), "profile", {"name": "Alice", "role": "admin", "active": True})
store.put(("users", "bob"),   "profile", {"name": "Bob",   "role": "user",  "active": False})
store.put(("users", "carol"), "profile", {"name": "Carol", "role": "admin", "active": True})

# Get a specific item
alice = store.get(("users", "alice"), "profile")
print(alice.value["name"])  # Alice

# Filter search — no embeddings needed
admins = store.search(("users",), filter={"role": "admin", "active": True})
print([item.value["name"] for item in admins])  # ['Alice', 'Carol']

# List all namespaces
namespaces = store.list_namespaces(prefix=("users",))
print(sorted(str(ns) for ns in namespaces))
# ["('users', 'alice')", "('users', 'bob')", "('users', 'carol')"]
```

### Example 2 — semantic search with `IndexConfig` and an embedding function

```python
from langgraph.store.memory import InMemoryStore

# Simple deterministic embedding for testing (do NOT use in production)
def mock_embed(texts: list[str]) -> list[list[float]]:
    """Embed by hashing characters into a 4-dim vector."""
    import hashlib
    results = []
    for text in texts:
        h = hashlib.md5(text.encode()).digest()
        vec = [b / 255.0 for b in h[:4]]
        total = sum(v**2 for v in vec) ** 0.5 or 1.0
        results.append([v / total for v in vec])  # normalise
    return results

store = InMemoryStore(index={
    "dims": 4,
    "embed": mock_embed,
    "fields": ["text"],  # only embed the "text" field of each stored value
})

# Store knowledge base entries
store.put(("kb",), "doc1", {"text": "LangGraph state machines for agents", "category": "framework"})
store.put(("kb",), "doc2", {"text": "Python async event loop internals",    "category": "python"})
store.put(("kb",), "doc3", {"text": "Graph neural networks for NLP",        "category": "ml"})

# Semantic search — returns SearchItem with .score
results = store.search(("kb",), query="building AI agents with graphs", limit=2)
for item in results:
    print(f"score={item.score:.3f}  text={item.value['text'][:40]}")
# Highest-scoring item is the one most semantically similar to the query
```

### Example 3 — using `InMemoryStore` with a graph for long-term memory

```python
from dataclasses import dataclass
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver

@dataclass
class Context:
    user_id: str

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    response: str

store = InMemoryStore()

def personalized_node(state: State, runtime: Runtime[Context]) -> dict:
    """Node that reads user preferences from long-term store."""
    user_id = runtime.context.user_id

    # Load preference from store
    pref = runtime.store.get(("prefs", user_id), "theme") if runtime.store else None
    theme = pref.value["value"] if pref else "light"

    # Save a new preference
    if runtime.store:
        runtime.store.put(("prefs", user_id), "theme", {"value": "dark"})

    response = f"Hello! Your theme is {theme} (updated to dark)."
    return {"response": response, "messages": [AIMessage(response)]}

builder = StateGraph(State, context_schema=Context)
builder.add_node("greet", personalized_node)
builder.add_edge(START, "greet")
builder.add_edge("greet", END)

graph = builder.compile(checkpointer=InMemorySaver(), store=store)

# First call: theme defaults to "light"
result1 = graph.invoke(
    {"messages": [HumanMessage("hi")]},
    config={"configurable": {"thread_id": "t1"}},
    context=Context(user_id="user42"),
)
print(result1["response"])  # Hello! Your theme is light (updated to dark).

# Second call: theme is now "dark" from the store
result2 = graph.invoke(
    {"messages": [HumanMessage("hi again")]},
    config={"configurable": {"thread_id": "t2"}},
    context=Context(user_id="user42"),
)
print(result2["response"])  # Hello! Your theme is dark (updated to dark).
```

---

## 9 · `add_messages` advanced patterns

**Module:** `langgraph.graph.message`

`add_messages` is a **reducer function** for the `messages` channel — it handles message deduplication, deletion, and format conversion. It is not a class but a function registered via `Annotated[list[BaseMessage], add_messages]`. Understanding its full behaviour unlocks patterns like safe parallel updates, message tombstoning, OpenAI-format conversion, and the `push_message` dual-write API.

**Key source facts (`langgraph/graph/message.py`):**

- **ID deduplication**: if the incoming list contains a message whose `id` matches an existing message, the existing message is replaced (not appended). This powers "edit in place" patterns.
- **`RemoveMessage(id=...)`**: a tombstone that deletes the message with that id. Raises `ValueError` if the id is not found.
- **`REMOVE_ALL_MESSAGES`**: a sentinel that clears the entire messages list in one update.
- **`format='langchain-openai'`**: calls `_format_messages` converting LangChain-style messages (with `type` field) to OpenAI-style messages (with `role` field). Passed as `add_messages.format` at graph compile time.
- **`push_message(msg, config)`**: writes `msg` to both the **custom stream** (immediately visible to streaming consumers) and the `messages` channel (persisted in state). Requires a `StreamWriter` context.
- `_messages_delta_reducer` is the internal function that applies the full dedup/delete logic.

### Example 1 — ID deduplication and `RemoveMessage`

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def node(state: State) -> dict:
    return {}  # no-op

builder = StateGraph(State)
builder.add_node("noop", node)
builder.add_edge(START, "noop")
builder.add_edge("noop", END)
graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "dedup-demo"}}

# Initial messages with explicit IDs
msg1 = HumanMessage("Hello", id="msg-1")
msg2 = AIMessage("Hi there!", id="msg-2")
graph.invoke({"messages": [msg1, msg2]}, config=config)

state = graph.get_state(config)
print("Before edit:", [m.content for m in state.values["messages"]])
# ['Hello', 'Hi there!']

# Update: replace msg2 content by supplying same id
corrected = AIMessage("Hi! How can I help?", id="msg-2")
graph.update_state(config, {"messages": [corrected]})

state = graph.get_state(config)
print("After edit:", [m.content for m in state.values["messages"]])
# ['Hello', 'Hi! How can I help?']

# Remove msg1
graph.update_state(config, {"messages": [RemoveMessage(id="msg-1")]})
state = graph.get_state(config)
print("After remove:", [m.content for m in state.values["messages"]])
# ['Hi! How can I help?']
```

### Example 2 — `REMOVE_ALL_MESSAGES` wipe and conversation reset

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session: int

def chat_node(state: State) -> dict:
    return {"messages": [AIMessage(f"Session {state['session']} reply")]}

def reset_node(state: State) -> dict:
    """Wipe all messages and start a new session."""
    return {
        "messages": REMOVE_ALL_MESSAGES,  # clears the entire list
        "session": state["session"] + 1,
    }

builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.add_node("reset", reset_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "reset-demo"}}

# Build up conversation history
for msg in ["Hi", "Tell me more", "Thanks"]:
    graph.invoke({"messages": [HumanMessage(msg)], "session": 1}, config=config)

state = graph.get_state(config)
print(f"Before reset: {len(state.values['messages'])} messages")

# Apply the wipe via update_state
graph.update_state(config, {"messages": REMOVE_ALL_MESSAGES, "session": 2})

state = graph.get_state(config)
print(f"After reset: {len(state.values['messages'])} messages, session={state.values['session']}")
# After reset: 0 messages, session=2
```

### Example 3 — `push_message` for real-time streaming

```python
import asyncio
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.message import push_message
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def streaming_node(state: State, config: RunnableConfig) -> dict:
    """Node that streams token-by-token via push_message."""
    tokens = ["Hello", ", ", "streaming", " user", "!"]
    full_content = ""
    for token in tokens:
        full_content += token
        # push_message writes to the stream immediately AND queues for state
        push_message(AIMessage(full_content, id="stream-msg"), config)
    # Final return is not needed — push_message handled state write
    return {}

builder = StateGraph(State)
builder.add_node("stream", streaming_node)
builder.add_edge(START, "stream")
builder.add_edge("stream", END)

graph = builder.compile(checkpointer=InMemorySaver())

async def main():
    config = {"configurable": {"thread_id": "push-demo"}}
    async for event in graph.astream(
        {"messages": [HumanMessage("Start")]},
        config=config,
        stream_mode="messages",
    ):
        chunk, metadata = event
        if metadata.get("langgraph_node") == "stream":
            print(f"[stream] {chunk.content!r}")

asyncio.run(main())
```

---

## 10 · `RunControl` · `GraphDrained` · `Runtime.drain_requested`

**Module:** `langgraph.runtime` / `langgraph.errors` / `langgraph.types`

`RunControl` is a **cooperative drain signal** injected into `Runtime.control` for every graph run. It is a thin dataclass with a single `_drain_reason: str | None` attribute, making it thread-safe without a lock (single attribute write is atomic in CPython). `GraphDrained` is the `GraphBubbleUp` subclass that surfaces when the drain is processed. `Runtime.drain_requested` and `Runtime.drain_reason` are convenience properties that delegate to the embedded `RunControl`.

**Key source facts (`langgraph/runtime.py`, `langgraph/errors.py`):**

- `RunControl.__init__` sets `self._drain_reason = None`.
- `RunControl.request_drain(reason="shutdown")` — writes to `_drain_reason`. Safe from any thread.
- `RunControl.drain_requested: bool` — `self._drain_reason is not None`.
- `RunControl.drain_reason: str | None` — the string passed to `request_drain`.
- `GraphDrained(reason: str)` — `GraphBubbleUp` subclass with a `reason` field. Raised by the Pregel loop when a drain is requested and the current super-step completes.
- Nodes access drain state via `runtime.drain_requested` or by injecting `Runtime` directly.
- `Runtime.control` is `None` outside an active graph run; always check before accessing.
- Intended use: stop long-running graphs gracefully on shutdown signals without killing mid-step.

### Example 1 — node checks `runtime.drain_requested` for graceful exit

```python
import asyncio
from dataclasses import dataclass
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime, RunControl
from langgraph.checkpoint.memory import InMemorySaver

@dataclass
class AppContext:
    max_turns: int

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    turn: int

def agent_node(state: State, runtime: Runtime[AppContext]) -> dict:
    """Agent that voluntarily drains when the drain signal is set."""
    if runtime.drain_requested:
        reason = runtime.drain_reason
        return {"messages": [AIMessage(f"Stopping gracefully: {reason}")], "turn": state["turn"]}

    turn = state["turn"] + 1
    return {
        "messages": [AIMessage(f"Processing turn {turn}...")],
        "turn": turn,
    }

def should_continue(state: State) -> str:
    return "agent" if state["turn"] < 3 else END

builder = StateGraph(State, context_schema=AppContext)
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)

graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "drain-demo"}}

result = graph.invoke(
    {"messages": [], "turn": 0},
    config=config,
    context=AppContext(max_turns=3),
)
print(f"Final turn: {result['turn']}")
print(f"Last message: {result['messages'][-1].content}")
```

### Example 2 — `request_drain` from a background thread

```python
import threading
import time
from dataclasses import dataclass
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime, RunControl
from langgraph.checkpoint.memory import InMemorySaver

@dataclass
class Ctx:
    control: RunControl  # shared control object

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step: int

def slow_node(state: State, runtime: Runtime[Ctx]) -> dict:
    """Simulates slow work; checks drain signal between steps."""
    time.sleep(0.05)  # simulate work
    if runtime.drain_requested:
        return {"messages": [AIMessage("Drain requested — stopping")], "step": -1}
    return {"messages": [AIMessage(f"Step {state['step'] + 1}")], "step": state["step"] + 1}

def should_loop(state: State) -> str:
    return END if state["step"] < 0 or state["step"] >= 10 else "work"

builder = StateGraph(State, context_schema=Ctx)
builder.add_node("work", slow_node)
builder.add_edge(START, "work")
builder.add_conditional_edges("work", should_loop)

graph = builder.compile(checkpointer=InMemorySaver())

# Create a shared RunControl that we can signal from outside
ctrl = RunControl()

def send_drain_after(delay: float) -> None:
    time.sleep(delay)
    ctrl.request_drain("shutdown signal received")

# Trigger drain from a background thread after 0.1s
threading.Thread(target=send_drain_after, args=(0.1,), daemon=True).start()

result = graph.invoke(
    {"messages": [], "step": 0},
    config={"configurable": {"thread_id": "drain-thread"}},
    context=Ctx(control=ctrl),
)
print(f"Stopped at step: {result['step']}")
print(f"Last message: {result['messages'][-1].content}")
```

### Example 3 — catching `GraphDrained` at the caller level

```python
from dataclasses import dataclass
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime, RunControl
from langgraph.errors import GraphDrained
from langgraph.checkpoint.memory import InMemorySaver

@dataclass
class Ctx:
    control: RunControl

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    processed: int

def processor(state: State, runtime: Runtime[Ctx]) -> dict:
    # Request drain after processing 2 items to simulate graceful shutdown
    if state["processed"] >= 2:
        if runtime.control:
            runtime.control.request_drain("processed limit reached")
    return {"messages": [AIMessage(f"item-{state['processed'] + 1}")], "processed": state["processed"] + 1}

builder = StateGraph(State, context_schema=Ctx)
builder.add_node("process", processor)
builder.add_edge(START, "process")
builder.add_conditional_edges("process", lambda s: "process" if s["processed"] < 5 else END)

graph = builder.compile(checkpointer=InMemorySaver())

ctrl = RunControl()
try:
    result = graph.invoke(
        {"messages": [], "processed": 0},
        config={"configurable": {"thread_id": "drain-catch"}},
        context=Ctx(control=ctrl),
    )
    print("Completed normally, processed:", result["processed"])
except GraphDrained as e:
    print(f"Graph drained: {e.reason}")
    # On GraphDrained, retrieve the last persisted state from the checkpointer
    snapshot = graph.get_state({"configurable": {"thread_id": "drain-catch"}})
    print(f"State at drain: processed={snapshot.values['processed']}")
```

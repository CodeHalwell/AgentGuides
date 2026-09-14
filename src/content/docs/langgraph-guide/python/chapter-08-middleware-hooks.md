---
title: "Chapter 8 — Middleware & Hooks"
description: "Intercept model and tool calls in LangGraph using pre_model_hook, post_model_hook, per-node error handlers, retry policies, and set_node_defaults — all verified against langgraph==1.2.11."
framework: langgraph
language: python
sidebar:
  label: "8 · Middleware & Hooks"
  order: 8
---

# Chapter 8 — Middleware & Hooks

**What you'll learn:** how to intercept, transform, and guard model and tool calls without rewriting node business logic. LangGraph 1.2.11 provides four complementary mechanisms: `pre_model_hook` / `post_model_hook` on `create_react_agent`, per-node `error_handler` and `retry_policy` on `add_node`, `set_node_defaults` for graph-wide policy, and `ToolNode`'s `handle_tool_errors` for tool-specific error handling.

Verified against **`langgraph==1.2.11`** (modules: `langgraph.prebuilt`, `langgraph.graph`, `langgraph.types`).

**Time:** ~25 minutes.

> Prereqs: [Chapter 4 — Tools](/langgraph-guide/python/chapter-04-tools/).

> **Important note on `langchain.agents.middleware`.** This module does not exist in the current release. If you encountered references to `create_agent(middleware=[...])` or `AgentMiddleware` from `langchain.agents`, those APIs are not available in the installed packages. The real primitives are documented here.

## Where hooks and policies live

| Mechanism | Import | Scope |
|---|---|---|
| `pre_model_hook` / `post_model_hook` | `langgraph.prebuilt.create_react_agent` kwargs | Before/after every model call in a ReAct agent |
| `error_handler` | `StateGraph.add_node(error_handler=...)` | Per-node error recovery |
| `retry_policy` | `StateGraph.add_node(retry_policy=...)` or `@task(retry_policy=...)` | Per-node/task retry with backoff |
| `cache_policy` | `StateGraph.add_node(cache_policy=...)` | Per-node result caching |
| `timeout` | `StateGraph.add_node(timeout=...)` | Per-node wall-clock and idle timeout |
| `set_node_defaults` | `StateGraph.set_node_defaults(...)` | Graph-wide defaults for all nodes |
| `handle_tool_errors` | `ToolNode(handle_tool_errors=...)` | Tool execution error handling |

---

## 1. `pre_model_hook` — transform state before the model sees it

`pre_model_hook` is a callable (or `Runnable`) that receives the agent state and returns an updated state dict. It runs on every iteration of the agent loop, immediately before the model call.

Common uses: trimming the message history, injecting a system prompt, adding per-request metadata.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

# --- Example 1: Trim messages to avoid context overflow ---

def trim_to_last_n(state: dict) -> dict:
    """Keep only the last 20 messages to avoid blowing the context window."""
    msgs = state.get("messages", [])
    if len(msgs) > 20:
        # Preserve the system message if present, then take the tail
        system = [m for m in msgs if isinstance(m, SystemMessage)]
        rest = [m for m in msgs if not isinstance(m, SystemMessage)][-19:]
        return {"messages": system + rest}
    return {}   # Return empty dict = no change


agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet-20241022",
    tools=[],
    pre_model_hook=trim_to_last_n,
)
```

```python
# --- Example 2: Inject a dynamic system prompt each turn ---
from langchain_core.messages import SystemMessage

def inject_system_prompt(state: dict) -> dict:
    """Prepend a fresh system prompt at the start of each model call."""
    system = SystemMessage(content="You are a concise assistant. Respond in ≤3 sentences.")
    msgs = [m for m in state.get("messages", []) if not isinstance(m, SystemMessage)]
    return {"messages": [system] + msgs}


agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet-20241022",
    tools=[],
    pre_model_hook=inject_system_prompt,
)
```

```python
# --- Example 3: Token-budget guardrail via pre_model_hook ---
from langchain_core.messages import SystemMessage, HumanMessage

MAX_INPUT_CHARS = 12_000

def truncate_long_messages(state: dict) -> dict:
    """Hard truncate message content that exceeds a per-message character cap."""
    updated = []
    for msg in state.get("messages", []):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            if len(msg.content) > MAX_INPUT_CHARS:
                truncated = msg.content[:MAX_INPUT_CHARS] + " [truncated]"
                msg = msg.model_copy(update={"content": truncated})
        updated.append(msg)
    return {"messages": updated}


agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet-20241022",
    tools=[],
    pre_model_hook=truncate_long_messages,
)
```

**Return value rules:**
- Return a `dict` with the keys you want to update — unchanged keys are left as-is.
- Return an empty dict `{}` or `None` to pass the state through unchanged.
- The hook **must not** raise; uncaught exceptions propagate to the caller.

---

## 2. `post_model_hook` — inspect or transform the model's response

`post_model_hook` runs immediately after each model call. It receives the state (which now includes the latest `AIMessage` from the model) and can return updates.

Common uses: logging token usage, enforcing response policies, adding metadata, early-exit guards.

```python
from langgraph.prebuilt import create_react_agent

# --- Example 1: Log token usage on every model call ---

def log_token_usage(state: dict) -> dict:
    last_msg = state["messages"][-1]
    usage = getattr(last_msg, "usage_metadata", None)
    if usage:
        print(
            f"[tokens] in={usage.get('input_tokens', 0)} "
            f"out={usage.get('output_tokens', 0)} "
            f"total={usage.get('total_tokens', 0)}"
        )
    return {}   # no state change


agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet-20241022",
    tools=[],
    post_model_hook=log_token_usage,
)
```

```python
# --- Example 2: Stateful cost accumulator using a class-based hook ---

class CostTracker:
    """Accumulate token costs across the full agent run."""

    INPUT_PRICE_PER_1M = 3.00   # USD per 1M input tokens (Sonnet 3.5)
    OUTPUT_PRICE_PER_1M = 15.00

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def __call__(self, state: dict) -> dict:
        last_msg = state["messages"][-1]
        usage = getattr(last_msg, "usage_metadata", None) or {}
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        return {}

    @property
    def total_cost_usd(self) -> float:
        return (
            self.input_tokens / 1_000_000 * self.INPUT_PRICE_PER_1M
            + self.output_tokens / 1_000_000 * self.OUTPUT_PRICE_PER_1M
        )


tracker = CostTracker()

agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet-20241022",
    tools=[],
    post_model_hook=tracker,
)

result = agent.invoke({"messages": [{"role": "user", "content": "Explain LangGraph in one sentence."}]})
print(f"Run cost: ${tracker.total_cost_usd:.6f}")
```

```python
# --- Example 3: Stop the agent early if the model response is empty ---
from langchain_core.messages import AIMessage
from langgraph.types import Command

def guard_empty_response(state: dict) -> dict | Command:
    """If the model returned an empty response, end the run immediately."""
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and not last_msg.content:
        # Returning Command(goto=END) exits the agent loop immediately.
        from langgraph.graph import END
        return Command(goto=END)
    return {}


agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet-20241022",
    tools=[],
    post_model_hook=guard_empty_response,
)
```

---

## 3. Combining both hooks

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 22°C in {city}"


def pre_hook(state: dict) -> dict:
    """Ensure a system message is always present."""
    has_system = any(isinstance(m, SystemMessage) for m in state.get("messages", []))
    if not has_system:
        system = SystemMessage(content="Be brief. Use metric units.")
        return {"messages": [system] + state["messages"]}
    return {}


def post_hook(state: dict) -> dict:
    """Log the model's output content length."""
    last = state["messages"][-1]
    content = getattr(last, "content", "")
    if isinstance(content, str):
        print(f"[post_hook] response length: {len(content)} chars")
    return {}


agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet-20241022",
    tools=[get_weather],
    pre_model_hook=pre_hook,
    post_model_hook=post_hook,
)

result = agent.invoke({"messages": [{"role": "user", "content": "Weather in Paris?"}]})
```

---

## 4. Per-node `error_handler` on `add_node`

Every `add_node` call accepts an `error_handler` — a node function called when the primary node raises. It receives the same state the node received and can return a partial state update (to write an error message, skip, or set a fallback value).

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class PipelineState(TypedDict):
    input: str
    result: str
    error: str


def risky_transform(state: PipelineState) -> dict:
    """Simulates a node that can fail."""
    if "bad" in state["input"]:
        raise ValueError("Detected bad input")
    return {"result": state["input"].upper()}


def handle_transform_error(state: PipelineState) -> dict:
    """Recover from risky_transform failure gracefully."""
    return {
        "result": "",
        "error": f"Transform failed for input: {state['input']!r}",
    }


def summarize(state: PipelineState) -> dict:
    if state["error"]:
        return {"result": f"[skipped: {state['error']}]"}
    return {"result": f"Done: {state['result']}"}


builder = StateGraph(PipelineState)
builder.add_node(
    "transform",
    risky_transform,
    error_handler=handle_transform_error,   # called on any exception from risky_transform
)
builder.add_node("summarize", summarize)
builder.add_edge(START, "transform")
builder.add_edge("transform", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()

# Happy path
r = graph.invoke({"input": "hello", "result": "", "error": ""})
print(r)  # {'input': 'hello', 'result': 'Done: HELLO', 'error': ''}

# Error path — error_handler fires, graph continues
r = graph.invoke({"input": "bad input", "result": "", "error": ""})
print(r)  # {'input': 'bad input', 'result': '[skipped: Transform failed ...]', 'error': '...'}
```

---

## 5. Per-node `retry_policy` on `add_node`

`retry_policy` is a `RetryPolicy` NamedTuple (from `langgraph.types`) that configures exponential-backoff retries for transient failures. Specify which exception types trigger a retry via `retry_on`.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict
import random


class FetchState(TypedDict):
    url: str
    content: str


_attempt_count = 0

def flaky_fetch(state: FetchState) -> dict:
    """Simulates a network call that fails ~50% of the time."""
    global _attempt_count
    _attempt_count += 1
    if random.random() < 0.5:
        raise ConnectionError("Network blip")
    return {"content": f"<html>{state['url']}</html>"}


builder = StateGraph(FetchState)
builder.add_node(
    "fetch",
    flaky_fetch,
    retry_policy=RetryPolicy(
        initial_interval=0.1,   # first retry after 100 ms
        backoff_factor=2.0,     # double the interval each retry
        max_interval=5.0,       # cap at 5 s
        max_attempts=4,         # give up after 4 total attempts
        jitter=True,            # add random jitter
        retry_on=ConnectionError,   # only retry on network errors
    ),
)
builder.add_edge(START, "fetch")
builder.add_edge("fetch", END)

graph = builder.compile()
result = graph.invoke({"url": "https://example.com", "content": ""})
print(result["content"])
```

**`RetryPolicy` fields** (all optional, shown with defaults):

| Field | Default | Meaning |
|---|---|---|
| `initial_interval` | `0.5` | Seconds before first retry |
| `backoff_factor` | `2.0` | Multiplier applied after each attempt |
| `max_interval` | `128.0` | Upper bound on the interval (seconds) |
| `max_attempts` | `3` | Total attempts including the first |
| `jitter` | `True` | Add random jitter to each interval |
| `retry_on` | all exceptions | Exception type(s) or `Callable[[Exception], bool]` |

---

## 6. `set_node_defaults` — graph-wide policy

Instead of passing `retry_policy`, `cache_policy`, `error_handler`, or `timeout` to every `add_node` call, set them once for the whole graph. Individual nodes can still override.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy, CachePolicy, TimeoutPolicy
from typing_extensions import TypedDict


class State(TypedDict):
    query: str
    answer: str


def node_a(state: State) -> dict:
    return {"answer": f"A({state['query']})"}


def node_b(state: State) -> dict:
    return {"answer": f"B({state['answer']})"}


def global_error_handler(state: State) -> dict:
    return {"answer": f"[error] could not process: {state['query']}"}


builder = StateGraph(State)

# Apply defaults before adding nodes — they are inherited by every add_node call.
builder.set_node_defaults(
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.5),
    error_handler=global_error_handler,
    timeout=TimeoutPolicy(run_timeout=30.0),
)

builder.add_node("a", node_a)                # inherits defaults
builder.add_node("b", node_b)                # inherits defaults

builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)

graph = builder.compile()
result = graph.invoke({"query": "hello", "answer": ""})
print(result)   # {'query': 'hello', 'answer': 'B(A(hello))'}
```

You can override defaults on specific nodes:

```python
from langgraph.types import RetryPolicy

# node_b gets a different retry policy; node_a keeps the graph default
builder.add_node(
    "b",
    node_b,
    retry_policy=RetryPolicy(max_attempts=1),  # no retries for this node
)
```

---

## 7. `ToolNode` — fine-grained tool error handling

`ToolNode` from `langgraph.prebuilt` handles errors during tool execution. Configure `handle_tool_errors` to control what the model sees when a tool raises.

```python
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b


@tool
def fetch_data(key: str) -> str:
    """Fetch data by key from a remote store."""
    raise TimeoutError("Store unavailable")


# Strategy 1: Return the default exception message as a ToolMessage
tool_node_default = ToolNode(
    tools=[divide, fetch_data],
    handle_tool_errors=True,   # catches all exceptions, returns default message
)

# Strategy 2: Return a fixed string for all errors
tool_node_fixed = ToolNode(
    tools=[divide, fetch_data],
    handle_tool_errors="Tool failed. Please try a different approach.",
)

# Strategy 3: Only catch specific exception types
tool_node_selective = ToolNode(
    tools=[divide, fetch_data],
    handle_tool_errors=(ZeroDivisionError, TimeoutError),
)

# Strategy 4: Custom callable — full control over the error message
def format_error(exc: Exception) -> str:
    return f"[ERROR {type(exc).__name__}] {exc}. Retry with different arguments."

tool_node_custom = ToolNode(
    tools=[divide, fetch_data],
    handle_tool_errors=format_error,
)
```

### Using `ToolNode` in a graph with `tools_condition`

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic


@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol."""
    prices = {"AAPL": "189.30", "GOOG": "175.50"}
    if ticker not in prices:
        raise ValueError(f"Unknown ticker: {ticker!r}")
    return f"{ticker}: ${prices[ticker]}"


model = ChatAnthropic(model="claude-3-5-sonnet-20241022").bind_tools([get_stock_price])

tool_node = ToolNode(
    tools=[get_stock_price],
    handle_tool_errors=True,   # unknown tickers return an error ToolMessage
)

builder = StateGraph(MessagesState)

def call_model(state: MessagesState) -> dict:
    return {"messages": [model.invoke(state["messages"])]}

builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()

result = graph.invoke({"messages": [{"role": "user", "content": "What's the price of AAPL and ZZZZ?"}]})
for msg in result["messages"]:
    print(f"{type(msg).__name__}: {getattr(msg, 'content', '')[:80]}")
```

---

## 8. `MessagesState` — the built-in shorthand

For agents that only need a `messages` list, `MessagesState` saves you from writing the TypedDict yourself:

```python
from langgraph.graph import MessagesState, StateGraph, START, END

# MessagesState is equivalent to:
#   class MessagesState(TypedDict):
#       messages: Annotated[list[AnyMessage], add_messages]

builder = StateGraph(MessagesState)
# ... add nodes and edges
```

This is the same state type used by `create_react_agent` internally.

---

## 9. `timeout` per node — `TimeoutPolicy`

Prevent runaway nodes with per-node timeouts. Async-only — sync nodes raise `ValueError` at decoration time if `timeout` is set.

```python
from datetime import timedelta
from langgraph.graph import StateGraph, START, END
from langgraph.types import TimeoutPolicy
from typing_extensions import TypedDict
import asyncio


class State(TypedDict):
    result: str


async def slow_node(state: State) -> dict:
    await asyncio.sleep(60)   # simulates a hung external call
    return {"result": "done"}


builder = StateGraph(State)
builder.add_node(
    "slow",
    slow_node,
    timeout=TimeoutPolicy(
        run_timeout=10.0,       # abort after 10 seconds regardless
        idle_timeout=5.0,       # abort if no progress signal for 5 seconds
        refresh_on="auto",      # internal LangGraph signals refresh the idle clock
    ),
)
builder.add_edge(START, "slow")
builder.add_edge("slow", END)

graph = builder.compile()
# graph.ainvoke({"result": ""}) will raise asyncio.TimeoutError after ~5-10 s
```

**`TimeoutPolicy` fields:**

| Field | Type | Default | Meaning |
|---|---|---|---|
| `run_timeout` | `float \| timedelta \| None` | `None` | Hard wall-clock cap. Never refreshed by progress signals. |
| `idle_timeout` | `float \| timedelta \| None` | `None` | Max silence between progress events. |
| `refresh_on` | `"auto" \| "heartbeat"` | `"auto"` | `"auto"` = refreshed by LangGraph events; `"heartbeat"` = only by `runtime.heartbeat()`. |

---

## 10. Realistic production stack

Putting it all together: hooks for context management + graph-wide retry + per-node error recovery + tool error handling.

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langgraph.types import RetryPolicy, TimeoutPolicy
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic


# ── Tools ──────────────────────────────────────────────────────────────────

@tool
def search_docs(query: str) -> str:
    """Search the internal knowledge base."""
    return f"Results for '{query}': ..."

@tool
def send_alert(message: str) -> str:
    """Send a Slack alert."""
    print(f"[ALERT] {message}")
    return "Alert sent."


# ── Hooks ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful internal assistant. "
    "Be concise. Cite sources when answering from search results."
))

def enforce_system_prompt(state: dict) -> dict:
    """Ensure the system prompt is always the first message."""
    msgs = state.get("messages", [])
    if not msgs or not isinstance(msgs[0], SystemMessage):
        return {"messages": [SYSTEM_PROMPT] + list(msgs)}
    return {}


class TokenBudget:
    """Abort the run if cumulative token usage exceeds a budget."""
    BUDGET = 50_000

    def __init__(self) -> None:
        self.total = 0

    def __call__(self, state: dict) -> dict:
        last = state["messages"][-1]
        usage = getattr(last, "usage_metadata", None) or {}
        self.total += usage.get("total_tokens", 0)
        if self.total > self.BUDGET:
            raise RuntimeError(f"Token budget exceeded: {self.total} > {self.BUDGET}")
        return {}


budget = TokenBudget()

# ── Graph ──────────────────────────────────────────────────────────────────

# create_react_agent with hooks handles the agent loop.
agent = create_react_agent(
    model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
    tools=[search_docs, send_alert],
    pre_model_hook=enforce_system_prompt,
    post_model_hook=budget,
)

# For a custom graph: use set_node_defaults for graph-wide resilience.
# (Shown separately — create_react_agent builds its own StateGraph internally.)

# Invoke
from langgraph.checkpoint.memory import InMemorySaver
agent_with_memory = create_react_agent(
    model=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
    tools=[search_docs, send_alert],
    pre_model_hook=enforce_system_prompt,
    post_model_hook=budget,
    checkpointer=InMemorySaver(),
)

cfg = {"configurable": {"thread_id": "session-1"}}
result = agent_with_memory.invoke(
    {"messages": [{"role": "user", "content": "What does LangGraph do?"}]},
    config=cfg,
)
print(result["messages"][-1].content)
```

---

## Comparison: hooks vs. dedicated nodes vs. `set_node_defaults`

| Pattern | When to use |
|---|---|
| `pre_model_hook` | Trim messages, inject prompts, add per-call metadata — applies before **every** model call in the agent loop. |
| `post_model_hook` | Log tokens, enforce output policies, early-exit guards — applies after **every** model call. |
| Dedicated node | Business logic that transforms state as part of your graph (classification, enrichment, routing). Keep visible in the graph topology. |
| `add_node(error_handler=...)` | Per-node graceful degradation when a specific node might fail. |
| `add_node(retry_policy=...)` | Per-node retry for transient errors (network, rate limits). |
| `set_node_defaults(...)` | Graph-wide safety net — retry, timeout, error fallback applied to every node by default. |
| `ToolNode(handle_tool_errors=...)` | Control what the model sees when a tool raises — prevent stack traces from leaking into the conversation. |

The rule of thumb: if a reader tracing your graph would be surprised the behaviour exists, it probably belongs in a `pre_model_hook` or `post_model_hook`, not a node. If it's part of the business flow, it belongs in the graph as a node.

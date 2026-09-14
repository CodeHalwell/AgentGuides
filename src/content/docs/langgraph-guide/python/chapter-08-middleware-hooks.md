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
from langchain_anthropic import ChatAnthropic

# --- Example 1: Trim messages to avoid context overflow ---

def trim_to_last_n(state: dict) -> dict:
    """Keep only the last 20 messages, preserving complete tool-call/result pairs."""
    from langchain_core.messages import ToolMessage, AIMessage as AI
    msgs = state.get("messages", [])
    if len(msgs) <= 20:
        return {}   # Nothing to trim
    # Preserve at most ONE system message (the first), then take the tail.
    system_msgs = [m for m in msgs if isinstance(m, SystemMessage)][:1]
    non_system = [m for m in msgs if not isinstance(m, SystemMessage)]
    tail = non_system[-(20 - len(system_msgs)):]
    # Drop leading ToolMessages whose AIMessage pair was trimmed away — providers
    # reject a history that starts with a tool result without the preceding tool call.
    while tail and isinstance(tail[0], ToolMessage):
        tail = tail[1:]
    # Edge case: if the most recent exchange alone exceeds N (e.g. an agent made
    # 20+ parallel tool calls in one turn), the tail above is all ToolMessages and
    # stripping orphaned leading ones empties it. Fall back to the entire most recent
    # complete exchange so the model always receives at least one meaningful turn.
    if not tail and non_system:
        for i in range(len(non_system) - 1, -1, -1):
            if isinstance(non_system[i], AI):
                tail = non_system[i:]
                break
    # Drop a trailing AIMessage that has tool_calls but whose ToolMessage results
    # were trimmed away — this would also produce a malformed exchange.
    while tail and isinstance(tail[-1], AI) and getattr(tail[-1], "tool_calls", []):
        tail = tail[:-1]
    # Verify the first AIMessage with tool_calls has ALL its results present.
    # If an AIMessage made 2 tool calls but only 1 ToolMessage follows in the tail,
    # the exchange is still malformed; drop it (and its partial results) and retry.
    changed = True
    while changed and tail:
        changed = False
        for idx, msg in enumerate(tail):
            if not (isinstance(msg, AI) and getattr(msg, "tool_calls", [])):
                continue
            needed = {tc["id"] for tc in msg.tool_calls}
            found: set = set()
            for m in tail[idx + 1:]:
                if isinstance(m, ToolMessage) and hasattr(m, "tool_call_id"):
                    found.add(m.tool_call_id)
                elif not isinstance(m, ToolMessage):
                    break
            if not needed.issubset(found):
                # Drop this AI message and its (partial) results then re-scan.
                j = idx + 1
                while j < len(tail) and isinstance(tail[j], ToolMessage):
                    j += 1
                tail = tail[:idx] + tail[j:]
                changed = True
                break
    # "llm_input_messages" is passed to the model WITHOUT updating persistent state.
    return {"llm_input_messages": system_msgs + tail}


agent = create_react_agent(
    model=ChatAnthropic(model="claude-sonnet-5"),
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
    # "llm_input_messages" is model-only; it does NOT write back to persistent state.
    return {"llm_input_messages": [system] + msgs}


agent = create_react_agent(
    model=ChatAnthropic(model="claude-sonnet-5"),
    tools=[],
    pre_model_hook=inject_system_prompt,
)
```

```python
# --- Example 3: Per-message size guardrail via pre_model_hook ---
from langchain_core.messages import SystemMessage, HumanMessage

MAX_INPUT_CHARS = 12_000

_TRUNC_MARKER = " [truncated]"
_MARKER_LEN = len(_TRUNC_MARKER)


def truncate_long_messages(state: dict) -> dict:
    """Hard truncate message content that exceeds a per-message character cap.

    Handles both string content and block-based (multimodal) content so the cap
    applies regardless of how a message was constructed. The marker itself is
    counted inside the cap, so the final string never exceeds MAX_INPUT_CHARS.
    Non-text blocks (images, audio) are always passed through unchanged.
    """
    updated = []
    for msg in state.get("messages", []):
        if not hasattr(msg, "content"):
            updated.append(msg)
            continue
        content = msg.content
        if isinstance(content, str):
            if len(content) > MAX_INPUT_CHARS:
                # Slice so that slice + marker together stay within the cap.
                msg = msg.model_copy(update={
                    "content": content[:MAX_INPUT_CHARS - _MARKER_LEN] + _TRUNC_MARKER
                })
        elif isinstance(content, list):
            # Block-based content (e.g. Anthropic multimodal messages).
            # Track a shared per-message budget so the aggregate text across ALL
            # blocks cannot exceed MAX_INPUT_CHARS, not just each block individually.
            new_blocks = []
            remaining = MAX_INPUT_CHARS
            for block in content:
                if isinstance(block, str):
                    if remaining <= 0:
                        continue  # budget exhausted — skip remaining text blocks
                    if len(block) > remaining:
                        # Guard: if the budget is too small to fit the marker,
                        # just hard-truncate to remaining chars with no marker.
                        if remaining >= _MARKER_LEN:
                            block = block[:remaining - _MARKER_LEN] + _TRUNC_MARKER
                        else:
                            block = block[:remaining]
                        remaining = 0
                    else:
                        remaining -= len(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    text = block["text"]
                    if remaining <= 0:
                        # Skip exhausted text-dict blocks entirely — providers
                        # such as Anthropic reject empty {"text": ""} blocks.
                        # Non-text blocks fall through to new_blocks.append below.
                        continue
                    elif len(text) > remaining:
                        if remaining >= _MARKER_LEN:
                            block = {**block, "text": text[:remaining - _MARKER_LEN] + _TRUNC_MARKER}
                        else:
                            block = {**block, "text": text[:remaining]}
                        remaining = 0
                    else:
                        remaining -= len(text)
                # Non-text blocks (images, audio) are always appended unchanged;
                # binary content has no meaningful character count to truncate.
                new_blocks.append(block)
            msg = msg.model_copy(update={"content": new_blocks})
        updated.append(msg)
    return {"llm_input_messages": updated}


agent = create_react_agent(
    model=ChatAnthropic(model="claude-sonnet-5"),
    tools=[],
    pre_model_hook=truncate_long_messages,
)
```

**Return value rules:**
- Return `{"llm_input_messages": [...]}` to pass a **different** message list to the model without touching persistent state. This is the right key for trimming, truncating, or injecting a fresh system prompt.
- Return `{"messages": [...]}` only when you want changes to be **persisted** back into the graph's message history (goes through the `add_messages` reducer).
- Return an empty dict `{}` or `None` to pass the state through unchanged.
- Hooks **may** raise; the exception propagates out of the agent invocation. This is intentional for guards like the budget example in section 10.

---

## 2. `post_model_hook` — inspect or transform the model's response

`post_model_hook` runs immediately after each model call. It receives the state (which now includes the latest `AIMessage` from the model) and can return updates.

Common uses: logging token usage, enforcing response policies, adding metadata, early-exit guards.

```python
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

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
    model=ChatAnthropic(model="claude-sonnet-5"),
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


# tracker and agent are module-scoped — token counts accumulate across ALL invocations.
# For per-run cost, create a fresh CostTracker and agent inside a helper function.
tracker = CostTracker()

agent = create_react_agent(
    model=ChatAnthropic(model="claude-sonnet-5"),
    tools=[],
    post_model_hook=tracker,
)

result = agent.invoke({"messages": [{"role": "user", "content": "Explain LangGraph in one sentence."}]})
print(f"Cumulative cost so far: ${tracker.total_cost_usd:.6f}")
```

```python
# --- Example 3: Stop the agent early if the model response is empty ---
from langchain_core.messages import AIMessage
from langgraph.types import Command

def guard_empty_response(state: dict) -> dict | Command:
    """If the model returned an empty final response (no tool calls), end the run."""
    last_msg = state["messages"][-1]
    # Only exit early when content is empty AND there are no pending tool calls —
    # tool-calling AIMessages legitimately have empty text content.
    has_tool_calls = bool(getattr(last_msg, "tool_calls", None))
    if isinstance(last_msg, AIMessage) and not last_msg.content and not has_tool_calls:
        from langgraph.graph import END
        return Command(goto=END)
    return {}


agent = create_react_agent(
    model=ChatAnthropic(model="claude-sonnet-5"),
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
from langchain_anthropic import ChatAnthropic


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 22°C in {city}"


def pre_hook(state: dict) -> dict:
    """Ensure a system message is always present (model-only, not persisted)."""
    has_system = any(isinstance(m, SystemMessage) for m in state.get("messages", []))
    if not has_system:
        system = SystemMessage(content="Be brief. Use metric units.")
        return {"llm_input_messages": [system] + state["messages"]}
    return {}


def post_hook(state: dict) -> dict:
    """Log the model's output content length."""
    last = state["messages"][-1]
    content = getattr(last, "content", "")
    if isinstance(content, str):
        print(f"[post_hook] response length: {len(content)} chars")
    return {}


agent = create_react_agent(
    model=ChatAnthropic(model="claude-sonnet-5"),
    tools=[get_weather],
    pre_model_hook=pre_hook,
    post_model_hook=post_hook,
)

result = agent.invoke({"messages": [{"role": "user", "content": "Weather in Paris?"}]})
```

---

## 4. Per-node `error_handler` on `add_node`

Every `add_node` call accepts an `error_handler` — a node function called when the primary node raises **after all configured retries are exhausted**. It receives the state the node last received and a `NodeError` context object, and can return a partial state update (to write an error message, skip, or set a fallback value).

```python
from langgraph.graph import StateGraph, START, END
from langgraph.errors import NodeError
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


def handle_transform_error(state: PipelineState, error: NodeError) -> dict:
    """Recover from risky_transform failure gracefully.

    LangGraph passes a NodeError context object, not the raw exception.
    Access the underlying exception via error.error; the failed node name via error.node.
    """
    # Log the full exception server-side; expose only safe metadata to state.
    import logging
    logging.getLogger(__name__).error(
        "risky_transform failed on node %s: %s", error.node, error.error
    )
    return {
        "result": "",
        "error": f"Transform failed (node: {error.node}, type: {type(error.error).__name__})",
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
| `retry_on` | `default_retry_on` (transient errors only — see below) | Exception type(s) or `Callable[[Exception], bool]` |

> **`default_retry_on`** checks exceptions in this order:
> 1. `ConnectionError` → **retry** (even though `ConnectionError` is an `OSError` subclass, it is checked first and retried)
> 2. `httpx.HTTPStatusError` with 5xx status → **retry**
> 3. `requests.HTTPError` with 5xx status → **retry**
> 4. `ValueError`, `TypeError`, `ArithmeticError`, `ImportError`, `LookupError`, `NameError`, `SyntaxError`, `RuntimeError`, `ReferenceError`, `StopIteration`, `StopAsyncIteration`, `OSError` (generic) → **do not retry**
> 5. Any other exception → **retry**
>
> Pass a custom predicate — `retry_on=lambda exc: isinstance(exc, (ConnectionError, TimeoutError))` — to control exactly what triggers a retry.

---

## 6. `set_node_defaults` — graph-wide policy

Instead of passing `retry_policy`, `cache_policy`, `error_handler`, or `timeout` to every `add_node` call, set them once for the whole graph. Individual nodes can still override.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.errors import NodeError
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict


class State(TypedDict):
    query: str
    answer: str


def node_a(state: State) -> dict:
    return {"answer": f"A({state['query']})"}


def node_b(state: State) -> dict:
    return {"answer": f"B({state['answer']})"}


def global_error_handler(state: State, error: NodeError) -> dict:
    # Log the full exception server-side; return a sanitized fixed message so raw
    # exception text (which can include internal URLs or request details) is never
    # written into user-visible state.
    import logging
    logging.getLogger(__name__).error("Node %s failed: %s", error.node, error.error)
    return {"answer": f"[error] Node {error.node} could not complete — please retry."}


builder = StateGraph(State)

# set_node_defaults resolves at compile time (builder.compile()), not at add_node time,
# so call order relative to add_node does not matter.
# Note: TimeoutPolicy only applies to async nodes; omit it for sync nodes.
builder.set_node_defaults(
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.5),
    error_handler=global_error_handler,
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

You can override defaults on specific nodes by passing the policy directly to `add_node`. In a fresh builder this replaces the default for that node only — don't call `add_node` for the same name twice in the same builder:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.errors import NodeError
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict

class State(TypedDict):
    query: str
    answer: str

def node_a(state: State) -> dict:
    return {"answer": f"A({state['query']})"}

def node_b(state: State) -> dict:
    return {"answer": f"B({state['answer']})"}

def global_error_handler(state: State, error: NodeError) -> dict:
    import logging
    logging.getLogger(__name__).error("Node %s failed: %s", error.node, error.error)
    return {"answer": f"[error] Node {error.node} could not complete — please retry."}

builder2 = StateGraph(State)
builder2.set_node_defaults(
    retry_policy=RetryPolicy(max_attempts=3),
    error_handler=global_error_handler,
)

builder2.add_node("a", node_a)   # inherits: max_attempts=3
builder2.add_node(               # overrides: no retries for this node
    "b",
    node_b,
    retry_policy=RetryPolicy(max_attempts=1),
)
builder2.add_edge(START, "a")
builder2.add_edge("a", "b")
builder2.add_edge("b", END)
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


# Strategy 1: Use LangGraph's default formatted error template as a ToolMessage.
# The template reads: "Error: <exception>\nPlease fix your mistakes." — not raw str(exc).
tool_node_default = ToolNode(
    tools=[divide, fetch_data],
    handle_tool_errors=True,
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
    # Only include the exception type — raw str(exc) can leak internal URLs or
    # request data that the model (and user) should not see.
    return f"Tool call failed ({type(exc).__name__}). Retry with different arguments."

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


model = ChatAnthropic(model="claude-sonnet-5").bind_tools([get_stock_price])

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

Note: `create_react_agent` uses an extended version of this state that also includes `remaining_steps` (an internal step counter). Hook callables passed to `create_react_agent` receive this extended state.

---

## 9. `timeout` per node — `TimeoutPolicy`

Prevent runaway nodes with per-node timeouts. Async-only — sync nodes raise `ValueError` at node registration time (i.e. when `add_node` is called) if `timeout` is set.

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
# graph.ainvoke({"result": ""}) will raise langgraph.errors.NodeTimeoutError after ~5-10 s
# Note: this is NOT asyncio.TimeoutError — catch NodeTimeoutError specifically.
```

**`TimeoutPolicy` fields:**

| Field | Type | Default | Meaning |
|---|---|---|---|
| `run_timeout` | `float \| timedelta \| None` | `None` | Hard wall-clock cap. Never refreshed by progress signals. |
| `idle_timeout` | `float \| timedelta \| None` | `None` | Max silence between progress events. |
| `refresh_on` | `"auto" \| "heartbeat"` | `"auto"` | `"auto"` = refreshed by LangGraph events; `"heartbeat"` = only by `runtime.heartbeat()`. |

---

## 10. Realistic production stack

Putting it all together. Two complementary patterns: (A) `create_react_agent` with pre/post hooks for model-call interceptors, and (B) a manual `StateGraph` with `set_node_defaults` for graph-wide retry + per-node error recovery + `ToolNode` error handling.

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
    """Always prepend the canonical system prompt, stripping any pre-existing SystemMessages.

    Stripping first ensures a caller cannot bypass the canonical instructions by
    supplying a different leading SystemMessage in the invocation input.
    """
    non_system = [m for m in state.get("messages", []) if not isinstance(m, SystemMessage)]
    return {"llm_input_messages": [SYSTEM_PROMPT] + non_system}


class TokenBudget:
    """Abort the run if cumulative token usage exceeds a budget.

    Instantiate once PER INVOCATION (not at module scope) so each run
    starts with a fresh counter and concurrent calls don't race.
    """
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


# ── Invoke ─────────────────────────────────────────────────────────────────
# IMPORTANT: the budget is bound at agent-construction time, not at invoke time.
# Each call to invoke() on the SAME agent instance accumulates tokens in the SAME
# budget object. To get a truly per-invocation budget, build a fresh agent per call:

def invoke_once(user_message: str) -> str:
    """Build a fresh agent (and fresh budget) for each independent request."""
    budget = TokenBudget()   # brand-new counter for this request only
    agent = create_react_agent(
        model=ChatAnthropic(model="claude-sonnet-5"),
        tools=[search_docs, send_alert],
        pre_model_hook=enforce_system_prompt,
        post_model_hook=budget,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
    return result["messages"][-1].content

print(invoke_once("What does LangGraph do?"))
```

### Pattern B — Custom `StateGraph` with retry, error recovery, and `ToolNode`

When you build your own graph (instead of using `create_react_agent`), combine `set_node_defaults`, per-node `error_handler`, and `ToolNode(handle_tool_errors=...)`:

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.errors import NodeError
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic


@tool
def search_docs(query: str) -> str:
    """Search the internal knowledge base."""
    return f"Results for '{query}': ..."


@tool
def send_alert(message: str) -> str:
    """Send a Slack alert."""
    print(f"[ALERT] {message}")
    return "Alert sent."


model = ChatAnthropic(model="claude-sonnet-5").bind_tools([search_docs, send_alert])


def call_model(state: MessagesState) -> dict:
    return {"messages": [model.invoke(state["messages"])]}


def fallback_handler(state: MessagesState, error: NodeError) -> dict:
    """Return a safe fallback message if call_model fails."""
    import logging
    from langchain_core.messages import AIMessage
    logging.getLogger(__name__).error(
        "call_model failed on node %s: %s", error.node, error.error
    )
    return {"messages": [AIMessage(content="I encountered an error and couldn't complete your request. Please try again.")]}


def format_error(exc: Exception) -> str:
    # Only include the exception type — raw str(exc) can leak internal URLs or
    # request data that the model (and user) should not see.
    return f"Tool call failed ({type(exc).__name__}). Retry with different arguments."


tool_node = ToolNode(
    tools=[search_docs, send_alert],
    # format_error returns only the exception type name, avoiding leakage of
    # internal URLs or stack traces into ToolMessage content visible to the model.
    # Tool errors handled here are NEVER seen by the graph's RetryPolicy — only errors
    # from call_model can trigger graph-level retries.
    handle_tool_errors=format_error,
)

builder = StateGraph(MessagesState)

# Graph-wide defaults — applied to every node that doesn't override them.
# TimeoutPolicy only applies to async nodes; omit it for sync nodes like call_model.
builder.set_node_defaults(
    # Omit retry_on so it uses default_retry_on, which covers ConnectionError,
    # httpx/requests 5xx, and transport timeouts — not just ConnectionError alone.
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.5),
    error_handler=fallback_handler,
)

builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()

result = graph.invoke({"messages": [{"role": "user", "content": "Search for LangGraph docs."}]})
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

---
title: "Chapter 8 — Agent Middleware"
description: "Customise a create_agent agent with middleware: hooks before and after model calls, wrappers around model and tool calls, and the built-in middleware for approvals, summarisation, limits, retries and PII."
framework: "langgraph"
language: "python"
sidebar:
  label: "8 · Agent middleware"
  order: 8
---

**What you'll learn:** how to change what a `create_agent` agent does on each step (trim history, pick a model, handle tool errors, stop the loop, pause for approval) with middleware, and which middleware ships with `langchain`.

Checked against `langchain` %%version=langchain%% and `langgraph` %%version=langgraph%%. **Time:** ~25 minutes.

> Prereqs: [Chapter 2](/langgraph-guide/python/chapter-02-simple-agents/) (`create_agent`) and [Chapter 4](/langgraph-guide/python/chapter-04-tools/) (tools).

Middleware is a `langchain` feature: `create_agent` and `langchain.agents.middleware` live in the `langchain` package (`pip install langchain`), and `create_agent` compiles to a LangGraph graph. If you build your own `StateGraph` instead, the equivalents are node policies and error handlers; see [In your own graph](#in-your-own-graph) below.

## Where middleware runs

`create_agent` runs a loop: call the model, run any tools it asked for, repeat until the model answers without tool calls. Middleware attaches at fixed points in that loop.

| Hook | Runs | Typical use |
|---|---|---|
| `before_agent` | Once, when the run starts | Load user data, validate input |
| `before_model` | Before every model call; can update state | Trim or summarise history |
| `wrap_model_call` | Around every model call; gets the request and a `handler` | Change model, prompt, tools or messages for one call; retries; fallbacks; logging |
| `after_model` | After every model call; can update state or jump | Guardrails on the response, counters |
| `wrap_tool_call` | Around every tool execution | Tool error handling, retries, approval |
| `after_agent` | Once, when the run ends | Clean-up, logging |

With several middleware in `middleware=[A, B]`, the `before_*` hooks run in list order (A then B), the `after_*` hooks in reverse (B then A), and the `wrap_*` hooks nest with A outermost.

## Writing middleware with decorators

Each decorator turns a function into a middleware object. This agent uses five of them:

```python
from dataclasses import dataclass

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (
    before_model, after_model, dynamic_prompt, wrap_model_call, wrap_tool_call,
    ModelRequest, ModelResponse,
)
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, RemoveMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime


@dataclass
class Context:
    tier: str = "standard"


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by ID."""
    if order_id == "0":
        raise ValueError("order not found")
    return f"Order {order_id}: shipped"


# 1. before_model: runs before every model call and can update state.
@before_model
def trim_history(state: AgentState, runtime: Runtime) -> dict | None:
    messages = state["messages"]
    if len(messages) <= 30:
        return None
    # Keep the first message and the 20 most recent ones (permanently).
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), messages[0], *messages[-20:]]}


# 2. dynamic_prompt: build the system prompt per call from context or state.
@dynamic_prompt
def prompt_for_tier(request: ModelRequest) -> str:
    return f"You are a support agent. Customer tier: {request.runtime.context.tier}. Be brief."


# 3. wrap_model_call: wraps the model call itself; override() changes the request.
small_model = init_chat_model("anthropic:%%model=anthropic-small%%")


@wrap_model_call
def cheap_model_for_short_chats(request: ModelRequest, handler) -> ModelResponse:
    if len(request.messages) < 4:
        request = request.override(model=small_model)
    return handler(request)


# 4. wrap_tool_call: wraps each tool execution.
@wrap_tool_call
def tool_errors_to_messages(request, handler):
    try:
        return handler(request)
    except Exception as exc:
        return ToolMessage(content=f"Tool failed: {exc}. Try another approach.",
                           tool_call_id=request.tool_call["id"])


# 5. after_model: inspect the response; jump_to="end" stops the loop.
@after_model(can_jump_to=["end"])
def block_secrets(state: AgentState, runtime: Runtime) -> dict | None:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and "API_KEY" in str(last.content):
        return {"messages": [AIMessage("I can't share that.")], "jump_to": "end"}
    return None


agent = create_agent(
    "anthropic:%%model=anthropic%%",
    tools=[lookup_order],
    middleware=[trim_history, prompt_for_tier, cheap_model_for_short_chats,
                tool_errors_to_messages, block_secrets],
    context_schema=Context,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Where is order 42?"}]},
    context=Context(tier="gold"),
)
print(result["messages"][-1].content)
```

Notes on the hooks used:

- **`before_model` / `after_model`** return `None` (no change) or a state update. Updates go through the state's reducers, which is why trimming uses `RemoveMessage(id=REMOVE_ALL_MESSAGES)` followed by the messages to keep.
- **`jump_to`** is allowed only when the hook declares it: `@after_model(can_jump_to=["end"])`. Targets are `"end"`, `"model"` and `"tools"`.
- **`wrap_model_call`** receives a `ModelRequest` (`model`, `messages`, `system_message`, `tools`, `tool_choice`, `response_format`, `state`, `runtime`, `model_settings`). `request.override(...)` returns a modified copy, so the change applies to this call only.
- **`dynamic_prompt`** is a shortcut for a `wrap_model_call` that replaces the system prompt.
- **`wrap_tool_call`** receives a `ToolCallRequest` (`request.tool_call` is the call dict) and returns a `ToolMessage` or a `Command`.

Async agents (`ainvoke`, `astream`) use async functions with the same decorators.

## Class-based middleware

Subclass `AgentMiddleware` when a middleware needs configuration, several hooks, or its own state keys. Declaring `state_schema` adds fields to the agent's state:

```python
from typing import Any
from typing_extensions import NotRequired

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langgraph.runtime import Runtime


class UsageState(AgentState):
    model_calls: NotRequired[int]


class UsageMiddleware(AgentMiddleware[UsageState]):
    """Counts model calls in state and shows the model only recent history."""

    state_schema = UsageState

    def __init__(self, window: int = 20) -> None:
        super().__init__()
        self.window = window

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        # Trim what the model sees for this call only; state keeps every message.
        return handler(request.override(messages=request.messages[-self.window:]))

    def after_model(self, state: UsageState, runtime: Runtime) -> dict[str, Any] | None:
        return {"model_calls": state.get("model_calls", 0) + 1}


agent = create_agent("anthropic:%%model=anthropic%%", tools=[], middleware=[UsageMiddleware(window=10)])
result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})
print(result["model_calls"])   # 1
```

## Built-in middleware

All of these are in `langchain.agents.middleware`.

| Middleware | What it does |
|---|---|
| `HumanInTheLoopMiddleware(interrupt_on={...})` | Pauses before chosen tools run, for approve / edit / reject / respond decisions. Needs a checkpointer. |
| `SummarizationMiddleware(model, trigger=..., keep=...)` | Replaces older messages with a summary once a token, message-count or context-fraction threshold is reached |
| `ContextEditingMiddleware(edits=[ClearToolUsesEdit(...)])` | Clears old tool results from the context instead of summarising |
| `ModelCallLimitMiddleware(run_limit=..., thread_limit=...)` | Caps model calls per run or per thread; `exit_behavior="end"` or `"error"` |
| `ToolCallLimitMiddleware(tool_name=..., run_limit=..., thread_limit=...)` | Caps calls to one tool (or all tools) |
| `ModelRetryMiddleware(max_retries=...)`, `ToolRetryMiddleware(...)` | Retries failed model or tool calls with exponential backoff |
| `ModelFallbackMiddleware(first_model, *others)` | Tries other models in order when the main one fails |
| `ToolErrorMiddleware(on_error=...)` | Turns tool exceptions into messages the model can read |
| `PIIMiddleware(pii_type, strategy=...)` | Detects e-mail addresses, card numbers, IPs, MAC addresses, URLs or a custom pattern, and blocks, redacts, masks or hashes them |
| `LLMToolSelectorMiddleware(model=..., max_tools=...)` | Uses a small model to choose which tools to offer when there are many |
| `TodoListMiddleware()` | Gives the agent a `write_todos` planning tool |
| `ShellToolMiddleware`, `FilesystemFileSearchMiddleware`, `LLMToolEmulator` | Shell and file-search tools; emulated tools for testing |

A support agent that combines several:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware, ModelCallLimitMiddleware, ModelFallbackMiddleware,
    ModelRetryMiddleware, PIIMiddleware, SummarizationMiddleware, ToolCallLimitMiddleware,
)
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"sent to {to}"


@tool
def search_docs(query: str) -> str:
    """Search internal documentation."""
    return "Refunds take 5 working days."


agent = create_agent(
    "anthropic:%%model=anthropic%%",
    tools=[send_email, search_docs],
    middleware=[
        # Summarise older messages once the history passes ~4,000 tokens; keep the last 20.
        SummarizationMiddleware(model="anthropic:%%model=anthropic-small%%",
                                trigger=("tokens", 4000), keep=("messages", 20)),
        # Mask card numbers in user input before the model sees them.
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        # Retry failed model calls, then fall back to another model.
        ModelRetryMiddleware(max_retries=2),
        ModelFallbackMiddleware("anthropic:%%model=anthropic-small%%"),
        # Hard limits: at most 10 model calls per run, 3 e-mails per thread.
        ModelCallLimitMiddleware(run_limit=10, exit_behavior="end"),
        ToolCallLimitMiddleware(tool_name="send_email", thread_limit=3),
        # Pause for approval before send_email runs; search_docs runs freely.
        HumanInTheLoopMiddleware(interrupt_on={
            "send_email": {"allowed_decisions": ["approve", "edit", "reject"]},
            "search_docs": False,
        }),
    ],
    checkpointer=InMemorySaver(),   # required for the approval pause
)

config = {"configurable": {"thread_id": "t1"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Email the refund policy to sam@example.com"}]},
    config,
)

# The run pauses before send_email; the interrupt describes the pending call.
request = result["__interrupt__"][0].value
print(request["action_requests"][0]["name"], request["action_requests"][0]["args"])

# Resume with one decision per pending call: approve, edit or reject.
result = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
print(result["messages"][-1].content)
```

Each decision in `resume={"decisions": [...]}` is one of:

- `{"type": "approve"}`
- `{"type": "edit", "edited_action": {"name": "send_email", "args": {...}}}`
- `{"type": "reject", "message": "Not now"}`: the tool does not run and the message goes back to the model
- `{"type": "respond", "message": "..."}`

[Chapter 7](/langgraph-guide/python/chapter-07-human-in-the-loop/) covers `interrupt()` itself, which this middleware uses.

## In your own graph

Middleware belongs to `create_agent`. In a hand-built `StateGraph`, a node is your own function, so the same needs are met differently:

| Need | In a `StateGraph` |
|---|---|
| Retry, timeout or cache a step | `add_node(..., retry_policy=, timeout=, cache_policy=)` and `set_node_defaults(...)`: see [Node policies](/langgraph-guide/python/reference-state-graph/#node-policies) |
| Recover from a failed node | `add_node(..., error_handler=fn)` with a `NodeError` parameter: see [Error handlers](/langgraph-guide/python/reference-state-graph/#error-handlers) |
| Tool errors | `ToolNode(handle_tool_errors=...)` or `ToolNode(wrap_tool_call=...)`: see [Chapter 4](/langgraph-guide/python/chapter-04-tools/) |
| Trim history, guardrails | Do it in the node, or add a node before or after the model node |
| Approval | `interrupt()` in a node: see [Chapter 7](/langgraph-guide/python/chapter-07-human-in-the-loop/) |

You can also use a `create_agent` agent as a node (or subgraph) inside a larger `StateGraph`, and keep its middleware.

## Migrating from `create_react_agent` hooks

`langgraph.prebuilt.create_react_agent` is deprecated (it warns with `LangGraphDeprecatedSinceV10` and is due for removal in 2.0). Its `pre_model_hook` and `post_model_hook` arguments map onto middleware:

| `create_react_agent` | `create_agent` |
|---|---|
| `prompt=` | `system_prompt=` or `@dynamic_prompt` |
| `pre_model_hook` returning `{"messages": ...}` | `@before_model` returning a state update |
| `pre_model_hook` returning `{"llm_input_messages": ...}` (model-only view) | `@wrap_model_call` with `request.override(messages=...)` |
| `post_model_hook` | `@after_model` |
| `state_schema=` (must include `remaining_steps`) | `state_schema=` subclassing `langchain.agents.AgentState`, or a middleware `state_schema` |
| `interrupt_before=["tools"]` | `HumanInTheLoopMiddleware` |
| `response_format=` | `response_format=` |

For reference, the legacy form looks like this:

```python
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langgraph.prebuilt import create_react_agent   # deprecated: emits LangGraphDeprecatedSinceV10


def pre_model_hook(state):
    trimmed = trim_messages(state["messages"], strategy="last", max_tokens=2000,
                            token_counter=count_tokens_approximately, start_on="human")
    return {"llm_input_messages": trimmed}   # what the model sees; state is unchanged


legacy = create_react_agent("anthropic:%%model=anthropic%%", tools=[], pre_model_hook=pre_model_hook)
print(legacy.invoke({"messages": [{"role": "user", "content": "Hi"}]})["messages"][-1].content)
```

Next: [Chapter 9 — Advanced patterns](/langgraph-guide/python/chapter-09-advanced-patterns/).

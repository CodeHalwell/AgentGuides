---
title: "Chapter 8 — Middleware"
description: "Cross-cutting hooks that wrap every model and tool call — PII redaction, summarization, retries, HITL, tool limits — via the real langchain.agents middleware system."
framework: langgraph
language: python
sidebar:
  label: "8 · Middleware"
  order: 8
---

# Chapter 8 — Middleware

**What you'll learn:** how to intercept every model and tool call in a LangGraph agent without rewriting nodes. LangChain's new agent middleware system (built on LangGraph) gives you a stack of composable middlewares — PII redaction, summarization, retries, human-in-the-loop, tool-call limits — applied by passing a `middleware=[...]` list to `create_agent`.

**Time:** ~20 minutes.

> Prereqs: [Chapter 4 — Tools](/langgraph-guide/python/chapter-04-tools/).

> **Heads-up.** Middleware in LangGraph lives in the `langchain` package, not `langgraph` directly. The split reflects how the two work together: `langgraph` provides the runtime; `langchain.agents` provides the agent factory and its middleware stack. Install with `pip install langchain langgraph`.

## Where middleware lives

| Component | Import | Role |
|---|---|---|
| `create_agent` | `from langchain.agents import create_agent` | Agent factory (replaces deprecated `create_react_agent`) |
| `AgentMiddleware` | `from langchain.agents.middleware import AgentMiddleware` | Base class for custom middleware |
| `before_model`, `after_model`, `wrap_model_call` | `from langchain.agents.middleware import before_model, after_model, wrap_model_call` | Decorator-style hooks around model calls |
| `before_agent`, `after_agent` | same module | Decorator-style hooks around the whole agent loop |
| `wrap_tool_call` | same module | Decorator-style hook around each tool call |
| Prebuilt middleware (see table below) | same module | Production-grade middleware you can drop in |

## Minimum viable middleware

Three ways to write middleware, from easiest to most flexible:

### 1. Decorator hooks (`before_model`, `after_model`)

```python
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model
from langchain.agents.middleware.types import ModelRequest, ModelResponse

@before_model
def log_request(request: ModelRequest) -> ModelRequest:
    print(f"→ calling model with {len(request.messages)} messages")
    return request

@after_model
def log_response(response: ModelResponse) -> ModelResponse:
    usage = getattr(response.message, "usage_metadata", None)
    if usage:
        print(f"← tokens in={usage['input_tokens']} out={usage['output_tokens']}")
    return response

agent = create_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    tools=[...],
    middleware=[log_request, log_response],
)
```

### 2. `wrap_model_call` (true "around" hook)

Use this when you need to transform the request, inspect the response, and/or short-circuit:

```python
from langchain.agents.middleware import wrap_model_call
from langchain.agents.middleware.types import ModelRequest, ModelResponse

@wrap_model_call
def rate_limit(request: ModelRequest, handler) -> ModelResponse:
    check_quota(request.runtime.context.user_id)  # raises on exceed
    return handler(request)  # delegate to next middleware / the model
```

### 3. `AgentMiddleware` subclass (stateful / multi-hook)

```python
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse

class CostTracker(AgentMiddleware):
    """Sum token usage across every model call in the run."""
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def before_model(self, request: ModelRequest) -> ModelRequest:
        return request  # no-op; override only the hooks you need

    def after_model(self, response: ModelResponse) -> ModelResponse:
        usage = getattr(response.message, "usage_metadata", {}) or {}
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        return response

tracker = CostTracker()
agent = create_agent(model="...", tools=[...], middleware=[tracker])
```

## Prebuilt middleware

`langchain.agents.middleware` ships production-grade middleware. Import and drop into your `middleware=[...]` list — most require zero custom code.

| Middleware | What it does |
|---|---|
| `PIIMiddleware(...)` | Detects and redacts PII (names, emails, phone numbers, addresses) from messages and tool outputs before they reach the model. |
| `SummarizationMiddleware(...)` | When the conversation exceeds a token threshold, compacts older messages into a summary so you never blow the context window. |
| `ContextEditingMiddleware(...)` | Edit conversation context in flight — strip system messages, prepend instructions, truncate. |
| `ModelCallLimitMiddleware(max_calls=N)` | Aborts the run after N model calls — the simplest runaway-cost guardrail. |
| `ModelFallbackMiddleware([a, b, c])` | Falls back from primary model to a cheaper/stronger one if the primary errors. |
| `ModelRetryMiddleware(...)` | Retries transient model errors with backoff. |
| `ToolCallLimitMiddleware(limits={...})` | Per-tool invocation caps to prevent loops. |
| `ToolRetryMiddleware(...)` | Retries transient tool errors. |
| `HumanInTheLoopMiddleware(interrupt_on={...})` | Pauses for human approval on specific tool calls. This **is** how HITL is usually wired in the new API — chapter 7's `interrupt()` is the lower-level primitive. |
| `LLMToolSelectorMiddleware(...)` | Uses an LLM to pre-select which tools are visible per turn — stops the model picking from 50 tools when 3 are relevant. |
| `LLMToolEmulator(...)` | Useful for tests: runs tools through an LLM simulator instead of hitting the real endpoint. |
| `TodoListMiddleware()` | Maintains a model-visible TODO list across turns. |
| `ShellToolMiddleware(...)` | Executes shell commands with a sandbox policy (`CodexSandboxExecutionPolicy`, `DockerExecutionPolicy`, `HostExecutionPolicy`). |
| `FilesystemFileSearchMiddleware(...)` | Exposes filesystem search as a tool with a root-directory constraint. |

## A realistic stack

Production agents usually compose several at once. Here's a safety + cost envelope:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    PIIMiddleware,
    SummarizationMiddleware,
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    HumanInTheLoopMiddleware,
)

agent = create_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    tools=[search, calculator, send_email],
    middleware=[
        # Input side — sanitise before anything else runs
        PIIMiddleware(redact=True),
        # Context management — keep the window under control
        SummarizationMiddleware(max_tokens=8000, keep_last=10),
        # Resilience — retry transient errors once
        ModelRetryMiddleware(max_retries=1),
        # Budgeting — stop runaway loops
        ModelCallLimitMiddleware(max_calls=20),
        ToolCallLimitMiddleware(limits={"send_email": 3}),
        # Safety gate — pause before irreversible tools
        HumanInTheLoopMiddleware(interrupt_on={"send_email"}),
    ],
)
```

## Legacy: `create_react_agent` hooks

If you're still on `langgraph.prebuilt.create_react_agent` (deprecated but functional in 1.1.x), it exposes two kwargs — `pre_model_hook` and `post_model_hook` — that accept a `Runnable` or plain callable transforming the state. These are **parameters on the function**, not standalone decorators:

```python
from langgraph.prebuilt import create_react_agent

def trim_messages(state):
    # keep only the last 20 messages
    return {"messages": state["messages"][-20:]}

agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    tools=[...],
    pre_model_hook=trim_messages,
    post_model_hook=None,
)
```

Migrate to `create_agent` + the middleware list when you can — it composes better and gives you the prebuilt middleware in the table above.

## When to reach for middleware vs. a node

- **Middleware** — cross-cutting concerns that apply to *every* model or tool call (PII, cost, retries, limits, HITL approval).
- **A dedicated node** — transformations that are part of your business logic graph (classification, routing, enrichment). Don't hide those in middleware; they belong in the graph.

The rule of thumb: if a reader tracing your graph would be surprised the behaviour exists, it doesn't belong in middleware.

## What this chapter replaced

Earlier drafts of this chapter documented `@pre_model_hook` / `@post_model_hook` decorators imported from `langgraph.llm_hooks`. **That module does not exist.** The real primitives are `create_agent(middleware=[...])` in `langchain.agents` (new, recommended) and `create_react_agent(pre_model_hook=..., post_model_hook=...)` in `langgraph.prebuilt` (older, deprecated). This chapter has been rewritten against the installed packages (`langchain` and `langgraph==1.1.10`).

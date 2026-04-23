---
title: "Microsoft Agent Framework (Python) — Tools"
description: "The @tool decorator, FunctionTool class, approval gates, explicit schemas, runtime context, invocation limits, and result parsers. All verified against agent-framework-core 1.1.0."
framework: microsoft-agent-framework
language: python
---

# Tools — Python

Tools are how agents call back into your code. Agent Framework offers two ways to define them — the `@tool` decorator for the common case and the `FunctionTool` class for advanced construction. Both target the same `FunctionTool` object that the agent sees.

This page covers first-party function tools. For MCP tools see the [MCP page](./microsoft_agent_framework_python_mcp/); for skill-based tools see the [Skills page](./microsoft_agent_framework_python_skills/).

Verified against `agent-framework-core==1.1.0` (`agent_framework._tools`).

## Minimal `@tool`

```python
from typing import Annotated
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient


@tool
def get_weather(
    location: Annotated[str, "The city or region name"],
    unit: Annotated[str, "celsius or fahrenheit"] = "celsius",
) -> str:
    """Get the current weather for a location."""
    return f"{location}: 22°{unit[0].upper()}"


agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a weather assistant.",
    tools=[get_weather],
)

response = await agent.run("What's the weather in Amsterdam?")
```

What the decorator does automatically:

- Uses the function name as the tool name.
- Uses the docstring as the tool description.
- Builds a Pydantic input model from the signature.
- Converts `Annotated[T, "..."]` hints into parameter descriptions in the JSON schema.
- Wraps sync and async functions identically.

## Customising via decorator args

```python
from pydantic import BaseModel, Field


@tool(
    name="weather_tool",
    description="Returns current conditions for a city.",
    approval_mode="never_require",   # or "always_require"
    max_invocations=50,              # lifetime cap on this tool instance
    max_invocation_exceptions=3,     # stop after N errors
)
async def get_weather(
    location: Annotated[str, Field(description="City name, e.g. 'Seattle'.")],
) -> str:
    ...
```

`@tool` accepts the full set of `FunctionTool` options — see the FunctionTool section below for the rest.

## Explicit schemas

Skip signature inference by passing `schema=`. Useful when you want to present a narrower or differently shaped schema to the model than the function exposes.

```python
from pydantic import BaseModel, Field


class WeatherInput(BaseModel):
    location: Annotated[str, Field(description="City name")]
    unit: Annotated[str, Field(description="celsius|fahrenheit")] = "celsius"


@tool(schema=WeatherInput)
def get_weather(location: str, unit: str = "celsius") -> str:
    return f"{location}: 22 {unit}"
```

`schema=` also accepts a JSON schema dict (flat object with `properties`) for cases where you're assembling the schema elsewhere:

```python
@tool(
    name="get_weather",
    schema={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    },
)
def get_weather(location: str, unit: str = "celsius") -> str: ...
```

## Runtime context — hidden parameters

Add a `FunctionInvocationContext` parameter anywhere in the signature to receive per-call state. It's hidden from the JSON schema the model sees.

```python
from agent_framework import tool, FunctionInvocationContext


@tool
def get_user_orders(
    status: Annotated[str, "pending|shipped|delivered"],
    ctx: FunctionInvocationContext,
) -> str:
    user_id = ctx.kwargs.get("user_id")
    session_id = ctx.session.id if ctx.session else "no-session"
    # …run your query…
    return f"{user_id} has 3 {status} orders"


# kwargs forwarded from the outer agent.run(...) call:
await agent.run(
    "Where are my orders?",
    function_invocation_kwargs={"user_id": "u-123"},
)
```

`ctx.metadata` also flows from any function middleware in the pipeline — see the [Middleware page](./microsoft_agent_framework_python_middleware/).

## Approval gates

```python
@tool(approval_mode="always_require")
def delete_file(path: str) -> str:
    import os
    os.remove(path)
    return f"deleted {path}"
```

Per-tool gate values:

- `"never_require"` (default) — always runs.
- `"always_require"` — pauses and emits a `function_approval_request` event; caller approves via `event.data.to_function_approval_response(approved=True)`.

See the [Human-in-the-loop page](./microsoft_agent_framework_python_hitl/#tool-approval) for the approval loop pattern.

## Result parsing

By default, the return value is `str(...)`-ed (or JSON-serialised for dataclasses and Pydantic models). Override with `result_parser=` to emit multi-part content, attach images, or trim long output.

The framework uses a single unified `Content` class with classmethod constructors (`Content.from_text`, `Content.from_uri`, `Content.from_data`, `Content.from_error`, …). A result parser returns either a string or a `list[Content]`:

```python
from agent_framework import tool, Content


def parse_weather_result(result: dict) -> list[Content]:
    parts: list[Content] = [
        Content.from_text(f"{result['temp']}°C — {result['summary']}")
    ]
    if icon_url := result.get("icon_url"):
        parts.append(Content.from_uri(uri=icon_url, media_type="image/png"))
    return parts


@tool(result_parser=parse_weather_result)
def get_weather(location: str) -> dict:
    return {
        "temp": 22,
        "summary": "Sunny",
        "icon_url": "https://cdn.example.com/weather/sunny.png",
    }
```

Return a `str` to keep things simple — the framework wraps it in a single `Content.from_text(...)` for you. Return `list[Content]` only when you need multi-part output (image + caption, JSON + human-readable summary).

### Trimming verbose tool output

The model does not need the whole body of a 400-row SQL result — shrink it before the tokens land in context:

```python
import json
from agent_framework import tool, Content


def top_n_preview(result: list[dict]) -> str:
    head = result[:10]
    summary = f"{len(result)} rows (showing first 10)\n"
    return summary + json.dumps(head, default=str, indent=2)


@tool(result_parser=top_n_preview)
def run_query(sql: str) -> list[dict]:
    return execute(sql)                # might return thousands of rows
```

## Invocation limits

`max_invocations` caps the **lifetime** of a tool instance. For module-level tools in long-running servers, either:

- Reset explicitly: `get_weather.invocation_count = 0`
- Use per-request limits instead: pass `FunctionInvocationConfiguration(max_function_calls=3)` to `agent.run(...)`.

`max_invocation_exceptions` puts a ceiling on how many times a tool can error before the agent stops calling it — a simple circuit breaker.

### Per-request tool-loop caps

`FunctionInvocationConfiguration` is the authoritative knob for runaway tool calls on a single `agent.run(...)`. Two levers that work together:

- `max_iterations` caps the number of **model roundtrips** in the tool loop. Each roundtrip may contain several parallel tool calls, so this alone does not bound total executions.
- `max_function_calls` caps the **total number of tool executions** across all iterations. This is the primary cost guard.

```python
from agent_framework import FunctionInvocationConfiguration

config: FunctionInvocationConfiguration = {
    "max_iterations": 5,
    "max_function_calls": 20,
    "max_consecutive_errors_per_request": 3,
    "terminate_on_unknown_calls": True,
    "include_detailed_errors": False,        # avoid leaking stack traces to the model
}

await agent.run(
    "Research the order pipeline",
    function_invocation_configuration=config,
)
```

Applied at the chat-client level, the same settings act as a default for every run:

```python
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient()
client.function_invocation_configuration["max_function_calls"] = 20
```

`max_function_calls` is a **best-effort** limit — it's checked *after* each batch of parallel calls completes. A single iteration that emits 20 parallel calls will run all 20 even if the limit is 10; the next iteration then bails out. Combine with `max_iterations` to bound worst-case wall time.

## Declaration-only tools

Define a tool the model can *reason about* but don't implement it — useful for client-side rendering, where the actual execution happens in a frontend:

```python
from agent_framework import FunctionTool

request_user_location = FunctionTool(
    name="request_user_location",
    description="Ask the user to share their GPS location.",
    func=None,                         # no implementation in Python
    input_model={                      # tell the model what args to produce
        "type": "object",
        "properties": {"reason": {"type": "string"}},
        "required": ["reason"],
    },
)
```

The model will emit a function call; you intercept it and handle it in your UI layer instead of letting the framework auto-invoke.

## `FunctionTool` direct construction

Equivalent to `@tool`, but useful when you're building tools dynamically (e.g. from a database of available APIs):

```python
from agent_framework import FunctionTool


def fetch(endpoint: str) -> str:
    ...


fetcher = FunctionTool(
    name="fetch",
    description="HTTP GET an internal endpoint",
    func=fetch,
    approval_mode="always_require",
    max_invocations=100,
)

agent = Agent(client=OpenAIChatClient(), tools=[fetcher])
```

### Generating tools from a spec

Build tools at startup from a config table, a JSON spec, or an OpenAPI document — the same `FunctionTool` surface handles both the declaration and the callable:

```python
from agent_framework import FunctionTool, Agent
from agent_framework.openai import OpenAIChatClient

API_SPECS = [
    {"name": "list_customers", "endpoint": "/customers", "description": "List all customers"},
    {"name": "get_customer", "endpoint": "/customers/{id}", "description": "Fetch one customer"},
]


def make_tool(spec: dict) -> FunctionTool:
    async def call(**kwargs) -> str:
        # Late-bind the endpoint template to runtime args
        url = spec["endpoint"].format(**kwargs)
        return await http_get(url)

    call.__name__ = spec["name"]                     # important — drives schema naming

    return FunctionTool(
        name=spec["name"],
        description=spec["description"],
        func=call,
        input_model={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": [] if "{id}" not in spec["endpoint"] else ["id"],
        },
    )


tools = [make_tool(s) for s in API_SPECS]
agent = Agent(client=OpenAIChatClient(), tools=tools)
```

This keeps the agent definition stable — you add a new row to `API_SPECS`, redeploy, and the model automatically sees the new tool.

## Passing tools to the agent

Tools go on the constructor or on the run call. Run-level tools are additive for that one run.

```python
agent = Agent(client=OpenAIChatClient(), tools=[get_weather, search_web])

# Run-level override/addition
await agent.run("Find a Thai restaurant", tools=[search_restaurants])
```

Supported values: single tool, list of tools, or a mix of function tools, `MCPStdioTool`/`MCPStreamableHTTPTool`, provider-hosted tools (Bedrock Guardrails, Foundry toolbox), and raw `@tool`-decorated callables.

## Error handling

Raise to signal failure — the framework surfaces the exception message to the model so it can recover:

```python
@tool
def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero — please ask for a non-zero divisor.")
    return a / b
```

For transient failures, add a `FunctionMiddleware` retry wrapper (see [Middleware → Retrying a failed tool call](./microsoft_agent_framework_python_middleware/#retrying-a-failed-tool-call)).

## Streaming tool outputs

Chat clients that support rich function output can stream content as the tool runs. Use an async generator and return `ResponseStream`-compatible content — or more commonly, just return the final value and rely on the model's own streaming.

## Patterns

**One agent, many backends.** Keep tool signatures stable and swap implementations via DI. Register the same set with every agent.

**Tenant-scoped data.** Pass `function_invocation_kwargs={"tenant_id": ...}` on `agent.run(...)` and read it from `FunctionInvocationContext.kwargs` — tools stay tenant-agnostic.

**Dangerous operations.** `approval_mode="always_require"` + function middleware that logs proposed arguments gives you an approvals audit log.

**Server-side tools vs client-side UI.** Use declaration-only `FunctionTool(func=None)` for UI rendering, real `@tool`-decorated functions for server-side actions — the model treats them identically.

**Hosted tools.** When the provider exposes hosted tools (Bedrock Guardrails, Foundry toolbox, OpenAI file-search, web-search, code-interpreter) the chat client exposes `SupportsWebSearchTool`, `SupportsFileSearchTool`, `SupportsCodeInterpreterTool` protocols — pass provider-specific tool descriptors in the same `tools=[...]` list.

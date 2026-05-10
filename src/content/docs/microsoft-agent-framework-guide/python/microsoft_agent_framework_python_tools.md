---
title: "Microsoft Agent Framework (Python) — Tools"
description: "The @tool decorator, FunctionTool class, approval gates, explicit schemas, runtime context, invocation limits, as_tool() / as_mcp_server() composition, and result parsers. All verified against agent-framework-core 1.3.0."
framework: microsoft-agent-framework
language: python
---

# Tools — Python

Tools are how agents call back into your code. Agent Framework offers two ways to define them — the `@tool` decorator for the common case and the `FunctionTool` class for advanced construction. Both target the same `FunctionTool` object that the agent sees.

This page covers first-party function tools. For MCP tools see the [MCP page](./microsoft_agent_framework_python_mcp/); for skill-based tools see the [Skills page](./microsoft_agent_framework_python_skills/).

Verified against `agent-framework-core==1.3.0` (`agent_framework._tools`).

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
- Use per-request limits instead: set `client.function_invocation_configuration["max_function_calls"] = 3` at the client level (see [Per-request tool-loop caps](#per-request-tool-loop-caps) below).

`max_invocation_exceptions` puts a ceiling on how many times a tool can error before the agent stops calling it — a simple circuit breaker.

### Reading `invocation_count` for cheap dashboards

Every `FunctionTool` increments a public `invocation_count` integer on each call. Sample it from middleware (or a periodic job) without holding any state of your own:

```python
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient


@tool
def fetch_inventory(sku: str) -> str:
    return query_inventory(sku)


agent = Agent(client=OpenAIChatClient(), tools=[fetch_inventory])

# After a batch of runs:
print(f"fetch_inventory called {fetch_inventory.invocation_count} times")
fetch_inventory.invocation_count = 0    # reset for the next reporting window
```

Combine with `max_invocations=N` to cap **per-process** spend on expensive tools (e.g. paid third-party APIs) without setting a per-request `FunctionInvocationConfiguration`.

### Per-request tool-loop caps

`FunctionInvocationConfiguration` is a `TypedDict` that configures the tool execution loop. Apply it at the **chat-client level** to set defaults for every run on that client, or override per-agent with `default_options`:

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

# Configure at the client level — applies to every agent using this client
client = OpenAIChatClient()
client.function_invocation_configuration["max_iterations"] = 5
client.function_invocation_configuration["max_function_calls"] = 20
client.function_invocation_configuration["max_consecutive_errors_per_request"] = 3
client.function_invocation_configuration["terminate_on_unknown_calls"] = True
client.function_invocation_configuration["include_detailed_errors"] = False

agent = Agent(client=client, instructions="You are a research assistant.")
await agent.run("Research the order pipeline")
```

All `FunctionInvocationConfiguration` keys (all optional):

| Key | Effect | Default |
|---|---|---|
| `enabled` | Master switch for the tool loop | `True` |
| `max_iterations` | Max LLM roundtrips per request | `100` |
| `max_function_calls` | Max total tool executions per request | `None` (unlimited) |
| `max_consecutive_errors_per_request` | Halt after N consecutive tool errors | framework default |
| `terminate_on_unknown_calls` | Raise if model calls an unregistered tool | `False` |
| `include_detailed_errors` | Include stack traces in tool error messages sent to model | `False` |
| `additional_tools` | Extra tools available but not advertised to the model | `[]` |

`max_function_calls` is a **best-effort** limit — it's checked *after* each batch of parallel calls completes. A single iteration that emits 20 parallel calls will run all 20 even if the limit is 10; the next iteration then bails out. Combine with `max_iterations` to bound worst-case wall time.

## Tool classification — `kind`

`kind` is a free-form string the framework propagates to provider adapters and observability layers. The first-party providers use it to decide how each tool is rendered to the model — for example, OpenAI's Responses API may surface `kind="hosted"` tools as a different shape than ordinary function tools. Use the same string to filter tools in your own dashboards or middleware:

```python
from agent_framework import tool


@tool(kind="readonly", additional_properties={"team": "platform", "owner": "search"})
def search_kb(query: str) -> str:
    """Read-only access to the internal knowledge base."""
    ...


@tool(kind="mutating", additional_properties={"team": "platform", "requires_audit": True})
def archive_ticket(ticket_id: str) -> str:
    """Move the ticket to long-term storage."""
    ...
```

Read both fields from middleware to enforce policy:

```python
from agent_framework import FunctionMiddleware, MiddlewareTermination


class ReadOnlyGuard(FunctionMiddleware):
    """Block mutating tools unless the caller passed approval=True."""

    async def process(self, context, call_next) -> None:
        kind = context.function.kind
        approved = context.kwargs.get("approval") is True
        if kind == "mutating" and not approved:
            raise MiddlewareTermination(
                f"{context.function.name} is mutating; approval=True required",
            )
        await call_next()
```

`additional_properties` is just a `dict[str, Any]` — drop anything in there that travels with the tool: cost class, owning team, downstream rate limit headers, audit category. The framework never inspects the contents; it's a slot for your own metadata so you don't have to maintain a parallel registry.

## Declaration-only tools

Define a tool the model can *reason about* without a server-side implementation. The model will emit a function call; you intercept it and handle it elsewhere — in a browser, a native app, or another service.

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

When `func=None` is combined with no `input_model`, the tool is valid but presents an empty schema to the model (zero parameters). Always pass `input_model` for declaration-only tools so the model knows what arguments to emit.

### The `declaration_only` property

Inspect whether a tool has an implementation at runtime:

```python
print(request_user_location.declaration_only)   # True
```

Use this from middleware to block accidental invocation of client-side tools:

```python
from agent_framework import FunctionMiddleware, FunctionInvocationContext, ToolException


class ClientSideOnlyGuard(FunctionMiddleware):
    """Raise before the framework tries to invoke a declaration-only tool."""

    async def process(self, context: FunctionInvocationContext, call_next) -> None:
        if context.function.declaration_only:
            raise ToolException(
                f"Tool '{context.function.name}' is declaration-only. "
                "Handle it client-side before calling the agent again."
            )
        await call_next()
```

### Use cases beyond client-side rendering

Declaration-only tools are also useful for:

- **Model capability testing** — define a schema and let the model reason about which arguments it would supply, without ever executing anything.
- **Approval-gate placeholders** — expose the tool to the model, but hold the `func=None` slot until a human approves the action. Swap in a real callable after approval.
- **OpenAPI stubs** — generate `FunctionTool` objects from an OpenAPI spec at startup; wire in the actual HTTP calls lazily once auth tokens are ready.

```python
from agent_framework import FunctionTool, Agent
from agent_framework.openai import OpenAIChatClient
from pydantic import BaseModel, Field


class BookFlightArgs(BaseModel):
    origin: str = Field(description="IATA airport code e.g. LHR")
    destination: str = Field(description="IATA airport code e.g. NRT")
    date: str = Field(description="ISO 8601 date e.g. 2026-09-15")


# Declare without implementation — the model plans the booking
plan_flight = FunctionTool(
    name="book_flight",
    description="Book a commercial flight. Requires human approval before execution.",
    func=None,
    input_model=BookFlightArgs,
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="You are a travel assistant. Propose flights; a human will confirm.",
    tools=[plan_flight],
)
```

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

## Composing agents — `as_tool()` and `as_mcp_server()`

Every concrete agent (`Agent`, custom `BaseAgent` subclasses) can be turned into a tool that other agents call. There are two doors:

- `agent.as_tool()` — wraps the agent in a `FunctionTool`. Other agents (in this process) call it like any other function.
- `agent.as_mcp_server()` — wraps the agent in an MCP `Server` with one tool. Anything that speaks MCP can call it remotely.

### `as_tool()` — supervisor / specialist composition

Use this when the supervisor agent needs to delegate a narrow task to a specialist with its own instructions, model, or tool set:

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


# Specialist agent — narrow job, narrow toolset
billing = Agent(
    client=OpenAIChatClient(model="gpt-4o-mini"),     # cheaper model for the specialist
    name="billing_specialist",
    description="Resolves invoice, refund and subscription questions.",
    instructions="Only answer billing questions. Refuse anything else politely.",
    tools=[lookup_invoice, refund_charge],
)

# Supervisor agent — broad job, delegates via the wrapped specialist
supervisor = Agent(
    client=OpenAIChatClient(model="gpt-4o"),
    name="customer_support",
    instructions="Triage user questions. Delegate billing topics to the specialist tool.",
    tools=[
        billing.as_tool(
            name="ask_billing",
            description="Delegate billing questions; returns the specialist's reply verbatim.",
            arg_name="question",
            arg_description="The user's billing question, copied verbatim.",
        ),
    ],
)

response = await supervisor.run("Why was I charged twice on April 1st?")
```

Tunable knobs (all keyword-only):

| Argument | Effect |
|---|---|
| `name` | Tool name surfaced to the model. Defaults to the agent's sanitised name. |
| `description` | Tool description. Defaults to the agent's `description`. |
| `arg_name` | Parameter the model fills with the task. Defaults to `"task"`. |
| `arg_description` | Description of that parameter. Defaults to `"Task for {tool_name}"`. |
| `approval_mode` | `"always_require"` to gate the delegation behind a HITL check. |
| `stream_callback` | Async/sync callable invoked for each `AgentResponseUpdate` from the specialist's stream. |
| `propagate_session` | When `True`, the supervisor's `AgentSession` is forwarded into the specialist's `run()` so they share history. Default `False`. |

### Streaming the specialist's progress

Hook `stream_callback` to surface intermediate updates from the specialist to your UI without changing the supervisor's contract:

```python
async def on_specialist_update(update) -> None:
    if update.text:
        await ws.send_text(update.text)        # forward partial output to the browser


supervisor = Agent(
    client=OpenAIChatClient(),
    name="supervisor",
    tools=[
        billing.as_tool(
            name="ask_billing",
            stream_callback=on_specialist_update,
        ),
    ],
)
```

The supervisor still sees a single string return value. The callback only side-channels the streamed tokens.

### Sharing a session — `propagate_session=True`

By default, every delegated call starts a fresh conversation in the specialist. Flip `propagate_session=True` when you want the specialist to **see** the same conversation history as the supervisor (e.g. a multi-turn debugging assistant where a sub-agent should know what was already said):

```python
debugger = Agent(client=OpenAIChatClient(), name="debugger", instructions="…")

supervisor = Agent(
    client=OpenAIChatClient(),
    name="supervisor",
    tools=[debugger.as_tool(propagate_session=True)],
)

session = supervisor.create_session()
await supervisor.run("My function returns None on Monday — here's the code.", session=session)
await supervisor.run("Now find the bug.", session=session)
# The debugger sees both turns when it gets called.
```

### `as_mcp_server()` — expose an agent over MCP

Same wrapping, different transport. `as_mcp_server()` returns an `mcp.server.lowlevel.Server` that lists exactly one tool — the agent itself. Drop it into any MCP-compatible runner:

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

agent = Agent(
    client=OpenAIChatClient(),
    name="docs_agent",
    description="Answers questions about our internal documentation.",
    instructions="Only answer using the indexed docs.",
)

server = agent.as_mcp_server(
    server_name="docs-mcp",
    version="1.0.0",
    instructions="Use the docs_agent tool for any question about internal docs.",
)
```

Now any MCP client — Claude Desktop, an LLM IDE, or another `Agent` configured with `MCPStdioTool` / `MCPStreamableHTTPTool` — can drive your agent through the standard tool surface. This is the cleanest way to turn a single-purpose agent into a service consumable by **other** agents written in any language.

> Requires the optional `mcp` dependency. `as_mcp_server()` raises `ModuleNotFoundError` with a friendly message if it's missing.

## Tools that return rich content — `Content.from_*`

When you opt into `result_parser`, the framework gives you a unified `Content` model with classmethods for every payload kind. A single tool can return text, JSON, errors, and binary data in one list — the agent and the chat client decide how to render each piece.

```python
from agent_framework import tool, Content


def parse_inventory(payload: dict) -> list[Content]:
    parts: list[Content] = []

    # Human-readable summary
    parts.append(Content.from_text(f"{payload['count']} items in stock"))

    # Raw JSON for the model to reason about
    parts.append(Content.from_text(payload["json_blob"]))

    # An image from the warehouse camera
    if image_url := payload.get("camera_url"):
        parts.append(Content.from_uri(uri=image_url, media_type="image/jpeg"))

    # Embedded binary (e.g. a small PDF report)
    if pdf_bytes := payload.get("report_pdf"):
        parts.append(Content.from_data(data=pdf_bytes, media_type="application/pdf"))

    # Surface a soft error without raising
    if payload.get("warning"):
        parts.append(Content.from_error(message=payload["warning"]))

    return parts


@tool(result_parser=parse_inventory)
def get_inventory(sku: str) -> dict:
    return inventory_lookup(sku)
```

The four constructors you'll reach for:

| Method | When | Carries |
|---|---|---|
| `Content.from_text` | Human-readable strings, JSON, summaries | `text` |
| `Content.from_uri` | Remote images, PDFs, audio | `uri`, `media_type` |
| `Content.from_data` | Inline binary (base64 internally) | `data`, `media_type` |
| `Content.from_error` | Soft failures the model should reason about | `message`, optional `code` |

Returning a single `str` is shorthand for `Content.from_text(...)`. Use the explicit list only when you need the multi-part shape.

## Skipping result parsing — `SKIP_PARSING`

The framework wraps every tool result in `list[Content]` by default. If you're piping the raw return value straight into another sandbox (e.g. a Python interpreter loop, a custom executor, or a reasoning harness), the wrapping is wasted work and lossy. Two ways to opt out:

```python
from agent_framework import tool, FunctionTool, SKIP_PARSING


# Tool always returns raw value
@tool(result_parser=SKIP_PARSING)
def crunch_numbers(values: list[float]) -> dict:
    return {"mean": sum(values) / len(values), "raw": values}


# Or skip per-call without changing the tool definition
result = await some_tool.invoke(arguments={"x": 1}, skip_parsing=True)
# result is the raw return value, NOT list[Content]
```

`SKIP_PARSING` makes `invoke()` return whatever the wrapped function returned — a dict, a Pydantic model, a numpy array. Useful when an outer harness already understands the type and would only have to undo the `Content` wrapping.

## Patterns

**One agent, many backends.** Keep tool signatures stable and swap implementations via DI. Register the same set with every agent.

**Tenant-scoped data.** Pass `function_invocation_kwargs={"tenant_id": ...}` on `agent.run(...)` and read it from `FunctionInvocationContext.kwargs` — tools stay tenant-agnostic.

**Dangerous operations.** `approval_mode="always_require"` + function middleware that logs proposed arguments gives you an approvals audit log.

**Server-side tools vs client-side UI.** Use declaration-only `FunctionTool(func=None)` for UI rendering, real `@tool`-decorated functions for server-side actions — the model treats them identically.

**Hosted tools.** When the provider exposes hosted tools (Bedrock Guardrails, Foundry toolbox, OpenAI file-search, web-search, code-interpreter) the chat client exposes `SupportsWebSearchTool`, `SupportsFileSearchTool`, `SupportsCodeInterpreterTool` protocols — pass provider-specific tool descriptors in the same `tools=[...]` list.

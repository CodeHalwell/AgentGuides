---
title: "Microsoft Agent Framework (Python) — Tools"
description: "The @tool decorator and FunctionTool class: schemas, runtime context, approval, result parsing, invocation limits, the tool-loop configuration, provider-hosted tools, and agents as tools or MCP servers."
framework: "microsoft-agent-framework"
language: "python"
sidebar:
  label: "Tools"
  order: 11
---

Tools are how an agent calls back into your code. You define them with the `@tool` decorator for the common case, or construct a `FunctionTool` directly when tools are built at runtime. Either way the agent sees a `FunctionTool`: a name, a description, a JSON schema for the arguments, and a callable.

This page covers function tools, the tool loop that runs them, and provider-hosted tools. MCP servers are on the [MCP page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_mcp/) and skill tools on the [Skills page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_skills/).

Checked against `agent-framework` %%version=agent-framework%% (`agent_framework._tools`), %%versions-checked%%.

## Minimal `@tool`

```python
import asyncio
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


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a weather assistant.",
        tools=[get_weather],
    )
    response = await agent.run("What's the weather in Amsterdam?")
    print(response.text)


asyncio.run(main())
```

What the decorator does:

- The function name becomes the tool name and the docstring the description.
- A Pydantic input model is built from the signature, and its JSON schema is what the model sees. `Annotated[T, "text"]` and `Annotated[T, Field(description=...)]` both become parameter descriptions; parameters with defaults are optional.
- Arguments from the model are validated against that input model before your function runs.
- Async functions run on the event loop. **Sync functions run in a worker thread** (`asyncio.to_thread`), so blocking I/O does not stall other runs, but the function must be thread-safe.

The schema for the tool above, from `get_weather.parameters()`:

```text
{"properties": {"location": {"description": "The city or region name", "title": "Location", "type": "string"},
                "unit": {"default": "celsius", "description": "celsius or fahrenheit", "title": "Unit", "type": "string"}},
 "required": ["location"], "title": "get_weather_input", "type": "object"}
```

Plain functions passed in `tools=[...]` are wrapped the same way, so `@tool` is only required when you want to set options.

## Decorator options

```python
from typing import Annotated

from pydantic import Field
from agent_framework import tool


@tool(
    name="weather_tool",
    description="Returns current conditions for a city.",
    approval_mode="never_require",   # or "always_require"
    max_invocations=50,              # lifetime cap on this tool instance
    max_invocation_exceptions=3,     # stop calling it after 3 failures
    kind="readonly",                 # free-form label (see "Tool kind" below)
    additional_properties={"team": "platform"},
)
async def get_weather(
    location: Annotated[str, Field(description="City name, e.g. 'Seattle'.")],
) -> str:
    return f"{location}: sunny"
```

The full signature is `tool(func=None, *, name=None, description=None, schema=None, approval_mode=None, kind=None, max_invocations=None, max_invocation_exceptions=None, additional_properties=None, result_parser=None)`.

## Explicit schemas

Pass `schema=` to present a different schema from the one the signature implies, for example to add enum constraints or hide optional parameters. It takes a Pydantic model:

```python
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from agent_framework import tool


class WeatherInput(BaseModel):
    location: Annotated[str, Field(description="City name")]
    unit: Literal["celsius", "fahrenheit"] = "celsius"


@tool(schema=WeatherInput)
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the weather."""
    return f"{location}: 22 {unit}"
```

or a JSON schema dict (an object schema with `properties`):

```python
from agent_framework import tool


@tool(
    name="get_weather",
    description="Get the weather.",
    schema={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    },
)
def get_weather(location: str, unit: str = "celsius") -> str:
    return f"{location}: 22 {unit}"
```

The function is still called with keyword arguments named after the schema's properties, so keep the two in step.

## Runtime context: hidden parameters

Add a parameter annotated `FunctionInvocationContext` anywhere in the signature to receive per-call state. It is left out of the schema the model sees.

```python
import asyncio
from typing import Annotated

from agent_framework import Agent, FunctionInvocationContext, tool
from agent_framework.openai import OpenAIChatClient


@tool
def get_user_orders(
    status: Annotated[str, "pending, shipped or delivered"],
    ctx: FunctionInvocationContext,
) -> str:
    """List the current user's orders with a given status."""
    user_id = ctx.kwargs.get("user_id")
    session_id = ctx.session.session_id if ctx.session else "no-session"
    return f"{user_id} ({session_id}) has 3 {status} orders"


async def main() -> None:
    agent = Agent(client=OpenAIChatClient(), instructions="Answer order questions.", tools=[get_user_orders])
    session = agent.create_session()
    response = await agent.run(
        "Where are my orders?",
        session=session,
        function_invocation_kwargs={"user_id": "u-123"},   # reaches ctx.kwargs, never the model
    )
    print(response.text)


asyncio.run(main())
```

`function_invocation_kwargs` is the right channel for tenant IDs, auth tokens and trace IDs: the model cannot see or forge them.

### `FunctionInvocationContext` fields

| Field | Type | What it carries |
|---|---|---|
| `function` | `FunctionTool` | The tool being invoked: `function.name`, `function.kind`, `function.additional_properties`. |
| `arguments` | `BaseModel \| Mapping[str, Any]` | The call arguments. In function middleware you may replace them before `call_next()`; they are validated again just before the function runs. |
| `session` | `AgentSession \| None` | The session of the current run, if any. |
| `kwargs` | `Mapping[str, Any]` | The `function_invocation_kwargs` passed to `run()`. |
| `metadata` | `dict[str, Any]` | Scratch space shared by every function middleware in this one invocation, and visible to the tool. |
| `result` | `Any` | `None` before `call_next()`; the tool's result afterwards. A middleware can overwrite it. |
| `tools` | `list \| None` | The tools available in this run. |

### Sharing data between middleware and the tool

A function middleware can stamp a value into `context.metadata`, and the tool reads it from the same context object. Each invocation gets its own context, so nothing leaks between concurrent calls.

```python
import asyncio
from typing import Annotated

from agent_framework import Agent, FunctionInvocationContext, function_middleware, tool
from agent_framework.openai import OpenAIChatClient


@function_middleware
async def stamp_trace_id(context: FunctionInvocationContext, call_next) -> None:
    context.metadata["trace_id"] = context.session.session_id if context.session else "no-session"
    await call_next()


@tool
async def fetch_report(
    report_id: Annotated[str, "The report identifier."],
    ctx: FunctionInvocationContext,
) -> str:
    """Fetch a report by id."""
    trace = ctx.metadata.get("trace_id", "unknown")
    return f"[trace={trace}] Report {report_id}: revenue £42,000"


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="You are a reporting assistant.",
        tools=[fetch_report],
        middleware=[stamp_trace_id],
    )
    session = agent.create_session(session_id="req-abc123")
    response = await agent.run("Show me the Q1 revenue report.", session=session)
    print(response.text)


asyncio.run(main())
```

## Approval

```python
import os

from agent_framework import tool


@tool(approval_mode="always_require")
def delete_file(path: str) -> str:
    """Delete a file."""
    os.remove(path)
    return f"deleted {path}"
```

`approval_mode` is `"never_require"` (the default) or `"always_require"`. When the model calls an approval-gated tool, the run stops before the tool executes and the response carries a `function_approval_request` in `response.user_input_requests`. Turn each request into an answer with `request.to_function_approval_response(approved=True | False)` and send it back as a user message **on the same session**:

```python
import asyncio

from agent_framework import Agent, Message, tool
from agent_framework.openai import OpenAIChatClient


@tool(approval_mode="always_require")
def refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} on {order_id}"


async def main() -> None:
    agent = Agent(client=OpenAIChatClient(), instructions="Handle refunds.", tools=[refund])
    session = agent.create_session()

    response = await agent.run("Refund £20 on order A-17.", session=session)
    while response.user_input_requests:
        answers = []
        for request in response.user_input_requests:
            call = request.function_call
            print(f"Approve {call.name}({call.arguments})?")
            answers.append(request.to_function_approval_response(approved=True))
        response = await agent.run(Message(role="user", contents=answers), session=session)

    print(response.text)


asyncio.run(main())
```

The session matters: by default the framework only accepts an approval that matches a request it recorded in that session, so an approval fabricated in caller-supplied history cannot run a tool (`disable_approval_response_binding` in the tool-loop configuration below turns this off). Workflows, streaming and Agent-as-tool approvals are on the [human-in-the-loop page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_hitl/#tool-approval).

## What the model sees: result parsing and errors

Every result is converted to `list[Content]` before it goes back to the model. The default conversion:

| Return value | Becomes |
|---|---|
| `str` | One text item. |
| `None` | One empty text item. |
| `Content`, or a list containing `Content` | Kept as is (non-`Content` items in the list are JSON-encoded). |
| Pydantic model, dataclass, dict, list, number | JSON text. |

### Custom result parsers

`result_parser=` takes the raw return value and returns a `str` or a `list[Content]`. Use it to trim large results before they reach the context window, or to return several parts:

```python
import json

from agent_framework import Content, tool


def top_ten(rows: list[dict]) -> str:
    return f"{len(rows)} rows (first 10 shown)\n" + json.dumps(rows[:10], default=str)


@tool(result_parser=top_ten)
def run_query(sql: str) -> list[dict]:
    """Run a read-only SQL query."""
    return [{"id": i, "total": i * 10} for i in range(400)]


def weather_parts(result: dict) -> list[Content]:
    parts = [Content.from_text(f"{result['temp']}°C, {result['summary']}")]
    if icon := result.get("icon_url"):
        parts.append(Content.from_uri(icon, media_type="image/png"))
    return parts


@tool(result_parser=weather_parts)
def get_weather(location: str) -> dict:
    """Get the weather."""
    return {"temp": 22, "summary": "Sunny", "icon_url": "https://cdn.example.com/sunny.png"}
```

The `Content` constructors you will use most in results:

| Method | Signature | Use |
|---|---|---|
| `Content.from_text` | `(text)` | Text, JSON, summaries. |
| `Content.from_uri` | `(uri, *, media_type=None)` | A remote image, PDF or audio file, or a `data:` URI. |
| `Content.from_data` | `(data: bytes, media_type)` | Inline binary content. |
| `Content.from_error` | `(*, message=None, error_code=None, error_details=None)` | A failure the model should reason about, without raising. |

Whether a provider can pass images or files back to the model as a tool result depends on the provider's API; text always works.

### `SKIP_PARSING`

`result_parser=SKIP_PARSING` makes `invoke()` return the function's raw value. You can also skip parsing for one direct call:

```python
import asyncio

from agent_framework import SKIP_PARSING, tool


@tool(result_parser=SKIP_PARSING)
def crunch(values: list[float]) -> dict:
    """Summarise numbers."""
    return {"mean": sum(values) / len(values)}


@tool
def stats(values: list[float]) -> dict:
    """Summarise numbers."""
    return {"n": len(values)}


async def main() -> None:
    print(await crunch.invoke(arguments={"values": [1.0, 2.0]}))                     # {'mean': 1.5}
    print(await stats.invoke(arguments={"values": [1.0]}, skip_parsing=True))       # {'n': 1}
    print((await stats.invoke(arguments={"values": [1.0]}))[0].text)                 # '{"n": 1}'


asyncio.run(main())
```

This is for code that calls tools itself (a harness, a code-execution sandbox, tests). A tool the agent calls still needs a result the model can read.

### Errors

If a tool raises, the loop catches the exception and sends the model a function result that says **only** `Error: Function failed.`, and logs a warning. The exception text is withheld because it can contain secrets, paths or stack details. You have three options:

- Return an explanatory string (or `Content.from_error(message=...)`) for errors the model should act on, such as "Unknown timezone; use an IANA name".
- Set `include_detailed_errors=True` in the tool-loop configuration (below) to append the exception text to the error result. Only do this when the exception text is safe to show the model.
- Raise `MiddlewareFailure` from function middleware when an enforcement check fails and the whole run must stop; ordinary exceptions never stop the run.

```python
from agent_framework import tool


@tool
def divide(a: float, b: float) -> str:
    """Divide a by b."""
    if b == 0:
        return "Cannot divide by zero. Ask the user for a non-zero divisor."
    return str(a / b)
```

After `max_consecutive_errors_per_request` failures in a row (default 3) the loop stops calling tools for that request. For transient failures, wrap the call in a retrying function middleware ([Middleware → retrying a failed tool call](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_middleware/#retrying-a-failed-tool-call)).

## Invocation limits

`max_invocations=N` caps the **lifetime** of the tool instance: once it has run N times, further calls raise `ToolException` (which the model sees as a failed call). Calls already requested in the same batch still complete. The counter lives on the instance and is never reset automatically, so for module-level tools in a long-running server either reset it yourself or use a per-request limit (`max_function_calls`, below).

`max_invocation_exceptions=N` does the same for failures: after N exceptions the tool stops being called.

Both counters are public:

```python
from agent_framework import tool


@tool(max_invocations=1000)
def fetch_inventory(sku: str) -> str:
    """Look up stock for a SKU."""
    return f"{sku}: 12 in stock"


# Later, in a periodic job:
print(fetch_inventory.invocation_count, fetch_inventory.invocation_exception_count)
fetch_inventory.invocation_count = 0     # start a new reporting window
```

## Configuring the tool loop

The loop that sends tool results back to the model and asks again is configured per chat client with a `FunctionInvocationConfiguration` (a `TypedDict`). Pass it to the client constructor, or change keys on `client.function_invocation_configuration` afterwards. Every agent using that client gets the same settings.

```python
import asyncio

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

client = OpenAIChatClient(
    function_invocation_configuration={
        "max_iterations": 5,          # model round trips per run
        "max_function_calls": 20,     # total tool executions per run
        "max_duration_seconds": 60.0,
    }
)
client.function_invocation_configuration["terminate_on_unknown_calls"] = True


async def main() -> None:
    agent = Agent(client=client, instructions="You are a research assistant.")
    print((await agent.run("Research the order pipeline")).text)


asyncio.run(main())
```

| Key | Default | Effect |
|---|---|---|
| `enabled` | `True` | Master switch. `False` leaves function calls in the response for you to handle. |
| `max_iterations` | `40` | Maximum model round trips in one run. |
| `max_function_calls` | `None` | Maximum tool executions in one run. When reached, tools are withdrawn and the model is asked for a text answer. |
| `max_duration_seconds` | `None` | Wall-clock budget for the loop, including time spent waiting for approvals; same graceful stop. |
| `max_consecutive_errors_per_request` | `3` | Stop calling tools after this many failures in a row. |
| `terminate_on_unknown_calls` | `False` | Raise instead of returning an error result when the model calls a tool that does not exist. |
| `include_detailed_errors` | `False` | Append the exception text to error results sent to the model. |
| `allow_concurrent_invocation` | `True` | Run several calls from one model response concurrently. `False` runs them in order. |
| `additional_tools` | `[]` | Tools that can be executed if called but are not advertised to the model. |
| `disable_approval_response_binding` | `False` | Accept approval responses that do not match a request recorded in the session. Leave it off unless you enforce the same check elsewhere. |

`max_function_calls` and `max_duration_seconds` are checked after each batch of parallel calls, so one response asking for 20 calls runs all 20 even if the limit is 10.

## Tool kind and metadata

`kind` is an optional classification string, and `additional_properties` a free-form dict. The framework does not interpret either, with one exception: `kind="shell"` marks a shell tool, which the OpenAI and Anthropic clients send as their native shell tool (see `client.get_shell_tool(...)`). Any other value is yours to use, typically in function middleware:

```python
from agent_framework import FunctionInvocationContext, FunctionMiddleware, MiddlewareTermination, tool


@tool(kind="readonly", additional_properties={"owner": "search"})
def search_kb(query: str) -> str:
    """Search the knowledge base."""
    return "3 articles"


@tool(kind="mutating", additional_properties={"requires_audit": True})
def archive_ticket(ticket_id: str) -> str:
    """Move a ticket to long-term storage."""
    return f"archived {ticket_id}"


class MutatingGuard(FunctionMiddleware):
    """Refuse mutating tools unless the caller passed approved=True in function_invocation_kwargs."""

    async def process(self, context: FunctionInvocationContext, call_next) -> None:
        if context.function.kind == "mutating" and context.kwargs.get("approved") is not True:
            raise MiddlewareTermination(f"{context.function.name} needs approval")
        await call_next()
```

`MiddlewareTermination` raised from function middleware ends the tool loop gracefully: the run returns what it has so far. To fail the run with an exception instead, raise `MiddlewareFailure`. The [middleware page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_middleware/) covers both.

## Constructing `FunctionTool` directly

`FunctionTool` takes the same options as the decorator, all keyword-only, with the schema passed as `input_model=`:

```text
FunctionTool(*, name, description="", approval_mode=None, kind=None, max_invocations=None,
             max_invocation_exceptions=None, additional_properties=None, func=None,
             input_model=None, result_parser=None)
```

Use it when tools are generated at runtime, for example from a table of internal API endpoints:

```python
import asyncio

from agent_framework import Agent, FunctionTool
from agent_framework.openai import OpenAIChatClient

API_SPECS = [
    {"name": "list_customers", "endpoint": "/customers", "description": "List all customers."},
    {"name": "get_customer", "endpoint": "/customers/{id}", "description": "Fetch one customer by id."},
]


def make_tool(spec: dict) -> FunctionTool:
    needs_id = "{id}" in spec["endpoint"]

    async def call(**kwargs) -> str:
        url = spec["endpoint"].format(**kwargs)
        return f"GET {url} -> 200"            # replace with a real HTTP call

    return FunctionTool(
        name=spec["name"],
        description=spec["description"],
        func=call,
        input_model={
            "type": "object",
            "properties": {"id": {"type": "string"}} if needs_id else {},
            "required": ["id"] if needs_id else [],
        },
    )


async def main() -> None:
    agent = Agent(client=OpenAIChatClient(), instructions="Use the API tools.", tools=[make_tool(s) for s in API_SPECS])
    print((await agent.run("Show customer 42")).text)


asyncio.run(main())
```

### Declaration-only tools

With `func=None` the model can call the tool but the framework cannot run it. The function call stays in the response for your application to handle, which suits actions carried out in a front end (asking for a location, rendering a confirmation dialog):

```python
from agent_framework import Agent, FunctionTool
from agent_framework.openai import OpenAIChatClient

confirm_payment = FunctionTool(
    name="confirm_payment",
    description="Ask the user to confirm a payment before it is processed.",
    func=None,
    input_model={
        "type": "object",
        "properties": {
            "amount": {"type": "number", "description": "Amount in GBP"},
            "merchant": {"type": "string"},
        },
        "required": ["amount", "merchant"],
    },
)

agent = Agent(
    client=OpenAIChatClient(),
    instructions="Use confirm_payment before processing any transaction.",
    tools=[confirm_payment],
)
# After agent.run(...), look for function_call contents named "confirm_payment"
# in response.messages, show the dialog, then send a function result back.
```

Calling `invoke()` on a declaration-only tool raises `ToolException`.

## Passing tools to an agent

Tools go on the constructor (defaults for every run) or on `run()` (added for that run only):

```python
agent = Agent(client=OpenAIChatClient(), tools=[get_weather])
await agent.run("Find a Thai restaurant near me", tools=[search_restaurants])
```

`tools=` accepts a single tool or a list mixing `@tool` functions, plain callables, `FunctionTool` instances, MCP tools (`MCPStdioTool`, `MCPStreamableHTTPTool`, `MCPWebsocketTool`), provider-hosted tool descriptors and tool collections such as a Foundry toolbox, which is flattened into its tools.

## Provider-hosted tools

Some services run tools themselves: web search, code interpreter, file search, image generation, hosted MCP. The chat client builds the descriptor, and you pass it in `tools=` like any other tool:

```python
import asyncio

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    client = OpenAIChatClient()
    agent = Agent(
        client=client,
        instructions="Use web search for anything recent and the code interpreter for arithmetic.",
        tools=[
            client.get_web_search_tool(search_context_size="low"),
            client.get_code_interpreter_tool(),
        ],
    )
    print((await agent.run("What is 2**64, and what was in today's tech news?")).text)


asyncio.run(main())
```

Which factories a client has depends on the service:

| Client | Hosted tool factories |
|---|---|
| `OpenAIChatClient` (Responses API) | `get_web_search_tool`, `get_code_interpreter_tool`, `get_file_search_tool`, `get_image_generation_tool`, `get_mcp_tool`, `get_shell_tool`, `get_computer_tool` |
| `OpenAIChatCompletionClient` | `get_web_search_tool` |
| `FoundryChatClient` | The OpenAI set plus `get_azure_ai_search_tool`, `get_bing_grounding_tool`, `get_bing_custom_search_tool`, `get_sharepoint_tool`, `get_fabric_tool`, `get_browser_automation_tool`, `get_memory_search_tool`, `get_a2a_tool` |
| `AnthropicClient` | `get_web_search_tool`, `get_code_interpreter_tool`, `get_mcp_tool`, `get_shell_tool` |

Code that should work with any client can check the `SupportsWebSearchTool`, `SupportsCodeInterpreterTool`, `SupportsFileSearchTool`, `SupportsImageGenerationTool`, `SupportsMCPTool` and `SupportsShellTool` protocols with `isinstance`. Hosted tools run on the provider's side, so your function middleware and approval settings do not see them; hosted MCP tools have their own `approval_mode=`.

Amazon Bedrock Guardrails are not a tool: they are configured on `BedrockChatClient` through `BedrockGuardrailConfig` (see [model providers](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_model_providers/)).

## Agents as tools: `as_tool()`

`agent.as_tool()` wraps an agent in a `FunctionTool`, so a supervisor agent can delegate to a specialist with its own instructions, model and tools:

```python
import asyncio

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient


@tool
def lookup_invoice(invoice_id: str) -> str:
    """Look up an invoice."""
    return f"{invoice_id}: £40.00, paid twice on 1 April"


billing = Agent(
    client=OpenAIChatClient(model="%%model=openai-small%%"),
    name="billing_specialist",
    description="Resolves invoice, refund and subscription questions.",
    instructions="Only answer billing questions.",
    tools=[lookup_invoice],
)

supervisor = Agent(
    client=OpenAIChatClient(model="%%model=openai%%"),
    name="customer_support",
    instructions="Triage user questions. Delegate billing topics to ask_billing.",
    tools=[
        billing.as_tool(
            name="ask_billing",
            description="Delegate a billing question; returns the specialist's answer.",
            arg_name="question",
            arg_description="The user's billing question, copied verbatim.",
        )
    ],
)


async def main() -> None:
    print((await supervisor.run("Why was I charged twice for invoice INV-9?")).text)


asyncio.run(main())
```

| Argument | Default | Effect |
|---|---|---|
| `name` | the agent's `name`, sanitised | Tool name. A `ValueError` is raised if neither is set. |
| `description` | the agent's `description` or `""` | Tool description; this is what the supervisor's model reads to decide when to delegate. |
| `arg_name` | `"task"` | The single string parameter the model fills in. |
| `arg_description` | `"Task for {tool_name}"` | Its description. |
| `approval_mode` | `"never_require"` | `"always_require"` puts the delegation behind an approval. |
| `stream_callback` | `None` | Sync or async callable that receives each `AgentResponseUpdate` from the specialist as it streams. |
| `propagate_session` | `False` | Pass the supervisor's `AgentSession` into the specialist's run, so both share history and session state. |

The supervisor's model sees only the specialist's final text. `stream_callback` is a side channel for showing progress:

```python
async def on_specialist_update(update) -> None:
    if update.text:
        print(update.text, end="", flush=True)     # or forward to a websocket


billing_tool = billing.as_tool(name="ask_billing", stream_callback=on_specialist_update)
```

With `propagate_session=True` the specialist runs on the supervisor's session, so it sees the turns that came before the delegation. Leave it off when the specialist should only see the task text, which is cheaper and keeps the specialist's context focused.

## Agents as MCP servers: `as_mcp_server()`

`agent.as_mcp_server(server_name="Agent", version=None, instructions=None, lifespan=None)` returns an `mcp.server.lowlevel.Server` with exactly one tool: the agent, named after `agent.name`, taking a `task` string. Any MCP client, in any language, can then call the agent:

```python
import asyncio

from mcp.server.stdio import stdio_server
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

agent = Agent(
    client=OpenAIChatClient(),
    name="docs_agent",
    description="Answers questions about our internal documentation.",
    instructions="Only answer using the indexed docs.",
)
server = agent.as_mcp_server(server_name="docs-mcp", version="1.0.0")


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
```

The `mcp` package is installed with `agent-framework-core[all]` (and so with `agent-framework`); with a bare `agent-framework-core`, `as_mcp_server()` raises `ModuleNotFoundError` telling you to install it. Serving over streamable HTTP is shown in [recipe 16](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_recipes/#recipe-16--expose-an-agent-over-mcp).

## Patterns

**Tenant-scoped data.** Pass `function_invocation_kwargs={"tenant_id": ...}` to `run()` and read `ctx.kwargs` in the tool. The tool's schema stays tenant-agnostic and the model cannot change the tenant.

**Dangerous operations.** `approval_mode="always_require"`, plus a function middleware that logs the proposed arguments, gives an audit trail of what was approved.

**Large results.** Use a `result_parser` to cut the result down before it is added to the conversation. Every later model call in the session pays for those tokens again.

**Workflow steps are not tools.** Deterministic pipeline steps belong in a workflow (`@executor` functions in a `WorkflowBuilder` graph, or `@step` in a functional workflow); see the [comprehensive guide](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_comprehensive_guide/#custom-executors) and [functional workflows](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_functional_workflows/).

## Revision history

| Date | Version | Changes |
|---|---|---|
| 2026-10-24 | 1.22.0 | Corrected the error behaviour (the model sees "Error: Function failed." unless `include_detailed_errors=True`), the approval flow (no `responses=` argument on `Agent.run`), the meaning of `kind` (only `"shell"` is interpreted), `Content.from_error` parameters and the tool-loop configuration table (added `max_duration_seconds`, `allow_concurrent_invocation`, `disable_approval_response_binding`). Added sync-in-thread behaviour, the default result conversion table, provider-hosted tools and runnable examples throughout. Removed an invented "streaming tool outputs" section and the workflow executor and visualisation sections, which were wrong (return values of `@executor` functions are not sent downstream) and are covered on the comprehensive guide. |

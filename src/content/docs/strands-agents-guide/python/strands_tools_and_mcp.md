---
title: "Strands Agents (Python): Tools and MCP"
description: "Defining tools in Strands Python: the @tool decorator, schemas, ToolContext, streaming and async tools, module and class-based tools, executors, strands-agents-tools and its deprecations, vended tools, sandboxes and MCP clients."
framework: "strands"
language: "python"
sidebar:
  label: "Tools and MCP"
  order: 2
---

Checked against `strands-agents` %%version=strands-agents%% and `strands-agents-tools` %%version=strands-agents-tools%% (%%versions-checked%%). Examples were run against an offline Bedrock stub; the MCP examples ran against real local MCP servers.

## The @tool decorator

`@tool` turns a function into a tool. The function name is the tool name, the docstring's summary is the description, and the parameters become a JSON schema built with Pydantic from the type hints. Parameter descriptions come from a Google-style `Args:` section:

```python
from typing import Literal

from strands import Agent, tool


@tool
def search_products(query: str, category: Literal["books", "music", "games"] | None = None, limit: int = 5) -> list[dict]:
    """Search the product catalogue.

    Args:
        query: Free-text search terms.
        category: Restrict results to one category.
        limit: Maximum number of results.
    """
    return [{"sku": "B-101", "title": f"Result for {query}"}][:limit]


print(search_products.tool_spec["inputSchema"]["json"]["properties"]["category"])
agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[search_products], callback_handler=None)
```

Parameters without a default are required. The decorated object still works as a plain function (`search_products("jazz")`), which makes unit-testing tools easy.

**Overrides.** `@tool(name=..., description=..., inputSchema=...)` replaces any of the generated values. `inputSchema` takes `{"json": {...}}`.

**Descriptions without a docstring section.** `Annotated[str, "description"]` works. `Annotated[str, Field(...)]` does not: the decorator raises `NotImplementedError` in this release.

**Pydantic parameters.** A parameter typed as a Pydantic model gets a nested schema and is validated, but the function receives a plain `dict`, not a model instance. Convert it yourself:

```python
from pydantic import BaseModel, Field
from strands import tool


class Address(BaseModel):
    street: str
    postcode: str = Field(description="UK postcode")


@tool
def ship_order(order_id: str, address: Address) -> str:
    """Ship an order.

    Args:
        order_id: Order to ship.
        address: Delivery address.
    """
    addr = Address.model_validate(address)   # arrives as a dict
    return f"{order_id} shipping to {addr.postcode}"
```

**Validation errors.** If the model's arguments fail validation, the tool is not called; the model receives an error result naming the field, and usually corrects itself on the next turn.

### Return values and errors

| Return | What the model sees |
|--------|---------------------|
| `str`, number | A text block |
| `dict`, `list` | JSON text |
| A dict with `status` and `content` keys | Used as the tool result as-is (content blocks such as `{"text": ...}`, `{"json": ...}`, `{"image": ...}`) |

```python
from strands import tool


@tool
def chart_data(metric: str) -> dict:
    """Return a metric series.

    Args:
        metric: Metric name.
    """
    return {
        "status": "success",
        "content": [
            {"text": f"Series for {metric}"},
            {"json": {"points": [3, 5, 8]}},
        ],
    }
```

Raising an exception does not stop the agent. The model gets `{"status": "error", "content": [{"text": "Error: RuntimeError - ..."}]}` and can retry, try another tool or explain the failure. Raise with a message the model can act on.

## Tool context and per-request data

`@tool(context=True)` injects a `ToolContext` into a parameter named `tool_context` (or pass a string to name the parameter yourself). It gives the tool:

| Attribute | Contents |
|-----------|----------|
| `tool_use` | The tool call (`toolUseId`, `name`, `input`) |
| `agent` | The calling agent (`agent.state`, `agent.messages`, `agent.name`) |
| `invocation_state` | The `invocation_state` passed to this call |
| `cancel_signal` | A `threading.Event` set when the call is cancelled |
| `interrupt(name, reason=...)` | Pause for human input (see [interrupts](/strands-agents-guide/python/strands_hooks_and_interrupts/#interrupts)) |

```python
from strands import Agent, ToolContext, tool


@tool(context="ctx")
def record_preference(key: str, value: str, ctx: ToolContext) -> str:
    """Remember a user preference.

    Args:
        key: Preference name.
        value: Preference value.
    """
    prefs = ctx.agent.state.get("prefs") or {}
    prefs[key] = value
    ctx.agent.state.set("prefs", prefs)
    return f"Saved {key} for user {ctx.invocation_state['user_id']}"


agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[record_preference], callback_handler=None)
agent("Please use metric units from now on", invocation_state={"user_id": "u-17"})
```

The context parameter is not part of the schema the model sees.

## Async and streaming tools

Tools can be `async def`. With the default `ConcurrentToolExecutor`, several tool calls from one model turn run concurrently; synchronous tools run in worker threads.

A tool written as an async generator streams progress: each `yield` is surfaced as a `tool_stream_event` in `stream_async`, and **the last value yielded is the tool result**:

```python
import asyncio

from strands import Agent, tool


@tool
async def export_report(report_id: str):
    """Export a report to PDF.

    Args:
        report_id: Report to export.
    """
    yield "Collecting data..."
    await asyncio.sleep(0.1)
    yield "Rendering PDF..."
    yield f"s3://reports/{report_id}.pdf"     # final value becomes the result


agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[export_report], callback_handler=None)


async def main() -> None:
    async for event in agent.stream_async("Export report R-9"):
        if "tool_stream_event" in event:
            print("progress:", event["tool_stream_event"]["data"])


asyncio.run(main())
```

## Calling tools directly

`agent.tool.<name>(**kwargs)` runs a tool without the model. It is useful for deterministic set-up steps and for tests. By default the call and its result are recorded in `agent.messages` so the model knows it happened; pass `record_direct_tool_call=False` to keep it out:

```python
from strands import Agent, tool


@tool
def load_profile(user_id: str) -> dict:
    """Load a user profile.

    Args:
        user_id: User ID.
    """
    return {"user_id": user_id, "name": "Priya", "plan": "pro"}


agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[load_profile], callback_handler=None)
result = agent.tool.load_profile(user_id="u-17")
print(result["status"], result["content"][0]["text"])
agent.tool.load_profile(user_id="u-18", record_direct_tool_call=False)
```

Names with hyphens are reachable with underscores (`agent.tool.my_tool` for `my-tool`).

## Other ways to define tools

### Methods

`@tool` works on methods, so a tool can carry configuration or a client:

```python
from strands import Agent, tool


class Catalogue:
    def __init__(self, prices: dict[str, float]):
        self.prices = prices

    @tool
    def price(self, sku: str) -> float:
        """Look up the price of a SKU in GBP.

        Args:
            sku: Product SKU.
        """
        return self.prices[sku]


catalogue = Catalogue({"A-1": 9.50})
agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[catalogue.price], callback_handler=None)
```

### Module tools with TOOL_SPEC

A module with a `TOOL_SPEC` dict and a function of the same name is a tool. This is the format most of `strands-agents-tools` uses, and it gives full control over the schema and the result:

```python
# weather_tool.py
from typing import Any

from strands.types.tools import ToolResult, ToolUse

TOOL_SPEC = {
    "name": "weather_tool",
    "description": "Get the forecast for a city.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        }
    },
}


def weather_tool(tool_use: ToolUse, **kwargs: Any) -> ToolResult:
    city = tool_use["input"]["city"]
    return {
        "toolUseId": tool_use["toolUseId"],
        "status": "success",
        "content": [{"text": f"Sunny in {city}"}],
    }
```

```python
import weather_tool
from strands import Agent

agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[weather_tool], callback_handler=None)
print(agent.tool_names)   # ['weather_tool']
```

`tools=` also accepts a file path (`"./tools/weather_tool.py"`), and `load_tools_from_directory=True` loads and hot-reloads every tool in `./tools/`, which is convenient while developing and best left off in production.

### AgentTool subclasses and tool providers

For tools built at run time, subclass `strands.types.tools.AgentTool` (implement `tool_name`, `tool_spec`, `tool_type` and an async-generator `stream`), or wrap a function with `PythonAgentTool(tool_name, tool_spec, tool_func)`. A `strands.tools.ToolProvider` supplies a set of tools with a lifecycle (`load_tools`, `add_consumer`, `remove_consumer`); `MCPClient` is one.

## Tool execution

| Executor | Behaviour |
|----------|-----------|
| `ConcurrentToolExecutor()` | Default. All tool calls in one model turn run at once |
| `SequentialToolExecutor()` | One at a time, in the order the model asked |

```python
from strands import Agent
from strands.tools.executors import SequentialToolExecutor

agent = Agent(
    model="us.anthropic.%%model=anthropic%%",
    tool_executor=SequentialToolExecutor(),   # tools that share a resource or must run in order
    callback_handler=None,
)
```

To stop a run after a tool, a hook can set `event.end_turn` on `AfterToolsEvent`, or a tool can use the vended `stop` tool; `Agent.as_tool(delegate=True)` ends the turn with a sub-agent's answer. Retrying a tool is a hook concern (`AfterToolCallEvent.retry = True`), covered on the [hooks page](/strands-agents-guide/python/strands_hooks_and_interrupts/#retrying-tool-and-model-calls).

## strands-agents-tools

`strands-agents-tools` is a separate package of ready-made tools, imported as modules:

```python
from strands import Agent
from strands_tools import file_read, use_aws, workflow

agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[file_read, use_aws, workflow], callback_handler=None)
```

Tools that write files, run code or call AWS ask for confirmation on the terminal before acting. Set `BYPASS_TOOL_CONSENT=true` to skip the prompt in non-interactive environments, and only where the agent's permissions are already narrow.

### Deprecations in 0.8

Version 0.8 deprecates most of the general-purpose tools in favour of tools vended by `strands-agents` itself, MCP servers or SDK features. They still work but log a deprecation warning, which the package says becomes an error log in 0.9.0:

| Deprecated (`strands_tools`) | Replacement |
|------------------------------|-------------|
| `shell` | `strands.vended_tools.shell`, which runs through the agent's sandbox instead of directly on the host |
| `calculator`, `environment` | `strands.vended_tools.shell` (note: arbitrary commands, so a wider boundary than the tools it replaces) |
| `editor` | `strands.vended_tools.file_editor` |
| `http_request` | `strands.vended_tools.http_request` and `web_fetch` |
| `python_repl`, `sleep` | `strands.vended_tools.python_repl`, `sleep` |
| `journal` | `strands.vended_tools.notebook` |
| `mcp_client` | `strands.vended_tools.make_mcp_router` |
| `a2a_client` | `strands.vended_tools.make_a2a_client` |
| `batch` | Nothing: concurrent tool execution is the default |
| `current_time` | Inject the time as context (`ContextInjector` plugin) |
| `think` | The provider's native reasoning (extended thinking) |
| `memory`, `retrieve`, `mem0_memory` | `MemoryManager` with `BedrockKnowledgeBaseStore`, or the `mem0-strands` store |
| `tavily`, `exa`, `bright_data`, `slack`, `chat_video`, `search_video` | The vendor's official MCP server |
| `cron` | EventBridge Scheduler (or `shell`) |
| `diagram`, `rss` | No replacement tool: have the model write code with graphviz, Mermaid or the `diagrams` package (`diagram`), or parse feeds with `feedparser` (`rss`) |

Not deprecated in this release: `file_read`, `file_write`, `use_aws`, `workflow`, `swarm`, `graph`, `agent_graph`, `use_agent`, `use_llm`, `handoff_to_user`, `stop`, `load_tool`, `image_reader`, `generate_image`, `generate_image_stability`, `nova_reels`, `speak`, `use_computer`, `agent_core_memory`, `elasticsearch_memory`, `mongodb_memory`, `code_interpreter` (AgentCore Code Interpreter) and `browser` (AgentCore Browser or local Chromium). Some need extras, for example `strands-agents-tools[agent-core-code-interpreter]`.

### Vended tools in strands-agents

`strands.vended_tools` ships tools with the SDK. Each has a ready instance and a `make_*` factory for configuration:

| Tool | Factory options |
|------|-----------------|
| `shell` | `make_shell(sandbox=..., name=..., description=...)` |
| `file_editor` | `make_file_editor(sandbox=...)`; commands `view`, `create`, `str_replace`, `insert` |
| `http_request` | `make_http_request(client=httpx.AsyncClient(...))` |
| `web_fetch` | Needs the `web-fetch` extra |
| `python_repl`, `notebook`, `sleep`, `stop`, `handoff_to_user` | |
| `subagent`, `swarm` | Let the model spawn sub-agents or a swarm |
| `make_mcp_router(...)`, `make_a2a_client(allowed_endpoints=[...])` | Factories only |

```python
from strands import Agent
from strands.vended_tools import file_editor, http_request, make_shell

agent = Agent(
    model="us.anthropic.%%model=anthropic%%",
    tools=[make_shell(), file_editor, http_request],
    callback_handler=None,
)
print(agent.tool_names)
```

## Sandboxes

`shell` and `file_editor` run through the agent's sandbox. Without one they run directly on the host (`NotASandboxLocalEnvironment`), with the same permissions as your process. Pass a sandbox to run them elsewhere:

```python
from strands import Agent
from strands.sandbox.docker import DockerSandbox

# docker run -d --name agent-sandbox python:3.12-slim sleep infinity
agent = Agent(
    model="us.anthropic.%%model=anthropic%%",
    sandbox=DockerSandbox("agent-sandbox", working_dir="/workspace"),
    callback_handler=None,
)
print(agent.tool_names)   # ['sandbox_file_editor', 'sandbox_shell']
```

`DockerSandbox(container, working_dir=None, user=None)` runs commands with `docker exec` in a container you started and vends `sandbox_shell` and `sandbox_file_editor`. `strands.sandbox.ssh.SshSandbox(host, working_dir=..., identity_file=...)` does the same over SSH, and `PosixShellSandbox` is the base class for your own. On AWS, the managed option for model-written code is the AgentCore Code Interpreter (`strands_tools.code_interpreter`, or the [AgentCore client](/amazon-bedrock-agents-guide/python/bedrock_agentcore_comprehensive_guide/#code-interpretation)).

## MCP

`strands.tools.mcp.MCPClient` connects to a Model Context Protocol server and exposes its tools to an agent. The agent starts the client when it loads its tools and stops it on `agent.cleanup()` or when the agent is garbage-collected.

Strands %%version=strands-agents%% accepts `mcp` from 1.23 up to (not including) 2.2, so a fresh install resolves to the 2.x series. In `mcp` 2.x, `FastMCP` has been renamed to `MCPServer` (`from mcp.server.mcpserver import MCPServer`); server code written for `mcp` 1.x that imports `mcp.server.fastmcp` fails with `ModuleNotFoundError` there. Pin `mcp<2` if you need the old import.

### A local stdio server

A small server to test against:

```python
# inventory_server.py
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("inventory")


@mcp.tool()
def stock_level(sku: str) -> int:
    """Return the number of units in stock for a SKU."""
    return {"A-1": 12, "B-2": 0}.get(sku, 0)


@mcp.tool()
def delete_sku(sku: str) -> str:
    """Delete a SKU from the catalogue."""
    return f"deleted {sku}"


if __name__ == "__main__":
    mcp.run()  # stdio
```

An agent using it, with one tool filtered out and the rest prefixed:

```python
import sys

from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.tools.mcp import MCPClient

inventory = MCPClient(
    lambda: stdio_client(StdioServerParameters(command=sys.executable, args=["inventory_server.py"])),
    tool_filters={"rejected": ["delete_sku"]},
    prefix="inv",
)

agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[inventory], callback_handler=None)
print(agent.tool_names)            # ['inv_stock_level']
agent("How many A-1 are in stock?")
agent.cleanup()                    # stops the server process
```

Published servers are launched the same way, for example `StdioServerParameters(command="uvx", args=["awslabs.aws-documentation-mcp-server@latest"])`.

### Remote servers

For a Streamable HTTP server, pass `url=` instead of a transport callable:

```python
from strands import Agent
from strands.tools.mcp import MCPClient

internal = MCPClient(
    url="https://mcp.example.com/mcp",
    headers={"Authorization": "Bearer <token>"},
    startup_timeout=30,
)
agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[internal], callback_handler=None)
```

`auth=` takes OAuth client credentials (machine-to-machine) and `auth_provider=` any `httpx.Auth`. For SSE servers pass `lambda: sse_client(url)` from `mcp.client.sse`. AgentCore Gateway is an MCP server, so it is reached the same way (see [Gateway](/amazon-bedrock-agents-guide/python/bedrock_agentcore_comprehensive_guide/#pattern-1-gateway--expose-existing-apis-as-mcp-tools)).

### Servers from a config file

Many MCP hosts describe their servers in an `mcpServers` JSON file. `MCPClient.load_servers()` builds one client per enabled server from such a file (a path) or from the same structure as a dict. A server with `command` is started over stdio, one with `url` uses Streamable HTTP (set `transport` to `"sse"` or `"streamable-http"` to override), and `${VAR}` in string values is filled from the environment:

```python
import sys

from strands import Agent
from strands.tools.mcp import MCPClient

config = {
    "mcpServers": {
        "inventory": {"command": sys.executable, "args": ["inventory_server.py"]},
        "docs": {
            "url": "https://mcp.example.com/mcp",
            "headers": {"Authorization": "Bearer ${MCP_TOKEN}"},
            "continue_on_error": True,
        },
        "legacy": {"command": "node", "args": ["server.js"], "disabled": True},
    }
}

clients = MCPClient.load_servers(config, prefix_with_server_name=True)
agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=clients, callback_handler=None)
print(agent.tool_names)      # ['inventory_stock_level', 'inventory_delete_sku']: "docs" was skipped
agent.cleanup()
```

`disabled` servers are skipped. A server whose config cannot be resolved (here `MCP_TOKEN` is not set) is skipped if it has `"continue_on_error": true`, and raises otherwise; the `continue_on_error=` argument sets that default for every server. Malformed JSON or a malformed entry always raises. `prefix_with_server_name=True` prefixes each server's tools with its key (characters outside `[A-Za-z0-9_-]` become `_`), so two servers that both offer `search` do not clash. Note that this example does not filter `delete_sku`; pass `tool_filters` by building the client directly when you need that.

### Using the client directly

As a context manager, the client can list and call tools, prompts and resources without an agent:

```python
import sys

from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient

with MCPClient(lambda: stdio_client(StdioServerParameters(command=sys.executable, args=["inventory_server.py"]))) as client:
    print([t.tool_name for t in client.list_tools_sync()])
    result = client.call_tool_sync(tool_use_id="t-1", name="stock_level", arguments={"sku": "B-2"})
    print(result["status"], result["content"])
```

`list_prompts_sync`, `get_prompt_sync`, `list_resources_sync` and `read_resource_sync` cover the rest of the protocol, with `_async` variants of the call methods.

### Options

| `MCPClient` argument | Use |
|----------------------|-----|
| `tool_filters` | `{"allowed": [...], "rejected": [...]}`: exact names, compiled regexes or callables |
| `prefix` | Prefix tool names to avoid clashes between servers |
| `startup_timeout` | Seconds to wait for the server (default 30) |
| `continue_on_error` | Let the agent start without this server if it fails |
| `elicitation_callback` | Answer server elicitation requests |
| `progress_callback` | Receive progress notifications |
| `tasks_config` | Use MCP tasks for long-running tools |
| `on_tools_changed` | React when the server's tool list changes |

If a server cannot start, the agent constructor raises a `ValueError` (wrapping a `ToolProviderException`) unless the client was created with `continue_on_error=True`.

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-24 | %%version=strands-agents%% | Audit against the installed packages. Corrected the `mcp` dependency range (1.23 up to 2.2, not 2.x only), the `shell` replacement note and the `diagram`/`rss` guidance. Added `MCPClient.load_servers()` for `mcpServers` config files. |
| 2026-10-24 | 1.62.0 | New page. Documents schema generation limits (`Annotated` with `Field` unsupported; Pydantic parameters arrive as dicts), return-value handling, streaming tools, direct calls, module and class tools, executors, the `strands-agents-tools` 0.8 deprecations, vended tools, sandboxes and MCP with `mcp` 2.x. |

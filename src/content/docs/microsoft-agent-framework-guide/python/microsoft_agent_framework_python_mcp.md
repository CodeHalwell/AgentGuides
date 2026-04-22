---
title: "Microsoft Agent Framework (Python) — MCP Integration"
description: "Plug Model Context Protocol servers into agent-framework via MCPStdioTool, MCPStreamableHTTPTool, and MCPWebsocketTool. Approval gates, header injection, and sampling callbacks."
framework: microsoft-agent-framework
language: python
---

# MCP Integration — Python

Model Context Protocol (MCP) servers are first-class tool sources in `agent-framework`. Three transports ship in `agent_framework`:

| Class | Transport | Typical use |
|---|---|---|
| `MCPStdioTool` | Subprocess over stdio | Local tools — filesystem, git, npm-hosted servers |
| `MCPStreamableHTTPTool` | Streamable HTTP (SSE) | Remote / hosted MCP services |
| `MCPWebsocketTool` | WebSocket | Bidirectional streaming services |

All three are async context managers that connect lazily, discover tools and prompts from the server, and register them as `FunctionTool` instances on the agent.

Verified against `agent-framework-core==1.1.0` and `mcp==1.27`.

## Install

The `mcp` package is required for any MCP tool:

```bash
pip install agent-framework  # pulls mcp transitively
# or, for pruned installs:
pip install agent-framework-core mcp
pip install 'mcp[ws]'        # only if you need MCPWebsocketTool
```

## Stdio — local MCP servers

```python
import asyncio
from agent_framework import Agent, MCPStdioTool
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    async with MCPStdioTool(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        description="Read and write files under /tmp",
    ) as fs:
        agent = Agent(
            client=OpenAIChatClient(),
            instructions="You help the user manage files in /tmp.",
            tools=fs,
        )
        response = await agent.run("List the files in /tmp and summarise their names.")
        print(response.text)


asyncio.run(main())
```

Notes:

- `name` is a *tool group* name — it becomes the prefix for the tools exposed to the model (e.g. `filesystem_read_file`). Override with `tool_name_prefix="fs"` to pick a shorter prefix.
- `command` + `args` + `env` are forwarded to `mcp.client.stdio.StdioServerParameters`.
- Use `async with` so the subprocess is cleaned up when the agent finishes.

## Streamable HTTP — remote MCP servers

```python
from agent_framework import Agent, MCPStreamableHTTPTool

async with MCPStreamableHTTPTool(
    name="learn",
    url="https://learn.microsoft.com/api/mcp",
    description="Search official Microsoft Learn documentation.",
    request_timeout=30,
) as learn:
    agent = Agent(
        client=OpenAIChatClient(),
        instructions="Use the learn tool to answer Microsoft documentation questions.",
        tools=learn,
    )
    response = await agent.run("How do I configure FoundryChatClient with Entra?")
```

### Per-request headers (auth tokens, tenant IDs)

Use `header_provider` to inject a header derived from `function_invocation_kwargs` on the outer `agent.run(...)` call. This avoids building a new `httpx.AsyncClient` per tenant.

```python
mcp = MCPStreamableHTTPTool(
    name="billing-api",
    url="https://mcp.example.com",
    header_provider=lambda kwargs: {"Authorization": f"Bearer {kwargs['token']}"},
)

async with mcp:
    agent = Agent(client=OpenAIChatClient(), tools=mcp)
    await agent.run(
        "What's my balance?",
        function_invocation_kwargs={"token": user_token},
    )
```

### Bring your own HTTP client

For custom TLS, retries, or observability pass an `httpx.AsyncClient`:

```python
import httpx

client = httpx.AsyncClient(timeout=30, verify="/etc/ssl/corp-ca.pem")
mcp = MCPStreamableHTTPTool(name="internal", url="https://mcp.corp/api", http_client=client)
```

## WebSocket

```python
from agent_framework import MCPWebsocketTool

async with MCPWebsocketTool(
    name="realtime",
    url="wss://service.example.com/mcp",
    description="Subscribe to real-time events.",
) as rt:
    agent = Agent(client=OpenAIChatClient(), tools=rt)
```

## Approval gates

Require human approval before specific MCP tools run:

```python
mcp = MCPStdioTool(
    name="git",
    command="uvx",
    args=["mcp-server-git"],
    approval_mode={
        "always_require_approval": ["git_push", "git_reset"],
        "never_require_approval": ["git_status", "git_diff"],
    },
)
```

Alternatives:

- `approval_mode="always_require"` — every tool invocation emits an approval event.
- `approval_mode="never_require"` — bypass approval entirely.
- `approval_mode=None` (default) — inherit the server's default.

When approval is required the workflow emits a `function_approval_request` event; respond with `event.data.to_function_approval_response(approved=True)` and re-run with that response. See the [Human-in-the-loop page](./microsoft_agent_framework_python_hitl/) for the full loop.

## Filtering which tools load

MCP servers sometimes expose hundreds of tools. Restrict what the model sees:

```python
mcp = MCPStreamableHTTPTool(
    name="github",
    url="https://mcp.github.com",
    allowed_tools=["list_issues", "create_issue", "get_pr"],
)
```

Or disable MCP prompts entirely when you only want tools:

```python
MCPStdioTool(name="fs", command="...", load_prompts=False)
```

## Parsing tool results

The default parser coerces MCP `CallToolResult` into a string for the model. Override it for structured returns (images, multi-part content):

```python
from mcp import types
from agent_framework import Content, MCPStreamableHTTPTool


def parse_image_result(result: types.CallToolResult) -> list[Content]:
    # Surface images as ImageContent instead of stringifying them
    out: list[Content] = []
    for c in result.content:
        if isinstance(c, types.ImageContent):
            out.append(...)  # build agent_framework ImageContent
        else:
            out.append(...)
    return out


mcp = MCPStreamableHTTPTool(
    name="diagrammer",
    url="https://diagrammer.example.com/mcp",
    parse_tool_results=parse_image_result,
)
```

## MCP sampling callbacks

Some MCP servers call back into the client to perform model sampling on the client's behalf. Pass a chat client so the tool can satisfy those callbacks:

```python
from agent_framework.openai import OpenAIChatClient

mcp = MCPStreamableHTTPTool(
    name="planner",
    url="https://planner.example.com/mcp",
    client=OpenAIChatClient(model="gpt-5"),
)
```

## Exposing an agent as an MCP server

Flip the direction — let other agents consume yours over MCP. The `agent_framework.devui` or `agent_framework_chatkit` hosting packages expose an agent as a streamable-HTTP MCP endpoint; see those sub-packages for the deployment recipe.

## Common patterns

**Multi-MCP agent.** Pass a list — every MCP tool's public functions are aggregated under its own prefix:

```python
async with (
    MCPStdioTool(name="fs", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]) as fs,
    MCPStreamableHTTPTool(name="learn", url="https://learn.microsoft.com/api/mcp") as learn,
):
    agent = Agent(client=OpenAIChatClient(), tools=[fs, learn])
```

**Tool + MCP mix.** MCP tools combine with plain `@tool`-decorated functions:

```python
from agent_framework import tool

@tool
def summarise(text: str) -> str:
    return " ".join(text.split()[:50])

async with MCPStdioTool(name="fs", command="...") as fs:
    agent = Agent(client=OpenAIChatClient(), tools=[fs, summarise])
```

**Quarantine risky servers.** Combine `approval_mode="always_require"` with function middleware that logs every invocation before approval.

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

Verified against `agent-framework-core==1.2.1` and `mcp==1.27`.

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
from agent_framework.openai import OpenAIChatClient

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
from agent_framework import Agent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatClient

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
from agent_framework import Agent, MCPWebsocketTool
from agent_framework.openai import OpenAIChatClient

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

### Inspecting what the server advertises

After entering the `async with` block the tool populates `functions` (and optionally `prompts`). Use this to log what's been loaded or to build a runtime tool picker:

```python
async with MCPStdioTool(
    name="github",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
) as mcp:
    for fn in mcp.functions:
        print(f"{fn.name}: {fn.description[:60]}")
    # Prompts advertised by the server (if load_prompts was True):
    for prompt in getattr(mcp, "prompts", []):
        print(f"prompt {prompt.name}: {prompt.description}")
```

## Parsing tool results

The default parser coerces MCP `CallToolResult` into a string for the model. Override it when the server returns structured data you want to surface as multi-part content (images plus alt text, JSON plus a summary, etc.):

```python
from mcp import types
from agent_framework import Content, MCPStreamableHTTPTool


def parse_image_result(result: types.CallToolResult) -> list[Content]:
    out: list[Content] = []
    for c in result.content:
        if isinstance(c, types.TextContent):
            out.append(Content.from_text(c.text))
        elif isinstance(c, types.ImageContent):
            # MCP images come as base64 — preserve them as a data URI.
            data_uri = f"data:{c.mimeType};base64,{c.data}"
            out.append(Content.from_uri(uri=data_uri, media_type=c.mimeType))
        elif isinstance(c, types.EmbeddedResource):
            res = c.resource
            text = getattr(res, "text", None)
            if text:
                out.append(Content.from_text(text))
    if result.isError:
        out.append(Content.from_error(message="MCP server returned an error"))
    return out


mcp = MCPStreamableHTTPTool(
    name="diagrammer",
    url="https://diagrammer.example.com/mcp",
    parse_tool_results=parse_image_result,
)
```

Per-tool overrides are also possible — connect once, inspect `mcp.functions`, then set `result_parser` on the individual `FunctionTool` instances you care about:

```python
async with MCPStreamableHTTPTool(name="analytics", url="https://analytics.example.com/mcp") as mcp:
    # Find the specific tool and override only its parser.
    query = next(f for f in mcp.functions if f.name == "analytics_run_query")
    query.result_parser = lambda r: Content.from_text(f"rows: {len(r.content)}")

    agent = Agent(client=OpenAIChatClient(), tools=mcp)
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

## Hosted MCP — the `SupportsMCPTool` protocol

The three MCP classes above run the MCP client **in your process**. Some providers (notably OpenAI's Responses API via certain deployments, and the Foundry model garden) can run the MCP client **server-side** — you tell the provider the MCP URL, the provider opens the connection, discovers tools, and calls them for you. No subprocess or HTTP client in your Python process.

Chat clients that support this implement `SupportsMCPTool`. Feature-detect at runtime before calling `get_mcp_tool(...)`:

```python
from agent_framework import Agent, SupportsMCPTool
from agent_framework.openai import OpenAIChatClient


client = OpenAIChatClient(model="gpt-5")

if isinstance(client, SupportsMCPTool):
    mcp_tool = client.get_mcp_tool(
        name="learn",
        url="https://learn.microsoft.com/api/mcp",
    )
    agent = Agent(
        client=client,
        instructions="Answer Microsoft documentation questions.",
        tools=[mcp_tool],
    )
    response = await agent.run("How does DefaultAzureCredential pick a credential?")
else:
    # Fall back to in-process MCP.
    from agent_framework import MCPStreamableHTTPTool
    async with MCPStreamableHTTPTool(name="learn", url="https://learn.microsoft.com/api/mcp") as mcp_tool:
        agent = Agent(client=client, tools=[mcp_tool])
        response = await agent.run("...")
```

When to prefer hosted MCP:

- The MCP server and the model provider already have a trust relationship — no need to re-mint auth tokens in your code.
- You don't want to manage a long-running HTTP client or handle reconnects.
- Latency matters — the provider can often keep a warm connection open across requests.

When to stick with in-process (`MCPStdioTool` / `MCPStreamableHTTPTool` / `MCPWebsocketTool`):

- You need `header_provider=` for multi-tenant auth — hosted MCP typically doesn't support per-request headers.
- You want `approval_mode` gating every individual tool — this is an in-process feature.
- You need to parse `CallToolResult` with a custom `parse_tool_results=` callback.

The `SupportsMCPTool` protocol is `runtime_checkable`, so the `isinstance(...)` guard is a normal runtime check — no stub subclassing needed. See the [Advanced → Capability protocols](./microsoft_agent_framework_python_advanced/#capability-protocols--supports) section for the full set of `Supports*` protocols (file search, web search, code interpreter, image generation).

## Lifecycle — connect, reset, close

The `async with mcp_tool:` idiom handles connect + close for you. For long-running servers where you want the same tool instance to survive across many agent invocations, manage the lifecycle explicitly:

```python
from agent_framework import MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatClient

mcp = MCPStreamableHTTPTool(name="learn", url="https://learn.microsoft.com/api/mcp")
client = OpenAIChatClient()                                  # reuse one HTTP pool

await mcp.connect()                                          # open once
try:
    for _ in range(100):
        agent = Agent(client=client, tools=mcp)              # reuses the open session
        await agent.run("…")
finally:
    await mcp.close()                                        # close once
```

### Reconnecting after a transport error

If the remote server drops the connection or rotates a token, call `connect(reset=True)` to tear down the stale session and open a fresh one without replacing the tool instance:

```python
try:
    response = await agent.run("Query the MCP server")
except ConnectionError:
    await mcp.connect(reset=True)                           # reset + reconnect in one call
    response = await agent.run("Query the MCP server")
```

Pair `reset=True` with your own backoff logic when you want a long-lived MCP tool that self-heals across transient failures — the default `async with` scope can't cover that because it wants a clean open/close.

### Lifetime and transport-level termination

`terminate_on_close=True` (the default for HTTP transports) closes the underlying httpx connection when `close()` runs. Set it to `False` when your `http_client=` is shared with other callers and you don't want the tool to terminate the pool:

```python
import httpx
from agent_framework import MCPStreamableHTTPTool

shared = httpx.AsyncClient(timeout=30)                      # used by other parts of your app

mcp = MCPStreamableHTTPTool(
    name="internal",
    url="https://mcp.corp/api",
    http_client=shared,
    terminate_on_close=False,                               # don't kill the shared client
    request_timeout=15,                                     # per-request timeout (seconds)
)
```

`request_timeout` applies to every MCP call (tool invocation, prompt fetch, server ping) — set it independently of the httpx client's global timeout when you want tighter per-call bounds.

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

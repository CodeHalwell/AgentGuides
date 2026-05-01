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

### Passing environment + encoding to a stdio child process

`MCPStdioTool` forwards `env`, `args`, and `encoding` straight through to `StdioServerParameters`. Anything else — buffering knobs, custom `cwd`, etc. — comes through `**kwargs` (the constructor merges it with the explicit args before constructing `StdioServerParameters`). Use it to ship secrets to a child server or to pin the wire encoding when running on Windows:

```python
import os
from agent_framework import MCPStdioTool

postgres_mcp = MCPStdioTool(
    name="pg",
    command="uvx",
    args=["mcp-server-postgres"],
    env={
        # Passed to the MCP child as environment variables. The agent process
        # still reads them here, so treat them as sensitive in logs/traces;
        # the model won't see them unless you explicitly forward them.
        "DATABASE_URL": os.environ["DATABASE_URL"],
        "PGPASSWORD": os.environ["DB_PASSWORD"],
    },
    encoding="utf-8",         # Avoid Windows cp1252 mojibake on logs / SQL output.
    request_timeout=15,        # Per-MCP-call timeout (seconds), independent of the child.
)
```

Treat the spawned process like any other dependency — log its stderr through your normal subprocess plumbing if the MCP server doesn't already forward useful diagnostics over the JSON-RPC channel.

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

The `approval_mode` parameter accepts four shapes — three string sentinels and one typed dict:

| Value | Effect |
|---|---|
| `"always_require"` | Every tool invocation emits a `function_approval_request` event. |
| `"never_require"` | Bypass approval entirely — the tool runs as soon as the model calls it. |
| `MCPSpecificApproval` (dict) | Per-tool whitelist / blacklist. |
| `None` (default) | Inherit the server's default — usually `"never_require"`. |

For per-tool control, use the typed dict — it gives you static type checking and the exact key names the framework expects:

```python
from agent_framework import MCPStdioTool, MCPSpecificApproval

# Use the TypedDict for the dict literal — IDE autocompletes the keys.
git_approval: MCPSpecificApproval = {
    "always_require_approval": ["git_push", "git_reset", "git_force_push"],
    "never_require_approval": ["git_status", "git_diff", "git_log"],
}

mcp = MCPStdioTool(
    name="git",
    command="uvx",
    args=["mcp-server-git"],
    approval_mode=git_approval,
)
```

Tools listed in **both** lists require approval (the safe default). Tools not in either list inherit the server-side default. When approval is required the workflow emits a `function_approval_request` event; respond with `event.data.to_function_approval_response(approved=True)` and re-run with that response. See the [Human-in-the-loop page](./microsoft_agent_framework_python_hitl/) for the full loop.

### Approval + middleware audit log

Pair the per-tool whitelist with a `FunctionMiddleware` that logs every approval-required call so a security team can review history asynchronously:

```python
import json
import logging
from agent_framework import FunctionMiddleware, FunctionInvocationContext

audit_log = logging.getLogger("agent.audit")


class ApprovalAudit(FunctionMiddleware):
    """Log structured records for every tool that requires approval, before it runs."""

    async def process(self, context: FunctionInvocationContext, call_next) -> None:
        if context.function.approval_mode == "always_require":
            audit_log.info(
                "approval_required",
                extra={
                    "tool": context.function.name,
                    "args": json.dumps(context.arguments or {}, default=str),
                    "session_id": (context.session.session_id if context.session else None),
                },
            )
        await call_next()


agent = Agent(
    client=OpenAIChatClient(),
    middleware=[ApprovalAudit()],
    tools=[mcp],
)
```

Combine with structured logging (JSON output → SIEM) and you have an auditable record of every privileged MCP invocation.

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

Flip the direction — let other agents consume yours over MCP. Every `Agent` (and concrete `BaseAgent` subclass) ships an `as_mcp_server()` helper that wraps the agent in an `mcp.server.lowlevel.Server` exposing one tool. Drop it into any MCP runner — stdio, the official `mcp dev` harness, or a hosted streamable-HTTP service:

```python
import anyio
from mcp.server.stdio import stdio_server
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    agent = Agent(
        client=OpenAIChatClient(),
        name="docs_agent",
        description="Answers questions about our internal documentation.",
        instructions="Only answer using the indexed docs.",
    )

    server = agent.as_mcp_server(
        server_name="docs-mcp",
        version="1.0.0",
        instructions="Use docs_agent for any question about internal docs.",
    )

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


anyio.run(main)
```

What the wrapper does for you:

- Calls `agent.as_tool()` internally to produce the single advertised tool.
- Wires `list_tools`, `call_tool`, and `set_logging_level` handlers on the MCP server.
- Forwards the agent's `name` / `description` to the MCP tool surface.
- Maps the agent's text output to MCP `TextContent`. Image/audio outputs are dropped with a warning — MCP server-side rich content forwarding isn't implemented yet.

Now any MCP client — Claude Desktop, an LLM IDE, or another `Agent` configured with `MCPStdioTool` / `MCPStreamableHTTPTool` — can drive the agent through the standard tool surface. This is the cleanest way to publish a single-purpose agent for cross-language consumption.

### As an MCP child of another agent

`as_mcp_server()` plus stdio plus another agent's `MCPStdioTool` lets you compose two agents over MCP locally without any HTTP plumbing. Useful for pipeline-style architectures where each step is a small, replaceable agent:

```python
# child_agent.py — packaged as an executable script
import anyio
from mcp.server.stdio import stdio_server
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

async def main() -> None:
    agent = Agent(client=OpenAIChatClient(), name="summariser", instructions="Summarise.")
    server = agent.as_mcp_server(server_name="summariser")
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())

anyio.run(main)
```

```python
# supervisor.py
async with MCPStdioTool(
    name="summariser",
    command="python",
    args=["child_agent.py"],
) as child:
    supervisor = Agent(client=OpenAIChatClient(), name="supervisor", tools=child)
    response = await supervisor.run("Summarise the attached doc.")
```

The two processes talk over MCP — kill or restart the child without touching the supervisor.

### Deployed MCP servers

For HTTP/SSE deployments, the `agent_framework.devui` and `agent_framework_chatkit` hosting packages turn `as_mcp_server()` output into a streamable-HTTP endpoint with auth, multi-session routing, and OpenTelemetry tracing — see those sub-packages for production recipes.

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

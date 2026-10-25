---
title: "Microsoft Agent Framework (Python) — MCP Integration"
description: "Plug Model Context Protocol servers into agent-framework with MCPStdioTool and MCPStreamableHTTPTool, or hand it your own mcp.Client. Approval gates, header injection, connection lifecycle, sampling, and serving an agent over MCP."
framework: "microsoft-agent-framework"
language: "python"
sidebar:
  label: "MCP"
  order: 12
---

Model Context Protocol (MCP) servers are first-class tool sources in `agent-framework`. Two transports ship in `agent_framework`:

| Class | Transport | Typical use |
|---|---|---|
| `MCPStdioTool` | Subprocess over stdio | Local tools — filesystem, git, npm-hosted servers |
| `MCPStreamableHTTPTool` | Streamable HTTP | Remote / hosted MCP services |

Both are async context managers. Entering one connects to the server, lists its tools and prompts (unless `load_tools=False` / `load_prompts=False`) and exposes each as a `FunctionTool`, available afterwards in `.functions`. Pass the MCP tool object itself in `tools=`; the agent expands it into those functions. An agent given an MCP tool that is not yet connected connects it on first use and closes it in `agent.close()`.

Since 1.23.0 the framework is built on version 2 of the MCP Python SDK (`mcp>=2.2,<3`). That release removed the WebSocket transport and MCP tasks, changed the SDK's own types to snake_case attributes, and added a high-level `mcp.Client` that you can hand to the framework yourself (`mcp_client=`, [below](#bring-your-own-mcp-client)). Code that only uses the framework's own classes mostly carries on working; code that touches `mcp` types or `MCPTool.session` needs the changes noted on this page.

Verified against `agent-framework` %%version=agent-framework%% and the `mcp` SDK it resolves (2.3 at the %%versions-checked%% check; first written for 1.6.0 and `mcp` 1.27). The examples were run against local stdio and streamable HTTP test servers built with `mcp`'s `MCPServer`.

## Install

```bash
pip install agent-framework  # pulls mcp and tenacity transitively
# or, for pruned installs:
pip install agent-framework-core 'mcp>=2.2,<3' tenacity
```

Since 1.23.0 a pruned install needs `tenacity` as well as `mcp`: without it, connecting fails with a `ModuleNotFoundError` ("MCP subscription recovery requires tenacity"). `agent-framework` and `agent-framework-core[all]` include both. An environment that still pins `mcp<2` cannot install `agent-framework` 1.23.0 or later.

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

- `name` identifies the MCP tool object (in logs and telemetry). It is **not** a prefix: the model sees the server's own tool names (`read_file`, `list_directory`, ...). Pass `tool_name_prefix="fs"` to expose `fs_read_file` and so on, which you need when two servers offer tools with the same name.
- `command` + `args` + `env` are forwarded to `mcp.client.stdio.StdioServerParameters`.
- Use `async with` so the subprocess is cleaned up when the agent finishes.

### Passing environment + encoding to a stdio child process

`MCPStdioTool` forwards `env`, `args`, and `encoding` straight through to `StdioServerParameters`; other `StdioServerParameters` fields, such as `cwd`, can be passed as extra keyword arguments. Use them to ship secrets to a child server or to pin the wire encoding when running on Windows:

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

Use `header_provider` to inject a header derived from `function_invocation_kwargs` on the outer `agent.run(...)` call. This avoids building a new `httpx.AsyncClient` per tenant. For headers that never change (an API key for a service account), use `static_headers={"X-Api-Key": ...}` instead.

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

With `header_provider=`, the tool connects using the `function_invocation_kwargs` of the run that first needs it, and reconnects when a later run supplies different headers, so one tool instance can serve several tenants in turn. Entering the tool with `async with` before any run (as above) also works.

### Bring your own HTTP client

For custom TLS, proxies or timeouts, pass your own client. MCP SDK 2.x uses [`httpx2`](https://pypi.org/project/httpx2/) (installed with `mcp`), so `http_client=` is typed as `httpx2.AsyncClient` since 1.23.0:

```python
import httpx2
from agent_framework import MCPStreamableHTTPTool

client = httpx2.AsyncClient(timeout=30, verify="/etc/ssl/corp-ca.pem")
mcp = MCPStreamableHTTPTool(name="internal", url="https://mcp.corp/api", http_client=client)
```

A client you pass in stays yours: the tool never closes it, so close it yourself when you are done. If you set authentication headers on that client directly, they go to every origin it talks to; `static_headers=` and `header_provider=` are only sent to the origin of `url`.

## Bring your own MCP client

`MCPStdioTool` and `MCPStreamableHTTPTool` normally build and own their connection. Since 1.23.0 you can instead pass an `mcp.Client` you have already entered, with `mcp_client=` in place of `command=` or `url=`. Use it when you need a setting the framework does not expose (callbacks, extensions, caching), or one connection shared between a tool and something else, such as [`MCPSkillsSource`](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_skills/#skills-served-over-mcp-mcpskillssource):

```python
import asyncio
from mcp import Client
from agent_framework import Agent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatClient


async def main() -> None:
    async with (
        Client("https://learn.microsoft.com/api/mcp") as mcp_client,   # a URL, StdioServerParameters or a transport
        MCPStreamableHTTPTool(name="learn", mcp_client=mcp_client) as learn,
    ):
        agent = Agent(client=OpenAIChatClient(), tools=learn)
        print((await agent.run("How do I configure FoundryChatClient with Entra?")).text)


asyncio.run(main())
```

The caller owns that client: closing or resetting the tool leaves it open, and you close it by leaving its `async with`. Connection options the framework would otherwise apply (`request_timeout`, `http_client`, `static_headers`, `header_provider`, `terminate_on_close`; `args`, `env` and `encoding` for stdio) raise `ValueError` when combined with `mcp_client=`, and sampling options on the tool are ignored with a warning; configure them on the `Client`.

`session=` (a low-level `ClientSession`) and the `MCPTool.session` property still exist but are deprecated in favour of `mcp_client=`. For a connection the framework owns, `MCPTool.session` now returns `None`.

## WebSocket (removed)

MCP SDK 2.x dropped the WebSocket transport, which was never part of the MCP specification. `MCPWebsocketTool` is still importable in 1.23.0 so old imports do not break, but constructing it emits a `DeprecationWarning` and connecting always fails with `ToolException` ("MCP WebSocket transport was removed in MCP v2"). Move to `MCPStreamableHTTPTool`; if the server only speaks WebSocket, stay on `agent-framework` 1.22 and `mcp` 1.x until it offers streamable HTTP.

## Approval gates

The `approval_mode` parameter accepts four values:

| Value | Effect |
|---|---|
| `"always_require"` | Every call to any of the server's tools needs approval. |
| `"never_require"` | No approval; the tool runs as soon as the model calls it. |
| `MCPSpecificApproval` (dict) | Per-tool lists, below. |
| `None` (default) | Same as `"never_require"`. (MCP servers have no approval setting of their own.) |

For per-tool control, use the typed dict — it gives you static type checking and the exact key names the framework expects:

```python
from agent_framework import MCPStdioTool
from agent_framework._mcp import MCPSpecificApproval  # TypedDict, not re-exported from agent_framework

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

Names are the server's own tool names (or the prefixed names, when the raw name is already in normalised form). Tools listed in **both** lists require approval. Tools in **neither** list run without approval, so list every tool you consider risky, or use `"always_require"` and put the safe ones in `never_require_approval`.

When a tool requires approval, the run stops with a `function_approval_request` in `response.user_input_requests`. Answer with `request.to_function_approval_response(approved=True)` on the same session; the loop is the same as for any tool, see the [Human-in-the-loop page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_hitl/#tool-approval).

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

`allowed_tools` matches the server's raw tool names. An empty list exposes nothing (use `load_tools=False` if you only want prompts). Or disable MCP prompts entirely when you only want tools:

```python
MCPStdioTool(name="fs", command="...", load_prompts=False)
```

### Inspecting what the server advertises

After entering the `async with` block the tool populates `functions`. The server's prompts are loaded into the same list as functions (named after the prompt), so the model can call them too; there is no separate `prompts` attribute. Use this to log what's been loaded or to build a runtime tool picker:

```python
async with MCPStdioTool(
    name="github",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-github"],
) as mcp:
    for fn in mcp.functions:
        print(f"{fn.name}: {(fn.description or '')[:60]}")
```

### Progressive disclosure for large servers

With `use_progressive_disclosure=True` (experimental, `ExperimentalFeature.PROGRESSIVE_TOOLS`) the model initially sees only the tools named in `always_load=` plus three small loader tools that list, load and unload the server's other tools on demand. This keeps a 200-tool server from filling the context window with schemas:

```python
from agent_framework import MCPStreamableHTTPTool

github = MCPStreamableHTTPTool(
    name="github",
    url="https://api.githubcopilot.com/mcp/",
    use_progressive_disclosure=True,
    always_load=["search_issues"],      # visible from the start
)
```

## Parsing tool results

By default, when a result has both `content` and `structured_content`, the structured part is preferred for the model-visible text (`tool_result_content="structured_first"`; the other modes are `"content_first"`, `"content_only"`, `"structured_only"` and `"both"`). Override it when the server returns structured data you want to surface as multi-part content (images plus alt text, JSON plus a summary, etc.). MCP SDK 2.x types use snake_case attributes (`mime_type`, `is_error`, `structured_content`); the camelCase names from SDK 1.x raise `AttributeError`:

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
            data_uri = f"data:{c.mime_type};base64,{c.data}"
            out.append(Content.from_uri(uri=data_uri, media_type=c.mime_type))
        elif isinstance(c, types.EmbeddedResource):
            res = c.resource
            text = getattr(res, "text", None)
            if text:
                out.append(Content.from_text(text))
    if result.is_error:
        out.append(Content.from_error(message="MCP server returned an error"))
    return out


mcp = MCPStreamableHTTPTool(
    name="diagrammer",
    url="https://diagrammer.example.com/mcp",
    parse_tool_results=parse_image_result,
)
```

Per-tool overrides are also possible — connect once, inspect `mcp.functions`, then set `result_parser` on the individual `FunctionTool` instances you care about. A `result_parser` runs after the MCP parsing above, so it receives the already-converted `list[Content]`, not the `CallToolResult`, and returns a string:

```python
async with MCPStreamableHTTPTool(name="analytics", url="https://analytics.example.com/mcp") as mcp:
    # Find the specific tool and override only its parser.
    query = next(f for f in mcp.functions if f.name == "run_query")   # no tool_name_prefix, so the raw name
    query.result_parser = lambda contents: f"rows: {len(contents)}"

    agent = Agent(client=OpenAIChatClient(), tools=mcp)
```

## Calling MCP tools directly

You don't need an agent to invoke MCP tools — `call_tool()` and `get_prompt()` let you drive the MCP server directly from your own code. Useful for testing, for orchestrators that don't use the agent tool loop, and for inspecting what a tool returns before wiring it to an agent:

```python
import asyncio
from agent_framework import MCPStdioTool, MCPStreamableHTTPTool


async def direct_invocation() -> None:
    # Stdio example — query a filesystem server directly
    async with MCPStdioTool(
        name="fs",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    ) as fs:
        # List what tools are available after connection
        for fn in fs.functions:
            print(fn.name, "—", (fn.description or "")[:60])

        # Call a tool by the server's own name; kwargs match its JSON schema parameters
        result = await fs.call_tool("list_directory", path="/tmp")
        print("directory listing:", result)


async def http_direct() -> None:
    # HTTP example — call a remote tool without an agent
    async with MCPStreamableHTTPTool(
        name="learn",
        url="https://learn.microsoft.com/api/mcp",
        allowed_tools=["search"],
    ) as learn:
        # Verify the server is connected
        assert learn.is_connected

        result = await learn.call_tool("search", query="FoundryChatClient authentication")
        print(result)


asyncio.run(direct_invocation())
asyncio.run(http_direct())
```

`call_tool` takes the server's tool name and returns a `list[Content]` with the default parser (or whatever your `parse_tool_results=` returns). The tool must be in `allowed_tools` (if that param is set) and the session must be connected — otherwise the call raises.

`get_prompt(name, **arguments)` renders one of the server's prompts and returns the text as a `str`. The filesystem server above defines no prompts, so the call only makes sense against a server that does; it raises if the prompt does not exist. For example, against a server that defines a `summarise` prompt: `text = await mcp.get_prompt("summarise", path="/tmp")`. Prompts are only loaded when `load_prompts=True` (the default); disable with `MCPStdioTool(..., load_prompts=False)` to skip the prompts handshake entirely on servers that don't use them.

### Checking connection state

`mcp.is_connected` is `True` after a successful `connect()` or `__aenter__` and `False` after `close()` or `__aexit__`. It does not notice a server that has gone away, so do not use it as a health check (see [reconnecting](#reconnecting-after-a-transport-error)).

```python
async with MCPStreamableHTTPTool(name="api", url="https://api.example.com/mcp") as mcp:
    print(mcp.is_connected)   # True
    print(len(mcp.functions)) # number of tools loaded from the server

# Outside the context manager:
print(mcp.is_connected)       # False
```

## MCP sampling callbacks

Some MCP servers call back into the client to perform model sampling on the client's behalf (`sampling/createMessage`). The framework treats servers as untrusted, so **every sampling request is denied** unless you pass both a `client=` to run it and a `sampling_approval_callback=` that approves it. Approved requests are capped by `sampling_max_tokens` (default 4096) and `sampling_max_requests` per connection (default 25). Sampling is deprecated in the MCP specification (2026-07-28 revision); since 1.23.0 setting any sampling option emits a deprecation warning when the tool is constructed. With a caller-owned `mcp_client=`, these options are ignored; set `sampling_callback=` on the `Client` instead.

```python
from agent_framework import Agent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatClient

def approve_sampling(params) -> bool:
    # params is the raw CreateMessageRequestParams (snake_case in MCP SDK 2.x); refuse anything large.
    return (params.max_tokens or 0) <= 1_000


mcp = MCPStreamableHTTPTool(
    name="planner",
    url="https://planner.example.com/mcp",
    client=OpenAIChatClient(model="%%model=openai-small%%"),   # runs approved sampling requests
    sampling_approval_callback=approve_sampling,
    sampling_max_requests=10,
)

async with mcp:
    agent = Agent(client=OpenAIChatClient(), tools=mcp)
    response = await agent.run("Build a week-long study plan for distributed systems.")
    print(response.text)
```

Leaving both unset is the safe default for servers that don't need sampling.

## Long-running MCP tasks (removed)

1.22 could drive MCP tasks (tools advertising `execution.taskSupport == "required"`) with `task_options=MCPTaskOptions(...)`. The MCP Python SDK 2.x does not expose the Tasks runtime, so 1.23.0 removed that support: `MCPTaskOptions` is kept only so imports still work, and passing `task_options=` is ignored with a `DeprecationWarning`. A tool that requires task execution can no longer be called through these classes; Microsoft tracks restoring it in [agent-framework#9215](https://github.com/microsoft/agent-framework/issues/9215).

## Hosted MCP — the `SupportsMCPTool` protocol

The three MCP classes above run the MCP client **in your process**. `OpenAIChatClient`, `FoundryChatClient` and `AnthropicClient` can instead have the provider run it **server-side**: you give the provider the MCP URL, and it opens the connection, discovers tools and calls them. No subprocess or HTTP client runs in your Python process, and the server must be reachable from the provider's network.

Chat clients that support this implement `SupportsMCPTool`. Feature-detect at runtime before calling `get_mcp_tool(...)`:

```python
from agent_framework import Agent, SupportsMCPTool
from agent_framework.openai import OpenAIChatClient


client = OpenAIChatClient(model="%%model=openai%%")

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

When to stick with in-process (`MCPStdioTool` / `MCPStreamableHTTPTool`):

- You need `header_provider=` for per-request, multi-tenant auth. Hosted MCP takes fixed `headers=` when you build the tool.
- You want your own function middleware and FIDES policies to see each MCP call; hosted calls never pass through them. (Hosted MCP does have its own `approval_mode=`, enforced by the provider.)
- The MCP server is on a private network or `localhost`.
- You need to parse `CallToolResult` with a custom `parse_tool_results=` callback.

The `SupportsMCPTool` protocol is `runtime_checkable`, so the `isinstance(...)` guard is a normal runtime check — no stub subclassing needed. See the [Advanced → Capability protocols](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_advanced/#capability-protocols--supports) section for the full set of `Supports*` protocols (file search, web search, code interpreter, image generation).

## Lifecycle — connect, reset, close

The framework distinguishes connections it owns (built from `command=` or `url=`) from connections you own (`mcp_client=`, or the deprecated `session=`). For its own, `connect()` opens, `connect(reset=True)` replaces and `close()` closes the connection; for yours, those calls only attach and detach the tool, and the connection's lifetime is the `async with` of your `Client`.

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

Inside an agent run, the framework checks a connection it owns before using it and reconnects if the server has gone away, so a restarted server is picked up on the next `agent.run()`. A call that still fails does not raise out of `agent.run()`; like any tool failure it becomes an error result for the model.

When you call the server directly with `call_tool()`, a dropped connection raises `ToolExecutionException` (a `ToolException`), and in 1.23.0 the tool does not reconnect by itself on that path. `is_connected` stays `True` afterwards, so catch the exception and call `connect(reset=True)`, which tears down the stale connection and opens a fresh one without replacing the tool instance:

```python
from agent_framework.exceptions import ToolException

try:
    result = await mcp.call_tool("add", a=1, b=2)
except ToolException:
    await mcp.connect(reset=True)                           # reset + reconnect in one call
    result = await mcp.call_tool("add", a=1, b=2)
```

Add backoff if the server may stay down for a while. `connect(reset=True)` does not replace a caller-owned `mcp_client`; recovering that connection is up to the `Client`.

### Lifetime and transport-level termination

`terminate_on_close=True` (the default) sends an HTTP `DELETE` to end the MCP session on the server when `close()` runs. Set it to `False` to leave the server-side session for the server to expire, for example when another process resumes it. It does not affect the HTTP client: one you pass in is never closed by the tool, whatever this flag says.

```python
import httpx2
from agent_framework import MCPStreamableHTTPTool

shared = httpx2.AsyncClient(timeout=30)                     # used by other parts of your app

mcp = MCPStreamableHTTPTool(
    name="internal",
    url="https://mcp.corp/api",
    http_client=shared,
    terminate_on_close=False,                               # leave the server-side MCP session open
    request_timeout=15,                                     # per-request timeout (seconds)
)
```

`request_timeout` applies to every MCP call (tool invocation, prompt fetch, server ping) — set it independently of the httpx client's global timeout when you want tighter per-call bounds.

## Exposing an agent as an MCP server

Flip the direction — let other agents consume yours over MCP. Every `Agent` (and concrete `BaseAgent` subclass) ships an `as_mcp_server()` helper that wraps the agent in an `mcp.server.lowlevel.Server` exposing one tool. Drop it into any MCP runner — stdio or a streamable-HTTP service. The examples below run unchanged on MCP SDK 2.x; the server logs an `MCPDeprecationWarning` because it still declares the logging capability, which the 2026-07-28 MCP revision deprecates:

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

### Serving over HTTP

`as_mcp_server()` returns a low-level MCP `Server`; to serve it over streamable HTTP, mount it with the MCP SDK's `StreamableHTTPSessionManager` in an ASGI app (Starlette and uvicorn are installed with `mcp`):

```python
# http_server.py — run with: python http_server.py
import contextlib
from collections.abc import AsyncIterator

import uvicorn
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

agent = Agent(
    client=OpenAIChatClient(),
    name="docs_agent",
    description="Answers questions about our internal documentation.",
)
server = agent.as_mcp_server(server_name="docs-mcp", version="1.0.0")
session_manager = StreamableHTTPSessionManager(app=server, stateless=True, json_response=True)


@contextlib.asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    async with session_manager.run():
        yield


app = Starlette(routes=[Mount("/mcp", app=session_manager.handle_request)], lifespan=lifespan)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

Clients connect to `http://127.0.0.1:8000/mcp/` (for example `MCPStreamableHTTPTool(name="docs", url="http://127.0.0.1:8000/mcp/")`). The endpoint has no authentication: put your own in front of it (Starlette middleware, or a gateway) before exposing it beyond your machine. `agent-framework` itself has no first-party HTTP host for MCP (DevUI and ChatKit do not serve it).

### `agent-framework-hosting-mcp` (alpha)

`agent-framework-hosting-mcp` (alpha, `1.0.0aYYMMDD` versions, install with `pip install --pre agent-framework-hosting-mcp`) is a small set of conversion helpers for when you want to own the MCP server yourself rather than take `as_mcp_server()`'s fixed shape. It does not run a server: `AgentMCPTool(agent, name=..., argument_description=...)` generates the MCP tool definition and runs the agent for `list_tools` / `call_tool` handlers you register on your own low-level `Server`; `WorkflowMCPTool` does the same for a workflow; `mcp_to_run()` and `mcp_from_run()` convert arguments and results by hand. It can also add extra schema parameters, forward selected ones as chat options, and keep an `AgentSession` per caller-supplied session id through `AgentState`. The API may change between alpha releases.

```python
from typing import Any

from mcp import types
from mcp.server import ServerRequestContext
from mcp.server.lowlevel import Server
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_framework_hosting_mcp import AgentMCPTool

agent = Agent(client=OpenAIChatClient(), name="docs_agent", description="Answers questions about our docs.")
agent_tool = AgentMCPTool(agent, name="run_agent", argument_description="The question for the agent.")


async def list_tools(ctx: ServerRequestContext[Any], params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
    return await agent_tool.list_tools()


async def call_tool(ctx: ServerRequestContext[Any], params: types.CallToolRequestParams) -> types.CallToolResult:
    return await agent_tool.call_tool(params.name, params.arguments)


server = Server("docs-mcp", on_list_tools=list_tools, on_call_tool=call_tool)
# Serve `server` over stdio or with StreamableHTTPSessionManager exactly as above.
```

## Common patterns

**Multi-MCP agent.** Pass a list. Give each server a `tool_name_prefix` when their tool names could collide:

```python
async with (
    MCPStdioTool(name="fs", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"], tool_name_prefix="fs") as fs,
    MCPStreamableHTTPTool(name="learn", url="https://learn.microsoft.com/api/mcp", tool_name_prefix="learn") as learn,
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

## Revision history

| Date | Version | Changes |
|---|---|---|
| 2026-10-25 | 1.23.0 | Updated for the move to MCP SDK 2.x: pruned installs need `tenacity`; WebSocket transport and MCP tasks removed (`MCPWebsocketTool`, `MCPTaskOptions` and `task_options` are deprecated stubs); `mcp` types are snake_case (`mime_type`, `is_error`, `max_tokens`); `http_client=` takes an `httpx2.AsyncClient`. Added `mcp_client=` and the caller-owned versus framework-owned lifecycle, a verified streamable-HTTP host for `as_mcp_server()` and the alpha `agent-framework-hosting-mcp`. Fixed earlier errors: prompts are in `.functions` (there is no `prompts` attribute), `result_parser` receives `list[Content]`, `is_connected` does not detect a dropped server, `terminate_on_close` ends the MCP session rather than closing the HTTP client. |
| 2026-10-24 | 1.22.0 | Corrected tool naming (`name` is not a prefix; use `tool_name_prefix`), the default approval (`None` means no approval; unlisted tools in an `MCPSpecificApproval` dict run unapproved), sampling (denied unless `sampling_approval_callback` approves; deprecated in MCP), WebSocket auth (no header support), hosted-MCP trade-offs and the reconnect example; removed an invented DevUI/ChatKit MCP hosting claim. Added `static_headers`, progressive disclosure, `tool_result_content` and MCP tasks. |

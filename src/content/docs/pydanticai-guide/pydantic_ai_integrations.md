---
title: "PydanticAI: Integrations — MCP, A2A and UI Protocols"
description: "Connecting PydanticAI agents to the outside world: MCP clients and servers (MCPToolset, MCP capability, MCPServerTool), Agent-to-Agent via fasta2a, AG-UI and Vercel AI SDK adapters, the built-in web chat UI and the CLI."
framework: "pydanticai"
language: "python"
sidebar:
  label: "Integrations (MCP, A2A, UI)"
  order: 18
---

Verified against **pydantic-ai %%version=pydantic-ai%%** (checked %%versions-checked%%). The MCP examples were run against that release with FastMCP 4.1 (client from the `mcp` extra, server from `pip install fastmcp`); the A2A app was built with fasta2a 2.1.

## Model Context Protocol (MCP)

[MCP](https://modelcontextprotocol.io) is an open protocol for exposing tools, resources and prompts to LLM applications. PydanticAI can be an MCP **client** in three ways, and you can put an agent behind an MCP **server**.

| Option | Where the MCP client runs | Use when |
|---|---|---|
| `MCPToolset(...)` | Your process | Full control; stdio servers; any model |
| `MCP(url)` capability | Your process by default; the provider's with `native=True` | One-line remote server, optionally offloaded to the provider |
| `NativeTool(MCPServerTool(...))` | The model provider | You always want the provider to call the server (OpenAI Responses, Anthropic, xAI) |

### `MCPToolset` — the client

`MCPToolset` replaced V1's `MCPServerStdio`, `MCPServerSSE`, `MCPServerStreamableHTTP` and `MCPServerHTTP` in 2.0. It is built on the FastMCP client and accepts:

- a URL (`'https://example.com/mcp'` — Streamable HTTP, or SSE for URLs ending `/sse`);
- a path to a Python or JavaScript server script, launched over stdio (pass a `pathlib.Path`: FastMCP 4 deprecates inferring stdio from a plain string, and FastMCP 5 will remove it);
- a FastMCP transport (`StdioTransport('uvx', args=[...])`, `StreamableHttpTransport(url, headers=...)`), a `fastmcp.Client`, or an in-process `FastMCP` server object.

```python
import asyncio
import os
from pathlib import Path

from fastmcp.client.transports import PythonStdioTransport
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

# The server script below is launched as a subprocess over stdio. Stdio servers start with a
# sanitised environment (PATH, HOME and a few others), so pass on what the server needs.
transport = PythonStdioTransport(
    Path('mcp_server.py'),
    env={'ANTHROPIC_API_KEY': os.environ['ANTHROPIC_API_KEY']},
)
# init_timeout covers start-up plus the initialize handshake (default 5 s); a server that
# imports pydantic_ai itself needs longer on a cold start.
toolset = MCPToolset(transport, init_timeout=30)
agent = Agent('anthropic:%%model=anthropic%%', toolsets=[toolset])


async def main():
    async with agent:                       # opens the MCP session once for several runs
        result = await agent.run('Write me a poem about the sea.')
    print(result.output)


asyncio.run(main())
```

- `async with agent:` keeps the MCP sessions open across several runs; without it each run connects and disconnects.
- **Environment variables do not reach stdio servers by default.** The MCP SDK passes only a short allow-list (`PATH`, `HOME`, `USER`, `SHELL`, `TERM` and their Windows equivalents). API keys and other settings must be given explicitly with `env=` on the transport, or with an `"env"` entry in an `mcpServers` config file.
- Useful options: `tool_error_behavior=` (`'retry'`, `'error'`, `'failed'`), `process_tool_call=` (inject deps or headers per call), `sampling_model=` (let the server call back into an LLM), `elicitation_handler=`, `cache_tools=`, `include_instructions=` (forward the server's instructions to the model), `headers=` / `auth=` for HTTP servers.
- Wrap in `PrefixedToolset` (`toolset.prefixed('docs')`) when two servers expose tools with the same name.
- `agent.set_mcp_sampling_model(model)` sets the sampling model on every `MCPToolset` the agent has. Sampling and elicitation need a session the server can call back over; with a FastMCP 4 server that negotiates the newer stateless session, PydanticAI warns that `sampling_model` will never be called.

### Resources and prompts

Tools are only one of the three things an MCP server offers. `MCPToolset` also reads **resources** (documents the server publishes by URI) and **prompts** (templated messages). These are not given to the model automatically; your code fetches them and decides where they go — typically into instructions or the user prompt:

```python
import asyncio
import os
from pathlib import Path

from fastmcp.client.transports import PythonStdioTransport
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

transport = PythonStdioTransport(Path('mcp_server.py'), env={'ANTHROPIC_API_KEY': os.environ['ANTHROPIC_API_KEY']})
toolset = MCPToolset(transport, init_timeout=30)
agent = Agent('anthropic:%%model=anthropic%%', toolsets=[toolset])


async def main():
    async with agent:
        print([r.uri for r in await toolset.list_resources()])
        #> ['style://house']
        style = await toolset.read_resource('style://house')        # str, BinaryContent or a list of them
        print([p.name for p in await toolset.list_prompts()])
        #> ['critique']
        prompt = await toolset.get_prompt('critique', {'poem': 'Roses are red'})
        print(prompt.messages[0].content)

        result = await agent.run('Write a poem about the sea.', instructions=f'House style: {style}')
        print(result.output)


asyncio.run(main())
```

`list_resource_templates()` lists parameterised resources, and `cache_resources=` / `cache_prompts=` (on by default) cache the listings until the server reports a change.

### Several servers from a config file

`load_mcp_toolsets()` reads the `mcpServers` JSON format used by Claude Desktop, Claude Code and Cursor, expands environment variables, and returns one prefixed `MCPToolset` per server:

```json
{
  "mcpServers": {
    "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/srv/data"]},
    "docs": {"url": "https://docs.example.com/mcp", "headers": {"Authorization": "Bearer ${DOCS_TOKEN}"}}
  }
}
```

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import load_mcp_toolsets

agent = Agent('anthropic:%%model=anthropic%%', toolsets=load_mcp_toolsets('mcp_config.json'))
```

### The `MCP` capability

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP

# Runs a local MCP client against the URL (the default since 2.0)
agent = Agent('anthropic:%%model=anthropic%%', capabilities=[MCP('https://mcp.example.com/mcp')])

# Ask the provider to call the server itself (V1's default behaviour)
agent = Agent(
    'anthropic:%%model=anthropic%%',
    capabilities=[MCP('https://mcp.example.com/mcp', native=True, authorization_token='...')],
)
```

### Exposing an agent as an MCP server

Any MCP server framework works. The `mcp` extra installs only FastMCP's client (`fastmcp-slim[client]`); for the server side install `fastmcp` itself (`pip install fastmcp`). This is the `mcp_server.py` the client examples above launch:

```python
from fastmcp import FastMCP
from pydantic_ai import Agent

poet = Agent('anthropic:%%model=anthropic-small%%', instructions='Write a short poem on the given theme.')
server = FastMCP('poetry-server')


@server.tool()
async def poem(theme: str) -> str:
    """Write a short poem about a theme."""
    result = await poet.run(f'Theme: {theme}')
    return result.output


@server.resource('style://house')
def house_style() -> str:
    """The house style guide."""
    return 'Four lines. No rhyming couplets.'


@server.prompt()
def critique(poem: str) -> str:
    """Ask for a critique of a poem."""
    return f'Critique this poem in two sentences:\n\n{poem}'


if __name__ == '__main__':
    server.run()   # stdio by default; server.run(transport='http', port=8000) for HTTP
```

For tests, pass the `FastMCP` object straight to `MCPToolset(server)`: it runs in-process with no subprocess or port (see [testing](/pydanticai-guide/pydantic_ai_testing/#6-test-an-mcp-toolset-in-process)).

---

## Agent-to-Agent (A2A)

[A2A](https://a2a-protocol.org) is a protocol for agents to call each other over HTTP. Since 2.0 the integration lives in the separate **`fasta2a`** package (V1's `Agent.to_a2a()` was removed):

```bash
pip install "fasta2a[pydantic-ai]"
```

```python
from fasta2a.pydantic_ai import agent_to_a2a
from pydantic_ai import Agent

agent = Agent('openai:%%model=openai%%', name='support', instructions='Help with orders.')
app = agent_to_a2a(agent, name='support', description='Answers order questions')
# uvicorn my_module:app --port 8000
```

`agent_to_a2a` returns an ASGI app serving the agent card and the A2A task endpoints. It accepts `storage=` and `broker=` for persistent task storage and distributed workers, plus `url=`, `version=`, `skills=` and the usual Starlette `routes=` / `middleware=` / `lifespan=`. To *call* another A2A agent from PydanticAI, wrap an A2A client call in an ordinary tool.

---

## UI protocols and front ends

Since 2.0 the UI adapters live under `pydantic_ai.ui` and need the `ui` (Vercel AI) or `ag-ui` extra. Both expose the same `dispatch_request(request, agent=...)` entry point, which parses the front end's request, runs the agent and streams the protocol's events back:

```python
from fastapi import FastAPI, Request
from starlette.responses import Response
from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui import AGUIAdapter
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

agent = Agent('openai:%%model=openai%%', instructions='Be helpful.')
app = FastAPI()


@app.post('/ag-ui')            # CopilotKit and other AG-UI clients
async def ag_ui(request: Request) -> Response:
    return await AGUIAdapter.dispatch_request(request, agent=agent)


@app.post('/api/chat')         # Vercel AI SDK useChat()
async def vercel_chat(request: Request) -> Response:
    return await VercelAIAdapter.dispatch_request(request, agent=agent)
```

`dispatch_request` also takes per-request `deps=`, `model=`, `toolsets=`, `usage_limits=`, `message_history=` and `deferred_tool_results=` (for human-in-the-loop approvals in the UI), plus `on_complete=` to persist the conversation. V1's `Agent.to_ag_ui()` and `AGUIApp` were removed in 2.0. For a hand-rolled SSE stream instead, see the [FastAPI streaming server guide](/pydanticai-guide/pydantic_ai_streaming_server_fastapi/).

### Built-in chat UI and CLI

```python
from pydantic_ai import Agent

agent = Agent('openai:%%model=openai%%', instructions='Be helpful.')

app = agent.to_web()           # Starlette app with a ready-made chat UI; run with: uvicorn my_module:app

if __name__ == '__main__':
    agent.to_cli_sync()        # interactive terminal chat
```

`to_web()` accepts `models=` (a model picker), `deps=`, `model_settings=`, `instructions=`, `html_source=` and `allowed_hosts=`; the same deps are used for every request, so use the adapters above when deps vary per user.

The `pai` command (installed by the `cli` extra, which the default bundle includes; also published standalone on PyPI as `clai`) gives the same chat without writing code:

```bash
pai -m anthropic:%%model=anthropic%% 'Summarise the A2A protocol in two lines.'   # one-shot
pai -a my_module:agent                       # chat with your own agent (or an AgentSpec YAML/JSON file)
pai --mcp-config mcp_config.json             # with the MCP servers from a config file
pai web -a my_module:agent --port 8000       # the web UI, served from the command line
```

---

## Gotchas

- **Install the extras.** `ag-ui`, `ui` and `temporal`-style extras are not in the default bundle since 2.0; `mcp` is, but only FastMCP's client — add `fastmcp` to write servers.
- **`MCP(url=...)` changed meaning in 2.0**: it now runs the client locally. Add `native=True` if you relied on the provider calling the server.
- **Tool name collisions** between MCP servers and your own tools raise at run time — prefix one side.
- **Untrusted servers**: an MCP server's tool descriptions and results go straight into the model's context. Treat third-party servers as untrusted input, and restrict them with `FilteredToolset` or approval (`ApprovalRequiredToolset`) for anything destructive.

## Reference

- `MCPToolset`, `load_mcp_toolsets` — `pydantic_ai.mcp`
- `MCP` capability — `pydantic_ai.capabilities`; `MCPServerTool` — `pydantic_ai.native_tools`
- `agent_to_a2a` — `fasta2a.pydantic_ai`
- `AGUIAdapter` — `pydantic_ai.ui.ag_ui`; `VercelAIAdapter` — `pydantic_ai.ui.vercel_ai`
- `Agent.to_web()`, `Agent.to_cli()` / `to_cli_sync()` — `pydantic_ai.agent`

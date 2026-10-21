---
title: "Pydantic AI: Integrations — MCP, A2A and UI Protocols"
description: "Connecting Pydantic AI agents to the outside world: MCP clients and servers (MCPToolset, MCP capability, MCPServerTool), Agent-to-Agent via fasta2a, AG-UI and Vercel AI SDK adapters, the built-in web chat UI and the CLI."
framework: pydanticai
---

# Pydantic AI: Integrations — MCP, A2A and UI Protocols

Verified against **pydantic-ai %%version=pydantic-ai%%** (checked %%versions-checked%%). The MCP, A2A and web-UI examples were run against that release (fastmcp via the `mcp` extra, fasta2a 2.1).

> **This page was rewritten for V2.** Earlier editions used classes that never existed in Pydantic AI (`MCPClient`, `MultiMCPClient`, `pydantic_ai.a2a.A2AAgent`, `A2AOrchestrator`, `StructuredEventStream`) and V1 provider prefixes such as `grok:` and `google-gla:`. For model providers, see the [model providers guide](/pydanticai-guide/pydantic_ai_model_providers/).

---

## Model Context Protocol (MCP)

[MCP](https://modelcontextprotocol.io) is an open protocol for exposing tools, resources and prompts to LLM applications. Pydantic AI can be an MCP **client** in three ways, and you can put an agent behind an MCP **server**.

| Option | Where the MCP client runs | Use when |
|---|---|---|
| `MCPToolset(...)` | Your process | Full control; stdio servers; any model |
| `MCP(url)` capability | Your process by default; the provider's with `native=True` | One-line remote server, optionally offloaded to the provider |
| `NativeTool(MCPServerTool(...))` | The model provider | You always want the provider to call the server (OpenAI Responses, Anthropic, xAI) |

### `MCPToolset` — the client

`MCPToolset` replaced V1's `MCPServerStdio`, `MCPServerSSE`, `MCPServerStreamableHTTP` and `MCPServerHTTP` in 2.0. It is built on the FastMCP client and accepts:

- a URL (`'https://example.com/mcp'` — Streamable HTTP, or SSE for URLs ending `/sse`);
- a path to a Python or JavaScript server script (launched over stdio);
- a FastMCP transport (`StdioTransport('uv', args=[...])`, `StreamableHttpTransport(url, headers=...)`), a `fastmcp.Client`, or an in-process `FastMCP` server.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.test import TestModel

toolset = MCPToolset('mcp_server.py')     # a .py path is launched over stdio
agent = Agent(TestModel(), toolsets=[toolset])

async def main():
    async with agent:                       # opens the MCP session once for several runs
        result = await agent.run('Write me a poem about the sea.')
    print(result.output)

asyncio.run(main())
```

- `async with agent:` keeps the MCP sessions open across several runs; without it each run connects and disconnects.
- Useful options: `tool_error_behavior=` (`'retry'`, `'error'`, `'failed'`), `process_tool_call=` (inject deps or headers per call), `sampling_model=` (let the server call back into an LLM), `elicitation_handler=`, `cache_tools=`, `include_instructions=` (forward the server's instructions to the model), `headers=` / `auth=` for HTTP servers.
- Wrap in `PrefixedToolset` (`toolset.prefixed('docs')`) when two servers expose tools with the same name.
- `agent.set_mcp_sampling_model(model)` sets the sampling model on every `MCPToolset` the agent has.

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

agent = Agent('anthropic:claude-sonnet-4-6', toolsets=load_mcp_toolsets('mcp_config.json'))
```

### The `MCP` capability

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP

# Runs a local MCP client against the URL (the default since 2.0)
agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[MCP('https://mcp.example.com/mcp')])

# Ask the provider to call the server itself (V1's default behaviour)
agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[MCP('https://mcp.example.com/mcp', native=True, authorization_token='...')],
)
```

### Exposing an agent as an MCP server

Any MCP server framework works; FastMCP ships with the `mcp` extra. This is the `mcp_server.py` the client example above launches:

```python
from fastmcp import FastMCP
from pydantic_ai import Agent

poet = Agent('test', instructions='Write a short poem on the given theme.')
server = FastMCP('poetry-server')


@server.tool()
async def poem(theme: str) -> str:
    """Write a short poem about a theme."""
    result = await poet.run(f'Theme: {theme}')
    return result.output


if __name__ == '__main__':
    server.run()   # stdio by default; server.run(transport='http', port=8000) for HTTP
```

---

## Agent-to-Agent (A2A)

[A2A](https://a2a-protocol.org) is a protocol for agents to call each other over HTTP. Since 2.0 the integration lives in the separate **`fasta2a`** package (V1's `Agent.to_a2a()` was removed):

```bash
pip install "fasta2a[pydantic-ai]"
```

```python
from fasta2a.pydantic_ai import agent_to_a2a
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2', name='support', instructions='Help with orders.')
app = agent_to_a2a(agent, name='support', description='Answers order questions')
# uvicorn my_module:app --port 8000
```

`agent_to_a2a` returns an ASGI app serving the agent card and the A2A task endpoints. It accepts `storage=` and `broker=` for persistent task storage and distributed workers, plus `url=`, `version=`, `skills=` and the usual Starlette `routes=` / `middleware=` / `lifespan=`. To *call* another A2A agent from Pydantic AI, wrap an A2A client call in an ordinary tool.

---

## UI protocols and front ends

Since 2.0 the UI adapters live under `pydantic_ai.ui` and need the `ui` (Vercel AI) or `ag-ui` extra. Both expose the same `dispatch_request(request, agent=...)` entry point, which parses the front end's request, runs the agent and streams the protocol's events back:

```python
from fastapi import FastAPI, Request
from starlette.responses import Response
from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui import AGUIAdapter
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

agent = Agent('openai:gpt-5.2', instructions='Be helpful.')
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
app = agent.to_web()           # Starlette app with a ready-made chat UI; run with uvicorn
agent.to_cli_sync()            # interactive terminal chat
```

`to_web()` accepts `models=` (a model picker), `deps=`, `model_settings=`, `instructions=` and `allowed_hosts=`; the same deps are used for every request, so use the adapters above when deps vary per user. The `clai` command (from the `cli` extra) gives the same terminal chat without writing code.

---

## Gotchas

- **Install the extras.** `ag-ui`, `ui` and `temporal`-style extras are not in the default bundle since 2.0; `mcp` is.
- **`MCP(url=...)` changed meaning in 2.0**: it now runs the client locally. Add `native=True` if you relied on the provider calling the server.
- **Tool name collisions** between MCP servers and your own tools raise at run time — prefix one side.
- **Untrusted servers**: an MCP server's tool descriptions and results go straight into the model's context. Treat third-party servers as untrusted input, and restrict them with `FilteredToolset` or approval (`ApprovalRequiredToolset`) for anything destructive.

## Reference

- `MCPToolset`, `load_mcp_toolsets` — `pydantic_ai.mcp`
- `MCP` capability — `pydantic_ai.capabilities`; `MCPServerTool` — `pydantic_ai.native_tools`
- `agent_to_a2a` — `fasta2a.pydantic_ai`
- `AGUIAdapter` — `pydantic_ai.ui.ag_ui`; `VercelAIAdapter` — `pydantic_ai.ui.vercel_ai`
- `Agent.to_web()`, `Agent.to_cli()` / `to_cli_sync()` — `pydantic_ai.agent`

---
title: "MCP and A2A"
description: "Consume MCP tools, expose agents as A2A servers, and call remote A2A agents from ADK."
framework: google-adk
language: python
sidebar:
  order: 70
---

Verified against google-adk==2.0.0 (`google/adk/tools/mcp_tool/`, `google/adk/agents/remote_a2a_agent.py`, `google/adk/a2a/`).

ADK supports both **Model Context Protocol** (Anthropic's tool-server protocol) and **Agent-to-Agent** (Google's cross-framework agent-handoff protocol). MCP flows are client-side tool toolsets; A2A flows let you expose or consume whole agents.

## Minimal example — MCP client

```python
from mcp import StdioServerParameters
from google.adk.agents import LlmAgent
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool import StdioConnectionParams

fs_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/work"],
        ),
        timeout=5.0,
    ),
    tool_filter=["read_file", "list_directory"],
    tool_name_prefix="fs_",
)

agent = LlmAgent(
    name="fs_agent",
    model="gemini-2.5-flash",
    instruction="Help the user browse the filesystem.",
    tools=[fs_toolset],
)
```

`Runner.close()` cleans toolsets automatically. For stand-alone use, call `await fs_toolset.close()`.

## MCP connection types

From `google/adk/tools/mcp_tool/mcp_session_manager.py`:

| Class | For |
|---|---|
| `StdioConnectionParams(server_params: StdioServerParameters, timeout: float = 5.0)` | Local stdio server (`npx ...`, `python3 -m ...`) |
| `SseConnectionParams(url, headers=None, timeout=5.0, sse_read_timeout=300.0, httpx_client_factory=...)` | Remote SSE MCP server |
| `StreamableHTTPConnectionParams(url, headers=None, timeout=5.0, sse_read_timeout=300.0, terminate_on_close=True, httpx_client_factory=...)` | Streamable HTTP MCP server |

`StdioServerParameters` is Anthropic's MCP type — pass `command` and `args`. ADK also accepts a bare `StdioServerParameters` directly for backwards compat, but prefer `StdioConnectionParams` when you need a timeout.

## `McpToolset` constructor

```python
toolset = McpToolset(
    connection_params=...,
    tool_filter=["read_file"],        # or a ToolPredicate callable
    tool_name_prefix="fs_",           # prepended to each tool's name
    errlog=sys.stderr,                # where the server's stderr goes
    auth_scheme=None,                 # OAuth/API-key auth for the MCP server
    auth_credential=None,
    require_confirmation=False,       # bool or predicate applied to every tool
    header_provider=lambda ctx: {"X-Tenant": ctx.state.get("tenant_id")},
    progress_callback=None,
    use_mcp_resources=False,          # adds `load_mcp_resource` tool when True
    sampling_callback=None,
    sampling_capabilities=None,
    credential_key=None,              # key for storing/loading this credential in credential service
)
```

All args are keyword-only (see `mcp_toolset.py:97-160`). Key behaviours:

- **Filtering** — `tool_filter=["name1", "name2"]` or a `ToolPredicate` `(tool, ctx) -> bool`.
- **Auth** — `auth_scheme` + `auth_credential` drive ADK's auth flow; exchanged tokens are injected as `Authorization` headers on each MCP request (`mcp_toolset.py:206-245`).
- **Progress** — `progress_callback` can be a single `ProgressFnT(progress, total, message)` or a factory that returns per-tool callbacks.
- **Resources** — set `use_mcp_resources=True` to expose MCP resources via a `load_mcp_resource` tool that the model can call.
- **Credential key** — `credential_key` is a user-specified string used to load and save this toolset's credential in a credential service. When two toolsets share the same `credential_key`, they share the same exchanged token, avoiding duplicate OAuth flows.

## Sampling (reverse-MCP)

MCP servers can call back into **your** model via the sampling mechanism. Pass `sampling_callback` and `sampling_capabilities` to let the server request completions:

```python
from mcp.types import SamplingCapability
async def handle_sampling(request, ctx):
    # delegate to your model/agent
    ...

toolset = McpToolset(
    connection_params=...,
    sampling_callback=handle_sampling,
    sampling_capabilities=SamplingCapability(...),
)
```

## Expose an ADK agent over A2A (server)

```python
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

agent = LlmAgent(name="solver", model="gemini-2.5-flash", instruction="Solve math problems.")
app = to_a2a(agent, host="0.0.0.0", port=8000, protocol="http")

# Run with:  uvicorn module_name:app --host 0.0.0.0 --port 8000
```

`to_a2a` returns a **Starlette** app. It:

1. Builds an `AgentCard` from the agent (or accepts a pre-built one via `agent_card=`).
2. Wraps the agent in a `Runner` with in-memory services (override via `runner=`).
3. Mounts the A2A RPC endpoint.
4. Optionally runs a user `lifespan` context manager for DB setup / shutdown.

Signature:

```python
to_a2a(
    agent: BaseAgent,
    *,
    host: str = "localhost",
    port: int = 8000,
    protocol: str = "http",
    agent_card: AgentCard | str | None = None,    # or path to JSON
    push_config_store: PushNotificationConfigStore | None = None,
    runner: Runner | None = None,
    lifespan: Callable | None = None,
) -> Starlette
```

For custom integration, use `A2aAgentExecutor` from `google.adk.a2a.executor.a2a_agent_executor` directly — it plugs into any A2A `DefaultRequestHandler`.

## Call a remote A2A agent (client)

Use `RemoteA2aAgent` to wrap a remote agent so it behaves like a local `BaseAgent`:

```python
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

remote_solver = RemoteA2aAgent(
    name="remote_solver",
    agent_card="https://agents.example.com/.well-known/agent.json",  # URL, path, or AgentCard
    description="Math solver hosted elsewhere",
    timeout=30.0,
)

# Compose into a larger system
root = LlmAgent(
    name="dispatcher",
    model="gemini-2.5-flash",
    instruction="For maths, transfer_to_agent('remote_solver').",
    sub_agents=[remote_solver],
)
```

Agent card sources:

- `AgentCard` object — passed straight through.
- `str` starting with `http://` / `https://` — fetched via `A2ACardResolver`.
- Any other `str` — treated as a local file path.

Constructor accepts `httpx_client`, `timeout`, `a2a_client_factory`, `a2a_request_meta_provider`, `full_history_when_stateless`, `config: A2aRemoteAgentConfig`, and `use_legacy: bool = True`. `use_legacy=False` emits the new-integration extension header (`remote_a2a_agent.py:108-212`).

## `A2aRemoteAgentConfig`

```python
from google.adk.a2a.agent.config import A2aRemoteAgentConfig, RequestInterceptor

cfg = A2aRemoteAgentConfig(
    parameters=...,
    request_interceptors=[RequestInterceptor(...)],
)
agent = RemoteA2aAgent(name="r", agent_card="...", config=cfg)
```

Interceptors mutate outgoing A2A requests (add tenant headers, signatures, logging). See `a2a/agent/config.py`.

## MCP server from ADK agent

To expose an ADK toolset (not a whole agent) as an MCP server, ADK includes helpers in `google.adk.tools.mcp_tool.conversion_utils`:

- `adk_to_mcp_tool_type(tool: BaseTool)` — convert a `BaseTool` to an MCP tool definition.
- `gemini_to_json_schema(schema)` — normalise Gemini schemas.

Wire these into a standard `mcp` server implementation. (There's no one-line `to_mcp` helper yet — the pattern is to run an `mcp.Server`, register tool definitions produced from your ADK tools, and dispatch tool calls back through `BaseTool.run_async`.)

## Patterns

### 1 — Multi-tenant filesystem MCP
One `McpToolset` per tenant with a unique `tool_name_prefix`. `header_provider=lambda ctx: {...}` rewrites tenant info per-turn. Register all toolsets on a single `LlmAgent`.

### 2 — ADK agent fronting an MCP proxy
`LlmAgent` + `McpToolset(connection_params=StreamableHTTPConnectionParams(url=...))` acts as a model-aware gateway to an existing MCP server. Add `require_confirmation=True` to gate destructive tools.

### 3 — Microservice of agents via A2A
Each team ships `to_a2a(agent, port=XXXX)`. Your orchestrator uses `RemoteA2aAgent` in `sub_agents=` to route between them. Add auth with `a2a_request_meta_provider` to sign requests.

### 4 — Hybrid local + remote
`sub_agents=[local_agent, RemoteA2aAgent(name="specialist", agent_card=...)]`. The LLM emits `transfer_to_agent("specialist")` and ADK routes through A2A transparently.

### 5 — HITL via MCP sampling
MCP server requests sampling via `sampling_callback`. ADK forwards the request to your model (possibly a different agent), returns the completion to the server. Useful for tool workflows that need human-style reasoning.

## Gotchas

- `McpToolset` is **session-scoped** — it holds a live MCP client. Always let the `Runner` manage lifecycle (it calls `close()` on shutdown), or use `async with toolset:` yourself.
- The MCP stdio server runs as a child process. Failures in the command (e.g. wrong `args`) surface as timeouts — check `errlog` for stderr.
- `RemoteA2aAgent` with `use_legacy=True` (the default) talks the legacy A2A protocol. Set `use_legacy=False` after upgrading both peers.
- `to_a2a` builds **in-memory** services when `runner=None`. For production, build your own `Runner` with Vertex / database services and pass it explicitly.
- `use_mcp_resources=True` on `McpToolset` adds a `load_mcp_resource` tool and injects available resources into the agent context — disabled by default to keep the prompt small.
- A2A classes under `google.adk.a2a` are `@a2a_experimental` — expect breaking changes.
- MCP connection params use `StdioServerParameters` from `mcp`, not from ADK. Import it from `mcp` (or `mcp.client.stdio`) depending on your `mcp` version.

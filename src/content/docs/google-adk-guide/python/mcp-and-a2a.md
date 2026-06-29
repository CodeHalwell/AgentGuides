---
title: "MCP and A2A"
description: "Consume MCP tools, expose agents as A2A servers, and call remote A2A agents from ADK."
framework: google-adk
language: python
sidebar:
  order: 70
---

Verified against google-adk==2.3.0 (`google/adk/tools/mcp_tool/`, `google/adk/agents/remote_a2a_agent.py`, `google/adk/a2a/`).

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
    tool_name_prefix="fs",
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
    tool_name_prefix="fs",             # ADK adds "_" automatically: "fs_read_file"
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
from google.adk.a2a.agent.config import (
    A2aRemoteAgentConfig,
    ParametersConfig,
    RequestInterceptor,
)
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

# A2aRemoteAgentConfig fields: converter hooks + request_interceptors.
# There is no top-level `parameters` field — attach metadata via a
# before_request interceptor that receives and mutates ParametersConfig.
async def add_auth_header(invocation_context, a2a_message, params: ParametersConfig):
    params.request_metadata = {"x-tenant-id": "acme", "Authorization": "Bearer tok"}
    return a2a_message, params

cfg = A2aRemoteAgentConfig(
    request_interceptors=[
        RequestInterceptor(before_request=add_auth_header),
    ],
)
agent = RemoteA2aAgent(name="r", agent_card="https://remote.example.com/.well-known/agent.json", config=cfg)
```

`ParametersConfig` carries `request_metadata` and `client_call_context` per outgoing message. Modify it inside a `before_request` hook; the return value `(message, params)` is forwarded to the A2A send call. `after_request` receives each incoming `A2AEvent` and can filter or transform it. See `a2a/agent/config.py`.

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

## Complete examples

### Example A — MCP filesystem toolset with filtering and prefix

```python
import asyncio
from google.genai import types
from mcp import StdioServerParameters
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.apps import App
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool import StdioConnectionParams

async def main():
    # Only expose read-only tools; ADK prepends the prefix with "_",
    # so tool_name_prefix="fs" produces "fs_read_file", "fs_list_directory", etc.
    fs_toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/workspace"],
            ),
            timeout=10.0,
        ),
        tool_filter=["read_file", "list_directory", "get_file_info"],
        tool_name_prefix="fs",
    )

    agent = LlmAgent(
        name="file_assistant",
        model="gemini-2.0-flash",
        instruction=(
            "Help the user navigate and read files. "
            "Use fs_list_directory to explore and fs_read_file to read content."
        ),
        tools=[fs_toolset],
    )

    session_service = InMemorySessionService()
    app = App(name="fs_app", root_agent=agent)
    runner = Runner(app=app, session_service=session_service)
    session = await session_service.create_session(app_name="fs_app", user_id="user1")

    async for event in runner.run_async(
        user_id="user1", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="What files are in /tmp/workspace?")]),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)

    await runner.close()

asyncio.run(main())
```

### Example B — MCP over HTTP (Streamable HTTP transport)

```python
import asyncio
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.apps import App
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool import StreamableHTTPConnectionParams

async def main():
    # Connect to a remote MCP server over HTTP
    toolset = McpToolset(
        connection_params=StreamableHTTPConnectionParams(
            url="https://my-mcp-server.example.com/mcp",
            headers={"Authorization": "Bearer my-api-key"},
            timeout=30.0,
            sse_read_timeout=300.0,
        ),
        tool_filter=["search", "get_document"],
        tool_name_prefix="kb",  # knowledge base prefix
    )

    agent = LlmAgent(
        name="kb_agent",
        model="gemini-2.0-flash",
        instruction="Search and retrieve documents from the knowledge base.",
        tools=[toolset],
    )

    session_service = InMemorySessionService()
    app = App(name="kb_app", root_agent=agent)
    runner = Runner(app=app, session_service=session_service)
    session = await session_service.create_session(app_name="kb_app", user_id="user1")

    async for event in runner.run_async(
        user_id="user1", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="Find documents about machine learning.")]),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)

    await runner.close()

asyncio.run(main())
```

### Example C — exposing an ADK agent as an A2A server

```python
# server.py — run with: python server.py
import asyncio
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.apps import App

agent = LlmAgent(
    name="weather_agent",
    model="gemini-2.0-flash",
    description="Provides weather information for any city.",
    instruction=(
        "You are a weather specialist. "
        "Answer weather questions for any city with helpful detail."
    ),
)

session_service = InMemorySessionService()
app = App(name="weather_app", root_agent=agent)
runner = Runner(app=app, session_service=session_service)

# to_a2a() wraps the runner in a Starlette ASGI app serving A2A protocol.
# The agent card is auto-generated from agent.name and agent.description.
a2a_app = to_a2a(agent, host="0.0.0.0", port=8080, runner=runner)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(a2a_app, host="0.0.0.0", port=8080)
```

### Example D — consuming a remote A2A agent with `RemoteA2aAgent`

```python
import asyncio
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.apps import App

async def main():
    # RemoteA2aAgent fetches the agent card from the remote server
    # and presents it as a local sub-agent.
    weather_remote = RemoteA2aAgent(
        name="weather_specialist",
        description="Remote weather agent that can answer weather questions.",
        # The agent card URL — served by the A2A server at /.well-known/agent.json
        agent_card="http://localhost:8080/.well-known/agent.json",
    )

    # Orchestrator routes questions to the remote agent via A2A protocol
    orchestrator = LlmAgent(
        name="orchestrator",
        model="gemini-2.0-flash",
        instruction=(
            "Route weather questions to weather_specialist. "
            "Handle other questions yourself."
        ),
        sub_agents=[weather_remote],
    )

    session_service = InMemorySessionService()
    app = App(name="main_app", root_agent=orchestrator)
    runner = Runner(app=app, session_service=session_service)
    session = await session_service.create_session(
        app_name="main_app", user_id="user1"
    )

    async for event in runner.run_async(
        user_id="user1", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="What's the weather like in Tokyo?")]),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Example E — MCP toolset with per-turn dynamic auth headers

```python
import asyncio
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.apps import App
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool import SseConnectionParams

async def main():
    # header_provider is called every turn — inject tenant-specific headers
    # ctx is a ReadonlyContext; read session state for dynamic values
    def get_tenant_headers(ctx) -> dict:
        tenant_id = ctx.state.get("tenant_id", "default")
        api_token = ctx.state.get("api_token", "")
        return {
            "X-Tenant-ID": tenant_id,
            "Authorization": f"Bearer {api_token}",
        }

    toolset = McpToolset(
        connection_params=SseConnectionParams(
            url="https://api.example.com/mcp/sse",
            timeout=15.0,
            sse_read_timeout=120.0,
        ),
        header_provider=get_tenant_headers,
        tool_name_prefix="api",
    )

    agent = LlmAgent(
        name="api_agent",
        model="gemini-2.0-flash",
        instruction="Help users interact with the API.",
        tools=[toolset],
    )

    session_service = InMemorySessionService()
    app = App(name="api_app", root_agent=agent)
    runner = Runner(app=app, session_service=session_service)

    # Create session with tenant context in state
    session = await session_service.create_session(
        app_name="api_app",
        user_id="user1",
        state={"tenant_id": "tenant-123", "api_token": "tok_abc"},
    )

    async for event in runner.run_async(
        user_id="user1", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="List all available resources.")]),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)

    await runner.close()

asyncio.run(main())
```

## Gotchas

- `McpToolset` is **session-scoped** — it holds a live MCP client. Always let the `Runner` manage lifecycle (it calls `close()` on shutdown), or use `async with toolset:` yourself.
- The MCP stdio server runs as a child process. Failures in the command (e.g. wrong `args`) surface as timeouts — check `errlog` for stderr.
- `RemoteA2aAgent` with `use_legacy=True` (the default) talks the legacy A2A protocol. Set `use_legacy=False` after upgrading both peers.
- `to_a2a` builds **in-memory** services when `runner=None`. For production, build your own `Runner` with Vertex / database services and pass it explicitly.
- `use_mcp_resources=True` on `McpToolset` adds a `load_mcp_resource` tool and injects available resources into the agent context — disabled by default to keep the prompt small.
- A2A classes under `google.adk.a2a` are `@a2a_experimental` — expect breaking changes.
- MCP connection params use `StdioServerParameters` from `mcp`, not from ADK. Import it from `mcp` (or `mcp.client.stdio`) depending on your `mcp` version.

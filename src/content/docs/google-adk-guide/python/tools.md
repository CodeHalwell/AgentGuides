---
title: "Tools (functions, agents, built-ins)"
description: "FunctionTool, AgentTool, Google built-ins, long-running tools, and confirmation flows."
framework: google-adk
language: python
sidebar:
  order: 30
---

Verified against google-adk==2.3.0 (`google/adk/tools/__init__.py`, `google/adk/tools/function_tool.py`).

Tools are the mechanism by which an `LlmAgent` calls code. Three flavours: **plain callable** (auto-wrapped into `FunctionTool`), **`BaseTool` subclass** (the built-ins + your own), and **`BaseToolset`** (dynamic tool lists — MCP, OpenAPI, custom).

## Minimal example

```python
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool, google_search

def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

agent = LlmAgent(
    name="math_and_search",
    model="gemini-2.5-flash",
    instruction="Use `add` for arithmetic. Use `google_search` for facts.",
    tools=[
        add,                       # callable → wrapped as FunctionTool
        google_search,             # built-in singleton
        FunctionTool(func=add, require_confirmation=True),  # explicit wrap
    ],
)
```

`LlmAgent` wraps bare callables with `FunctionTool(func=...)` at registration time (`llm_agent.py:178-182`). Wrap manually only when you need `require_confirmation=`.

## Public surface

Everything in `google.adk.tools` is lazy-loaded (`tools/__init__.py`):

| Name | Kind | Import note |
|---|---|---|
| `BaseTool`, `BaseToolset` | Abstract | Subclass for custom tools |
| `FunctionTool` | Class | Wraps a callable |
| `LongRunningFunctionTool` | Class | Wraps an async long-running callable |
| `AgentTool` | Class | Wraps a `BaseAgent` as a tool |
| `ExampleTool` | Class | Few-shot example injector |
| `AuthToolArguments` | Class | Auth-required tool arguments |
| `TransferToAgentTool`, `transfer_to_agent` | Class + singleton | Injected automatically when `sub_agents` is set |
| `McpToolset` | Class | Connects to an MCP server (also exported as `MCPToolset` for back-compat) |
| `APIHubToolset` | Class | Wraps APIs registered in Google API Hub |
| `ApiRegistry` | Class | Wraps Cloud API Registry MCP servers as `McpToolset` instances |
| `ToolContext` | Class | Passed to every tool via `tool_context=` |
| `google_search` | Singleton | Built-in Google Search (Gemini-side) |
| `url_context` | Singleton | Built-in URL context (Gemini-side) |
| `google_maps_grounding` | Singleton | Built-in Maps grounding |
| `enterprise_web_search` | Singleton | Enterprise web search |
| `VertexAiSearchTool` | Class | Vertex AI Search data store |
| `DiscoveryEngineSearchTool` | Class | Discovery Engine search |
| `SearchResultMode` | Enum | For `DiscoveryEngineSearchTool` |
| `load_memory`, `preload_memory` | Singletons | Long-term memory access |
| `load_artifacts` | Singleton | Reads artifacts into the prompt |
| `exit_loop` | Singleton | Sets `actions.escalate=True` from inside `LoopAgent`/`Workflow` |
| `get_user_choice` | `LongRunningFunctionTool` | HITL multi-choice prompt |

## FunctionTool

```python
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

def list_files(folder: str, tool_context: ToolContext) -> dict:
    """List files in a given folder.

    Args:
      folder: The folder path.
    Returns:
      A dict with keys `files` and `count`.
    """
    tool_context.state["last_listed"] = folder
    return {"files": ["a.txt", "b.txt"], "count": 2}

tool = FunctionTool(func=list_files, require_confirmation=False)
```

Signature rules (`function_tool.py`):

- The tool **name** is `func.__name__` (or `func.__class__.__name__` for callable objects).
- The tool **description** is the docstring — one sentence + Google-style `Args`/`Returns`. It's passed to the model verbatim, so keep it tight.
- Parameters are introspected with `inspect.signature` + `get_type_hints`. Pydantic model params are auto-converted (`_preprocess_args`, `function_tool.py:106`).
- A parameter named `tool_context` (or typed as `ToolContext`) gets the `ToolContext` injected — it is **not** exposed to the model.
- Sync and async callables both work.

**Missing mandatory args** short-circuit to an `{"error": ...}` response without calling the function, so the LLM can retry (`function_tool.py:219-224`).

### `require_confirmation`

```python
def wipe_all(scope: str) -> dict:
    "Irreversibly wipes data."
    return {"wiped": True}

tool = FunctionTool(
    func=wipe_all,
    require_confirmation=lambda scope: scope != "dry-run",
)
```

Bool or predicate. When the callable returns truthy, the tool returns `{"error": "This tool call requires confirmation..."}` and sets `tool_context.actions.skip_summarization = True`. The user then sends back a `FunctionResponse` carrying a `ToolConfirmation` payload on the next turn.

## LongRunningFunctionTool

`LongRunningFunctionTool` is a subclass of `FunctionTool` that sets `is_long_running = True` and appends a note to the tool description instructing the model **not to call the tool again if it has already returned a pending/intermediate status** (verified in `tools/long_running_tool.py`).

```python
from google.adk.tools import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext

async def start_report_job(project_id: str, tool_context: ToolContext) -> dict:
    """Launch a long-running report generation job.

    Args:
      project_id: The GCP project to generate the report for.
    Returns:
      A dict with `status` ("pending" or "done") and optionally `job_id` or `result`.
    """
    job_id = await report_service.submit(project_id)
    # Persist the job id so a follow-up poll tool can check it
    tool_context.state["report_job_id"] = job_id
    return {"status": "pending", "job_id": job_id, "message": "Report queued — check back in ~30 s"}

report_tool = LongRunningFunctionTool(func=start_report_job)

# Companion poll tool — plain callable, auto-wrapped by ADK when passed to tools=
async def check_report_status(tool_context: ToolContext) -> dict:
    """Check the status of the previously submitted report job."""
    job_id = tool_context.state.get("report_job_id")
    if not job_id:
        return {"error": "No job in progress"}
    result = await report_service.get_status(job_id)
    return result   # {"status": "done", "url": "gs://..."} or {"status": "pending"}
```

The key contract: the function **returns immediately** with a `{"status": "pending", ...}` dict. ADK delivers that response to the model, which then waits for the user to poll or for the next invocation to arrive. Do not block inside the function — that freezes the event loop.

### Distinguishing `FunctionTool` vs `LongRunningFunctionTool`

| Aspect | `FunctionTool` | `LongRunningFunctionTool` |
|---|---|---|
| `is_long_running` flag | `False` | `True` |
| Declaration description | Unchanged | "`[LONG RUNNING TOOL]` …do not call again if pending" appended |
| Return value | Anything JSON-serialisable | Same — **must** return immediately; typically `{"status": "pending", "job_id": ...}` |
| Follow-up tool needed? | No | Yes (companion poll/status tool reads from `tool_context.state`) |

### Multi-phase job with progress updates

```python
import asyncio
from google.adk.tools import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext

# Phase 1: submit the job
async def export_dataset(dataset_id: str, format: str, tool_context: ToolContext) -> dict:
    """Export a dataset in the requested format. Returns immediately with a job ID.

    Args:
      dataset_id: The dataset identifier.
      format: Export format — 'csv', 'json', or 'parquet'.
    Returns:
      A dict with status ('pending') and job_id.
    """
    job_id = f"exp-{dataset_id}-{format}"
    # Kick off async work (e.g. Cloud Run job, BigQuery export, etc.)
    asyncio.create_task(_run_export(job_id, dataset_id, format))
    tool_context.state[f"export_job:{job_id}"] = {"status": "pending", "pct": 0}
    return {"status": "pending", "job_id": job_id, "eta_seconds": 30}

async def _run_export(job_id: str, dataset_id: str, fmt: str):
    """Background coroutine — updates state for the poll tool to read."""
    await asyncio.sleep(15)   # simulate work
    # In production, update state via a callback or shared store
    print(f"[background] {job_id} complete")

# Phase 2: poll status
async def get_export_status(job_id: str, tool_context: ToolContext) -> dict:
    """Check the status of a dataset export job.

    Args:
      job_id: The job ID returned by export_dataset.
    Returns:
      A dict with status ('pending' or 'done') and optionally a download_url.
    """
    info = tool_context.state.get(f"export_job:{job_id}")
    if info is None:
        return {"error": f"No job found for id {job_id!r}"}
    return info

export_tool = LongRunningFunctionTool(func=export_dataset)
# get_export_status is a regular FunctionTool (auto-wrapped)
```

## AuthenticatedFunctionTool

`AuthenticatedFunctionTool` (experimental) is a `FunctionTool` subclass that handles the ADK authentication flow before invoking your function. It:

1. **First call** — credential not yet available → calls `CredentialManager.request_credential`, adds the auth flow to `actions.requested_auth_configs`, and returns `response_for_auth_required` (default: `"Pending User Authorization."`).
2. **Subsequent call** — credential exchanged → injects it as a `credential` keyword argument and runs your function normally.

The `credential` parameter is **not exposed to the model** — it is filtered from the function declaration.

Source: `tools/authenticated_function_tool.py`.

### API-key tool (pre-configured key)

```python
from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes
from google.adk.auth.auth_schemes import CustomAuthScheme
import httpx

# Declare the auth scheme (OpenAPI-style apiKey in header)
api_key_scheme = CustomAuthScheme(type="apiKey", **{"in": "header", "name": "X-API-Key"})

auth_cfg = AuthConfig(
    auth_scheme=api_key_scheme,
    raw_auth_credential=AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY,
        api_key="sk-my-secret-api-key",   # loaded from Secret Manager in production
    ),
)

async def search_products(query: str, max_results: int = 5, credential=None) -> dict:
    """Search the product catalogue.

    Args:
      query: Search terms.
      max_results: Maximum number of results.
    Returns:
      A dict with 'products' list.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://catalogue.internal/search",
            params={"q": query, "limit": max_results},
            headers={"X-API-Key": credential.api_key if credential else ""},
        )
        resp.raise_for_status()
        return resp.json()

search_tool = AuthenticatedFunctionTool(func=search_products, auth_config=auth_cfg)
```

### OAuth2 / OIDC (user consent flow)

```python
from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig

google_oidc = OpenIdConnectWithConfig(
    authorization_endpoint="https://accounts.google.com/o/oauth2/auth",
    token_endpoint="https://oauth2.googleapis.com/token",
    scopes=["https://www.googleapis.com/auth/calendar.readonly"],
)

calendar_auth = AuthConfig(
    auth_scheme=google_oidc,
    raw_auth_credential=AuthCredential(
        auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
        oauth2=OAuth2Auth(
            client_id="YOUR_CLIENT_ID.apps.googleusercontent.com",
            client_secret="YOUR_CLIENT_SECRET",
            redirect_uri="https://myapp.example.com/oauth/callback",
        ),
    ),
    credential_key="google-calendar",  # share token across multiple calendar tools
)

async def list_calendar_events(days_ahead: int = 7, credential=None) -> dict:
    """List upcoming calendar events.

    Args:
      days_ahead: Number of days ahead to look.
    Returns:
      A dict with 'events' list.
    """
    import httpx
    from datetime import datetime, timezone, timedelta

    if not credential or not credential.oauth2 or not credential.oauth2.access_token:
        return {"error": "no credential"}

    now = datetime.now(timezone.utc)
    time_max = now + timedelta(days=days_ahead)
    async with httpx.AsyncClient() as c:
        resp = await c.get(
            "https://www.googleapis.com/calendar/v3/calendars/primary/events",
            params={
                "timeMin": now.isoformat(),
                "timeMax": time_max.isoformat(),
                "maxResults": 20,
                "singleEvents": True,
                "orderBy": "startTime",
            },
            headers={"Authorization": f"Bearer {credential.oauth2.access_token}"},
        )
        return resp.json()

calendar_tool = AuthenticatedFunctionTool(
    func=list_calendar_events,
    auth_config=calendar_auth,
    response_for_auth_required={
        "status": "auth_required",
        "message": "Please grant calendar access via the provided link.",
    },
)
```

### Bearer token (already obtained upstream)

```python
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, HttpAuth, HttpCredentials
from google.adk.auth.auth_schemes import CustomAuthScheme
from google.adk.auth.auth_tool import AuthConfig
from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool

bearer_auth = AuthConfig(
    auth_scheme=CustomAuthScheme(type="http", scheme="bearer"),
    raw_auth_credential=AuthCredential(
        auth_type=AuthCredentialTypes.HTTP,
        http=HttpAuth(
            scheme="bearer",
            credentials=HttpCredentials(token="ya29.ALREADY_OBTAINED"),
        ),
    ),
)

async def call_internal_api(endpoint: str, credential=None) -> dict:
    """Call an internal API with the user's bearer token.

    Args:
      endpoint: API endpoint path (relative to base URL).
    Returns:
      A dict with the API response.
    """
    import httpx
    token = credential.http.credentials.token if credential else ""
    async with httpx.AsyncClient() as c:
        resp = await c.get(
            f"https://internal.example.com/api/{endpoint}",
            headers={"Authorization": f"Bearer {token}"},
        )
        return resp.json()

internal_tool = AuthenticatedFunctionTool(func=call_internal_api, auth_config=bearer_auth)
```

### `credential_key` — share tokens across tools

When multiple tools use the same OAuth provider, set the same `credential_key` so the user only completes the OAuth flow once:

```python
SHARED_GOOGLE_AUTH = AuthConfig(
    auth_scheme=OpenIdConnectWithConfig(
        authorization_endpoint="https://accounts.google.com/o/oauth2/auth",
        token_endpoint="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/calendar.readonly"],
    ),
    raw_auth_credential=AuthCredential(
        auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
        oauth2=OAuth2Auth(client_id="CLIENT_ID", client_secret="SECRET"),
    ),
    credential_key="google-workspace",  # same key → same cached token
)

gmail_tool = AuthenticatedFunctionTool(func=read_gmail, auth_config=SHARED_GOOGLE_AUTH)
calendar_tool = AuthenticatedFunctionTool(func=read_calendar, auth_config=SHARED_GOOGLE_AUTH)
```

## ExecuteBashTool and BashToolPolicy

`ExecuteBashTool` (experimental, `@experimental(FeatureName.SKILL_TOOLSET)`) lets an agent run shell commands in a sandboxed workspace. It always requests user confirmation before executing.

Source: `tools/bash_tool.py`.

### Constructor

```python
from google.adk.tools.bash_tool import ExecuteBashTool, BashToolPolicy
import pathlib

# ── Minimal — allow all commands, 30 s timeout ────────────────────────────────
bash = ExecuteBashTool()

# ── Custom workspace ──────────────────────────────────────────────────────────
bash = ExecuteBashTool(workspace=pathlib.Path("/tmp/agent-sandbox"))

# ── With policy ───────────────────────────────────────────────────────────────
policy = BashToolPolicy(
    allowed_command_prefixes=("git", "python3", "pip"),  # block everything else
    blocked_operators=("|", ";", "&&", "||", "`"),       # prevent chaining
    timeout_seconds=60,
    max_memory_bytes=512 * 1024 * 1024,   # 512 MB
    max_file_size_bytes=100 * 1024 * 1024,  # 100 MB per write
    max_child_processes=10,
)
bash = ExecuteBashTool(
    workspace=pathlib.Path("/tmp/sandbox"),
    policy=policy,
)
```

### `BashToolPolicy` fields (frozen dataclass)

| Field | Type | Default | Purpose |
|---|---|---|---|
| `allowed_command_prefixes` | `tuple[str, ...]` | `("*",)` | `"*"` = allow all; otherwise restrict to listed prefixes |
| `blocked_operators` | `tuple[str, ...]` | `()` | Shell operators that are rejected (e.g. `";", "&&", "\|"`) |
| `timeout_seconds` | `int \| None` | `30` | Wall-clock timeout; process is killed with SIGKILL on breach |
| `max_memory_bytes` | `int \| None` | `None` | Process virtual memory limit (`RLIMIT_AS`) |
| `max_file_size_bytes` | `int \| None` | `None` | Max size of any file the process writes (`RLIMIT_FSIZE`) |
| `max_child_processes` | `int \| None` | `None` | Max subprocess count (`RLIMIT_NPROC`) |

### Return format

`ExecuteBashTool.run_async` always returns a dict:

```python
{
    "stdout": "<captured stdout or '<no stdout captured>'>",
    "stderr": "<captured stderr>",
    "returncode": 0,   # int exit code
    # Present only on validation failure or execution error:
    "error": "<reason>"
}
```

### Usage example

```python
import asyncio, pathlib
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk.tools.bash_tool import ExecuteBashTool, BashToolPolicy
from google.genai import types

# Enable the experimental flag first
import os
os.environ["GOOGLE_ADK_ALLOW_FEATURES"] = "skill_toolset"

policy = BashToolPolicy(
    allowed_command_prefixes=("ls", "cat", "echo", "python3 -c"),
    blocked_operators=("|", ";", "&&", "||", "&"),
    timeout_seconds=10,
)
bash_tool = ExecuteBashTool(
    workspace=pathlib.Path("/tmp/work"),
    policy=policy,
)

agent = LlmAgent(
    name="code_runner",
    model="gemini-2.5-flash",
    instruction=(
        "You can run shell commands in /tmp/work. "
        "Always confirm before executing anything."
    ),
    tools=[bash_tool],
)

async def main():
    app = App(name="shell_demo", root_agent=agent)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="shell_demo", user_id="dev"
    )
    async for event in runner.run_async(
        user_id="dev",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="List files in the workspace")]
        ),
    ):
        if event.is_final_response() and event.content:
            print("→", "".join(p.text or "" for p in event.content.parts))

asyncio.run(main())
```

### Validation order

Before spawning a subprocess, `ExecuteBashTool` runs `_validate_command`:

1. **Prefix check** — if `allowed_command_prefixes != ("*",)`, the command must start with one of the listed prefixes (case-sensitive). Fails → `{"error": "Command not allowed..."}`.
2. **Operator check** — if any `blocked_operators` token appears in the command string (simple substring match), fails → `{"error": "Operator ... is not allowed..."}`.
3. **Confirmation** — `tool_context.request_confirmation(...)` is called on every valid command.

After validation, the subprocess is launched with:
- `cwd=workspace`
- `start_new_session=True` (for clean signal propagation)
- Resource limits applied via `preexec_fn` if `policy.max_*` fields are set

### Security notes

- `ExecuteBashTool` always requests confirmation (`request_confirmation` called before every run). In headless environments the confirmation callback must be handled by your `App` or a plugin.
- Prefix matching is a **prefix** check, not a full command parser. `allowed_command_prefixes=("git",)` would permit `git-annex` as well. Use `blocked_operators` to prevent shell injection.
- Use `workspace=` to confine the working directory; note that the process can still read/write absolute paths outside the workspace unless `RLIMIT_AS` / container isolation is also in place.

## AgentTool

Wrap a whole agent as a callable tool. The agent's `input_schema` becomes the tool's parameter schema; its reply becomes the tool's return value.

```python
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool
from pydantic import BaseModel

class ResearchIn(BaseModel):
    topic: str

researcher = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction="Research the topic and return a citation-rich paragraph.",
    input_schema=ResearchIn,
    tools=[google_search],
)

writer = LlmAgent(
    name="writer",
    model="gemini-2.5-flash",
    instruction="Use the `researcher` tool, then write a crisp 150-word brief.",
    tools=[AgentTool(agent=researcher, skip_summarization=False)],
)
```

Constructor args (`agent_tool.py:111-122`):

| Arg | Default | Purpose |
|---|---|---|
| `agent` | required | Any `BaseAgent` |
| `skip_summarization` | `False` | If `True`, the caller's model sees the raw agent output rather than summarising it |
| `include_plugins` | `True` | Inherits parent runner's plugins |
| `propagate_grounding_metadata` | `False` | Forwards grounding citations up |

## Built-in Gemini tools

These run **server-side inside Gemini** and cannot be combined freely. When mixed with custom tools, ADK wraps them automatically to stay within Gemini's single-built-in constraint (see `llm_agent.py:149-176`):

| Tool | What it does | Multi-tool-safe |
|---|---|---|
| `google_search` | Gemini's built-in Google Search grounding | Auto-wrapped as `GoogleSearchAgentTool` if needed |
| `url_context` | Gemini's built-in URL-fetch grounding | Single-use |
| `google_maps_grounding` | Gemini's Maps grounding | Single-use |
| `enterprise_web_search` | Enterprise web search grounding | Single-use |
| `VertexAiSearchTool(data_store_id=..., ...)` | Vertex AI Search data store | Auto-substituted for `DiscoveryEngineSearchTool` when mixed |
| `DiscoveryEngineSearchTool(...)` | Discovery Engine (client-side) | Fine with other tools |

```python
from google.adk.tools import VertexAiSearchTool

tool = VertexAiSearchTool(
    data_store_id="projects/my-project/locations/global/collections/default_collection/dataStores/my-store",
    bypass_multi_tools_limit=True,   # auto-substitute with DiscoveryEngine if needed
)
```

## Memory and artifact tools

```python
from google.adk.tools import load_memory, preload_memory, load_artifacts

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-pro",
    instruction="Use `load_memory` to recall past facts.",
    tools=[load_memory, preload_memory, load_artifacts],
)
```

- `load_memory` — the model calls it explicitly with a query; returns memory entries.
- `preload_memory` — **no model-visible tool call**; automatically front-loads the top-k memories into the prompt before each turn.
- `load_artifacts` — lets the model fetch a saved artifact (file) by name; requires an artifact service to be configured on the runner.

## MCP toolset

```python
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool import StdioConnectionParams
from mcp import StdioServerParameters

fs_tools = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/work"],
        ),
        timeout=5.0,
    ),
    tool_filter=["read_file", "list_directory"],
)

agent = LlmAgent(name="fs_agent", tools=[fs_tools])
```

Connection params:

| Class | For | Import |
|---|---|---|
| `StdioConnectionParams(server_params, timeout)` | Local stdio MCP server (`npx`, `python3 -m ...`) | `google.adk.tools.mcp_tool` |
| `SseConnectionParams(url, headers, timeout, sse_read_timeout, httpx_client_factory)` | Remote SSE | same |
| `StreamableHTTPConnectionParams(url, headers, timeout, sse_read_timeout, terminate_on_close, ...)` | Streamable HTTP | same |

`tool_filter` accepts a list of tool names or a `ToolPredicate` callable. `McpToolset` also supports `auth_scheme` / `auth_credential` for OAuth-gated servers, `require_confirmation=` (bool or predicate), `progress_callback=`, `use_mcp_resources=True` to expose MCP resources via a `load_mcp_resource` tool, and `credential_key` to namespace credential storage in a shared credential service.

## OpenAPI tools

### `OpenAPIToolset`

`OpenAPIToolset` parses an OpenAPI 3.x spec and generates one `RestApiTool` per operation. Each tool's name comes from the operation's `operationId` (snake-cased and truncated to 60 characters).

```python
import yaml
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from google.adk.agents import LlmAgent

# --- From a YAML spec string ---------------------------------------------------
with open("petstore.yaml") as f:
    spec_yaml = f.read()

toolset = OpenAPIToolset(
    spec_str=spec_yaml,
    spec_str_type="yaml",           # or "json"
    tool_name_prefix="petstore_",   # avoids collisions when using multiple specs
)

# --- From a pre-parsed dict ----------------------------------------------------
with open("petstore.yaml") as f:
    spec_dict = yaml.safe_load(f)

toolset = OpenAPIToolset(spec_dict=spec_dict)

# --- Use all tools from the spec -----------------------------------------------
agent = LlmAgent(
    name="petstore_agent",
    model="gemini-2.5-flash",
    instruction="Help the user browse and manage the Petstore catalogue.",
    tools=[toolset],
)

# --- Use only a specific operation ---------------------------------------------
list_pets_tool = toolset.get_tool("list_pets")   # by operationId (snake_case)
agent2 = LlmAgent(
    name="lister",
    model="gemini-2.5-flash",
    tools=[list_pets_tool],
)
```

**Constructor args** (`tools/openapi_tool/openapi_spec_parser/openapi_toolset.py`):

| Arg | Default | Purpose |
|---|---|---|
| `spec_dict` | `None` | Pre-parsed spec dictionary |
| `spec_str` | `None` | Raw spec string (use when `spec_dict` is `None`) |
| `spec_str_type` | `"json"` | `"json"` or `"yaml"` |
| `auth_scheme` | `None` | Applied to every generated tool |
| `auth_credential` | `None` | Applied to every generated tool |
| `credential_key` | `None` | Shared credential cache key for all tools |
| `tool_filter` | `None` | List of operationIds or `ToolPredicate` |
| `tool_name_prefix` | `None` | Prepended to every tool name |
| `ssl_verify` | `None` | `True` / `False` / path to CA bundle / `ssl.SSLContext` |
| `header_provider` | `None` | `(ReadonlyContext) -> dict[str, str]` — dynamic per-request headers |
| `preserve_property_names` | `False` | Keep camelCase names instead of converting to snake_case |

### Adding auth to an OpenAPI toolset

Use the `auth_helpers` module to create scheme/credential pairs without building the Pydantic objects manually:

```python
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from google.adk.tools.openapi_tool.auth.auth_helpers import (
    token_to_scheme_credential,
    service_account_scheme_credential,
    openid_url_to_scheme_credential,
    service_account_dict_to_scheme_credential,
)
from google.adk.auth.auth_credential import ServiceAccount, ServiceAccountCredential

# ── API key in a header ────────────────────────────────────────────────────────
scheme, cred = token_to_scheme_credential(
    token_type="apikey",
    location="header",
    name="X-API-Key",
    credential_value="my-secret-api-key",
)
toolset = OpenAPIToolset(spec_dict=spec, auth_scheme=scheme, auth_credential=cred)

# ── Bearer token (OAuth2 token already obtained) ───────────────────────────────
scheme, cred = token_to_scheme_credential(
    token_type="oauth2Token",
    location="header",
    name="Authorization",
    credential_value="ya29.access_token...",
)

# ── Google Service Account (JSON key file) ─────────────────────────────────────
import json
with open("service_account.json") as f:
    sa_dict = json.load(f)

scheme, cred = service_account_dict_to_scheme_credential(
    config=sa_dict,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
toolset = OpenAPIToolset(spec_dict=spec, auth_scheme=scheme, auth_credential=cred)

# ── OpenID Connect via discovery URL ──────────────────────────────────────────
scheme, cred = openid_url_to_scheme_credential(
    openid_url="https://accounts.google.com/.well-known/openid-configuration",
    scopes=["openid", "email", "profile"],
    credential_dict={
        "client_id": "YOUR_CLIENT_ID",
        "client_secret": "YOUR_CLIENT_SECRET",
        "redirect_uri": "http://localhost:8080/callback",
    },
)
```

### Dynamic headers per request

```python
from google.adk.agents.readonly_context import ReadonlyContext

def add_tenant_header(ctx: ReadonlyContext) -> dict[str, str]:
    return {
        "X-Tenant-ID": ctx.state.get("tenant_id", "default"),
        "X-Correlation-ID": ctx.invocation_id[:16],
    }

toolset = OpenAPIToolset(
    spec_dict=spec,
    header_provider=add_tenant_header,
)
```

### SSL certificate pinning (enterprise proxy)

```python
import ssl

# Trust a custom CA bundle (e.g. corporate TLS-intercepting proxy)
toolset = OpenAPIToolset(
    spec_dict=spec,
    ssl_verify="/etc/ssl/corp-ca-bundle.pem",  # path to PEM CA file
)

# Or pass an ssl.SSLContext for full control
ctx = ssl.create_default_context()
ctx.load_verify_locations("/etc/ssl/corp-ca.pem")
toolset = OpenAPIToolset(spec_dict=spec, ssl_verify=ctx)
```

### `APIHubToolset`

`APIHubToolset` fetches specs from Google Cloud API Hub and wraps them as `OpenAPIToolset` instances. Requires `google-adk[extensions]`.

```python
from google.adk.tools import APIHubToolset

toolset = APIHubToolset(
    apihub_resource_name=(
        "projects/my-project/locations/us-central1"
        "/apis/petstore-api/versions/v1/specs/openapi"
    ),
    auth_scheme=scheme,
    auth_credential=cred,
)
agent = LlmAgent(name="hub_agent", tools=[toolset])
```

## `ApplicationIntegrationToolset`

`ApplicationIntegrationToolset` generates tools from a Google Cloud Application Integration workflow or an Integration Connector resource. It automatically calls the integration's trigger or the connector's entity/action APIs.

```python
from google.adk.tools.application_integration_tool.application_integration_toolset import (
    ApplicationIntegrationToolset,
)
from google.adk.agents import LlmAgent

# ── Trigger an API-triggered integration ──────────────────────────────────────
integration_toolset = ApplicationIntegrationToolset(
    project="my-gcp-project",
    location="us-central1",
    integration="order-processor",           # integration name
    triggers=["api_trigger/process_order"],  # trigger IDs
    tool_name_prefix="order_",
    tool_instructions="Use these tools to manage order processing workflows.",
)

agent = LlmAgent(
    name="order_agent",
    model="gemini-2.5-flash",
    instruction="Help users track and process orders.",
    tools=[integration_toolset],
)
```

```python
# ── Use an Integration Connector (Salesforce, ServiceNow, etc.) ───────────────
connector_toolset = ApplicationIntegrationToolset(
    project="my-gcp-project",
    location="us-central1",
    connection="salesforce-prod",            # connector name in Integration Connectors
    entity_operations={
        "Account": ["LIST", "GET", "CREATE"],
        "Opportunity": ["LIST", "GET"],
    },
    actions=["QueryRecords"],
    tool_name_prefix="sf_",
)
```

**Constructor args:**

| Arg | Default | Purpose |
|---|---|---|
| `project` | required | GCP project ID |
| `location` | required | GCP region |
| `integration` | `None` | Integration name (for API-triggered flows) |
| `triggers` | `None` | Trigger IDs within the integration |
| `connection` | `None` | Connector name (for connector-based flows) |
| `entity_operations` | `None` | `{entity_id: ["LIST","GET","CREATE",...]}` — `[]` means all ops |
| `actions` | `None` | Connector action names to expose |
| `tool_name_prefix` | `""` | Prepended to generated tool names |
| `tool_instructions` | `""` | Appended to each tool description |
| `service_account_json` | `None` | Service account JSON string (falls back to ADC when `None`) |
| `auth_scheme` | `None` | Override auth scheme |
| `auth_credential` | `None` | Override auth credential |
| `tool_filter` | `None` | Filter generated tools |

> Exactly one of (`integration` + `triggers`) or (`connection` + one of `entity_operations`/`actions`) must be provided.

## `ToolboxToolset`

`ToolboxToolset` connects to a running [MCP Toolbox for Databases](https://github.com/googleapis/mcp-toolbox-sdk-python) server and exposes its registered tools as ADK tools. It requires `pip install google-adk[toolbox]`.

```python
from google.adk.tools.toolbox_toolset import ToolboxToolset
from google.adk.agents import LlmAgent

# ── Connect to a local toolbox server ─────────────────────────────────────────
toolset = ToolboxToolset(
    server_url="http://127.0.0.1:5000",
    toolset_name="my-db-toolset",       # expose only this named toolset
    bound_params={
        "user_id": lambda: "current_user",   # bind param from a callable
        "max_rows": 100,                      # or a static value
    },
)

agent = LlmAgent(
    name="db_agent",
    model="gemini-2.5-flash",
    instruction="Answer database questions using the available tools.",
    tools=[toolset],
)
```

```python
# ── Load specific tools by name ───────────────────────────────────────────────
toolset = ToolboxToolset(
    server_url="http://toolbox.internal:5000",
    tool_names=["search_products", "get_order"],   # load only these tools
    auth_token_getters={
        "google-auth": lambda: _get_google_id_token(),  # for OIDC-gated toolboxes
    },
    additional_headers={"X-Tenant": "acme"},
)
```

**Constructor args:**

| Arg | Default | Purpose |
|---|---|---|
| `server_url` | required | URL of the running toolbox server |
| `toolset_name` | `None` | Load all tools from this named toolset |
| `tool_names` | `None` | Load only these specific tools |
| `auth_token_getters` | `None` | `{service_name: () -> str}` for per-service auth |
| `bound_params` | `None` | `{param: value_or_callable}` — pre-bound SQL/tool params |
| `credentials` | `None` | `CredentialConfig` from `toolbox_adk` |
| `additional_headers` | `None` | Static headers for every toolbox request |

> When both `toolset_name` and `tool_names` are omitted, **all** registered tools are loaded.

## Custom `BaseTool`

Subclass `BaseTool` when you need full control over the tool schema or must modify the `LlmRequest` before it is sent. `FunctionTool` is sufficient for 95% of use cases.

```python
from typing import Any
from google.genai import types
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext


class ProductLookupTool(BaseTool):
    """Look up a product by SKU from the catalogue database."""

    def __init__(self, db_pool):
        super().__init__(
            name="lookup_product",
            description="Retrieve product details by SKU from the catalogue.",
        )
        self._db = db_pool

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "sku": types.Schema(
                        type=types.Type.STRING,
                        description="The SKU identifier (e.g. 'WIDGET-42').",
                    ),
                },
                required=["sku"],
            ),
        )

    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> dict:
        sku = args.get("sku", "").strip()
        if not sku:
            return {"error": "sku is required"}
        # Parameterised query — never interpolate LLM-supplied values directly
        row = await self._db.fetchrow(
            "SELECT name, price, stock FROM products WHERE sku = $1", sku
        )
        if row is None:
            return {"error": f"SKU {sku!r} not found"}
        tool_context.state["last_sku"] = sku
        return {"name": row["name"], "price": row["price"], "stock": row["stock"]}
```

Key overrides (`tools/base_tool.py`):

| Method | When to override |
|---|---|
| `_get_declaration()` | To define the function schema shown to the model |
| `run_async(*, args, tool_context)` | The actual execution; return a JSON-serialisable dict |
| `process_llm_request(*, tool_context, llm_request)` | To inject the tool into the request in a non-standard way (e.g. as a built-in Gemini tool block) |

Do **not** override `process_llm_request` unless you also suppress `_get_declaration` (return `None`). The default implementation calls `llm_request.append_tools([self])` which relies on `_get_declaration`.

## Custom `BaseToolset`

`BaseToolset` provides a **dynamic** list of tools — useful when available tools differ by user, tenant, or context. Implement `get_tools`.

```python
from typing import Optional
from google.adk.tools.base_toolset import BaseToolset, ToolPredicate
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.agents.readonly_context import ReadonlyContext


class RoleBasedToolset(BaseToolset):
    """Expose different tools based on the role stored in session state."""

    def __init__(self):
        super().__init__()

    async def get_tools(
        self, readonly_context: Optional[ReadonlyContext] = None
    ) -> list[BaseTool]:
        role = "guest"
        if readonly_context:
            role = readonly_context.state.get("user_role", "guest")

        tools: list[BaseTool] = [FunctionTool(func=self._read_data)]
        if role in ("editor", "admin"):
            tools.append(FunctionTool(func=self._write_data))
        if role == "admin":
            tools.append(FunctionTool(func=self._delete_data))
        return tools

    # Simple in-memory store for illustration; replace with a real DB in production
    _store: dict = {}

    async def _read_data(self, key: str) -> dict:
        """Read a value from the shared data store.

        Args:
          key: The key to read.
        Returns:
          A dict with the value.
        """
        return {"value": self._store.get(key)}

    async def _write_data(self, key: str, value: str) -> dict:
        """Write a value to the shared data store.

        Args:
          key: The key to write.
          value: The value to write.
        Returns:
          A dict with `ok: true`.
        """
        self._store[key] = value
        return {"ok": True}

    async def _delete_data(self, key: str) -> dict:
        """Delete a key from the shared data store.

        Args:
          key: The key to delete.
        Returns:
          A dict with `deleted: true`.
        """
        self._store.pop(key, None)
        return {"deleted": True}

    async def close(self) -> None:
        pass  # release DB connections, etc.


agent = LlmAgent(
    name="data_agent",
    model="gemini-2.5-flash",
    tools=[RoleBasedToolset()],
)
```

`BaseToolset` notes:
- `get_tools_with_prefix` is `@final` — override only `get_tools`.
- Results are **cached per invocation ID** to avoid redundant calls. Set `self._use_invocation_cache = False` in `__init__` to disable caching for toolsets whose tool list changes mid-turn.
- Pass a `ToolPredicate` or list of tool names to the `tool_filter` constructor arg to filter exposed tools without touching `get_tools`.
- `tool_name_prefix` prefixes every returned tool name, preventing collisions when the same toolset class is registered multiple times.

## `ToolContext` (and `Context`) API

`ToolContext` is an alias for `Context` (`tools/tool_context.py:ToolContext = Context`). Both callbacks and tools receive the same object — the type name differs only by convention. `Context` extends `ReadonlyContext` (which itself wraps `InvocationContext`).

### State

```python
# Session-scoped (default)
tool_context.state["key"] = "value"

# App-scoped (all sessions for this app)
tool_context.state["app:flag"] = True

# User-scoped (all sessions for this user)
tool_context.state["user:lang"] = "en"

# Temp (current invocation only — not persisted)
tool_context.state["temp:scratch"] = [1, 2, 3]
```

### `actions` — event steering

`tool_context.actions` is an `EventActions` object. Setting fields here modifies the emitted event:

| Field | Type | Effect |
|---|---|---|
| `skip_summarization` | `bool` | Suppress the model from narrating the tool result |
| `transfer_to_agent` | `str` | Programmatically route control to another agent |
| `escalate` | `bool` | Exit a `LoopAgent` / workflow loop |
| `state_delta` | `dict` | Merged into session state when the event is appended |
| `artifact_delta` | `dict[str, int]` | Filename → version map, recorded automatically by `save_artifact` |

### Artifacts

```python
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types as gtypes

async def export_report(format: str, tool_context: ToolContext) -> dict:
    """Export the current analysis as a file.

    Args:
      format: File format — 'pdf' or 'csv'.
    Returns:
      A dict with `filename` and `version`.
    """
    data = generate_report(format)
    mime = "application/pdf" if format == "pdf" else "text/csv"
    part = gtypes.Part(inline_data=gtypes.Blob(mime_type=mime, data=data))

    # save_artifact returns the 0-based version int
    version = await tool_context.save_artifact(
        filename=f"report.{format}",
        artifact=part,
        custom_metadata={"generated_by": "export_report"},
    )
    tool_context.actions.skip_summarization = True
    return {"filename": f"report.{format}", "version": version}

# Load it back
async def read_report(format: str, tool_context: ToolContext) -> dict:
    """Read a previously saved report.

    Args:
      format: 'pdf' or 'csv'.
    Returns:
      A dict with metadata.
    """
    part = await tool_context.load_artifact(f"report.{format}")
    if part is None:
        return {"error": "no report found"}
    meta = await tool_context.get_artifact_version(f"report.{format}")
    return {"version": meta.version, "uri": meta.canonical_uri}

# List all artifacts in this session
async def list_reports(tool_context: ToolContext) -> list[str]:
    """List all saved report filenames."""
    return await tool_context.list_artifacts()
```

### Memory (from tools)

All three memory methods require a `memory_service` to be configured on the `Runner`.

```python
from google.adk.memory.memory_entry import MemoryEntry

async def remember_preference(pref: str, tool_context: ToolContext) -> dict:
    """Explicitly store a user preference in long-term memory.

    Args:
      pref: The preference to remember (e.g. 'prefers metric units').
    Returns:
      A dict with `ok: true`.
    """
    await tool_context.add_memory(
        memories=[MemoryEntry(content=pref)],
    )
    return {"ok": True}

async def recall(query: str, tool_context: ToolContext) -> dict:
    """Search long-term memory for relevant past information.

    Args:
      query: Search query.
    Returns:
      A dict with `results` list.
    """
    response = await tool_context.search_memory(query)
    return {"results": [m.content for m in response.memories]}

# End-of-turn: commit entire session to memory (usually done in after_agent_callback)
async def flush_to_memory(tool_context: ToolContext) -> dict:
    """Commit this session's conversation to long-term memory."""
    await tool_context.add_session_to_memory()
    return {"flushed": True}
```

`add_events_to_memory` is also available — pass a list of `Event` objects if you only want to store specific turns rather than the whole session.

### Credentials (OAuth / API-key flows)

```python
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig

GOOGLE_OAUTH_CONFIG = AuthConfig(
    auth_scheme=OpenIdConnectWithConfig(
        authorization_endpoint="https://accounts.google.com/o/oauth2/auth",
        token_endpoint="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/calendar.readonly"],
    ),
    raw_auth_credential=None,  # filled in after the OAuth flow
)

async def list_calendar_events(tool_context: ToolContext) -> dict:
    """List upcoming calendar events."""
    cred = tool_context.get_auth_response(GOOGLE_OAUTH_CONFIG)
    if cred is None:
        # Pause the tool and request the OAuth flow
        tool_context.request_credential(GOOGLE_OAUTH_CONFIG)
        return {"status": "auth_required"}
    # Use cred.oauth2.access_token to call the Calendar API
    ...
    return {"events": [...]}
```

`request_credential` sets `actions.requested_auth_configs` on the current event. The framework pauses execution; when the user completes OAuth the runner resumes. On the next turn `get_auth_response` returns the exchanged token.

Use `save_credential` / `load_credential` in **callback** contexts where `function_call_id` is not available.

### Confirmation (HITL gate)

```python
async def wipe_database(scope: str, tool_context: ToolContext) -> dict:
    """Wipe a database scope.

    Args:
      scope: Target scope ('staging' or 'production').
    Returns:
      A dict with the operation outcome.
    """
    confirmed = tool_context.tool_confirmation
    if confirmed is None:
        # Ask for confirmation on first call
        tool_context.request_confirmation(
            hint=f"Confirm wiping '{scope}' database? Reply 'yes' to proceed.",
            payload={"scope": scope},
        )
        return {"status": "awaiting_confirmation"}
    # Second call arrives with tool_context.tool_confirmation set
    if confirmed.payload.get("scope") == scope:
        _do_wipe(scope)
        return {"wiped": True, "scope": scope}
    return {"error": "scope mismatch"}
```

### Workflow-specific properties

These are only meaningful when the tool/callback runs inside a `Workflow` node (not a plain `LlmAgent`):

| Property | Type | Purpose |
|---|---|---|
| `tool_context.route` | `str \| bool \| int \| list` | Set routing value for conditional workflow edges |
| `tool_context.output` | `Any` | Set the node's output value directly |
| `tool_context.attempt_count` | `int` | 1-based retry attempt counter (1 = first run) |
| `tool_context.resume_inputs` | `dict[str, Any]` | Input values returned from HITL interrupts, keyed by interrupt ID |
| `tool_context.node_path` | `str` | Full path of the current node in the workflow graph |
| `tool_context.run_id` | `str` | Execution ID of the current node run |

### ReadonlyContext

`ReadonlyContext` is a read-only view of the session context, passed to:
- Dynamic instruction providers (`instruction=my_fn` on `LlmAgent`)
- `BaseToolset.get_tools(readonly_context=...)`

```python
from google.adk.agents.readonly_context import ReadonlyContext

async def dynamic_instruction(ctx: ReadonlyContext) -> str:
    lang = ctx.state.get("user:preferred_language", "en")
    return f"Always respond in language code '{lang}'."

agent = LlmAgent(
    name="localised",
    model="gemini-2.5-flash",
    instruction=dynamic_instruction,
)
```

Available on `ReadonlyContext`: `user_content`, `invocation_id`, `agent_name`, `state` (read-only `MappingProxyType`), `session`, `user_id`, `run_config`, `get_credential(key)`.

## PubSubToolset (experimental)

`PubSubToolset` exposes three Google Cloud Pub/Sub operations as ADK tools: `publish_message`, `pull_messages`, and `acknowledge_messages`. Install prerequisite: `pip install google-cloud-pubsub`.

```python
from google.adk.tools.pubsub.pubsub_toolset import PubSubToolset
from google.adk.tools.pubsub.config import PubSubToolConfig
from google.adk.agents import LlmAgent

toolset = PubSubToolset(
    tool_filter=["publish_message", "pull_messages"],    # optional filter
    pubsub_tool_config=PubSubToolConfig(project_id="my-gcp-project"),
)

agent = LlmAgent(
    name="event_router",
    model="gemini-2.5-flash",
    instruction="Publish and consume messages on Pub/Sub topics as instructed.",
    tools=[toolset],
)
```

> Full constructor reference, message tool signatures, and more examples → [Class deep dives — vol. 2](./google_adk_class_deep_dives_v2/#5--pubsubtoolset-experimental).

## SpannerToolset (experimental)

`SpannerToolset` exposes Cloud Spanner schema inspection and SQL execution. Provides `spanner_list_table_names`, `spanner_get_table_schema`, `spanner_execute_sql`, `spanner_similarity_search`, and more. Requires `pip install google-cloud-spanner`.

```python
from google.adk.tools.spanner.spanner_toolset import SpannerToolset
from google.adk.tools.spanner.settings import SpannerToolSettings, QueryResultMode
from google.adk.agents import LlmAgent

settings = SpannerToolSettings(
    max_executed_query_result_rows=50,
    query_result_mode=QueryResultMode.DICT_LIST,  # {column: value} per row
)

toolset = SpannerToolset(spanner_tool_settings=settings)

agent = LlmAgent(
    name="sql_assistant",
    model="gemini-2.5-pro",
    instruction=(
        "You are a Spanner SQL assistant. List tables first, then inspect schemas "
        "before writing queries. Cap all results at 50 rows."
    ),
    tools=[toolset],
)
```

> Full constructor reference, vector similarity search setup, and more examples → [Class deep dives — vol. 2](./google_adk_class_deep_dives_v2/#6--spannertoolset-experimental).

## SkillToolset (experimental)

`SkillToolset` is an experimental feature that adds a skills system to an `LlmAgent`. A "skill" is a folder containing a `SKILL.md` instruction file (with optional `references/`, `assets/`, `scripts/` subfolders). The toolset dynamically exposes tools for the model to discover, load, and execute those skills.

```python
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.skills import Skill, SkillFrontmatter
from google.adk.agents import LlmAgent

# Build a skill from content in memory
python_skill = Skill(
    name="python_helper",
    frontmatter=SkillFrontmatter(
        name="python_helper",
        description="Helps write and explain Python code.",
        version="1.0.0",
    ),
    instructions="""
# Python Helper

Use this skill when the user asks for help with Python.

## Steps
1. Understand the user's Python question.
2. Provide a concise, working code example.
3. Explain each line briefly.
""",
)

toolset = SkillToolset(skills=[python_skill])

agent = LlmAgent(
    name="coder",
    model="gemini-2.5-flash",
    instruction="You are a coding assistant with access to skills.",
    tools=[toolset],
)
```

**Tools the model receives:**

| Tool name | Purpose |
|---|---|
| `list_skills` | Lists all registered skills with names and descriptions |
| `load_skill(skill_name)` | Reads the full `SKILL.md` for a given skill |
| `load_skill_resource(skill_name, file_path)` | Reads a file from `references/`, `assets/`, or `scripts/` |
| `run_skill_script(skill_name, script_path, ...)` | Runs a script from `scripts/` (requires `code_executor`) |
| `search_skills(query)` | Semantic search (requires `registry=`) |

**Constructor args:**

| Arg | Type | Default | Purpose |
|---|---|---|---|
| `skills` | `list[Skill]` | `[]` | Statically-defined skills |
| `registry` | `SkillRegistry \| None` | `None` | Dynamic skill registry (enables `search_skills`) |
| `code_executor` | `BaseCodeExecutor \| None` | `None` | Executor for `run_skill_script` |
| `script_timeout` | `int` | `300` | Shell-script timeout in seconds |
| `additional_tools` | `list[ToolUnion] \| None` | `None` | Extra tools unlocked when a skill is activated |

**Loading from disk:**

```python
from google.adk.skills import Skill
import pathlib

skill_dir = pathlib.Path("my_skills/python_helper")
skill = Skill.from_directory(skill_dir)   # reads SKILL.md + subfolders
toolset = SkillToolset(skills=[skill])
```

> **Experimental**: `SkillToolset` is decorated with `@experimental(FeatureName.SKILL_TOOLSET)`. Its API may change in future releases. Suppress the warning with `GOOGLE_ADK_IGNORE_WARNINGS=skill_toolset`.

## Agent transfer

`transfer_to_agent` and `TransferToAgentTool` are injected automatically by ADK when the LLM agent has `sub_agents`. You rarely construct them yourself, but you can inspect them for logging.

## HITL tools

- `get_user_choice` — a `LongRunningFunctionTool` that prompts the user with a list; the LLM picks from the returned choice.
- `request_input` via `ToolContext.request_confirmation()` — any tool can pause and solicit input.

## Patterns

### 1 — Typed function tools
Annotate parameters with Pydantic models. `FunctionTool` converts `dict` → model via `model_validate`. The model sees the JSON schema; your function receives a validated Pydantic instance.

### 2 — Tool chains via `AgentTool`
Wrap a specialist agent as a tool for a generalist. Set `skip_summarization=True` when the specialist's output is already polished.

### 3 — Guardrail with `require_confirmation`
For destructive ops, pass a predicate that returns `True` only for risky inputs (e.g. `scope != "dry-run"`).

### 4 — Gemini-side search + local DB
Put `google_search` first and a `FunctionTool` wrapping your DB helper second. ADK auto-wraps `google_search` so the two coexist.

### 5 — Dynamic MCP toolset
Spin up `McpToolset` at runtime (e.g. per-tenant filesystem); pass `tool_name_prefix=` to avoid collisions with other toolsets. The `Runner` auto-closes toolsets on `runner.close()`.

## Gotchas

- In ADK 2.3.0 `output_schema=` and `tools=` can be used together on an `LlmAgent` — tools run during the thought loop and the schema is enforced only on the final reply.
- `tool_context` is injected by parameter name (`tool_context`) **or** type (`ToolContext`). Any other parameter of type `ToolContext` would also be treated as the context slot.
- `FunctionTool` treats the first sentence of the docstring as the tool description. Keep it focused — the model obeys it.
- Built-in Gemini tools (`google_search`, `url_context`, `google_maps_grounding`) cannot coexist freely. ADK tries to wrap them, but if you hit `400 INVALID_ARGUMENT` try `bypass_multi_tools_limit=True` where available.
- `LongRunningFunctionTool` is just a `FunctionTool` with `is_long_running=True`. The model is separately instructed not to re-call it while pending.
- Mutating `tool_context.state` with a reserved prefix (`app:`, `user:`, `temp:`) changes scope — see [runner-and-sessions](./runner-and-sessions/).

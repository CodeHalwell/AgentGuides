---
title: "Class deep dives — volume 5 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.1.0 classes: VertexAiSessionService, VertexAiSearchTool, VertexAiCodeExecutor, APIHubToolset, ToolboxToolset, ConversationScenario/ConversationScenarios, TrajectoryEvaluator/ToolTrajectoryCriterion, AuthConfig/AuthHandler, PreloadMemoryTool, and CodeExecutorContext."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 5"
  order: 64
---

Source-verified against **google-adk==2.1.0** (installed from PyPI, May 2026). Every field name, signature, and code example is taken directly from the installed package source.

| # | Class | Module | Status |
|---|---|---|---|
| 1 | `VertexAiSessionService` | `google.adk.sessions.vertex_ai_session_service` | Stable |
| 2 | `VertexAiSearchTool` | `google.adk.tools.vertex_ai_search_tool` | Stable |
| 3 | `VertexAiCodeExecutor` | `google.adk.code_executors.vertex_ai_code_executor` | Stable |
| 4 | `APIHubToolset` | `google.adk.tools.apihub_tool` | Stable |
| 5 | `ToolboxToolset` | `google.adk.tools.toolbox_toolset` | Stable |
| 6 | `ConversationScenario` + `ConversationScenarios` + `ConversationGenerationConfig` | `google.adk.evaluation.conversation_scenarios` | Stable |
| 7 | `TrajectoryEvaluator` + `ToolTrajectoryCriterion` | `google.adk.evaluation` | Stable |
| 8 | `AuthConfig` + `AuthHandler` | `google.adk.auth` | Stable |
| 9 | `PreloadMemoryTool` | `google.adk.tools.preload_memory_tool` | Stable |
| 10 | `CodeExecutorContext` | `google.adk.code_executors` | Stable |

---

## 1 · `VertexAiSessionService`

`google.adk.sessions.vertex_ai_session_service.VertexAiSessionService` connects your agent to the **Vertex AI Agent Engine Session Service** — a managed, persistent, cloud-backed store for session state and event history. It is the production alternative to `InMemorySessionService` and `SqliteSessionService`, providing automatic scaling, multi-tenant isolation, and event replay.

Internally the service wraps the `vertexai.Client().aio` async API client. The `app_name` you pass to `create_session` / `get_session` is resolved to a **Reasoning Engine ID** by one of three strategies (verified from `_get_reasoning_engine_id`):

1. If `agent_engine_id` was provided at construction time, that ID is always used regardless of `app_name`.
2. If `app_name` is a plain integer string (`"12345678"`), it is used directly as the reasoning engine ID.
3. Otherwise `app_name` must be the full resource name `projects/{project}/locations/{location}/reasoningEngines/{id}` — the ID is extracted with a regex.

### Constructor (verified `vertex_ai_session_service.py`)

```python
VertexAiSessionService(
    project: Optional[str] = None,
    location: Optional[str] = None,
    agent_engine_id: Optional[str] = None,
    *,
    express_mode_api_key: Optional[str] = None,
)
```

| Parameter | Type | Notes |
|-----------|------|-------|
| `project` | `str \| None` | GCP project ID. Falls back to ADC / env. |
| `location` | `str \| None` | GCP region, e.g. `"us-central1"`. |
| `agent_engine_id` | `str \| None` | Numeric reasoning engine ID. Overrides `app_name` resolution when set. |
| `express_mode_api_key` | `str \| None` | API key for Vertex AI Express Mode. Falls back to `GOOGLE_API_KEY` env var when `GOOGLE_GENAI_USE_VERTEXAI=true`. Do **not** use a Google AI Studio key here. |

Requires `google-cloud-aiplatform` extra: `pip install google-adk[gcp]`.

### Key methods

| Method | Description |
|--------|-------------|
| `create_session(app_name, user_id, state, session_id, **kwargs)` | Creates and returns a `Session`. Pass `expire_time='2026-10-01T00:00:00Z'` via `**kwargs` to set TTL. |
| `get_session(app_name, user_id, session_id, config)` | Returns `Session` with full event history. `GetSessionConfig.num_recent_events=0` skips event loading. |
| `list_sessions(app_name, user_id)` | Returns `ListSessionsResponse`. `user_id=None` returns all sessions for the app. |
| `delete_session(app_name, user_id, session_id)` | Permanently deletes a session from Vertex AI. |
| `append_event(session, event)` | Serialises and persists an event via the sessions events API. Handles `raw_event` vs legacy field fallback transparently. |

### Example 1 — standard Vertex AI session store

```python
import asyncio
import os
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService

# Authentication: set GOOGLE_APPLICATION_CREDENTIALS or run `gcloud auth application-default login`
session_service = VertexAiSessionService(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location="us-central1",
    agent_engine_id=os.environ["AGENT_ENGINE_ID"],  # numeric resource ID
)

agent = LlmAgent(
    name="support_agent",
    model="gemini-2.5-flash",
    instruction="You are a helpful customer support agent.",
)

async def main():
    runner = Runner(
        agent=agent,
        app_name=os.environ["AGENT_ENGINE_ID"],  # plain numeric ID
        session_service=session_service,
    )

    # Create a session with initial state
    session = await session_service.create_session(
        app_name=os.environ["AGENT_ENGINE_ID"],
        user_id="user-42",
        state={"tier": "premium", "region": "EMEA"},
        session_id="session-001",                  # optional; auto-generated if omitted
    )
    print(f"Created session: {session.id}")

    # Run the agent
    from google.genai import types
    async for event in runner.run_async(
        user_id="user-42",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="Hello!")]),
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Example 2 — using full resource name as app_name

```python
import asyncio, os
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService

# Full resource name — no agent_engine_id needed at construction time
session_service = VertexAiSessionService(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location="us-central1",
)

full_resource_name = (
    f"projects/{os.environ['GOOGLE_CLOUD_PROJECT']}"
    f"/locations/us-central1/reasoningEngines/{os.environ['AGENT_ENGINE_ID']}"
)

async def main():
    session = await session_service.create_session(
        app_name=full_resource_name,
        user_id="user-99",
        expire_time="2026-12-31T23:59:59Z",    # **kwargs forwarded to API
    )
    print(f"Session expires: {session.id}")
    
    # List all sessions for a user
    response = await session_service.list_sessions(
        app_name=full_resource_name,
        user_id="user-99",
    )
    print(f"Sessions: {[s.id for s in response.sessions]}")

asyncio.run(main())
```

### Example 3 — Express Mode (no GCP project needed)

```python
import asyncio, os
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService

# Express Mode: GOOGLE_GENAI_USE_VERTEXAI=true + a Vertex Express API key
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

session_service = VertexAiSessionService(
    express_mode_api_key=os.environ["VERTEX_EXPRESS_API_KEY"],
    # project / location / agent_engine_id omitted for Express Mode
)

agent = LlmAgent(name="my_agent", model="gemini-2.5-flash", instruction="Help the user.")

async def main():
    runner = Runner(agent=agent, app_name="my-app", session_service=session_service)
    session = await session_service.create_session(app_name="my-app", user_id="u1")
    from google.genai import types
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="Tell me a joke.")]),
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

asyncio.run(main())
```

### When to use `VertexAiSessionService` vs alternatives

| Service | Persistence | Multi-tenant | Use case |
|---------|-------------|--------------|----------|
| `InMemorySessionService` | No | No | Testing, demos |
| `SqliteSessionService` | Yes (file) | Single-process | Local dev, low-scale |
| `DatabaseSessionService` | Yes (DB) | Yes | Self-hosted production |
| `VertexAiSessionService` | Yes (managed) | Yes | GCP production, Agent Engine |

---

## 2 · `VertexAiSearchTool`

`google.adk.tools.vertex_ai_search_tool.VertexAiSearchTool` is a **model built-in tool** that injects a Vertex AI Search retrieval configuration directly into the Gemini `GenerateContentConfig`. The model calls the data store or search engine transparently — no Python function execution involved.

Unlike `FunctionTool`, the `name` and `description` fields are unused; the retrieval is handled server-side by Gemini. The tool overrides `process_llm_request` to append a `types.Tool(retrieval=types.Retrieval(vertex_ai_search=...))` config block.

A key design feature: **subclass and override `_build_vertex_ai_search_config`** to dynamically compute filters from the session state at runtime.

### Constructor (verified `vertex_ai_search_tool.py`)

```python
VertexAiSearchTool(
    *,
    data_store_id: Optional[str] = None,
    data_store_specs: Optional[list[types.VertexAISearchDataStoreSpec]] = None,
    search_engine_id: Optional[str] = None,
    filter: Optional[str] = None,
    max_results: Optional[int] = None,
    bypass_multi_tools_limit: bool = False,
)
```

**Constraint:** exactly one of `data_store_id` or `search_engine_id` must be provided (not both, not neither). If `data_store_specs` is set, `search_engine_id` is required. `ValueError` is raised otherwise.

**Gemini 1.x constraint:** cannot be combined with other tools. Use Gemini 2.x (`bypass_multi_tools_limit=True` lifts the internal check for engines that support it).

### Example 1 — data store search

```python
from google.adk.agents import LlmAgent
from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool
from google.adk.runners import InMemoryRunner
import asyncio

# Full data store resource name
DATA_STORE = (
    "projects/my-project/locations/global"
    "/collections/default_collection"
    "/dataStores/my-support-docs"
)

search_tool = VertexAiSearchTool(
    data_store_id=DATA_STORE,
    max_results=5,
)

agent = LlmAgent(
    name="support_agent",
    model="gemini-2.5-flash",
    instruction="Answer questions using the company knowledge base.",
    tools=[search_tool],
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="support")
    await runner.session_service.create_session(
        app_name="support", user_id="u1", session_id="s1"
    )
    from google.genai import types
    async for event in runner.run_async(
        user_id="u1",
        session_id="s1",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="How do I reset my password?")]
        ),
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Example 2 — search engine with multiple data store specs

```python
from google.adk.agents import LlmAgent
from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool
from google.genai import types

SEARCH_ENGINE = (
    "projects/my-project/locations/global"
    "/collections/default_collection"
    "/engines/my-blended-engine"
)

# Blend two data stores in a single search engine request
search_tool = VertexAiSearchTool(
    search_engine_id=SEARCH_ENGINE,
    data_store_specs=[
        types.VertexAISearchDataStoreSpec(
            data_store=(
                "projects/my-project/locations/global"
                "/collections/default_collection"
                "/dataStores/public-docs"
            )
        ),
        types.VertexAISearchDataStoreSpec(
            data_store=(
                "projects/my-project/locations/global"
                "/collections/default_collection"
                "/dataStores/internal-kb"
            )
        ),
    ],
    max_results=10,
    bypass_multi_tools_limit=True,  # allow alongside other tools
)

agent = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction="Research questions using both public and internal docs.",
    tools=[search_tool],
)
```

### Example 3 — dynamic filter via subclass override

```python
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool
from google.adk.agents.readonly_context import ReadonlyContext
from google.genai import types
import asyncio

DATA_STORE = "projects/my-project/locations/global/collections/default_collection/dataStores/products"

class UserSpecificSearchTool(VertexAiSearchTool):
    """Restricts search to products available in the user's region."""

    def _build_vertex_ai_search_config(
        self, ctx: ReadonlyContext
    ) -> types.VertexAISearch:
        region = ctx.state.get("user_region", "US")
        tier = ctx.state.get("user_tier", "standard")
        return types.VertexAISearch(
            datastore=self.data_store_id,
            filter=f"region = '{region}' AND tier IN ('{tier}', 'all')",
            max_results=self.max_results,
        )

search_tool = UserSpecificSearchTool(data_store_id=DATA_STORE, max_results=5)

agent = LlmAgent(
    name="product_agent",
    model="gemini-2.5-flash",
    instruction="Help users find products available in their region.",
    tools=[search_tool],
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="shop")
    await runner.session_service.create_session(
        app_name="shop",
        user_id="u1",
        session_id="s1",
        state={"user_region": "EMEA", "user_tier": "premium"},
    )
    from google.genai import types as gtypes
    async for event in runner.run_async(
        user_id="u1",
        session_id="s1",
        new_message=gtypes.Content(
            role="user",
            parts=[gtypes.Part(text="Show me laptops available for me.")]
        ),
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

asyncio.run(main())
```

---

## 3 · `VertexAiCodeExecutor`

`google.adk.code_executors.vertex_ai_code_executor.VertexAiCodeExecutor` executes Python code through the **Vertex AI Code Interpreter Extension** — a managed sandboxed runtime on GCP. It supports uploading input files, executing multi-step code, and retrieving output files (images, CSVs, data files).

Internally it manages an `Extension` object. If `resource_name` is provided, it loads the existing extension; otherwise it creates a new one. Output files are returned as `File` objects attached to the `CodeExecutionResult`.

### Constructor (verified `vertex_ai_code_executor.py`)

```python
VertexAiCodeExecutor(
    resource_name: str = None,
    **data,
)
```

| Parameter | Type | Notes |
|-----------|------|-------|
| `resource_name` | `str \| None` | Existing Code Interpreter Extension resource name. Format: `projects/{id}/locations/{location}/extensions/{ext_id}`. If `None`, a new extension is created automatically. |
| `**data` | — | Forwarded to `BaseCodeExecutor` (e.g. `optimize_data_file=True`, `stateful=True`). |

Requires: `pip install google-adk[gcp]` and the Vertex AI Code Interpreter Extension API enabled in your project.

### Example 1 — data analysis with output files

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.code_executors.vertex_ai_code_executor import VertexAiCodeExecutor
from google.adk.runners import Runner
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService
import os

code_executor = VertexAiCodeExecutor()  # creates a new Code Interpreter Extension

agent = LlmAgent(
    name="data_analyst",
    model="gemini-2.5-flash",
    instruction=(
        "You are a data analyst. When asked to analyse data, "
        "write and execute Python code using pandas and matplotlib."
    ),
    code_executor=code_executor,
)

async def main():
    session_service = VertexAiSessionService(
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location="us-central1",
        agent_engine_id=os.environ["AGENT_ENGINE_ID"],
    )
    runner = Runner(
        agent=agent,
        app_name=os.environ["AGENT_ENGINE_ID"],
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name=os.environ["AGENT_ENGINE_ID"],
        user_id="analyst-1",
    )
    from google.genai import types
    async for event in runner.run_async(
        user_id="analyst-1",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=(
                "Plot a bar chart showing monthly sales: "
                "Jan=120, Feb=95, Mar=140, Apr=180. "
                "Save the chart as sales.png."
            ))],
        ),
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Example 2 — reuse an existing extension resource

```python
import asyncio, os
from google.adk.agents import LlmAgent
from google.adk.code_executors.vertex_ai_code_executor import VertexAiCodeExecutor

# Load a pre-existing extension to avoid recreation cost on every deploy
EXTENSION_RESOURCE = (
    f"projects/{os.environ['GOOGLE_CLOUD_PROJECT']}"
    "/locations/us-central1/extensions/1234567890"
)

code_executor = VertexAiCodeExecutor(resource_name=EXTENSION_RESOURCE)

agent = LlmAgent(
    name="code_agent",
    model="gemini-2.5-flash",
    instruction="Execute the code snippets requested by the user.",
    code_executor=code_executor,
)
```

### Comparison: code executor options

| Executor | Where code runs | Network access | Files | GCP required |
|----------|----------------|----------------|-------|--------------|
| `UnsafeLocalCodeExecutor` | Local process | Yes (unsafe) | Host FS | No |
| `BuiltInCodeExecutor` | Gemini sandbox | No | Limited | No |
| `VertexAiCodeExecutor` | Vertex AI Extension | Controlled | Full | Yes |
| `AgentEngineSandboxCodeExecutor` | Agent Engine sandbox | No | Session | Yes |
| `ContainerCodeExecutor` | Docker container | Configurable | Volume | No |

---

## 4 · `APIHubToolset`

`google.adk.tools.apihub_tool.APIHubToolset` auto-generates ADK tools from an **API Hub** resource. Given a resource name that points to an OpenAPI spec, it parses the spec and creates a `RestApiTool` (or `OpenAPIToolset`) for each operation — no manual tool writing needed.

`apihub_resource_name` resolution (verified from source):
- If the name includes a spec path segment, that spec's content is used directly.
- If the name ends at an API or version, the **first spec of the first version** is automatically selected.
- Access tokens or service account JSON are passed to the internal `APIHubClient` for fetching specs from the Hub.

### Constructor (verified `apihub_toolset.py`)

```python
APIHubToolset(
    *,
    apihub_resource_name: str,
    access_token: Optional[str] = None,
    service_account_json: Optional[str] = None,
    name: str = "",
    description: str = "",
    lazy_load_spec: bool = False,
    auth_scheme: Optional[AuthScheme] = None,
    auth_credential: Optional[AuthCredential] = None,
    apihub_client: Optional[APIHubClient] = None,
    tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
)
```

| Parameter | Type | Notes |
|-----------|------|-------|
| `apihub_resource_name` | `str` | **Required.** Full API Hub resource path. |
| `access_token` | `str \| None` | Short-lived access token (`gcloud auth print-access-token`). |
| `service_account_json` | `str \| None` | Service account key JSON string. |
| `lazy_load_spec` | `bool` | If `True`, spec is fetched on first `get_tools()` call, not at construction time. |
| `auth_scheme` | `AuthScheme \| None` | Auth scheme applied to all generated tools. |
| `auth_credential` | `AuthCredential \| None` | Credential applied to all generated tools. |
| `tool_filter` | `ToolPredicate \| list[str] \| None` | Predicate or name list to expose a subset of tools. |

### Example 1 — load all operations from an API Hub spec

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.tools.apihub_tool import APIHubToolset
from google.adk.runners import InMemoryRunner

# Replace with your actual project and API Hub resource
APIHUB_RESOURCE = (
    "projects/my-project/locations/us-central1"
    "/apis/customer-api/versions/v2"
)

toolset = APIHubToolset(
    apihub_resource_name=APIHUB_RESOURCE,
    service_account_json=open("sa-key.json").read(),
)

agent = LlmAgent(
    name="api_agent",
    model="gemini-2.5-flash",
    instruction="Use the customer API to fulfil user requests.",
    tools=[toolset],
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="apihub_demo")
    await runner.session_service.create_session(
        app_name="apihub_demo", user_id="u1", session_id="s1"
    )
    from google.genai import types
    async for event in runner.run_async(
        user_id="u1",
        session_id="s1",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="List all customers in the UK.")]
        ),
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Example 2 — filter to a subset of operations

```python
from google.adk.agents import LlmAgent
from google.adk.tools.apihub_tool import APIHubToolset

APIHUB_RESOURCE = "projects/my-project/locations/us-central1/apis/orders-api"

# Only expose read operations via a list filter
toolset = APIHubToolset(
    apihub_resource_name=APIHUB_RESOURCE,
    service_account_json=open("sa-key.json").read(),
    tool_filter=["getOrder", "listOrders", "searchOrders"],
    lazy_load_spec=True,  # defer spec fetch until first use
)

agent = LlmAgent(
    name="orders_reader",
    model="gemini-2.5-flash",
    instruction="Answer read-only questions about orders.",
    tools=[toolset],
)
```

### Example 3 — API Hub toolset with OAuth2 credentials

```python
from google.adk.agents import LlmAgent
from google.adk.tools.apihub_tool import APIHubToolset
from google.adk.auth import AuthCredential, AuthCredentialTypes
from google.adk.auth.auth_schemes import OAuthGrantType
from fastapi.openapi.models import OAuth2, OAuthFlowClientCredentials, OAuthFlows

APIHUB_RESOURCE = "projects/my-project/locations/us-central1/apis/crm-api"

oauth_scheme = OAuth2(
    flows=OAuthFlows(
        clientCredentials=OAuthFlowClientCredentials(
            tokenUrl="https://auth.example.com/token",
            scopes={"crm.read": "Read CRM data"},
        )
    )
)

oauth_credential = AuthCredential(
    auth_type=AuthCredentialTypes.OAUTH2,
    oauth2={
        "client_id": "my-client-id",
        "client_secret": "my-client-secret",
    },
)

toolset = APIHubToolset(
    apihub_resource_name=APIHUB_RESOURCE,
    access_token="ya29.xxxxx",    # for Hub API access
    auth_scheme=oauth_scheme,     # applied to generated tool calls
    auth_credential=oauth_credential,
)

agent = LlmAgent(
    name="crm_agent",
    model="gemini-2.5-flash",
    instruction="Query the CRM system for customer information.",
    tools=[toolset],
)
```

---

## 5 · `ToolboxToolset`

`google.adk.tools.toolbox_toolset.ToolboxToolset` integrates with **MCP Toolbox for Databases** — Google's open-source middleware that exposes SQL databases (PostgreSQL, AlloyDB, Spanner, BigQuery, MySQL), REST APIs, and other backends as MCP-compatible tools over HTTP. The ADK class is a thin delegation wrapper around `toolbox-adk.ToolboxToolset`.

Install extra: `pip install google-adk[toolbox]`. Requires a running Toolbox server (`docker run us-central1-docker.pkg.dev/database-toolbox/toolbox/toolbox:latest`).

### Constructor (verified `toolbox_toolset.py`)

```python
ToolboxToolset(
    server_url: str,
    toolset_name: Optional[str] = None,
    tool_names: Optional[List[str]] = None,
    auth_token_getters: Optional[Mapping[str, Callable[[], str]]] = None,
    bound_params: Optional[Mapping[str, Union[Callable[[], Any], Any]]] = None,
    credentials: Optional[CredentialConfig] = None,
    additional_headers: Optional[Mapping[str, str]] = None,
    **kwargs,
)
```

| Parameter | Type | Notes |
|-----------|------|-------|
| `server_url` | `str` | URL of the running Toolbox server. |
| `toolset_name` | `str \| None` | Load a named toolset from the server config. |
| `tool_names` | `list[str] \| None` | Load specific tools by name. `toolset_name` and `tool_names` can both be set — union is loaded. If both omitted, **all tools** are loaded. |
| `auth_token_getters` | `Mapping[str, Callable[[], str]] \| None` | Mapping of auth service name → callable returning a bearer token. |
| `bound_params` | `Mapping[str, ...] \| None` | Bind parameter names to static values or callables (e.g. inject `user_id` from session). |
| `credentials` | `CredentialConfig \| None` | Toolbox-specific credential config. |
| `additional_headers` | `Mapping[str, str] \| None` | Static HTTP headers sent with every request to the server. |

### Example 1 — basic PostgreSQL toolset

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.tools.toolbox_toolset import ToolboxToolset
from google.adk.runners import InMemoryRunner

# Toolbox server exposes tools defined in its YAML config
# pointing at a PostgreSQL database
toolset = ToolboxToolset(
    server_url="http://127.0.0.1:5000",
    toolset_name="postgres-tools",
)

agent = LlmAgent(
    name="db_agent",
    model="gemini-2.5-flash",
    instruction="Answer questions about orders by querying the database.",
    tools=[toolset],
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="db_demo")
    await runner.session_service.create_session(
        app_name="db_demo", user_id="u1", session_id="s1"
    )
    from google.genai import types
    async for event in runner.run_async(
        user_id="u1",
        session_id="s1",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="How many orders were placed in March 2026?")]
        ),
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Example 2 — specific tools with bound user context

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.toolbox_toolset import ToolboxToolset
from google.adk.tools.tool_context import ToolContext

# Inject `user_id` from the session state into every tool call automatically
def get_user_id(tc: ToolContext) -> str:
    return tc.state.get("user_id", "anonymous")

toolset = ToolboxToolset(
    server_url="http://127.0.0.1:5000",
    tool_names=["get_user_orders", "get_order_details", "cancel_order"],
    bound_params={"user_id": get_user_id},    # injected per-call; not exposed to LLM
    auth_token_getters={
        "my-auth-service": lambda: "Bearer ya29.xxxxxxx"
    },
)

agent = LlmAgent(
    name="order_agent",
    model="gemini-2.5-flash",
    instruction="Help users manage their orders. The user_id is injected automatically.",
    tools=[toolset],
)
```

### Example 3 — AlloyDB with additional auth headers

```python
from google.adk.agents import LlmAgent
from google.adk.tools.toolbox_toolset import ToolboxToolset
import os

# Toolbox running in Cloud Run, secured with an Identity token
def get_identity_token() -> str:
    import subprocess
    result = subprocess.run(
        ["gcloud", "auth", "print-identity-token"],
        capture_output=True, text=True
    )
    return result.stdout.strip()

toolset = ToolboxToolset(
    server_url=os.environ["TOOLBOX_URL"],           # Cloud Run service URL
    toolset_name="alloydb-analytics",
    additional_headers={"Authorization": f"Bearer {get_identity_token()}"},
)

agent = LlmAgent(
    name="analytics_agent",
    model="gemini-2.5-pro",
    instruction="Perform data analysis on the AlloyDB analytics database.",
    tools=[toolset],
)
```

---

## 6 · `ConversationScenario`, `ConversationScenarios`, `ConversationGenerationConfig`

These three classes (`google.adk.evaluation.conversation_scenarios`) power **scenario-based evaluation** — multi-turn conversations between a **simulated user** and the agent under test. Unlike `EvalCase`/`EvalSet` (which compare against fixed expected outputs), scenario eval generates dynamic user turns guided by a **conversation plan**.

### `ConversationScenario` (verified source)

```python
class ConversationScenario(EvalBaseModel):
    starting_prompt: str
    """Fixed first user message sent to the agent."""

    conversation_plan: str
    """Natural language instructions for the user simulator."""

    user_persona: Optional[UserPersona] = None
    """A UserPersona object or a persona ID string (auto-resolved to defaults)."""
```

### `ConversationScenarios`

```python
class ConversationScenarios(EvalBaseModel):
    scenarios: list[ConversationScenario]
```

A container for serialising/deserialising a list of scenarios to/from JSON.

### `ConversationGenerationConfig`

```python
class ConversationGenerationConfig(EvalBaseModel):
    count: int
    """Number of scenarios to generate."""

    generation_instruction: Optional[str] = None
    """Natural language goal to guide generation."""

    environment_context: Optional[str] = None
    """'Ground truth' context: data the agent's tools can access."""

    model_name: str
    """Gemini model used for generation, e.g. 'gemini-2.5-flash'."""
```

### Example 1 — manual scenario construction

```python
from google.adk.evaluation.conversation_scenarios import (
    ConversationScenario,
    ConversationScenarios,
)
from google.adk.evaluation import AgentEvaluator

travel_scenarios = ConversationScenarios(scenarios=[
    ConversationScenario(
        starting_prompt="I need to book a flight.",
        conversation_plan=(
            "First, book a one-way flight from SFO to LAX for next Tuesday. "
            "Prefer a morning flight, budget under $150. If the agent finds a valid "
            "flight, confirm the booking. Then rent a standard car for three days "
            "from the airport. Once both tasks complete, your goal is done."
        ),
    ),
    ConversationScenario(
        starting_prompt="What hotels are near the conference centre?",
        conversation_plan=(
            "You need a hotel within walking distance of the Moscone Centre in SF, "
            "checking in Friday, checking out Sunday. Budget is $250/night. "
            "If found, ask about breakfast options. Accept the first suitable option."
        ),
        user_persona="budget_traveler",  # auto-resolved from default persona registry
    ),
])

# Serialise to JSON for storage
import json
print(json.dumps(travel_scenarios.model_dump(), indent=2))
```

### Example 2 — run scenario eval with AgentEvaluator

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.evaluation import AgentEvaluator, PrebuiltMetrics
from google.adk.evaluation.conversation_scenarios import (
    ConversationScenario,
    ConversationScenarios,
)

# Define the agent under test
agent = LlmAgent(
    name="travel_agent",
    model="gemini-2.5-flash",
    instruction="You are a travel booking assistant.",
    tools=[],   # add real booking tools here
)

scenarios = ConversationScenarios(scenarios=[
    ConversationScenario(
        starting_prompt="I need to fly from NYC to London next month.",
        conversation_plan=(
            "You want to fly business class from JFK to LHR, departing around the "
            "15th. Budget is $3000. If the agent finds options, pick the one with "
            "the fewest stops. Once booked, ask to add a hotel for 5 nights near "
            "central London. Accept if under $200/night."
        ),
    ),
])

async def run_scenario_eval():
    evaluator = AgentEvaluator(
        agent=agent,
        eval_dataset=scenarios,
        metrics=[PrebuiltMetrics.RESPONSE_EVALUATION_SCORE],
    )
    results = await evaluator.evaluate_async()
    for result in results:
        print(f"Score: {result.overall_score:.2f} — {result.overall_eval_status}")

asyncio.run(run_scenario_eval())
```

### Example 3 — LLM-generated scenarios via `ConversationGenerationConfig`

```python
from google.adk.evaluation.conversation_scenarios import ConversationGenerationConfig

# Describe the agent + available data, and the LLM will generate scenarios
gen_config = ConversationGenerationConfig(
    count=5,
    generation_instruction=(
        "Generate scenarios that test both happy-path bookings and edge cases "
        "like sold-out flights, missing passport info, and budget conflicts."
    ),
    environment_context=(
        "Available flights: SFO→LAX (AM, $129), SFO→LAX (PM, $89), "
        "JFK→LHR (non-stop, $2800), JFK→LHR (1-stop, $1900). "
        "Hotels: Marriott SF ($220/night), Hilton SF ($195/night, no breakfast)."
    ),
    model_name="gemini-2.5-flash",
)

# Pass gen_config to AgentEvaluator or ConversationScenarios generator
# (generation via AgentEvaluator API — see evaluation guide for full pipeline)
print(gen_config.model_dump_json(indent=2))
```

### `user_persona` values

`ConversationScenario.user_persona` accepts either a `UserPersona` object or a string ID resolved against the default `UserPersonaRegistry`. Common default persona IDs vary by version; pass a fully-constructed `UserPersona` for deterministic behaviour.

---

## 7 · `TrajectoryEvaluator` + `ToolTrajectoryCriterion`

`google.adk.evaluation.trajectory_evaluator.TrajectoryEvaluator` evaluates whether the agent called the **right tools in the right order**. It compares the agent's actual tool call sequence against a reference sequence using one of three match modes defined in `ToolTrajectoryCriterion.MatchType`.

### `ToolTrajectoryCriterion.MatchType` (verified source)

| Mode | Description | Extra calls allowed? |
|------|-------------|---------------------|
| `EXACT` | Perfect match — no extra, no missing tool calls | No |
| `IN_ORDER` | All expected calls present and in order, but other calls may appear in between | Yes |
| `ANY_ORDER` | All expected calls present (any order), other calls may appear | Yes |

### `TrajectoryEvaluator` constructor

```python
TrajectoryEvaluator(
    threshold: Optional[float] = None,
    eval_metric: Optional[EvalMetric] = None,
)
```

Provide **either** `threshold` (simple float cutoff) **or** `eval_metric` (full `EvalMetric` including a `ToolTrajectoryCriterion`). Not both.

Score per invocation: `1.0` if the tool trajectory matches the criterion, `0.0` otherwise. Final score = mean across all invocations.

### Example 1 — EXACT trajectory evaluation

```python
import asyncio
from google.adk.evaluation import AgentEvaluator
from google.adk.evaluation.eval_case import EvalCase, Invocation
from google.adk.evaluation.eval_metrics import EvalMetric, ToolTrajectoryCriterion
from google.adk.evaluation.trajectory_evaluator import TrajectoryEvaluator
from google.adk.agents import LlmAgent

def search_products(query: str) -> dict:
    """Search the product catalogue."""
    return {"results": [{"id": "P001", "name": "Laptop Pro 14"}]}

def add_to_cart(product_id: str, quantity: int) -> dict:
    """Add a product to the shopping cart."""
    return {"cart_id": "C123", "product_id": product_id}

def checkout(cart_id: str) -> dict:
    """Complete the checkout process."""
    return {"order_id": "O789", "status": "confirmed"}

agent = LlmAgent(
    name="shopping_agent",
    model="gemini-2.5-flash",
    instruction="Help users shop. Search, add to cart, then checkout in that order.",
    tools=[search_products, add_to_cart, checkout],
)

# Define expected tool trajectory
eval_metric = EvalMetric(
    metric_name="tool_trajectory_exact",
    threshold=1.0,
    criterion=ToolTrajectoryCriterion(
        match_type=ToolTrajectoryCriterion.MatchType.EXACT,
        threshold=1.0,
    ),
)

evaluator = TrajectoryEvaluator(eval_metric=eval_metric)
print(f"Criterion type: {evaluator.criterion_type}")
```

### Example 2 — IN_ORDER matching for flexible pipelines

```python
from google.adk.evaluation.eval_metrics import EvalMetric, ToolTrajectoryCriterion
from google.adk.evaluation.trajectory_evaluator import TrajectoryEvaluator

# Allow the agent to call diagnostic tools between the required steps
eval_metric = EvalMetric(
    metric_name="tool_trajectory_in_order",
    threshold=0.8,                             # pass if ≥ 80% of invocations match
    criterion=ToolTrajectoryCriterion(
        match_type=ToolTrajectoryCriterion.MatchType.IN_ORDER,
        threshold=0.8,
    ),
)

evaluator = TrajectoryEvaluator(eval_metric=eval_metric)

# Expected: [fetch_user_data, validate_address, create_shipment]
# Actual:   [fetch_user_data, log_request, validate_address, check_inventory, create_shipment]
# → PASS under IN_ORDER because all expected tools appear in order
```

### Example 3 — ANY_ORDER for parallel search agents

```python
from google.adk.evaluation.eval_metrics import EvalMetric, ToolTrajectoryCriterion
from google.adk.evaluation.trajectory_evaluator import TrajectoryEvaluator

# A research agent that must call 3 search tools but order doesn't matter
eval_metric = EvalMetric(
    metric_name="research_coverage",
    threshold=1.0,
    criterion=ToolTrajectoryCriterion(
        match_type=ToolTrajectoryCriterion.MatchType.ANY_ORDER,
        threshold=1.0,
    ),
)

evaluator = TrajectoryEvaluator(eval_metric=eval_metric)

# Expected: [search_papers, search_patents, search_market_data]
# Actual:   [search_patents, search_market_data, search_papers, summarise]
# → PASS under ANY_ORDER because all 3 expected tools appear
```

### Match type quick reference

```python
from google.adk.evaluation.eval_metrics import ToolTrajectoryCriterion

MatchType = ToolTrajectoryCriterion.MatchType

# String coercion also works (normalises to UPPER_CASE, strips hyphens/spaces)
criterion = ToolTrajectoryCriterion(match_type="in_order")  # MatchType.IN_ORDER
```

---

## 8 · `AuthConfig` + `AuthHandler`

`AuthConfig` (`google.adk.auth.AuthConfig`) and `AuthHandler` (`google.adk.auth.auth_handler.AuthHandler`) form the two-layer auth pipeline in ADK: `AuthConfig` describes **what credentials are needed and their current state**, while `AuthHandler` orchestrates the **OAuth2 token exchange flow**.

### `AuthConfig` (verified source)

```python
class AuthConfig(BaseModelWithConfig):
    auth_scheme: AuthScheme
    raw_auth_credential: Optional[AuthCredential] = None
    exchanged_auth_credential: Optional[AuthCredential] = None
    credential_key: Optional[str] = None
```

| Field | Purpose |
|-------|---------|
| `auth_scheme` | The OpenAPI security scheme (API key, HTTP bearer, OAuth2, OIDC, service account) |
| `raw_auth_credential` | The original credential with client ID + secret (before exchange) |
| `exchanged_auth_credential` | Filled by ADK/client after exchange — contains access token, auth URI, or OAuth2 state |
| `credential_key` | Lookup key for the `CredentialService`. Auto-derived from scheme + credential hash if omitted. |

`AuthConfig.__init__` automatically resolves `credential_key` from `raw_auth_credential.model_extra` or `auth_scheme.model_extra` if not explicitly set.

### `AuthCredentialTypes` enum

```python
from google.adk.auth import AuthCredentialTypes

AuthCredentialTypes.API_KEY          # "apiKey"
AuthCredentialTypes.HTTP             # "http" (Basic, Bearer)
AuthCredentialTypes.OAUTH2           # "oauth2"
AuthCredentialTypes.OPEN_ID_CONNECT  # "openIdConnect"
AuthCredentialTypes.SERVICE_ACCOUNT  # "serviceAccount"
```

### `AuthHandler` (verified source)

```python
class AuthHandler:
    def __init__(self, auth_config: AuthConfig): ...
    async def exchange_auth_token(self) -> AuthCredential: ...
    async def parse_and_store_auth_response(self, state: State) -> None: ...
    def get_auth_response(self, state: State) -> AuthCredential: ...
    def generate_auth_request(self) -> AuthConfig: ...
```

`generate_auth_request()` prepares the `AuthConfig` with the OAuth2 authorization URI embedded in `exchanged_auth_credential`. The client (browser or your code) uses this URI to complete the user-facing OAuth2 flow.

### Example 1 — API key auth on an OpenAPI tool

```python
from google.adk.auth import AuthConfig, AuthCredential, AuthCredentialTypes
from google.adk.auth.auth_schemes import APIKey
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

api_key_scheme = APIKey(name="X-API-Key", in_="header")

api_key_credential = AuthCredential(
    auth_type=AuthCredentialTypes.API_KEY,
    api_key="sk-my-secret-api-key",
)

toolset = OpenAPIToolset(
    spec_str=open("petstore.yaml").read(),
    spec_str_type="yaml",
    auth_scheme=api_key_scheme,
    auth_credential=api_key_credential,
)
```

### Example 2 — OAuth2 client-credentials flow

```python
from google.adk.auth import AuthConfig, AuthCredential, AuthCredentialTypes
from google.adk.auth.auth_handler import AuthHandler
from fastapi.openapi.models import OAuth2, OAuthFlowClientCredentials, OAuthFlows

# 1. Describe the scheme
oauth_scheme = OAuth2(
    flows=OAuthFlows(
        clientCredentials=OAuthFlowClientCredentials(
            tokenUrl="https://auth.example.com/oauth/token",
            scopes={"api.read": "Read access", "api.write": "Write access"},
        )
    )
)

# 2. Provide the credential (client ID + secret, before exchange)
raw_credential = AuthCredential(
    auth_type=AuthCredentialTypes.OAUTH2,
    oauth2={
        "client_id": "my-client-id",
        "client_secret": "my-client-secret",
    },
)

auth_config = AuthConfig(
    auth_scheme=oauth_scheme,
    raw_auth_credential=raw_credential,
    credential_key="my-api-client-creds",
)

# 3. Exchange for an access token
import asyncio

async def get_token():
    handler = AuthHandler(auth_config)
    exchanged = await handler.exchange_auth_token()
    print(f"Access token: {exchanged.oauth2.access_token}")

asyncio.run(get_token())
```

### Example 3 — Bearer token (HTTP auth)

```python
from google.adk.auth import AuthCredential, AuthCredentialTypes
from google.adk.auth.auth_credential import HttpAuth, HttpCredentials

bearer_credential = AuthCredential(
    auth_type=AuthCredentialTypes.HTTP,
    http=HttpAuth(
        scheme="bearer",
        credentials=HttpCredentials(token="eyJhbGciOiJSUzI1NiIs..."),
    ),
)

# Use in AuthConfig for a tool that requires bearer auth
from google.adk.auth import AuthConfig
from fastapi.openapi.models import HTTPBearer

auth_config = AuthConfig(
    auth_scheme=HTTPBearer(),
    raw_auth_credential=bearer_credential,
)
```

### Example 4 — Service account credential

```python
from google.adk.auth import AuthCredential, AuthCredentialTypes
from google.adk.auth.auth_credential import ServiceAccount

sa_credential = AuthCredential(
    auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
    service_account=ServiceAccount(
        service_account_json={
            "type": "service_account",
            "project_id": "my-project",
            "private_key_id": "key-id",
            "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
            "client_email": "my-sa@my-project.iam.gserviceaccount.com",
            "token_uri": "https://oauth2.googleapis.com/token",
        },
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    ),
)
```

### Auth flow summary

```
Tool declares AuthConfig (scheme + raw_credential)
        ↓
AuthHandler.generate_auth_request()   → embeds auth_uri in exchanged_credential
        ↓
Client completes OAuth2 flow          → calls auth_response_uri
        ↓
AuthHandler.parse_and_store_auth_response(state)  → stores token in session state
        ↓
AuthHandler.exchange_auth_token()     → exchanges code for access token
        ↓
Tool's API call uses exchanged access token
```

---

## 9 · `PreloadMemoryTool`

`google.adk.tools.preload_memory_tool.PreloadMemoryTool` is an **implicit tool** — it never appears in the model's tool list and is never called by Gemini. Instead, it hooks into `process_llm_request` and automatically injects relevant **past conversation memories** as a system instruction before every LLM call.

It works by:
1. Extracting the user's latest message text.
2. Calling `tool_context.search_memory(user_query)` — a semantic search against the configured `MemoryService`.
3. Formatting matching memories as a `<PAST_CONVERSATIONS>` block and appending it to `llm_request` instructions.

If `search_memory` raises an exception or returns no results, it silently skips (no error propagation).

### Constructor

```python
PreloadMemoryTool()
```

No parameters. The tool is associated with the agent's `MemoryService` through the runner context.

### Example 1 — automatic memory injection

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.genai import types

# Session 1: initial conversation that gets saved to memory
async def session_one():
    memory_service = InMemoryMemoryService()
    session_service = InMemorySessionService()

    agent = LlmAgent(
        name="assistant",
        model="gemini-2.5-flash",
        instruction="You are a helpful assistant.",
        tools=[PreloadMemoryTool()],
    )

    runner = Runner(
        agent=agent,
        app_name="memory_demo",
        session_service=session_service,
        memory_service=memory_service,
    )

    session = await session_service.create_session(
        app_name="memory_demo", user_id="u1", session_id="s1"
    )

    async for event in runner.run_async(
        user_id="u1",
        session_id="s1",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="My name is Alice and I prefer dark mode.")],
        ),
    ):
        if event.is_final_response() and event.content:
            print(f"Session 1: {event.content.parts[0].text}")

    # Save the session to memory so it's available in future sessions
    await memory_service.add_session_to_memory(session)
    return memory_service

# Session 2: the memory is automatically injected into the next conversation
async def session_two(memory_service):
    session_service = InMemorySessionService()

    agent = LlmAgent(
        name="assistant",
        model="gemini-2.5-flash",
        instruction="You are a helpful assistant.",
        tools=[PreloadMemoryTool()],    # will inject memories from session 1
    )

    runner = Runner(
        agent=agent,
        app_name="memory_demo",
        session_service=session_service,
        memory_service=memory_service,
    )

    session = await session_service.create_session(
        app_name="memory_demo", user_id="u1", session_id="s2"
    )

    async for event in runner.run_async(
        user_id="u1",
        session_id="s2",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="What's my name and what UI preference do I have?")],
        ),
    ):
        if event.is_final_response() and event.content:
            print(f"Session 2: {event.content.parts[0].text}")
            # Output: "Session 2: Your name is Alice and you prefer dark mode."

async def main():
    mem = await session_one()
    await session_two(mem)

asyncio.run(main())
```

### Example 2 — with VertexAiMemoryBankService

```python
import os
from google.adk.agents import LlmAgent
from google.adk.memory.vertex_ai_memory_bank_service import VertexAiMemoryBankService
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

# Production: memories stored and searched in Vertex AI Memory Bank
memory_service = VertexAiMemoryBankService(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location="us-central1",
)

agent = LlmAgent(
    name="personal_assistant",
    model="gemini-2.5-flash",
    instruction=(
        "You are a personal assistant. You remember user preferences "
        "and past interactions."
    ),
    tools=[PreloadMemoryTool()],
)
# Wire memory_service into Runner(memory_service=memory_service)
```

### How the injection looks in the prompt

When memories are found, the following block is appended to the system instructions before the LLM call:

```
The following content is from your previous conversations with the user.
They may be useful for answering the user's current query.
<PAST_CONVERSATIONS>
Time: 2026-05-01T10:30:00
user: My name is Alice and I prefer dark mode.
model: Nice to meet you Alice! I'll remember your dark mode preference.
</PAST_CONVERSATIONS>
```

Each memory entry includes `author:` prefix and `Time:` timestamp when available. Text-only parts are extracted (verified from `_memory_entry_utils.extract_text`).

---

## 10 · `CodeExecutorContext`

`google.adk.code_executors.CodeExecutorContext` manages the **persistent state** of a code execution session across multiple LLM invocations within a session. It is not used directly by application code, but understanding it is essential for debugging code executor behaviour, building custom code executors, and tuning retry/error handling.

The context is stored inside `session.state[_CONTEXT_KEY]` (a key like `"_code_executor_context"`) and survives across turns via the session service. A `CodeExecutorContext` is instantiated per-invocation by the code execution flow processor.

### Constructor (verified `code_executor_context.py`)

```python
CodeExecutorContext(session_state: State)
```

Reads the existing context dict from `session_state` (or initialises it to `{}` on first use).

### Key methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_execution_id()` | `-> Optional[str]` | Returns the persistent session ID for the code interpreter (e.g. Vertex AI Extension session). |
| `set_execution_id(session_id)` | `-> None` | Stores the code interpreter session ID for reuse across turns. |
| `get_processed_file_names()` | `-> list[str]` | Returns names of files already uploaded to the executor. |
| `add_processed_file_names(file_names)` | `-> None` | Records newly uploaded file names to avoid re-uploading. |
| `get_input_files()` | `-> list[File]` | Returns input files stored in session state. |
| `add_input_files(input_files)` | `-> None` | Adds `File` objects to session state for the executor to consume. |
| `clear_input_files()` | `-> None` | Clears uploaded files and processed names. |
| `get_error_count(invocation_id)` | `-> int` | Returns the error count for a given invocation (for retry logic). |
| `increment_error_count(invocation_id)` | `-> None` | Increments the error count for the invocation. |
| `reset_error_count(invocation_id)` | `-> None` | Resets the count after a successful execution. |
| `update_code_execution_result(...)` | `-> None` | Appends a timestamped `{code, stdout, stderr}` record to the session history. |
| `get_state_delta()` | `-> dict` | Returns the diff to persist back to session state. |

### Example 1 — inspecting code executor state in a callback

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.code_executors import BuiltInCodeExecutor, CodeExecutorContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.runners import InMemoryRunner
from google.adk.agents import RunConfig

def after_agent_callback(callback_context: CallbackContext):
    """Log code executor state after each agent run."""
    state = callback_context.state
    ctx = CodeExecutorContext(state)

    exec_id = ctx.get_execution_id()
    processed = ctx.get_processed_file_names()
    input_files = ctx.get_input_files()

    print(f"Execution session ID: {exec_id}")
    print(f"Processed files: {processed}")
    print(f"Pending input files: {[f.name for f in input_files]}")

agent = LlmAgent(
    name="code_agent",
    model="gemini-2.5-flash",
    instruction="Write and run Python code to answer questions.",
    code_executor=BuiltInCodeExecutor(),
    after_agent_callback=after_agent_callback,
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="code_demo")
    await runner.session_service.create_session(
        app_name="code_demo", user_id="u1", session_id="s1"
    )
    from google.genai import types
    async for event in runner.run_async(
        user_id="u1",
        session_id="s1",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="What is 2 ** 32?")]
        ),
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Example 2 — providing input files for code execution

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.code_executors import BuiltInCodeExecutor, CodeExecutorContext
from google.adk.code_executors.code_executor_context import File
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

session_service = InMemorySessionService()

async def main():
    session = await session_service.create_session(
        app_name="files_demo", user_id="u1", session_id="s1"
    )

    # Inject a CSV file into the session so the code executor can access it
    ctx = CodeExecutorContext(session.state)
    csv_content = b"month,sales\nJan,120\nFeb,95\nMar,140\n"
    ctx.add_input_files([
        File(name="sales.csv", content=csv_content, mime_type="text/csv"),
    ])

    # Persist state delta back to session
    session.state.update(ctx.get_state_delta())

    agent = LlmAgent(
        name="csv_agent",
        model="gemini-2.5-flash",
        instruction="Analyse CSV files provided by the user.",
        code_executor=BuiltInCodeExecutor(),
    )

    runner = Runner(
        agent=agent,
        app_name="files_demo",
        session_service=session_service,
    )

    async for event in runner.run_async(
        user_id="u1",
        session_id="s1",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="Plot the monthly sales from the CSV file.")]
        ),
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Example 3 — custom executor: read error count for retry logic

```python
from google.adk.code_executors.base_code_executor import BaseCodeExecutor
from google.adk.code_executors import CodeExecutorContext
from google.adk.code_executors.code_execution_utils import CodeExecutionInput, CodeExecutionResult

class RetryingCodeExecutor(BaseCodeExecutor):
    """Custom executor that retries up to 3 times on error."""

    MAX_RETRIES = 3

    def execute_code(
        self,
        invocation_context,
        code_execution_input: CodeExecutionInput,
    ) -> CodeExecutionResult:
        ctx = CodeExecutorContext(invocation_context.session.state)
        invocation_id = invocation_context.invocation_id

        error_count = ctx.get_error_count(invocation_id)
        if error_count >= self.MAX_RETRIES:
            ctx.reset_error_count(invocation_id)
            return CodeExecutionResult(
                stdout="",
                stderr=f"Maximum retries ({self.MAX_RETRIES}) exceeded.",
                output_files=[],
            )

        result = self._try_execute(code_execution_input.code)

        if result.stderr:
            ctx.increment_error_count(invocation_id)
        else:
            ctx.reset_error_count(invocation_id)

        ctx.update_code_execution_result(
            invocation_id,
            code=code_execution_input.code,
            result_stdout=result.stdout,
            result_stderr=result.stderr,
        )
        invocation_context.session.state.update(ctx.get_state_delta())
        return result

    def _try_execute(self, code: str) -> CodeExecutionResult:
        # Real execution logic would go here
        import subprocess
        proc = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=30
        )
        return CodeExecutionResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            output_files=[],
        )
```

---

## Summary

| Class | Key use case | Import path |
|-------|-------------|-------------|
| `VertexAiSessionService` | Managed cloud session storage for production deployments | `google.adk.sessions.vertex_ai_session_service` |
| `VertexAiSearchTool` | Vertex AI Search as a Gemini model built-in retrieval tool | `google.adk.tools.vertex_ai_search_tool` |
| `VertexAiCodeExecutor` | Sandboxed code execution via Vertex AI Code Interpreter Extension | `google.adk.code_executors.vertex_ai_code_executor` |
| `APIHubToolset` | Auto-generate ADK tools from API Hub OpenAPI specs | `google.adk.tools.apihub_tool` |
| `ToolboxToolset` | Connect agents to SQL databases and APIs via MCP Toolbox | `google.adk.tools.toolbox_toolset` |
| `ConversationScenario` / `ConversationScenarios` | Multi-turn scenario-based eval with LLM-simulated users | `google.adk.evaluation.conversation_scenarios` |
| `TrajectoryEvaluator` + `ToolTrajectoryCriterion` | Evaluate tool call order (EXACT / IN_ORDER / ANY_ORDER) | `google.adk.evaluation.trajectory_evaluator` |
| `AuthConfig` + `AuthHandler` | OAuth2 / API key / bearer / service account auth pipeline | `google.adk.auth` |
| `PreloadMemoryTool` | Automatic semantic memory injection before every LLM call | `google.adk.tools.preload_memory_tool` |
| `CodeExecutorContext` | Persistent code executor state management across invocations | `google.adk.code_executors` |

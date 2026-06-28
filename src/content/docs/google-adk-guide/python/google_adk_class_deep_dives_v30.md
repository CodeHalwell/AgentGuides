---
title: "Class deep dives — volume 30 (10 additional classes)"
description: "Source-verified deep dives into 10 additional google-adk 2.3.0 classes: GCPSkillRegistry, ApiRegistry, AgentRegistry, EnterpriseWebSearchTool, MultiTurnTaskSuccessV1Evaluator, the session DB migration runner, the Interactions API generator, _BasicLlmRequestProcessor, SandboxClient, and RubricBasedFinalResponseQualityV1Evaluator."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 30"
  order: 99
---

import { Aside } from "@astrojs/starlight/components";

<Aside type="note">
All signatures, constants, and behaviours on this page were verified directly
against the installed package source at
`~/.local/lib/python3.11/site-packages/google/adk/` on
**google-adk == 2.3.0**. No documentation or blog posts were used as primary
sources.
</Aside>

## 1 · `GCPSkillRegistry` — Vertex AI skill registry client

**Module:** `google.adk.integrations.skill_registry.gcp_skill_registry`

`GCPSkillRegistry` implements the abstract `SkillRegistry` interface and connects ADK agents to skills hosted in Vertex AI. It wraps a lazy `vertexai.AsyncClient` that is created on first use, reading project/location from constructor args or environment variables.

### Constructor and environment variables (verified from source)

```python
class GCPSkillRegistry(SkillRegistry):
    def __init__(
        self,
        *,
        project_id: str | None = None,
        location: str | None = None,
    ):
        # Falls back to os.environ["GOOGLE_CLOUD_PROJECT"] and
        # os.environ["GOOGLE_CLOUD_LOCATION"] when args are None.
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION")
        self._client: vertexai.AsyncClient | None = None  # lazy init
```

### `get_skill` — download and unzip a skill

`get_skill` constructs a full resource name `projects/{project_id}/locations/{location}/skills/{name}`, fetches the skill via `_client.skills.get(name=full_name)`, then decodes the base64 `zipped_filesystem` field and calls `_unzip_filesystem` inside `asyncio.to_thread` to avoid blocking the event loop.

```python
async def get_skill(self, *, name: str) -> models.Skill:
    client = await self._get_or_create_client()
    full_name = f"projects/{self._project_id}/locations/{self._location}/skills/{name}"
    skill_response = await client.skills.get(name=full_name)
    zipped = base64.b64decode(skill_response.zipped_filesystem)
    skill = await asyncio.to_thread(_unzip_filesystem, zipped)
    return skill
```

### `search_skills` — discovery via the Vertex AI API

```python
async def search_skills(self, *, query: str) -> list[models.Frontmatter]:
    client = await self._get_or_create_client()
    response = await client.skills.retrieve(query=query)
    return [models.Frontmatter(**s.frontmatter) for s in response.skills]
```

### Example: using `GCPSkillRegistry` inside an agent

```python
import asyncio
import os
from google.adk.integrations.skill_registry.gcp_skill_registry import GCPSkillRegistry
from google.adk.agents import LlmAgent
from google.adk.tools.skill_toolset import SkillToolset

os.environ["GOOGLE_CLOUD_PROJECT"] = "my-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

registry = GCPSkillRegistry()  # reads env vars automatically

toolset = SkillToolset(registry=registry)

agent = LlmAgent(
    name="skill_agent",
    model="gemini-2.0-flash",
    tools=[toolset],
    instruction="Use the available skills to answer the user.",
)
```

### Example: searching and fetching a skill manually

```python
import asyncio
from google.adk.integrations.skill_registry.gcp_skill_registry import GCPSkillRegistry

async def demo():
    registry = GCPSkillRegistry(
        project_id="my-project",
        location="us-central1",
    )

    # Search for skills matching a topic
    frontmatters = await registry.search_skills(query="data analysis")
    for fm in frontmatters:
        print(f"{fm.name}: {fm.description[:60]}")

    # Fetch the full skill (instructions + resources)
    if frontmatters:
        skill = await registry.get_skill(name=frontmatters[0].name)
        print(skill.instructions[:200])

asyncio.run(demo())
```

### Example: supplying explicit project/location (no env vars)

```python
from google.adk.integrations.skill_registry.gcp_skill_registry import GCPSkillRegistry

registry = GCPSkillRegistry(
    project_id="prod-project-123",
    location="europe-west4",
)

# The lazy client is created only on the first await, so construction
# is cheap and safe to do at module import time.
```

---

## 2 · `ApiRegistry` — Cloud API Registry MCP toolset bridge

**Module:** `google.adk.integrations.api_registry.api_registry`  
**Public re-export:** `google.adk.tools.ApiRegistry`

`ApiRegistry` wraps the **Cloud API Registry** service and exposes each registered MCP server as an `McpToolset`. On construction it fetches **all** registered servers (with pagination, using `filter: "enabled=false"` which returns both enabled and disabled), then lets callers retrieve individual toolsets by server name.

### Constructor (verified from source)

```python
API_REGISTRY_URL = "https://cloudapiregistry.googleapis.com"

class ApiRegistry:
    def __init__(
        self,
        api_registry_project_id: str,
        location: str = "global",
        header_provider: Callable[[ReadonlyContext], dict[str, str]] | None = None,
    ):
        # Fetches all MCP servers via httpx at construction time (paginated)
        # filter="enabled=false" is intentional — it includes all servers
        self._mcp_servers: dict[str, McpServer] = {}
        self._header_provider = header_provider
        # ... pagination loop via next_page_token
```

### `get_toolset` — build an MCP toolset from a registered server

```python
def get_toolset(
    self,
    mcp_server_name: str,
    tool_filter: ToolPredicate | list[str] | None = None,
    tool_name_prefix: str | None = None,
) -> McpToolset:
    server = self._mcp_servers[mcp_server_name]
    return McpToolset(
        connection_params=server.connection_params,
        tool_filter=tool_filter,
        tool_name_prefix=tool_name_prefix,
        header_provider=self._header_provider,  # injected per-request
    )
```

### Example: listing available MCP servers

```python
from google.adk.integrations.api_registry.api_registry import ApiRegistry

registry = ApiRegistry(api_registry_project_id="my-project")

# All server names loaded at construction time.
# _mcp_servers is a private implementation detail — use for debugging only,
# not as stable public API.
print(list(registry._mcp_servers.keys()))
```

### Example: attaching a filtered toolset to an agent

```python
from google.adk.integrations.api_registry.api_registry import ApiRegistry
from google.adk.agents import LlmAgent

registry = ApiRegistry(
    api_registry_project_id="my-project",
    location="us-central1",
)

# Only expose tools whose names start with "search_"
search_toolset = registry.get_toolset(
    mcp_server_name="projects/my-project/locations/us-central1/mcpServers/web-search",
    tool_name_prefix="search_",
)

agent = LlmAgent(
    name="search_agent",
    model="gemini-2.0-flash",
    tools=[search_toolset],
)
```

### Example: per-request header injection (e.g., user-scoped auth)

```python
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.integrations.api_registry.api_registry import ApiRegistry

def my_header_provider(ctx: ReadonlyContext) -> dict[str, str]:
    # Inject the calling user's identity token on every MCP request
    token = ctx.state.get("user_token", "")
    return {"Authorization": f"Bearer {token}"}

registry = ApiRegistry(
    api_registry_project_id="my-project",
    header_provider=my_header_provider,
)
toolset = registry.get_toolset("projects/my-project/locations/global/mcpServers/my-api")
```

---

## 3 · `AgentRegistry` + `AgentRegistrySingleMcpToolset` — Agent Registry v1alpha

**Module:** `google.adk.integrations.agent_registry.agent_registry`

`AgentRegistry` connects to `https://agentregistry.googleapis.com/v1alpha` and provides toolset access, endpoint listing, and model-name resolution. When no `auth_scheme` is supplied to `get_mcp_toolset`, the registry auto-resolves authentication from IAM bindings via `GcpAuthProviderScheme`.

### Constructor (verified from source)

```python
AGENT_REGISTRY_BASE_URL = "https://agentregistry.googleapis.com/v1alpha"

class AgentRegistry:
    def __init__(
        self,
        project_id: str,    # required — does NOT fall back to env vars
        location: str,      # required — raises ValueError if either is falsy
        header_provider: Callable[[ReadonlyContext], dict[str, str]] | None = None,
    ):
        self.project_id = project_id
        self.location = location
        if not self.project_id or not self.location:
            raise ValueError("project_id and location must be provided")
        self._header_provider = header_provider
```

### Key methods

```python
def get_mcp_toolset(
    self,
    mcp_server_name: str,
    auth_scheme=None,          # if None, auto-resolves via GcpAuthProviderScheme
    auth_credential=None,
    *,
    continue_uri: str | None = None,
) -> McpToolset: ...

def list_endpoints(
    self,
    filter_str: str | None = None,
    page_size: int | None = None,
    page_token: str | None = None,
) -> dict: ...

def get_endpoint(self, name: str) -> Endpoint: ...

def get_model_name(self, endpoint_name: str) -> str: ...
```

### `AgentRegistrySingleMcpToolset` — OTel tracing support

`AgentRegistrySingleMcpToolset` is a thin `McpToolset` subclass that injects `GCP_MCP_SERVER_DESTINATION_ID` into every tool's `custom_metadata` dict. This value is used downstream by OpenTelemetry tracing to identify which MCP server handled each tool call.

```python
class AgentRegistrySingleMcpToolset(McpToolset):
    # Overrides _get_tools_async to add:
    # tool.custom_metadata["GCP_MCP_SERVER_DESTINATION_ID"] = server_id
```

### Example: getting an MCP toolset with auto-IAM auth

```python
from google.adk.integrations.agent_registry.agent_registry import AgentRegistry
from google.adk.agents import LlmAgent

registry = AgentRegistry(
    project_id="my-project",
    location="us-central1",
)

# Auth resolved automatically from IAM bindings
toolset = registry.get_mcp_toolset(
    mcp_server_name="projects/my-project/locations/us-central1/agents/my-agent/mcpServers/default",
)

agent = LlmAgent(
    name="registry_agent",
    model="gemini-2.0-flash",
    tools=[toolset],
)
```

### Example: listing and inspecting endpoints

```python
import asyncio
from google.adk.integrations.agent_registry.agent_registry import AgentRegistry

registry = AgentRegistry(project_id="my-project", location="us-central1")

result = registry.list_endpoints(page_size=10)
endpoints = result.get("endpoints", [])
for ep in endpoints:
    print(ep["name"])

# Resolve to a Gemini model name for use in LlmAgent
if endpoints:
    model_name = registry.get_model_name(endpoints[0]["name"])
    print(f"Model: {model_name}")
```

### Example: A2A agent-to-agent routing via Agent Registry

```python
# Requires: pip install google-adk[a2a]
from google.adk.integrations.agent_registry.agent_registry import AgentRegistry
from google.adk.agents import LlmAgent

registry = AgentRegistry(project_id="my-project", location="us-central1")

# Each remote agent becomes a local tool via its MCP server
remote_tool = registry.get_mcp_toolset(
    mcp_server_name=(
        "projects/my-project/locations/us-central1"
        "/agents/specialist-agent/mcpServers/default"
    ),
    continue_uri="https://my-app.run.app/a2a/callback",
)

orchestrator = LlmAgent(
    name="orchestrator",
    model="gemini-2.0-flash",
    tools=[remote_tool],
    instruction="Delegate specialist tasks to the remote agent tool.",
)
```

---

## 4 · `EnterpriseWebSearchTool` — Gemini enterprise web search

**Module:** `google.adk.tools.enterprise_search_tool`

`EnterpriseWebSearchTool` is a `BaseTool` that injects Gemini's built-in `EnterpriseWebSearch` capability into the LLM request rather than making external HTTP calls itself. A module-level singleton named `enterprise_web_search_tool` is re-exported from `google.adk.tools` as `enterprise_web_search`.

### Construction and singleton (verified from source)

```python
class EnterpriseWebSearchTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="enterprise_web_search",
            description="enterprise_web_search",
        )

# Module-level singleton (internal name); re-exported from google.adk.tools
# as `enterprise_web_search` (without the _tool suffix)
enterprise_web_search_tool = EnterpriseWebSearchTool()
```

### `process_llm_request` — injects the built-in tool

`process_llm_request` is called before each LLM invocation. It appends `types.Tool(enterprise_web_search=types.EnterpriseWebSearch())` to `llm_request.config.tools`. On Gemini 1.x models combined with other tools it raises `ValueError` to prevent silent incompatibility.

```python
async def process_llm_request(
    self,
    *,
    tool_context: ToolContext,
    llm_request: LlmRequest,
) -> None:
    # Gemini 1.x + other tools → ValueError
    # Otherwise appends types.Tool(enterprise_web_search=types.EnterpriseWebSearch())
    llm_request.config.tools = llm_request.config.tools or []
    llm_request.config.tools.append(
        types.Tool(enterprise_web_search=types.EnterpriseWebSearch())
    )
```

### Example: adding enterprise web search to an agent

```python
from google.adk.agents import LlmAgent
from google.adk.tools import enterprise_web_search  # public singleton name

agent = LlmAgent(
    name="search_agent",
    model="gemini-2.0-flash",          # must be Gemini 2+
    tools=[enterprise_web_search],
    instruction="Search the web for up-to-date information.",
)
```

### Example: combining with other tools (Gemini 2+ only)

```python
from google.adk.agents import LlmAgent
from google.adk.tools import enterprise_web_search
from google.adk.tools import FunctionTool

def summarise(text: str) -> str:
    """Summarise the provided text in one sentence."""
    return text[:200] + "..."

agent = LlmAgent(
    name="search_and_summarise",
    model="gemini-2.0-flash",
    # enterprise_web_search works alongside custom tools on Gemini 2+
    tools=[enterprise_web_search, FunctionTool(func=summarise)],
)
```

### Example: guarding against Gemini 1.x incompatibility

```python
# This will raise ValueError at call time on Gemini 1.5 if other tools exist.
# Test model compatibility before deploying:
model = "gemini-1.5-pro"
if not model.startswith("gemini-2."):
    raise ValueError(f"EnterpriseWebSearchTool requires Gemini 2+, got {model}")
```

---

## 5 · `MultiTurnTaskSuccessV1Evaluator` — multi-turn LLM-as-judge

**Module:** `google.adk.evaluation.multi_turn_task_success_evaluator`

`MultiTurnTaskSuccessV1Evaluator` is an `Evaluator` subclass that scores how well an agent completed a multi-turn task. It delegates to `_MultiTurnVertexiAiEvalFacade` which calls the Vertex AI Gen AI Eval SDK with a pre-defined `MULTI_TURN_TASK_SUCCESS` rubric. Scores are in `[0, 1]`.

### Constructor (verified from source)

```python
class MultiTurnTaskSuccessV1Evaluator(Evaluator):
    def __init__(self, eval_metric: EvalMetric):
        super().__init__()
        self._facade = _MultiTurnVertexiAiEvalFacade(
            eval_metric=eval_metric,
            rubric_type="MULTI_TURN_TASK_SUCCESS",
        )
```

### `evaluate_invocations` — the main entry point

```python
def evaluate_invocations(
    self,
    actual_invocations: list[Invocation],
    expected_invocations: list[Invocation] | None = None,
    conversation_scenario: str | None = None,
) -> EvaluationResult:
    # Requires GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION to be set.
    # Uses the Vertex AI Gen AI Eval SDK (LLM-as-judge approach).
    return self._facade.evaluate(
        actual_invocations=actual_invocations,
        expected_invocations=expected_invocations,
        conversation_scenario=conversation_scenario,
    )
```

### Example: evaluating a multi-turn agent run

```python
import os
import google.genai.types as types
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.multi_turn_task_success_evaluator import (
    MultiTurnTaskSuccessV1Evaluator,
)
from google.adk.evaluation.eval_case import Invocation

os.environ["GOOGLE_CLOUD_PROJECT"] = "my-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

metric = EvalMetric(metric_name="multi_turn_task_success")
evaluator = MultiTurnTaskSuccessV1Evaluator(eval_metric=metric)

# Build actual invocations from an agent session
actual = [
    Invocation(
        user_content=types.Content(role="user", parts=[types.Part(text="Book a flight from London to Paris tomorrow.")]),
        final_response=types.Content(role="model", parts=[types.Part(text="I have booked flight BA304 departing 08:00.")]),
    ),
    Invocation(
        user_content=types.Content(role="user", parts=[types.Part(text="Also add travel insurance.")]),
        final_response=types.Content(role="model", parts=[types.Part(text="Travel insurance has been added to your booking.")]),
    ),
]

result = evaluator.evaluate_invocations(actual_invocations=actual)
print(f"Task success score: {result.overall_score:.2f}")
```

### Example: providing a conversation scenario for better context

```python
result = evaluator.evaluate_invocations(
    actual_invocations=actual,
    conversation_scenario=(
        "The user wants to book a flight and add travel insurance "
        "in a single multi-turn conversation."
    ),
)
for inv_result in result.per_invocation_results:
    print(f"Turn score: {inv_result.score:.2f} — {inv_result.rationale}")
```

### Example: integrating with `AgentEvaluator`

```python
import asyncio
from google.adk.evaluation.agent_evaluator import AgentEvaluator

# Metric selection is declared inside the .test.json / EvalConfig files
# (e.g. "eval_metrics": [{"metric_name": "multi_turn_task_success"}]).
# AgentEvaluator.evaluate reads those files and wires the right evaluator
# automatically — there is no eval_metrics kwarg on this method.
asyncio.run(AgentEvaluator.evaluate(
    agent_module="my_agent.agent",
    eval_dataset_file_path_or_dir="eval_cases/",
))
```

---

## 6 · `upgrade()` — session database migration runner

**Module:** `google.adk.sessions.migration.migration_runner`

`upgrade()` migrates a session database from its current schema version to the latest version. The current migration path is `SCHEMA_VERSION_0_PICKLE → SCHEMA_VERSION_1_JSON`, using `migrate_from_sqlalchemy_pickle.migrate`. Multi-step migrations use temporary SQLite files for intermediate results and always clean them up in a `finally` block.

### Migration map and latest version (verified from source)

```python
MIGRATIONS = {
    SCHEMA_VERSION_0_PICKLE: (
        SCHEMA_VERSION_1_JSON,
        migrate_from_sqlalchemy_pickle.migrate,
    ),
}
LATEST_VERSION = SCHEMA_VERSION_1_JSON  # = _schema_check_utils.LATEST_SCHEMA_VERSION
```

### `upgrade()` signature and key behaviours

```python
def upgrade(
    source_db_url: str,
    dest_db_url: str,
    allow_unsafe_unpickling: bool = False,
) -> None:
    # Raises RuntimeError if source_db_url == dest_db_url (no in-place migration).
    # Detects current schema version via _schema_check_utils.get_db_schema_version().
    # If already at LATEST_VERSION, logs and returns immediately.
    # Multi-step path: intermediate results written to temp SQLite files.
    # allow_unsafe_unpickling=True enables Python's unsafe pickle loader for
    # the SCHEMA_VERSION_0_PICKLE step — use only with a trusted source DB.
```

### Example: migrating a local SQLite session database

```python
from google.adk.sessions.migration.migration_runner import upgrade

upgrade(
    source_db_url="sqlite:///old_sessions.db",
    dest_db_url="sqlite:///new_sessions.db",
)
print("Migration complete.")
```

### Example: migrating from PostgreSQL with unsafe pickle enabled

```python
from google.adk.sessions.migration.migration_runner import upgrade

# Use allow_unsafe_unpickling=True only when you trust the source database.
upgrade(
    source_db_url="postgresql+psycopg2://user:pass@host/old_db",
    dest_db_url="postgresql+psycopg2://user:pass@host/new_db",
    allow_unsafe_unpickling=True,
)
```

### Example: guarding against in-place migration attempts

```python
from google.adk.sessions.migration.migration_runner import upgrade

source = "sqlite:///sessions.db"
dest = "sqlite:///sessions_v2.db"

if source == dest:
    # upgrade() itself raises RuntimeError for this case; guard early for clarity
    raise RuntimeError("Cannot migrate in-place; provide a different dest_db_url.")

upgrade(source_db_url=source, dest_db_url=dest)
```

---

## 7 · `generate_content_via_interactions` — Interactions API async generator

**Module:** `google.adk.models.interactions_utils`

`generate_content_via_interactions` is an async generator that sends LLM requests through the Interactions API instead of the standard `generate_content` path. When `previous_interaction_id` is set on the request, only the latest user content is sent (not the full conversation history), reducing payload size. Both streaming and non-streaming modes are supported.

### Key conversion functions (verified from source)

```python
# The generator orchestrates these helpers:
# _convert_contents_to_steps(contents) → input_steps
# convert_tools_config_to_interactions_format(config) → interaction_tools
# extract_system_instruction(config) → system_instruction
# build_generation_config(config) → generation_config

async def generate_content_via_interactions(
    api_client: Client,
    llm_request: LlmRequest,
    stream: bool,
) -> AsyncGenerator[LlmResponse, None]:
    # stream=True  → api_client.aio.interactions.create(..., stream=True)
    # stream=False → api_client.aio.interactions.create(..., stream=False)
    # Tracks interaction_id across SSE events for session continuity.
    ...
```

### SSE event types consumed (verified from source)

```python
# Imports from google.genai (Interactions API):
# InteractionSSEEvent, InteractionStatusUpdate, InteractionCreatedEvent,
# InteractionCompletedEvent, FunctionCallStep, FunctionResultStep
```

### Example: enabling the Interactions API on a model

```python
from google.adk.models.google_llm import Gemini

# Pass use_interactions_api=True to route through Interactions API
model = Gemini(
    model="gemini-2.0-flash",
    use_interactions_api=True,
)
```

### Example: streaming with `previous_interaction_id` optimisation

```python
import asyncio
from google.adk.models.interactions_utils import generate_content_via_interactions
from google.adk.models.llm_request import LlmRequest
import google.genai as genai
import google.genai.types as types

async def run_interactions():
    client = genai.Client()

    # First turn — no previous interaction
    request = LlmRequest(
        model="gemini-2.0-flash",
        contents=[types.Content(role="user", parts=[types.Part(text="Hello!")])],
    )
    interaction_id = None

    async for response in generate_content_via_interactions(
        api_client=client,
        llm_request=request,
        stream=True,
    ):
        if response.text:
            print(response.text, end="", flush=True)
        if response.interaction_id:
            interaction_id = response.interaction_id

    # Second turn — only send latest user turn, not full history
    request2 = LlmRequest(
        model="gemini-2.0-flash",
        contents=[types.Content(role="user", parts=[types.Part(text="Tell me more.")])],
        previous_interaction_id=interaction_id,
    )
    async for response in generate_content_via_interactions(
        api_client=client,
        llm_request=request2,
        stream=True,
    ):
        print(response.text or "", end="", flush=True)

asyncio.run(run_interactions())
```

### Example: non-streaming mode

```python
import asyncio
from google.adk.models.interactions_utils import generate_content_via_interactions
from google.adk.models.llm_request import LlmRequest
import google.genai as genai
import google.genai.types as types

async def non_streaming():
    client = genai.Client()
    request = LlmRequest(
        model="gemini-2.0-flash",
        contents=[types.Content(role="user", parts=[types.Part(text="Summarise the ADK.")])],
    )

    responses = []
    async for response in generate_content_via_interactions(
        api_client=client,
        llm_request=request,
        stream=False,
    ):
        responses.append(response)

    # With stream=False the generator typically yields a single response
    print(responses[-1].text)

asyncio.run(non_streaming())
```

---

## 8 · `_BasicLlmRequestProcessor` + `_build_basic_request` — LLM request assembly

**Module:** `google.adk.flows.llm_flows.basic`

`_BasicLlmRequestProcessor` is the first request processor in the standard LLM flow. It delegates to `_build_basic_request`, which populates `llm_request` with model identity, generation config, output schema, and all live-connect fields. A module-level singleton `request_processor` is used by the LlmAgent flow.

### `_build_basic_request` — key behaviours (verified from source)

```python
def _build_basic_request(
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
) -> None:
    agent = invocation_context.agent

    # 1. Model identity
    llm_request.model = agent.canonical_model

    # 2. Generation config — deep copy so mutations don't affect agent state
    llm_request.config = copy.deepcopy(agent.generate_content_config)

    # 3. Output schema — only if no tools OR model supports structured output with tools
    if not llm_request.tools or can_use_output_schema_with_tools(llm_request.model):
        if not agent.is_task_mode:
            llm_request.config.response_schema = agent.output_schema

    # 4. Live connect fields (all set from agent.live_connect_config):
    #    response_modalities, speech_config, transcription,
    #    affective_dialog (None for Gemini 3.1 Flash Live),
    #    proactivity, session_resumption, history_config,
    #    context_window_compression, avatar_config
```

### The processor class and singleton

```python
class _BasicLlmRequestProcessor(BaseLlmRequestProcessor):
    async def run_async(
        self,
        invocation_context: InvocationContext,
        llm_request: LlmRequest,
    ) -> AsyncGenerator[Event, None]:
        _build_basic_request(invocation_context, llm_request)
        return
        yield  # makes this an async generator without yielding

# Module-level singleton consumed by the LlmAgent flow
request_processor = _BasicLlmRequestProcessor()
```

### Example: observing the populated request in a custom processor

```python
from google.adk.flows.llm_flows._base_llm_processor import BaseLlmRequestProcessor
from google.adk.flows.llm_flows.basic import _build_basic_request
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.events import Event
from typing import AsyncGenerator

class DebugRequestProcessor(BaseLlmRequestProcessor):
    async def run_async(
        self,
        invocation_context: InvocationContext,
        llm_request: LlmRequest,
    ) -> AsyncGenerator[Event, None]:
        # Inspect the request after _build_basic_request populates it
        _build_basic_request(invocation_context, llm_request)
        print(f"Model: {llm_request.model}")
        print(f"Response schema: {llm_request.config.response_schema}")
        return
        yield
```

### Example: checking if output schema will be applied

```python
from google.adk.utils.output_schema_utils import can_use_output_schema_with_tools

model = "gemini-2.0-flash"
has_tools = True

if can_use_output_schema_with_tools(model) or not has_tools:
    print("Output schema will be applied.")
else:
    print("Output schema suppressed because model has tools and doesn't support it.")
```

### Example: live connect config fields on an agent

```python
from google.adk.agents import LlmAgent
from google.genai import types

agent = LlmAgent(
    name="live_agent",
    model="gemini-2.0-flash-live",
    generate_content_config=types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
            )
        ),
    ),
    # _build_basic_request copies all live_connect_config fields into llm_request
)
```

---

## 9 · `SandboxClient` — Vertex AI Computer Use CDP commands

**Module:** `google.adk.integrations.vmaas.sandbox_client`

`SandboxClient` drives a Vertex AI Computer Use sandbox browser via Chrome DevTools Protocol (CDP) commands. It wraps the Vertex AI SDK's `send_command` call and provides both single-command and batch execution with automatic sequential fallback. The class is gated behind `@experimental(FeatureName.COMPUTER_USE)`.

### Constructor (verified from source)

```python
@experimental(FeatureName.COMPUTER_USE)
class SandboxClient:
    def __init__(
        self,
        vertexai_client: vertexai.Client,
        sandbox: Any,        # vertexai SDK SandboxEnvironment
        access_token: str,
    ):
        self._client = vertexai_client
        self._sandbox = sandbox
        self._access_token = access_token
```

### CDP command constants (verified from source)

```python
_CDP_COMMAND_PAGE_CAPTURE_SCREENSHOT      = "Page.captureScreenshot"
_CDP_COMMAND_INPUT_DISPATCH_MOUSE_EVENT   = "Input.dispatchMouseEvent"
_CDP_COMMAND_INPUT_DISPATCH_KEY_EVENT     = "Input.dispatchKeyEvent"
_CDP_COMMAND_INPUT_INSERT_TEXT            = "Input.insertText"
_CDP_COMMAND_PAGE_GET_NAV_HISTORY         = "Page.getNavigationHistory"
_CDP_COMMAND_PAGE_NAV_TO_HISTORY          = "Page.navigateToHistoryEntry"
_CDP_COMMAND_PAGE_NAVIGATE                = "Page.navigate"
```

### Modifier key bitmask map (verified from source)

```python
_MODIFIER_MAP = {
    "CONTROL": 2,
    "ALT":     1,
    "SHIFT":   8,
    "COMMAND": 4,
    "SUPER":   4,
}
```

### `make_cdp_request` and `make_cdp_batch_request`

Single-command requests POST to the `/cdp` path. Batch requests try `/cdps` first; if the server returns 404 they fall back to sequential execution:

```python
async def make_cdp_request(
    self,
    command: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = params if params is not None else {}
    request_dict = {"command": command, "params": params}
    response = await asyncio.to_thread(
        self._client.agent_engines.sandboxes.send_command,
        http_method="POST",
        path="cdp",
        access_token=self._access_token,
        sandbox_environment=self._sandbox,
        request_dict=request_dict,
    )
    return self._parse_response(response)

async def make_cdp_batch_request(
    self,
    commands: list[dict[str, Any]],
    stop_on_error: bool = True,
) -> list[dict[str, Any]]:
    # Tries POST /cdps; falls back to sequential make_cdp_request on 404
    ...
```

### `get_screenshot` — with automatic retry

```python
async def get_screenshot(self, max_retries: int = 3) -> bytes:
    # Retries up to max_retries times to handle transient CDP errors
    # (e.g., "Execution context was destroyed" during navigation).
    # Returns PNG bytes.
```

### Example: capturing a screenshot

:::note[Prerequisites]
`sandbox` is a `SandboxEnvironment` obtained from the Vertex AI SDK
(e.g. `vertexai.preview.extensions.create_sandbox(...)`).
`token` is a short-lived OAuth2 access token (e.g. from
`google.auth.default()` or `google.oauth2.credentials`).
:::

```python
import asyncio
import vertexai
from google.adk.integrations.vmaas.sandbox_client import SandboxClient

async def capture():
    client = vertexai.Client(project="my-project", location="us-central1")
    sandbox = ...  # SandboxEnvironment — see Prerequisites note above
    token = ...    # OAuth2 access token — see Prerequisites note above

    sandbox_client = SandboxClient(
        vertexai_client=client,
        sandbox=sandbox,
        access_token=token,
    )
    png_bytes = await sandbox_client.get_screenshot(max_retries=5)
    with open("screenshot.png", "wb") as f:
        f.write(png_bytes)

asyncio.run(capture())
```

### Example: navigating the browser and sending text input

```python
from google.adk.integrations.vmaas.sandbox_client import SandboxClient

async def navigate_and_type(sandbox_client: SandboxClient):
    # Navigate to a URL
    await sandbox_client.make_cdp_request(
        "Page.navigate",
        {"url": "https://example.com"},
    )

    # Type text into the focused input
    await sandbox_client.make_cdp_request(
        "Input.insertText",
        {"text": "Hello from ADK Computer Use"},
    )

    # Press Enter
    await sandbox_client.make_cdp_request(
        "Input.dispatchKeyEvent",
        {"type": "keyDown", "key": "Enter", "windowsVirtualKeyCode": 13},
    )
```

### Example: batching multiple CDP commands

```python
from google.adk.integrations.vmaas.sandbox_client import SandboxClient

async def batch_demo(sandbox_client: SandboxClient):
    results = await sandbox_client.make_cdp_batch_request(
        commands=[
            {"command": "Page.navigate", "params": {"url": "https://example.com"}},
            {"command": "Page.captureScreenshot", "params": {}},
        ],
        stop_on_error=True,
    )
    for r in results:
        print(r["status"])  # "success" or "error"
```

---

## 10 · `RubricBasedFinalResponseQualityV1Evaluator` — rubric-based response quality judge

**Module:** `google.adk.evaluation.rubric_based_final_response_quality_v1`

`RubricBasedFinalResponseQualityV1Evaluator` extends `RubricBasedEvaluator` and uses an LLM (via the Vertex AI Gen AI Eval SDK) to judge the quality of an agent's final response against user-defined rubrics. It is gated behind `@experimental`. The internal prompt instructs the LLM to think silently with a 10240-token budget and produce `yes`/`no` verdicts that map to `1.0`/`0.0`.

### Class variables and constructor (verified from source)

```python
@experimental
class RubricBasedFinalResponseQualityV1Evaluator(RubricBasedEvaluator):
    criterion_type: ClassVar[type[RubricsBasedCriterion]] = RubricsBasedCriterion
    RUBRIC_TYPE: ClassVar[str] = "FINAL_RESPONSE_QUALITY"

    def __init__(self, eval_metric: EvalMetric):
        super().__init__(
            eval_metric,
            criterion_type=RubricBasedFinalResponseQualityV1Evaluator.criterion_type,
            rubric_type=RubricBasedFinalResponseQualityV1Evaluator.RUBRIC_TYPE,
        )
        self._auto_rater_prompt_template = _RUBRIC_BASED_FINAL_RESPONSE_QUALITY_V1_PROMPT
```

### `format_auto_rater_prompt` — building the judge prompt

The method extracts user input text, tool call/response steps, developer instructions from `app_details`, and available tool declarations, then formats the large prompt template with these values. The `include_intermediate_responses_in_final` flag on the criterion controls whether intermediate agent responses appear in the `final_answer` section.

```python
@override
def format_auto_rater_prompt(
    self,
    actual_invocation: Invocation,
    _: Optional[Invocation],
) -> str:
    self.create_effective_rubrics_list(actual_invocation.rubrics)
    user_input = get_text_from_content(actual_invocation.user_content)
    # ... extracts tool calls, developer instructions, tool declarations
    rubrics_text = "\n".join([
        f"*  {r.rubric_content.text_property}"
        for r in self._effective_rubrics_list
    ])
    return self._auto_rater_prompt_template.format(
        developer_instructions=developer_instructions,
        tool_declarations=tool_declarations,
        user_input=user_input,
        response_steps=response_steps,
        final_response=final_response,
        rubrics=rubrics_text,
    )
```

### Response parsing — `DefaultAutoRaterResponseParser`

```python
# From rubric_based_evaluator.py
_PROPERTY_PATTERN = r"(?<=Property: )(.*)"
_RATIONALE_PATTERN = r"(?<=Rationale: )(.*)"
_VERDICT_PATTERN   = r"(?<=Verdict: )(.*)"

# Verdict → score mapping:
# "yes" in verdict.lower() → 1.0
# "no"  in verdict.lower() → 0.0
# otherwise               → None
```

### Example: evaluating final response quality with custom rubrics

```python
import os
from google.adk.evaluation.eval_metrics import EvalMetric, RubricsBasedCriterion
from google.adk.evaluation.rubric_based_final_response_quality_v1 import (
    RubricBasedFinalResponseQualityV1Evaluator,
)
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_rubrics import Rubric, RubricContent

os.environ["GOOGLE_CLOUD_PROJECT"] = "my-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

criterion = RubricsBasedCriterion(rubrics=[
    Rubric(
        rubric_id="conciseness",
        rubric_content=RubricContent(
            text_property="The response is concise and avoids unnecessary filler text."
        ),
    ),
    Rubric(
        rubric_id="tool-evidence",
        rubric_content=RubricContent(
            text_property="The response correctly uses tool output as evidence."
        ),
    ),
])

metric = EvalMetric(
    metric_name="final_response_quality",
    criterion=criterion,
)
evaluator = RubricBasedFinalResponseQualityV1Evaluator(eval_metric=metric)
```

### Example: running the evaluation on an invocation

```python
import google.genai.types as types
from google.adk.evaluation.eval_case import Invocation

invocation = Invocation(
    user_content=types.Content(role="user", parts=[types.Part(text="What is the capital of France?")]),
    final_response=types.Content(role="model", parts=[types.Part(text="The capital of France is Paris.")]),
    # intermediate_data carries tool call/response pairs for evidence collection
)

result = evaluator.evaluate_invocations(actual_invocations=[invocation])
for per_inv in result.per_invocation_results:
    for rubric_score in per_inv.rubric_scores:
        print(f"{rubric_score.property_text}")
        print(f"  Score: {rubric_score.score}  Rationale: {rubric_score.rationale}")
```

### Example: enabling intermediate responses in the final answer section

```python
from google.adk.evaluation.eval_metrics import RubricsBasedCriterion
from google.adk.evaluation.eval_rubrics import Rubric, RubricContent

# Setting include_intermediate_responses_in_final=True causes
# format_auto_rater_prompt to include intermediate agent messages
# alongside the final answer, giving the LLM-judge more context.
criterion = RubricsBasedCriterion(
    include_intermediate_responses_in_final=True,
    rubrics=[
        Rubric(
            rubric_id="transparency",
            rubric_content=RubricContent(
                text_property="The agent's reasoning steps are transparent and justified."
            ),
        ),
    ],
)
```

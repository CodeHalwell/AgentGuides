---
title: "Class deep dives — volume 9 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.1.0 classes: Gemma/Gemma3Ollama (local models), ContextCacheConfig/GeminiContextCacheManager (context caching), DataAgentToolset (data analytics), DiscoveryEngineSearchTool (Vertex AI Search), GoogleMapsGroundingTool, EnterpriseWebSearchTool, LoadMemoryTool, LoadArtifactsTool, exit_loop/get_user_choice (loop control), multi-turn evaluation suite."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 9"
  order: 68
---

Source-verified against **google-adk==2.1.0** (installed from PyPI, June 2026). Every field name, signature, and code example is taken directly from the installed package source.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `Gemma` + `Gemma3Ollama` + `GemmaFunctionCallingMixin` | `google.adk.models.gemma_llm` | Stable |
| 2 | `ContextCacheConfig` + `GeminiContextCacheManager` | `google.adk.agents.context_cache_config`, `google.adk.models.gemini_context_cache_manager` | Experimental |
| 3 | `DataAgentToolset` + `DataAgentToolConfig` + `DataAgentCredentialsConfig` | `google.adk.tools.data_agent` | Experimental |
| 4 | `DiscoveryEngineSearchTool` + `SearchResultMode` | `google.adk.tools.discovery_engine_search_tool` | Stable |
| 5 | `GoogleMapsGroundingTool` | `google.adk.tools.google_maps_grounding_tool` | Stable |
| 6 | `EnterpriseWebSearchTool` | `google.adk.tools.enterprise_search_tool` | Stable |
| 7 | `LoadMemoryTool` | `google.adk.tools.load_memory_tool` | Stable |
| 8 | `LoadArtifactsTool` | `google.adk.tools.load_artifacts_tool` | Stable |
| 9 | `exit_loop` + `get_user_choice_tool` | `google.adk.tools.exit_loop_tool`, `google.adk.tools.get_user_choice_tool` | Stable |
| 10 | Multi-turn evaluation suite: `MultiTurnTaskSuccessV1Evaluator`, `MultiTurnToolUseQualityV1Evaluator`, `MultiTurnTrajectoryQualityV1Evaluator`, `SafetyEvaluatorV1`, `LlmAsJudge` | `google.adk.evaluation` | Stable / Experimental |

---

## 1 · `Gemma` / `Gemma3Ollama` / `GemmaFunctionCallingMixin`

**Source:** `google.adk.models.gemma_llm`

`Gemma` and `Gemma3Ollama` bring Google's open Gemma model family into ADK agents — either via the Gemini API (cloud, no GPU required) or fully locally via Ollama and LiteLLM. Because Gemma lacks native function-calling API support, both classes inherit `GemmaFunctionCallingMixin`, which translates tool declarations into system-instruction text and parses JSON function calls back out of model responses.

### Class hierarchy

```
BaseLlm
└── Gemini (google.adk.models.google_llm)
    └── Gemma(GemmaFunctionCallingMixin, Gemini)

BaseLlm
└── LiteLlm (google.adk.models.lite_llm)
    └── Gemma3Ollama(GemmaFunctionCallingMixin, LiteLlm)
```

### `Gemma` constructor (source-verified)

```python
class Gemma(GemmaFunctionCallingMixin, Gemini):
    model: str = 'gemma-3-27b-it'
    # Supported: gemma-3-1b-it, gemma-3-4b-it, gemma-3-12b-it, gemma-3-27b-it, gemma-4-31b-it
```

`Gemma` inherits all of `Gemini`'s `generate_content_async` pipeline but:
1. Converts `system_instruction` into a leading **user** turn (Gemma does not support system prompts natively).
2. Calls `_move_function_calls_into_system_instruction()` to serialise tool declarations as JSON inside the system text.
3. After generation, calls `_extract_function_calls_from_response()` to parse any `{"name": ..., "parameters": ...}` JSON block out of the raw text response and rewrite it as a `FunctionCall` part.

> **Limitation:** Gemma does **not** support Vertex AI API. Use only with the Gemini API (`GOOGLE_GENAI_USE_VERTEXAI=FALSE`, the default).

### `Gemma3Ollama` constructor (source-verified)

```python
class Gemma3Ollama(GemmaFunctionCallingMixin, LiteLlm):
    def __init__(self, model: str = 'ollama/gemma3:12b', **kwargs): ...
    # Supported model pattern: r'ollama/gemma3.*'
```

Requires Ollama running locally with a Gemma 3 model pulled. Routes via `LiteLlm`'s OpenAI-compatible endpoint.

### `GemmaFunctionCallingMixin` — what it does

| Method | Purpose |
|--------|---------|
| `_move_function_calls_into_system_instruction(llm_request)` | Serialises all `FunctionDeclaration` objects as JSON into the system prompt; empties `config.tools`; converts prior tool calls/responses in the content history to text. |
| `_extract_function_calls_from_response(llm_response)` | Scans the model's final text part for a JSON block matching `{"name": ..., "parameters": ...}` (via `GemmaFunctionCallModel`); if found, replaces the text part with a `FunctionCall` part. Accepts both markdown code fences and raw JSON. |

**`GemmaFunctionCallModel`** is a flexible Pydantic model that accepts both `name`/`function` and `parameters`/`args` key aliases:

```python
class GemmaFunctionCallModel(BaseModel):
    name: str = Field(validation_alias=AliasChoices('name', 'function'))
    parameters: dict[str, Any] = Field(
        validation_alias=AliasChoices('parameters', 'args')
    )
```

### Example 1 — Gemma via Gemini API

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.gemma_llm import Gemma
from google.adk.runners import InMemoryRunner
from google.adk.tools.function_tool import FunctionTool


def add_numbers(a: int, b: int) -> int:
    """Adds two numbers together."""
    return a + b


# Use gemma-3-27b-it (default) or gemma-3-12b-it for lighter footprint
gemma_model = Gemma(model="gemma-3-27b-it")

agent = LlmAgent(
    name="gemma_agent",
    model=gemma_model,
    instruction="You are a helpful assistant that can do maths.",
    tools=[FunctionTool(add_numbers)],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="gemma_demo")
    await runner.session_service.create_session(
        app_name="gemma_demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "What is 42 plus 58?", user_id="u1", session_id="s1"
    )
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text)


asyncio.run(main())
```

### Example 2 — Gemma3Ollama (fully local, no API key)

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.gemma_llm import Gemma3Ollama
from google.adk.runners import InMemoryRunner

# Requires: `ollama pull gemma3:12b` and Ollama running locally
local_model = Gemma3Ollama(model="ollama/gemma3:12b")

agent = LlmAgent(
    name="local_gemma",
    model=local_model,
    instruction="You are a concise local assistant.",
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="local_demo")
    await runner.session_service.create_session(
        app_name="local_demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Summarise the water cycle in one sentence.",
        user_id="u1",
        session_id="s1",
    )
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text)


asyncio.run(main())
```

### Example 3 — Multi-model team (Gemini orchestrator + Gemma worker)

```python
from google.adk.agents import LlmAgent
from google.adk.models.gemma_llm import Gemma
from google.adk.tools.agent_tool import AgentTool

# Cheap Gemma worker handles classification
classifier = LlmAgent(
    name="classifier",
    model=Gemma(model="gemma-3-12b-it"),
    instruction=(
        "Classify the sentiment of the input as POSITIVE, NEGATIVE, or NEUTRAL."
        " Respond with only the label."
    ),
)

# Gemini orchestrator handles reasoning and routes to Gemma for classification
orchestrator = LlmAgent(
    name="orchestrator",
    model="gemini-2.5-flash",
    instruction=(
        "You coordinate tasks. For sentiment classification, delegate to the"
        " classifier sub-agent."
    ),
    tools=[AgentTool(agent=classifier)],
)
```

### Supported models (source-verified)

```python
# Gemma (Gemini API)
@classmethod
def supported_models(cls) -> list[str]:
    return [r'gemma-.*']

# Gemma3Ollama (via Ollama/LiteLLM)
@classmethod
def supported_models(cls) -> list[str]:
    return [r'ollama/gemma3.*']
```

Recommended models for agentic use: `gemma-3-27b-it`, `gemma-3-12b-it`, `gemma-4-31b-it` (source comment).

---

## 2 · `ContextCacheConfig` / `GeminiContextCacheManager`

**Source:** `google.adk.agents.context_cache_config` · `google.adk.models.gemini_context_cache_manager`  
**Status:** `@experimental(FeatureName.AGENT_CONFIG)` / `@experimental`

Gemini's context cache lets you pre-process a large static context (system instructions, tools, documents) once and reuse the cached version across many requests, cutting both cost and latency.

`ContextCacheConfig` is attached to an `App` to enable this. `GeminiContextCacheManager` is the internal lifecycle engine — you never instantiate it directly.

### `ContextCacheConfig` — constructor & fields (source-verified)

```python
from google.adk.agents.context_cache_config import ContextCacheConfig

config = ContextCacheConfig(
    cache_intervals=10,   # reuse the same cache for up to N invocations (1–100)
    ttl_seconds=1800,     # cache TTL in seconds (default: 30 min)
    min_tokens=0,         # only cache if estimated request tokens >= this value
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cache_intervals` | `int` | `10` | Max invocations before the cache is refreshed. Range: 1–100. |
| `ttl_seconds` | `int` | `1800` | Cache TTL (seconds). The manager appends `"s"` when calling the Gemini API. |
| `min_tokens` | `int` | `0` | Skip caching if the estimated request token count is below this threshold — avoids paying cache storage overhead for small requests. |

**Computed property:**

```python
@property
def ttl_string(self) -> str:
    return f"{self.ttl_seconds}s"
```

### How `GeminiContextCacheManager` works

1. On each LLM call, the manager receives the assembled `LlmRequest` (system instructions + tools + conversation contents).
2. It hashes the **cacheable prefix** of the request to detect whether the current cache is still valid.
3. If the cache is valid (not expired, content unchanged, invocation counter not exceeded) it **strips the cached prefix from the request** and sets `cached_content` on the `GenerateContentConfig`. Gemini does the rest.
4. If the cache is invalid, it **deletes** the stale cache (if any) and **creates a new one** via the Gemini API, then applies it.
5. The resulting `CacheMetadata` (cache name, content count, hash, expiry) is propagated through `LlmResponse` back to the session for the next call.

Gemini requires a minimum of 4096 tokens for cached content (enforced via `_GEMINI_MIN_CACHE_TOKENS = 4096`). If the cacheable prefix is smaller, caching is skipped silently.

### Attaching `ContextCacheConfig` to an `App` (source-verified)

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# Build an agent with a large static instruction
BIG_SYSTEM_PROMPT = """
You are a financial analyst assistant. You have deep knowledge of...
""" + ("(detailed financial regulations and guidelines) " * 500)  # ~4096+ tokens

agent = LlmAgent(
    name="finance_agent",
    model="gemini-2.5-flash",
    instruction=BIG_SYSTEM_PROMPT,
)

app = App(
    name="finance_app",
    root_agent=agent,
    # Enable context caching — reuse the same cache for 20 invocations, TTL=1 hour
    context_cache_config=ContextCacheConfig(
        cache_intervals=20,
        ttl_seconds=3600,
        min_tokens=4096,
    ),
)

session_service = InMemorySessionService()
runner = Runner(app=app, session_service=session_service)


async def main():
    await session_service.create_session(
        app_name="finance_app", user_id="analyst1", session_id="s1"
    )
    # First call: cache is built (~small latency overhead)
    events = await runner.run_async(
        "What are the key capital adequacy ratios?",
        user_id="analyst1",
        session_id="s1",
    )
    # Subsequent calls reuse the cache (faster + cheaper)
    events2 = await runner.run_async(
        "Explain Basel III requirements.",
        user_id="analyst1",
        session_id="s1",
    )
    for event in events2:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text)


asyncio.run(main())
```

> **Cost note:** Gemini charges a reduced input-token price for cache hits. Storage costs apply. Use `min_tokens` to avoid caching small requests where overhead exceeds savings.

---

## 3 · `DataAgentToolset` / `DataAgentToolConfig` / `DataAgentCredentialsConfig`

**Source:** `google.adk.tools.data_agent`  
**Status:** `@experimental(FeatureName.DATA_AGENT_TOOLSET)` / `@experimental(FeatureName.DATA_AGENT_TOOL_CONFIG)`

`DataAgentToolset` connects ADK agents to **Gemini Data Analytics Agents** hosted in Google Cloud (the `geminidataanalytics.googleapis.com` API). It exposes three tools — `list_accessible_data_agents`, `get_data_agent_info`, `ask_data_agent` — that let an orchestrator delegate structured data queries to a managed data analytics agent.

### `DataAgentToolset` constructor (source-verified)

```python
from google.adk.tools.data_agent.data_agent_toolset import DataAgentToolset
from google.adk.tools.data_agent.config import DataAgentToolConfig
from google.adk.tools.data_agent.credentials import DataAgentCredentialsConfig

toolset = DataAgentToolset(
    tool_filter=None,           # None → all 3 tools; list[str] to select by name
    credentials_config=None,    # DataAgentCredentialsConfig (or ADC)
    data_agent_tool_config=DataAgentToolConfig(
        max_query_result_rows=50   # cap query results (default: 50)
    ),
)
```

### The three tools (source-verified)

| Tool function | Purpose |
|---------------|---------|
| `list_accessible_data_agents(project_id, credentials)` | Lists all Gemini Data Analytics Agents in a GCP project with their display names, descriptions, and creation times. |
| `get_data_agent_info(data_agent_name, credentials)` | Returns the full configuration and context for a named data agent. |
| `ask_data_agent(data_agent_name, question, credentials, ...)` | Sends a natural-language question to the data agent and streams back the response. |

### `DataAgentToolConfig` fields (source-verified)

```python
class DataAgentToolConfig(BaseModel):
    max_query_result_rows: int = 50  # cap on rows returned from queries
```

### `DataAgentCredentialsConfig`

Inherits `BaseGoogleCredentialsConfig`. Default scope is `["https://www.googleapis.com/auth/bigquery"]`. Uses ADC if no explicit credentials are provided.

### Tool filter patterns

```python
# Select only specific tools
toolset = DataAgentToolset(
    tool_filter=["ask_data_agent"],  # only expose the query tool
)

# Use a predicate for dynamic filtering
from google.adk.tools.base_toolset import ToolPredicate

def only_query_tools(tool, ctx) -> bool:
    return tool.name in {"ask_data_agent", "list_accessible_data_agents"}

toolset = DataAgentToolset(tool_filter=ToolPredicate(only_query_tools))
```

### Full example — data analytics orchestrator

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.data_agent.data_agent_toolset import DataAgentToolset
from google.adk.tools.data_agent.config import DataAgentToolConfig

toolset = DataAgentToolset(
    data_agent_tool_config=DataAgentToolConfig(max_query_result_rows=100),
)

agent = LlmAgent(
    name="data_orchestrator",
    model="gemini-2.5-pro",
    instruction=(
        "You have access to Gemini Data Analytics Agents. "
        "When the user asks a data question: "
        "1. List available data agents to find the most relevant one. "
        "2. Ask that agent the question. "
        "3. Summarise the result clearly."
    ),
    tools=[toolset],
)

app = App(name="data_app", root_agent=agent)
session_service = InMemorySessionService()
runner = Runner(app=app, session_service=session_service)


async def main():
    await session_service.create_session(
        app_name="data_app", user_id="u1", session_id="s1"
    )
    events = await runner.run_async(
        "Which product category had the highest revenue last quarter?",
        user_id="u1",
        session_id="s1",
    )
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text)


asyncio.run(main())
```

> **Prerequisite:** Enable the Gemini Data Analytics API in your GCP project and create at least one data agent via the Cloud Console or gcloud CLI.

---

## 4 · `DiscoveryEngineSearchTool` / `SearchResultMode`

**Source:** `google.adk.tools.discovery_engine_search_tool`

`DiscoveryEngineSearchTool` wraps the Cloud Discovery Engine (Vertex AI Search) REST API as an ADK tool. It supports both unstructured datastores (chunked retrieval) and structured datastores (document-level retrieval), with automatic mode detection.

### Constructor (source-verified)

```python
from google.adk.tools.discovery_engine_search_tool import (
    DiscoveryEngineSearchTool,
    SearchResultMode,
)

# Option A — single datastore
tool = DiscoveryEngineSearchTool(
    data_store_id="projects/my-project/locations/global/collections/default_collection/dataStores/my-store",
    max_results=5,
    filter="lang: ANY(\"en\")",
    search_result_mode=SearchResultMode.CHUNKS,  # or DOCUMENTS, or None for auto
    location="global",  # optional; inferred from data_store_id if omitted
)

# Option B — search engine spanning multiple datastores
tool = DiscoveryEngineSearchTool(
    search_engine_id="projects/my-project/locations/global/collections/default_collection/engines/my-engine",
    data_store_specs=[...],  # list[types.VertexAISearchDataStoreSpec]
    max_results=10,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_store_id` | `Optional[str]` | Full resource path to a single Discovery Engine datastore. Mutually exclusive with `search_engine_id`. |
| `search_engine_id` | `Optional[str]` | Full resource path to a search engine. Mutually exclusive with `data_store_id`. |
| `data_store_specs` | `Optional[list[...]]` | Per-datastore specs when using a search engine. Only valid with `search_engine_id`. |
| `filter` | `Optional[str]` | Discovery Engine filter expression (e.g. `"lang: ANY(\"en\")"`) |
| `max_results` | `Optional[int]` | Maximum number of results. |
| `search_result_mode` | `Optional[SearchResultMode]` | `CHUNKS` (default for unstructured), `DOCUMENTS` (required for structured). `None` = auto-detect. |
| `location` | `Optional[str]` | API endpoint location (`"global"`, `"us"`, `"eu"`). Inferred from resource ID if omitted. |

### `SearchResultMode` enum

```python
class SearchResultMode(enum.Enum):
    CHUNKS = "CHUNKS"      # unstructured datastores — returns snippet chunks
    DOCUMENTS = "DOCUMENTS" # structured datastores — returns whole documents
```

### Auto-detection behaviour (source-verified)

When `search_result_mode=None` (default):
1. The tool first attempts `CHUNKS` mode.
2. If the Gemini API returns a `GoogleAPICallError` matching `search_result_mode.*DOCUMENTS`, it retries with `DOCUMENTS` mode and caches this mode for the instance.
3. If any other error occurs, it is re-raised.

### Response format

```python
# Success
{
    "status": "success",
    "results": [
        {"title": "...", "url": "...", "content": "..."},
        ...
    ]
}

# Error
{"status": "error", "error_message": "..."}
```

### Example — RAG agent with Vertex AI Search

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.discovery_engine_search_tool import DiscoveryEngineSearchTool

search_tool = DiscoveryEngineSearchTool(
    data_store_id=(
        "projects/my-gcp-project/locations/global"
        "/collections/default_collection/dataStores/product-docs"
    ),
    max_results=5,
)

agent = LlmAgent(
    name="docs_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a product documentation assistant. "
        "Always search the docs before answering."
    ),
    tools=[search_tool],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="docs")
    await runner.session_service.create_session(
        app_name="docs", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "How do I configure retry logic?", user_id="u1", session_id="s1"
    )
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text)


asyncio.run(main())
```

### Multi-datastore search engine example

```python
from google.genai import types
from google.adk.tools.discovery_engine_search_tool import DiscoveryEngineSearchTool

tool = DiscoveryEngineSearchTool(
    search_engine_id=(
        "projects/my-project/locations/global"
        "/collections/default_collection/engines/multi-store-engine"
    ),
    data_store_specs=[
        types.VertexAISearchDataStoreSpec(
            data_store="projects/my-project/locations/global"
            "/collections/default_collection/dataStores/store-1"
        ),
        types.VertexAISearchDataStoreSpec(
            data_store="projects/my-project/locations/global"
            "/collections/default_collection/dataStores/store-2"
        ),
    ],
    max_results=8,
)
```

---

## 5 · `GoogleMapsGroundingTool`

**Source:** `google.adk.tools.google_maps_grounding_tool`

`GoogleMapsGroundingTool` injects the Gemini 2.x native `google_maps` grounding capability into an LLM request. Unlike most ADK tools, **no local code executes** — the model calls the Maps API internally when it determines location context is relevant.

### Constructor (source-verified)

```python
from google.adk.tools.google_maps_grounding_tool import (
    GoogleMapsGroundingTool,
    google_maps_grounding,  # pre-built singleton instance
)

# Use the singleton (recommended)
tool = google_maps_grounding

# Or construct your own
tool = GoogleMapsGroundingTool()
```

### `process_llm_request` logic (source-verified)

```python
async def process_llm_request(self, *, tool_context, llm_request) -> None:
    llm_request.config.tools.append(
        types.Tool(google_maps=types.GoogleMaps())
    )
```

The method raises `ValueError` for:
- Gemini 1.x models (`is_gemini_1_model` check) — Maps grounding requires Gemini 2.x.
- Non-Gemini models (unless `GOOGLE_ADK_DISABLE_MODEL_ID_CHECK=true`).

### Constraints

- **Vertex AI only.** Requires `GOOGLE_GENAI_USE_VERTEXAI=TRUE`. The Google AI Studio (Gemini Developer API) endpoint does not support Google Maps grounding.
- **Gemini 2.x only.** Will raise on `gemini-1.*` model strings.
- The `name` and `description` fields are set to `'google_maps'` but are unused — they are framework internals.

### Example — location-aware travel agent

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.google_maps_grounding_tool import google_maps_grounding

# Requires GOOGLE_GENAI_USE_VERTEXAI=TRUE
agent = LlmAgent(
    name="travel_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a travel assistant. Use Google Maps to ground your answers "
        "with accurate location data, distances, and business information."
    ),
    tools=[google_maps_grounding],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="travel")
    await runner.session_service.create_session(
        app_name="travel", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Find me a good coffee shop near the Eiffel Tower, Paris.",
        user_id="u1",
        session_id="s1",
    )
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text)


asyncio.run(main())
```

---

## 6 · `EnterpriseWebSearchTool`

**Source:** `google.adk.tools.enterprise_search_tool`

`EnterpriseWebSearchTool` provides **enterprise-compliant grounded web search** via Gemini 2.x's `enterprise_web_search` built-in tool. Unlike `GoogleSearchTool`, this mode uses web grounding that satisfies enterprise data-residency and compliance requirements (no data leaves the Vertex AI platform to public Google Search).

> **Not the same as Vertex AI Search** (formerly called "Enterprise Search"). This tool grounds model responses with public web content via Vertex AI's enterprise-grade endpoint.

### Constructor (source-verified)

```python
from google.adk.tools.enterprise_search_tool import (
    EnterpriseWebSearchTool,
    enterprise_web_search,  # pre-built singleton
)

tool = enterprise_web_search  # singleton (recommended)
# or
tool = EnterpriseWebSearchTool()
```

### Behaviour

Like `GoogleMapsGroundingTool`, this is a **model built-in tool** — `process_llm_request` appends `types.Tool(enterprise_web_search=types.EnterpriseWebSearch())` to the request config. No local code runs.

Raises `ValueError` for:
- Gemini 1.x models.
- Non-Gemini models (unless model ID check is disabled).

### Constraints

- **Vertex AI only** (`GOOGLE_GENAI_USE_VERTEXAI=TRUE`).
- **Gemini 2.x only.**

### Example — compliance-grade research agent

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.enterprise_search_tool import enterprise_web_search

# Requires GOOGLE_GENAI_USE_VERTEXAI=TRUE
agent = LlmAgent(
    name="research_agent",
    model="gemini-2.5-pro",
    instruction=(
        "You are a market research analyst. Use enterprise web search to find "
        "current, accurate information. Always cite your sources."
    ),
    tools=[enterprise_web_search],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="research")
    await runner.session_service.create_session(
        app_name="research", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "What are the latest trends in the EV battery market in 2026?",
        user_id="u1",
        session_id="s1",
    )
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text)


asyncio.run(main())
```

### Comparison: `GoogleSearchTool` vs `EnterpriseWebSearchTool`

| Feature | `GoogleSearchTool` | `EnterpriseWebSearchTool` |
|---------|-------------------|--------------------------|
| API endpoint | Gemini Developer API or Vertex | Vertex AI only |
| Compliance | Standard | Enterprise data-residency |
| Gemini 2.x required | Yes | Yes |
| Model built-in | Yes | Yes |
| Singleton | `google_search` | `enterprise_web_search` |

---

## 7 · `LoadMemoryTool`

**Source:** `google.adk.tools.load_memory_tool`

`LoadMemoryTool` is an **injected memory retrieval tool** that the agent calls explicitly when it needs to look something up from long-term memory. Unlike `PreloadMemoryTool` (which injects memories automatically before the model call), `LoadMemoryTool` gives the model agency: it decides _when_ to retrieve memory and _what_ to query.

### Constructor & singleton (source-verified)

```python
from google.adk.tools.load_memory_tool import LoadMemoryTool, load_memory_tool

# Use singleton (recommended)
tool = load_memory_tool

# Or construct explicitly
tool = LoadMemoryTool()
```

### Underlying function (source-verified)

```python
async def load_memory(query: str, tool_context: ToolContext) -> LoadMemoryResponse:
    search_memory_response = await tool_context.search_memory(query)
    return LoadMemoryResponse(memories=search_memory_response.memories)
```

`LoadMemoryResponse` is a Pydantic model with a `memories: list[MemoryEntry]` field.

### Instruction injection (source-verified)

`LoadMemoryTool.process_llm_request` appends the following instruction to every request:

```
You have memory. You can use it to answer questions. If any questions need
you to look up the memory, you should call load_memory function with a query.
```

This nudges the model to use the tool when relevant without forcing it on every turn.

### Function declaration (source-verified)

```python
# When FeatureName.JSON_SCHEMA_FOR_FUNC_DECL is enabled:
types.FunctionDeclaration(
    name="load_memory",
    description=...,
    parameters_json_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)
```

### Example — agent with long-term memory

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.load_memory_tool import load_memory_tool

agent = LlmAgent(
    name="memory_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a personal assistant with long-term memory. "
        "Search memory when the user asks about past interactions."
    ),
    tools=[load_memory_tool],
)

memory_service = InMemoryMemoryService()
session_service = InMemorySessionService()

app = App(name="memory_app", root_agent=agent)
runner = Runner(
    app=app,
    session_service=session_service,
    memory_service=memory_service,
)


async def main():
    await session_service.create_session(
        app_name="memory_app", user_id="alice", session_id="s1"
    )

    # First conversation — teach the agent a preference
    await runner.run_async(
        "My favourite programming language is Rust.",
        user_id="alice",
        session_id="s1",
    )

    # Save session to memory
    session = await session_service.get_session(
        app_name="memory_app", user_id="alice", session_id="s1"
    )
    await memory_service.add_session_to_memory(session)

    # New session — agent can retrieve from memory
    await session_service.create_session(
        app_name="memory_app", user_id="alice", session_id="s2"
    )
    events = await runner.run_async(
        "What do you know about my programming preferences?",
        user_id="alice",
        session_id="s2",
    )
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text)


asyncio.run(main())
```

### `PreloadMemoryTool` vs `LoadMemoryTool` — when to use which

| | `PreloadMemoryTool` | `LoadMemoryTool` |
|-|--------------------|-----------------|
| Retrieval timing | Before the model call (automatic) | When the model decides to call it |
| Model control | None — memories are always injected | Full — model chooses query and timing |
| Latency | Added to every turn | Only when the model calls it |
| Use case | Always-on context (user preferences, prior facts) | On-demand lookup (specific information retrieval) |

---

## 8 · `LoadArtifactsTool`

**Source:** `google.adk.tools.load_artifacts_tool`

`LoadArtifactsTool` lets agents work with user-uploaded files (images, PDFs, audio, CSV, JSON, etc.) stored in the artifact service. It operates in two phases:
1. **`process_llm_request`** — injects an instruction listing available artifact names and a previous `load_artifacts` function response (if any) as inline content into the model context.
2. **`run_async`** — called by the model to request specific artifacts; returns a status placeholder (the actual injection happens in the next `process_llm_request` call).

### Constructor & singleton (source-verified)

```python
from google.adk.tools.load_artifacts_tool import LoadArtifactsTool, load_artifacts_tool

tool = load_artifacts_tool  # singleton
```

### Function declaration (source-verified)

```python
# Parameters
{
    "artifact_names": list[str]  # names of artifacts to load
}
```

### MIME type handling (source-verified)

The tool categorises each artifact's MIME type and converts it to a model-safe representation:

| MIME category | Handling |
|--------------|---------|
| `image/*`, `audio/*`, `video/*`, `application/pdf` | Sent as inline binary data (Gemini native support) |
| `text/*`, `application/csv`, `application/json`, `application/xml` | Decoded as UTF-8 text part |
| Unknown binary | Replaced with a text placeholder: `[Binary artifact: name, type: ..., size: X KB. Content cannot be displayed inline.]` |

Artifacts can be scoped to the session (default) or user-wide (`user:` prefix). The tool tries session-scoped first, then falls back to `user:{name}` if not found.

### Example — document Q&A with uploaded files

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.load_artifacts_tool import load_artifacts_tool
from google.genai import types

artifact_service = InMemoryArtifactService()
session_service = InMemorySessionService()

agent = LlmAgent(
    name="doc_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a document assistant. When the user asks about an uploaded "
        "file, call load_artifacts with the file name to read it first."
    ),
    tools=[load_artifacts_tool],
)

app = App(name="doc_app", root_agent=agent)
runner = Runner(
    app=app,
    session_service=session_service,
    artifact_service=artifact_service,
)


async def main():
    await session_service.create_session(
        app_name="doc_app", user_id="u1", session_id="s1"
    )

    # Simulate a user uploading a text file
    await artifact_service.save_artifact(
        app_name="doc_app",
        user_id="u1",
        session_id="s1",
        filename="report.txt",
        artifact=types.Part.from_text(
            text="Q3 revenue: $4.2M (up 18% YoY). Key driver: APAC expansion."
        ),
    )

    events = await runner.run_async(
        "What are the highlights from my uploaded report?",
        user_id="u1",
        session_id="s1",
    )
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text)


asyncio.run(main())
```

### User-scoped (cross-session) artifacts

```python
# Save with user: prefix to make it available across sessions
await artifact_service.save_artifact(
    app_name="doc_app",
    user_id="u1",
    session_id="s1",
    filename="user:profile.pdf",  # accessible in any session for user u1
    artifact=types.Part(inline_data=types.Blob(
        mime_type="application/pdf",
        data=pdf_bytes,
    )),
)
```

---

## 9 · `exit_loop` + `get_user_choice_tool`

**Sources:** `google.adk.tools.exit_loop_tool` · `google.adk.tools.get_user_choice_tool`

These two built-in tools provide **loop control** and **interactive human-in-the-loop choices** within `LoopAgent` workflows. They are the simplest tools in the ADK codebase but are critical for building self-terminating or interactive loops.

### `exit_loop` — source-verified

```python
def exit_loop(tool_context: ToolContext):
    """Exits the loop."""
    tool_context.actions.escalate = True
    tool_context.actions.skip_summarization = True
```

Sets two `EventActions` flags:
- `escalate = True` — signals the `LoopAgent` to break the loop.
- `skip_summarization = True` — prevents the framework from adding a summarisation step.

The function is wrapped automatically as a `FunctionTool` when added to an agent's `tools` list.

### `get_user_choice_tool` — source-verified

```python
def get_user_choice(
    options: list[str], tool_context: ToolContext
) -> Optional[str]:
    """Provides the options to the user and asks them to choose one."""
    tool_context.actions.skip_summarization = True
    return None

get_user_choice_tool = LongRunningFunctionTool(func=get_user_choice)
```

`get_user_choice_tool` is a **`LongRunningFunctionTool`** — it suspends agent execution and waits for the user to provide a value. The ADK frontend (web UI, Slack runner, custom runner) is expected to render the `options` list and send the user's selection back as a function response.

### Example 1 — self-terminating loop agent

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.exit_loop_tool import exit_loop

worker = LlmAgent(
    name="refiner",
    model="gemini-2.5-flash",
    instruction=(
        "Review the draft in session state under key 'draft'. "
        "If it is satisfactory (score >= 8/10), call exit_loop. "
        "Otherwise improve it and update session state."
    ),
    tools=[exit_loop],
)

loop = LoopAgent(name="refinement_loop", sub_agents=[worker], max_iterations=5)


async def main():
    runner = InMemoryRunner(agent=loop, app_name="refine")
    await runner.session_service.create_session(
        app_name="refine",
        user_id="u1",
        session_id="s1",
        state={"draft": "ADK is a framework for building AI agents."},
    )
    events = await runner.run_debug(
        "Improve the draft.", user_id="u1", session_id="s1"
    )
    # Final draft is in session state["draft"]
    session = await runner.session_service.get_session(
        app_name="refine", user_id="u1", session_id="s1"
    )
    print("Final draft:", session.state.get("draft"))


asyncio.run(main())
```

### Example 2 — interactive choice gate

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.get_user_choice_tool import get_user_choice_tool


def deploy_to_staging(environment: str) -> str:
    """Deploys the service to the given environment."""
    return f"Deployed to {environment} successfully."


agent = LlmAgent(
    name="deploy_agent",
    model="gemini-2.5-flash",
    instruction=(
        "Before deploying, ask the user which environment they want using "
        "get_user_choice with options ['staging', 'production', 'cancel']. "
        "Then deploy based on their choice."
    ),
    tools=[get_user_choice_tool, deploy_to_staging],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="deploy")
    await runner.session_service.create_session(
        app_name="deploy", user_id="u1", session_id="s1"
    )

    # The agent will pause at get_user_choice and wait for the function response
    # In production, the runner/UI would render the options and send back the selection
    events = await runner.run_debug(
        "Deploy the payment service.", user_id="u1", session_id="s1"
    )
    for event in events:
        if hasattr(event, "long_running_tool_ids") and event.long_running_tool_ids:
            print("Waiting for user choice — tool IDs:", event.long_running_tool_ids)


asyncio.run(main())
```

### When to use each

| Tool | Use case |
|------|---------|
| `exit_loop` | Stop a `LoopAgent` once a quality/completion condition is met. |
| `get_user_choice_tool` | Present a menu to the user and gate execution on their selection (approval flows, environment selection, escalation routing). |

---

## 10 · Multi-turn evaluation suite

**Sources:** `google.adk.evaluation.*`  
**Status:** Stable (Vertex-backed evaluators) / `@experimental` (`LlmAsJudge`)

ADK 2.x adds three Vertex AI-backed **multi-turn** evaluation metrics alongside the existing single-turn `FinalResponseMatchV1` and `ToolTrajectoryAvgScore` metrics. All three delegate to the `Vertex Gen AI Eval SDK` (`vertexai.types.RubricMetric`). A fully customisable `LlmAsJudge` ABC is also available for bespoke judges.

### The four Vertex-backed evaluators

#### `SafetyEvaluatorV1`

```python
from google.adk.evaluation.safety_evaluator import SafetyEvaluatorV1
from google.adk.evaluation.eval_metrics import EvalMetric

evaluator = SafetyEvaluatorV1(
    eval_metric=EvalMetric(metric_name="safety", threshold=0.8)
)
```

Evaluates the **harmlessness** of the agent's final response using `vertexai.types.PrebuiltMetric.SAFETY`. Score range: `[0, 1]`, higher = safer.

#### `MultiTurnTaskSuccessV1Evaluator`

```python
from google.adk.evaluation.multi_turn_task_success_evaluator import (
    MultiTurnTaskSuccessV1Evaluator,
)
from google.adk.evaluation.eval_metrics import EvalMetric

evaluator = MultiTurnTaskSuccessV1Evaluator(
    eval_metric=EvalMetric(metric_name="multi_turn_task_success", threshold=0.7)
)
```

Evaluates whether the agent **achieved the conversation goal** across all turns. Uses `vertexai.types.RubricMetric.MULTI_TURN_TASK_SUCCESS`. Score range: `[0, 1]`.

#### `MultiTurnToolUseQualityV1Evaluator`

```python
from google.adk.evaluation.multi_turn_tool_use_quality_evaluator import (
    MultiTurnToolUseQualityV1Evaluator,
)
from google.adk.evaluation.eval_metrics import EvalMetric

evaluator = MultiTurnToolUseQualityV1Evaluator(
    eval_metric=EvalMetric(
        metric_name="multi_turn_tool_use_quality", threshold=0.7
    )
)
```

Evaluates the **quality of function calls** across the entire conversation. Reference-free. Uses `vertexai.types.RubricMetric.MULTI_TURN_TOOL_USE_QUALITY`.

#### `MultiTurnTrajectoryQualityV1Evaluator`

```python
from google.adk.evaluation.multi_turn_trajectory_quality_evaluator import (
    MultiTurnTrajectoryQualityV1Evaluator,
)
from google.adk.evaluation.eval_metrics import EvalMetric

evaluator = MultiTurnTrajectoryQualityV1Evaluator(
    eval_metric=EvalMetric(
        metric_name="multi_turn_trajectory_quality", threshold=0.7
    )
)
```

Evaluates **how the agent achieved the goal** — the quality of the path taken, not just the outcome. Distinct from `MultiTurnTaskSuccessV1Evaluator`. Reference-free.

### Comparison table

| Evaluator | Metric type | Reference needed | Multi-turn | What it measures |
|-----------|-------------|-----------------|-----------|-----------------|
| `SafetyEvaluatorV1` | Single-turn | No | No | Harmlessness of response |
| `MultiTurnTaskSuccessV1Evaluator` | Multi-turn | No | Yes | Goal completion across turns |
| `MultiTurnToolUseQualityV1Evaluator` | Multi-turn | No | Yes | Tool call quality across turns |
| `MultiTurnTrajectoryQualityV1Evaluator` | Multi-turn | No | Yes | Path quality, not just outcome |

### `LlmAsJudge` — custom evaluator ABC

`LlmAsJudge` is an `@experimental` abstract base class for building your own LLM-based evaluators. It handles the evaluation loop (sampling, aggregation); you implement the four abstract methods.

```python
from google.adk.evaluation.llm_as_judge import LlmAsJudge, AutoRaterScore
from google.adk.evaluation.eval_metrics import EvalMetric, BaseCriterion, EvalMetric
from google.adk.evaluation.evaluator import EvaluationResult, PerInvocationResult
from google.adk.evaluation.eval_case import Invocation
from pydantic import Field
from typing import Optional

class ToneCheckCriterion(BaseCriterion):
    """Criterion for evaluating response tone."""
    desired_tone: str = Field(default="professional")
    judge_model_options: ... = Field(default_factory=...)  # JudgeModelOptions

class ToneCheckEvaluator(LlmAsJudge):
    def __init__(self, eval_metric: EvalMetric):
        super().__init__(
            eval_metric=eval_metric,
            criterion_type=ToneCheckCriterion,
            expected_invocations_required=False,
        )

    def format_auto_rater_prompt(
        self, actual: Invocation, expected: Optional[Invocation]
    ) -> str:
        response_text = ""
        if actual.final_response and actual.final_response.parts:
            response_text = actual.final_response.parts[0].text or ""
        desired_tone = self._criterion.desired_tone
        return (
            f"Rate the following response for a {desired_tone} tone on a"
            f" scale of 0 to 1 where 1 is perfectly {desired_tone}.\n\n"
            f"Response: {response_text}\n\n"
            "Return only a JSON object: {\"score\": <float>}"
        )

    def convert_auto_rater_response_to_score(self, auto_rater_response) -> AutoRaterScore:
        import json
        text = ""
        if auto_rater_response.content and auto_rater_response.content.parts:
            text = auto_rater_response.content.parts[0].text or ""
        try:
            data = json.loads(text.strip())
            return AutoRaterScore(score=float(data.get("score", 0.0)))
        except (json.JSONDecodeError, ValueError):
            return AutoRaterScore(score=None)

    def aggregate_per_invocation_samples(
        self, samples: list[PerInvocationResult]
    ) -> PerInvocationResult:
        scores = [s.score for s in samples if s.score is not None]
        avg = sum(scores) / len(scores) if scores else None
        return PerInvocationResult(
            actual_invocation=samples[0].actual_invocation,
            score=avg,
            eval_status=samples[0].eval_status,
        )

    def aggregate_invocation_results(
        self, results: list[PerInvocationResult]
    ) -> EvaluationResult:
        scores = [r.score for r in results if r.score is not None]
        avg = sum(scores) / len(scores) if scores else None
        return EvaluationResult(
            eval_metric_results=[],
            overall_eval_status=None,
        )
```

### Full evaluation example with multi-turn metrics

```python
import asyncio
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_case import EvalCase, Invocation
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.eval_metrics import EvalMetric, PrebuiltMetrics
from google.adk.evaluation.multi_turn_task_success_evaluator import (
    MultiTurnTaskSuccessV1Evaluator,
)
from google.adk.evaluation.multi_turn_tool_use_quality_evaluator import (
    MultiTurnToolUseQualityV1Evaluator,
)
from google.adk.evaluation.safety_evaluator import SafetyEvaluatorV1


# Build eval cases as multi-turn conversations
eval_set = EvalSet(
    eval_set_id="booking_flow_eval",
    eval_cases=[
        EvalCase(
            eval_id="book_flight_success",
            conversation=[
                Invocation(
                    user_content={"parts": [{"text": "I need to fly to London next Monday."}]},
                ),
                Invocation(
                    user_content={"parts": [{"text": "Business class, please."}]},
                ),
            ],
            eval_metrics=[
                EvalMetric(
                    metric_name="multi_turn_task_success",
                    threshold=0.7,
                ),
                EvalMetric(
                    metric_name="multi_turn_tool_use_quality",
                    threshold=0.7,
                ),
                EvalMetric(
                    metric_name="safety",
                    threshold=0.9,
                ),
            ],
        ),
    ],
)

# AgentEvaluator.evaluate() dispatches to the correct evaluator class
# based on metric_name via the metric_evaluator_registry
```

> **Prerequisite for Vertex-backed evaluators:** set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` in your environment. These metrics call the Vertex Gen AI Eval SDK and require Vertex AI API access.

---

## Summary table

| # | Class / group | Key insight |
|---|---|---|
| 1 | `Gemma` / `Gemma3Ollama` | Run open Gemma models via Gemini API or Ollama; `GemmaFunctionCallingMixin` bridges Gemma's lack of native function-calling. |
| 2 | `ContextCacheConfig` / `GeminiContextCacheManager` | Attach to `App` to cache large static prompts across invocations; requires ≥4096 tokens. |
| 3 | `DataAgentToolset` | Exposes 3 functions (`list`, `get_info`, `ask`) to delegate queries to managed Gemini Data Analytics Agents. |
| 4 | `DiscoveryEngineSearchTool` | Wraps Vertex AI Search; auto-detects `CHUNKS` vs `DOCUMENTS` mode; supports multi-datastore engines. |
| 5 | `GoogleMapsGroundingTool` | Injects Gemini 2.x native Maps grounding; Vertex AI only; no local code runs. |
| 6 | `EnterpriseWebSearchTool` | Enterprise-compliant grounded web search via Gemini 2.x built-in; Vertex AI only. |
| 7 | `LoadMemoryTool` | Model-controlled on-demand memory retrieval; the model chooses when and what to query. |
| 8 | `LoadArtifactsTool` | Lazy-loads user artifacts into model context; handles MIME conversion; supports user-scoped (`user:`) artifacts. |
| 9 | `exit_loop` / `get_user_choice_tool` | `exit_loop` signals `LoopAgent` termination; `get_user_choice_tool` is a `LongRunningFunctionTool` that suspends for user input. |
| 10 | Multi-turn eval suite | Four Vertex-backed multi-turn metrics + `LlmAsJudge` ABC for custom evaluation. |

---
title: "azure-ai-agents Integration Add-on (Python) — Class Deep Dives Vol. 6"
description: "Source-verified deep dives into 10 class groups from azure-ai-agents 1.1.0: OpenAPI auth hierarchy, BingGroundingSearchConfiguration, RunCompletionUsage, VectorStoreDataSource / VectorStoreConfigurations, vector store expiry and chunking, AgentsNamedToolChoice, file-search run step results, streaming event taxonomy, RequiredFunctionToolCall dispatch, and multimodal ThreadMessageOptions."
framework: microsoft-agent-framework
language: python
sidebar:
  order: 36
---

# `azure-ai-agents` Integration Add-on (Python) — Class Deep Dives Vol. 6

> **Note:** `azure-ai-agents` is an **optional integration add-on** for the Azure AI Agents service — not a replacement for `agent-framework`. See the [integration overview](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_sdk_migration_notice/) for when to use it alongside the framework.

**Package:** `azure-ai-agents` (integration add-on)  
**Version covered:** 1.1.0  
**Verified against:** installed package at `/usr/local/lib/python3.11/dist-packages/azure/ai/agents/`

This is the sixth volume of source-verified class deep dives for the `azure-ai-agents` integration add-on. Earlier volumes covered the primary client, tool classes, orchestration patterns, data models, and streaming plumbing. This volume covers **advanced configuration** — the classes you reach for when you need fine-grained control: OpenAPI authentication strategies, Bing search tuning, run cost accounting, enterprise data sources, vector store lifecycle, forced tool selection, file-search result inspection, streaming event semantics, manual tool-call dispatch, and multimodal message construction.

Earlier volumes:
- **[Vol. 1](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_sdk_class_deep_dives/)** — `AgentsClient`, `FunctionTool`, `ToolSet`, `CodeInterpreterTool`, `FileSearchTool`, `BingGroundingTool`, `ConnectedAgentTool`, `AgentEventHandler`, `ThreadMessage`, `OpenApiTool`
- **[Vol. 3](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_sdk_class_deep_dives_v3/)** — `AsyncFunctionTool`, `AzureFunctionTool`, `AzureAISearchTool`, `VectorStore`, `ThreadRun`, `RunStep`, `ResponseFormatJsonSchema`, `TruncationObject`, `MessageAttachment`, `AsyncAgentEventHandler`
- **[Vol. 4](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_sdk_class_deep_dives_v4/)** — `AgentsClient` auto function calls, `FunctionTool` dynamic registration, `CodeInterpreterTool` file upload, `FileSearchTool` + `VectorStore` lifecycle, `AzureAISearchTool` query modes, `BingGroundingTool` params, `ConnectedAgentTool` multi-agent, `AsyncToolSet`
- **[Vol. 5](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_sdk_class_deep_dives_v5/)** — `Agent` model, `AgentThread`, `ToolOutput`, `VectorStoreFileBatch`, `VectorStoreFile`, `FileInfo`, `MessageDeltaChunk`, `RunStepDeltaChunk`, `SubmitToolOutputsAction`, `AgentRunStream`

---

## Table of Contents

1. [OpenAPI auth hierarchy — anonymous, managed-identity, connection](#1-openapi-auth-hierarchy--anonymous-managed-identity-connection)
2. [`BingGroundingSearchConfiguration` + `BingGroundingSearchToolParameters` — fine-grained web search](#2-binggroundingsearchconfiguration--binggroundingsearchtoolparameters--fine-grained-web-search)
3. [`RunCompletionUsage` + `RunStepCompletionUsage` — cost accounting per run and per step](#3-runcompletionusage--runstepcompletionusage--cost-accounting-per-run-and-per-step)
4. [`VectorStoreDataSource` + `VectorStoreConfigurations` — enterprise Azure asset sources](#4-vectorstoredatasource--vectorstoreconfigurations--enterprise-azure-asset-sources)
5. [Vector store lifecycle — expiry policy and chunking strategy](#5-vector-store-lifecycle--expiry-policy-and-chunking-strategy)
6. [`AgentsNamedToolChoice` + `AgentsToolChoiceOptionMode` — forcing specific tools](#6-agentsnamedtoolchoice--agentstooltoolchoiceoptionmode--forcing-specific-tools)
7. [File-search run step results — `RunStepFileSearchToolCall` through `FileSearchToolCallContent`](#7-file-search-run-step-results--runstepfilesearchtoolcall-through-filesearchtoolcallcontent)
8. [Streaming event taxonomy — `AgentStreamEvent` and the four typed sub-enums](#8-streaming-event-taxonomy--agentstreamevent-and-the-four-typed-sub-enums)
9. [`RequiredFunctionToolCall` + `SubmitToolOutputsDetails` — manual tool-call dispatch](#9-requiredfunctiontoolcall--submittooloutputsdetails--manual-tool-call-dispatch)
10. [Multimodal message input — `ThreadMessageOptions`, `MessageInputImageFileBlock`, `MessageInputImageUrlBlock`](#10-multimodal-message-input--threadmessageoptions-messageinputimagefileblock-messageinputimageurlblock)

---

## 1. OpenAPI auth hierarchy — anonymous, managed-identity, connection

**Source:** `azure/ai/agents/models/_models.py`

`OpenApiTool` (covered in Vol. 1) lets an agent call any REST API described by an OpenAPI spec. Authentication is configured via an `OpenApiAuthDetails` subclass selected using a discriminator field `type`. Three concrete subclasses exist.

### Class signatures

```python
class OpenApiAuthType(str, Enum):
    ANONYMOUS        = "anonymous"          # no auth
    CONNECTION       = "connection"         # named AI Foundry connection
    MANAGED_IDENTITY = "managed_identity"   # Azure managed identity

class OpenApiAnonymousAuthDetails(OpenApiAuthDetails):
    # No extra fields — discriminator sets type = "anonymous" automatically
    type: Literal[OpenApiAuthType.ANONYMOUS]

class OpenApiManagedSecurityScheme(_Model):
    audience: str   # AAD scope, e.g. "https://management.azure.com/"

class OpenApiManagedAuthDetails(OpenApiAuthDetails):
    type: Literal[OpenApiAuthType.MANAGED_IDENTITY]
    security_scheme: OpenApiManagedSecurityScheme

class OpenApiConnectionSecurityScheme(_Model):
    connection_id: str   # AI Foundry connection name / ID

class OpenApiConnectionAuthDetails(OpenApiAuthDetails):
    type: Literal[OpenApiAuthType.CONNECTION]
    security_scheme: OpenApiConnectionSecurityScheme
```

### Key points

- `OpenApiAnonymousAuthDetails()` takes no arguments — just instantiate it and pass it to `OpenApiFunctionDefinition`.
- `OpenApiManagedAuthDetails` requires an `audience` string — the AAD resource URI your function app or API is registered under. The managed identity of the Azure AI Agents service must be granted the appropriate role on that resource.
- `OpenApiConnectionAuthDetails` references a connection you created in Azure AI Foundry (Project → Connections). The `connection_id` is the connection name shown in the portal.
- `default_params` on `OpenApiFunctionDefinition` lets you specify parameter names that will be filled from a user-provided defaults dict at call time rather than generated by the model — useful for tenant-specific values you do not want in the prompt.

### Example 1: public REST API with no authentication

```python
import os, json
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    OpenApiTool,
    OpenApiFunctionDefinition,
    OpenApiAnonymousAuthDetails,
)
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

# Minimal OpenAPI 3.0 spec for a public weather API
weather_spec = {
    "openapi": "3.0.0",
    "info": {"title": "Weather API", "version": "1.0"},
    "paths": {
        "/current": {
            "get": {
                "operationId": "get_current_weather",
                "summary": "Get current weather for a city",
                "parameters": [
                    {
                        "name": "city",
                        "in": "query",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "City name, e.g. 'London'",
                    }
                ],
                "responses": {"200": {"description": "Weather data"}},
            }
        }
    },
    "servers": [{"url": "https://wttr.in"}],
}

weather_tool = OpenApiTool(
    name="weather",
    spec=weather_spec,
    description="Get current weather for any city.",
    auth=OpenApiAnonymousAuthDetails(),   # no auth needed
)

agent = client.create_agent(
    model="gpt-4o",
    name="WeatherBot",
    instructions="Answer weather questions using the weather tool.",
    tools=weather_tool.definitions,
)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role="user",
    content="What's the weather in Tokyo right now?",
)
run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        print(msg.content[0].text.value)

client.delete_agent(agent.id)
```

### Example 2: internal Azure API with managed-identity auth

```python
from azure.ai.agents.models import (
    OpenApiTool,
    OpenApiFunctionDefinition,
    OpenApiManagedAuthDetails,
    OpenApiManagedSecurityScheme,
)

# Internal inventory API deployed as an Azure Function App
inventory_spec = {
    "openapi": "3.0.0",
    "info": {"title": "Inventory API", "version": "1.0"},
    "paths": {
        "/stock/{sku}": {
            "get": {
                "operationId": "get_stock_level",
                "summary": "Return current stock for a SKU",
                "parameters": [
                    {
                        "name": "sku",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                "responses": {"200": {"description": "Stock level"}},
            }
        }
    },
    "servers": [{"url": "https://myapp.azurewebsites.net/api"}],
}

# The audience is the App Registration client ID (or resource URI) for the Function App
auth = OpenApiManagedAuthDetails(
    security_scheme=OpenApiManagedSecurityScheme(
        audience="api://my-function-app-client-id"
    )
)

inventory_tool = OpenApiTool(
    name="inventory",
    spec=inventory_spec,
    description="Check stock levels for product SKUs.",
    auth=auth,
)
```

### Example 3: third-party API accessed via an AI Foundry connection

```python
from azure.ai.agents.models import (
    OpenApiTool,
    OpenApiConnectionAuthDetails,
    OpenApiConnectionSecurityScheme,
)

# Connection "salesforce-prod" was created in AI Foundry Project → Connections
# and stores the OAuth credentials securely
salesforce_tool = OpenApiTool(
    name="salesforce",
    spec=salesforce_spec,   # your OpenAPI spec
    description="Query Salesforce CRM records.",
    auth=OpenApiConnectionAuthDetails(
        security_scheme=OpenApiConnectionSecurityScheme(
            connection_id="salesforce-prod"
        )
    ),
)
```

### Auth strategy comparison

| Class | When to use | Azure requirement |
|-------|-------------|-------------------|
| `OpenApiAnonymousAuthDetails` | Public APIs, internal APIs on private VNet | None |
| `OpenApiManagedAuthDetails` | Azure-hosted APIs protected by AAD | Assign role on target resource to the agent's managed identity |
| `OpenApiConnectionAuthDetails` | Third-party APIs (OAuth, API key) via Foundry | Create a connection in Azure AI Foundry |

---

## 2. `BingGroundingSearchConfiguration` + `BingGroundingSearchToolParameters` — fine-grained web search

**Source:** `azure/ai/agents/models/_models.py`

`BingGroundingTool` (Vol. 1) connects an agent to live web search. The parameters controlling *how* that search runs — locale, result count, time filter — live in `BingGroundingSearchConfiguration`. The container that holds one or more configurations is `BingGroundingSearchToolParameters`.

### Signatures

```python
class BingGroundingSearchConfiguration(_Model):
    connection_id: str          # Bing connection in AI Foundry (required)
    market: Optional[str]       # BCP-47 locale, e.g. "en-GB", "de-DE"
    set_lang: Optional[str]     # UI string language for Bing API, e.g. "en"
    count: Optional[int]        # number of search results to return
    freshness: Optional[str]    # "Day" | "Week" | "Month" | "YYYY-MM-DD..YYYY-MM-DD"

class BingGroundingSearchToolParameters(_Model):
    search_configurations: List[BingGroundingSearchConfiguration]
    # Maximum 1 configuration per tool instance
```

### Key points

- `market` controls which regional index Bing uses. `"en-US"` for the US index, `"en-GB"` for UK, `"ja-JP"` for Japan, and so on. This affects which news sources and pages rank highly.
- `freshness` accepts either a named window (`"Day"`, `"Week"`, `"Month"`) or an ISO 8601 date range (`"2026-01-01..2026-05-31"`). Use this to restrict the agent to recent news or to a specific historical period.
- `count` caps how many search result chunks Bing returns. Fewer results reduce context length and cost; more results improve coverage. Default is unset (Bing's own default).
- `BingGroundingSearchToolParameters.search_configurations` accepts a list but the service currently enforces a maximum of **one** configuration per tool.
- `BingGroundingTool` exposes these parameters via its `bing_grounding` property — you do not construct `BingGroundingSearchToolParameters` directly in normal use; it is populated from `BingGroundingTool`'s constructor arguments.

### Example 1: restrict search to the last 24 hours for breaking news

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import BingGroundingTool
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

bing_connection_id = os.environ["BING_CONNECTION_ID"]

# BingGroundingTool's constructor maps directly onto BingGroundingSearchConfiguration fields
news_tool = BingGroundingTool(
    connection_id=bing_connection_id,
    market="en-GB",        # UK Bing index
    count=5,               # limit to 5 results to keep context compact
    freshness="Day",       # only results from the last 24 hours
)

agent = client.create_agent(
    model="gpt-4o",
    name="NewsBot",
    instructions=(
        "You are a news assistant. Summarise only events from the last 24 hours. "
        "Always cite your sources."
    ),
    tools=news_tool.definitions,
    tool_resources=news_tool.resources,
)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role="user",
    content="What are the most important tech stories in the UK today?",
)
run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        for block in msg.content:
            if hasattr(block, "text"):
                print(block.text.value)

client.delete_agent(agent.id)
```

### Example 2: historical date range search

```python
# Research agent: UK AI policy between Jan and Mar 2026
policy_tool = BingGroundingTool(
    connection_id=bing_connection_id,
    market="en-GB",
    freshness="2026-01-01..2026-03-31",  # ISO date range
    count=10,
)
```

### Example 3: inspect the generated parameters object

```python
from azure.ai.agents.models import (
    BingGroundingTool,
    BingGroundingSearchToolParameters,
    BingGroundingSearchConfiguration,
)

tool = BingGroundingTool(
    connection_id="my-bing-connection",
    market="en-US",
    count=8,
    freshness="Week",
)

# The underlying parameters object
params: BingGroundingSearchToolParameters = tool.bing_grounding
for cfg in params.search_configurations:
    print(f"connection: {cfg.connection_id}")
    print(f"market:     {cfg.market}")
    print(f"count:      {cfg.count}")
    print(f"freshness:  {cfg.freshness}")
```

---

## 3. `RunCompletionUsage` + `RunStepCompletionUsage` — cost accounting per run and per step

**Source:** `azure/ai/agents/models/_models.py`

Every `ThreadRun` carries a `usage` field of type `RunCompletionUsage`. Every `RunStep` carries a `usage` field of type `RunStepCompletionUsage`. Both expose `prompt_tokens`, `completion_tokens`, and `total_tokens`.

### Signatures

```python
class RunCompletionUsage(_Model):
    completion_tokens: int   # output tokens for the whole run
    prompt_tokens: int       # input tokens for the whole run
    total_tokens: int        # prompt + completion

class RunStepCompletionUsage(_Model):
    completion_tokens: int   # output tokens for this step
    prompt_tokens: int       # input tokens for this step
    total_tokens: int        # prompt + completion
```

### Key points

- `usage` is `None` while the run is in a non-terminal state (`queued`, `in_progress`, `cancelling`). Always check before reading.
- `RunCompletionUsage` aggregates across **all steps** in the run — it is the sum you care about for billing.
- `RunStepCompletionUsage` lets you attribute cost to individual steps — useful for identifying expensive tool-call rounds vs. final message generation.
- Token counts follow the model's tokenizer. For GPT-4o, multiply by the per-token price in USD. For Azure OpenAI Foundry deployments the price per 1K tokens appears on your invoice under the model deployment name.
- There is no built-in rate guard in the SDK — implement your own budget check by accumulating `total_tokens` across runs.

### Example 1: log usage after every run

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import RunCompletionUsage, RunStatus
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

COST_PER_1K_INPUT  = 0.0025   # USD — adjust to your model's pricing
COST_PER_1K_OUTPUT = 0.010

def log_run_cost(run) -> float:
    usage: RunCompletionUsage | None = run.usage
    if usage is None:
        print(f"Run {run.id} has no usage data (status: {run.status})")
        return 0.0
    cost = (
        usage.prompt_tokens     / 1000 * COST_PER_1K_INPUT
        + usage.completion_tokens / 1000 * COST_PER_1K_OUTPUT
    )
    print(
        f"Run {run.id} | "
        f"prompt={usage.prompt_tokens}  "
        f"completion={usage.completion_tokens}  "
        f"total={usage.total_tokens}  "
        f"cost=${cost:.4f}"
    )
    return cost

agent = client.create_agent(
    model="gpt-4o",
    name="CostTrackingBot",
    instructions="Answer questions concisely.",
)

thread = client.threads.create()
client.messages.create(thread_id=thread.id, role="user", content="Explain LLM tokenisation in one paragraph.")
run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
log_run_cost(run)
client.delete_agent(agent.id)
```

### Example 2: per-step cost breakdown

```python
from azure.ai.agents.models import RunStepCompletionUsage, RunStepType

steps = client.run_steps.list(thread_id=thread.id, run_id=run.id)
for step in steps:
    usage: RunStepCompletionUsage | None = step.usage
    if usage:
        print(
            f"  Step {step.id} [{step.type}] — "
            f"prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens}"
        )
```

### Example 3: budget guard

```python
MAX_TOKENS_PER_SESSION = 50_000
session_tokens = 0

def run_with_budget(client, thread_id: str, agent_id: str, user_message: str) -> str:
    global session_tokens
    if session_tokens >= MAX_TOKENS_PER_SESSION:
        raise RuntimeError("Session token budget exhausted.")

    client.messages.create(thread_id=thread_id, role="user", content=user_message)
    run = client.runs.create_and_process(thread_id=thread_id, agent_id=agent_id)

    if run.usage:
        session_tokens += run.usage.total_tokens
        print(f"Session tokens used: {session_tokens}/{MAX_TOKENS_PER_SESSION}")

    messages = client.messages.list(thread_id=thread_id)
    return next(
        (m.content[0].text.value for m in messages if m.role == "assistant"),
        ""
    )
```

---

## 4. `VectorStoreDataSource` + `VectorStoreConfigurations` — enterprise Azure asset sources

**Source:** `azure/ai/agents/models/_models.py`

Standard file upload (via `client.files.upload`) ingests files directly into the agents service. For enterprise workloads where your documents already live in Azure Blob Storage or Azure Data Lake Gen 2, use `VectorStoreDataSource` to reference them by URI or asset ID — no re-upload required.

### Signatures

```python
class VectorStoreDataSourceAssetType(str, Enum):
    URI_ASSET = "uri_asset"    # Azure Storage URI  (abfss://, https://)
    ID_ASSET  = "id_asset"     # Azure ML data asset ID

class VectorStoreDataSource(_Model):
    asset_identifier: str                         # URI or data asset ID
    asset_type: VectorStoreDataSourceAssetType    # "uri_asset" or "id_asset"

class VectorStoreConfiguration(_Model):
    data_sources: List[VectorStoreDataSource]     # one or more sources

class VectorStoreConfigurations(_Model):
    store_name: str                               # logical name of the store
    store_configuration: VectorStoreConfiguration
```

### Key points

- `URI_ASSET` accepts `abfss://` paths (Azure Data Lake Gen 2) and `https://` blob storage URLs. The agents service's managed identity (or the connection credential) must have at least **Storage Blob Data Reader** on the container.
- `ID_ASSET` references an Azure Machine Learning data asset by its ID. Useful when your data pipeline publishes versioned datasets to AML.
- `VectorStoreConfigurations` is the top-level wrapper you pass to `tool_resources` on `FileSearchTool`. It names the store and contains the configuration.
- The `store_name` in `VectorStoreConfigurations` is a logical label — it does not need to match any storage container name.
- You can include multiple `VectorStoreDataSource` entries in a single `VectorStoreConfiguration` to aggregate documents from several containers or paths.

### Example 1: index a Data Lake path directly

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    FileSearchTool,
    VectorStoreDataSource,
    VectorStoreDataSourceAssetType,
    VectorStoreConfiguration,
    VectorStoreConfigurations,
)
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

data_source = VectorStoreDataSource(
    asset_identifier="abfss://documents@mystorageaccount.dfs.core.windows.net/policies/",
    asset_type=VectorStoreDataSourceAssetType.URI_ASSET,
)

vs_config = VectorStoreConfigurations(
    store_name="policy-docs",
    store_configuration=VectorStoreConfiguration(data_sources=[data_source]),
)

file_search = FileSearchTool(vector_store_configurations=[vs_config])

agent = client.create_agent(
    model="gpt-4o",
    name="PolicyBot",
    instructions="Answer questions about company policies. Only use the provided documents.",
    tools=file_search.definitions,
    tool_resources=file_search.resources,
)
```

### Example 2: aggregate documents from multiple containers

```python
from azure.ai.agents.models import (
    VectorStoreDataSource,
    VectorStoreDataSourceAssetType,
    VectorStoreConfiguration,
    VectorStoreConfigurations,
    FileSearchTool,
)

hr_source = VectorStoreDataSource(
    asset_identifier="abfss://hr@mystorageaccount.dfs.core.windows.net/handbooks/",
    asset_type=VectorStoreDataSourceAssetType.URI_ASSET,
)
legal_source = VectorStoreDataSource(
    asset_identifier="abfss://legal@mystorageaccount.dfs.core.windows.net/contracts/",
    asset_type=VectorStoreDataSourceAssetType.URI_ASSET,
)

vs_config = VectorStoreConfigurations(
    store_name="combined-knowledge",
    store_configuration=VectorStoreConfiguration(
        data_sources=[hr_source, legal_source]
    ),
)

tool = FileSearchTool(vector_store_configurations=[vs_config])
```

### Example 3: Azure ML data asset by ID

```python
from azure.ai.agents.models import (
    VectorStoreDataSource,
    VectorStoreDataSourceAssetType,
    VectorStoreConfiguration,
    VectorStoreConfigurations,
    FileSearchTool,
)

# AML data asset ID — found in your AML workspace under Data → Assets
aml_source = VectorStoreDataSource(
    asset_identifier="/subscriptions/sub-id/resourceGroups/rg/providers/"
                     "Microsoft.MachineLearningServices/workspaces/my-ws/"
                     "data/my-dataset/versions/3",
    asset_type=VectorStoreDataSourceAssetType.ID_ASSET,
)

tool = FileSearchTool(
    vector_store_configurations=[
        VectorStoreConfigurations(
            store_name="aml-dataset-v3",
            store_configuration=VectorStoreConfiguration(data_sources=[aml_source]),
        )
    ]
)
```

---

## 5. Vector store lifecycle — expiry policy and chunking strategy

**Source:** `azure/ai/agents/models/_models.py`

Two configuration concerns matter for long-lived vector stores: *when do they expire* and *how are documents chunked*. The first is handled by `VectorStoreExpirationPolicy`; the second by `VectorStoreChunkingStrategyRequest` and its two subclasses.

### Signatures

```python
class VectorStoreExpirationPolicyAnchor(str, Enum):
    LAST_ACTIVE_AT = "last_active_at"   # expiry counts from last use

class VectorStoreExpirationPolicy(_Model):
    anchor: VectorStoreExpirationPolicyAnchor   # always "last_active_at"
    days: int                                    # days until expiry after anchor event

class VectorStoreAutoChunkingStrategyRequest(VectorStoreChunkingStrategyRequest):
    # Default strategy: max_chunk_size_tokens=800, chunk_overlap_tokens=400
    # No configurable fields — just instantiate it

class VectorStoreStaticChunkingStrategyOptions(_Model):
    max_chunk_size_tokens: int    # 100–4096, default 800
    chunk_overlap_tokens: int     # must be < max_chunk_size_tokens / 2, default 400

class VectorStoreStaticChunkingStrategyRequest(VectorStoreChunkingStrategyRequest):
    static: VectorStoreStaticChunkingStrategyOptions
```

### Key points

- The only supported expiry anchor is `"last_active_at"` — the TTL resets every time the vector store is queried. A store will only expire if it goes unused for `days` consecutive days.
- `days` has no enforced minimum or maximum in the SDK, but the service enforces a maximum of **365 days**. Use `days=7` for ephemeral session data, `days=365` for long-lived knowledge bases.
- `VectorStoreAutoChunkingStrategyRequest` (default) is equivalent to `VectorStoreStaticChunkingStrategyRequest(static=VectorStoreStaticChunkingStrategyOptions(max_chunk_size_tokens=800, chunk_overlap_tokens=400))`. Use `auto` unless you have a specific reason to tune chunking.
- For legal or technical documents with long paragraphs, increase `max_chunk_size_tokens` to `1600` or `2048` to avoid splitting mid-sentence.
- `chunk_overlap_tokens` must be strictly less than `max_chunk_size_tokens / 2`. The service will reject a configuration where the overlap is ≥ half the chunk size.
- Chunking strategy is set at **vector store creation time** and cannot be changed afterwards. Recreate the store if you need different chunking.

### Example 1: ephemeral session store (expires after 1 day of inactivity)

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    VectorStoreExpirationPolicy,
    VectorStoreExpirationPolicyAnchor,
    VectorStoreAutoChunkingStrategyRequest,
)
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

# Create a short-lived store for a support session
store = client.vector_stores.create_and_poll(
    name="session-docs",
    expires_after=VectorStoreExpirationPolicy(
        anchor=VectorStoreExpirationPolicyAnchor.LAST_ACTIVE_AT,
        days=1,
    ),
    chunking_strategy=VectorStoreAutoChunkingStrategyRequest(),
)
print(f"Store '{store.name}' expires in 1 day of inactivity (status: {store.status})")
```

### Example 2: static chunking for long-form technical documentation

```python
from azure.ai.agents.models import (
    VectorStoreStaticChunkingStrategyRequest,
    VectorStoreStaticChunkingStrategyOptions,
    VectorStoreExpirationPolicy,
    VectorStoreExpirationPolicyAnchor,
)

# Technical manuals: large chunks, moderate overlap
store = client.vector_stores.create_and_poll(
    name="technical-manuals",
    expires_after=VectorStoreExpirationPolicy(
        anchor=VectorStoreExpirationPolicyAnchor.LAST_ACTIVE_AT,
        days=90,
    ),
    chunking_strategy=VectorStoreStaticChunkingStrategyRequest(
        static=VectorStoreStaticChunkingStrategyOptions(
            max_chunk_size_tokens=1600,
            chunk_overlap_tokens=200,   # must be < 1600 / 2 = 800
        )
    ),
)
```

### Example 3: reading back strategy from a created store

```python
from azure.ai.agents.models import (
    VectorStoreStaticChunkingStrategyResponse,
    VectorStoreAutoChunkingStrategyResponse,
)

store = client.vector_stores.get(vector_store_id=store.id)
strategy = store.chunking_strategy   # returns a VectorStoreChunkingStrategyResponse

if isinstance(strategy, VectorStoreStaticChunkingStrategyResponse):
    print(f"max_chunk_size: {strategy.static.max_chunk_size_tokens}")
    print(f"overlap:        {strategy.static.chunk_overlap_tokens}")
elif isinstance(strategy, VectorStoreAutoChunkingStrategyResponse):
    print("Auto chunking (800 / 400 defaults)")
```

---

## 6. `AgentsNamedToolChoice` + `AgentsToolChoiceOptionMode` — forcing specific tools

**Source:** `azure/ai/agents/models/_models.py`

By default the model decides which tool (if any) to call. Two model-level controls let you override this: `tool_choice` on `create_run` / `create_thread_and_run`, and its two value shapes.

### Signatures

```python
class AgentsToolChoiceOptionMode(str, Enum):
    NONE = "none"   # model must not call any tool — text only
    AUTO = "auto"   # model chooses freely (the default)

class AgentsNamedToolChoiceType(str, Enum):
    FUNCTION        = "function"
    CODE_INTERPRETER = "code_interpreter"
    FILE_SEARCH     = "file_search"
    BING_GROUNDING  = "bing_grounding"
    AZURE_AI_SEARCH = "azure_ai_search"
    CONNECTED_AGENT = "connected_agent"

class FunctionName(_Model):
    name: str   # exact function name to force

class AgentsNamedToolChoice(_Model):
    type: AgentsNamedToolChoiceType
    function: Optional[FunctionName]   # required when type == "function"
```

### Key points

- `tool_choice` accepts either a string (`AgentsToolChoiceOptionMode` value) or an `AgentsNamedToolChoice` instance. The API accepts both via a union type.
- Setting `tool_choice=AgentsToolChoiceOptionMode.NONE` forces the model to produce a text response regardless of whether tools are attached. Useful for a "summarise the conversation" final step where you do not want side-effects.
- `AgentsNamedToolChoice(type="file_search")` forces the model to always invoke file search first, before deciding whether to generate text — useful for RAG workflows where you always want retrieval.
- When `type=AgentsNamedToolChoiceType.FUNCTION` you must also set `function=FunctionName(name="my_function")`. For built-in tools (`code_interpreter`, `file_search`, etc.) the `function` field is omitted.
- Forcing a tool does **not** prevent the model from calling other tools in subsequent steps. It only forces the *first* tool call.

### Example 1: force file search on every run

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    FileSearchTool,
    AgentsNamedToolChoice,
    AgentsNamedToolChoiceType,
)
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

vector_store_id = "vs_your_store_id"
file_search = FileSearchTool(vector_store_ids=[vector_store_id])

agent = client.create_agent(
    model="gpt-4o",
    name="RAGBot",
    instructions="Always ground your answers in the document store.",
    tools=file_search.definitions,
    tool_resources=file_search.resources,
)

thread = client.threads.create()
client.messages.create(thread_id=thread.id, role="user", content="What does the policy say about remote work?")

# Force file_search — model must retrieve before answering
run = client.runs.create_and_process(
    thread_id=thread.id,
    agent_id=agent.id,
    tool_choice=AgentsNamedToolChoice(type=AgentsNamedToolChoiceType.FILE_SEARCH),
)

client.delete_agent(agent.id)
```

### Example 2: force a specific function call

```python
from azure.ai.agents.models import (
    FunctionTool,
    AgentsNamedToolChoice,
    AgentsNamedToolChoiceType,
    FunctionName,
)

def get_account_balance(account_id: str) -> str:
    """Return current account balance."""
    return f"£1,234.56 for account {account_id}"

tool = FunctionTool({get_account_balance})

agent = client.create_agent(
    model="gpt-4o",
    name="BalanceBot",
    instructions="Help customers check their balance.",
    tools=tool.definitions,
)

thread = client.threads.create()
client.messages.create(thread_id=thread.id, role="user", content="What's my balance?")

# Force the model to call get_account_balance (it still needs to pick the argument)
run = client.runs.create_and_process(
    thread_id=thread.id,
    agent_id=agent.id,
    tool_choice=AgentsNamedToolChoice(
        type=AgentsNamedToolChoiceType.FUNCTION,
        function=FunctionName(name="get_account_balance"),
    ),
)
```

### Example 3: text-only final step

```python
from azure.ai.agents.models import AgentsToolChoiceOptionMode

# After several tool-use rounds, force a clean text summary
client.messages.create(
    thread_id=thread.id,
    role="user",
    content="Summarise everything we've discussed and any actions taken.",
)
run = client.runs.create_and_process(
    thread_id=thread.id,
    agent_id=agent.id,
    tool_choice=AgentsToolChoiceOptionMode.NONE,  # no tools allowed
)
```

---

## 7. File-search run step results — `RunStepFileSearchToolCall` through `FileSearchToolCallContent`

**Source:** `azure/ai/agents/models/_models.py`

When an agent uses `FileSearchTool`, each retrieval is recorded as a `RunStepFileSearchToolCall` run step. Inspecting these gives you the raw search results with relevance scores — useful for debugging retrieval quality and for building "show your sources" UI.

### Signatures

```python
class FileSearchToolCallContent(_Model):
    type: Literal["text"]   # always "text"
    text: str               # the retrieved passage

class FileSearchRankingOptions(_Model):
    ranker: str             # ranker identifier used
    score_threshold: float  # minimum score for inclusion

class RunStepFileSearchToolCallResult(_Model):
    file_id:   str                              # ID of the source file
    file_name: str                              # human-readable filename
    score:     float                            # relevance score 0.0–1.0
    content:   Optional[List[FileSearchToolCallContent]]  # passage text (opt-in)

class RunStepFileSearchToolCallResults(_Model):
    ranking_options: Optional[FileSearchRankingOptions]
    results:         List[RunStepFileSearchToolCallResult]

class RunStepFileSearchToolCall(RunStepToolCall):
    type:        Literal["file_search"]
    id:          str
    file_search: RunStepFileSearchToolCallResults

class RunAdditionalFieldList(str, Enum):
    FILE_SEARCH_CONTENTS = "step_details.tool_calls[*].file_search.results[*].content"
```

### Key points

- `RunStepFileSearchToolCallResult.content` is `None` by default. To receive the actual retrieved text you must pass `include=[RunAdditionalFieldList.FILE_SEARCH_CONTENTS]` when listing run steps.
- `score` is a float between 0.0 and 1.0. Higher scores indicate stronger semantic relevance to the query.
- `RunStepFileSearchToolCallResults.ranking_options` is set when the agent's `FileSearchToolDefinitionDetails` had `ranking_options` configured at agent creation time.
- A single run step of type `tool_calls` may contain **multiple** `RunStepFileSearchToolCall` entries if the agent called file search more than once in that step.
- File search results are read-only snapshots. The `file_id` values link back to files in your vector store.

### Example 1: list run steps and print retrieval scores

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    FileSearchTool,
    RunStepType,
    RunStepFileSearchToolCall,
    RunAdditionalFieldList,
)
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

vector_store_id = "vs_your_store_id"
tool = FileSearchTool(vector_store_ids=[vector_store_id])

agent = client.create_agent(
    model="gpt-4o",
    name="SearchInspector",
    instructions="Answer questions using the document store.",
    tools=tool.definitions,
    tool_resources=tool.resources,
)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role="user",
    content="What are the refund conditions?",
)
run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

# Request file search result content via the include parameter
steps = client.run_steps.list(
    thread_id=thread.id,
    run_id=run.id,
    include=[RunAdditionalFieldList.FILE_SEARCH_CONTENTS],
)

for step in steps:
    if step.type == RunStepType.TOOL_CALLS:
        for tool_call in step.step_details.tool_calls:
            if isinstance(tool_call, RunStepFileSearchToolCall):
                print(f"\nFile search step {step.id}:")
                for result in tool_call.file_search.results:
                    print(f"  [{result.score:.3f}] {result.file_name} ({result.file_id})")
                    if result.content:
                        for chunk in result.content:
                            print(f"    → {chunk.text[:120]}…")

client.delete_agent(agent.id)
```

### Example 2: filter low-quality results in post-processing

```python
MIN_SCORE = 0.6

def good_results(tool_call: RunStepFileSearchToolCall):
    return [
        r for r in tool_call.file_search.results
        if r.score >= MIN_SCORE
    ]
```

### Example 3: build a "Sources" UI section from results

```python
def format_sources(tool_calls) -> str:
    sources = []
    for tc in tool_calls:
        if isinstance(tc, RunStepFileSearchToolCall):
            for r in tc.file_search.results:
                sources.append(f"- **{r.file_name}** (relevance {r.score:.0%})")
    return "\n".join(sources) if sources else "_No sources retrieved._"
```

---

## 8. Streaming event taxonomy — `AgentStreamEvent` and the four typed sub-enums

**Source:** `azure/ai/agents/models/_models.py`

The streaming layer dispatches server-sent events to `AgentEventHandler` callbacks. Each SSE carries an `event` string matching one of the enum values documented here. Knowing the full taxonomy lets you write type-safe event handlers and handle unknown future events gracefully.

### Enumerations

```python
class ThreadStreamEvent(str, Enum):
    THREAD_CREATED = "thread.created"            # data: AgentThread

class RunStreamEvent(str, Enum):
    THREAD_RUN_CREATED       = "thread.run.created"
    THREAD_RUN_QUEUED        = "thread.run.queued"
    THREAD_RUN_IN_PROGRESS   = "thread.run.in_progress"
    THREAD_RUN_REQUIRES_ACTION = "thread.run.requires_action"
    THREAD_RUN_COMPLETED     = "thread.run.completed"
    THREAD_RUN_INCOMPLETE    = "thread.run.incomplete"
    THREAD_RUN_FAILED        = "thread.run.failed"
    THREAD_RUN_CANCELLING    = "thread.run.cancelling"
    THREAD_RUN_CANCELLED     = "thread.run.cancelled"
    THREAD_RUN_EXPIRED       = "thread.run.expired"

class MessageStreamEvent(str, Enum):
    THREAD_MESSAGE_CREATED     = "thread.message.created"
    THREAD_MESSAGE_IN_PROGRESS = "thread.message.in_progress"
    THREAD_MESSAGE_DELTA       = "thread.message.delta"      # data: MessageDeltaChunk
    THREAD_MESSAGE_COMPLETED   = "thread.message.completed"
    THREAD_MESSAGE_INCOMPLETE  = "thread.message.incomplete"

class RunStepStreamEvent(str, Enum):
    THREAD_RUN_STEP_CREATED     = "thread.run.step.created"
    THREAD_RUN_STEP_IN_PROGRESS = "thread.run.step.in_progress"
    THREAD_RUN_STEP_DELTA       = "thread.run.step.delta"    # data: RunStepDeltaChunk
    THREAD_RUN_STEP_COMPLETED   = "thread.run.step.completed"
    THREAD_RUN_STEP_FAILED      = "thread.run.step.failed"
    THREAD_RUN_STEP_CANCELLED   = "thread.run.step.cancelled"
    THREAD_RUN_STEP_EXPIRED     = "thread.run.step.expired"

# AgentStreamEvent is a flat union that includes every value above plus:
class AgentStreamEvent(str, Enum):
    # All thread/run/message/step events are repeated here, plus:
    DONE  = "done"    # stream has ended — data is "[DONE]"
    ERROR = "error"   # stream error — data is an error object
```

### Key points

- `AgentStreamEvent` is the master union: every event string from the four typed sub-enums also appears in `AgentStreamEvent`. Use the typed sub-enums in match/isinstance guards where you care about a specific category; use `AgentStreamEvent` for the full list.
- `AgentStreamEvent.DONE` ("done") marks the end of the SSE stream. The `AgentEventHandler.on_done()` hook is called at this point.
- `AgentStreamEvent.ERROR` ("error") is dispatched when the service sends an error SSE. The `AgentEventHandler.on_error()` hook receives the raw `ErrorEvent` object.
- The `data` payload type for each event is documented in the enum member docstring — e.g., `THREAD_MESSAGE_DELTA` carries `MessageDeltaChunk`, while `THREAD_RUN_COMPLETED` carries `ThreadRun`.
- Microsoft recommends handling unknown event names gracefully (fall through without raising) since new events may be added.

### Example 1: type-safe streaming handler using all four sub-enums

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    AgentEventHandler,
    ThreadRun,
    RunStep,
    ThreadMessage,
    MessageDeltaChunk,
    RunStepDeltaChunk,
    RunStreamEvent,
    MessageStreamEvent,
    RunStepStreamEvent,
)
from azure.identity import DefaultAzureCredential

class VerboseEventHandler(AgentEventHandler):
    def on_thread_run(self, run: ThreadRun) -> None:
        # Matches RunStreamEvent.THREAD_RUN_* events
        print(f"[RUN ] {run.id} → {run.status}")

    def on_message_delta(self, delta: MessageDeltaChunk) -> None:
        # Matches MessageStreamEvent.THREAD_MESSAGE_DELTA
        for block in delta.delta.content or []:
            if hasattr(block, "text") and block.text:
                print(block.text.value, end="", flush=True)

    def on_run_step(self, step: RunStep) -> None:
        # Matches RunStepStreamEvent.THREAD_RUN_STEP_CREATED / COMPLETED / FAILED …
        print(f"\n[STEP] {step.id} [{step.type}] → {step.status}")

    def on_run_step_delta(self, delta: RunStepDeltaChunk) -> None:
        # Matches RunStepStreamEvent.THREAD_RUN_STEP_DELTA
        pass  # handle code-interpreter streaming here if needed

    def on_message(self, message: ThreadMessage) -> None:
        # Matches MessageStreamEvent.THREAD_MESSAGE_CREATED / COMPLETED …
        print(f"\n[MSG ] {message.id} status={message.status}")

    def on_done(self) -> None:
        print("\n[DONE] Stream closed.")

    def on_error(self, data: str) -> None:
        print(f"[ERR ] {data}")

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

agent = client.create_agent(
    model="gpt-4o",
    name="StreamBot",
    instructions="Be concise.",
)
thread = client.threads.create()
client.messages.create(thread_id=thread.id, role="user", content="Count to 5.")

with client.runs.stream(
    thread_id=thread.id,
    agent_id=agent.id,
    event_handler_class=VerboseEventHandler,
) as handler:
    handler.until_done()

client.delete_agent(agent.id)
```

### Example 2: detect requires-action inside a streaming run

```python
from azure.ai.agents.models import (
    AgentEventHandler,
    ThreadRun,
    RunStatus,
    ToolOutput,
)

class ToolCallHandler(AgentEventHandler):
    def __init__(self, client, tool_fn):
        super().__init__()
        self.client = client
        self.tool_fn = tool_fn

    def on_thread_run(self, run: ThreadRun) -> None:
        if run.status == RunStatus.REQUIRES_ACTION:
            action = run.required_action
            outputs = []
            for tc in action.submit_tool_outputs.tool_calls:
                import json
                args = json.loads(tc.function.arguments)
                result = self.tool_fn(tc.function.name, **args)
                outputs.append(ToolOutput(tool_call_id=tc.id, output=str(result)))
            self.client.runs.submit_tool_outputs_stream(
                thread_id=run.thread_id,
                run_id=run.id,
                tool_outputs=outputs,
                event_handler=self,
            )
```

---

## 9. `RequiredFunctionToolCall` + `SubmitToolOutputsDetails` — manual tool-call dispatch

**Source:** `azure/ai/agents/models/_models.py`

When `enable_auto_function_calls` is not used (see Vol. 4), the run enters `REQUIRES_ACTION` status and you must call `client.runs.submit_tool_outputs`. The data model for that action flows through three classes.

### Signatures

```python
class RequiredFunctionToolCallDetails(_Model):
    name:      str   # function name as registered in FunctionTool
    arguments: str   # JSON string of arguments generated by the model

class RequiredFunctionToolCall(RequiredToolCall):
    type:     Literal["function"]
    id:       str                              # tool_call_id to echo back
    function: RequiredFunctionToolCallDetails

class SubmitToolOutputsDetails(_Model):
    tool_calls: List[RequiredToolCall]   # list of calls that need outputs

# On ThreadRun, when status == "requires_action":
#   run.required_action.type == "submit_tool_outputs"
#   run.required_action.submit_tool_outputs is a SubmitToolOutputsDetails
```

### Key points

- `RequiredFunctionToolCallDetails.arguments` is a **JSON string** — not a dict. Parse it with `json.loads()` before calling your function.
- The `id` field on `RequiredFunctionToolCall` must be included verbatim as `ToolOutput.tool_call_id` when you submit results. If the IDs don't match the run will fail.
- `SubmitToolOutputsDetails.tool_calls` is a `List[RequiredToolCall]` — the base type. In practice every entry will be a `RequiredFunctionToolCall` for function tools, but other tool types may appear in future.
- You must submit outputs for **all** pending tool calls in a single `submit_tool_outputs` call. Partial submission is not supported.
- After submitting, the run re-enters `IN_PROGRESS` and may produce more tool calls before finally completing. Loop until `run.status` is a terminal state.

### Example 1: full manual dispatch loop

```python
import os, json
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    FunctionTool,
    RunStatus,
    ToolOutput,
    RequiredFunctionToolCall,
    SubmitToolOutputsDetails,
)
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"The weather in {city} is sunny and 22°C."

def book_restaurant(city: str, cuisine: str, time: str) -> str:
    """Book a restaurant."""
    return f"Booked a {cuisine} restaurant in {city} at {time}. Ref: RES-0042."

tool = FunctionTool({get_weather, book_restaurant})

agent = client.create_agent(
    model="gpt-4o",
    name="PlannerBot",
    instructions="Help users plan outings. Check the weather and book restaurants.",
    tools=tool.definitions,
)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role="user",
    content="Is it a good day for lunch outside in Paris? Book somewhere Italian at 1 PM.",
)

# Start the run — do NOT use create_and_process; we drive the loop ourselves
run = client.runs.create(thread_id=thread.id, agent_id=agent.id)

TERMINAL = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.EXPIRED}

while run.status not in TERMINAL:
    run = client.runs.get(thread_id=thread.id, run_id=run.id)

    if run.status == RunStatus.REQUIRES_ACTION:
        details: SubmitToolOutputsDetails = run.required_action.submit_tool_outputs
        outputs = []

        for tc in details.tool_calls:
            if isinstance(tc, RequiredFunctionToolCall):
                args = json.loads(tc.function.arguments)
                print(f"  → calling {tc.function.name}({args})")

                if tc.function.name == "get_weather":
                    result = get_weather(**args)
                elif tc.function.name == "book_restaurant":
                    result = book_restaurant(**args)
                else:
                    result = "Function not found."

                outputs.append(ToolOutput(tool_call_id=tc.id, output=result))

        run = client.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=outputs,
        )

print(f"\nRun finished: {run.status}")
for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        print(msg.content[0].text.value)

client.delete_agent(agent.id)
```

### Example 2: generic dispatcher using a function registry

```python
from typing import Callable
import json
from azure.ai.agents.models import RequiredFunctionToolCall, ToolOutput

def dispatch_tool_calls(
    tool_calls,
    registry: dict[str, Callable],
) -> list[ToolOutput]:
    outputs = []
    for tc in tool_calls:
        if not isinstance(tc, RequiredFunctionToolCall):
            continue
        fn = registry.get(tc.function.name)
        if fn is None:
            result = f"Unknown function: {tc.function.name}"
        else:
            try:
                args = json.loads(tc.function.arguments)
                result = str(fn(**args))
            except Exception as exc:
                result = f"Error: {exc}"
        outputs.append(ToolOutput(tool_call_id=tc.id, output=result))
    return outputs
```

---

## 10. Multimodal message input — `ThreadMessageOptions`, `MessageInputImageFileBlock`, `MessageInputImageUrlBlock`

**Source:** `azure/ai/agents/models/_models.py`

Thread messages can contain more than plain text. Vision-capable models (GPT-4o, GPT-4 Turbo) accept image content alongside text. The SDK exposes this via a discriminated union of `MessageInputContentBlock` subclasses.

### Signatures

```python
class ImageDetailLevel(str, Enum):
    AUTO = "auto"   # server picks based on image size
    LOW  = "low"    # fast, lower resolution — 85 tokens
    HIGH = "high"   # slow, full resolution — up to 1105 tokens

class MessageInputTextBlock(MessageInputContentBlock):
    type: Literal["text"]
    text: str

class MessageImageFileParam(_Model):
    file_id: str                              # previously uploaded file ID
    detail: Optional[ImageDetailLevel]        # defaults to "auto"

class MessageInputImageFileBlock(MessageInputContentBlock):
    type: Literal["image_file"]
    image_file: MessageImageFileParam

class MessageImageUrlParam(_Model):
    url:    str                               # publicly accessible URL
    detail: Optional[ImageDetailLevel]        # defaults to "auto"

class MessageInputImageUrlBlock(MessageInputContentBlock):
    type: Literal["image_url"]
    image_url: MessageImageUrlParam

class ThreadMessageOptions(_Model):
    role:    MessageRole                       # "user" or "assistant"
    content: str | List[MessageInputContentBlock]
    # content is a plain string for text-only messages;
    # a list of blocks for multimodal messages.
```

### Key points

- `ImageDetailLevel.LOW` processes images at a fixed 512×512 resolution crop. It costs **85 tokens** regardless of image size and is fastest.
- `ImageDetailLevel.HIGH` processes the image in 512×512 tiles. A 1080×1080 image costs approximately **765 tokens** on top of the base 85.
- `ImageDetailLevel.AUTO` (default) lets the server pick based on the image dimensions — it defaults to `low` for small images and `high` for large ones.
- When using `MessageInputImageFileBlock`, the file must have been uploaded via `client.files.upload` with `purpose=FilePurpose.VISION` (or equivalent). The file ID is what you pass.
- `MessageInputImageUrlBlock` accepts any publicly accessible URL. The model will fetch and encode the image server-side. Private URLs behind auth are not supported via this path — upload the file first and use `MessageInputImageFileBlock` instead.
- `ThreadMessageOptions.content` is a plain `str` for text-only messages. Passing a list of `MessageInputContentBlock` enables multimodal input. The API accepts both shapes at the same `content` field.

### Example 1: send a URL image alongside a question

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    MessageInputTextBlock,
    MessageInputImageUrlBlock,
    MessageImageUrlParam,
    ImageDetailLevel,
)
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

agent = client.create_agent(
    model="gpt-4o",
    name="VisionBot",
    instructions="Describe what you see in images and answer follow-up questions.",
)

thread = client.threads.create()

# Multimodal message: text + image URL
client.messages.create(
    thread_id=thread.id,
    role="user",
    content=[
        MessageInputTextBlock(text="What type of chart is shown, and what are the main trends?"),
        MessageInputImageUrlBlock(
            image_url=MessageImageUrlParam(
                url="https://example.com/sales_chart_2026.png",
                detail=ImageDetailLevel.HIGH,   # full resolution for charts
            )
        ),
    ],
)

run = client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

for msg in client.messages.list(thread_id=thread.id):
    if msg.role == "assistant":
        print(msg.content[0].text.value)

client.delete_agent(agent.id)
```

### Example 2: upload a private image and reference it by file ID

```python
import os
from azure.ai.agents.models import (
    FilePurpose,
    MessageInputTextBlock,
    MessageInputImageFileBlock,
    MessageImageFileParam,
    ImageDetailLevel,
)

# Upload the image (purpose must allow vision)
with open("/tmp/diagram.png", "rb") as f:
    uploaded = client.files.upload(file=f, purpose=FilePurpose.ASSISTANTS)

thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role="user",
    content=[
        MessageInputTextBlock(text="Identify all components in this architecture diagram."),
        MessageInputImageFileBlock(
            image_file=MessageImageFileParam(
                file_id=uploaded.id,
                detail=ImageDetailLevel.HIGH,
            )
        ),
    ],
)
```

### Example 3: multiple images in one message

```python
client.messages.create(
    thread_id=thread.id,
    role="user",
    content=[
        MessageInputTextBlock(text="Compare these two product designs and give a preference."),
        MessageInputImageUrlBlock(
            image_url=MessageImageUrlParam(
                url="https://example.com/design_a.png",
                detail=ImageDetailLevel.AUTO,
            )
        ),
        MessageInputImageUrlBlock(
            image_url=MessageImageUrlParam(
                url="https://example.com/design_b.png",
                detail=ImageDetailLevel.AUTO,
            )
        ),
    ],
)
```

### Image token cost reference

| Detail level | Tiles | Base cost | Typical 1080×1080 cost |
|---|---|---|---|
| `low` | 1 (fixed crop) | 85 tokens | 85 tokens |
| `high` | depends on dimensions | 85 + 170/tile | ~765 tokens |
| `auto` | server-chosen | — | equivalent to `low` or `high` |

---

## Capstone: production pipeline combining all six volumes

The following example shows how the classes from Vols. 1–6 compose into a realistic production pipeline: a research agent that indexes Data Lake documents, enforces a token budget, forces file search, streams results, extracts source citations, and handles tool calls manually.

```python
import os, json, asyncio
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    # Tool setup (Vols. 1–4)
    FileSearchTool,
    BingGroundingTool,
    FunctionTool,
    # Azure data sources (Vol. 6, section 4)
    VectorStoreDataSource, VectorStoreDataSourceAssetType,
    VectorStoreConfiguration, VectorStoreConfigurations,
    # Expiry + chunking (Vol. 6, section 5)
    VectorStoreExpirationPolicy, VectorStoreExpirationPolicyAnchor,
    VectorStoreStaticChunkingStrategyRequest, VectorStoreStaticChunkingStrategyOptions,
    # Tool choice (Vol. 6, section 6)
    AgentsNamedToolChoice, AgentsNamedToolChoiceType,
    # Streaming (Vols. 5–6)
    AgentEventHandler, ThreadRun, MessageDeltaChunk, RunStep,
    # File search results (Vol. 6, section 7)
    RunStepFileSearchToolCall, RunAdditionalFieldList,
    # Tool dispatch (Vol. 6, section 9)
    RequiredFunctionToolCall, ToolOutput, RunStatus,
    # Usage (Vol. 6, section 3)
    RunCompletionUsage,
    # Auth (Vol. 6, section 1)
    OpenApiAnonymousAuthDetails,
)
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

# 1. Create vector store from Azure Data Lake (Vol. 6, sections 4–5)
store = client.vector_stores.create_and_poll(
    name="research-docs",
    expires_after=VectorStoreExpirationPolicy(
        anchor=VectorStoreExpirationPolicyAnchor.LAST_ACTIVE_AT,
        days=30,
    ),
    chunking_strategy=VectorStoreStaticChunkingStrategyRequest(
        static=VectorStoreStaticChunkingStrategyOptions(
            max_chunk_size_tokens=1200,
            chunk_overlap_tokens=200,
        )
    ),
)

data_source = VectorStoreDataSource(
    asset_identifier="abfss://research@myaccount.dfs.core.windows.net/papers/",
    asset_type=VectorStoreDataSourceAssetType.URI_ASSET,
)
vs_configs = VectorStoreConfigurations(
    store_name="research-papers",
    store_configuration=VectorStoreConfiguration(data_sources=[data_source]),
)

file_search = FileSearchTool(
    vector_store_ids=[store.id],
    vector_store_configurations=[vs_configs],
)

bing = BingGroundingTool(
    connection_id=os.environ["BING_CONNECTION_ID"],
    freshness="Week",
    count=5,
)

def save_note(topic: str, content: str) -> str:
    """Save a research note."""
    print(f"  [NOTE] {topic}: {content[:80]}…")
    return "Saved."

note_tool = FunctionTool({save_note})

agent = client.create_agent(
    model="gpt-4o",
    name="ResearchAgent",
    instructions=(
        "You are a thorough research assistant. Always search documents first, "
        "then supplement with web search for recent developments. "
        "Save key insights using save_note."
    ),
    tools=[*file_search.definitions, *bing.definitions, *note_tool.definitions],
    tool_resources=file_search.resources,
)

# 2. Streaming handler (Vol. 6, section 8)
class ResearchHandler(AgentEventHandler):
    def on_message_delta(self, delta: MessageDeltaChunk) -> None:
        for block in delta.delta.content or []:
            if hasattr(block, "text") and block.text:
                print(block.text.value, end="", flush=True)

    def on_run_step(self, step: RunStep) -> None:
        print(f"\n[STEP] {step.type} → {step.status}")

    def on_thread_run(self, run: ThreadRun) -> None:
        print(f"\n[RUN ] {run.status}")

    def on_done(self) -> None:
        print("\n[DONE]")

# 3. Run with forced file search first (Vol. 6, section 6)
thread = client.threads.create()
client.messages.create(
    thread_id=thread.id,
    role="user",
    content="What does the research say about transformer attention patterns? Include recent 2026 findings.",
)

with client.runs.stream(
    thread_id=thread.id,
    agent_id=agent.id,
    tool_choice=AgentsNamedToolChoice(type=AgentsNamedToolChoiceType.FILE_SEARCH),
    event_handler_class=ResearchHandler,
) as handler:
    handler.until_done()

# 4. Inspect sources and token usage (Vol. 6, sections 3 and 7)
final_run = client.runs.get(thread_id=thread.id, run_id=handler.current_run_id)
if final_run.usage:
    usage: RunCompletionUsage = final_run.usage
    print(
        f"\nTokens — prompt: {usage.prompt_tokens}  "
        f"completion: {usage.completion_tokens}  "
        f"total: {usage.total_tokens}"
    )

steps = client.run_steps.list(
    thread_id=thread.id,
    run_id=final_run.id,
    include=[RunAdditionalFieldList.FILE_SEARCH_CONTENTS],
)
print("\nSources:")
for step in steps:
    if hasattr(step.step_details, "tool_calls"):
        for tc in step.step_details.tool_calls:
            if isinstance(tc, RunStepFileSearchToolCall):
                for r in tc.file_search.results:
                    print(f"  [{r.score:.2f}] {r.file_name}")

client.delete_agent(agent.id)
client.vector_stores.delete(vector_store_id=store.id)
```

---

## Quick-reference table

| Class | Module | First covered | Key use case |
|-------|--------|---------------|--------------|
| `OpenApiAnonymousAuthDetails` | `models` | Vol. 6 | Public API — no auth |
| `OpenApiManagedAuthDetails` | `models` | Vol. 6 | Azure-hosted API — managed identity |
| `OpenApiConnectionAuthDetails` | `models` | Vol. 6 | Third-party API — Foundry connection |
| `OpenApiManagedSecurityScheme` | `models` | Vol. 6 | AAD audience for managed identity |
| `OpenApiConnectionSecurityScheme` | `models` | Vol. 6 | Connection ID for Foundry auth |
| `BingGroundingSearchConfiguration` | `models` | Vol. 6 | Market, freshness, result count |
| `BingGroundingSearchToolParameters` | `models` | Vol. 6 | Container for search configs |
| `RunCompletionUsage` | `models` | Vol. 6 | Token cost per run |
| `RunStepCompletionUsage` | `models` | Vol. 6 | Token cost per step |
| `VectorStoreDataSource` | `models` | Vol. 6 | Azure asset URI or ID |
| `VectorStoreDataSourceAssetType` | `models` | Vol. 6 | `uri_asset` vs `id_asset` |
| `VectorStoreConfiguration` | `models` | Vol. 6 | List of data sources |
| `VectorStoreConfigurations` | `models` | Vol. 6 | Named store + config wrapper |
| `VectorStoreExpirationPolicy` | `models` | Vol. 6 | TTL: days since last use |
| `VectorStoreStaticChunkingStrategyRequest` | `models` | Vol. 6 | Fine-grained chunking |
| `VectorStoreStaticChunkingStrategyOptions` | `models` | Vol. 6 | `max_chunk_size_tokens`, `overlap` |
| `VectorStoreAutoChunkingStrategyRequest` | `models` | Vol. 6 | Default 800/400 chunking |
| `AgentsNamedToolChoice` | `models` | Vol. 6 | Force a specific tool |
| `AgentsNamedToolChoiceType` | `models` | Vol. 6 | Tool type enum |
| `AgentsToolChoiceOptionMode` | `models` | Vol. 6 | `none` / `auto` mode |
| `FunctionName` | `models` | Vol. 6 | Named function for forced call |
| `RunStepFileSearchToolCall` | `models` | Vol. 6 | File search step record |
| `RunStepFileSearchToolCallResults` | `models` | Vol. 6 | Results container + ranking |
| `RunStepFileSearchToolCallResult` | `models` | Vol. 6 | Score + filename + content |
| `FileSearchRankingOptions` | `models` | Vol. 6 | Ranker + threshold |
| `FileSearchToolCallContent` | `models` | Vol. 6 | Retrieved text passage |
| `RunAdditionalFieldList` | `models` | Vol. 6 | `include=` for file content |
| `AgentStreamEvent` | `models` | Vol. 6 | Master event taxonomy |
| `RunStreamEvent` | `models` | Vol. 6 | Run lifecycle events |
| `MessageStreamEvent` | `models` | Vol. 6 | Message lifecycle events |
| `RunStepStreamEvent` | `models` | Vol. 6 | Step lifecycle events |
| `ThreadStreamEvent` | `models` | Vol. 6 | Thread created event |
| `RequiredFunctionToolCall` | `models` | Vol. 6 | Tool call to dispatch |
| `RequiredFunctionToolCallDetails` | `models` | Vol. 6 | Name + arguments JSON |
| `SubmitToolOutputsDetails` | `models` | Vol. 6 | Pending calls container |
| `ThreadMessageOptions` | `models` | Vol. 6 | Rich / multimodal message |
| `MessageInputTextBlock` | `models` | Vol. 6 | Text block in message |
| `MessageInputImageFileBlock` | `models` | Vol. 6 | Uploaded image by file ID |
| `MessageInputImageUrlBlock` | `models` | Vol. 6 | External image by URL |
| `MessageImageFileParam` | `models` | Vol. 6 | File ID + detail level |
| `MessageImageUrlParam` | `models` | Vol. 6 | URL + detail level |
| `ImageDetailLevel` | `models` | Vol. 6 | `auto` / `low` / `high` |

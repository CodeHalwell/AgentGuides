---
title: "Class deep dives — volume 17 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.2.0 classes: LongRunningFunctionTool (async background task pattern), OpenAPIToolset + RestApiTool (REST API from OpenAPI specs), PubSubToolset + PubSubToolConfig (event-driven pub/sub, @experimental), VertexAiRagRetrieval (built-in Vertex AI RAG), RetryConfig (workflow node retry with exponential backoff + jitter), RemoteA2aAgent (calling remote A2A services from ADK graphs), LiveRequest + LiveRequestQueue (bidirectional streaming live agents), AuthHandler + AuthSchemes (full OAuth2 flow orchestration), SkillToolset (agent skill discovery + code execution), GoogleSearchTool + UrlContextTool + GoogleMapsGroundingTool (model-native built-in tools)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 17"
  order: 86
---

Source-verified against **google-adk==2.2.0** (installed from PyPI, June 2026). Every field name, signature, and code example is drawn from the installed package source at `/usr/local/lib/python3.11/dist-packages/google/adk/`.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `LongRunningFunctionTool` | `google.adk.tools.long_running_tool` | Stable |
| 2 | `OpenAPIToolset` + `RestApiTool` | `google.adk.tools.openapi_tool` | Stable |
| 3 | `PubSubToolset` + `PubSubToolConfig` + `PubSubCredentialsConfig` | `google.adk.tools.pubsub` | `@experimental` |
| 4 | `VertexAiRagRetrieval` | `google.adk.tools.retrieval.vertex_ai_rag_retrieval` | Stable |
| 5 | `RetryConfig` | `google.adk.workflow._retry_config` | Stable |
| 6 | `RemoteA2aAgent` | `google.adk.agents.remote_a2a_agent` | Stable |
| 7 | `LiveRequest` + `LiveRequestQueue` | `google.adk.agents.live_request_queue` | Stable |
| 8 | `AuthHandler` + `AuthSchemes` | `google.adk.auth.auth_handler`, `google.adk.auth.auth_schemes` | Stable |
| 9 | `SkillToolset` | `google.adk.tools.skill_toolset` | Stable |
| 10 | `GoogleSearchTool` + `UrlContextTool` + `GoogleMapsGroundingTool` | `google.adk.tools` | Stable |

---

## 1 · `LongRunningFunctionTool`

**Source:** `google.adk.tools.long_running_tool`

`LongRunningFunctionTool` is a thin subclass of `FunctionTool` that marks an operation as long-running. The framework interprets this flag and returns the function's result **asynchronously** — identified by `function_call_id` — rather than blocking the agent turn. It also appends a guidance note to the tool's description, instructing the model not to re-invoke the tool if an intermediate or pending status has already been returned.

### Constructor (source-verified)

```python
from google.adk.tools.long_running_tool import LongRunningFunctionTool

LongRunningFunctionTool(func: Callable)
```

The only difference from `FunctionTool` is:
- `self.is_long_running = True` — the framework checks this flag.
- The description is auto-amended with: `"NOTE: This is a long-running operation. Do not call this tool again if it has already returned some intermediate or pending status."`

### When to use it

Use `LongRunningFunctionTool` for operations that:
- Take more than a few seconds (data exports, image generation, ETL runs, ML training jobs).
- Need to return a **pending status** immediately and stream or poll for the final result.
- Must not be re-called if the model sees a partial response.

### Example 1 — export report (returns pending → final)

```python
import asyncio
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.tool_context import ToolContext

_JOBS: dict[str, dict] = {}


async def export_report(
    report_type: str,
    date_range: str,
    tool_context: ToolContext,
) -> dict:
    """Export a report.  Returns a job_id immediately; the report is ready when status is DONE.

    Args:
        report_type: Type of report to export, e.g. 'sales' or 'inventory'.
        date_range: Date range string, e.g. '2026-01-01/2026-06-01'.

    Returns:
        dict with job_id and status.
    """
    job_id = f"job_{report_type}_{id(tool_context)}"
    _JOBS[job_id] = {"status": "PENDING", "report_type": report_type, "date_range": date_range}

    # Fire-and-forget: simulate long work
    async def _do_work():
        await asyncio.sleep(5)  # real work goes here
        _JOBS[job_id]["status"] = "DONE"
        _JOBS[job_id]["download_url"] = f"https://reports.example.com/{job_id}.csv"

    asyncio.create_task(_do_work())
    return {"job_id": job_id, "status": "PENDING"}


async def get_report_status(job_id: str) -> dict:
    """Check the status of a previously submitted export job.

    Args:
        job_id: The job_id returned by export_report.

    Returns:
        dict with status and optionally a download_url.
    """
    return _JOBS.get(job_id, {"status": "NOT_FOUND"})


export_tool = LongRunningFunctionTool(func=export_report)

agent = LlmAgent(
    name="report_agent",
    model="gemini-2.0-flash",
    instruction=(
        "Help users export and download reports. "
        "Call export_report to start a job, then call get_report_status with the job_id "
        "to check when it is DONE before giving a download URL."
    ),
    tools=[export_tool, get_report_status],
)
```

### Example 2 — ML training job with intermediate progress

```python
import time
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.agents.llm_agent import LlmAgent


_TRAINING_JOBS: dict = {}


def start_training(
    model_name: str,
    dataset: str,
    epochs: int = 10,
) -> dict:
    """Start a model training job.

    Args:
        model_name: Name of the model to train.
        dataset: Name of the dataset to use.
        epochs: Number of training epochs.

    Returns:
        dict with job_id and status.
    """
    job_id = f"train_{model_name}_{int(time.time())}"
    _TRAINING_JOBS[job_id] = {
        "status": "RUNNING",
        "current_epoch": 0,
        "total_epochs": epochs,
        "model_name": model_name,
        "dataset": dataset,
    }
    # In production, submit to Vertex AI Training / similar
    return {"job_id": job_id, "status": "RUNNING", "message": f"Training {model_name} started."}


def check_training(job_id: str) -> dict:
    """Check the progress of a training job.

    Args:
        job_id: The job ID returned by start_training.

    Returns:
        dict with status, current_epoch, and optionally model_path.
    """
    job = _TRAINING_JOBS.get(job_id)
    if not job:
        return {"status": "NOT_FOUND"}
    return job


training_tool = LongRunningFunctionTool(func=start_training)

agent = LlmAgent(
    name="ml_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You manage ML training jobs. "
        "When a user asks to train a model, call start_training and report the job_id. "
        "Poll check_training until status is COMPLETED before announcing success."
    ),
    tools=[training_tool, check_training],
)
```

### Example 3 — checking `is_long_running` at runtime

```python
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.function_tool import FunctionTool

def slow_op(x: int) -> int:
    """A slow operation."""
    return x * 2

def fast_op(x: int) -> int:
    """A fast operation."""
    return x + 1

long_tool = LongRunningFunctionTool(func=slow_op)
fast_tool = FunctionTool(func=fast_op)

print(long_tool.is_long_running)   # True
print(fast_tool.is_long_running)   # False  (FunctionTool default)

# The description carries the ADK-injected instruction
decl = long_tool._get_declaration()
assert "long-running operation" in decl.description
```

---

## 2 · `OpenAPIToolset` + `RestApiTool`

**Sources:** `google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset`, `google.adk.tools.openapi_tool.openapi_spec_parser.rest_api_tool`

`OpenAPIToolset` parses an OpenAPI 3.x spec (JSON or YAML) and generates one `RestApiTool` per operation. Each `RestApiTool` is a fully wired HTTP client that maps LLM arguments to request parameters, handles auth, and returns the parsed JSON response. Property names are converted to snake_case by default; set `preserve_property_names=True` if the API requires camelCase.

### Constructor (source-verified)

```python
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

OpenAPIToolset(
    *,
    spec_dict: dict | None = None,                 # already-parsed spec dict
    spec_str: str | None = None,                   # raw JSON/YAML string
    spec_str_type: Literal["json", "yaml"] = "json",
    auth_scheme: AuthScheme | None = None,
    auth_credential: AuthCredential | None = None,
    credential_key: str | None = None,             # stable key for credential caching
    tool_filter: ToolPredicate | list[str] | None = None,
    tool_name_prefix: str | None = None,
    ssl_verify: bool | str | ssl.SSLContext | None = None,  # None = default (True)
    header_provider: Callable[[ReadonlyContext], dict[str, str]] | None = None,
    httpx_client_factory: Callable[[], httpx.AsyncClient] | None = None,
    preserve_property_names: bool = False,         # keep original camelCase names
)
```

Either `spec_dict` or `spec_str` must be provided (not both).

### Example 1 — load a public OpenAPI spec from YAML

```python
import httpx
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from google.adk.agents.llm_agent import LlmAgent

# Inline minimal spec — in production, load from file or URL
PETSTORE_SPEC = """
openapi: "3.0.0"
info:
  title: Pet Store
  version: "1.0"
paths:
  /pets:
    get:
      operationId: list_pets
      summary: List all pets
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
      responses:
        "200":
          description: A list of pets
  /pets/{petId}:
    get:
      operationId: show_pet_by_id
      summary: Info for a specific pet
      parameters:
        - name: petId
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          description: Expected response to a valid request
servers:
  - url: https://petstore.example.com/v1
"""

toolset = OpenAPIToolset(spec_str=PETSTORE_SPEC, spec_str_type="yaml")

agent = LlmAgent(
    name="pet_agent",
    model="gemini-2.0-flash",
    instruction="Help users browse the pet store. Use list_pets and show_pet_by_id.",
    tools=[toolset],
)

# Inspect generated tool names
import asyncio
tools = asyncio.run(toolset.get_tools())
print([t.name for t in tools])  # ['list_pets', 'show_pet_by_id']
```

### Example 2 — OpenAPI spec with API-key auth

```python
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from google.adk.tools.openapi_tool.auth.auth_helpers import (
    api_key_scheme_with_header,
    api_key_credential,
)
from google.adk.agents.llm_agent import LlmAgent

spec_dict = {
    "openapi": "3.0.0",
    "info": {"title": "Weather API", "version": "1.0"},
    "paths": {
        "/current": {
            "get": {
                "operationId": "get_current_weather",
                "summary": "Get current weather",
                "parameters": [
                    {"name": "city", "in": "query", "required": True,
                     "schema": {"type": "string"}},
                ],
                "responses": {"200": {"description": "Weather data"}},
            }
        }
    },
    "servers": [{"url": "https://api.weatherapi.example.com/v1"}],
}

toolset = OpenAPIToolset(
    spec_dict=spec_dict,
    auth_scheme=api_key_scheme_with_header("X-Api-Key"),
    auth_credential=api_key_credential("YOUR_API_KEY_HERE"),
)

agent = LlmAgent(
    name="weather_agent",
    model="gemini-2.0-flash",
    instruction="Answer questions about current weather using get_current_weather.",
    tools=[toolset],
)
```

### Example 3 — OAuth2 with a Bearer token + dynamic headers

```python
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.openapi_tool.auth.auth_helpers import (
    bearer_token_scheme,
    bearer_token_credential,
)
from google.adk.agents.llm_agent import LlmAgent

def _headers(ctx: ReadonlyContext) -> dict[str, str]:
    """Inject a correlation ID from session state into every request."""
    return {"X-Correlation-Id": ctx.state.get("correlation_id", "unknown")}

toolset = OpenAPIToolset(
    spec_str=open("crm_api_spec.yaml").read(),
    spec_str_type="yaml",
    auth_scheme=bearer_token_scheme(),
    auth_credential=bearer_token_credential("MY_BEARER_TOKEN"),
    header_provider=_headers,
    tool_name_prefix="crm",         # tools become crm_list_contacts, crm_create_lead, …
    tool_filter=["crm_list_contacts", "crm_create_lead"],
)

agent = LlmAgent(
    name="crm_agent",
    model="gemini-2.0-flash",
    instruction="Help users manage CRM contacts and leads.",
    tools=[toolset],
)
```

### Example 4 — custom SSL context (corporate proxy with custom CA)

```python
import ssl
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

ctx = ssl.create_default_context()
ctx.load_verify_locations("/etc/ssl/corporate-ca.crt")

toolset = OpenAPIToolset(
    spec_str=open("internal_api.json").read(),
    spec_str_type="json",
    ssl_verify=ctx,          # custom SSLContext
    preserve_property_names=True,  # API wants camelCase params
)
```

---

## 3 · `PubSubToolset` + `PubSubToolConfig` + `PubSubCredentialsConfig`

**Sources:** `google.adk.tools.pubsub.pubsub_toolset`, `google.adk.tools.pubsub.config`, `google.adk.tools.pubsub.pubsub_credentials`

`PubSubToolset` is marked `@experimental(FeatureName.PUBSUB_TOOLSET)` and exposes three tools for interacting with Google Cloud Pub/Sub:

| Tool name | Function |
|---|---|
| `publish_message` | Publish a message to a topic |
| `pull_messages` | Pull messages from a subscription |
| `acknowledge_messages` | Acknowledge pulled messages (prevent re-delivery) |

All three tools are wrapped as `GoogleTool` instances, so they support the same credential injection patterns as the rest of the Google tool ecosystem.

### Constructors (source-verified)

```python
from google.adk.tools.pubsub.pubsub_toolset import PubSubToolset
from google.adk.tools.pubsub.config import PubSubToolConfig
from google.adk.tools.pubsub.pubsub_credentials import PubSubCredentialsConfig

PubSubToolset(
    *,
    tool_filter: ToolPredicate | list[str] | None = None,
    credentials_config: PubSubCredentialsConfig | None = None,
    pubsub_tool_config: PubSubToolConfig | None = None,
)

PubSubToolConfig(
    project_id: str | None = None,   # GCP project; inferred from env if None
)

# PubSubCredentialsConfig inherits from BaseGoogleCredentialsConfig
# and supports the same 3 auth options:
#   credentials=<google.auth.credentials.Credentials>
#   external_access_token_key="session_state_key"
#   client_id + client_secret + scopes
```

### Tool signatures (source-verified from `message_tool.py`)

```python
# publish_message
publish_message(
    topic_name: str,              # "projects/my-project/topics/my-topic"
    message: str,
    credentials: Credentials,     # injected by GoogleTool
    settings: PubSubToolConfig,   # injected by GoogleTool
    attributes: dict[str, str] | None = None,
    ordering_key: str = "",
) -> dict  # {"message_id": "..."}

# pull_messages
pull_messages(
    subscription_name: str,       # "projects/my-project/subscriptions/my-sub"
    credentials: Credentials,
    settings: PubSubToolConfig,
    *,
    max_messages: int = 1,
    auto_ack: bool = False,       # True → ack immediately after pull
) -> dict  # {"messages": [{message_id, data, attributes, ordering_key, publish_time, ack_id}, ...]}

# acknowledge_messages
acknowledge_messages(
    subscription_name: str,
    ack_ids: list[str],           # ack_ids from pull_messages response
    credentials: Credentials,
    settings: PubSubToolConfig,
) -> dict  # {"status": "SUCCESS"} or {"status": "ERROR", "error_details": "..."}
```

### Example 1 — event dispatcher agent (publish)

```python
from google.adk.tools.pubsub.pubsub_toolset import PubSubToolset
from google.adk.tools.pubsub.config import PubSubToolConfig
from google.adk.agents.llm_agent import LlmAgent
import google.auth

# Use Application Default Credentials
adc_credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/pubsub"]
)

from google.adk.tools.pubsub.pubsub_credentials import PubSubCredentialsConfig

toolset = PubSubToolset(
    credentials_config=PubSubCredentialsConfig(credentials=adc_credentials),
    pubsub_tool_config=PubSubToolConfig(project_id="my-gcp-project"),
    tool_filter=["publish_message"],  # publish only; no pull/ack
)

agent = LlmAgent(
    name="event_dispatcher",
    model="gemini-2.0-flash",
    instruction=(
        "You dispatch order events. When a user submits an order, "
        "publish it to projects/my-gcp-project/topics/orders as a JSON string."
    ),
    tools=[toolset],
)
```

### Example 2 — consumer agent (pull, process, acknowledge)

```python
from google.adk.tools.pubsub.pubsub_toolset import PubSubToolset
from google.adk.tools.pubsub.pubsub_credentials import PubSubCredentialsConfig
from google.adk.tools.pubsub.config import PubSubToolConfig
from google.adk.agents.llm_agent import LlmAgent
import google.auth

creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/pubsub"])

toolset = PubSubToolset(
    credentials_config=PubSubCredentialsConfig(credentials=creds),
    pubsub_tool_config=PubSubToolConfig(project_id="my-gcp-project"),
    # Expose all 3 tools
)

agent = LlmAgent(
    name="order_processor",
    model="gemini-2.0-flash",
    instruction=(
        "Process order events from Pub/Sub. "
        "Call pull_messages on projects/my-gcp-project/subscriptions/orders-sub "
        "with max_messages=5. For each message, summarise the order details, "
        "then call acknowledge_messages with the ack_ids to confirm processing."
    ),
    tools=[toolset],
)
```

### Example 3 — ordered publish with attributes

```python
# The agent calls publish_message; here is what the LLM args resolve to:
import json

# The tool is called with these args by the model:
publish_args = {
    "topic_name": "projects/my-project/topics/transactions",
    "message": json.dumps({"txn_id": "T-999", "amount": 150.00, "currency": "GBP"}),
    "attributes": {"region": "eu-west-1", "priority": "high"},
    "ordering_key": "customer-42",   # ensures ordered delivery per customer
}
# The tool automatically uses publishing with enable_message_ordering=True
# when ordering_key is non-empty.
```

### Message decoding note

`pull_messages` tries to UTF-8 decode each message's `data` field. If UTF-8 decoding fails (binary payload), it falls back to base64-encoding the bytes as an ASCII string. Always check the `data` field type before parsing as JSON.

---

## 4 · `VertexAiRagRetrieval`

**Source:** `google.adk.tools.retrieval.vertex_ai_rag_retrieval`

`VertexAiRagRetrieval` bridges Vertex AI RAG corpora into ADK agents. For Gemini 2+ models it uses the **model-native** RAG retrieval path (`types.Retrieval(vertex_rag_store=...)` injected into the LLM request config) — no function call round-trip, no token overhead. For older models it falls back to the standard `run_async` function-call path using `vertexai.rag.retrieval_query`.

### Constructor (source-verified)

```python
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval

VertexAiRagRetrieval(
    *,
    name: str,
    description: str,
    rag_corpora: list[str] | None = None,          # corpus resource names
    rag_resources: list[rag.RagResource] | None = None,  # fine-grained resource specs
    similarity_top_k: int | None = None,           # number of top chunks to return
    vector_distance_threshold: float | None = None, # filter out distant matches
)
```

Use either `rag_corpora` (simple list of corpus names) or `rag_resources` (fine-grained `RagResource` objects that let you pin specific files or corpora). Not both.

### Example 1 — simple corpus lookup

```python
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from google.adk.agents.llm_agent import LlmAgent

rag_tool = VertexAiRagRetrieval(
    name="search_company_docs",
    description=(
        "Search the company knowledge base for policy documents, "
        "onboarding guides, and technical specifications."
    ),
    rag_corpora=["projects/123456/locations/us-central1/ragCorpora/my-corpus"],
    similarity_top_k=5,
    vector_distance_threshold=0.5,
)

agent = LlmAgent(
    name="hr_assistant",
    model="gemini-2.0-flash",
    instruction=(
        "Answer employee questions about company policies and procedures. "
        "Always search the knowledge base before answering."
    ),
    tools=[rag_tool],
)
```

### Example 2 — multi-corpus with fine-grained `RagResource`

```python
import vertexai
from vertexai.preview import rag
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from google.adk.agents.llm_agent import LlmAgent

vertexai.init(project="my-project", location="us-central1")

# Search two corpora with different resource filters
rag_tool = VertexAiRagRetrieval(
    name="search_docs",
    description="Search product and support documentation.",
    rag_resources=[
        rag.RagResource(
            rag_corpus="projects/my-project/locations/us-central1/ragCorpora/product-docs",
        ),
        rag.RagResource(
            rag_corpus="projects/my-project/locations/us-central1/ragCorpora/support-tickets",
            rag_file_ids=["file_abc123"],  # only search a specific file
        ),
    ],
    similarity_top_k=3,
)

agent = LlmAgent(
    name="support_agent",
    model="gemini-2.0-flash",
    instruction="Help users solve product issues using the documentation and past support tickets.",
    tools=[rag_tool],
)
```

### Example 3 — combining RAG with function tools

```python
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from google.adk.agents.llm_agent import LlmAgent
from datetime import datetime

def get_current_date() -> str:
    """Returns today's date as ISO 8601."""
    return datetime.today().date().isoformat()

rag_tool = VertexAiRagRetrieval(
    name="search_regulations",
    description="Search current financial regulations and compliance documents.",
    rag_corpora=["projects/my-project/locations/us-central1/ragCorpora/regulations"],
    similarity_top_k=4,
)

agent = LlmAgent(
    name="compliance_agent",
    model="gemini-2.0-flash",
    instruction=(
        "Answer compliance questions. "
        "Use search_regulations to find relevant rules. "
        "Use get_current_date to check if time-sensitive rules apply."
    ),
    tools=[rag_tool, get_current_date],
)
```

### Gemini version behaviour

| Model | RAG path |
|---|---|
| Gemini 2.x / EAP | Native `types.Retrieval` injected into config — model fetches chunks internally |
| Gemini 1.x | `run_async` → `vertexai.rag.retrieval_query` function call |

Always use a Gemini 2+ model to get the native path (lower latency, grounding metadata in the response).

---

## 5 · `RetryConfig`

**Source:** `google.adk.workflow._retry_config`, `google.adk.workflow.utils._retry_utils`

`RetryConfig` is a Pydantic model that configures exponential-backoff retries for Workflow nodes. Assign it to a node via the `@node(retry_config=...)` decorator or the `add_node(retry_config=...)` method. The retry logic lives in `_retry_utils._should_retry_node` and `_get_retry_delay`.

### Constructor (source-verified)

```python
from google.adk.workflow._retry_config import RetryConfig

RetryConfig(
    max_attempts: int | None = None,    # total attempts incl. original; default=5; 0 or 1 = no retry
    initial_delay: float | None = None, # seconds before first retry; default=1.0
    max_delay: float | None = None,     # cap on delay; default=60.0 seconds
    backoff_factor: float | None = None,# delay multiplier per attempt; default=2.0
    jitter: float | None = None,        # randomness ±jitter*delay; default=1.0; 0.0=no jitter
    exceptions: list[str | type[BaseException]] | None = None,
                                        # None = retry on all; list = only named exceptions
)
```

`exceptions` accepts both strings (`["httpx.ConnectError"]`) and class objects (`[httpx.ConnectError]`). The validator normalises them to class name strings.

### Delay formula (source-verified)

```
delay = min(initial_delay * backoff_factor ** (attempt - 1), max_delay)
delay += random.uniform(-jitter * delay, jitter * delay)  # if jitter > 0
delay = max(0.0, delay)
```

`attempt` starts at 1 for the first retry (i.e. exponent is 0 for the first retry → `initial_delay` seconds).

### Example 1 — basic retry on network errors

```python
from google.adk.workflow._retry_config import RetryConfig
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._node import node
from google.adk.agents.context import Context
import httpx

retry = RetryConfig(
    max_attempts=4,        # 1 original + 3 retries
    initial_delay=1.0,
    backoff_factor=2.0,    # 1s, 2s, 4s
    max_delay=10.0,
    jitter=0.5,            # ±50% randomness
    exceptions=[httpx.ConnectError, httpx.TimeoutException],
)

@node(retry_config=retry)
async def fetch_api_data(ctx: Context, endpoint: str) -> dict:
    """Fetch data from an external API with automatic retry."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(endpoint)
        resp.raise_for_status()
        return resp.json()

wf = Workflow(name="fetch_pipeline")
wf.add_node(fetch_api_data)
```

### Example 2 — retry all exceptions, conservative backoff

```python
from google.adk.workflow._retry_config import RetryConfig
from google.adk.workflow._node import node
from google.adk.agents.context import Context

conservative_retry = RetryConfig(
    max_attempts=5,
    initial_delay=2.0,
    backoff_factor=1.5,
    max_delay=30.0,
    jitter=0.0,       # deterministic delays: 2s, 3s, 4.5s, 6.75s
    # exceptions=None → retry on ANY exception
)

@node(retry_config=conservative_retry)
async def write_to_database(ctx: Context, records: list[dict]) -> dict:
    """Write records to database; retry on any transient error."""
    # ... database write logic ...
    return {"written": len(records)}
```

### Example 3 — per-node retry via `add_node`

```python
from google.adk.workflow._retry_config import RetryConfig
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._node import node
from google.adk.agents.context import Context

@node
async def unreliable_step(ctx: Context) -> str:
    """This step sometimes fails."""
    import random
    if random.random() < 0.3:
        raise RuntimeError("Transient failure")
    return "success"

wf = Workflow(name="resilient_pipeline")
wf.add_node(
    unreliable_step,
    retry_config=RetryConfig(
        max_attempts=3,
        initial_delay=0.5,
        backoff_factor=2.0,
        exceptions=["RuntimeError"],
    ),
)
```

### Example 4 — disabling retries for a critical node

```python
from google.adk.workflow._retry_config import RetryConfig
from google.adk.workflow._node import node
from google.adk.agents.context import Context

no_retry = RetryConfig(max_attempts=1)   # 0 or 1 = no retries

@node(retry_config=no_retry)
async def send_payment(ctx: Context, amount: float, account: str) -> dict:
    """Idempotency-unsafe — never retry a payment."""
    # ... payment API call ...
    return {"txn_id": "TXN-123", "status": "SENT"}
```

---

## 6 · `RemoteA2aAgent`

**Source:** `google.adk.agents.remote_a2a_agent`

`RemoteA2aAgent` is a `BaseAgent` subclass that wraps a remote A2A-compatible service (e.g. another ADK app, a LangGraph server, or any A2A-compliant agent) and makes it callable from within an ADK graph. It resolves the remote agent's `AgentCard`, builds an A2A client, and translates ADK `Event` objects to/from A2A protocol messages.

### Constructor (source-verified)

```python
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

RemoteA2aAgent(
    name: str,
    agent_card: AgentCard | str,   # AgentCard object, HTTPS URL, or file path
    timeout: float = 30.0,         # HTTP timeout in seconds
)
```

`agent_card` supports three forms:
- **`AgentCard` object** — passed directly, no resolution needed.
- **URL string** — fetched from `{url}/.well-known/agent.json` (or the literal URL if it ends with `.json`).
- **File path string** — read and parsed as JSON.

### Example 1 — agent calling a remote A2A service by URL

```python
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.agents.llm_agent import LlmAgent

# The remote agent runs at https://translation-agent.example.com
translation_agent = RemoteA2aAgent(
    name="translation_service",
    agent_card="https://translation-agent.example.com",  # resolved via /.well-known/agent.json
    timeout=15.0,
)

orchestrator = LlmAgent(
    name="orchestrator",
    model="gemini-2.0-flash",
    instruction=(
        "You are an orchestrator. "
        "When the user needs text translated, delegate to translation_service."
    ),
    sub_agents=[translation_agent],
)
```

### Example 2 — using a pre-built `AgentCard`

```python
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

card = AgentCard(
    name="calculator",
    description="Performs arithmetic calculations.",
    url="https://calc-agent.internal.example.com",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="calculate",
            name="Calculate",
            description="Evaluate a mathematical expression.",
        )
    ],
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
)

calc_agent = RemoteA2aAgent(
    name="calculator",
    agent_card=card,
    timeout=5.0,
)
```

### Example 3 — loading an AgentCard from a local JSON file

```python
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

# Write agent card JSON to disk for local development / testing
import json, pathlib
card_path = pathlib.Path("/tmp/search_agent_card.json")
card_path.write_text(json.dumps({
    "name": "search_agent",
    "description": "Web search specialist.",
    "url": "https://search.example.com",
    "version": "2.0.0",
    "capabilities": {"streaming": True},
    "skills": [{"id": "search", "name": "Search", "description": "Search the web."}],
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain"],
}))

search_agent = RemoteA2aAgent(
    name="search_agent",
    agent_card=str(card_path),  # file path string
    timeout=20.0,
)
```

### Example 4 — multi-agent pipeline with remote specialists

```python
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.llm_agent import LlmAgent

# Specialists running as separate A2A services
summariser = RemoteA2aAgent(
    name="summariser",
    agent_card="https://summariser.example.com",
    timeout=30.0,
)
sentiment_analyser = RemoteA2aAgent(
    name="sentiment",
    agent_card="https://sentiment.example.com",
    timeout=10.0,
)

# Sequential pipeline: summarise → analyse sentiment
pipeline = SequentialAgent(
    name="document_pipeline",
    sub_agents=[summariser, sentiment_analyser],
)

orchestrator = LlmAgent(
    name="orchestrator",
    model="gemini-2.0-flash",
    instruction="Process documents by summarising them and analysing sentiment.",
    sub_agents=[pipeline],
)
```

### Error handling

`RemoteA2aAgent` raises `AgentCardResolutionError` (importable from the same module) when the AgentCard cannot be resolved:

```python
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent, AgentCardResolutionError

try:
    agent = RemoteA2aAgent(
        name="unknown",
        agent_card="https://does-not-exist.example.com",
    )
    # Resolution is lazy — happens on first invocation
except AgentCardResolutionError as e:
    print(f"Could not resolve agent card: {e}")
```

---

## 7 · `LiveRequest` + `LiveRequestQueue`

**Source:** `google.adk.agents.live_request_queue`

`LiveRequestQueue` is a thin asyncio-based queue that drives bidirectional streaming ("live") sessions. You push `LiveRequest` objects into it; the ADK runner reads from it and forwards them to the model's live API. Live mode enables real-time audio/video streaming, voice interruption, and activity-start/end signalling — capabilities not available in standard turn-by-turn mode.

### Classes (source-verified)

```python
from google.adk.agents.live_request_queue import LiveRequest, LiveRequestQueue

# LiveRequest fields (priority order: activity_start > activity_end > blob > content):
LiveRequest(
    content: types.Content | None = None,       # turn-by-turn text/function response
    blob: types.Blob | None = None,             # realtime audio/video bytes
    activity_start: types.ActivityStart | None = None,  # begin voice utterance
    activity_end: types.ActivityEnd | None = None,      # end voice utterance
    close: bool = False,                        # drain and close the session
)

# LiveRequestQueue methods:
queue = LiveRequestQueue()
queue.send_content(content: types.Content)
queue.send_realtime(blob: types.Blob)
queue.send_activity_start()
queue.send_activity_end()
queue.send(req: LiveRequest)
queue.close()

# Read by the ADK runner:
req: LiveRequest = await queue.get()
```

### Example 1 — text-based live session

```python
import asyncio
from google.genai import types
from google.adk.agents.live_request_queue import LiveRequestQueue, LiveRequest
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import InMemoryRunner

agent = LlmAgent(
    name="live_assistant",
    model="gemini-2.0-flash-live",
    instruction="You are a helpful real-time assistant.",
)

runner = InMemoryRunner(agent=agent, app_name="live_demo")

async def run_live_session():
    session = await runner.session_service.create_session(
        app_name="live_demo", user_id="user-1"
    )
    live_queue = LiveRequestQueue()

    async def feed_input():
        # Send a text message
        live_queue.send_content(
            types.Content(role="user", parts=[types.Part(text="Hello! What can you do?")])
        )
        await asyncio.sleep(0.1)
        live_queue.send_content(
            types.Content(role="user", parts=[types.Part(text="Tell me a short joke.")])
        )
        await asyncio.sleep(0.1)
        live_queue.close()

    async def read_output(events):
        async for event in events:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"[{event.author}]: {part.text}")

    events = runner.run_live(
        user_id="user-1",
        session_id=session.id,
        live_request_queue=live_queue,
    )

    await asyncio.gather(feed_input(), read_output(events))

asyncio.run(run_live_session())
```

### Example 2 — realtime audio streaming (voice input)

```python
import asyncio
from google.genai import types
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import InMemoryRunner

agent = LlmAgent(
    name="voice_assistant",
    model="gemini-2.0-flash-live",
    instruction="You are a voice assistant. Respond conversationally.",
)

runner = InMemoryRunner(agent=agent, app_name="voice_demo")

async def stream_audio(live_queue: LiveRequestQueue, audio_chunks: list[bytes]):
    """Push PCM audio chunks into the live session."""
    live_queue.send_activity_start()
    for chunk in audio_chunks:
        live_queue.send_realtime(
            types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
        )
        await asyncio.sleep(0.02)  # 20ms chunks
    live_queue.send_activity_end()
    live_queue.close()
```

### Example 3 — interleaved text and function responses in live mode

```python
from google.genai import types
from google.adk.agents.live_request_queue import LiveRequest, LiveRequestQueue

# After the model emits a function call, push the function response back
def push_function_response(
    queue: LiveRequestQueue,
    function_call_id: str,
    result: dict,
):
    queue.send_content(
        types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=function_call_id,
                        name="get_weather",
                        response=result,
                    )
                )
            ],
        )
    )
```

### Example 4 — manual `send()` for full `LiveRequest` control

```python
from google.adk.agents.live_request_queue import LiveRequest, LiveRequestQueue
from google.genai import types

queue = LiveRequestQueue()

# Equivalent to queue.send_activity_start()
queue.send(LiveRequest(activity_start=types.ActivityStart()))

# Interleave a function response (content wins over blob in priority)
queue.send(LiveRequest(
    content=types.Content(
        role="user",
        parts=[types.Part(function_response=types.FunctionResponse(
            id="call_01", name="lookup_price", response={"price": 42.0}
        ))],
    )
))

# Equivalent to queue.close()
queue.send(LiveRequest(close=True))
```

---

## 8 · `AuthHandler` + `AuthSchemes`

**Sources:** `google.adk.auth.auth_handler`, `google.adk.auth.auth_schemes`

`AuthHandler` orchestrates the OAuth2 / OpenID Connect flow on behalf of an agent tool. It generates an auth URI for the user to visit, stores the exchanged credential in session state, and on subsequent calls retrieves it — all keyed by `auth_config.credential_key`. `AuthSchemes` defines the type hierarchy: `SecurityScheme` (from OpenAPI 3.0), `OpenIdConnectWithConfig`, and `CustomAuthScheme`.

### `AuthHandler` (source-verified)

```python
from google.adk.auth.auth_handler import AuthHandler
from google.adk.auth.auth_tool import AuthConfig

handler = AuthHandler(auth_config: AuthConfig)

# Key methods:
handler.generate_auth_request()               # → AuthConfig with auth_uri for user consent
handler.generate_auth_uri()                   # → AuthCredential with oauth2.auth_uri populated
await handler.exchange_auth_token()            # → AuthCredential with access_token
await handler.parse_and_store_auth_response(state)  # store credential in session state
handler.get_auth_response(state)              # → AuthCredential | None from session state
```

`AuthHandler` is typically used by ADK internals (the `AuthPreprocessor` and `RestApiTool`), but you can use it directly in custom tools that need to orchestrate OAuth2 flows.

### `AuthSchemes` type hierarchy (source-verified)

```python
from google.adk.auth.auth_schemes import (
    AuthScheme,            # Union type = SecurityScheme | OpenIdConnectWithConfig | CustomAuthScheme
    AuthSchemeType,        # re-export of SecuritySchemeType from OpenAPI 3.0
    OAuthGrantType,        # Enum: CLIENT_CREDENTIALS | AUTHORIZATION_CODE | IMPLICIT | PASSWORD
    OpenIdConnectWithConfig,  # flat OIDC config with discovery endpoints
    CustomAuthScheme,      # base for custom schemes
    ExtendedOAuth2,        # OAuth2 + issuer_url for auto-discovery (@experimental)
)
```

### `OpenIdConnectWithConfig` fields (source-verified)

```python
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig

OpenIdConnectWithConfig(
    authorization_endpoint: str,   # REQUIRED
    token_endpoint: str,           # REQUIRED
    userinfo_endpoint: str | None = None,
    revocation_endpoint: str | None = None,
    token_endpoint_auth_methods_supported: list[str] | None = None,
    grant_types_supported: list[str] | None = None,
    scopes: list[str] | None = None,
)
```

### Example 1 — detecting OAuth2 grant type from a flow

```python
from fastapi.openapi.models import OAuthFlows, OAuthFlowAuthorizationCode
from google.adk.auth.auth_schemes import OAuthGrantType

flows = OAuthFlows(
    authorizationCode=OAuthFlowAuthorizationCode(
        authorizationUrl="https://accounts.google.com/o/oauth2/auth",
        tokenUrl="https://oauth2.googleapis.com/token",
        scopes={"openid": "OpenID scope", "email": "Email scope"},
    )
)
print(OAuthGrantType.from_flow(flows))  # OAuthGrantType.AUTHORIZATION_CODE
```

### Example 2 — building an OpenID Connect config manually

```python
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig

oidc_scheme = OpenIdConnectWithConfig(
    authorization_endpoint="https://accounts.google.com/o/oauth2/auth",
    token_endpoint="https://oauth2.googleapis.com/token",
    userinfo_endpoint="https://openidconnect.googleapis.com/v1/userinfo",
    revocation_endpoint="https://oauth2.googleapis.com/revoke",
    scopes=["openid", "email", "profile"],
)
```

### Example 3 — using `AuthHandler` in a custom OAuth2 tool

```python
import asyncio
from google.adk.auth.auth_handler import AuthHandler
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.auth.auth_credential import AuthCredential, OAuth2Auth
from google.adk.tools.tool_context import ToolContext


OIDC_SCHEME = OpenIdConnectWithConfig(
    authorization_endpoint="https://auth.example.com/authorize",
    token_endpoint="https://auth.example.com/token",
    scopes=["openid", "profile", "email"],
)

_AUTH_KEY = "example_oidc"


async def get_user_profile(
    tool_context: ToolContext,
    user_id: str,
) -> dict:
    """Return the user's profile from the identity provider.

    Initiates an OAuth2 consent flow on first call; uses cached token thereafter.
    """
    auth_config = AuthConfig(
        auth_scheme=OIDC_SCHEME,
        raw_auth_credential=AuthCredential(
            auth_type="oauth2",
            oauth2=OAuth2Auth(
                client_id="YOUR_CLIENT_ID",
                client_secret="YOUR_CLIENT_SECRET",
                redirect_uri="https://yourapp.example.com/callback",
            ),
        ),
        credential_key=_AUTH_KEY,
    )

    handler = AuthHandler(auth_config=auth_config)

    # Check for an already-stored credential
    stored = handler.get_auth_response(tool_context.state)
    if stored and stored.oauth2 and stored.oauth2.access_token:
        token = stored.oauth2.access_token
    else:
        # No stored credential → request user consent
        auth_request = handler.generate_auth_request()
        # In a real app, return auth_request.exchanged_auth_credential.oauth2.auth_uri
        # to the user so they can sign in. The callback will populate the state.
        return {
            "status": "AUTH_REQUIRED",
            "auth_uri": (
                auth_request.exchanged_auth_credential.oauth2.auth_uri
                if auth_request.exchanged_auth_credential
                else None
            ),
        }

    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://auth.example.com/userinfo",
            headers={"Authorization": f"Bearer {token}"},
        )
        return resp.json()
```

### Example 4 — custom auth scheme for API-key-in-header

```python
from google.adk.auth.auth_schemes import CustomAuthScheme

class ApiKeyHeaderScheme(CustomAuthScheme):
    """Custom scheme that passes an API key in a specific header."""
    type_: str = "apiKey"
    header_name: str = "X-Custom-Api-Key"

scheme = ApiKeyHeaderScheme(header_name="X-Internal-Api-Key")
print(scheme.type_)         # "apiKey"
print(scheme.header_name)   # "X-Internal-Api-Key"
```

---

## 9 · `SkillToolset`

**Source:** `google.adk.tools.skill_toolset`

`SkillToolset` gives agents the ability to **discover and invoke agent "skills"** — named, versioned capabilities registered in a `SkillRegistry`. Skills can be invoked by description lookup, direct name/version reference, or — if a code executor is provided — via Python script execution. It is the runtime side of the ADK Skills system.

### Constructor (source-verified)

```python
from google.adk.tools.skill_toolset import SkillToolset

SkillToolset(
    *,
    skill_registry: SkillRegistry,
    tool_filter: ToolPredicate | list[str] | None = None,
    code_executor: BaseCodeExecutor | None = None,   # enables script execution
    script_timeout: int = 300,                        # seconds for script execution
    tool_name_prefix: str | None = None,
)
```

`SkillToolset` generates several internal tools: a `list_skills` tool (discover what skills exist), a `get_skill_details` tool (inspect a skill's API), a `use_skill` tool (invoke a skill directly), and — when `code_executor` is provided — a `run_skill_script` tool (execute Python that uses skills).

### Example 1 — skill discovery and invocation

```python
from google.adk.skills import SkillRegistry
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.agents.llm_agent import LlmAgent

# Create and populate a skill registry
registry = SkillRegistry()

# Register a skill
@registry.skill(name="summarise_text", description="Summarise a block of text into bullet points.")
def summarise_text(text: str, max_bullets: int = 5) -> str:
    """Summarise text into bullet points."""
    sentences = text.split(". ")[:max_bullets]
    return "\n".join(f"• {s.strip()}" for s in sentences if s.strip())

toolset = SkillToolset(skill_registry=registry)

agent = LlmAgent(
    name="skill_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You help users by discovering and using registered skills. "
        "Start by listing available skills, then use them as needed."
    ),
    tools=[toolset],
)
```

### Example 2 — skill toolset with code executor (script mode)

```python
from google.adk.skills import SkillRegistry
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.code_executors.local_code_executor import LocalCodeExecutor
from google.adk.agents.llm_agent import LlmAgent

registry = SkillRegistry()

@registry.skill(name="compute_stats", description="Compute basic statistics on a list of numbers.")
def compute_stats(numbers: list[float]) -> dict:
    """Return mean, min, max, and count for a list of numbers."""
    if not numbers:
        return {"error": "empty list"}
    return {
        "count": len(numbers),
        "mean": sum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }

executor = LocalCodeExecutor()

toolset = SkillToolset(
    skill_registry=registry,
    code_executor=executor,
    script_timeout=30,    # 30-second cap on script execution
)

agent = LlmAgent(
    name="data_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You analyse data using registered skills. "
        "Use run_skill_script to write Python that chains multiple skills together."
    ),
    tools=[toolset],
)
```

### Example 3 — prefixed tool names to avoid collisions

```python
from google.adk.skills import SkillRegistry
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.agents.llm_agent import LlmAgent

registry = SkillRegistry()

toolset = SkillToolset(
    skill_registry=registry,
    tool_name_prefix="myapp",  # tools become myapp_list_skills, myapp_use_skill, …
)

agent = LlmAgent(
    name="prefixed_agent",
    model="gemini-2.0-flash",
    instruction="Use the myapp_ prefixed tools to discover and invoke skills.",
    tools=[toolset],
)
```

### Example 4 — filtering which skill tools are exposed

```python
from google.adk.skills import SkillRegistry
from google.adk.tools.skill_toolset import SkillToolset

registry = SkillRegistry()

# Expose only discovery tools; disable direct invocation and scripting
toolset = SkillToolset(
    skill_registry=registry,
    tool_filter=["list_skills", "get_skill_details"],
)
```

---

## 10 · `GoogleSearchTool` + `UrlContextTool` + `GoogleMapsGroundingTool`

**Sources:** `google.adk.tools.google_search_tool`, `google.adk.tools.url_context_tool`, `google.adk.tools.google_maps_grounding_tool`

These three tools are **model-native built-ins** — they work by injecting entries into the `GenerateContentConfig.tools` list sent to the Gemini API rather than executing local code. The model handles them internally, so they have no `run_async` body and impose no round-trip latency for tool calling.

### `GoogleSearchTool` (source-verified)

```python
from google.adk.tools.google_search_tool import GoogleSearchTool, google_search

GoogleSearchTool(
    *,
    bypass_multi_tools_limit: bool = False,  # allow use alongside other tools on Gemini 2+
    model: str | None = None,                # override the LLM request model
)

# Pre-instantiated singleton:
from google.adk.tools.google_search_tool import google_search
```

**Model compatibility:**
- **Gemini 1.x**: Injects `types.Tool(google_search_retrieval=types.GoogleSearchRetrieval())`. **Cannot** be combined with other tools — raises `ValueError` if any other tools are already present.
- **Gemini 2.x**: Injects `types.Tool(google_search=types.GoogleSearch())`. Can be combined with other tools (including function tools) when `bypass_multi_tools_limit=False`.

### `UrlContextTool` (source-verified)

```python
from google.adk.tools.url_context_tool import UrlContextTool, url_context

UrlContextTool()  # no configuration parameters

# Pre-instantiated singleton:
from google.adk.tools.url_context_tool import url_context
```

Injects `types.Tool(url_context=types.UrlContext())`. The model fetches and reads URL content when the user (or another tool) provides a URL. Requires **Gemini 2.x or EAP** — raises `ValueError` on Gemini 1.x.

### `GoogleMapsGroundingTool` (source-verified)

```python
from google.adk.tools.google_maps_grounding_tool import GoogleMapsGroundingTool, google_maps_grounding

GoogleMapsGroundingTool()  # no configuration parameters

# Pre-instantiated singleton:
from google.adk.tools.google_maps_grounding_tool import google_maps_grounding
```

Injects `types.Tool(google_maps=types.GoogleMaps())`. Grounds agent responses with real-time Google Maps data. Requires **Gemini 2.x via the Vertex AI API** (`GOOGLE_GENAI_USE_VERTEXAI=TRUE`) — raises `ValueError` otherwise.

### Example 1 — Google Search with a Gemini 2 agent

```python
from google.adk.tools.google_search_tool import google_search
from google.adk.agents.llm_agent import LlmAgent

agent = LlmAgent(
    name="research_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You research topics using Google Search. "
        "Cite your sources in every answer."
    ),
    tools=[google_search],
)
```

### Example 2 — Google Search + function tools on Gemini 2 (`bypass_multi_tools_limit`)

On Gemini 2+ the multi-tool restriction is lifted by default, but if your model variant still enforces it you can opt out:

```python
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.agents.llm_agent import LlmAgent
from datetime import datetime

def get_current_date() -> str:
    """Returns today's date."""
    return datetime.today().date().isoformat()

search = GoogleSearchTool(bypass_multi_tools_limit=True)

agent = LlmAgent(
    name="date_search_agent",
    model="gemini-2.0-flash",
    instruction="Answer questions using search and the current date.",
    tools=[search, get_current_date],
)
```

### Example 3 — URL context: reading a live web page

```python
from google.adk.tools.url_context_tool import url_context
from google.adk.agents.llm_agent import LlmAgent

agent = LlmAgent(
    name="url_reader",
    model="gemini-2.0-flash",
    instruction=(
        "When the user provides a URL, fetch and summarise the content at that URL. "
        "Quote specific sections when answering questions about the page."
    ),
    tools=[url_context],
)

# Usage: user sends "Summarise https://example.com/blog/post-1"
# The model fetches the page and uses it as grounding context.
```

### Example 4 — Google Maps grounding (Vertex AI only)

```python
import os
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"

from google.adk.tools.google_maps_grounding_tool import google_maps_grounding
from google.adk.agents.llm_agent import LlmAgent

agent = LlmAgent(
    name="location_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You help users find places and get directions. "
        "Always ground your answers in real Google Maps data."
    ),
    tools=[google_maps_grounding],
)
```

### Example 5 — combining all three built-in tools

```python
from google.adk.tools.google_search_tool import google_search
from google.adk.tools.url_context_tool import url_context
from google.adk.tools.google_maps_grounding_tool import google_maps_grounding
from google.adk.agents.llm_agent import LlmAgent
import os

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"

agent = LlmAgent(
    name="super_grounded_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You are a research assistant with full grounding capabilities. "
        "Use Google Search for current events, URL context for reading links, "
        "and Google Maps for location and navigation questions."
    ),
    tools=[google_search, url_context, google_maps_grounding],
)
```

### Model-version compatibility matrix

| Tool | Gemini 1.x | Gemini 2.x (AI Studio) | Gemini 2.x (Vertex AI) |
|---|---|---|---|
| `GoogleSearchTool` | ✓ (solo only) | ✓ | ✓ |
| `UrlContextTool` | ✗ | ✓ | ✓ |
| `GoogleMapsGroundingTool` | ✗ | ✗ | ✓ |

---

## Quick reference

| Class | Module | Key feature |
|---|---|---|
| `LongRunningFunctionTool` | `google.adk.tools.long_running_tool` | Marks a tool as async/background; adds LLM guidance note |
| `OpenAPIToolset` | `google.adk.tools.openapi_tool` | Parses OpenAPI 3.x specs → `RestApiTool` instances |
| `RestApiTool` | `google.adk.tools.openapi_tool` | HTTP client tool generated from a single OpenAPI operation |
| `PubSubToolset` | `google.adk.tools.pubsub.pubsub_toolset` | 3 Pub/Sub tools: publish, pull, acknowledge (`@experimental`) |
| `PubSubToolConfig` | `google.adk.tools.pubsub.config` | `project_id` config for all Pub/Sub tools (`@experimental`) |
| `VertexAiRagRetrieval` | `google.adk.tools.retrieval.vertex_ai_rag_retrieval` | Native Gemini 2 RAG retrieval; falls back to function call for older models |
| `RetryConfig` | `google.adk.workflow._retry_config` | Exponential backoff + jitter for Workflow nodes |
| `RemoteA2aAgent` | `google.adk.agents.remote_a2a_agent` | Wraps a remote A2A service as a local `BaseAgent` |
| `AgentCardResolutionError` | `google.adk.agents.remote_a2a_agent` | Raised when `RemoteA2aAgent` cannot resolve an `AgentCard` |
| `LiveRequest` | `google.adk.agents.live_request_queue` | Wrapper for a single live session input (text / audio blob / activity signal) |
| `LiveRequestQueue` | `google.adk.agents.live_request_queue` | asyncio queue that drives bidirectional streaming live agents |
| `AuthHandler` | `google.adk.auth.auth_handler` | Orchestrates OAuth2 / OIDC flow; stores credentials in session state |
| `AuthScheme` | `google.adk.auth.auth_schemes` | Union type: `SecurityScheme \| OpenIdConnectWithConfig \| CustomAuthScheme` |
| `OpenIdConnectWithConfig` | `google.adk.auth.auth_schemes` | Flat OIDC config (auth + token + userinfo endpoints) |
| `OAuthGrantType` | `google.adk.auth.auth_schemes` | Enum: CLIENT_CREDENTIALS \| AUTHORIZATION_CODE \| IMPLICIT \| PASSWORD |
| `CustomAuthScheme` | `google.adk.auth.auth_schemes` | Base for custom auth schemes not in OpenAPI 3.0 |
| `SkillToolset` | `google.adk.tools.skill_toolset` | Discover and invoke `SkillRegistry` skills; optional script execution |
| `GoogleSearchTool` | `google.adk.tools.google_search_tool` | Model-native Google Search (Gemini 1.x solo; Gemini 2.x combinable) |
| `UrlContextTool` | `google.adk.tools.url_context_tool` | Model-native URL content fetching (Gemini 2.x+) |
| `GoogleMapsGroundingTool` | `google.adk.tools.google_maps_grounding_tool` | Model-native Google Maps grounding (Vertex AI Gemini 2.x only) |

---
title: "Class deep dives — volume 15 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.2.0 classes: ReflectAndRetryToolPlugin (self-healing tool errors, configurable TrackingScope, custom extract_error_from_result), OpenAPIToolset + RestApiTool (REST API generation from OpenAPI specs; auth schemes; ssl_verify; header_provider; tool_name_prefix), ExampleTool + BaseExampleProvider (few-shot example injection via list or dynamic provider; VertexAiExampleStore), EnvironmentSimulationConfig + ToolSimulationConfig + InjectionConfig (tool environment simulation; probability-based injection; error/latency/response injection; TOOL_SPEC mock strategy), LlmBackedUserSimulator + LlmBackedUserSimulatorConfig + UserPersona (LLM-backed user simulation for evaluation; custom personas; conversation scenarios), BaseCodeExecutor + VertexAiCodeExecutor + UnsafeLocalCodeExecutor (code execution backends; stateful sessions; error retry; output files), ComputerUseTool + BaseComputer + ComputerUseToolset (computer use GUI automation; custom computer implementation; prepare hook), GEPARootAgentPromptOptimizer + SimplePromptOptimizer (GEPA/iterative prompt optimization; Sampler interface; OptimizerResult), SkillToolset + Skill + Frontmatter (skills system; dynamic tool loading; adk_additional_tools; SkillRegistry; RunSkillScriptTool), AutoTracingPlugin + DebugLoggingPlugin (OTel auto-instrumentation; YAML debug dump; ContextFilterPlugin)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 15"
  order: 80
---

Source-verified against **google-adk==2.2.0** (installed from PyPI, June 2026). Every field name, signature, and code example is drawn from the installed package source at `/usr/local/lib/python3.11/dist-packages/google/adk/`.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `ReflectAndRetryToolPlugin` + `TrackingScope` | `google.adk.plugins.reflect_retry_tool_plugin` | `@experimental` |
| 2 | `OpenAPIToolset` + `RestApiTool` | `google.adk.tools.openapi_tool.openapi_spec_parser` | Stable |
| 3 | `ExampleTool` + `BaseExampleProvider` | `google.adk.tools.example_tool`, `google.adk.examples` | Stable |
| 4 | `EnvironmentSimulationConfig` + `ToolSimulationConfig` + `InjectionConfig` | `google.adk.tools.environment_simulation` | `@experimental` |
| 5 | `LlmBackedUserSimulator` + `LlmBackedUserSimulatorConfig` + `UserPersona` | `google.adk.evaluation.simulation` | `@experimental` |
| 6 | `BaseCodeExecutor` + `VertexAiCodeExecutor` + `UnsafeLocalCodeExecutor` | `google.adk.code_executors` | Stable |
| 7 | `ComputerUseTool` + `BaseComputer` + `ComputerUseToolset` | `google.adk.tools.computer_use` | `@experimental` |
| 8 | `GEPARootAgentPromptOptimizer` + `SimplePromptOptimizer` | `google.adk.optimization` | `@experimental` |
| 9 | `SkillToolset` + `Skill` + `Frontmatter` | `google.adk.tools.skill_toolset`, `google.adk.skills.models` | Stable |
| 10 | `AutoTracingPlugin` + `DebugLoggingPlugin` | `google.adk.plugins` | Stable |

---

## 1 · `ReflectAndRetryToolPlugin` + `TrackingScope`

**Source:** `google.adk.plugins.reflect_retry_tool_plugin`

`ReflectAndRetryToolPlugin` intercepts tool failures and guides the LLM to reflect, correct its arguments, and retry — up to a configurable limit. It is **concurrency-safe** (uses an async lock) and tracks failure counts per-tool within a configurable scope.

### Constructor (source-verified)

```python
from google.adk.plugins.reflect_retry_tool_plugin import (
    ReflectAndRetryToolPlugin,
    TrackingScope,
)

ReflectAndRetryToolPlugin(
    name: str = "reflect_retry_tool_plugin",
    max_retries: int = 3,
    throw_exception_if_retry_exceeded: bool = True,
    tracking_scope: TrackingScope = TrackingScope.INVOCATION,
)
```

| Parameter | Default | Notes |
|---|---|---|
| `max_retries` | `3` | Max consecutive failures per tool before giving up. `0` = no retries. |
| `throw_exception_if_retry_exceeded` | `True` | If `False`, returns a guidance message to the LLM instead of raising. |
| `tracking_scope` | `TrackingScope.INVOCATION` | `INVOCATION` resets per-invocation; `GLOBAL` accumulates across all sessions. |

### `TrackingScope` enum (source-verified)

```python
class TrackingScope(Enum):
    INVOCATION = "invocation"   # failure count lives for one agent invocation
    GLOBAL = "global"           # failure count shared across all invocations
```

### How it works

The plugin implements two callbacks:

- **`after_tool_callback`** — intercepts every tool result. If the result is non-error, resets the failure counter for that tool. If `extract_error_from_result` returns a non-`None` value, treats it as an error.
- **`on_tool_error_callback`** — intercepts exceptions raised by tools.

Both funnel into `_handle_tool_error`, which atomically increments the per-tool counter and either:
1. Returns a structured reflection message (retry attempt 1…N)
2. Raises the original exception / returns an exhaustion message when `max_retries` is exceeded

### Overriding `extract_error_from_result` for soft failures

Tools that return `{"status": "error", ...}` without raising an exception won't be detected by default. Override to handle them:

```python
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from typing import Any, Optional

class ApiRetryPlugin(ReflectAndRetryToolPlugin):
    """Retries on API-level errors returned as dicts."""

    async def extract_error_from_result(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: Any,
    ) -> Optional[dict[str, Any]]:
        if isinstance(result, dict) and result.get("status") == "error":
            return result   # treat as failure → triggers retry logic
        return None         # success
```

### Example 1 — basic retry plugin on a runner

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin

call_count = 0

def flaky_database_lookup(record_id: str) -> dict:
    """Look up a record — fails the first two times."""
    global call_count
    call_count += 1
    if call_count < 3:
        raise ConnectionError(f"DB timeout on attempt {call_count}")
    return {"id": record_id, "name": "Alice", "status": "active"}

agent = LlmAgent(
    name="db_agent",
    model="gemini-2.5-flash",
    instruction="Look up records in the database.",
    tools=[flaky_database_lookup],
)

async def main():
    runner = InMemoryRunner(
        agent=agent,
        app_name="db_app",
        plugins=[ReflectAndRetryToolPlugin(max_retries=3)],
    )
    await runner.session_service.create_session(
        app_name="db_app", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Look up record with id='rec-42'.",
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 2 — global scope with soft-failure detection

```python
from google.adk.plugins.reflect_retry_tool_plugin import (
    ReflectAndRetryToolPlugin, TrackingScope,
)
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from typing import Any, Optional

class RateLimitRetryPlugin(ReflectAndRetryToolPlugin):
    """Retries when the API returns a 429 rate-limit response."""

    async def extract_error_from_result(
        self, *, tool: BaseTool, tool_args: dict, tool_context: ToolContext, result: Any
    ) -> Optional[dict]:
        if isinstance(result, dict) and result.get("error_code") == 429:
            return result
        return None

plugin = RateLimitRetryPlugin(
    max_retries=5,
    throw_exception_if_retry_exceeded=False,  # return guidance instead of raise
    tracking_scope=TrackingScope.GLOBAL,
)
```

### Example 3 — custom scoping (per-user retry isolation)

```python
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin
from google.adk.tools.tool_context import ToolContext

class PerUserRetryPlugin(ReflectAndRetryToolPlugin):
    """Each user gets their own independent failure counter."""

    def _get_scope_key(self, tool_context: ToolContext) -> str:
        # key = user_id + tool_name → fully isolated per user
        return tool_context.user_id
```

### Gotchas

- A success resets only **that tool's** counter — other tools' retry counts are unaffected.
- `max_retries=0` always raises (or returns guidance) immediately on the first failure.
- `TrackingScope.GLOBAL` keeps counters for the entire lifetime of the plugin object. Use this with care in long-running servers.
- The plugin's `after_tool_callback` skips results that already carry `response_type == REFLECT_AND_RETRY_RESPONSE_TYPE` to avoid double-processing.

---

## 2 · `OpenAPIToolset` + `RestApiTool`

**Source:** `google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset`, `.rest_api_tool`

`OpenAPIToolset` parses an OpenAPI 3.x spec (JSON or YAML) and auto-generates one `RestApiTool` per operation. Each `RestApiTool` is a fully-configured HTTP client that maps LLM function-call arguments to request parameters, handles auth, and returns the parsed response.

### `OpenAPIToolset` constructor (source-verified)

```python
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

OpenAPIToolset(
    spec_dict: Optional[dict] = None,
    spec_str: Optional[str] = None,
    spec_str_type: Literal["json", "yaml"] = "json",
    auth_scheme: Optional[AuthScheme] = None,
    auth_credential: Optional[AuthCredential] = None,
    credential_key: Optional[str] = None,
    tool_filter: Optional[Union[ToolPredicate, list[str]]] = None,
    tool_name_prefix: Optional[str] = None,
    ssl_verify: Optional[Union[bool, str, ssl.SSLContext]] = None,
    header_provider: Optional[Callable[[ReadonlyContext], dict[str, str]]] = None,
    httpx_client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,
    preserve_property_names: bool = False,
)
```

| Parameter | Notes |
|---|---|
| `spec_dict` / `spec_str` | Provide exactly one. `spec_str_type` controls JSON vs YAML parse. |
| `auth_scheme` + `auth_credential` | Applied to all generated tools. Use auth helpers from `google.adk.tools.openapi_tool.auth.auth_helpers`. |
| `credential_key` | Stable key for interactive auth caching across all tools in this toolset. |
| `tool_name_prefix` | Prepends a prefix to every generated tool name — avoids collisions when loading multiple specs. |
| `ssl_verify` | `True` (default), `False` (insecure), path to CA bundle, or `ssl.SSLContext`. Useful for enterprise TLS-intercepting proxies. |
| `header_provider` | `Callable[[ReadonlyContext], dict[str, str]]` — called per request; add correlation IDs, dynamic auth tokens, etc. |
| `httpx_client_factory` | Returns a fresh `httpx.AsyncClient` for each request — unlocks proxies, HTTP/2, request signing. |
| `preserve_property_names` | Default `False` converts parameter names to snake_case. Set `True` to keep camelCase as-is. |

### Example 1 — loading a JSON spec from a string

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

PETSTORE_SPEC = """
{
  "openapi": "3.0.0",
  "info": {"title": "Petstore", "version": "1.0.0"},
  "servers": [{"url": "https://petstore.example.com/api"}],
  "paths": {
    "/pets": {
      "get": {
        "operationId": "listPets",
        "summary": "List all pets",
        "parameters": [
          {"name": "limit", "in": "query", "schema": {"type": "integer"}}
        ],
        "responses": {"200": {"description": "A list of pets"}}
      }
    },
    "/pets/{petId}": {
      "get": {
        "operationId": "showPetById",
        "summary": "Get a pet by ID",
        "parameters": [
          {"name": "petId", "in": "path", "required": true, "schema": {"type": "string"}}
        ],
        "responses": {"200": {"description": "A pet"}}
      }
    }
  }
}
"""

toolset = OpenAPIToolset(spec_str=PETSTORE_SPEC, spec_str_type="json")
print([t.name for t in asyncio.run(toolset.get_tools())])
# ['listPets', 'showPetById']

agent = LlmAgent(
    name="petstore_agent",
    model="gemini-2.5-flash",
    instruction="Help users browse the pet store.",
    tools=[toolset],
)
```

### Example 2 — YAML spec with API key auth

```python
import yaml
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, HttpAuth, HttpCredentials
from google.adk.auth.auth_schemes import CustomAuthScheme
from google.genai import types as genai_types

SPEC_YAML = """
openapi: "3.0.0"
info:
  title: Weather API
  version: "1.0"
servers:
  - url: https://api.weather.example.com
paths:
  /forecast:
    get:
      operationId: getForecast
      summary: Get weather forecast
      parameters:
        - name: city
          in: query
          required: true
          schema:
            type: string
      responses:
        "200":
          description: Forecast data
"""

toolset = OpenAPIToolset(
    spec_str=SPEC_YAML,
    spec_str_type="yaml",
    auth_scheme=CustomAuthScheme(
        name="api_key",
        description="API key in X-API-Key header",
    ),
    auth_credential=AuthCredential(
        auth_type=AuthCredentialTypes.HTTP,
        http=HttpAuth(
            scheme="bearer",
            credentials=HttpCredentials(token="my-secret-api-key"),
        ),
    ),
)
```

### Example 3 — multiple specs with prefixes and dynamic headers

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

def make_correlation_headers(ctx: ReadonlyContext) -> dict[str, str]:
    """Inject a correlation ID from session state into every request."""
    correlation_id = ctx.state.get("correlation_id", "unknown")
    return {
        "X-Correlation-ID": correlation_id,
        "X-Client-Version": "2.2.0",
    }

orders_toolset = OpenAPIToolset(
    spec_dict=orders_spec,
    tool_name_prefix="orders_",
    header_provider=make_correlation_headers,
    ssl_verify=False,  # dev environment with self-signed cert
)

inventory_toolset = OpenAPIToolset(
    spec_dict=inventory_spec,
    tool_name_prefix="inventory_",
    header_provider=make_correlation_headers,
)

agent = LlmAgent(
    name="ecommerce_agent",
    model="gemini-2.5-flash",
    instruction="Help with orders and inventory. Use orders_* for order management, inventory_* for stock.",
    tools=[orders_toolset, inventory_toolset],
)
```

### Example 4 — filter to a subset of operations

```python
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

# Only expose read operations; filter out create/update/delete
toolset = OpenAPIToolset(
    spec_dict=full_api_spec,
    tool_filter=["listPets", "showPetById", "searchPets"],  # allowlist by operationId
)
```

### `RestApiTool` per-tool configuration (post-construction)

```python
tool = toolset.get_tool("listPets")
if tool:
    tool.configure_ssl_verify("/etc/ssl/certs/corporate-ca.pem")
    tool.configure_auth_scheme(my_auth_scheme)
    tool.configure_auth_credential(my_auth_credential)
```

### Gotchas

- Tool names are derived from `operationId`. Missing `operationId` causes the parser to skip that operation.
- `preserve_property_names=False` (default) converts `camelCase` parameters to `snake_case`. If the backend requires `camelCase` in the request body, set `preserve_property_names=True`.
- `httpx_client_factory` must return a **new** client on every call — it is closed as an async context manager after each request.
- The `ssl_verify` parameter applies globally. Call `configure_ssl_verify_all()` after construction to update all tools at once.

---

## 3 · `ExampleTool` + `BaseExampleProvider`

**Source:** `google.adk.tools.example_tool`, `google.adk.examples`

`ExampleTool` injects few-shot examples into the LLM system instruction at inference time. It is not a callable tool in the traditional sense — it has no function the LLM invokes. Instead, it hooks into `process_llm_request` and prepends formatted examples before the model call.

### `Example` model (source-verified)

```python
from google.adk.examples.example import Example
from google.genai import types

Example(
    input: types.Content,   # the example user message
    output: list[types.Content],  # the expected assistant response(s)
)
```

### `ExampleTool` constructor (source-verified)

```python
from google.adk.tools.example_tool import ExampleTool

ExampleTool(examples: list[Example] | BaseExampleProvider)
```

When a list is provided, all examples are injected for every request. When a `BaseExampleProvider` is provided, `get_examples(query)` is called with the user's message text, enabling dynamic/semantic retrieval.

### Example 1 — static few-shot examples

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.example_tool import ExampleTool
from google.adk.examples.example import Example
from google.genai import types

examples = [
    Example(
        input=types.Content(
            role="user",
            parts=[types.Part.from_text("What is 2 + 2?")],
        ),
        output=[
            types.Content(
                role="model",
                parts=[types.Part.from_text("4")],
            )
        ],
    ),
    Example(
        input=types.Content(
            role="user",
            parts=[types.Part.from_text("Capital of France?")],
        ),
        output=[
            types.Content(
                role="model",
                parts=[types.Part.from_text("Paris")],
            )
        ],
    ),
]

agent = LlmAgent(
    name="factual_agent",
    model="gemini-2.5-flash",
    instruction="Answer factual questions concisely.",
    tools=[ExampleTool(examples)],
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="facts")
    await runner.session_service.create_session(
        app_name="facts", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "What is 5 + 7?", user_id="u1", session_id="s1"
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### `BaseExampleProvider` — dynamic example retrieval

```python
import abc
from google.adk.examples.base_example_provider import BaseExampleProvider
from google.adk.examples.example import Example

class BaseExampleProvider(abc.ABC):
    @abc.abstractmethod
    def get_examples(self, query: str) -> list[Example]:
        """Return relevant examples for a given query."""
```

### Example 2 — keyword-based dynamic provider

```python
from google.adk.examples.base_example_provider import BaseExampleProvider
from google.adk.examples.example import Example
from google.genai import types

class KeywordExampleProvider(BaseExampleProvider):
    """Returns domain-specific examples based on keywords in the query."""

    def __init__(self):
        self._math_examples = [
            Example(
                input=types.Content(role="user", parts=[types.Part.from_text("What is 3 * 4?")]),
                output=[types.Content(role="model", parts=[types.Part.from_text("12")])],
            ),
        ]
        self._geo_examples = [
            Example(
                input=types.Content(role="user", parts=[types.Part.from_text("Capital of Japan?")]),
                output=[types.Content(role="model", parts=[types.Part.from_text("Tokyo")])],
            ),
        ]

    def get_examples(self, query: str) -> list[Example]:
        q = query.lower()
        if any(kw in q for kw in ["add", "subtract", "multiply", "divide", "+", "-", "*", "/"]):
            return self._math_examples
        if any(kw in q for kw in ["capital", "country", "city", "where is"]):
            return self._geo_examples
        return []  # no examples for unrecognised query types

from google.adk.tools.example_tool import ExampleTool
from google.adk.agents import LlmAgent

agent = LlmAgent(
    name="smart_agent",
    model="gemini-2.5-flash",
    instruction="Answer questions accurately.",
    tools=[ExampleTool(KeywordExampleProvider())],
)
```

### `VertexAiExampleStore` — semantic example retrieval

```python
from google.adk.examples.vertex_ai_example_store import VertexAiExampleStore
from google.adk.tools.example_tool import ExampleTool
from google.adk.agents import LlmAgent

# Requires an existing Vertex AI Example Store resource
example_store = VertexAiExampleStore(
    examples_store_name="projects/my-project/locations/us-central1/exampleStores/my-store"
)

agent = LlmAgent(
    name="semantic_agent",
    model="gemini-2.5-flash",
    instruction="Answer questions based on relevant examples.",
    tools=[ExampleTool(example_store)],
)
```

### Gotchas

- `ExampleTool` only fires when the **first part** of the current user message is text. If the user sends an image or audio as the first part, no examples are injected.
- Examples are prepended to the system instruction, not the conversation history. This means they count against the system prompt token budget, not the conversation context window.
- `get_examples(query)` is called synchronously — keep it fast. For slow async providers, pre-load examples in a background task.

---

## 4 · `EnvironmentSimulationConfig` + `ToolSimulationConfig` + `InjectionConfig`

**Source:** `google.adk.tools.environment_simulation`

The environment simulation module (`@experimental`) lets you **mock tool responses** during testing — without real API calls. It supports probability-based injection of pre-canned responses, errors, and latency, with a fallback to an LLM-driven mock strategy when no injection matches.

### Class hierarchy

```
EnvironmentSimulationConfig
  └── tool_simulation_configs: list[ToolSimulationConfig]
        ├── injection_configs: list[InjectionConfig]  (tried first, in order)
        └── mock_strategy_type: MockStrategy           (fallback when no injection hits)
```

### `EnvironmentSimulationConfig` fields (source-verified)

```python
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig, ToolSimulationConfig, InjectionConfig,
    InjectedError, MockStrategy,
)

EnvironmentSimulationConfig(
    tool_simulation_configs: list[ToolSimulationConfig],  # required, non-empty
    simulation_model: str = "gemini-2.5-flash",
    simulation_model_configuration: GenerateContentConfig = ...,
    tracing: Optional[str] = None,        # prior run trace JSON for context
    environment_data: Optional[str] = None,  # DB dump / reference data JSON
)
```

### `ToolSimulationConfig` fields

```python
ToolSimulationConfig(
    tool_name: str,                                # must match the tool's .name
    injection_configs: list[InjectionConfig] = [], # tried in order
    mock_strategy_type: MockStrategy = MockStrategy.MOCK_STRATEGY_UNSPECIFIED,
)
```

At least one of `injection_configs` or a non-`UNSPECIFIED` `mock_strategy_type` is required.

### `InjectionConfig` fields

```python
InjectionConfig(
    injection_probability: float = 1.0,          # 0.0–1.0; 1.0 = always inject
    match_args: Optional[dict[str, Any]] = None, # only inject if args match
    injected_latency_seconds: float = 0.0,       # max 120.0
    random_seed: Optional[int] = None,
    injected_error: Optional[InjectedError] = None,    # XOR with injected_response
    injected_response: Optional[dict[str, Any]] = None,
)
```

`injected_error` and `injected_response` are mutually exclusive — exactly one must be set.

### `MockStrategy` enum

| Value | Behaviour |
|---|---|
| `MOCK_STRATEGY_UNSPECIFIED` | No fallback mock — returns `None` (no-op) |
| `TOOL_SPEC_MOCK_STRATEGY` | Uses the LLM + tool spec + `environment_data`/`tracing` to generate a plausible response |

### Example 1 — deterministic response injection for unit testing

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig, ToolSimulationConfig, InjectionConfig,
)
from google.adk.tools.environment_simulation.environment_simulation_plugin import (
    EnvironmentSimulationPlugin,
)

def get_stock_price(ticker: str) -> dict:
    """Fetch the current stock price for a ticker symbol."""
    # In production this would call a real API
    ...

agent = LlmAgent(
    name="trader",
    model="gemini-2.5-flash",
    instruction="Provide stock price info.",
    tools=[get_stock_price],
)

sim_config = EnvironmentSimulationConfig(
    tool_simulation_configs=[
        ToolSimulationConfig(
            tool_name="get_stock_price",
            injection_configs=[
                InjectionConfig(
                    injection_probability=1.0,
                    match_args={"ticker": "GOOG"},
                    injected_response={"ticker": "GOOG", "price": 185.42, "currency": "USD"},
                ),
                InjectionConfig(
                    injection_probability=1.0,
                    match_args={"ticker": "AAPL"},
                    injected_response={"ticker": "AAPL", "price": 212.10, "currency": "USD"},
                ),
            ],
        )
    ]
)

async def main():
    runner = InMemoryRunner(
        agent=agent,
        app_name="trader",
        plugins=[EnvironmentSimulationPlugin(config=sim_config)],
    )
    await runner.session_service.create_session(
        app_name="trader", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "What's the current price of GOOG?", user_id="u1", session_id="s1"
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 2 — probabilistic error injection (chaos testing)

```python
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig, ToolSimulationConfig, InjectionConfig,
    InjectedError, MockStrategy,
)

sim_config = EnvironmentSimulationConfig(
    tool_simulation_configs=[
        ToolSimulationConfig(
            tool_name="send_email",
            injection_configs=[
                # 20% of calls get a 503 Service Unavailable
                InjectionConfig(
                    injection_probability=0.2,
                    random_seed=42,
                    injected_error=InjectedError(
                        injected_http_error_code=503,
                        error_message="Email service temporarily unavailable",
                    ),
                ),
                # 80% pass through to the mock strategy fallback
            ],
            mock_strategy_type=MockStrategy.TOOL_SPEC_MOCK_STRATEGY,
        )
    ],
    simulation_model="gemini-2.5-flash",
)
```

### Example 3 — LLM-generated mocks with context data

```python
import json
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig, ToolSimulationConfig, MockStrategy,
)

# Provide a snapshot of the "database" so the LLM generates realistic mocks
MOCK_DB = json.dumps({
    "users": [
        {"id": "u1", "name": "Alice", "email": "alice@example.com"},
        {"id": "u2", "name": "Bob",   "email": "bob@example.com"},
    ]
})

sim_config = EnvironmentSimulationConfig(
    tool_simulation_configs=[
        ToolSimulationConfig(
            tool_name="lookup_user",
            mock_strategy_type=MockStrategy.TOOL_SPEC_MOCK_STRATEGY,
        ),
        ToolSimulationConfig(
            tool_name="update_user",
            mock_strategy_type=MockStrategy.TOOL_SPEC_MOCK_STRATEGY,
        ),
    ],
    simulation_model="gemini-2.0-flash",
    environment_data=MOCK_DB,  # passed to mock strategy for context
)
```

### Gotchas

- `tool_simulation_configs` must be **non-empty** and contain no duplicate `tool_name` values (validated at construction).
- `injection_configs` are evaluated in order; the first matching config fires. Put specific `match_args` before catch-all configs.
- `injected_latency_seconds` is capped at `120.0` seconds.
- `TOOL_SPEC_MOCK_STRATEGY` makes real LLM calls — it is not free. Use deterministic `injection_configs` for unit tests; reserve `TOOL_SPEC_MOCK_STRATEGY` for exploratory testing.

---

## 5 · `LlmBackedUserSimulator` + `LlmBackedUserSimulatorConfig` + `UserPersona`

**Source:** `google.adk.evaluation.simulation`

`LlmBackedUserSimulator` drives automated multi-turn evaluation without a human in the loop. An LLM plays the role of the user, following a `ConversationScenario` and stopping when it detects the conversation is complete.

### `LlmBackedUserSimulatorConfig` fields (source-verified)

```python
from google.adk.evaluation.simulation.llm_backed_user_simulator import (
    LlmBackedUserSimulator, LlmBackedUserSimulatorConfig,
)

LlmBackedUserSimulatorConfig(
    model: str = "gemini-2.5-flash",
    model_configuration: GenerateContentConfig = ...,  # thinking enabled by default
    max_allowed_invocations: int = 20,      # -1 for unlimited (not recommended)
    custom_instructions: str | None = None, # must include {{ stop_signal }}, {{ conversation_plan }}, etc.
    include_function_calls: bool = False,   # whether to show tool calls in history
)
```

### `UserPersona` + `UserBehavior` (source-verified)

```python
from google.adk.evaluation.simulation.user_simulator_personas import (
    UserPersona, UserBehavior,
)
from google.adk.evaluation.simulation.pre_built_personas import PreBuiltBehaviors

UserPersona(
    name: str,
    description: str,
    behaviors: list[UserBehavior],
)

UserBehavior(
    name: str,
    description: str,
)
```

Pre-built behaviors are available in `PreBuiltBehaviors` — check the source for the full list of ready-made persona types.

### `ConversationScenario` (source-verified)

```python
from google.adk.evaluation.conversation_scenarios import ConversationScenario

ConversationScenario(
    starting_prompt: str,          # the very first message the simulator sends
    conversation_plan: str,        # multi-turn objective for the simulator
    user_persona: Optional[UserPersona] = None,
)
```

### Example 1 — basic automated evaluation loop

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.evaluation.conversation_scenarios import ConversationScenario
from google.adk.evaluation.simulation.llm_backed_user_simulator import (
    LlmBackedUserSimulator, LlmBackedUserSimulatorConfig,
)

agent = LlmAgent(
    name="support_agent",
    model="gemini-2.5-flash",
    instruction="You are a customer support agent. Help users reset their passwords.",
)

scenario = ConversationScenario(
    starting_prompt="Hi, I forgot my password and can't log in.",
    conversation_plan=(
        "You are a user who forgot their password. "
        "Follow the support agent's instructions to reset it. "
        "When the issue is resolved, end the conversation."
    ),
)

simulator_config = LlmBackedUserSimulatorConfig(
    model="gemini-2.5-flash",
    max_allowed_invocations=10,
)

simulator = LlmBackedUserSimulator(
    config=simulator_config,
    conversation_scenario=scenario,
)
```

### Example 2 — persona-driven simulation

```python
from google.adk.evaluation.simulation.user_simulator_personas import (
    UserPersona, UserBehavior,
)
from google.adk.evaluation.conversation_scenarios import ConversationScenario
from google.adk.evaluation.simulation.llm_backed_user_simulator import (
    LlmBackedUserSimulator, LlmBackedUserSimulatorConfig,
)

impatient_persona = UserPersona(
    name="impatient_user",
    description="A busy professional who wants quick answers and gets frustrated by long explanations.",
    behaviors=[
        UserBehavior(
            name="terse",
            description="Gives short, clipped responses. Doesn't provide extra context unless asked.",
        ),
        UserBehavior(
            name="escalates_quickly",
            description="After two unhelpful replies, immediately asks to speak to a manager.",
        ),
    ],
)

scenario = ConversationScenario(
    starting_prompt="My order #12345 hasn't arrived. Where is it?",
    conversation_plan=(
        "You are trying to track down a missing order. "
        "Follow the agent's instructions but lose patience quickly if they ask for "
        "information you've already provided."
    ),
    user_persona=impatient_persona,
)

simulator = LlmBackedUserSimulator(
    config=LlmBackedUserSimulatorConfig(max_allowed_invocations=8),
    conversation_scenario=scenario,
)
```

### Stop signal mechanism (source-verified)

The simulator adds a stop signal to its prompt: `_STOP_SIGNAL = "<<STOP>>"`. When the LLM response contains this string (case-insensitive), `get_next_user_message` returns `NextUserMessage(status=Status.STOP_SIGNAL_DETECTED)` and the loop terminates. This prevents the simulator from looping indefinitely.

### Gotchas

- `LlmBackedUserSimulator` is `@experimental` — the API may change.
- `max_allowed_invocations=-1` disables the turn limit and can cause infinite loops if the simulator never detects conversation completion.
- `custom_instructions` must include all four Jinja placeholders (`{{ stop_signal }}`, `{{ conversation_plan }}`, `{{ conversation_history }}`, `{{ persona }}`) or the simulator will raise a validation error.
- `get_simulation_evaluator()` raises `NotImplementedError` — you must provide your own evaluator.

---

## 6 · `BaseCodeExecutor` + `VertexAiCodeExecutor` + `UnsafeLocalCodeExecutor`

**Source:** `google.adk.code_executors`

Code executors allow an ADK agent to **execute Python code blocks** generated by the LLM and feed the results back into the response. The agent's LLM generates code; the executor runs it; stdout/stderr/output files are returned.

### `BaseCodeExecutor` fields (source-verified)

```python
from google.adk.code_executors.base_code_executor import BaseCodeExecutor

# All executors inherit these Pydantic fields:
optimize_data_file: bool = False     # extract CSV files from request, attach to executor
stateful: bool = False               # reuse execution session across turns
error_retry_attempts: int = 2        # retry on consecutive errors
code_block_delimiters: list[tuple[str, str]] = [
    ('```tool_code\n', '\n```'),
    ('```python\n', '\n```'),
]
execution_result_delimiters: tuple[str, str] = ('```tool_output\n', '\n```')
timeout_seconds: Optional[int] = None
```

### Executor comparison

| Executor | Backend | `stateful` | Safe | Best for |
|---|---|---|---|---|
| `BuiltInCodeExecutor` | Gemini's built-in code interpreter | ✓ | ✓ | Simple code tasks with Gemini models |
| `VertexAiCodeExecutor` | Vertex AI Code Interpreter Extension | ✓ | ✓ | Production workloads, managed sandbox |
| `UnsafeLocalCodeExecutor` | Local `multiprocessing.Process` (spawned) | ✗ | ✗ | Dev/testing only — no sandbox |
| `AgentEngineSandboxCodeExecutor` | Agent Engine managed sandbox | ✓ | ✓ | Cloud-native production |

### Example 1 — `BuiltInCodeExecutor` with a data analysis agent

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.code_executors.built_in_code_executor import BuiltInCodeExecutor

agent = LlmAgent(
    name="data_analyst",
    model="gemini-2.5-flash",
    instruction=(
        "You are a data analyst. When the user asks a question requiring "
        "calculations, write and execute Python code to compute the answer."
    ),
    code_executor=BuiltInCodeExecutor(),
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="analyst")
    await runner.session_service.create_session(
        app_name="analyst", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Calculate the mean and std dev of [12, 45, 23, 67, 34, 89, 11].",
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 2 — `VertexAiCodeExecutor` with stateful sessions and file output

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.code_executors.vertex_ai_code_executor import VertexAiCodeExecutor

executor = VertexAiCodeExecutor(
    stateful=True,          # session persists across turns — variables stay in scope
    optimize_data_file=True,  # extract CSV attachments and pass to executor
    error_retry_attempts=3,
    timeout_seconds=60,
)

agent = LlmAgent(
    name="chart_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a data visualisation expert. "
        "Use Python (matplotlib) to generate charts when requested."
    ),
    code_executor=executor,
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="charts")
    await runner.session_service.create_session(
        app_name="charts", user_id="u1", session_id="s1"
    )
    # Turn 1: define data
    await runner.run_debug(
        "Create a variable `sales = [100, 150, 120, 200, 180]`.",
        user_id="u1", session_id="s1",
    )
    # Turn 2: reuse variable from turn 1 (stateful=True)
    events = await runner.run_debug(
        "Now plot `sales` as a bar chart and save it as 'sales.png'.",
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 3 — `UnsafeLocalCodeExecutor` for fast local dev/testing

```python
from google.adk.agents import LlmAgent
from google.adk.code_executors.unsafe_local_code_executor import UnsafeLocalCodeExecutor

# WARNING: executes arbitrary Python in a spawned subprocess on your machine.
# Never use in production or with untrusted LLM output.
agent = LlmAgent(
    name="dev_agent",
    model="gemini-2.5-flash",
    instruction="Execute Python code for quick calculations.",
    code_executor=UnsafeLocalCodeExecutor(
        error_retry_attempts=1,
        timeout_seconds=10,
    ),
)
```

### `CodeExecutionResult` structure (source-verified)

```python
@dataclass
class CodeExecutionResult:
    stdout: str
    stderr: str
    output_files: list[File]  # images (PNG/JPG/GIF), CSVs, etc.

@dataclass
class File:
    name: str
    content: bytes   # base64-encoded by the executor
    mime_type: str
```

### Gotchas

- `UnsafeLocalCodeExecutor` **cannot** be `stateful=True` or `optimize_data_file=True` — setting either raises `ValueError` at construction.
- `VertexAiCodeExecutor` saves output images as `File` objects with `mime_type=image/{ext}`. These are available in the `CodeExecutionResult.output_files` but are **not** automatically saved as ADK artifacts — do that manually in a callback if needed.
- `stateful=True` sessions are keyed on `invocation_context.session.id`. If you switch session services mid-session, the executor state is lost.

---

## 7 · `ComputerUseTool` + `BaseComputer` + `ComputerUseToolset`

**Source:** `google.adk.tools.computer_use`

The computer use module (`@experimental`) gives an ADK agent a browser/GUI interface. `BaseComputer` defines the abstract interface; `ComputerUseTool` wraps individual actions; `ComputerUseToolset` aggregates them and manages the lifecycle.

### `BaseComputer` abstract methods (source-verified)

```python
class BaseComputer(abc.ABC):
    async def prepare(self, tool_context: ToolContext) -> None: ...  # optional setup

    async def screen_size(self) -> tuple[int, int]: ...
    async def open_web_browser(self) -> ComputerState: ...
    async def click_at(self, x: int, y: int) -> ComputerState: ...
    async def hover_at(self, x: int, y: int) -> ComputerState: ...
    async def type_text_at(self, x: int, y: int, text: str,
                           press_enter: bool = True,
                           clear_before_typing: bool = True) -> ComputerState: ...
    async def scroll_document(self, direction: Literal["up","down","left","right"]) -> ComputerState: ...
    async def scroll_at(self, x: int, y: int,
                        direction: Literal["up","down","left","right"],
                        magnitude: int) -> ComputerState: ...
    async def wait(self, seconds: int) -> ComputerState: ...
    async def go_back(self) -> ComputerState: ...
    async def go_forward(self) -> ComputerState: ...
    async def search(self) -> ComputerState: ...
    async def navigate(self, url: str) -> ComputerState: ...
    async def key_combination(self, keys: list[str]) -> ComputerState: ...
    async def drag_and_drop(self, x: int, y: int,
                            destination_x: int, destination_y: int) -> ComputerState: ...
```

### `ComputerState` (source-verified)

```python
@dataclass
class ComputerState:
    screenshot: bytes    # PNG bytes of the current screen
    url: Optional[str]   # current URL (browser environments)
```

Every action returns a `ComputerState` snapshot for the LLM to reason about.

### `ComputerEnvironment` enum

```python
class ComputerEnvironment(Enum):
    WEB_BROWSER = "web_browser"
```

### `ComputerUseToolset` constructor (source-verified)

```python
from google.adk.tools.computer_use.computer_use_toolset import ComputerUseToolset

ComputerUseToolset(
    computer: BaseComputer,
    environment: ComputerEnvironment = ComputerEnvironment.WEB_BROWSER,
    tool_filter: Optional[Union[ToolPredicate, list[str]]] = None,
)
```

### Example 1 — implementing a Playwright-based `BaseComputer`

```python
import asyncio
from playwright.async_api import async_playwright, Page
from google.adk.tools.computer_use.base_computer import BaseComputer, ComputerState
from google.adk.tools.tool_context import ToolContext
from typing import Literal

class PlaywrightComputer(BaseComputer):
    """Chromium-based computer using Playwright."""

    def __init__(self, headless: bool = True):
        self._headless = headless
        self._playwright = None
        self._browser = None
        self._page: Optional[Page] = None

    async def prepare(self, tool_context: ToolContext) -> None:
        if self._page is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=self._headless)
            self._page = await self._browser.new_page()

    async def _snapshot(self) -> ComputerState:
        screenshot = await self._page.screenshot(type="png")
        url = self._page.url
        return ComputerState(screenshot=screenshot, url=url)

    async def screen_size(self) -> tuple[int, int]:
        vp = self._page.viewport_size or {"width": 1280, "height": 800}
        return vp["width"], vp["height"]

    async def open_web_browser(self) -> ComputerState:
        await self._page.goto("about:blank")
        return await self._snapshot()

    async def navigate(self, url: str) -> ComputerState:
        await self._page.goto(url)
        return await self._snapshot()

    async def click_at(self, x: int, y: int) -> ComputerState:
        await self._page.mouse.click(x, y)
        return await self._snapshot()

    async def hover_at(self, x: int, y: int) -> ComputerState:
        await self._page.mouse.move(x, y)
        return await self._snapshot()

    async def type_text_at(self, x: int, y: int, text: str,
                           press_enter: bool = True,
                           clear_before_typing: bool = True) -> ComputerState:
        await self._page.mouse.click(x, y)
        if clear_before_typing:
            await self._page.keyboard.press("Control+a")
        await self._page.keyboard.type(text)
        if press_enter:
            await self._page.keyboard.press("Enter")
        return await self._snapshot()

    async def scroll_document(self, direction: str) -> ComputerState:
        delta = {"up": (0, -300), "down": (0, 300), "left": (-300, 0), "right": (300, 0)}
        dx, dy = delta[direction]
        await self._page.mouse.wheel(dx, dy)
        return await self._snapshot()

    async def scroll_at(self, x: int, y: int, direction: str, magnitude: int) -> ComputerState:
        delta = {"up": (0, -magnitude), "down": (0, magnitude),
                 "left": (-magnitude, 0), "right": (magnitude, 0)}
        dx, dy = delta[direction]
        await self._page.mouse.wheel(dx, dy)
        return await self._snapshot()

    async def wait(self, seconds: int) -> ComputerState:
        await asyncio.sleep(seconds)
        return await self._snapshot()

    async def go_back(self) -> ComputerState:
        await self._page.go_back()
        return await self._snapshot()

    async def go_forward(self) -> ComputerState:
        await self._page.go_forward()
        return await self._snapshot()

    async def search(self) -> ComputerState:
        await self._page.goto("https://google.com")
        return await self._snapshot()

    async def key_combination(self, keys: list[str]) -> ComputerState:
        await self._page.keyboard.press("+".join(keys))
        return await self._snapshot()

    async def drag_and_drop(self, x: int, y: int,
                            destination_x: int, destination_y: int) -> ComputerState:
        await self._page.drag_and_drop(
            f"[data-x='{x}']", f"[data-x='{destination_x}']"
        )
        return await self._snapshot()
```

### Example 2 — wiring the computer to an agent

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.computer_use.computer_use_toolset import ComputerUseToolset
from google.adk.tools.computer_use.base_computer import ComputerEnvironment

computer = PlaywrightComputer(headless=True)
toolset = ComputerUseToolset(
    computer=computer,
    environment=ComputerEnvironment.WEB_BROWSER,
)

agent = LlmAgent(
    name="web_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You control a web browser. To complete tasks: "
        "1. Open the browser. 2. Navigate to URLs. 3. Click, type, and scroll as needed. "
        "Always look at the screenshot to understand the current page state."
    ),
    tools=[toolset],
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="browser")
    await runner.session_service.create_session(
        app_name="browser", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Go to python.org and tell me the latest Python version.",
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Gotchas

- `ComputerUseTool` is `@experimental` — the interface may change.
- `prepare()` is called before each tool invocation, not once per session. Use `tool_context.state` to store and reuse the browser session across calls.
- All coordinates (`x`, `y`) are absolute pixels scaled to the screen dimensions returned by `screen_size()`. The LLM must interpret the screenshot correctly to compute valid coordinates.
- Screenshot bytes are returned as part of the tool response and counted against the LLM's context window — keep sessions short to avoid token bloat.

---

## 8 · `GEPARootAgentPromptOptimizer` + `SimplePromptOptimizer`

**Source:** `google.adk.optimization`

The optimization module (`@experimental`) provides two optimizer implementations for automatically improving an agent's instruction prompt using evaluation metrics.

### Optimizer interface (source-verified)

```python
from google.adk.optimization.agent_optimizer import AgentOptimizer

class AgentOptimizer(ABC):
    async def optimize(
        self,
        initial_agent: Agent,
        sampler: Sampler,
    ) -> OptimizerResult:
        ...
```

Both optimizers take an `initial_agent` and a `Sampler` (the evaluation harness) and return an `OptimizerResult` with the best found agent(s).

### `SimplePromptOptimizer` — iterative hill-climbing

```python
from google.adk.optimization.simple_prompt_optimizer import (
    SimplePromptOptimizer, SimplePromptOptimizerConfig,
)

config = SimplePromptOptimizerConfig(
    optimizer_model: str = "gemini-2.5-flash",
    model_configuration: GenerateContentConfig = ...,  # thinking enabled
    num_iterations: int = 10,   # how many candidate prompts to try
    batch_size: int = 5,        # training examples used per evaluation
)

optimizer = SimplePromptOptimizer(config=config)
```

**Algorithm:** At each iteration, the optimizer asks the LLM to suggest an improved prompt given the current best score. If the new prompt scores higher on a random batch of training examples, it replaces the current best. After all iterations, final validation runs on the full validation set.

### `GEPARootAgentPromptOptimizer` — GEPA framework

```python
from google.adk.optimization.gepa_root_agent_prompt_optimizer import (
    GEPARootAgentPromptOptimizer, GEPARootAgentPromptOptimizerConfig,
)

config = GEPARootAgentPromptOptimizerConfig(
    optimizer_model: str = "gemini-2.5-flash",
    model_configuration: GenerateContentConfig = ...,  # thinking budget: 10240
    max_metric_calls: int = 100,       # budget: total evaluations
    reflection_minibatch_size: int = 3, # examples per reflection step
    run_dir: Optional[str] = None,     # save intermediate results here
)

optimizer = GEPARootAgentPromptOptimizer(config=config)
```

**GEPA** (Generative Prompt Adaptation) uses a separate `reflection_lm` to analyse evaluation failures and propose targeted improvements. It requires the `gepa` extra (`pip install google-adk[gepa]`). Only the **root agent's** instruction is optimized — sub-agents are left unchanged (a warning is logged if sub-agents exist).

### `OptimizerResult` structure

```python
from google.adk.optimization.data_types import OptimizerResult, AgentWithScores

@dataclass
class OptimizerResult:
    optimized_agents: list[AgentWithScores]  # Pareto-optimal candidates

@dataclass
class AgentWithScores:
    optimized_agent: Agent
    overall_score: float
```

### Example 1 — `SimplePromptOptimizer` end-to-end

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.optimization.simple_prompt_optimizer import (
    SimplePromptOptimizer, SimplePromptOptimizerConfig,
)

# You must provide a Sampler implementation; see google.adk.optimization.sampler
# This example shows the optimizer setup; Sampler wiring is app-specific.

initial_agent = LlmAgent(
    name="qa_agent",
    model="gemini-2.5-flash",
    instruction="Answer questions accurately.",  # starting prompt to optimise
)

optimizer = SimplePromptOptimizer(
    config=SimplePromptOptimizerConfig(
        optimizer_model="gemini-2.5-flash",
        num_iterations=5,
        batch_size=3,
    )
)

async def run_optimization(sampler):
    result = await optimizer.optimize(initial_agent, sampler)
    best = result.optimized_agents[0]
    print(f"Best score: {best.overall_score:.3f}")
    print(f"Optimized instruction:\n{best.optimized_agent.instruction}")
    return best.optimized_agent
```

### Example 2 — `GEPARootAgentPromptOptimizer` with custom run directory

```python
import asyncio
import os
from google.adk.agents import LlmAgent
from google.adk.optimization.gepa_root_agent_prompt_optimizer import (
    GEPARootAgentPromptOptimizer, GEPARootAgentPromptOptimizerConfig,
)

initial_agent = LlmAgent(
    name="support_agent",
    model="gemini-2.5-flash",
    instruction="You are a support agent.",
)

optimizer = GEPARootAgentPromptOptimizer(
    config=GEPARootAgentPromptOptimizerConfig(
        optimizer_model="gemini-2.5-flash",
        max_metric_calls=50,          # keep budget low for quick experiments
        reflection_minibatch_size=5,
        run_dir="./optimization_runs/support_v1",  # save candidates here
    )
)

async def run(sampler):
    result = await optimizer.optimize(initial_agent, sampler)
    for i, candidate in enumerate(result.optimized_agents):
        print(f"Candidate {i}: score={candidate.overall_score:.3f}")
        print(candidate.optimized_agent.instruction[:200])
```

### Gotchas

- Both optimizers are `@experimental`.
- `GEPARootAgentPromptOptimizer` requires `pip install gepa` (separate package not bundled with `google-adk`).
- The `Sampler` interface is the most complex part — it wraps evaluation logic. See `google.adk.optimization.sampler.Sampler` source for the full contract.
- `SimplePromptOptimizer` uses random sampling from the training set; different `random.seed()` values give different trajectories.
- Training and validation UIDs that overlap generate a warning but are allowed — useful when you have a small dataset.

---

## 9 · `SkillToolset` + `Skill` + `Frontmatter`

**Source:** `google.adk.tools.skill_toolset`, `google.adk.skills.models`

The skills system lets an agent **discover and load self-contained capability bundles** (skills) at runtime. Each skill is defined by a `SKILL.md` file with frontmatter metadata, instruction text, and optional resources (scripts, assets, references).

### `Skill` model (source-verified)

```python
from google.adk.skills.models import Skill, Frontmatter, Resources, Script

class Skill(BaseModel):
    frontmatter: Frontmatter   # L1: discovery metadata
    instructions: str          # L2: markdown instructions loaded when skill triggers
    resources: Resources       # L3: scripts, assets, references

    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
```

### `Frontmatter` fields (source-verified)

```python
Frontmatter(
    name: str,                       # kebab-case or snake_case, max 64 chars
    description: str,                # max 1024 chars
    license: Optional[str] = None,
    compatibility: Optional[str] = None,  # max 500 chars
    allowed_tools: Optional[str] = None,  # space-delimited list of pre-approved tool names
    metadata: dict[str, Any] = {},   # custom key-value pairs
)
```

`metadata["adk_additional_tools"]` is a special key: when the skill is activated, the listed tool names are dynamically injected into the agent's tool list.

### `Resources` model (source-verified)

```python
Resources(
    references: dict[str, str | bytes] = {},  # instruction docs
    assets: dict[str, str | bytes] = {},       # schemas, templates, examples
    scripts: dict[str, Script] = {},           # executable scripts
)
```

### `SkillToolset` constructor (source-verified)

```python
from google.adk.tools.skill_toolset import SkillToolset

SkillToolset(
    skills: list[Skill] | None = None,
    registry: SkillRegistry | None = None,   # remote registry for dynamic fetch
    code_executor: BaseCodeExecutor | None = None,
    script_timeout: int = 300,               # seconds for subprocess scripts
    additional_tools: list[ToolUnion] | None = None,  # tools activated by adk_additional_tools
    tool_name_prefix: str | None = None,
    tool_filter: ToolPredicate | list[str] | None = None,
)
```

The toolset registers four built-in tools:
- `list_skills` — list all available skills
- `load_skill` — activate a skill (loads instructions into the agent's context)
- `load_skill_resource` — retrieve a skill's assets/references
- `run_skill_script` — execute a skill's shell/Python script
- `search_skills` — (only if `registry` is provided) semantic search for skills

### Example 1 — inline skill definition

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.skills.models import Skill, Frontmatter, Resources, Script
from google.adk.tools.skill_toolset import SkillToolset

# Define a "sql-query-writer" skill inline
sql_skill = Skill(
    frontmatter=Frontmatter(
        name="sql-query-writer",
        description=(
            "Helps write SQL queries. Activate when the user asks about "
            "SQL, databases, or data retrieval."
        ),
        metadata={
            "adk_additional_tools": ["run_query"],  # inject run_query when activated
        },
    ),
    instructions="""
# SQL Query Writer

You are an expert SQL author. Always:
- Use parameterised queries to prevent injection
- Add LIMIT 1000 unless the user specifies otherwise
- Explain the query after writing it

Supported dialects: PostgreSQL, BigQuery, SQLite.
""",
    resources=Resources(
        assets={"schema.sql": "CREATE TABLE orders (id INT, user_id INT, total DECIMAL);"},
        scripts={"validate.py": Script(src="import sys; print('Schema valid')")},
    ),
)

toolset = SkillToolset(skills=[sql_skill])

agent = LlmAgent(
    name="data_agent",
    model="gemini-2.5-flash",
    instruction="Help with data tasks. Use list_skills to discover what you can do.",
    tools=[toolset],
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="data")
    await runner.session_service.create_session(
        app_name="data", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "What skills do you have?", user_id="u1", session_id="s1"
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 2 — skill with `adk_additional_tools` dynamic injection

When a skill is activated via `load_skill`, any tools listed in `frontmatter.metadata["adk_additional_tools"]` are automatically added to the agent's tool list for that turn:

```python
from google.adk.skills.models import Skill, Frontmatter
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.agents import LlmAgent
import httpx

async def run_query(sql: str) -> dict:
    """Execute a SQL query against the data warehouse."""
    async with httpx.AsyncClient() as client:
        resp = await client.post("https://api.warehouse.example.com/query", json={"sql": sql})
        return resp.json()

skill = Skill(
    frontmatter=Frontmatter(
        name="data-warehouse-query",
        description="Query the corporate data warehouse with SQL.",
        metadata={"adk_additional_tools": ["run_query"]},
    ),
    instructions="Use the run_query tool to execute SQL against the warehouse.",
)

toolset = SkillToolset(
    skills=[skill],
    additional_tools=[run_query],  # pool of tools that can be injected
)

agent = LlmAgent(
    name="warehouse_agent",
    model="gemini-2.5-flash",
    instruction="Help with data warehouse queries. First load the data-warehouse-query skill.",
    tools=[toolset],
)
```

### Example 3 — `run_skill_script` with a validation script

```python
from google.adk.skills.models import Skill, Frontmatter, Resources, Script
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.code_executors.unsafe_local_code_executor import UnsafeLocalCodeExecutor

validation_skill = Skill(
    frontmatter=Frontmatter(
        name="schema-validator",
        description="Validates JSON against a schema. Activate before saving config.",
    ),
    instructions="Run the validate script to check JSON structure before persisting.",
    resources=Resources(
        scripts={
            "validate.py": Script(src="""
import json, sys
data = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
required = ["name", "version", "settings"]
missing = [k for k in required if k not in data]
if missing:
    print(f"INVALID: missing fields {missing}")
    sys.exit(1)
print("VALID")
""")
        }
    ),
)

toolset = SkillToolset(
    skills=[validation_skill],
    code_executor=UnsafeLocalCodeExecutor(),  # needed for run_skill_script
)
```

### Gotchas

- `Frontmatter.name` must be kebab-case or snake_case (≤64 chars). Mixing hyphens and underscores is not allowed.
- `adk_additional_tools` names must exactly match the `tool.name` of tools passed in `additional_tools`. Mismatches are silently skipped with an error log.
- `SkillToolset` caches fetched skill definitions per invocation ID (up to 16 turns) to reduce registry calls. Use `_use_invocation_cache = False` to disable.

---

## 10 · `AutoTracingPlugin` + `DebugLoggingPlugin`

**Source:** `google.adk.plugins.auto_tracing_plugin`, `google.adk.plugins.debug_logging_plugin`

### `AutoTracingPlugin` — zero-code OTel instrumentation

`AutoTracingPlugin` walks the agent object graph and wraps every public function/method in your code with an OpenTelemetry span — **without any manual instrumentation**. It fires before the first agent run and instruments all loaded modules whose names match the discovered scope prefixes.

```python
from google.adk.plugins.auto_tracing_plugin import AutoTracingPlugin
from opentelemetry import trace

AutoTracingPlugin(
    name: str = "AutoTracingPlugin",
    extra_scope_prefixes: tuple[str, ...] = (),  # extra module prefixes to instrument
    tracer: trace.Tracer | None = None,           # default: trace.get_tracer(__name__)
    max_repr_len: int = ...,           # truncation for argument repr in spans
    max_recorded_yields: int = ...,    # max yielded values captured per generator
    max_walk_depth: int = ...,         # depth of agent object-graph walk
)
```

The plugin discovers scope prefixes by walking the invocation context's agent object graph up to `max_walk_depth`. Any module encountered is added to the instrument set — your custom tool modules, callback modules, etc. are automatically covered.

### Example 1 — attach `AutoTracingPlugin` to a runner

```python
import asyncio
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.plugins.auto_tracing_plugin import AutoTracingPlugin

# Configure OTel
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
tracer = provider.get_tracer("my_app")

def get_news(topic: str) -> list[str]:
    """Fetch latest news headlines for a topic."""
    return [f"Breaking: {topic} update #1", f"Analysis: {topic} deep dive"]

agent = LlmAgent(
    name="news_agent",
    model="gemini-2.5-flash",
    instruction="Fetch and summarise news.",
    tools=[get_news],
)

async def main():
    runner = InMemoryRunner(
        agent=agent,
        app_name="news",
        plugins=[
            AutoTracingPlugin(
                tracer=tracer,
                extra_scope_prefixes=("my_app.",),  # instrument your own modules
            )
        ],
    )
    await runner.session_service.create_session(
        app_name="news", user_id="u1", session_id="s1"
    )
    await runner.run_debug(
        "What's happening with AI today?", user_id="u1", session_id="s1"
    )
    # Spans appear on stdout via ConsoleSpanExporter

asyncio.run(main())
```

### `DebugLoggingPlugin` — YAML debug dump

`DebugLoggingPlugin` records every LLM request/response, tool call/result, and session state to a YAML file. Each invocation is appended as a separate YAML document (`---` separator).

```python
from google.adk.plugins.debug_logging_plugin import DebugLoggingPlugin

DebugLoggingPlugin(
    name: str = "debug_logging_plugin",
    output_path: str = "adk_debug.yaml",
    include_session_state: bool = True,
    include_system_instruction: bool = True,
)
```

What it records per invocation:
- LLM requests: model, system instruction, conversation contents, tool declarations
- LLM responses: content parts, usage metadata, error codes
- Tool calls with arguments
- Tool responses with results
- Session state snapshot at end of invocation

### Example 2 — debug dump for CI/test debugging

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.plugins.debug_logging_plugin import DebugLoggingPlugin

def calculate_tax(income: float, rate: float) -> dict:
    """Calculate tax owed."""
    return {"income": income, "rate": rate, "tax": income * rate}

agent = LlmAgent(
    name="tax_agent",
    model="gemini-2.5-flash",
    instruction="Calculate tax for given income and rates.",
    tools=[calculate_tax],
)

async def main():
    runner = InMemoryRunner(
        agent=agent,
        app_name="tax",
        plugins=[
            DebugLoggingPlugin(
                output_path="./test_debug.yaml",
                include_session_state=True,
                include_system_instruction=True,
            )
        ],
    )
    await runner.session_service.create_session(
        app_name="tax", user_id="u1", session_id="s1"
    )
    await runner.run_debug(
        "What's the tax on an income of $50,000 at 25% rate?",
        user_id="u1", session_id="s1",
    )
    # Read ./test_debug.yaml to see the full LLM trace

asyncio.run(main())
```

### Example 3 — combining both plugins for production observability

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.plugins.auto_tracing_plugin import AutoTracingPlugin
from google.adk.plugins.debug_logging_plugin import DebugLoggingPlugin

agent = LlmAgent(
    name="production_agent",
    model="gemini-2.5-flash",
    instruction="Production agent with full observability.",
)

import os
runner = Runner(
    app_name="prod",
    agent=agent,
    session_service=InMemorySessionService(),
    plugins=[
        AutoTracingPlugin(),          # spans → your OTel collector
        DebugLoggingPlugin(
            output_path="/var/log/adk/debug.yaml",
            include_session_state=False,   # reduce file size in production
        ),
    ],
)
```

### `ContextFilterPlugin` — filter context by event type or author

```python
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin

# Only include events from specific agents in the LLM context window
plugin = ContextFilterPlugin(
    included_authors=["user", "root_agent"],  # filter out sub-agent noise
)
```

### Gotchas

- `AutoTracingPlugin` instruments modules **lazily** on the first `before_run_callback`. Modules imported after the first run will not be traced unless they share a prefix with already-discovered scopes.
- `AutoTracingPlugin` uses a threading lock (`_lock`) — safe for concurrent runners sharing the same plugin instance, but adds a small lock-contention cost.
- `DebugLoggingPlugin` **appends** to `output_path` across process restarts. In CI, delete or rotate the file between test runs to avoid stale data.
- The YAML output contains full system instructions which can be verbose. Set `include_system_instruction=False` to reduce file size if your instructions are stable.

---

## Version notes

Verified against **google-adk==2.2.0** (June 2026). All constructor signatures, field names, and default values in this document were read from the installed source at `/usr/local/lib/python3.11/dist-packages/google/adk/`.

Previous: [Class deep dives — vol. 14 →](./google_adk_class_deep_dives_v14/)

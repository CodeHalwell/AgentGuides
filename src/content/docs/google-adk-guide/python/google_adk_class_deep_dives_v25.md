---
title: "Class deep dives — volume 25 (FunctionTool, BaseToolset/ToolPredicate, LlmRequest, LlmResponse, GEPARootAgentOptimizer, TelemetryContext, CacheMetadata, GCSToolset/GCSAdminToolset, BaseAuthenticatedTool, BaseLlm)"
description: "Source-verified deep dives into 10 google-adk 2.3.0 classes: FunctionTool (Pydantic arg coercion, confirmation callables, live streaming); BaseToolset and ToolPredicate (invocation-scoped caching, prefix injection, process_llm_request hook); LlmRequest (append_instructions, append_tools, set_output_schema, tools_dict exclusion); LlmResponse (camelCase aliases, create() factory, partial streaming, go_away); GEPARootAgentOptimizer and Sampler (GEPA adapter, checkpoint resumption, Pareto front results); TelemetryContext and start_as_current_node_span (OTel span dispatch by node type, associated_event_ids); CacheMetadata (frozen model, active vs fingerprint-only state, expire_soon); GCSToolset and GCSAdminToolset (Capabilities enum, DEFAULT_GCS_TOOL_NAME_PREFIX, GoogleTool wrapping); BaseAuthenticatedTool (CredentialManager wiring, abstract _run_async_impl, response_for_auth_required); BaseLlm (abstract generate_content_async, streaming contract, connect(), LLMRegistry.register())."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 25"
  order: 94
---

Source-verified against **google-adk==2.3.0** (installed from PyPI, June 2026). Every field name, signature, constant, and code example is drawn from the installed package source at `/usr/local/lib/python3.11/dist-packages/google/adk/`.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `FunctionTool` | `google.adk.tools.function_tool` | Stable |
| 2 | `BaseToolset` + `ToolPredicate` | `google.adk.tools.base_toolset` | Stable |
| 3 | `LlmRequest` | `google.adk.models.llm_request` | Stable |
| 4 | `LlmResponse` | `google.adk.models.llm_response` | Stable |
| 5 | `GEPARootAgentOptimizer` + `GEPARootAgentOptimizerConfig` + `Sampler` + `OptimizerResult` + `AgentWithScores` | `google.adk.optimization.*` | `@experimental` |
| 6 | `TelemetryContext` + `start_as_current_node_span` | `google.adk.telemetry.node_tracing` | Stable |
| 7 | `CacheMetadata` | `google.adk.models.cache_metadata` | Stable |
| 8 | `GCSToolset` + `GCSAdminToolset` + `GCSToolSettings` + `Capabilities` | `google.adk.integrations.gcs.*` | `@experimental` |
| 9 | `BaseAuthenticatedTool` | `google.adk.tools.base_authenticated_tool` | `@experimental` |
| 10 | `BaseLlm` | `google.adk.models.base_llm` | Stable |

---

## 1 · `FunctionTool`

**Source:** `google.adk.tools.function_tool`

`FunctionTool` wraps any Python callable as an ADK tool. It auto-extracts the name and docstring, detects the context parameter by type annotation, and handles Pydantic arg coercion before invoking the function.

### Constructor (source-verified)

```python
FunctionTool(
    func: Callable[..., Any],
    *,
    require_confirmation: Union[bool, Callable[..., bool]] = False,
)
```

### Key behaviours (source-verified)

| Behaviour | Detail |
|---|---|
| `_context_param_name` | Detected via `find_context_parameter(func)` by type annotation; falls back to `'tool_context'` |
| `_ignore_params` | `[self._context_param_name, 'input_stream']` — stripped from the LLM function declaration |
| `_preprocess_args()` | Coerces `dict` → Pydantic model for `BaseModel`-typed params; handles `Optional[T]` and `list[BaseModel]` |
| `_get_mandatory_args()` | Returns params without defaults that are not `VAR_POSITIONAL`/`VAR_KEYWORD`; missing mandatory args return an error dict, not an exception |
| `require_confirmation` | `bool` or `Callable[..., bool]`; if callable, called with `args_to_call` before invocation |
| Confirmation denied | Sets `tool_context.actions.skip_summarization = True`, returns `{'error': 'This tool call requires confirmation...'}` |
| `run_async()` | Filters `args_to_call` to only params present in the function signature |
| `_call_live()` | Streams from async generator; reads `input_stream` from `invocation_context.active_streaming_tools[self.name].stream` |
| `_get_declaration()` | Calls `build_function_declaration()` then wraps in `types.FunctionDeclaration.model_validate()` |

### Example 1: Basic FunctionTool with Pydantic arg coercion

```python
from pydantic import BaseModel
from google.adk.tools.function_tool import FunctionTool
from google.adk.agents import LlmAgent

class SearchParams(BaseModel):
    query: str
    max_results: int = 10
    include_snippets: bool = True

def web_search(params: SearchParams, tool_context) -> dict:
    """Search the web for the given query and return structured results.

    Args:
        params: Search parameters including query and result settings.
        tool_context: Injected ADK tool context (ignored by LLM declaration).

    Returns:
        A dict with the search results.
    """
    # FunctionTool._preprocess_args() auto-coerces the LLM's dict payload
    # to a SearchParams instance before this function is called.
    print(f"Searching for: {params.query!r}, max={params.max_results}")
    return {
        "results": [
            {"title": "Example", "snippet": "...", "url": "https://example.com"}
        ],
        "total": 1,
    }

# FunctionTool auto-extracts name="web_search" from func.__name__
# and strips 'tool_context' + 'input_stream' from the LLM declaration
search_tool = FunctionTool(func=web_search)

agent = LlmAgent(
    name="searcher",
    model="gemini-2.5-flash",
    instruction="Use web_search to answer questions.",
    tools=[search_tool],
)
```

### Example 2: Dynamic confirmation callable

```python
from google.adk.tools.function_tool import FunctionTool

def transfer_funds(
    from_account: str,
    to_account: str,
    amount: float,
    tool_context,
) -> dict:
    """Transfer funds between accounts.

    Args:
        from_account: Source account ID.
        to_account: Destination account ID.
        amount: Transfer amount in USD.
        tool_context: Injected ADK tool context.
    """
    # This function is only reached after confirmation is approved.
    return {"status": "ok", "transferred": amount}

def needs_confirmation(**kwargs) -> bool:
    # FunctionTool._invoke_callable expands args as **kwargs, not a single dict.
    # tool_context is also injected, so use .get() / kwargs directly.
    try:
        amount = float(kwargs.get("amount", 0))
    except (ValueError, TypeError):
        amount = 0.0
    return amount > 500.0   # require confirmation for transfers > $500

transfer_tool = FunctionTool(
    func=transfer_funds,
    require_confirmation=needs_confirmation,
)
# For a $1000 transfer: FunctionTool calls needs_confirmation(from_account=..., amount=1000, ...)
# → returns True → tool_context.request_confirmation() is called
# → returns {'error': 'This tool call requires confirmation...'} to the agent
# → agent surfaces it to the user for approval
```

### Example 3: Declaration inspection and tool wrapping

```python
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.base_tool import BaseTool

def calculate_tax(income: float, tax_rate: float = 0.2) -> dict:
    """Calculate income tax.

    Args:
        income: Annual gross income in USD.
        tax_rate: Tax rate as a decimal (e.g. 0.2 for 20%).

    Returns:
        Tax owed and net income.
    """
    tax = income * tax_rate
    return {"tax_owed": tax, "net_income": income - tax}

tool = FunctionTool(func=calculate_tax)

# Inspect the function declaration that will be sent to the LLM
declaration = tool._get_declaration()
print(f"Tool name: {declaration.name}")          # "calculate_tax"
print(f"Description: {declaration.description}")
if declaration.parameters:
    for prop_name in declaration.parameters.properties:
        print(f"  Param: {prop_name}")

# Check which params are mandatory (no default value)
mandatory = tool._get_mandatory_args()
print(f"Mandatory args: {mandatory}")   # ["income"] — tax_rate has a default

# Wrap in a custom class for additional metadata
class AuditedFunctionTool(FunctionTool):
    def __init__(self, func, audit_tag: str, **kwargs):
        super().__init__(func, **kwargs)
        self.audit_tag = audit_tag

audited_tax_tool = AuditedFunctionTool(
    func=calculate_tax,
    audit_tag="finance-v1",
)
```

### Example 4: Live streaming tool with input_stream

```python
from typing import AsyncIterator
from google.adk.tools.function_tool import FunctionTool
from google.adk.agents import LlmAgent

async def transcribe_audio(
    language: str,
    input_stream: AsyncIterator[bytes],
    tool_context,
) -> AsyncIterator[str]:
    """Transcribe a live audio stream.

    Args:
        language: BCP-47 language tag, e.g. 'en-US'.
        input_stream: Live audio byte stream injected by ADK for live tools.
        tool_context: Injected ADK tool context.

    Yields:
        Transcription text chunks as they arrive.
    """
    # FunctionTool._call_live() populates input_stream from
    # invocation_context.active_streaming_tools[tool.name].stream
    # 'input_stream' is in _ignore_params so it never appears in the LLM declaration
    async for audio_chunk in input_stream:
        # Send chunk to a hypothetical ASR service
        yield f"[transcript chunk for {language}: {len(audio_chunk)} bytes]"

live_transcribe_tool = FunctionTool(func=transcribe_audio)

agent = LlmAgent(
    name="live_assistant",
    model="gemini-2.5-flash",
    instruction="Transcribe user audio and respond.",
    tools=[live_transcribe_tool],
)
```

---

## 2 · `BaseToolset` + `ToolPredicate`

**Source:** `google.adk.tools.base_toolset`

`BaseToolset` is the abstract base for all multi-tool collections. `ToolPredicate` is a `@runtime_checkable` Protocol used to dynamically filter which tools are exposed to the LLM.

### `ToolPredicate` (source-verified)

```python
@runtime_checkable
class ToolPredicate(Protocol):
    def __call__(
        self, tool: BaseTool, readonly_context: Optional[ReadonlyContext] = None
    ) -> bool: ...
```

Any callable with this signature is automatically a `ToolPredicate` — no subclassing required.

### `BaseToolset.__init__` (source-verified)

```python
BaseToolset(
    *,
    tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
    tool_name_prefix: Optional[str] = None,
)
```

### Key behaviours (source-verified)

| Behaviour | Detail |
|---|---|
| `_cached_invocation_id` + `_cached_prefixed_tools` | Per-invocation cache; avoids repeated `get_tools()` calls within the same invocation |
| `get_tools_with_prefix()` | `@final` — cannot be overridden; calls `get_tools()` then applies prefix if set |
| Prefix injection | Shallow-copies each tool; sets `tool_copy.name = f"{prefix}_{tool.name}"`; patches `_get_declaration()` via closure to also rename the declaration |
| `_is_tool_selected()` | `list[str]` → `tool.name in list`; `ToolPredicate` → calls it; `None` → always True |
| `get_auth_config()` | Returns `None` by default; override to return an `AuthConfig` for credential-gated toolsets |
| `process_llm_request()` | No-op by default; override for toolset-level LLM request mutation (e.g. `ComputerUseToolset`) |
| `from_config()` | Raises `ValueError` by default; override for YAML config loading |
| `close()` | Async no-op; override to release connections/resources |

### Example 1: Custom toolset with ToolPredicate filter (role-based access)

```python
from typing import Optional
from google.adk.tools.base_toolset import BaseToolset, ToolPredicate
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents import LlmAgent

# --- Define tools ---
def list_reports(tool_context) -> dict:
    """List available reports."""
    return {"reports": ["q1.pdf", "q2.pdf"]}

def delete_report(name: str, tool_context) -> dict:
    """Delete a report by name.

    Args:
        name: Report filename to delete.
        tool_context: Injected ADK tool context.
    """
    return {"deleted": name}

# --- Define a ToolPredicate ---
# Any callable matching (tool, readonly_context=None) -> bool is a ToolPredicate
def admin_only_predicate(
    tool: BaseTool, readonly_context: Optional[ReadonlyContext] = None
) -> bool:
    """Only expose delete tools when the session state marks the user as admin."""
    if readonly_context is None:
        return True
    is_admin = readonly_context.state.get("is_admin", False)
    if "delete" in tool.name and not is_admin:
        return False
    return True

# --- Build the toolset ---
class ReportToolset(BaseToolset):
    async def get_tools(
        self, readonly_context: Optional[ReadonlyContext] = None
    ) -> list[BaseTool]:
        tools = [
            FunctionTool(func=list_reports),
            FunctionTool(func=delete_report),
        ]
        # _is_tool_selected() is called by callers; apply it here explicitly
        return [t for t in tools if self._is_tool_selected(t, readonly_context)]

toolset = ReportToolset(tool_filter=admin_only_predicate)

agent = LlmAgent(
    name="report_agent",
    model="gemini-2.5-flash",
    instruction="Help users manage reports.",
    tools=[toolset],
)
```

### Example 2: Toolset with tool_name_prefix to avoid collisions

```python
from typing import Optional
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents import LlmAgent

def search(query: str, tool_context) -> dict:
    """Search the knowledge base.

    Args:
        query: Search query string.
        tool_context: Injected ADK tool context.
    """
    return {"results": []}

class KnowledgeBaseToolset(BaseToolset):
    async def get_tools(
        self, readonly_context: Optional[ReadonlyContext] = None
    ) -> list[BaseTool]:
        return [FunctionTool(func=search)]

class WebToolset(BaseToolset):
    async def get_tools(
        self, readonly_context: Optional[ReadonlyContext] = None
    ) -> list[BaseTool]:
        return [FunctionTool(func=search)]

# Without prefixes, both toolsets expose a tool named "search" — collision.
# With prefixes, the LLM sees "kb_search" and "web_search" — no collision.
kb_toolset = KnowledgeBaseToolset(tool_name_prefix="kb")
web_toolset = WebToolset(tool_name_prefix="web")

# get_tools_with_prefix() is @final — it calls get_tools() then renames each tool:
#   tool_copy.name = "kb_search"  (or "web_search")
#   tool_copy._get_declaration().name = "kb_search"  (via closure)

agent = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction="Use kb_search for internal docs and web_search for external info.",
    tools=[kb_toolset, web_toolset],
)
```

### Example 3: Custom toolset implementing from_config() for YAML loading

```python
from typing import Optional
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.agents.readonly_context import ReadonlyContext

class ConfigurableApiToolset(BaseToolset):
    """A toolset that can be instantiated from a YAML config block."""

    def __init__(self, *, base_url: str, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self._base_url = base_url
        self._api_key = api_key

    async def get_tools(
        self, readonly_context: Optional[ReadonlyContext] = None
    ) -> list[BaseTool]:
        def fetch_data(endpoint: str, tool_context) -> dict:
            """Fetch data from the configured API endpoint.

            Args:
                endpoint: API path to fetch (e.g. '/items').
                tool_context: Injected ADK tool context.
            """
            import urllib.request
            url = self._base_url.rstrip("/") + endpoint
            # Real implementation would use httpx / aiohttp
            return {"url": url, "status": "ok"}

        return [FunctionTool(func=fetch_data)]

    @classmethod
    def from_config(cls, config, config_abs_path: str):
        # config is a ToolArgsConfig dict-like object produced by the YAML loader
        # Example YAML:
        #   toolset:
        #     type: ConfigurableApiToolset
        #     base_url: "https://api.example.com"
        #     api_key: "${ENV_API_KEY}"
        return cls(
            base_url=config.get("base_url", ""),
            api_key=config.get("api_key", ""),
        )
```

### Example 4: Toolset with process_llm_request() hook

```python
from typing import Optional
from google.genai import types
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.tool_context import ToolContext

def get_weather(city: str, tool_context) -> dict:
    """Get current weather for a city.

    Args:
        city: City name to query.
        tool_context: Injected ADK tool context.
    """
    return {"city": city, "temp_c": 22, "condition": "sunny"}

class WeatherToolset(BaseToolset):
    """Toolset that injects a reminder into every LLM request."""

    async def get_tools(
        self, readonly_context: Optional[ReadonlyContext] = None
    ) -> list[BaseTool]:
        return [FunctionTool(func=get_weather)]

    async def process_llm_request(
        self, *, tool_context: ToolContext, llm_request: LlmRequest
    ) -> None:
        # Called before each LLM call; mutate llm_request here.
        # Example: append a reminder instruction to every request.
        llm_request.append_instructions([
            "Always include the temperature unit (Celsius or Fahrenheit) in weather responses."
        ])

toolset = WeatherToolset()
```

---

## 3 · `LlmRequest`

**Source:** `google.adk.models.llm_request`

`LlmRequest` is the mutable envelope passed to every LLM call. Callbacks and processors mutate it directly to inject instructions, tools, output schemas, and cache configuration.

### Field reference (source-verified)

```python
class LlmRequest(BaseModel):
    model: Optional[str] = None
    contents: list[types.Content] = Field(default_factory=list)
    config: types.GenerateContentConfig = Field(default_factory=types.GenerateContentConfig)
    live_connect_config: types.LiveConnectConfig = Field(default_factory=types.LiveConnectConfig)
    tools_dict: dict[str, BaseTool] = Field(default_factory=dict, exclude=True)
    cache_config: Optional[ContextCacheConfig] = None
    cache_metadata: Optional[CacheMetadata] = None
    cacheable_contents_token_count: Optional[int] = None
    previous_interaction_id: Optional[str] = None
```

`tools_dict` is `exclude=True` — it is never serialized to JSON; it exists so that processors can look up the actual `BaseTool` object by name during the request lifecycle.

### `append_instructions` (source-verified)

- `list[str]` → concatenates with `\n\n` into `config.system_instruction`
- `types.Content` → extracts text parts into system instruction; returns non-text parts (inline_data, file_data) as `list[types.Content]` user contents added to `llm_request.contents`

### `append_tools` (source-verified)

Calls `tool._get_declaration()` per tool; adds `tools_dict[name] = tool`; appends declarations to the first `types.Tool(function_declarations=...)` found in `config.tools`, or creates a new one if none exists.

### `set_output_schema` (source-verified)

Sets `config.response_schema = schema` and `config.response_mime_type = "application/json"`. Accepts `type[BaseModel]`, `list[type[BaseModel]]`, `list[primitive]`, `dict`, or `types.Schema`.

### Example 1: Reading LlmRequest in before_model_callback

```python
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.agents import LlmAgent

def inject_user_context(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """Append dynamic instructions based on session state."""
    user_tier = callback_context.state.get("subscription_tier", "free")
    language = callback_context.state.get("preferred_language", "en")

    instructions = [f"User tier: {user_tier}. Language preference: {language}."]
    if user_tier == "premium":
        instructions.append("This user has access to premium features.")

    llm_request.append_instructions(instructions)
    return None  # None = proceed with the mutated request

agent = LlmAgent(
    name="context_aware",
    model="gemini-2.5-flash",
    instruction="Assist the user based on their subscription tier.",
    before_model_callback=inject_user_context,
)
```

### Example 2: Mutating tools_dict in a custom processor

```python
from typing import Optional
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.tool_context import ToolContext

def admin_reset(tool_context) -> str:
    """Reset all user data (admin only)."""
    return "Data reset complete."

class ConditionalAdminToolset(BaseToolset):
    """Injects an admin tool only when the session state grants admin access."""

    async def get_tools(
        self, readonly_context: Optional[ReadonlyContext] = None
    ) -> list[BaseTool]:
        # Return base tools; admin tool is injected via process_llm_request
        return []

    async def process_llm_request(
        self, *, tool_context: ToolContext, llm_request: LlmRequest
    ) -> None:
        is_admin = tool_context.state.get("is_admin", False)
        if is_admin:
            # Inject the tool into both tools_dict and the function declarations
            admin_tool = FunctionTool(func=admin_reset)
            llm_request.append_tools([admin_tool])
            # tools_dict now has "admin_reset" → tool object available to dispatcher
            print(f"Admin tools injected: {list(llm_request.tools_dict.keys())}")
```

### Example 3: Using set_output_schema for structured JSON output

```python
from pydantic import BaseModel
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.agents import LlmAgent

class SentimentResult(BaseModel):
    sentiment: str        # "positive", "negative", or "neutral"
    confidence: float     # 0.0 – 1.0
    key_phrases: list[str]

def enforce_sentiment_schema(
    callback_context: CallbackContext, llm_request: LlmRequest
):
    # Force the model to return SentimentResult-shaped JSON on every call
    llm_request.set_output_schema(SentimentResult)
    # Equivalent to:
    #   llm_request.config.response_schema = SentimentResult
    #   llm_request.config.response_mime_type = "application/json"
    return None

agent = LlmAgent(
    name="sentiment_analyzer",
    model="gemini-2.5-flash",
    instruction="Analyse the sentiment of the provided text.",
    before_model_callback=enforce_sentiment_schema,
)
```

### Example 4: Reading cache_metadata to monitor cache hit status

```python
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.runners import Runner
from google.adk.apps.app import App
from google.adk.agents import LlmAgent
from google.adk.agents.context_cache_config import ContextCacheConfig

def log_cache_state(
    callback_context: CallbackContext, llm_request: LlmRequest
):
    """Log whether a prior-turn cache is being reused."""
    meta = llm_request.cache_metadata
    if meta is None:
        print("No cache metadata — first turn or caching not configured.")
    elif meta.cache_name is None:
        # Fingerprint-only state: no active cache yet
        print(f"Cache fingerprint computed: {meta.fingerprint[:8]}... "
              f"({meta.contents_count} contents)")
    else:
        # Active cache state
        print(f"Reusing cache: {meta.cache_name.split('/')[-1]}, "
              f"invocations_used={meta.invocations_used}, "
              f"expire_soon={meta.expire_soon}")
    return None

agent = LlmAgent(
    name="cached_agent",
    model="gemini-2.5-flash",
    instruction="You have a long system instruction here...",
    before_model_callback=log_cache_state,
)

app = App(
    name="cache-demo",
    root_agent=agent,
    context_cache_config=ContextCacheConfig(cache_intervals=5, ttl_seconds=1800),
)
```

---

## 4 · `LlmResponse`

**Source:** `google.adk.models.llm_response`

`LlmResponse` is the base response class for all ADK model calls. `Event` (used by the runner) extends it. The camelCase alias generator means JSON serialization uses camelCase keys while Python code uses snake_case attributes.

### Model config (source-verified)

```python
model_config = ConfigDict(
    extra='forbid',
    alias_generator=alias_generators.to_camel,
    populate_by_name=True,
)
```

`populate_by_name=True` means both `partial` and `"partial"` (snake_case) work alongside the camelCase alias `"partial"`.

### Field reference (source-verified)

| Field | Type | Notes |
|---|---|---|
| `model_version` | `Optional[str]` | Model version used to generate the response |
| `content` | `Optional[types.Content]` | Model output (text, function calls, function responses) |
| `grounding_metadata` | `Optional[types.GroundingMetadata]` | Search grounding data |
| `partial` | `Optional[bool]` | `True` = streaming chunk; `False`/`None` = final |
| `turn_complete` | `Optional[bool]` | Live mode turn completion |
| `turn_complete_reason` | `Optional[types.TurnCompleteReason]` | Live mode only |
| `finish_reason` | `Optional[types.FinishReason]` | Standard finish reason |
| `error_code` | `Optional[str]` | Error code (varies by model) |
| `error_message` | `Optional[str]` | Human-readable error description |
| `interrupted` | `Optional[bool]` | True when user interrupted bidi streaming |
| `custom_metadata` | `Optional[dict[str, Any]]` | Arbitrary JSON-serializable key-value pairs |
| `usage_metadata` | `Optional[types.GenerateContentResponseUsageMetadata]` | Token counts |
| `live_session_resumption_update` | `Optional[types.LiveServerSessionResumptionUpdate]` | Live session resume token |
| `live_session_id` | `Optional[str]` | Live session ID |
| `go_away` | `Optional[types.LiveServerGoAway]` | Server-initiated live session termination signal |
| `input_transcription` | `Optional[types.Transcription]` | Audio transcription of user input |
| `output_transcription` | `Optional[types.Transcription]` | Audio transcription of model output |
| `avg_logprobs` | `Optional[float]` | Average log probability of generated tokens |
| `logprobs_result` | `Optional[types.LogprobsResult]` | Per-token log probabilities |
| `cache_metadata` | `Optional[CacheMetadata]` | Cache hit/miss info populated by context cache manager |
| `citation_metadata` | `Optional[types.CitationMetadata]` | Citation data for grounded responses |
| `interaction_id` | `Optional[str]` | Interactions API interaction ID for stateful chaining |

### `LlmResponse.create()` — static factory (source-verified)

```python
# Handles 4 cases:
# 1. Candidates with content → LlmResponse(content=..., grounding_metadata=..., ...)
# 2. Candidates with non-STOP finish_reason → LlmResponse(error_code=..., error_message=...)
# 3. No candidates + prompt_feedback → LlmResponse(error_code=block_reason, ...)
# 4. Empty candidates + no prompt_feedback → LlmResponse(content=Content(role='model', parts=[]))
```

### Example 1: After-model callback inspecting function calls and injecting metrics

```python
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.agents import LlmAgent
import time

_call_log: list[dict] = []

def track_tool_calls(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse | None:
    """Log every function call the model emits."""
    func_calls = llm_response.get_function_calls()
    if func_calls:
        for fc in func_calls:
            _call_log.append({
                "agent": callback_context.agent_name,
                "tool": fc.name,
                "args": fc.args,
                "timestamp": time.time(),
            })
        print(f"[metrics] {len(func_calls)} tool call(s) by {callback_context.agent_name}")
    return None  # None = pass the response through unchanged

agent = LlmAgent(
    name="tooling_agent",
    model="gemini-2.5-flash",
    instruction="Use tools to answer questions.",
    after_model_callback=track_tool_calls,
)
```

### Example 2: Handling partial responses in streaming mode

```python
import asyncio
from google.genai import types
from google.adk.runners import Runner
from google.adk.apps.app import App
from google.adk.agents import LlmAgent

agent = LlmAgent(name="streamer", model="gemini-2.5-flash", instruction="Respond verbosely.")
app = App(name="stream-demo", root_agent=agent)

async def stream_response():
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    svc = InMemorySessionService()
    runner = Runner(app=app, session_service=svc)
    session = await svc.create_session(app_name="stream-demo", user_id="u1")

    accumulated_text = ""

    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(
            role="user", parts=[types.Part(text="Tell me about the solar system.")]
        ),
    ):
        if event.partial:
            # Intermediate chunk — accumulate text
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        accumulated_text += part.text
                        print(part.text, end="", flush=True)
        elif event.is_final_response():
            # Final response — partial=False — identical to non-streaming output
            print()  # newline after streaming
            print(f"[Final] Total chars: {len(accumulated_text)}")
            if event.usage_metadata:
                print(f"Tokens: {event.usage_metadata.total_token_count}")
```

### Example 3: Checking error_code and error_message in an error handler callback

```python
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.adk.models.llm_request import LlmRequest
from google.adk.agents import LlmAgent
from google.genai import types

def handle_model_error(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse | None:
    """Intercept and handle model errors gracefully."""
    if llm_response.error_code is not None:
        error_code = llm_response.error_code
        error_msg = llm_response.error_message or "Unknown error"
        print(f"[error] code={error_code}, message={error_msg!r}")

        # For safety-filter blocks, return a canned response instead
        if str(error_code) in ("SAFETY", "BLOCKED_REASON_SAFETY"):
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="I cannot help with that request.")],
                ),
                partial=False,
            )
    return None  # pass through all non-error responses

agent = LlmAgent(
    name="safe_agent",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
    after_model_callback=handle_model_error,
)
```

### Example 4: Using interaction_id for Interactions API chaining

```python
import asyncio
from google.genai import types
from google.adk.runners import Runner
from google.adk.apps.app import App
from google.adk.agents import LlmAgent
from google.adk.sessions.in_memory_session_service import InMemorySessionService

agent = LlmAgent(name="chat", model="gemini-2.5-flash", instruction="Have a conversation.")
app = App(name="interactions-demo", root_agent=agent)

async def chained_conversation():
    svc = InMemorySessionService()
    runner = Runner(app=app, session_service=svc)
    session = await svc.create_session(app_name="interactions-demo", user_id="u1")

    last_interaction_id: str | None = None

    for user_message in ["Hello!", "What is 2 + 2?", "Thanks, goodbye."]:
        async for event in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part(text=user_message)]
            ),
        ):
            if event.is_final_response():
                if event.interaction_id:
                    # Store interaction_id; pass as previous_interaction_id on next turn
                    # via LlmRequest.previous_interaction_id for stateful Interactions API
                    last_interaction_id = event.interaction_id
                    print(f"[interaction_id] {last_interaction_id}")
                if event.content and event.content.parts:
                    print(f"Agent: {event.content.parts[0].text}")
```

---

## 5 · `GEPARootAgentOptimizer` + supporting types

**Source:** `google.adk.optimization.gepa_root_agent_optimizer`, `google.adk.optimization.agent_optimizer`, `google.adk.optimization.sampler`, `google.adk.optimization.data_types`

`GEPARootAgentOptimizer` automatically improves a root agent's instruction using the GEPA (Gradient-based Evolution for Prompt Adaptation) framework. All classes in this group are `@experimental`.

### `GEPARootAgentOptimizerConfig` (source-verified)

```python
class GEPARootAgentOptimizerConfig(BaseModel):
    optimizer_model: str = "gemini-3.5-flash"
    model_configuration: genai_types.GenerateContentConfig  # defaults to HIGH thinking
    max_metric_calls: int = 100
    reflection_minibatch_size: int = 3
    run_dir: str | None = None   # set to enable checkpoint resumption
```

### Data-type hierarchy (source-verified)

| Class | Fields | Notes |
|---|---|---|
| `SamplingResult` | `scores: dict[str, float]` | Base; map from example UID → score |
| `UnstructuredSamplingResult` | `+ data: Optional[dict[str, dict[str, Any]]]` | Adds per-example trajectory/metric data |
| `AgentWithScores` | `optimized_agent: Agent`, `overall_score: Optional[float]` | One candidate on the Pareto front |
| `OptimizerResult[AgentWithScoresT]` | `optimized_agents: list[AgentWithScoresT]` | Pareto front — not necessarily a single best |
| `GEPARootAgentOptimizerResult` | `+ gepa_result: dict[str, Any] \| None` | Raw GEPA output dict |

### Internal candidate keys (source-verified)

```python
_AGENT_PROMPT_KEY = "agent_prompt"
_SKILL_KEY_TEMPLATE = "skill_instructions:{skill_name}"
# The optimizer builds a seed_candidate dict with these keys
# and proposes new values via the optimizer LLM + GEPA adapter
```

### Example 1: Implementing a custom Sampler

```python
import asyncio
from typing import Literal, Optional
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.apps.app import App
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.optimization.sampler import Sampler
from google.adk.optimization.data_types import UnstructuredSamplingResult
from google.genai import types

# Eval dataset: list of (uid, input, expected_output) tuples
EVAL_DATA = [
    ("ex1", "What is the capital of France?", "Paris"),
    ("ex2", "Translate 'hello' to Spanish.", "hola"),
    ("ex3", "What is 7 * 8?", "56"),
]
TRAIN_IDS = ["ex1", "ex2"]
VAL_IDS = ["ex3"]

class LocalEvalSampler(Sampler[UnstructuredSamplingResult]):
    """Evaluates a candidate agent against a local dataset."""

    def get_train_example_ids(self) -> list[str]:
        return TRAIN_IDS

    def get_validation_example_ids(self) -> list[str]:
        return VAL_IDS

    async def sample_and_score(
        self,
        candidate: LlmAgent,
        example_set: Literal["train", "validation"] = "validation",
        batch: Optional[list[str]] = None,
        capture_full_eval_data: bool = False,
    ) -> UnstructuredSamplingResult:
        ids_to_eval = batch or (TRAIN_IDS if example_set == "train" else VAL_IDS)
        svc = InMemorySessionService()
        runner = Runner(
            app=App(name="eval", root_agent=candidate),
            session_service=svc,
        )

        scores: dict[str, float] = {}
        data: dict[str, dict] = {}

        for uid in ids_to_eval:
            example = next(e for e in EVAL_DATA if e[0] == uid)
            _, user_input, expected = example

            session = await svc.create_session(app_name="eval", user_id="eval_user")
            response_text = ""

            async for event in runner.run_async(
                user_id="eval_user",
                session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text=user_input)]
                ),
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    response_text = event.content.parts[0].text or ""

            # Simple exact-match scoring
            score = 1.0 if expected.lower() in response_text.lower() else 0.0
            scores[uid] = score

            if capture_full_eval_data:
                data[uid] = {
                    "input": user_input,
                    "expected": expected,
                    "actual": response_text,
                    "score": score,
                }

        return UnstructuredSamplingResult(scores=scores, data=data if capture_full_eval_data else None)
```

### Example 2: Running GEPARootAgentOptimizer with checkpoint resumption

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.optimization.gepa_root_agent_optimizer import (
    GEPARootAgentOptimizer,
    GEPARootAgentOptimizerConfig,
)
# Requires: pip install google-adk[optimization]

async def run_optimization():
    initial_agent = LlmAgent(
        name="qa_agent",
        model="gemini-2.5-flash",
        instruction="Answer the user's question as accurately as possible.",
    )

    config = GEPARootAgentOptimizerConfig(
        optimizer_model="gemini-3.5-flash",
        max_metric_calls=50,
        reflection_minibatch_size=3,
        # Set run_dir to resume from checkpoint if the process is interrupted
        run_dir="./optimizer_checkpoints/qa_agent_v1",
    )

    optimizer = GEPARootAgentOptimizer(config=config)
    sampler = LocalEvalSampler()  # from Example 1 above

    result = await optimizer.optimize(
        initial_agent=initial_agent,
        sampler=sampler,
    )

    print(f"Optimization complete. Pareto front size: {len(result.optimized_agents)}")
    if result.gepa_result:
        print(f"GEPA raw result keys: {list(result.gepa_result.keys())}")

    return result
```

### Example 3: Reading OptimizerResult and picking the best agent

```python
from google.adk.optimization.data_types import OptimizerResult, AgentWithScores

def pick_best_agent(result) -> AgentWithScores | None:
    """Select the agent with the highest overall_score from the Pareto front."""
    candidates = result.optimized_agents
    if not candidates:
        return None

    # Filter out candidates where overall_score is None
    scored = [c for c in candidates if c.overall_score is not None]
    if not scored:
        # Fall back to first candidate if no scores available
        return candidates[0]

    best = max(scored, key=lambda c: c.overall_score)
    print(f"Best agent score: {best.overall_score:.4f}")
    print(f"Optimized instruction: {best.optimized_agent.instruction[:120]}...")
    return best

# Usage after optimization:
# result = await optimizer.optimize(initial_agent, sampler)
# best = pick_best_agent(result)
# if best:
#     deploy_agent(best.optimized_agent)
```

### Example 4: UnstructuredSamplingResult to capture trajectories

```python
from google.adk.optimization.data_types import UnstructuredSamplingResult

# When capture_full_eval_data=True, return trajectory data per example.
# The GEPA adapter uses this data in make_reflective_dataset() and
# propose_new_texts() to generate improved prompts.

def build_sampling_result_with_trajectories(
    scores: dict[str, float],
    trajectories: dict[str, dict],
) -> UnstructuredSamplingResult:
    """Wrap scores and trajectory data for the GEPA optimizer."""
    # 'data' maps example UID → JSON-serializable dict
    # Recommended contents: inputs, tool call history, model outputs, metric values
    return UnstructuredSamplingResult(
        scores=scores,
        data={
            uid: {
                "input": trajectories[uid].get("input"),
                "tool_calls": trajectories[uid].get("tool_calls", []),
                "final_output": trajectories[uid].get("final_output"),
                "metric_scores": {
                    "correctness": scores[uid],
                    "latency_ms": trajectories[uid].get("latency_ms"),
                },
            }
            for uid in scores
        },
    )

# Example:
result = build_sampling_result_with_trajectories(
    scores={"ex1": 0.9, "ex2": 0.6},
    trajectories={
        "ex1": {"input": "Q1", "tool_calls": [], "final_output": "A1", "latency_ms": 320},
        "ex2": {"input": "Q2", "tool_calls": [{"name": "search", "args": {}}], "final_output": "A2", "latency_ms": 890},
    },
)
print(result.scores)   # {'ex1': 0.9, 'ex2': 0.6}
```

---

## 6 · `TelemetryContext` + `start_as_current_node_span`

**Source:** `google.adk.telemetry.node_tracing`

`TelemetryContext` is a frozen dataclass that carries an OTel context alongside a list of event IDs emitted within a node's span. `start_as_current_node_span` is an async context manager that dispatches to the correct span type based on the node class.

### `TelemetryContext` (source-verified)

```python
@dataclass(frozen=True)
class TelemetryContext:
    otel_context: context_api.Context
    _associated_event_ids: list[str] = field(default_factory=list)

    def add_event(self, event: Event) -> None:
        self._associated_event_ids.append(event.id)
```

### Span dispatch logic (source-verified)

| Node type | Span created | Span name | Key attributes |
|---|---|---|---|
| `BaseAgent` | None — uses existing `otel_context` | N/A | N/A |
| `Workflow` | New span | `"invoke_workflow {workflow.name}"` | `GEN_AI_OPERATION_NAME: "invoke_workflow"`, `"gen_ai.workflow.name": workflow.name`, `GEN_AI_CONVERSATION_ID: session.id` |
| Other `BaseNode` | New span | `"invoke_node {node.name}"` | `GEN_AI_OPERATION_NAME: "invoke_node"`, `GEN_AI_CONVERSATION_ID: session.id` |

On span exit, if `len(telemetry_context._associated_event_ids) > 0`, the span attribute `"gcp.vertex.agent.associated_event_ids"` is set.

Semconv alignment:
- `invoke_agent` → OTel semconv 1.36 (backwards compatibility)
- `invoke_workflow` → OTel semconv 1.41
- `invoke_node` → not yet in any semconv release

### Example 1: Reading the current OTel span inside a custom node

```python
from opentelemetry import trace
from google.adk.workflow._base_node import BaseNode
from google.adk.agents.context import Context

class InstrumentedNode(BaseNode):
    """A custom node that adds custom attributes to the current OTel span."""

    async def _run_impl(self, context: Context):
        # The OTel span for this node was started by start_as_current_node_span()
        # before _run_impl is called. Access it via the standard OTel API.
        span = trace.get_current_span()

        if span.is_recording():
            span.set_attribute("custom.node.input_length", len(str(context.user_content)))
            span.set_attribute("custom.node.session_id", context.session.id)

        # Perform node logic
        result = await self._do_work(context)

        if span.is_recording():
            span.set_attribute("custom.node.output_length", len(str(result)))

        return result

    async def _do_work(self, context: Context):
        return {"processed": True}
```

### Example 2: Configuring a trace exporter for Google Cloud Trace

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
# Requires: pip install google-cloud-trace opentelemetry-exporter-gcp-trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

from google.adk.runners import Runner
from google.adk.apps.app import App
from google.adk.agents import LlmAgent

def setup_cloud_trace(project_id: str):
    """Configure OTel to export ADK node spans to Google Cloud Trace."""
    exporter = CloudTraceSpanExporter(project_id=project_id)
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

# Call once at startup before creating the Runner
setup_cloud_trace(project_id="my-gcp-project")

agent = LlmAgent(
    name="traced_agent",
    model="gemini-2.5-flash",
    instruction="Help with tasks.",
)
app = App(name="traced-app", root_agent=agent)
# ADK telemetry will now emit invoke_agent / invoke_workflow / invoke_node
# spans to Cloud Trace automatically during runner.run_async()
```

### Example 3: Custom BaseNode subclass that adds timing metrics to span attributes

```python
import time
from opentelemetry import trace
from google.adk.workflow._base_node import BaseNode
from google.adk.agents.context import Context

class TimedNode(BaseNode):
    """Wraps node execution with timing attributes on the OTel span."""

    async def _run_impl(self, context: Context):
        span = trace.get_current_span()
        start_ns = time.perf_counter_ns()

        try:
            result = await self._execute(context)
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

            if span.is_recording():
                span.set_attribute("node.duration_ms", round(elapsed_ms, 2))
                span.set_attribute("node.success", True)

            return result

        except Exception as exc:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            if span.is_recording():
                span.set_attribute("node.duration_ms", round(elapsed_ms, 2))
                span.set_attribute("node.success", False)
                span.set_attribute("node.error_type", type(exc).__name__)
                span.record_exception(exc)
            raise

    async def _execute(self, context: Context):
        # Subclass-specific logic goes here
        return {"result": "ok"}
```

---

## 7 · `CacheMetadata`

**Source:** `google.adk.models.cache_metadata`

`CacheMetadata` is a frozen Pydantic model that describes the cache state associated with an LLM request or response. It exists in one of two mutually exclusive states enforced by a model validator.

### States and field rules (source-verified)

| State | `cache_name` | `expire_time` | `invocations_used` | `created_at` | `contents_count` |
|---|---|---|---|---|---|
| **Active cache** | Full resource name | Unix timestamp | Count ≥ 0 | Unix timestamp | Number of cached contents |
| **Fingerprint-only** | `None` | `None` | `None` | `None` | Total contents in request |

The model validator `_enforce_active_state_invariant` raises `ValueError` if any of `(cache_name, expire_time, invocations_used)` is set while others are `None`.

### `expire_soon` property (source-verified)

```python
@property
def expire_soon(self) -> bool:
    if self.expire_time is None:
        return False
    buffer_seconds = 120  # 2-minute buffer
    return time.time() > (self.expire_time - buffer_seconds)
```

Returns `False` for fingerprint-only instances (no `expire_time`).

### `__str__` output (source-verified)

```python
# Fingerprint-only:  "Fingerprint-only: 5 contents, fingerprint=a1b2c3d4..."
# Active cache:      "Cache 456: used 3 invocations, cached 5 contents, expires in 14.3min"
```

### Example 1: Reading CacheMetadata from LlmResponse in after_model_callback

```python
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.adk.agents import LlmAgent

_cache_stats = {"hits": 0, "misses": 0}

def track_cache_hits(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse | None:
    """Track cache hit rate across invocations."""
    meta = llm_response.cache_metadata

    if meta is None or meta.cache_name is None:
        _cache_stats["misses"] += 1
    else:
        _cache_stats["hits"] += 1
        total = _cache_stats["hits"] + _cache_stats["misses"]
        hit_rate = _cache_stats["hits"] / total * 100
        print(
            f"Cache hit | {meta} | "
            f"session hit rate: {hit_rate:.1f}%"
        )
    return None

agent = LlmAgent(
    name="cache_tracked",
    model="gemini-2.5-flash",
    instruction="Answer questions about large documents.",
    after_model_callback=track_cache_hits,
)
```

### Example 2: Cache warmup check using expire_soon before a batch run

```python
import asyncio
import time
from google.adk.models.cache_metadata import CacheMetadata

def is_cache_warm(meta: CacheMetadata | None) -> bool:
    """Return True only if there is an active, non-expiring cache."""
    if meta is None:
        return False
    if meta.cache_name is None:
        return False   # Fingerprint-only: no active cache
    if meta.expire_soon:
        print(f"Cache expires soon: {meta}")
        return False
    return True

async def run_batch_with_warmup_check(runner, sessions, warmup_meta: CacheMetadata | None):
    """Warn if cache is cold before starting a long batch run."""
    if not is_cache_warm(warmup_meta):
        print(
            "WARNING: Cache is cold or expiring. The first batch turns will "
            "not benefit from context caching. Consider running a warmup turn first."
        )
    else:
        time_left = (warmup_meta.expire_time - time.time()) / 60
        print(
            f"Cache is warm ({time_left:.1f} min remaining, "
            f"{warmup_meta.invocations_used} prior invocations). Starting batch."
        )

    for session_id in sessions:
        pass  # process each session
```

### Example 3: Using CacheMetadata.model_copy() to pass cache state between invocations

```python
from google.adk.models.cache_metadata import CacheMetadata

# CacheMetadata is frozen=True — use model_copy() to create a modified version.
# The context cache manager does this internally when updating invocation counts.

def increment_invocations(meta: CacheMetadata) -> CacheMetadata:
    """Return a new CacheMetadata with invocations_used incremented by 1."""
    if meta.invocations_used is None:
        return meta  # fingerprint-only state; no invocation counter
    return meta.model_copy(update={"invocations_used": meta.invocations_used + 1})

# Example usage:
active_meta = CacheMetadata(
    cache_name="projects/123/locations/us-central1/cachedContents/456",
    expire_time=9999999999.0,
    fingerprint="a1b2c3d4e5f67890",
    invocations_used=3,
    contents_count=10,
    created_at=1700000000.0,
)

updated = increment_invocations(active_meta)
print(updated.invocations_used)   # 4
print(str(updated))               # "Cache 456: used 4 invocations, cached 10 contents, expires in Xmin"
```

---

## 8 · `GCSToolset` + `GCSAdminToolset` + `GCSToolSettings` + `Capabilities`

**Source:** `google.adk.integrations.gcs.storage_toolset`, `google.adk.integrations.gcs.admin_toolset`, `google.adk.integrations.gcs.settings`

`GCSToolset` and `GCSAdminToolset` provide tools for Google Cloud Storage interactions. Both are `@experimental` and require `pip install google-adk[google-cloud-storage]`.

### Constructor signatures (source-verified)

```python
GCSToolset(
    *,
    tool_filter: ToolPredicate | list[str] | None = None,
    credentials_config: GCSCredentialsConfig | None = None,
    gcs_tool_settings: GCSToolSettings | None = None,
)

GCSAdminToolset(
    *,
    tool_filter: ToolPredicate | list[str] | None = None,
    credentials_config: GCSCredentialsConfig | None = None,
    gcs_tool_settings: GCSToolSettings | None = None,
)
```

Both pass `tool_name_prefix=DEFAULT_GCS_TOOL_NAME_PREFIX` (`"gcs"`) to `BaseToolset.__init__`, so all tool names are prefixed `gcs_`.

### Tool inventory (source-verified)

| Toolset | Capabilities required | Tool names (after gcs_ prefix) |
|---|---|---|
| `GCSToolset` | `READ_ONLY` or `READ_WRITE` | `gcs_get_bucket`, `gcs_get_object_data`, `gcs_get_object_metadata`, `gcs_list_objects` |
| `GCSToolset` | `READ_WRITE` only | + `gcs_create_object`, `gcs_delete_objects` |
| `GCSAdminToolset` | `READ_ONLY` or `READ_WRITE` | `gcs_list_buckets` |
| `GCSAdminToolset` | `READ_WRITE` only | + `gcs_create_bucket`, `gcs_update_bucket`, `gcs_delete_bucket` |

### `GCSToolSettings` + `Capabilities` (source-verified)

```python
class Capabilities(Enum):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"

class GCSToolSettings(BaseModel):
    capabilities: list[Capabilities] = [Capabilities.READ_ONLY]
```

### Example 1: Read-only GCSToolset for a document reading agent

```python
from google.adk.integrations.gcs.storage_toolset import GCSToolset  # @experimental
from google.adk.integrations.gcs.settings import GCSToolSettings, Capabilities
from google.adk.agents import LlmAgent

# Read-only: agent can list objects, read content and metadata, inspect buckets
read_toolset = GCSToolset(
    gcs_tool_settings=GCSToolSettings(
        capabilities=[Capabilities.READ_ONLY]
    ),
)
# Tools exposed: gcs_get_bucket, gcs_get_object_data, gcs_get_object_metadata, gcs_list_objects

doc_agent = LlmAgent(
    name="doc_reader",
    model="gemini-2.5-flash",
    instruction=(
        "You have access to GCS. Use gcs_list_objects to find documents, "
        "then gcs_get_object_data to read them."
    ),
    tools=[read_toolset],
)
```

### Example 2: Read-write GCSToolset for a report generation agent

```python
from google.adk.integrations.gcs.storage_toolset import GCSToolset  # @experimental
from google.adk.integrations.gcs.settings import GCSToolSettings, Capabilities
from google.adk.integrations.gcs.gcs_credentials import GCSCredentialsConfig
from google.adk.agents import LlmAgent

# Read-write: agent can also upload new objects and delete existing ones
rw_toolset = GCSToolset(
    gcs_tool_settings=GCSToolSettings(
        capabilities=[Capabilities.READ_WRITE]
    ),
    # credentials_config=None means use Application Default Credentials (ADC)
)
# Tools exposed: gcs_get_bucket, gcs_get_object_data, gcs_get_object_metadata,
#                gcs_list_objects, gcs_create_object, gcs_delete_objects

report_agent = LlmAgent(
    name="report_writer",
    model="gemini-2.5-flash",
    instruction=(
        "Generate weekly reports and upload them to GCS using gcs_create_object. "
        "Store reports in the 'reports/weekly/' prefix."
    ),
    tools=[rw_toolset],
)
```

### Example 3: GCSAdminToolset combined with GCSToolset for full lifecycle management

```python
from google.adk.integrations.gcs.storage_toolset import GCSToolset   # @experimental
from google.adk.integrations.gcs.admin_toolset import GCSAdminToolset  # @experimental
from google.adk.integrations.gcs.settings import GCSToolSettings, Capabilities
from google.adk.agents import LlmAgent

storage_toolset = GCSToolset(
    gcs_tool_settings=GCSToolSettings(capabilities=[Capabilities.READ_WRITE])
)
admin_toolset = GCSAdminToolset(
    gcs_tool_settings=GCSToolSettings(capabilities=[Capabilities.READ_WRITE])
)
# Combined tools:
#   Storage: gcs_get_bucket, gcs_get_object_data, gcs_get_object_metadata,
#            gcs_list_objects, gcs_create_object, gcs_delete_objects
#   Admin:   gcs_list_buckets, gcs_create_bucket, gcs_update_bucket, gcs_delete_bucket

infra_agent = LlmAgent(
    name="gcs_infra_agent",
    model="gemini-2.5-flash",
    instruction=(
        "Manage GCS infrastructure. Use admin tools to create/delete buckets "
        "and storage tools to manage objects within buckets."
    ),
    tools=[storage_toolset, admin_toolset],
)
```

### Example 4: ToolPredicate filter to restrict deletes to admin sessions

```python
from typing import Optional
from google.adk.tools.base_tool import BaseTool
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.integrations.gcs.storage_toolset import GCSToolset   # @experimental
from google.adk.integrations.gcs.admin_toolset import GCSAdminToolset  # @experimental
from google.adk.integrations.gcs.settings import GCSToolSettings, Capabilities
from google.adk.agents import LlmAgent

def no_deletes_for_non_admins(
    tool: BaseTool, readonly_context: Optional[ReadonlyContext] = None
) -> bool:
    """Block delete tools unless session state grants admin access."""
    if "delete" in tool.name:
        if readonly_context is None:
            return False   # deny deletes when context unavailable
        return readonly_context.state.get("is_admin", False)
    return True

# Apply the predicate to both toolsets
storage_toolset = GCSToolset(
    tool_filter=no_deletes_for_non_admins,
    gcs_tool_settings=GCSToolSettings(capabilities=[Capabilities.READ_WRITE]),
)
admin_toolset = GCSAdminToolset(
    tool_filter=no_deletes_for_non_admins,
    gcs_tool_settings=GCSToolSettings(capabilities=[Capabilities.READ_WRITE]),
)
# Non-admin sessions: gcs_delete_objects and gcs_delete_bucket are filtered out
# Admin sessions (is_admin=True): all tools are available

agent = LlmAgent(
    name="gcs_agent",
    model="gemini-2.5-flash",
    instruction="Manage GCS resources according to your permissions.",
    tools=[storage_toolset, admin_toolset],
)
```

---

## 9 · `BaseAuthenticatedTool`

**Source:** `google.adk.tools.base_authenticated_tool`

`BaseAuthenticatedTool` is a `@experimental` abstract base class for class-based tools that require credential retrieval before execution. It handles the credential lifecycle (request → obtain → use) so subclasses only implement the post-auth logic.

### Constructor (source-verified)

```python
@experimental(FeatureName.BASE_AUTHENTICATED_TOOL)
class BaseAuthenticatedTool(BaseTool):
    def __init__(
        self,
        *,
        name: str,
        description: str,
        auth_config: AuthConfig = None,
        response_for_auth_required: Optional[Union[dict[str, Any], str]] = None,
    ): ...
```

### `run_async()` flow (source-verified)

```
1. If _credentials_manager is None → skip auth → call _run_async_impl(credential=None)
2. Else: credential = await _credentials_manager.get_auth_credential(tool_context)
3. If credential is None:
     → await _credentials_manager.request_credential(tool_context)  # triggers auth flow
     → return _response_for_auth_required or "Pending User Authorization."
4. Else: return await _run_async_impl(args=args, tool_context=tool_context, credential=credential)
```

### `_run_async_impl` (source-verified)

```python
@abstractmethod
async def _run_async_impl(
    self,
    *,
    args: dict[str, Any],
    tool_context: ToolContext,
    credential: AuthCredential,
) -> Any:
    ...
```

### When to use `BaseAuthenticatedTool` vs `AuthenticatedFunctionTool`

| | `BaseAuthenticatedTool` | `AuthenticatedFunctionTool` |
|---|---|---|
| Tool style | Class-based (`_run_async_impl` override) | Function-wrapping (wraps a `Callable`) |
| Use case | Complex tools with state, multiple methods, or non-trivial logic | Simple function tools that need a credential injected |
| Subclassing | Required (`@abstractmethod`) | Not required |

### Example 1: Custom BaseAuthenticatedTool for a GitHub API tool (OAuth2)

```python
from typing import Any, Optional
from google.adk.tools.base_authenticated_tool import BaseAuthenticatedTool  # @experimental
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_schemes import OAuthGrantType, OpenIdConnectWithConfig
from google.adk.auth.auth_credential import AuthCredential
from google.adk.tools.tool_context import ToolContext

class GitHubIssueTool(BaseAuthenticatedTool):
    """Lists open GitHub issues using an OAuth2 access token."""

    def __init__(self, github_auth_config: AuthConfig):
        super().__init__(
            name="list_github_issues",
            description="List open issues in a GitHub repository.",
            auth_config=github_auth_config,
            response_for_auth_required={
                "status": "auth_required",
                "message": "Please authorise access to your GitHub account.",
            },
        )

    def _get_declaration(self):
        from google.genai import types
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "owner": types.Schema(type="STRING", description="Repository owner"),
                    "repo": types.Schema(type="STRING", description="Repository name"),
                },
                required=["owner", "repo"],
            ),
        )

    async def _run_async_impl(
        self,
        *,
        args: dict[str, Any],
        tool_context: ToolContext,
        credential: AuthCredential,
    ) -> Any:
        # credential is ready-to-use; extract the OAuth token
        token = credential.oauth2.access_token if credential and credential.oauth2 else None
        if not token:
            return {"error": "No OAuth token available."}

        owner = args.get("owner", "")
        repo = args.get("repo", "")
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/issues?state=open",
                headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"},
                timeout=10.0,
            )
            resp.raise_for_status()
            issues = resp.json()
        return {"issues": [{"number": i["number"], "title": i["title"]} for i in issues[:10]]}
```

### Example 2: BaseAuthenticatedTool with custom response_for_auth_required

```python
from typing import Any
from google.adk.tools.base_authenticated_tool import BaseAuthenticatedTool  # @experimental
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_credential import AuthCredential
from google.adk.tools.tool_context import ToolContext

class SalesforceQueryTool(BaseAuthenticatedTool):
    """Queries Salesforce data via SOQL."""

    def __init__(self, auth_config: AuthConfig):
        super().__init__(
            name="query_salesforce",
            description="Run a SOQL query against Salesforce.",
            auth_config=auth_config,
            # Custom string message returned when credentials are not yet available
            response_for_auth_required=(
                "Salesforce access is required. "
                "Please complete the OAuth2 flow to connect your Salesforce account. "
                "A login link has been sent to your email."
            ),
        )

    def _get_declaration(self):
        from google.genai import types
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type="OBJECT",
                properties={"soql": types.Schema(type="STRING", description="SOQL query string")},
                required=["soql"],
            ),
        )

    async def _run_async_impl(
        self, *, args: dict[str, Any], tool_context: ToolContext, credential: AuthCredential
    ) -> Any:
        soql = args.get("soql", "")
        token = credential.oauth2.access_token if credential and credential.oauth2 else None
        return {"query": soql, "status": "executed", "rows": []}
```

### Example 3: BaseAuthenticatedTool vs AuthenticatedFunctionTool

```python
# --- Option A: AuthenticatedFunctionTool (function-based) ---
# Use when you have a simple function and just need a credential injected.
from google.adk.auth.auth_tool import AuthConfig
# from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool

async def fetch_private_data(
    resource_id: str,
    tool_context,
    credential=None,  # injected by AuthenticatedFunctionTool
) -> dict:
    """Fetch a private resource by ID.

    Args:
        resource_id: The ID of the resource to fetch.
        tool_context: Injected ADK tool context.
        credential: Injected auth credential.
    """
    token = credential.oauth2.access_token if credential and credential.oauth2 else None
    return {"resource_id": resource_id, "data": "...", "authed": token is not None}

# auth_config = AuthConfig(...)
# tool = AuthenticatedFunctionTool(func=fetch_private_data, auth_config=auth_config)

# --- Option B: BaseAuthenticatedTool (class-based) ---
# Use when the tool has complex state, helper methods, or multiple operations.
from google.adk.tools.base_authenticated_tool import BaseAuthenticatedTool  # @experimental
from google.adk.auth.auth_credential import AuthCredential
from google.adk.tools.tool_context import ToolContext
from typing import Any

class PrivateDataTool(BaseAuthenticatedTool):
    def __init__(self, auth_config: AuthConfig, cache_ttl: int = 300):
        super().__init__(
            name="fetch_private_data",
            description="Fetch a private resource by ID.",
            auth_config=auth_config,
        )
        self._cache_ttl = cache_ttl   # class state not possible with function tools
        self._cache: dict = {}

    def _get_declaration(self):
        from google.genai import types
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type="OBJECT",
                properties={"resource_id": types.Schema(type="STRING")},
                required=["resource_id"],
            ),
        )

    async def _run_async_impl(
        self, *, args: dict[str, Any], tool_context: ToolContext, credential: AuthCredential
    ) -> Any:
        resource_id = args.get("resource_id", "")
        if resource_id in self._cache:
            return self._cache[resource_id]
        result = {"resource_id": resource_id, "data": "fetched"}
        self._cache[resource_id] = result
        return result
```

### Example 4: BaseAuthenticatedTool with service account credentials

```python
from typing import Any
from google.adk.tools.base_authenticated_tool import BaseAuthenticatedTool  # @experimental
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_credential import (
    AuthCredential, AuthCredentialTypes, ServiceAccount, ServiceAccountCredential,
)
from google.adk.auth.auth_schemes import CustomAuthScheme
from google.adk.tools.tool_context import ToolContext

# Service account auth config — use with BigQuery, Cloud Storage APIs, etc.
# auth_scheme must be an AuthScheme (OpenAPI SecurityScheme or CustomAuthScheme);
# ServiceAccount belongs in raw_auth_credential.service_account, not auth_scheme.
sa_auth_config = AuthConfig(
    auth_scheme=CustomAuthScheme(type="serviceAccount"),
    raw_auth_credential=AuthCredential(
        auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
        service_account=ServiceAccount(
            scopes=["https://www.googleapis.com/auth/bigquery.readonly"],
            service_account_credential=ServiceAccountCredential(
                # In production, load from environment or Secret Manager
                service_account_json="{...}",
            ),
        ),
    ),
)

class BigQueryTool(BaseAuthenticatedTool):
    """Queries BigQuery using a service account credential."""

    def __init__(self):
        super().__init__(
            name="query_bigquery",
            description="Run a SQL query against BigQuery.",
            auth_config=sa_auth_config,
            response_for_auth_required="Service account credentials are not configured.",
        )

    def _get_declaration(self):
        from google.genai import types
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "sql": types.Schema(type="STRING", description="Standard SQL query"),
                    "project": types.Schema(type="STRING", description="GCP project ID"),
                },
                required=["sql", "project"],
            ),
        )

    async def _run_async_impl(
        self, *, args: dict[str, Any], tool_context: ToolContext, credential: AuthCredential
    ) -> Any:
        # credential.service_account contains the ready-to-use SA credentials
        # In real usage: use google-cloud-bigquery with the credentials object
        sql = args.get("sql", "")
        project = args.get("project", "")
        return {"sql": sql, "project": project, "rows": [], "status": "executed"}
```

---

## 10 · `BaseLlm`

**Source:** `google.adk.models.base_llm`

`BaseLlm` is the abstract base for all model backends. Implement it to plug in any LLM — local Ollama, a proprietary API, or a mock for testing — while retaining full ADK capability (tools, callbacks, structured output).

### Constructor + model config (source-verified)

```python
class BaseLlm(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: str   # e.g. "gemini-2.5-flash", "llama3", "gpt-4o"
```

### `generate_content_async` contract (source-verified)

```
Non-streaming (stream=False):
  → yields exactly one LlmResponse with partial=False

Streaming (stream=True):
  → yields N LlmResponse with partial=True (intermediate chunks)
  → yields one final LlmResponse with partial=False
  → final partial=False response is identical to non-streaming response

Text:           streams incrementally as tokens arrive
Function calls: may arrive in separate partial=True chunks
Thoughts:       stream when thinking_config enabled
Consecutive parts of same type SHOULD merge; client must not rely on this
```

### Other methods (source-verified)

| Method | Default | Override |
|---|---|---|
| `generate_content_async(llm_request, stream=False)` | `@abstractmethod` | Required |
| `connect(llm_request)` | raises `NotImplementedError` | Optional — for live/bidi |
| `supported_models()` | returns `[]` (not auto-registered) | Optional — for `LLMRegistry` |
| `_maybe_append_user_content(llm_request)` | Appends `"Handle the requests..."` if contents empty; `"Continue processing..."` if last content not `'user'` | Rarely overridden |

### Example 1: Custom BaseLlm wrapping a local Ollama endpoint

```python
from __future__ import annotations
import json
from typing import AsyncGenerator
import httpx
from google.genai import types
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

class OllamaLlm(BaseLlm):
    """BaseLlm implementation for a local Ollama endpoint."""

    base_url: str = "http://localhost:11434"

    @classmethod
    def supported_models(cls) -> list[str]:
        # Regex patterns — models matching these are auto-registered to OllamaLlm
        return [r"ollama/.*", r"llama3.*", r"mistral.*", r"phi3.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self._maybe_append_user_content(llm_request)

        # Build Ollama-style prompt from contents
        messages = []
        if llm_request.config and llm_request.config.system_instruction:
            sys_inst = llm_request.config.system_instruction
            if not isinstance(sys_inst, str):
                parts = getattr(sys_inst, "parts", []) or []
                sys_inst = " ".join(p.text for p in parts if getattr(p, "text", None))
            if sys_inst:
                messages.append({
                    "role": "system",
                    "content": sys_inst,
                })
        for content in llm_request.contents:
            text = " ".join(p.text for p in (content.parts or []) if p.text)
            messages.append({"role": content.role or "user", "content": text})

        payload = {
            "model": self.model.removeprefix("ollama/"),
            "messages": messages,
            "stream": stream,
        }

        async with httpx.AsyncClient() as client:
            if not stream:
                resp = await client.post(
                    f"{self.base_url}/api/chat", json=payload, timeout=60.0
                )
                resp.raise_for_status()
                data = resp.json()
                text = data.get("message", {}).get("content", "")
                yield LlmResponse(
                    content=types.Content(role="model", parts=[types.Part(text=text)]),
                    partial=False,
                )
            else:
                async with client.stream(
                    "POST", f"{self.base_url}/api/chat",
                    json=payload, timeout=120.0
                ) as resp:
                    resp.raise_for_status()
                    accumulated = ""
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        accumulated += token
                        if not chunk.get("done", False):
                            yield LlmResponse(
                                content=types.Content(
                                    role="model", parts=[types.Part(text=token)]
                                ),
                                partial=True,
                            )
                    # Final partial=False response with full accumulated text
                    yield LlmResponse(
                        content=types.Content(
                            role="model", parts=[types.Part(text=accumulated)]
                        ),
                        partial=False,
                    )
```

### Example 2: Custom BaseLlm with streaming (partial=True chunks + final partial=False)

```python
from __future__ import annotations
from typing import AsyncGenerator
from google.genai import types
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

class EchoLlm(BaseLlm):
    """A testing LLM that streams the user's message back word by word."""

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"echo.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        # Extract last user message
        user_text = ""
        for content in reversed(llm_request.contents):
            if content.role == "user" and content.parts:
                user_text = " ".join(p.text for p in content.parts if p.text)
                break

        reply = f"Echo: {user_text}"
        words = reply.split()

        if not stream:
            # Non-streaming: yield exactly one response with partial=False
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text=reply)]),
                partial=False,
            )
        else:
            # Streaming: yield partial=True chunks, then final partial=False
            accumulated = ""
            for i, word in enumerate(words):
                token = word + (" " if i < len(words) - 1 else "")
                accumulated += token
                yield LlmResponse(
                    content=types.Content(role="model", parts=[types.Part(text=token)]),
                    partial=True,
                )
            # Final response — partial=False — identical to non-streaming output
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text=reply)]),
                partial=False,
            )
```

### Example 3: Registering a custom LLM with LLMRegistry

```python
from google.adk.models.registry import LLMRegistry
from google.adk.agents import LlmAgent

# Method 1: Auto-registration via supported_models() at import time
# When OllamaLlm is imported, call LLMRegistry.register(OllamaLlm) to register it.
# ADK does this automatically for built-in models during package initialization.
LLMRegistry.register(OllamaLlm)    # registers all patterns from supported_models()
LLMRegistry.register(EchoLlm)

# Method 2: Verify resolution works
resolved_cls = LLMRegistry.resolve("ollama/llama3:8b")
print(resolved_cls)   # <class 'OllamaLlm'>

resolved_echo = LLMRegistry.resolve("echo-test")
print(resolved_echo)  # <class 'EchoLlm'>

# Method 3: Use the model string directly in LlmAgent
# LLMRegistry.new_llm() is called by the runner when the agent needs an LLM instance
ollama_agent = LlmAgent(
    name="local_agent",
    model="ollama/llama3:8b",   # resolved to OllamaLlm(model="llama3:8b")
    instruction="You are a helpful assistant running locally.",
)

echo_agent = LlmAgent(
    name="echo_agent",
    model="echo-v1",            # resolved to EchoLlm(model="echo-v1")
    instruction="Echo test agent.",
)
```

### Example 4: Custom LLM with connect() override for a bidirectional streaming backend

```python
from __future__ import annotations
from typing import AsyncGenerator
from google.genai import types
from google.adk.models.base_llm import BaseLlm
from google.adk.models.base_llm_connection import BaseLlmConnection
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

class BidiConnection(BaseLlmConnection):
    """A stub bidirectional connection for a custom streaming backend."""

    def __init__(self, session_url: str):
        self._session_url = session_url
        self._closed = False

    async def send_history(self, history: list[types.Content]) -> None:
        """Send conversation history right after opening the connection."""
        print(f"[BidiConnection] Sending history ({len(history)} turns)")

    async def send_content(self, content: types.Content) -> None:
        """Send user content to the model mid-stream."""
        print(f"[BidiConnection] Sending to {self._session_url}: {content}")

    async def send_realtime(self, blob: types.Blob) -> None:
        """Send an audio chunk or video frame for realtime voice/video input."""
        print(f"[BidiConnection] Realtime blob: mime={blob.mime_type} size={len(blob.data or b'')}")

    async def receive(self) -> AsyncGenerator[LlmResponse, None]:
        """Receive streaming responses from the model."""
        # In a real implementation, this reads from a WebSocket or gRPC stream
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="Bidi response")]),
            partial=False,
            turn_complete=True,
        )

    async def close(self) -> None:
        self._closed = True

class BidiStreamingLlm(BaseLlm):
    """LLM backend with bidirectional streaming support."""

    server_url: str = "wss://bidi.example.com/v1/stream"

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"bidi/.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        # Standard (non-bidi) generation — required by @abstractmethod
        self._maybe_append_user_content(llm_request)
        yield LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part(text="Standard response")]
            ),
            partial=False,
        )

    def connect(self, llm_request: LlmRequest) -> BaseLlmConnection:
        """Override connect() to return a live bidirectional connection."""
        # The runner calls this instead of generate_content_async() for live agents
        session_url = f"{self.server_url}?model={self.model}"
        return BidiConnection(session_url=session_url)
```

---

## Quick-reference table

| What you want | Class / field | Module |
|---|---|---|
| Wrap a Python function as a tool | `FunctionTool(func=my_fn)` | `google.adk.tools.function_tool` |
| Coerce LLM dict args to Pydantic | `FunctionTool` — auto via `_preprocess_args()` | `google.adk.tools.function_tool` |
| Require human confirmation before tool runs | `FunctionTool(require_confirmation=True)` or `callable` | `google.adk.tools.function_tool` |
| Stream audio/video to a tool | `async def tool(input_stream: AsyncIterator, ...)` + `FunctionTool` | `google.adk.tools.function_tool` |
| Group multiple tools with filtering | `BaseToolset(tool_filter=...)` | `google.adk.tools.base_toolset` |
| Avoid tool name collisions across toolsets | `BaseToolset(tool_name_prefix="prefix")` | `google.adk.tools.base_toolset` |
| Inject a predicate for tool visibility | `ToolPredicate` protocol | `google.adk.tools.base_toolset` |
| Mutate the LLM request in a toolset | `BaseToolset.process_llm_request()` override | `google.adk.tools.base_toolset` |
| Append dynamic system instructions | `llm_request.append_instructions([...])` | `google.adk.models.llm_request` |
| Inject a tool into the request | `llm_request.append_tools([tool])` | `google.adk.models.llm_request` |
| Force structured JSON output | `llm_request.set_output_schema(MyModel)` | `google.adk.models.llm_request` |
| Access tool objects by name in request | `llm_request.tools_dict["tool_name"]` | `google.adk.models.llm_request` |
| Inspect model function calls | `llm_response.get_function_calls()` | `google.adk.models.llm_response` |
| Check if model was interrupted (live) | `llm_response.interrupted` | `google.adk.models.llm_response` |
| Handle server-side live disconnect | `llm_response.go_away` | `google.adk.models.llm_response` |
| Chain Interactions API turns | `llm_response.interaction_id` + `llm_request.previous_interaction_id` | `google.adk.models.*` |
| Optimize agent prompt automatically | `GEPARootAgentOptimizer(config).optimize(agent, sampler)` | `google.adk.optimization.*` |
| Provide eval data to optimizer | `UnstructuredSamplingResult(scores=..., data=...)` | `google.adk.optimization.data_types` |
| Resume optimizer from checkpoint | `GEPARootAgentOptimizerConfig(run_dir="./ckpt")` | `google.adk.optimization.gepa_root_agent_optimizer` |
| Add custom OTel attributes to node spans | `trace.get_current_span().set_attribute(...)` inside a `BaseNode` | `google.adk.telemetry.node_tracing` |
| Export ADK spans to Cloud Trace | `CloudTraceSpanExporter` + `TracerProvider` at startup | `google.adk.telemetry.node_tracing` |
| Check if context cache is about to expire | `cache_metadata.expire_soon` | `google.adk.models.cache_metadata` |
| Immutably update a frozen cache metadata | `cache_metadata.model_copy(update={...})` | `google.adk.models.cache_metadata` |
| Read GCS objects in an agent | `GCSToolset(gcs_tool_settings=GCSToolSettings([READ_ONLY]))` | `google.adk.integrations.gcs.storage_toolset` |
| Upload files to GCS in an agent | `GCSToolset(gcs_tool_settings=GCSToolSettings([READ_WRITE]))` | `google.adk.integrations.gcs.storage_toolset` |
| Manage GCS buckets in an agent | `GCSAdminToolset(...)` | `google.adk.integrations.gcs.admin_toolset` |
| Class-based tool with auth | `BaseAuthenticatedTool` subclass + `_run_async_impl` | `google.adk.tools.base_authenticated_tool` |
| Custom response when auth pending | `BaseAuthenticatedTool(response_for_auth_required=...)` | `google.adk.tools.base_authenticated_tool` |
| Custom LLM backend (any provider) | `BaseLlm` subclass + `generate_content_async` | `google.adk.models.base_llm` |
| Register custom LLM for auto-resolution | `BaseLlm.supported_models()` + `LLMRegistry.register(cls)` | `google.adk.models.base_llm`, `google.adk.models.registry` |
| Live bidi connection for custom LLM | `BaseLlm.connect()` override → `BaseLlmConnection` | `google.adk.models.base_llm` |

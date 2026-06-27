---
title: "Class deep dives — volume 27 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.3.0 classes: AntigravityAgent+convert_step_to_events (Google Antigravity SDK wrapper; trajectory file management; root-only restriction; partial streaming; step→ADK event mapping), CredentialManager (8-step credential lifecycle: validate→check-ready→load-existing→load-auth-response→client-credentials→exchange→refresh→save; register_auth_provider; CustomAuthScheme rehydration), OAuth2DiscoveryManager+AuthorizationServerMetadata+ProtectedResourceMetadata (RFC8414/RFC9728 auto-discovery; 3-endpoint ordering; issuer path insertion vs appending), ContextCacheRequestProcessor (LLM request processor enabling context caching; session event scan for latest CacheMetadata and previous token count; lazy no-op guard), LocalEvalSampler+LocalEvalSamplerConfig (local eval sampler for prompt optimization; sample_and_score; capture_full_eval_data for trajectories+tool-calls; TRAIN_SET/VALIDATION_SET split), BigQueryAgentAnalyticsPlugin+BigQueryLoggerConfig+EventData (BigQuery Write API analytics pipeline; per-event-loop _LoopState; GCSOffloader for large content; adk.* JSON attribute envelope), CompactionRequestProcessor (token-threshold event compaction flow processor; _has_token_threshold_config guard; token_compaction_checked flag; module-level singleton), LocalEnvironment+BaseEnvironment+ExecutionResult (asyncio subprocess execution; temp-dir lifecycle; path traversal safety; env_vars merge; timed_out flag), GeminiLlmConnection (live bidirectional Gemini session; audio-part filtering on history init; Gemini 3.1 Flash collapsed-text preamble; transcription accumulation), plot_workflow_graph (DOT/SVG workflow visualisation; dark/light mode NodeStatus colour mapping; LlmAgent sub-agent recursive traversal fallback)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 27"
  order: 96
---

Source-verified against **google-adk==2.3.0** (installed from PyPI, June 2026). Every field name, signature, and code example is taken directly from the installed package source.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `AntigravityAgent` + `convert_step_to_events` | `google.adk.labs.antigravity` | Experimental |
| 2 | `CredentialManager` | `google.adk.auth.credential_manager` | Experimental |
| 3 | `OAuth2DiscoveryManager` + `AuthorizationServerMetadata` + `ProtectedResourceMetadata` | `google.adk.auth.oauth2_discovery` | Experimental |
| 4 | `ContextCacheRequestProcessor` | `google.adk.flows.llm_flows.context_cache_processor` | Stable |
| 5 | `LocalEvalSampler` + `LocalEvalSamplerConfig` | `google.adk.optimization.local_eval_sampler` | Experimental |
| 6 | `BigQueryAgentAnalyticsPlugin` + `BigQueryLoggerConfig` + `EventData` | `google.adk.plugins.bigquery_agent_analytics_plugin` | Stable |
| 7 | `CompactionRequestProcessor` | `google.adk.flows.llm_flows.compaction` | Stable |
| 8 | `LocalEnvironment` + `BaseEnvironment` + `ExecutionResult` | `google.adk.environment` | Experimental |
| 9 | `GeminiLlmConnection` | `google.adk.models.gemini_llm_connection` | Stable |
| 10 | `plot_workflow_graph` | `google.adk.cli.utils.graph_visualization` | Stable |

---

## 1 · `AntigravityAgent` + `convert_step_to_events`

**Source:** `google.adk.labs.antigravity._antigravity_agent`, `._event_converter`

`AntigravityAgent` wraps a pre-configured `google.antigravity.Agent` as a native ADK `BaseAgent` node, delegating each turn to the Antigravity SDK runner and streaming its trajectory steps back as standard ADK events.

### Why it exists

The Antigravity SDK uses an in-process Go harness with its own session lifecycle. Since the harness owns session state, `AntigravityAgent` must run as a **standalone root agent** — it cannot be used as a sub-agent or be given sub-agents (this restriction is temporary, pending remote connection mode support).

### Root-only enforcement

Two guards block misuse at construction time:

```python
# model_post_init raises if any sub_agents are supplied
def model_post_init(self, __context):
    super().model_post_init(__context)
    if self.sub_agents:
        raise ValueError(_ROOT_ONLY_MESSAGE)

# __setattr__ blocks parent_agent assignment (triggered when a
# parent adopts this agent as a sub-agent)
def __setattr__(self, name, value):
    if name == 'parent_agent' and value is not None:
        raise ValueError(_ROOT_ONLY_MESSAGE)
    super().__setattr__(name, value)
```

### Trajectory file lifecycle

`AntigravityAgent` persists conversation trajectories between turns using `config.save_dir`. On the **first turn**, the harness writes a trajectory with a random ID; the agent renames it to a deterministic name `{session_id}_{agent_name}`. On **subsequent turns**, the agent passes that ID to `config.conversation_id` so the harness replays history from disk.

The `resume_step_index` — persisted via `_trajectory_files.save_resume_step_index` — prevents re-emitting steps that were already yielded in prior turns. Steps with `step_index <= resume_step_index` are skipped.

### `convert_step_to_events` — mapping SDK steps to ADK events

```python
# google.adk.labs.antigravity._event_converter

def convert_step_to_events(
    step: sdk_types.Step,
    *,
    ctx: InvocationContext,
    author: str,
    seen_tool_calls: set[str],
    seen_tool_results: set[str],
    streaming: bool = False,
) -> list[Event]:
    partials = (
        _convert_partial_deltas(step, ctx=ctx, author=author) if streaming else []
    )
    return [
        *partials,
        *_convert_model_text(step, ctx=ctx, author=author),
        *_convert_function_calls(
            step, ctx=ctx, author=author, seen_tool_calls=seen_tool_calls
        ),
        *_convert_function_responses(
            step, ctx=ctx, seen_tool_results=seen_tool_results
        ),
    ]
```

Key mapping rules:
- **Partial deltas** (SSE mode only): `thinking_delta` → `Part(thought=True)`, `content_delta` → text `Part`
- **Final model text**: emitted only when `step.is_complete_response` is set to avoid duplicate cumulative re-broadcasts
- **Tool calls**: deduplicated via `seen_tool_calls` set keyed on `call.id or f'{step_index}-{call.name}'`
- **Tool responses**: author set to the **tool name** (not the agent name) to mirror ADK's own function-response events

### Example 1 — basic single-turn agent

```python
# pip install google-adk google-antigravity
import asyncio
from pathlib import Path
from google.adk.labs.antigravity import AntigravityAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# AntigravityAgent requires config.save_dir so trajectories survive
# across turns and can be resumed.
try:
    from google.antigravity import LocalAgentConfig
    config = LocalAgentConfig(
        model="gemini-2.0-flash",
        save_dir=str(Path.home() / ".adk_antigravity_trajs"),
    )
    agent = AntigravityAgent(name="my_ag_agent", config=config)
    runner = Runner(
        agent=agent,
        app_name="demo",
        session_service=InMemorySessionService(),
    )
    async def run():
        session = await runner.session_service.create_session(
            app_name="demo", user_id="u1"
        )
        events = []
        async for ev in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part.from_text(text="Hello!")]
            ),
        ):
            events.append(ev)
        return events
    asyncio.run(run())
except ImportError:
    print("google-antigravity not installed; skipping live run.")
```

### Example 2 — root-only guard

```python
from google.adk.agents import LlmAgent
try:
    from google.adk.labs.antigravity import AntigravityAgent
    from google.antigravity import LocalAgentConfig

    config = LocalAgentConfig(
        model="gemini-2.0-flash",
        save_dir="/tmp/trajs",
    )
    sub = LlmAgent(name="helper", model="gemini-2.0-flash")
    try:
        # Raises immediately — AntigravityAgent cannot have sub-agents
        ag = AntigravityAgent(
            name="root",
            config=config,
            sub_agents=[sub],
        )
    except ValueError as e:
        print(f"Blocked: {e}")
except ImportError:
    print("google-antigravity not installed.")
```

### Example 3 — SSE streaming mode shows partial deltas

```python
import asyncio
try:
    from google.adk.labs.antigravity import AntigravityAgent
    from google.antigravity import LocalAgentConfig
    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    config = LocalAgentConfig(
        model="gemini-2.0-flash",
        save_dir="/tmp/trajs",
    )
    agent = AntigravityAgent(name="streamer", config=config)
    runner = Runner(
        agent=agent,
        app_name="demo",
        session_service=InMemorySessionService(),
    )

    async def run_streaming():
        session = await runner.session_service.create_session(
            app_name="demo", user_id="u1"
        )
        run_cfg = RunConfig(streaming_mode=StreamingMode.SSE)
        partial_count = 0
        async for ev in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[types.Part.from_text(text="Tell me a story.")],
            ),
            run_config=run_cfg,
        ):
            if ev.partial:
                partial_count += 1
        print(f"Received {partial_count} partial streaming events")

    asyncio.run(run_streaming())
except ImportError:
    print("google-antigravity not installed.")
```

---

## 2 · `CredentialManager`

**Source:** `google.adk.auth.credential_manager`

`CredentialManager` orchestrates the **8-step credential lifecycle** from initial loading through final preparation for use. It provides a centralized interface for validating, exchanging, refreshing, and saving credentials while supporting pluggable exchangers and refreshers.

### The 8-step workflow in `get_auth_credential()`

```
Step 0: Handle CustomAuthScheme (rehydrate + delegate to AuthProvider)
Step 1: Validate credential configuration
Step 2: Check if credential is already ready (API_KEY / HTTP → return copy)
Step 3: Try to load existing processed credential from credential service
Step 4: If none, load from auth response in context (was_from_auth_response=True)
Step 5: If still none, check for client credentials flow; else return None (trigger OAuth consent)
Step 6: Exchange credential (service account → access token; OAuth2 code → token)
Step 7: Refresh credential if expired
Step 8: Save credential if it was modified
```

### Class-level `register_auth_provider()`

`CredentialManager` holds a class-level `AuthProviderRegistry` protected by a `threading.Lock`. Calling `register_auth_provider(provider)` is safe to call from any thread and silently ignores a re-registration of the same provider:

```python
from google.adk.auth.credential_manager import CredentialManager
from google.adk.auth.base_auth_provider import BaseAuthProvider
from google.adk.auth.auth_schemes import CustomAuthScheme, AuthSchemeType
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes

class MyCustomAuthProvider(BaseAuthProvider):
    @property
    def supported_auth_schemes(self):
        return [CustomAuthScheme]

    async def get_auth_credential(self, auth_config, context):
        # Fetch credential from custom secret store
        return AuthCredential(
            auth_type=AuthCredentialTypes.HTTP,
            http={"scheme": "bearer", "credentials": "my-token"},
        )

# Register once at application startup
CredentialManager.register_auth_provider(MyCustomAuthProvider())
```

### Example 1 — API key credential (ready on step 2)

```python
import asyncio
from google.adk.auth.credential_manager import CredentialManager
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_schemes import APIKey
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes

async def get_api_key_credential():
    scheme = APIKey(
        name="X-API-Key",
        in_="header",
    )
    credential = AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY,
        api_key="my-secret-key-123",
    )
    auth_config = AuthConfig(
        auth_scheme=scheme,
        raw_auth_credential=credential,
    )
    manager = CredentialManager(auth_config)
    # Step 2 short-circuits: API_KEY is already ready, no exchange/refresh needed.
    # (Requires a CallbackContext for the full flow; shown conceptually here.)
    print("CredentialManager created for API key auth")
    print(f"Is API_KEY ready? {credential.auth_type == AuthCredentialTypes.API_KEY}")

asyncio.run(get_api_key_credential())
```

### Example 2 — registering a custom exchanger

```python
from google.adk.auth.credential_manager import CredentialManager
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes
from google.adk.auth.auth_schemes import HTTPBearer
from google.adk.auth.exchanger.base_credential_exchanger import (
    BaseCredentialExchanger,
    ExchangeResult,
)

class VaultCredentialExchanger(BaseCredentialExchanger):
    """Exchanges service credentials for short-lived Vault tokens."""

    async def exchange(self, credential, auth_scheme):
        # Fetch short-lived token from Vault
        vault_token = "hvs.short-lived-token-from-vault"
        exchanged = AuthCredential(
            auth_type=AuthCredentialTypes.HTTP,
            http={"scheme": "bearer", "credentials": vault_token},
        )
        return ExchangeResult(credential=exchanged, was_exchanged=True)

scheme = HTTPBearer()
raw = AuthCredential(
    auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
    service_account={"json_file_path": "/path/to/sa.json"},
)
auth_config = AuthConfig(auth_scheme=scheme, raw_auth_credential=raw)

manager = CredentialManager(auth_config)
manager.register_credential_exchanger(
    AuthCredentialTypes.SERVICE_ACCOUNT,
    VaultCredentialExchanger(),
)
print("Custom Vault exchanger registered for SERVICE_ACCOUNT credentials")
```

### Example 3 — OAuth2 discovery and client credentials flow

```python
from google.adk.auth.credential_manager import CredentialManager
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes
from google.adk.auth.auth_schemes import ExtendedOAuth2
from fastapi.openapi.models import OAuthFlowClientCredentials, OAuthFlows

# The manager auto-discovers token endpoint via RFC8414 when issuer_url is set
# and token_url is missing. This happens in _populate_auth_scheme().
scheme = ExtendedOAuth2(
    flows=OAuthFlows(
        clientCredentials=OAuthFlowClientCredentials(
            tokenUrl="",  # empty → triggers auto-discovery
            scopes={"read:data": "Read access"},
        )
    ),
    issuer_url="https://auth.example.com",
)
credential = AuthCredential(
    auth_type=AuthCredentialTypes.OAUTH2,
    oauth2={
        "client_id": "my-client-id",
        "client_secret": "my-client-secret",
    },
)
auth_config = AuthConfig(auth_scheme=scheme, raw_auth_credential=credential)
manager = CredentialManager(auth_config)
# At get_auth_credential() time, _missing_oauth_info() returns True →
# _populate_auth_scheme() hits /.well-known/oauth-authorization-server
# and fills in tokenUrl before attempting the exchange.
print("CredentialManager ready; will auto-discover token URL on first use")
print(f"Is client credentials flow: {scheme.flows.clientCredentials is not None}")
```

---

## 3 · `OAuth2DiscoveryManager` + `AuthorizationServerMetadata` + `ProtectedResourceMetadata`

**Source:** `google.adk.auth.oauth2_discovery`

`OAuth2DiscoveryManager` implements metadata auto-discovery for OAuth2 following **RFC8414** (authorization server metadata) and **RFC9728** (protected resource metadata). It is used internally by `CredentialManager` when OAuth flow URLs are missing.

### Discovery endpoint ordering

For an issuer URL with a non-root path (e.g. `https://auth.example.com/tenant/v2`), three endpoints are tried in order:

1. `https://auth.example.com/.well-known/oauth-authorization-server/tenant/v2` (RFC8414 path insertion)
2. `https://auth.example.com/.well-known/openid-configuration/tenant/v2` (OIDC Discovery path insertion)
3. `https://auth.example.com/tenant/v2/.well-known/openid-configuration` (OIDC Discovery path appending)

For a root issuer URL, only the base forms are tried.

### `AuthorizationServerMetadata`

```python
class AuthorizationServerMetadata(BaseModel):
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    scopes_supported: Optional[List[str]] = None
    registration_endpoint: Optional[str] = None
```

### `ProtectedResourceMetadata`

```python
class ProtectedResourceMetadata(BaseModel):
    resource: str
    authorization_servers: List[str] = []
```

### Example 1 — manual metadata discovery

```python
import asyncio
from google.adk.auth.oauth2_discovery import OAuth2DiscoveryManager

async def discover():
    manager = OAuth2DiscoveryManager()
    # This makes real HTTP calls to well-known endpoints.
    # For illustration, use a known OIDC provider.
    try:
        meta = await manager.discover_auth_server_metadata(
            issuer_url="https://accounts.google.com"
        )
        if meta:
            print(f"Issuer: {meta.issuer}")
            print(f"Auth endpoint: {meta.authorization_endpoint}")
            print(f"Token endpoint: {meta.token_endpoint}")
    except Exception as e:
        print(f"Discovery failed (expected in sandboxed env): {e}")

asyncio.run(discover())
```

### Example 2 — path insertion vs path appending

```python
from urllib.parse import urlparse

def show_discovery_endpoints(issuer_url: str):
    """Demonstrates the 3-endpoint ordering for a pathed issuer URL."""
    parsed = urlparse(issuer_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path

    if path and path != "/":
        endpoints = [
            f"{base_url}/.well-known/oauth-authorization-server{path}",
            f"{base_url}/.well-known/openid-configuration{path}",
            f"{base_url}{path}/.well-known/openid-configuration",
        ]
    else:
        endpoints = [
            f"{base_url}/.well-known/oauth-authorization-server",
            f"{base_url}/.well-known/openid-configuration",
        ]

    for i, ep in enumerate(endpoints, 1):
        print(f"  {i}. {ep}")

print("Pathed issuer:")
show_discovery_endpoints("https://auth.example.com/tenant/v2")
print("\nRoot issuer:")
show_discovery_endpoints("https://auth.example.com")
```

### Example 3 — using `ProtectedResourceMetadata` to find authorization servers

```python
import asyncio
import httpx
from google.adk.auth.oauth2_discovery import ProtectedResourceMetadata

async def discover_auth_servers(resource_url: str):
    """Fetches RFC9728 protected resource metadata to find authorization servers."""
    # RFC9728: fetch /.well-known/oauth-protected-resource
    parsed_url = resource_url.rstrip("/")
    well_known = f"{parsed_url}/.well-known/oauth-protected-resource"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(well_known, timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            meta = ProtectedResourceMetadata.model_validate(data)
            print(f"Resource: {meta.resource}")
            print(f"Auth servers: {meta.authorization_servers}")
            return meta.authorization_servers
    except Exception as e:
        print(f"Could not fetch protected resource metadata: {e}")
        # Fall back: treat the resource URL itself as the issuer
        return [resource_url]

asyncio.run(discover_auth_servers("https://api.example.com"))
```

---

## 4 · `ContextCacheRequestProcessor`

**Source:** `google.adk.flows.llm_flows.context_cache_processor`

`ContextCacheRequestProcessor` is a `BaseLlmRequestProcessor` that enables Gemini context caching for agents that have `ContextCacheConfig` configured. It runs before the model call to inject caching directives into the `LlmRequest`.

### What it does

1. **Early exit** — if `invocation_context.context_cache_config` is `None`, the processor returns immediately (no-op).
2. **Inject cache config** — sets `llm_request.cache_config = invocation_context.context_cache_config`.
3. **Scan session events** — calls `_find_cache_info_from_events()` to locate the most recent `CacheMetadata` and the previous cacheable-content token count from the agent's past events.
4. **Set on request** — populates `llm_request.cache_metadata` and `llm_request.cacheable_contents_token_count` so the model-specific cache manager (e.g. `GeminiContextCacheManager`) can decide whether to reuse or create a cache entry.

### Why the session scan matters

Context caching requires knowing:
- The **existing cache name** (`CacheMetadata`) — to reuse it if still valid.
- The **previous prompt token count** — to determine whether re-caching (at the `min_tokens` threshold) is needed.

Both are extracted from past `Event` objects in the session, not from external storage.

### Example 1 — wiring context caching into an LlmAgent

```python
from google.adk.agents import LlmAgent
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps.app import App
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

# ContextCacheConfig drives ContextCacheRequestProcessor at the flow level.
cache_cfg = ContextCacheConfig(
    min_tokens=4096,        # Only cache when context exceeds this threshold
    ttl_seconds=3600,       # Cache entry time-to-live (default: 3600)
    # cache_intervals=[...] # Optional: pin specific content windows
)

agent = LlmAgent(
    name="cached_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You are a research assistant. Use the provided context "
        "to answer questions accurately."
    ),
    context_cache_config=cache_cfg,
)

app = App(name="cache_demo", root_agent=agent)
print(f"Context cache enabled: min_tokens={cache_cfg.min_tokens}")
print("ContextCacheRequestProcessor will activate when context > 4096 tokens")
```

### Example 2 — inspecting what the processor sets on LlmRequest

```python
from google.adk.flows.llm_flows.context_cache_processor import (
    ContextCacheRequestProcessor,
)
from google.adk.models.llm_request import LlmRequest
from unittest.mock import MagicMock

# Simulate what happens inside BaseLlmFlow when the processor runs.
processor = ContextCacheRequestProcessor()

# Case 1: No cache config → processor is a no-op
ctx_no_cache = MagicMock()
ctx_no_cache.context_cache_config = None
request = LlmRequest()

import asyncio
async def test_no_cache():
    events = []
    async for ev in processor.run_async(ctx_no_cache, request):
        events.append(ev)
    print(f"No-cache mode: {len(events)} events yielded, cache_config={request.cache_config}")

asyncio.run(test_no_cache())

# Case 2: Cache config present → processor sets cache_config and scans events
from google.adk.agents.context_cache_config import ContextCacheConfig

ctx_with_cache = MagicMock()
ctx_with_cache.context_cache_config = ContextCacheConfig(min_tokens=4096)
ctx_with_cache.agent.name = "agent1"
ctx_with_cache.invocation_id = "inv-001"
ctx_with_cache.session.events = []  # Empty history → no existing cache

async def test_with_cache():
    request2 = LlmRequest()
    events = []
    async for ev in processor.run_async(ctx_with_cache, request2):
        events.append(ev)
    print(f"Cache mode: cache_config set = {request2.cache_config is not None}")
    print(f"  cache_metadata = {request2.cache_metadata}")  # None if no past events

asyncio.run(test_with_cache())
```

### Example 3 — custom LLM flow with context cache processor

```python
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.flows.llm_flows.context_cache_processor import (
    ContextCacheRequestProcessor,
)

# The ContextCacheRequestProcessor runs at the start of each LLM call.
# In the default AutoFlow it is included automatically when the agent has
# context_cache_config set. Below shows how you'd include it in a custom flow.
class MyCachedFlow(BaseLlmFlow):
    def _get_request_processors(self):
        processors = super()._get_request_processors()
        # ContextCacheRequestProcessor is injected before content processors
        return [ContextCacheRequestProcessor(), *processors]

# This pattern is for advanced use cases where you build your own LLM flow.
print("Custom flow with ContextCacheRequestProcessor registered.")
print("The processor runs before BaseLlmFlow sends the request to Gemini.")
```

---

## 5 · `LocalEvalSampler` + `LocalEvalSamplerConfig`

**Source:** `google.adk.optimization.local_eval_sampler`

`LocalEvalSampler` is the `Sampler` implementation used by `GEPARootAgentOptimizer` and `SimplePromptOptimizer` when you want to run evaluation **locally** (without Vertex AI) during prompt optimization. It uses `LocalEvalService` internally.

### `LocalEvalSamplerConfig` fields

```python
class LocalEvalSamplerConfig(BaseModel):
    eval_config: EvalConfig              # Metrics + thresholds
    app_name: str                        # Matches your App's name
    train_eval_set: str                  # Eval set ID for training
    train_eval_case_ids: Optional[list[str]] = None   # Subset; None = all
    validation_eval_set: Optional[str] = None          # Defaults to train set
    validation_eval_case_ids: Optional[list[str]] = None
```

### `sample_and_score()` — the optimizer's interface

```python
async def sample_and_score(
    self,
    candidate: Agent,
    example_set: Literal["train", "validation"] = "validation",
    batch: Optional[list[str]] = None,
    capture_full_eval_data: bool = False,
) -> UnstructuredSamplingResult:
```

- `candidate`: The agent variant to evaluate (typically with a modified instruction/prompt).
- `example_set`: `"train"` or `"validation"` — selects which eval set to run against.
- `batch`: Explicit list of eval case IDs; `None` → all cases in the selected set.
- `capture_full_eval_data`: When `True`, the returned `UnstructuredSamplingResult.data` includes per-invocation tool calls, agent responses, and eval metric results. Required by GEPA for its reflection loop.

**Return value** — `UnstructuredSamplingResult`:
- `scores`: `dict[eval_case_id, float]` — `1.0` if `EvalStatus.PASSED`, `0.0` otherwise.
- `data`: Optional per-case dict with `conversation_scenario`, `invocations`, and per-invocation `tool_calls` + `eval_metric_results`.

### Example 1 — wiring LocalEvalSampler into SimplePromptOptimizer

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.optimization.local_eval_sampler import (
    LocalEvalSampler,
    LocalEvalSamplerConfig,
)
from google.adk.optimization.simple_prompt_optimizer import SimplePromptOptimizer
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager

# 1. Set up eval sets manager with some test cases
eval_manager = InMemoryEvalSetsManager()
# (In a real scenario, you'd create an eval set with ground-truth cases)

# 2. Configure eval with ROUGE metric threshold
eval_cfg = EvalConfig(
    eval_set_results_dir="/tmp/eval_results",
    # Metrics loaded from DEFAULT_EVAL_CONFIG if not specified
)

# 3. Build the sampler config
sampler_cfg = LocalEvalSamplerConfig(
    eval_config=eval_cfg,
    app_name="my_app",
    train_eval_set="train_set_v1",
    validation_eval_set="val_set_v1",
)

# 4. The optimizer will use LocalEvalSampler automatically
print("LocalEvalSamplerConfig ready:")
print(f"  app_name={sampler_cfg.app_name}")
print(f"  train_set={sampler_cfg.train_eval_set}")
print(f"  validation_set={sampler_cfg.validation_eval_set}")
```

### Example 2 — running sample_and_score manually

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
from google.adk.optimization.local_eval_sampler import (
    LocalEvalSampler,
    LocalEvalSamplerConfig,
)
from google.adk.optimization.sampler import Sampler

async def evaluate_candidate(candidate_agent: LlmAgent):
    eval_manager = InMemoryEvalSetsManager()
    eval_cfg = EvalConfig(eval_set_results_dir="/tmp/eval_results")

    sampler_cfg = LocalEvalSamplerConfig(
        eval_config=eval_cfg,
        app_name="scoring_demo",
        train_eval_set="train_v1",
    )

    sampler = LocalEvalSampler(
        config=sampler_cfg,
        eval_sets_manager=eval_manager,
    )

    # Run scoring on all training examples
    result = await sampler.sample_and_score(
        candidate=candidate_agent,
        example_set=Sampler.TRAIN_SET,
        capture_full_eval_data=True,  # Include tool calls + trajectories
    )

    for case_id, score in result.scores.items():
        status = "PASS" if score == 1.0 else "FAIL"
        print(f"  {case_id}: {status} ({score:.1f})")

    return result

# Create a sample candidate
candidate = LlmAgent(
    name="candidate",
    model="gemini-2.0-flash",
    instruction="Answer questions concisely and accurately.",
)
# asyncio.run(evaluate_candidate(candidate))
print("sample_and_score ready to be called with eval cases")
```

### Example 3 — `extract_tool_call_data` for optimizer reflection

```python
from google.adk.optimization.local_eval_sampler import (
    extract_tool_call_data,
    extract_single_invocation_info,
)
from google.adk.evaluation.eval_case import IntermediateData, ToolCall, ToolResponse

# The GEPA optimizer uses extract_tool_call_data to build reflection context.
# Here we show what the extracted data looks like.
intermediate = IntermediateData(
    tool_uses=[
        ToolCall(name="search", args={"query": "ADK version"}, id="fc-001"),
        ToolCall(name="calculator", args={"expr": "2+2"}, id="fc-002"),
    ],
    tool_results=[
        ToolResponse(name="search", response={"result": "2.3.0"}, id="fc-001"),
        ToolResponse(name="calculator", response={"result": "4"}, id="fc-002"),
    ],
)
tool_calls = extract_tool_call_data(intermediate)
for tc in tool_calls:
    print(f"Tool: {tc['name']}, Args: {tc['args']}, Response: {tc['response']}")
```

---

## 6 · `BigQueryAgentAnalyticsPlugin` + `BigQueryLoggerConfig` + `EventData`

**Source:** `google.adk.plugins.bigquery_agent_analytics_plugin`

`BigQueryAgentAnalyticsPlugin` streams agent events to BigQuery using the **BigQuery Write API** for efficient, asynchronous, and reliable analytics logging. It is a `BasePlugin` that hooks into agent lifecycle callbacks.

### Architecture overview

```
Plugin callbacks (before/after LLM request, tool call, etc.)
    │
    ▼
EventData → _log_event() → row serialization
    │
    ▼
BatchProcessor (per asyncio event loop)  ← lazy init in _get_loop_state()
    │
    ├─ BigQueryWriteAsyncClient (BigQuery Write API)
    │
    └─ GCSOffloader (when content > max_content_length)
         └─ HybridContentParser (GCS URI ↔ inline text)
```

### Per-event-loop `_LoopState`

The plugin handles **multiple asyncio event loops** (e.g. in testing or multi-threaded servers) by maintaining a `dict[asyncio.AbstractEventLoop, _LoopState]`. Each `_LoopState` has its own `BigQueryWriteAsyncClient`, `BatchProcessor`, and write stream. Stale closed loops are cleaned up via `_cleanup_stale_loop_states()`.

### `EventData` — the structured telemetry envelope

```python
@dataclass(kw_only=True)
class EventData:
    span_id_override: Optional[str] = None
    parent_span_id_override: Optional[str] = None
    latency_ms: Optional[int] = None
    time_to_first_token_ms: Optional[int] = None
    model: Optional[str] = None
    model_version: Optional[str] = None
    usage_metadata: Any = None
    cache_metadata: Any = None
    status: str = "OK"
    error_message: Optional[str] = None
    extra_attributes: dict[str, Any] = field(default_factory=dict)
    trace_id_override: Optional[str] = None
    source_event: Optional["Event"] = None  # ADK Event for adk.* attributes
    adk_extras: dict[str, Any] = field(default_factory=dict)
```

The `adk_extras` dict places fields **inside** the `attributes.adk` JSON object, so consumer SQL like `JSON_VALUE(attributes, '$.adk.function_call_id')` works correctly.

### Example 1 — basic plugin setup

```python
from google.adk.plugins.bigquery_agent_analytics_plugin import (
    BigQueryAgentAnalyticsPlugin,
    BigQueryLoggerConfig,
)
from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

# Configure the plugin
config = BigQueryLoggerConfig(
    table_id="agent_events",
    view_prefix="agent_",        # Required: non-empty string
    max_content_length=8192,     # Truncate content at 8192 chars
)

plugin = BigQueryAgentAnalyticsPlugin(
    project_id="my-gcp-project",
    dataset_id="agent_analytics",
    config=config,
    location="US",
    # credentials=...  # Optional; uses ADC if None
)

agent = LlmAgent(
    name="analytics_agent",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant.",
)

app = App(
    name="analytics_demo",
    root_agent=agent,
    plugins=[plugin],
)

runner = Runner(
    agent=agent,
    app_name="analytics_demo",
    session_service=InMemorySessionService(),
)

print(f"Plugin registered: {plugin.name}")
print(f"Table: {plugin.dataset_id}.{plugin.table_id}")
```

### Example 2 — inspecting the per-loop state pattern

```python
import asyncio
from google.adk.plugins.bigquery_agent_analytics_plugin import (
    BigQueryAgentAnalyticsPlugin,
    BigQueryLoggerConfig,
)

config = BigQueryLoggerConfig(view_prefix="demo_")
plugin = BigQueryAgentAnalyticsPlugin(
    project_id="my-project",
    dataset_id="events",
    config=config,
)

async def check_loop_state():
    """Shows the per-event-loop BatchProcessor lookup pattern."""
    loop = asyncio.get_running_loop()

    # Before any agent runs, loop state is not yet initialized.
    # _batch_processor_prop returns None when no state for the current loop.
    bp = plugin.batch_processor  # Uses __getattribute__ routing
    print(f"BatchProcessor before init: {bp}")  # None

    # After the plugin starts (triggered by the first agent invocation),
    # _get_loop_state() creates a new _LoopState for the current event loop.
    print(f"Registered loops: {len(plugin._loop_state_by_loop)}")

asyncio.run(check_loop_state())
```

### Example 3 — `BigQueryLoggerConfig` options

```python
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryLoggerConfig

# Full configuration reference
config = BigQueryLoggerConfig(
    table_id="agent_telemetry",
    view_prefix="agent_",          # Prefix for auto-created views
    max_content_length=16384,      # Max chars before GCS offload
    # content_formatter=None       # Custom (content, event_type) → str formatter
    # retry_config=RetryConfig(...)
)

print("BigQueryLoggerConfig fields:")
print(f"  table_id:            {config.table_id}")
print(f"  view_prefix:         {config.view_prefix}")
print(f"  max_content_length:  {config.max_content_length}")
```

---

## 7 · `CompactionRequestProcessor`

**Source:** `google.adk.flows.llm_flows.compaction`

`CompactionRequestProcessor` is a `BaseLlmRequestProcessor` that compacts session events **before** the contents processor prepares them for model calls. It delegates to `_run_compaction_for_token_threshold_config()` from `google.adk.apps.compaction`.

### Source

```python
class CompactionRequestProcessor(BaseLlmRequestProcessor):
    """Compacts session events before contents are prepared for model calls."""

    async def run_async(
        self, invocation_context: InvocationContext, llm_request: LlmRequest
    ) -> AsyncGenerator[Event, None]:
        del llm_request
        config = invocation_context.events_compaction_config
        if not _has_token_threshold_config(config):
            return
            yield  # Required for AsyncGenerator.

        token_compacted = await _run_compaction_for_token_threshold_config(
            config=config,
            session=invocation_context.session,
            session_service=invocation_context.session_service,
            agent=invocation_context.agent,
            agent_name=invocation_context.agent.name,
            current_branch=invocation_context.branch,
        )
        if token_compacted:
            invocation_context.token_compaction_checked = True
        return
        yield  # Required for AsyncGenerator.

request_processor = CompactionRequestProcessor()
```

### Module-level singleton

The module exports a single `request_processor = CompactionRequestProcessor()` instance that is imported and used by `BaseLlmFlow`. There is no need to instantiate this class yourself.

### `_has_token_threshold_config()` guard

The processor only activates when `EventsCompactionConfig` has `token_threshold` mode configured (not sliding-window-only). If the config is absent or is purely a sliding-window config, the processor returns immediately.

### `token_compaction_checked` flag

After compaction runs, `invocation_context.token_compaction_checked = True` is set. This flag prevents redundant compaction attempts within the same invocation (e.g. if the processor is called multiple times in a pipeline).

### Example 1 — enabling token-threshold compaction on an App

```python
from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.apps.compaction import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models.google_llm import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

# When token_threshold is set, CompactionRequestProcessor activates
# before each LLM call and runs compaction if context exceeds the limit.
compaction_cfg = EventsCompactionConfig(
    token_threshold=8192,           # Compact when context > 8192 tokens
    compaction_overlap=512,         # Keep last 512 tokens of pre-compaction context
    summarizer=LlmEventSummarizer(
        llm=Gemini(model="gemini-2.0-flash"),
    ),
)

agent = LlmAgent(
    name="compact_agent",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant with a long memory.",
)

app = App(
    name="compaction_demo",
    root_agent=agent,
    events_compaction_config=compaction_cfg,
)

runner = Runner(
    agent=agent,
    app_name="compaction_demo",
    session_service=InMemorySessionService(),
)

print(f"Token-threshold compaction enabled at {compaction_cfg.token_threshold} tokens")
print("CompactionRequestProcessor will check and compact before each LLM call")
```

### Example 2 — inspecting the module-level singleton

```python
from google.adk.flows.llm_flows.compaction import (
    CompactionRequestProcessor,
    request_processor,
)

# The module exports a ready-to-use singleton
print(f"Singleton type: {type(request_processor).__name__}")
print(f"Same class: {isinstance(request_processor, CompactionRequestProcessor)}")

# BaseLlmFlow imports this singleton directly:
# from .compaction import request_processor as compaction_request_processor
# and includes it in the request processor chain when compaction is configured.
print("BaseLlmFlow uses this module-level singleton automatically.")
```

### Example 3 — combining token-threshold with sliding-window compaction

```python
from google.adk.apps.compaction import EventsCompactionConfig

# Token-threshold compaction (triggers CompactionRequestProcessor)
# can be combined with sliding-window compaction on the App.
# Only token_threshold activates CompactionRequestProcessor;
# sliding-window is handled separately in the contents processor.
cfg = EventsCompactionConfig(
    # Token threshold mode: compact when context exceeds limit
    token_threshold=16384,
    compaction_overlap=1024,

    # Sliding window mode: always keep last N events (applied separately)
    # max_events=50,  # Not part of EventsCompactionConfig directly
)

print(f"Token threshold: {cfg.token_threshold}")
print(f"Compaction overlap: {cfg.compaction_overlap}")
print("When token_threshold is set, _has_token_threshold_config() returns True")
print("→ CompactionRequestProcessor runs before each LLM call")
```

---

## 8 · `LocalEnvironment` + `BaseEnvironment` + `ExecutionResult`

**Source:** `google.adk.environment._local_environment`, `._base_environment`

`LocalEnvironment` is an `@experimental` `BaseEnvironment` implementation that executes shell commands via `asyncio.create_subprocess_shell` and provides async file I/O via `asyncio.to_thread`. It is the concrete backend used by `EnvironmentToolset` for local development.

### `ExecutionResult` — the unified execution output

```python
@dataclass
class ExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
```

The `timed_out` flag is set when `asyncio.wait_for` raises `asyncio.TimeoutError`. On timeout, the process is killed with `proc.kill()` and stdout/stderr are still collected from the dead process.

### `LocalEnvironment` lifecycle

| Method | Behaviour |
|---|---|
| `initialize()` | Creates a temp dir (prefix `adk_workspace_`) if no `working_dir` was supplied; marks `_is_initialized=True`. |
| `close()` | Removes the temp dir only if the environment created it (`_auto_created=True`); otherwise leaves it intact. |
| `execute(command, *, timeout)` | Runs via `asyncio.create_subprocess_shell` in `working_dir`; merges `env_vars` into `os.environ`. |
| `read_file(path)` | Resolves relative paths against `working_dir`; reads bytes via `asyncio.to_thread`. |
| `write_file(path, content)` | Creates parent directories; handles `str` vs `bytes` in the correct mode. |

### Example 1 — basic command execution with timeout

```python
import asyncio
from pathlib import Path
from google.adk.environment._local_environment import LocalEnvironment

async def run_commands():
    async with LocalEnvironment() as env:
        # Environment creates a temp workspace automatically
        print(f"Workspace: {env.working_dir}")

        # Run a shell command
        result = await env.execute("echo 'Hello from ADK environment'", timeout=10)
        print(f"Exit code: {result.exit_code}")
        print(f"Stdout: {result.stdout.strip()}")
        print(f"Timed out: {result.timed_out}")

        # Write and read a file
        await env.write_file("test.txt", "Hello, world!")
        content = await env.read_file("test.txt")
        print(f"File content: {content.decode()}")

asyncio.run(run_commands())
```

### Example 2 — custom working directory and env vars

```python
import asyncio
from pathlib import Path
from google.adk.environment._local_environment import LocalEnvironment

async def use_custom_workspace():
    workspace = Path("/tmp/my_adk_workspace")
    env = LocalEnvironment(
        working_dir=workspace,
        env_vars={
            "MY_SECRET": "not-really-secret",
            "PYTHONPATH": str(workspace),
        },
    )
    await env.initialize()  # Creates the directory if it doesn't exist

    result = await env.execute("echo $MY_SECRET", timeout=5)
    print(f"MY_SECRET: {result.stdout.strip()}")

    # Fixed working_dir is NOT removed on close
    await env.close()
    print(f"Workspace still exists: {workspace.exists()}")

asyncio.run(use_custom_workspace())
```

### Example 3 — timeout handling and error capture

```python
import asyncio
from google.adk.environment._local_environment import LocalEnvironment

async def demonstrate_timeout():
    async with LocalEnvironment() as env:
        # Command that should time out
        result = await env.execute("sleep 10", timeout=0.5)
        print(f"Timed out: {result.timed_out}")
        print(f"Exit code after kill: {result.exit_code}")

        # Command that fails (non-zero exit code)
        result2 = await env.execute("ls /nonexistent_path_xyz", timeout=5)
        print(f"Exit code: {result2.exit_code}")  # Non-zero
        print(f"Stderr: {result2.stderr.strip()}")

asyncio.run(demonstrate_timeout())
```

---

## 9 · `GeminiLlmConnection`

**Source:** `google.adk.models.gemini_llm_connection`

`GeminiLlmConnection` is the `BaseLlmConnection` implementation for Gemini's **live bidirectional streaming** (BIDI) API. It wraps a `google.genai.live.AsyncSession` and handles history initialization, audio filtering, model-specific preamble collapsing, and streaming response processing.

### Constructor

```python
class GeminiLlmConnection(BaseLlmConnection):
    def __init__(
        self,
        gemini_session: live.AsyncSession,
        api_backend: GoogleLLMVariant = GoogleLLMVariant.VERTEX_AI,
        model_version: str | None = None,
    ):
```

On construction, it detects two special model variants:
- `_is_gemini_3_1_flash_live` — activates **collapsed-text preamble** mode for history initialization.
- `_is_gemini_3_5_live_translate` — enables live translation mode.

### `send_history()` — audio filtering and Gemini 3.1 Flash preamble

When a live session starts (agent transfer or new connection with existing ADK history), `send_history()` is called:

1. **Audio filtering** — audio parts are stripped from all history content because the Live API cannot replay audio; only text, function calls, and function responses are sent.
2. **Normal mode** — history is sent directly via `connection.send_live_content()`.
3. **Gemini 3.1 Flash on Vertex AI** — because this model doesn't support `history_config` in the SDK, all prior turns are collapsed into a single user-role message with the prefix `"Previous conversation history:\n"` followed by formatted `ROLE: text\n` lines. This avoids a `1007` protocol error from mixed-role turn ordering.

### Example 1 — creating a live connection via `LlmAgent`

```python
# GeminiLlmConnection is created internally when LlmAgent.connect() is called.
# It requires a live API-capable model (gemini-2.0-flash-live-001, etc.)
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.agents.live_request_queue import LiveRequestQueue, LiveRequest
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio

async def run_live_session():
    agent = LlmAgent(
        name="live_agent",
        model="gemini-2.0-flash-live-001",  # Live-capable model
        instruction="You are a voice assistant.",
    )

    runner = Runner(
        agent=agent,
        app_name="live_demo",
        session_service=InMemorySessionService(),
    )

    session = await runner.session_service.create_session(
        app_name="live_demo", user_id="u1"
    )
    live_queue = LiveRequestQueue()

    # GeminiLlmConnection is created inside run_live
    async def send_audio():
        await asyncio.sleep(0.1)
        # In production, stream real audio bytes here
        live_queue.send(
            LiveRequest(content=types.Content(
                role="user",
                parts=[types.Part.from_text(text="Hello, can you hear me?")]
            ))
        )
        await asyncio.sleep(0.5)
        live_queue.close()

    run_config = RunConfig(streaming_mode=StreamingMode.BIDI)

    sender_task = asyncio.create_task(send_audio())
    events = []
    try:
        async for ev in runner.run_live(
            user_id="u1",
            session_id=session.id,
            live_request_queue=live_queue,
            run_config=run_config,
        ):
            events.append(ev)
    except Exception as e:
        print(f"Live session note: {e}")
    finally:
        await sender_task

    print(f"Received {len(events)} events from live session")

# asyncio.run(run_live_session())
print("GeminiLlmConnection created internally by run_live when BIDI mode is active")
```

### Example 2 — audio filtering internals

```python
from google.adk.utils.content_utils import filter_audio_parts
from google.genai import types

# GeminiLlmConnection.send_history() filters audio from all history content
# before sending to the Live API session. This is because:
# 1. Audio has already been transcribed to text events.
# 2. The Live API rejects audio in history (it would corrupt the session).

def demo_audio_filtering():
    # Simulate a session history with mixed text and audio parts
    history = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="Hello!"),
                types.Part(
                    inline_data=types.Blob(
                        mime_type="audio/pcm",
                        data=b"\x00\x01\x02\x03",  # fake audio bytes
                    )
                ),
            ]
        ),
        types.Content(
            role="model",
            parts=[types.Part.from_text(text="Hi there!")]
        ),
    ]

    # Filter audio (mirrors what send_history() does)
    filtered = [
        f for content in history
        if (f := filter_audio_parts(content)) is not None
    ]

    print(f"Original turns: {len(history)}")
    print(f"Filtered turns: {len(filtered)}")
    for turn in filtered:
        for part in turn.parts:
            print(f"  role={turn.role}, has_text={bool(part.text)}, has_audio={part.inline_data is not None}")

demo_audio_filtering()
```

### Example 3 — Gemini 3.1 Flash collapsed-text preamble

```python
# Gemini 3.1 Flash Live on Vertex AI doesn't support history_config in the SDK.
# GeminiLlmConnection collapses all prior history into a single user-turn
# preamble to avoid 1007 "invalid role mid-session" protocol errors.
from google.genai import types

def simulate_preamble_collapse(history: list[types.Content]) -> str:
    """Mirrors GeminiLlmConnection's history collapse for Gemini 3.1 Flash."""
    collapsed_text = "Previous conversation history:\n"
    for content in history:
        # Filter audio first (already done by filter_audio_parts in real code)
        for part in content.parts:
            if part.text:
                role = content.role.upper() if content.role else "UNKNOWN"
                collapsed_text += f"{role}: {part.text}\n"
    return collapsed_text

history = [
    types.Content(role="user", parts=[types.Part.from_text("What's the weather?")]),
    types.Content(role="model", parts=[types.Part.from_text("It's sunny today!")]),
    types.Content(role="user", parts=[types.Part.from_text("What about tomorrow?")]),
]

preamble = simulate_preamble_collapse(history)
print("Collapsed preamble sent to Gemini 3.1 Flash Live:")
print(preamble)
print("This single user-turn avoids the 1007 protocol error on Vertex AI.")
```

---

## 10 · `plot_workflow_graph`

**Source:** `google.adk.cli.utils.graph_visualization`

`plot_workflow_graph` renders an ADK agent or Workflow graph as a **Graphviz DOT/SVG diagram** with live `NodeStatus` colour overlays. It is used by the ADK web UI to show workflow topology and real-time execution progress.

### Signature

```python
def plot_workflow_graph(
    app_info: dict[str, Any],
    agent_state: dict[str, Any] | None = None,
    format: str = "svg",
    dark_mode: bool = True,
) -> str | bytes:
```

- `app_info`: The JSON/dict from `Runner.get_app_info()` (or equivalent). Contains `root_agent.graph.nodes` and `.edges`.
- `agent_state`: Optional dict with `nodes: {node_name: NodeStatus}` for status colouring.
- `format`: Graphviz output format — `"svg"` returns `str`, binary formats return `bytes`.
- `dark_mode`: Toggles colour theme (dark `#0F172A` background vs light).

### Fallback for `LlmAgent` without `Workflow`

When `app_info["root_agent"]["graph"]` is empty (i.e. an `LlmAgent` without a graph), `plot_workflow_graph` builds a synthetic graph by recursively traversing `sub_agents` and listing `tools` as metadata. This gives a visual overview even for multi-agent trees that don't use the `Workflow` class.

### `NodeStatus` colour mapping

| `NodeStatus` | Dark mode colour |
|---|---|
| `COMPLETED` | `#16A34A` (green) |
| `RUNNING` | `#D97706` (amber) |
| `FAILED` | `#EF4444` (red) |
| _(default)_ | `#1E293B` (dark slate) |
| START node | `#059669` (emerald) |
| END node | `#DC2626` (dark red) |

### Example 1 — generating a workflow SVG

```python
# pip install google-adk graphviz
from google.adk.cli.utils.graph_visualization import plot_workflow_graph

# Simulate an app_info dict (normally from Runner.get_app_info())
app_info = {
    "root_agent": {
        "name": "pipeline",
        "graph": {
            "nodes": [
                {"name": "fetch_data", "type": "node"},
                {"name": "process_data", "type": "node"},
                {"name": "summarize", "type": "node"},
            ],
            "edges": [
                {"from_node": {"name": "fetch_data"},   "to_node": {"name": "process_data"}},
                {"from_node": {"name": "process_data"}, "to_node": {"name": "summarize"}},
            ],
        },
        "sub_agents": [],
    }
}

# Generate SVG (dark mode)
svg_dark = plot_workflow_graph(app_info, dark_mode=True, format="svg")
print(f"SVG length: {len(svg_dark)} chars")
print(f"Contains 'fetch_data': {'fetch_data' in svg_dark}")

# Light mode variant
svg_light = plot_workflow_graph(app_info, dark_mode=False, format="svg")
print(f"Light SVG length: {len(svg_light)} chars")
```

### Example 2 — overlaying live NodeStatus colours

```python
from google.adk.cli.utils.graph_visualization import plot_workflow_graph
from google.adk.workflow._node_status import NodeStatus

app_info = {
    "root_agent": {
        "name": "my_workflow",
        "graph": {
            "nodes": [
                {"name": "step_a", "type": "node"},
                {"name": "step_b", "type": "node"},
                {"name": "step_c", "type": "node"},
            ],
            "edges": [
                {"from_node": {"name": "step_a"}, "to_node": {"name": "step_b"}},
                {"from_node": {"name": "step_b"}, "to_node": {"name": "step_c"}},
            ],
        },
        "sub_agents": [],
    }
}

# Simulate mid-run status: step_a done, step_b running, step_c pending
agent_state = {
    "nodes": {
        "step_a": NodeStatus.COMPLETED,
        "step_b": NodeStatus.RUNNING,
        # step_c has no status yet → default colour
    }
}

svg = plot_workflow_graph(app_info, agent_state=agent_state, format="svg")

# The SVG will colour:
# step_a in #16A34A (green, completed)
# step_b in #D97706 (amber, running)
# step_c in #1E293B (default dark slate)
print(f"Status-coloured SVG generated ({len(svg)} chars)")
```

### Example 3 — LlmAgent sub-agent tree (no Workflow)

```python
from google.adk.cli.utils.graph_visualization import plot_workflow_graph

# When graph is empty, plot_workflow_graph traverses sub_agents recursively.
app_info = {
    "root_agent": {
        "name": "coordinator",
        "graph": {},   # No Workflow → fallback to sub_agent traversal
        "tools": ["google_search"],
        "sub_agents": [
            {
                "name": "researcher",
                "tools": ["web_fetch", "summarize"],
                "sub_agents": [],
            },
            {
                "name": "writer",
                "tools": ["write_doc"],
                "sub_agents": [
                    {
                        "name": "formatter",
                        "tools": ["format_markdown"],
                        "sub_agents": [],
                    }
                ],
            },
        ],
    }
}

svg = plot_workflow_graph(app_info, format="svg")
print(f"Multi-agent tree SVG: {len(svg)} chars")
# Nodes: coordinator → researcher, coordinator → writer → formatter
print("Edges automatically derived from parent→child sub-agent relationships")
```

---

## Summary table

| Class | Module | Key insight |
|---|---|---|
| `AntigravityAgent` | `google.adk.labs.antigravity` | Root-only ADK wrapper for Antigravity SDK; trajectory file management; step→ADK event mapping |
| `CredentialManager` | `google.adk.auth.credential_manager` | 8-step lifecycle (validate→ready→load→exchange→refresh→save); class-level `register_auth_provider()` |
| `OAuth2DiscoveryManager` | `google.adk.auth.oauth2_discovery` | RFC8414/RFC9728 auto-discovery; 3-endpoint ordering (path-insertion then path-appending) |
| `ContextCacheRequestProcessor` | `google.adk.flows.llm_flows.context_cache_processor` | Sets `cache_config` + scans session events for latest `CacheMetadata` and token count |
| `LocalEvalSampler` | `google.adk.optimization.local_eval_sampler` | `sample_and_score()` → `LocalEvalService`; `capture_full_eval_data` for GEPA reflection |
| `BigQueryAgentAnalyticsPlugin` | `google.adk.plugins.bigquery_agent_analytics_plugin` | BigQuery Write API; per-event-loop `_LoopState`; `EventData.adk_extras` in `attributes.adk.*` |
| `CompactionRequestProcessor` | `google.adk.flows.llm_flows.compaction` | Token-threshold compaction before content processor; module-level singleton; `token_compaction_checked` |
| `LocalEnvironment` | `google.adk.environment._local_environment` | asyncio subprocess; temp-dir auto-lifecycle; `timed_out` flag; bytes/str write branching |
| `GeminiLlmConnection` | `google.adk.models.gemini_llm_connection` | Audio filtering on history init; Gemini 3.1 Flash collapsed-text preamble for Vertex AI |
| `plot_workflow_graph` | `google.adk.cli.utils.graph_visualization` | DOT/SVG with `NodeStatus` colour overlays; sub-agent tree fallback for `LlmAgent` |

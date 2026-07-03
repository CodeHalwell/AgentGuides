---
title: "Class deep dives — volume 35 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.3.0 classes: ServiceAccountCredentialExchanger (access token vs ID token exchange; use_id_token; use_default_credential; audience-based Cloud Run auth), PubSub client factory (get_publisher_client/get_subscriber_client/cleanup_clients; TTL-30-min cache; BatchSettings(max_messages=1)), ScheduleDynamicNode Protocol (ctx.run_node() contract; 9 parameters: 3 positional + 6 keyword-only; fresh/dedup/resume dispatch semantics), build_node + is_node_like (NodeLike→BaseNode; LlmAgent mode inference; _ParallelWorker wrapping; _ToolNode/FunctionNode promotion), OAuthGrantType + ExtendedOAuth2 + OpenIdConnectWithConfig (extended auth scheme models; from_flow() factory; issuer_url discovery), ToolContextCredentialStore + AuthPreparationResult + ToolAuthHandler.prepare_auth_credentials (SHA-256 credential key derivation; OAuth2 field nulling before hashing; refresh on read), OperationParser (OpenAPI operation→Python params; preserve_property_names; snake_case; _dedupe_param_names; load() classmethod), UrlContextTool standalone (Gemini 2 built-in URL fetcher; process_llm_request hook; model guard; types.UrlContext injection), ApiParameter + rename_python_keywords (OpenAPI→Python parameter transformation; location-based defaults; TypeHintHelper), token_to_scheme_credential + openid_dict_to_scheme_credential + credential_to_param (auth helper factories; API key/bearer/OAuth2/OpenID scheme+credential pairs)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 35"
  order: 104
---

import { Aside } from "@astrojs/starlight/components";

<Aside type="note">
All signatures, constants, and behaviours on this page were verified directly
against the installed package source (locate yours with
`python -c 'import google.adk; print(google.adk.__file__)'`) on
**google-adk == 2.3.0**. No documentation or blog posts were used as primary
sources.
</Aside>

---

## 1 · `ServiceAccountCredentialExchanger` — service account → bearer token

**Source:** `google/adk/tools/openapi_tool/auth/credential_exchangers/service_account_exchanger.py`

`ServiceAccountCredentialExchanger` exchanges a Google Service Account
credential for either an **access token** (default, for calling Cloud APIs with
OAuth2 scopes) or an **ID token** (required by Cloud Run, Cloud Functions, and
any service that verifies caller identity via `Authorization: Bearer <id_token>`).

The choice is driven by `ServiceAccount.use_id_token`.  When `True`, the
exchanger calls `IDTokenCredentials.from_service_account_info()` and populates
the `audience` field; when `False` it calls
`service_account.Credentials.from_service_account_info()` and uses the
supplied `scopes` list.  Setting `use_default_credential = True` in either
branch skips the explicit JSON key and falls back to Application Default
Credentials (`google.auth.default()` / `fetch_id_token()`).

### Key method (verified `service_account_exchanger.py`)

```python
class ServiceAccountCredentialExchanger(BaseAuthCredentialExchanger):
    def exchange_credential(
        self,
        auth_scheme: AuthScheme,
        auth_credential: Optional[AuthCredential] = None,
    ) -> AuthCredential:
        # delegates to _exchange_for_id_token or _exchange_for_access_token
        ...

    def _exchange_for_id_token(self, sa_config: ServiceAccount) -> AuthCredential:
        # uses sa_config.audience; raises AuthCredentialMissingError on failure

    def _exchange_for_access_token(self, sa_config: ServiceAccount) -> AuthCredential:
        # requires sa_config.scopes when use_default_credential is False
        # injects x-goog-user-project header when quota_project_id is available
```

### Example 1 — exchange explicit SA key for an access token

```python
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, ServiceAccount, ServiceAccountCredential
from fastapi.openapi.models import HTTPBearer
from google.adk.tools.openapi_tool.auth.credential_exchangers.service_account_exchanger import ServiceAccountCredentialExchanger

sa_json = {
    "type": "service_account",
    "project_id": "my-project",
    "private_key_id": "key-id",
    "private_key": "-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----\n",
    "client_email": "my-sa@my-project.iam.gserviceaccount.com",
    "client_id": "123456789",
    "token_uri": "https://oauth2.googleapis.com/token",
}

auth_credential = AuthCredential(
    auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
    service_account=ServiceAccount(
        service_account_credential=ServiceAccountCredential.model_construct(**sa_json),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    ),
)

auth_scheme = HTTPBearer(bearerFormat="JWT")
exchanger = ServiceAccountCredentialExchanger()
# Replace the placeholder private_key above with a real key before running.
# exchange_credential() calls the token endpoint and will fail on a dummy key.
try:
    result = exchanger.exchange_credential(auth_scheme, auth_credential)
    print(result.auth_type)            # AuthCredentialTypes.HTTP
    print(result.http.scheme)          # bearer
    print(result.http.credentials.token[:20])
except Exception as exc:
    print(f"Provide a real service-account key to run this example: {exc}")
```

### Example 2 — exchange for an ID token (Cloud Run service-to-service)

```python
from google.adk.auth.auth_credential import (
    AuthCredential, AuthCredentialTypes, ServiceAccount, ServiceAccountCredential
)
from fastapi.openapi.models import HTTPBearer
from google.adk.tools.openapi_tool.auth.credential_exchangers.service_account_exchanger import ServiceAccountCredentialExchanger

sa_json = {
    "type": "service_account",
    "project_id": "my-project",
    "private_key_id": "key-id",
    "private_key": "-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----\n",
    "client_email": "my-sa@my-project.iam.gserviceaccount.com",
    "client_id": "123456789",
    "token_uri": "https://oauth2.googleapis.com/token",
}

auth_credential = AuthCredential(
    auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
    service_account=ServiceAccount(
        service_account_credential=ServiceAccountCredential.model_construct(**sa_json),
        use_id_token=True,                        # ← ID token branch
        audience="https://my-service.run.app",   # required for ID token
    ),
)

exchanger = ServiceAccountCredentialExchanger()
# Replace the placeholder private_key above with a real key before running.
try:
    result = exchanger.exchange_credential(HTTPBearer(), auth_credential)
    # result.http.credentials.token is a signed OIDC ID token valid for
    # the specified audience — suitable for Cloud Run Authorization headers
    print(result.http.credentials.token[:20])
except Exception as exc:
    print(f"Provide a real service-account key to run this example: {exc}")
```

### Example 3 — fall back to Application Default Credentials with custom scopes

```python
from google.adk.auth.auth_credential import (
    AuthCredential, AuthCredentialTypes, ServiceAccount
)
from fastapi.openapi.models import HTTPBearer
from google.adk.tools.openapi_tool.auth.credential_exchangers.service_account_exchanger import ServiceAccountCredentialExchanger

# No JSON key needed — reads credentials from the environment
# (GOOGLE_APPLICATION_CREDENTIALS, gcloud auth application-default login, etc.)
auth_credential = AuthCredential(
    auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
    service_account=ServiceAccount(
        use_default_credential=True,
        scopes=["https://www.googleapis.com/auth/bigquery.readonly"],
    ),
)

exchanger = ServiceAccountCredentialExchanger()
result = exchanger.exchange_credential(HTTPBearer(), auth_credential)
# result.http.credentials.token is a short-lived access token
# result.http.additional_headers may contain x-goog-user-project
```

---

## 2 · PubSub client factory — `get_publisher_client` + `get_subscriber_client` + `cleanup_clients`

**Source:** `google/adk/tools/pubsub/client.py`

ADK's PubSub toolset wraps `google-cloud-pubsub` clients behind a module-level
TTL cache so that repeated calls within the same process share a single
`PublisherClient` / `SubscriberClient` instead of opening new gRPC channels on
every tool invocation.

Key implementation details (verified `client.py`):

| Detail | Publisher | Subscriber |
|--------|-----------|------------|
| Cache TTL | 1800 s (30 min) | 1800 s (30 min) |
| Cache key | `(id(credentials), user_agents_key, id(publisher_options))` | `(id(credentials), user_agents_key)` |
| Batching | `BatchSettings(max_messages=1)` — effectively disables batching for synchronous publish | N/A |
| Thread safety | `threading.Lock` per cache dict | `threading.Lock` per cache dict |
| User-agent | Prefixes `"adk-pubsub-tool google-adk/<version>"` | Same prefix |

`cleanup_clients()` closes all cached transports and clears both caches — call
it in shutdown handlers to avoid dangling gRPC connections.

### Example 1 — obtain a cached publisher and publish a message

```python
from unittest.mock import MagicMock
from google.adk.tools.pubsub.client import get_publisher_client

# Fake credentials object (identity used as cache key)
creds = MagicMock()
creds.quota_project_id = None

client_a = get_publisher_client(credentials=creds)
client_b = get_publisher_client(credentials=creds)

# Same object identity proves the cache was hit
assert client_a is client_b, "Cache miss — unexpected new client created"
print(f"Cache hit: both calls returned the same {type(client_a).__name__} instance")
# To actually publish, supply real GCP credentials and call:
#   client_a.publish("projects/my-project/topics/my-topic", b"hello")
```

### Example 2 — separate caches for different credential objects

```python
from unittest.mock import MagicMock
from google.adk.tools.pubsub.client import get_publisher_client

creds_prod = MagicMock()
creds_staging = MagicMock()   # different Python object → different cache key

client_prod = get_publisher_client(credentials=creds_prod)
client_stag = get_publisher_client(credentials=creds_staging)

# Different credentials → different cache slots → different client objects
assert client_prod is not client_stag
print("Two independent publisher clients obtained")
```

### Example 3 — custom user-agent string and TTL cache expiry check

```python
from unittest.mock import MagicMock
from google.adk.tools.pubsub.client import (
    get_publisher_client, get_subscriber_client, cleanup_clients, _CACHE_TTL
)

creds = MagicMock()
ua = "my-app/1.0"

pub = get_publisher_client(credentials=creds, user_agent=ua)
sub = get_subscriber_client(credentials=creds, user_agent=ua)

# The ClientInfo passed to pubsub_v1 merges the ADK prefix and the custom UA:
# "adk-pubsub-tool google-adk/2.3.0 my-app/1.0"
print(f"Publisher client: {type(pub).__name__}")
print(f"Subscriber client: {type(sub).__name__}")
print(f"Cache TTL is {_CACHE_TTL} seconds ({_CACHE_TTL // 60} minutes)")

# Graceful shutdown — closes gRPC transports and clears both caches
cleanup_clients()
print("All cached clients closed and caches cleared")
```

---

## 3 · `ScheduleDynamicNode` Protocol — `ctx.run_node()` dispatch contract

**Source:** `google/adk/workflow/_schedule_dynamic_node.py`

`ScheduleDynamicNode` is a `typing.Protocol` that defines the internal contract
used by the workflow engine to dispatch `ctx.run_node()` calls.  Any object
that implements this Protocol can act as the scheduler — production code uses
`DynamicNodeScheduler`, but the Protocol decouples the calling node from the
concrete implementation (useful for testing).

### Full Protocol signature (verified `_schedule_dynamic_node.py`)

```python
class ScheduleDynamicNode(Protocol):
    def __call__(
        self,
        ctx: Context,
        node: Any,                          # usually a BaseNode subclass
        node_input: Any,                    # must match node.input_schema if defined
        *,
        node_name: str | None = None,       # deterministic tracking name (CRITICAL for resume)
        use_as_output: bool = False,        # replace calling node's output with child's
        run_id: str,                        # unique ID for this specific execution
        use_sub_branch: bool = False,       # run in isolated sub-branch
        override_branch: str | None = None, # explicit branch override
        override_isolation_scope: str | None = None,
    ) -> Awaitable[Context]:  # internal; ctx.run_node() unwraps this and returns child output (Any)
        ...
```

Three execution paths the scheduler resolves:
- **Fresh run** — node not seen before; executes natively.
- **Dedup (same-turn completed)** — node already completed this turn; returns cached output immediately.
- **Resume (cross-turn)** — node state found in session events; rehydrates interrupts or fast-forwards.

`node_name` is an internal scheduler parameter; the public `ctx.run_node()` API
uses `node.name` as the tracking key automatically.  The scheduler's `node_name`
must be deterministic across turns — avoid generated names like
`f"step_{uuid.uuid4()}"` or resume will never match.

**Public API vs. Protocol differences** — `ctx.run_node()` wraps the scheduler
and has a slightly different surface:
- Return type: the Protocol returns `Awaitable[Context]` (the child's full
  execution context); `ctx.run_node()` unwraps it and returns the child's output
  as `Any`.
- `run_id`: the Protocol declares it as a required `str`; `ctx.run_node()`
  exposes it as `run_id: str | None = None` (optional — pass a stable ID to
  correlate logs and events across retries).

### Example 1 — basic `ctx.run_node()` call pattern

```python
from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow

# ctx.run_node() returns the child's output directly (Any).
# Note: the internal ScheduleDynamicNode Protocol returns Awaitable[Context];
# ctx.run_node() unwraps that Context and surfaces only the output.
async def orchestrate(ctx):
    summariser = LlmAgent(
        name="summariser",
        model="gemini-2.5-flash",
        instruction="Summarise the text provided as input.",
    )
    # run_id is str | None = None in ctx.run_node(); pass a stable string to
    # correlate this execution in logs and events across retries
    output = await ctx.run_node(
        summariser,
        node_input={"text": ctx.input},
        run_id="run-001",
    )
    ctx.output = output   # run_node() returns the child's output directly

wf = Workflow(name="demo").add_node(orchestrate, name="orchestrate")
```

### Example 2 — `use_as_output=True` to propagate child output transparently

```python
from google.adk.agents import LlmAgent

async def transparent_wrapper(ctx):
    inner = LlmAgent(
        name="inner_agent",
        model="gemini-2.5-flash",
        instruction="Answer the question.",
    )
    # use_as_output=True: child output replaces this node's output event
    await ctx.run_node(
        inner,
        node_input=ctx.input,
        run_id="run-wrap-001",
        use_as_output=True,   # ← transparent pass-through
    )
    # ctx.output is automatically set to the child's output when use_as_output=True
```

### Example 3 — `use_sub_branch=True` for isolated message history

```python
import asyncio
from google.adk.agents import LlmAgent

async def parallel_evaluator(ctx):
    tasks = []
    for i, chunk in enumerate(ctx.input.get("chunks", [])):
        async def run_chunk(i=i, chunk=chunk):
            # run_node() returns the child output directly
            return await ctx.run_node(
                LlmAgent(name="evaluator", model="gemini-2.5-flash",
                         instruction="Evaluate quality."),
                node_input={"text": chunk},
                run_id=f"eval-{i}-001",
                use_sub_branch=True,      # each branch gets its own message history
            )
        tasks.append(run_chunk())
    results = await asyncio.gather(*tasks)
    ctx.output = {"scores": results}   # results is a list of child outputs
```

---

## 4 · `build_node` + `is_node_like` — NodeLike → BaseNode conversion

**Source:** `google/adk/workflow/utils/_workflow_graph_utils.py`

`build_node` is the central factory that converts any `NodeLike` value into a
`BaseNode` ready to be placed in a `Workflow` graph.  It is called internally
by `Workflow.add_node()`, `Workflow.add_edge()`, and related helpers.

`NodeLike` is a type alias for:
```python
NodeLike = Union[BaseNode, BaseTool, Callable, Literal["START"]]
```

### Conversion rules (verified `_workflow_graph_utils.py`)

| Input type | Output | Notes |
|-----------|--------|-------|
| `"START"` | `START` sentinel | No wrapping |
| `LlmAgent` (a `BaseNode`) | cloned `LlmAgent` | Sets `rerun_on_resume=True`; resolves `mode` |
| Other `BaseNode` | `model_copy(update=kwargs)` | Only if kwargs provided |
| `BaseTool` | `_ToolNode(tool=...)` | |
| `Callable` | `FunctionNode(func=...)` | `rerun_on_resume` defaults to `False` |

**LlmAgent mode resolution** (from source, lines ~75-90):
- If `agent.parent_agent is not None` → mode defaults to `'chat'` (enables agent transfer).
- Otherwise → mode defaults to `'single_turn'` (standalone graph node).
- Agents in `'task'` or `'chat'` mode automatically get `wait_for_output=True`.
- If `agent.parallel_worker=True`, the agent is wrapped in `_ParallelWorker` instead of returned directly.

### Example 1 — callables are wrapped in FunctionNode

```python
from google.adk.workflow.utils._workflow_graph_utils import build_node, is_node_like
from google.adk.workflow._function_node import FunctionNode

async def my_func(ctx):
    ctx.output = "done"

assert is_node_like(my_func)   # True — callable qualifies as NodeLike
node = build_node(my_func, name="step1", rerun_on_resume=True)
assert isinstance(node, FunctionNode)
assert node.name == "step1"
assert node.rerun_on_resume is True
print(f"Wrapped as: {type(node).__name__}(name={node.name!r})")
```

### Example 2 — BaseTool wraps in _ToolNode; non-NodeLike raises ValueError

```python
from google.adk.workflow.utils._workflow_graph_utils import build_node, is_node_like
from google.adk.tools.function_tool import FunctionTool

def my_tool_func(x: int) -> str:
    return str(x)

tool = FunctionTool(func=my_tool_func)

assert is_node_like(tool)   # True
assert is_node_like("START")   # True
assert not is_node_like(42)    # False — plain int is not NodeLike
assert not is_node_like("some_string")  # False — only "START" literal

from google.adk.workflow._tool_node import _ToolNode
tool_node = build_node(tool, name="my_tool_step")
assert isinstance(tool_node, _ToolNode)
print(f"Tool wrapped as: {type(tool_node).__name__}(name={tool_node.name!r})")
```

### Example 3 — LlmAgent mode inference in Workflow.add_node

```python
from google.adk.agents import LlmAgent
from google.adk.workflow.utils._workflow_graph_utils import build_node

standalone = LlmAgent(name="standalone", model="gemini-2.5-flash",
                       instruction="Answer questions.")

# Simulate adding to a workflow without a parent agent
built = build_node(standalone)
assert built.mode == "single_turn", f"Expected single_turn, got {built.mode}"

# Simulate as a sub-agent (parent set)
child = LlmAgent(name="child", model="gemini-2.5-flash", instruction="Help.")
parent = LlmAgent(name="parent", model="gemini-2.5-flash",
                   instruction="Delegate.", sub_agents=[child])
child.parent_agent = parent   # set by LlmAgent.__init__ when sub_agents provided

built_child = build_node(child)
assert built_child.mode == "chat", f"Expected chat, got {built_child.mode}"
assert built_child.wait_for_output is True

print("LlmAgent mode resolution working as expected")
```

---

## 5 · `OAuthGrantType` + `ExtendedOAuth2` + `OpenIdConnectWithConfig`

**Source:** `google/adk/auth/auth_schemes.py`

These three classes extend the standard FastAPI/OpenAPI security scheme models
to support ADK-specific features: grant-type introspection, OIDC discovery-URL
auto-population, and a rich endpoint-config model for OpenID Connect.

### `OAuthGrantType` — grant type enum with `from_flow()` factory

```python
class OAuthGrantType(str, Enum):
    CLIENT_CREDENTIALS = "client_credentials"
    AUTHORIZATION_CODE = "authorization_code"
    IMPLICIT = "implicit"
    PASSWORD = "password"

    @staticmethod
    def from_flow(flow: OAuthFlows) -> "OAuthGrantType":
        # Inspects flow.clientCredentials, .authorizationCode, .implicit, .password
```

### `ExtendedOAuth2` — OAuth2 with discovery URL

```python
@experimental
class ExtendedOAuth2(OAuth2):
    issuer_url: Optional[str] = None  # enables endpoint auto-discovery
```

### `OpenIdConnectWithConfig` — full OIDC endpoint model

```python
class OpenIdConnectWithConfig(SecurityBase):
    type_: SecuritySchemeType = SecuritySchemeType.openIdConnect
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: Optional[str] = None
    revocation_endpoint: Optional[str] = None
    token_endpoint_auth_methods_supported: Optional[List[str]] = None
    grant_types_supported: Optional[List[str]] = None
    scopes: Optional[List[str]] = None
```

### Example 1 — `OAuthGrantType.from_flow()` — detect grant type from OAuthFlows

```python
from fastapi.openapi.models import OAuthFlows, OAuthFlowClientCredentials, OAuthFlowAuthorizationCode
from google.adk.auth.auth_schemes import OAuthGrantType

cc_flows = OAuthFlows(
    clientCredentials=OAuthFlowClientCredentials(
        tokenUrl="https://auth.example.com/token",
    )
)
assert OAuthGrantType.from_flow(cc_flows) == OAuthGrantType.CLIENT_CREDENTIALS

ac_flows = OAuthFlows(
    authorizationCode=OAuthFlowAuthorizationCode(
        authorizationUrl="https://auth.example.com/authorize",
        tokenUrl="https://auth.example.com/token",
    )
)
assert OAuthGrantType.from_flow(ac_flows) == OAuthGrantType.AUTHORIZATION_CODE
print("Grant type detection works correctly")
```

### Example 2 — `OpenIdConnectWithConfig` for a Google-style OIDC endpoint

```python
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig

# OpenIdConnectWithConfig uses explicit endpoint fields; it does not have an
# openIdConnectUrl constructor param — use ExtendedOAuth2.issuer_url for discovery (see example 3)
google_oidc = OpenIdConnectWithConfig(
    authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
    token_endpoint="https://oauth2.googleapis.com/token",
    userinfo_endpoint="https://openidconnect.googleapis.com/v1/userinfo",
    revocation_endpoint="https://oauth2.googleapis.com/revoke",
    scopes=["openid", "email", "profile"],
)
print(f"Token endpoint: {google_oidc.token_endpoint}")
print(f"Scopes: {google_oidc.scopes}")
```

### Example 3 — `ExtendedOAuth2` with `issuer_url` for auto-discovery

```python
from fastapi.openapi.models import OAuthFlows, OAuthFlowClientCredentials
from google.adk.auth.auth_schemes import ExtendedOAuth2

# issuer_url tells the ADK auth system where to fetch the OIDC discovery doc
# (/.well-known/openid-configuration) to auto-populate tokenUrl, authorizationUrl, etc.
oauth2_with_discovery = ExtendedOAuth2(
    flows=OAuthFlows(
        clientCredentials=OAuthFlowClientCredentials(
            tokenUrl="",   # placeholder — will be discovered from issuer_url
        )
    ),
    issuer_url="https://accounts.google.com",   # discovery root
)
print(f"Issuer URL for discovery: {oauth2_with_discovery.issuer_url}")
print(f"Scheme type: {oauth2_with_discovery.type_}")
```

---

## 6 · `ToolContextCredentialStore` + `AuthPreparationResult` + `ToolAuthHandler.prepare_auth_credentials`

**Source:** `google/adk/tools/openapi_tool/openapi_spec_parser/tool_auth_handler.py`

This trio implements the **credential lifecycle** for `RestApiTool` calls:
storing exchanged tokens, looking them up by a stable key, refreshing stale
OAuth2 tokens, and orchestrating the multi-step exchange pipeline.

### `ToolContextCredentialStore` — SHA-256 keyed credential cache in session state

The store persists exchanged credentials in the agent's session state so they
survive across LLM turns.  The key is a 16-char SHA-256 prefix of the
(auth_scheme, auth_credential) pair serialised to JSON — **with OAuth2 volatile
fields zeroed out before hashing** (auth_uri, state, auth_response_uri,
auth_code, access_token, refresh_token, expires_at, expires_in).  This ensures
the same credential configuration always maps to the same key regardless of
which token is currently active.

### `AuthPreparationResult` — outcome model

```python
class AuthPreparationResult(BaseModel):
    state: Literal["pending", "done"]   # "pending" → user must complete OAuth flow
    auth_scheme: Optional[AuthScheme] = None
    auth_credential: Optional[AuthCredential] = None
```

### `prepare_auth_credentials` — 5-step pipeline (verified)

1. If no auth scheme → return `state="done"` immediately.
2. Look up existing credential from store; if OAuth2 and stale → refresh and re-persist.
3. If no credential or external OAuth2 exchange still pending → call `get_auth_response()`.
4. If still no credential → call `request_credential()` and return `state="pending"`.
5. Exchange credential (SA token, OAuth2 bearer conversion) → return `state="done"`.

### Example 1 — inspect credential key derivation without live credentials

```python
import hashlib
import json
from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowClientCredentials
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth

# Mirrors _stable_model_digest from google.adk.auth.auth_tool (used by get_credential_key)
def stable_model_digest(model) -> str:
    dumped = model.model_dump(by_alias=True, exclude_none=True, mode="json")
    canonical = json.dumps(dumped, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

# Auth scheme contributes the first half of the key
auth_scheme = OAuth2(
    flows=OAuthFlows(
        clientCredentials=OAuthFlowClientCredentials(
            tokenUrl="https://auth.example.com/token",
        )
    )
)

# Build the credential; volatile OAuth2 fields are zeroed before hashing
credential = AuthCredential(
    auth_type=AuthCredentialTypes.OAUTH2,
    oauth2=OAuth2Auth(
        client_id="my-client",
        client_secret="my-secret",
        access_token="short-lived-token",   # zeroed before hashing
        refresh_token="long-lived-token",   # zeroed before hashing
    ),
)

# Zero out volatile fields (mirrors ToolContextCredentialStore.get_credential_key)
cred_copy = credential.model_copy(deep=True)
cred_copy.oauth2.access_token = None
cred_copy.oauth2.refresh_token = None
cred_copy.oauth2.expires_at = None
cred_copy.oauth2.expires_in = None

# Full key format: {scheme_type}_{scheme_digest}_{cred_type}_{cred_digest}_existing_exchanged_credential
scheme_digest = stable_model_digest(auth_scheme)
cred_digest = stable_model_digest(cred_copy)
key = (
    f"{auth_scheme.type_.name}_{scheme_digest}"
    f"_{credential.auth_type.value}_{cred_digest}"
    f"_existing_exchanged_credential"
)
print(f"Session state key: {key}")
# → oauth2_<16-char-hex>_oauth2_<16-char-hex>_existing_exchanged_credential
# The key is stable regardless of which access_token is currently active
```

### Example 2 — `AuthPreparationResult` states in practice

```python
from google.adk.tools.openapi_tool.openapi_spec_parser.tool_auth_handler import (
    AuthPreparationResult
)

# "done" state — auth_credential is Optional[AuthCredential]; it may be
# populated (e.g. a freshly exchanged token) or None if the credential was
# consumed/stored downstream. Check before use.
ready = AuthPreparationResult(state="done")
assert ready.state == "done"
# auth_credential is Optional — do not assert it is always None for "done"
if ready.auth_credential is not None:
    print("Credential available:", ready.auth_credential.auth_type)
else:
    print("Credential already stored/consumed downstream")

# "pending" state — user must complete an OAuth2 browser flow
pending = AuthPreparationResult(
    state="pending",
    auth_scheme=None,   # omitted in this minimal example; real responses may include the scheme for re-rendering
    auth_credential=None,
)
print(f"Pending: waiting for user OAuth2 completion → state={pending.state!r}")
```

### Example 3 — wire up a `ToolAuthHandler` with explicit scheme/credential

```python
from fastapi.openapi.models import HTTPBearer
from google.adk.auth.auth_credential import (
    AuthCredential, AuthCredentialTypes, ServiceAccount
)
from google.adk.tools.openapi_tool.openapi_spec_parser.tool_auth_handler import (
    ToolAuthHandler
)
from google.adk.tools.openapi_tool.auth.credential_exchangers.service_account_exchanger import (
    ServiceAccountCredentialExchanger
)

# ToolAuthHandler.from_tool_context() requires a live ToolContext (ctx),
# so it is called from inside a tool implementation:
async def call_with_sa_auth(ctx):
    auth_scheme = HTTPBearer()
    auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
        service_account=ServiceAccount(use_default_credential=True,
                                       scopes=["https://www.googleapis.com/auth/cloud-platform"]),
    )
    handler = ToolAuthHandler.from_tool_context(
        tool_context=ctx,
        auth_scheme=auth_scheme,
        auth_credential=auth_credential,
        credential_exchanger=ServiceAccountCredentialExchanger(),
    )
    result = await handler.prepare_auth_credentials()
    if result.state == "done":
        # auth_credential is Optional — guard before dereferencing
        if result.auth_credential is not None:
            token = result.auth_credential.http.credentials.token
            print(token[:20])
        else:
            # For service-account exchangers, auth_credential=None on "done"
            # means exchange_credential() raised and the error was logged.
            # Check logs for the root cause (invalid key, quota issues, etc.)
            print("Exchange failed — auth_credential is None; check logs")
```

---

## 7 · `OperationParser` — OpenAPI operation → Python function parameters

**Source:** `google/adk/tools/openapi_tool/openapi_spec_parser/operation_parser.py`

`OperationParser` converts a single OpenAPI `Operation` object into a list of
`ApiParameter` instances that are used to generate Python function signatures,
docstrings, and JSON schema declarations for `RestApiTool`.

### Constructor (verified `operation_parser.py`)

```python
class OperationParser:
    def __init__(
        self,
        operation: Union[Operation, Dict[str, Any], str],
        should_parse: bool = True,          # if False, use load() classmethod instead
        *,
        preserve_property_names: bool = False,  # keep original camelCase names
    ):
        # Accepts Operation object, dict, or JSON string
        # Calls _process_operation_parameters, _process_request_body,
        # _process_return_value, _dedupe_param_names during __init__
```

When `preserve_property_names=False` (default), parameter names are converted
via `_to_snake_case()` and then sanitised with `rename_python_keywords()`.
When `True`, only Python keyword conflicts are resolved (e.g. `if` → `param_if`).

`_dedupe_param_names()` ensures no two parameters share the same `py_name` by
appending a numeric counter to later duplicates (`name_0`, `name_1`, …).

`load()` is a classmethod that constructs an `OperationParser` from a
pre-processed param list — useful when you need to inject synthetic parameters.

### Example 1 — parse a simple GET operation with query parameters

```python
from google.adk.tools.openapi_tool.openapi_spec_parser.operation_parser import OperationParser
from fastapi.openapi.models import Operation, Parameter, Schema

op = Operation(
    operationId="listPets",
    parameters=[
        Parameter(**{
            "name": "maxResults",    # camelCase — will be snake_cased
            "in": "query",
            "required": False,
            "schema": Schema(type="integer", description="Maximum number of results"),
        }),
        Parameter(**{
            "name": "pageToken",
            "in": "query",
            "required": False,
            "schema": Schema(type="string"),
        }),
    ],
    responses={},
)

parser = OperationParser(op)
for param in parser._params:
    print(f"original={param.original_name!r}  py_name={param.py_name!r}  "
          f"location={param.param_location}  required={param.required}")
# output: original='maxResults'  py_name='max_results'  location=query  required=False
# output: original='pageToken'   py_name='page_token'   location=query  required=False
```

### Example 2 — `preserve_property_names=True` keeps original names

```python
from google.adk.tools.openapi_tool.openapi_spec_parser.operation_parser import OperationParser
from fastapi.openapi.models import Operation, Parameter, Schema

op = Operation(
    parameters=[
        Parameter(**{"name": "xApiKey", "in": "header", "required": True,
                     "schema": Schema(type="string")}),
        Parameter(**{"name": "userId", "in": "path", "required": True,
                     "schema": Schema(type="string")}),
    ],
    responses={},
)

parser_snake = OperationParser(op, preserve_property_names=False)
parser_orig = OperationParser(op, preserve_property_names=True)

for p in parser_snake._params:
    print(f"snake:  {p.py_name}")
for p in parser_orig._params:
    print(f"original: {p.original_name}")
# snake:  x_api_key, user_id
# original: xApiKey, userId  (only keyword conflicts renamed)
```

### Example 3 — `load()` classmethod to inject synthetic parameters

```python
from google.adk.tools.openapi_tool.openapi_spec_parser.operation_parser import OperationParser
from google.adk.tools.openapi_tool.common.common import ApiParameter
from fastapi.openapi.models import Operation, Schema

op = Operation(responses={})

# Pre-built params (e.g. from a cache or custom generator)
custom_params = [
    ApiParameter(
        original_name="correlation_id",
        param_location="header",
        param_schema=Schema(type="string"),
        description="Distributed tracing correlation ID",
        required=False,
    )
]

parser = OperationParser.load(op, params=custom_params)
for p in parser._params:
    print(f"Injected: {p.py_name} ({p.param_location})")
# Injected: correlation_id (header)
```

---

## 8 · `UrlContextTool` — Gemini 2 built-in URL context fetcher

**Source:** `google/adk/tools/url_context_tool.py`

`UrlContextTool` is a **model-side built-in** that tells a Gemini 2 model to
fetch content from URLs referenced in the conversation before generating a
response.  Unlike `FunctionTool`, it never produces a Python function call —
instead its `process_llm_request` hook injects a `types.UrlContext()` object
into the `GenerateContentConfig.tools` list before the request reaches the LLM.
The model handles the fetch itself.

### Key implementation (verified `url_context_tool.py`)

```python
class UrlContextTool(BaseTool):
    def __init__(self):
        super().__init__(name='url_context', description='url_context')

    async def process_llm_request(
        self,
        *,
        tool_context: ToolContext,
        llm_request: LlmRequest,
    ) -> None:
        if is_gemini_1_model(llm_request.model):
            raise ValueError('Url context tool cannot be used in Gemini 1.x.')
        elif is_gemini_eap_or_2_or_above(llm_request.model):
            llm_request.config.tools.append(
                types.Tool(url_context=types.UrlContext())
            )

url_context = UrlContextTool()   # pre-built singleton
```

### Example 1 — add `url_context` to an LlmAgent

```python
from google.adk.agents import LlmAgent
from google.adk.tools.url_context_tool import url_context

agent = LlmAgent(
    name="web_researcher",
    model="gemini-2.5-flash",    # must be Gemini 2+
    instruction=(
        "When the user asks about a topic, fetch the provided URL and "
        "summarise its content."
    ),
    tools=[url_context],          # ← adds types.UrlContext() to every LLM request
)
# Usage:
# user: "Summarise https://example.com/paper.pdf"
# → Gemini fetches the URL, reads it, and summarises in its response
print(f"Tool name: {url_context.name}")
print("UrlContextTool added — Gemini will auto-fetch URLs in the conversation")
```

### Example 2 — pair with `google_search` for grounded, URL-enriched responses

```python
from google.adk.agents import LlmAgent
from google.adk.tools.url_context_tool import url_context
from google.adk.tools import google_search

agent = LlmAgent(
    name="deep_researcher",
    model="gemini-2.5-pro",
    instruction="Use web search to find relevant pages, then fetch and read them.",
    tools=[google_search, url_context],
)
# google_search returns URLs → url_context fetches their full content
# Both tools inject into llm_request.config.tools as built-in model capabilities
```

### Example 3 — Gemini 1.x guard raises an informative ValueError

```python
import asyncio
from unittest.mock import MagicMock
from google.adk.tools.url_context_tool import UrlContextTool
from google.adk.models.llm_request import LlmRequest
from google.genai import types

tool = UrlContextTool()

# Simulate a Gemini 1.x model request
llm_request = MagicMock(spec=LlmRequest)
llm_request.model = "gemini-1.5-flash"
llm_request.config = types.GenerateContentConfig(tools=[])

tool_context = MagicMock()

async def test_guard():
    try:
        await tool.process_llm_request(
            tool_context=tool_context,
            llm_request=llm_request,
        )
    except ValueError as e:
        print(f"Correctly raised: {e}")
        # "Url context tool cannot be used in Gemini 1.x."

asyncio.run(test_guard())
```

---

## 9 · `ApiParameter` + `rename_python_keywords` — OpenAPI → Python parameter transformation

**Source:** `google/adk/tools/openapi_tool/common/common.py`

`ApiParameter` is the intermediate representation of a single OpenAPI parameter
after it has been extracted from an `Operation`.  It carries the original API
name (`original_name`), the Python-safe name (`py_name`), the parameter
location (`query`, `path`, `header`, `cookie`, `body`), and the JSON Schema
(`param_schema`).

On construction (`model_post_init`), `ApiParameter` automatically:
1. Calls `_to_snake_case(original_name)` unless `py_name` is already set.
2. Passes the result through `rename_python_keywords()` to avoid collisions
   with Python keywords (`if`, `for`, `return`, etc.).
3. Falls back to a `param_location`-derived default name if both steps produce
   an empty string.
4. Calls `TypeHintHelper.get_type_value()` + `TypeHintHelper.get_type_hint()`
   to compute the Python type annotation.

### `rename_python_keywords` signature

```python
def rename_python_keywords(s: str, prefix: str = 'param_') -> str:
    """Returns prefix + s if s is a Python keyword, else s unchanged."""
    if keyword.iskeyword(s):
        return prefix + s
    return s
```

### Example 1 — `ApiParameter` auto-derives `py_name` and type hint

```python
from google.adk.tools.openapi_tool.common.common import ApiParameter
from fastapi.openapi.models import Schema

# camelCase → snake_case conversion happens automatically in model_post_init
param = ApiParameter(
    original_name="maxPageSize",
    param_location="query",
    param_schema=Schema(type="integer", description="Max items per page"),
    required=True,
)
print(f"original_name: {param.original_name}")  # maxPageSize
print(f"py_name:       {param.py_name}")         # max_page_size
print(f"type_hint:     {param.type_hint}")        # int
print(f"required:      {param.required}")          # True
print(f"str repr:      {str(param)}")              # max_page_size: int
```

### Example 2 — Python keyword collision is transparently renamed

```python
from google.adk.tools.openapi_tool.common.common import ApiParameter, rename_python_keywords
from fastapi.openapi.models import Schema
import keyword

# Verify rename_python_keywords directly
assert rename_python_keywords("if") == "param_if"
assert rename_python_keywords("return") == "param_return"
assert rename_python_keywords("user_id") == "user_id"   # unchanged
print("keyword.iskeyword check:", keyword.iskeyword("if"))   # True

# ApiParameter handles it transparently
param = ApiParameter(
    original_name="return",   # ← Python keyword
    param_location="header",
    param_schema=Schema(type="string"),
)
print(f"py_name for 'return': {param.py_name}")   # param_return
```

### Example 3 — location-based default when name is unrepresentable

```python
from google.adk.tools.openapi_tool.common.common import ApiParameter
from fastapi.openapi.models import Schema

# An empty or non-ASCII-only name falls back to location default
param = ApiParameter(
    original_name="",          # empty → fallback needed
    param_location="body",
    param_schema=Schema(type="object"),
    py_name="",                # also empty → triggers _default_py_name()
)
print(f"Default py_name for body param: {param.py_name}")   # body

# Other location defaults:
for loc, expected in [("query", "query_param"), ("path", "path_param"),
                       ("header", "header_param"), ("cookie", "cookie_param")]:
    p = ApiParameter(original_name="", param_location=loc,
                     param_schema=Schema(type="string"), py_name="")
    print(f"{loc:8s} → {p.py_name}")
```

---

## 10 · `token_to_scheme_credential` + `openid_dict_to_scheme_credential` + `credential_to_param`

**Source:** `google/adk/tools/openapi_tool/auth/auth_helpers.py`

This module provides **factory functions** that create matched `(AuthScheme,
AuthCredential)` pairs and convert them back into HTTP parameters for injection
into outgoing API requests.

| Function | Input | Output |
|----------|-------|--------|
| `token_to_scheme_credential` | `"apikey"` or `"oauth2Token"` + location + value | `(APIKey\|HTTPBearer, AuthCredential)` |
| `openid_dict_to_scheme_credential` | discovery config dict + scopes + credential dict | `(OpenIdConnectWithConfig, AuthCredential)` |
| `openid_url_to_scheme_credential` | OIDC discovery URL + scopes + credential dict | same — auto-fetches config |
| `service_account_dict_to_scheme_credential` | SA JSON dict + scopes | `(HTTPBearer, AuthCredential)` |
| `credential_to_param` | `(AuthScheme, AuthCredential)` | `(ApiParameter, {header_value_dict})` |
| `dict_to_auth_scheme` | raw dict with `"type"` key | appropriate `AuthScheme` subclass |

### Example 1 — `token_to_scheme_credential` for API key in a header

```python
from google.adk.tools.openapi_tool.auth.auth_helpers import token_to_scheme_credential
from fastapi.openapi.models import APIKey

scheme, credential = token_to_scheme_credential(
    "apikey",
    location="header",
    name="X-API-Key",
    credential_value="sk-abc123",
)

assert isinstance(scheme, APIKey)
assert scheme.name == "X-API-Key"
assert credential.api_key == "sk-abc123"
print(f"scheme.in_: {scheme.in_}")   # APIKeyIn.header
print(f"credential.api_key: {credential.api_key}")
```

### Example 2 — `token_to_scheme_credential` for a bearer OAuth2 token

```python
from google.adk.tools.openapi_tool.auth.auth_helpers import token_to_scheme_credential
from fastapi.openapi.models import HTTPBearer
from google.adk.auth.auth_credential import AuthCredentialTypes

scheme, credential = token_to_scheme_credential(
    "oauth2Token",
    credential_value="ya29.my-short-lived-token",
)

assert isinstance(scheme, HTTPBearer)
assert credential.auth_type == AuthCredentialTypes.HTTP
print(f"Token: {credential.http.credentials.token[:20]}...")
print(f"HTTP scheme: {credential.http.scheme}")   # bearer
```

### Example 3 — `credential_to_param` converts a credential back into HTTP headers

```python
from google.adk.tools.openapi_tool.auth.auth_helpers import (
    token_to_scheme_credential, credential_to_param
)

# Step 1: create scheme + credential
scheme, credential = token_to_scheme_credential(
    "oauth2Token",
    credential_value="ya29.my-token",
)

# Step 2: convert to an ApiParameter + kwargs dict for HTTP injection
param, kwargs = credential_to_param(scheme, credential)

print(f"param.original_name: {param.original_name}")   # Authorization
print(f"param.param_location: {param.param_location}") # header
# kwargs is injected as request headers by RestApiTool:
for k, v in kwargs.items():
    print(f"  header key: {k!r}")                        # _auth_prefix_vaf_Authorization
    print(f"  header val: {v[:30]}...")                  # Bearer ya29.my-token
```

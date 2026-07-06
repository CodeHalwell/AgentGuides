---
title: "Class deep dives — volume 36 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.3.0 classes: BaseAgent (run_async/run_live lifecycle; clone() deep-copy semantics; find_agent()/root_agent traversal; BaseAgentState; before/after callback chains), AuthCredential + AuthCredentialTypes + HttpAuth + ServiceAccount (all five credential shapes; model validators; resource_ref; PKCE code_verifier; use_id_token/audience), RunConfig + StreamingMode + ToolThreadPoolConfig (NONE/SSE/BIDI streaming; partial vs aggregated event patterns; thread-pool live-tool isolation; max_llm_calls; TelemetryConfig), LlmRequest (append_instructions text+binary; append_tools function declaration merging; set_output_schema; previous_interaction_id chaining), AuthHandler (generate_auth_uri PKCE; generate_auth_request; parse_and_store_auth_response; exchange_auth_token), EventsCompactionConfig + ResumabilityConfig (compaction_interval/overlap_size; token_threshold+event_retention_size pair; is_resumable), LlmEventSummarizer (maybe_summarize_events; custom prompt_template; _format_events_for_prompt thought/tool rendering; _truncate cap), ToolConfirmation (hint/confirmed/payload; @experimental TOOL_CONFIRMATION; camelCase alias; three confirmation patterns), LiveRequest + LiveRequestQueue + ActiveStreamingTool (priority-ordered fields; send_content/send_realtime/activity signals; streaming tool task+stream lifecycle), model name utilities (extract_model_name path patterns; is_gemini_model; is_gemini_eap_or_2_or_above; EAP regex; ADK_DISABLE_GEMINI_MODEL_ID_CHECK)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 36"
  order: 105
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

## 1 · `BaseAgent` — agent hierarchy foundation

**Source:** `google/adk/agents/base_agent.py`

`BaseAgent` is the abstract Pydantic base class that every concrete agent
(`LlmAgent`, `SequentialAgent`, `LoopAgent`, custom agents) inherits from.
It wires together the lifecycle methods (`run_async`, `run_live`), the
before/after callback chain, sub-agent tree management, and an optional
per-agent typed state (`BaseAgentState`).

### Constructor fields (verified `base_agent.py`)

```python
class BaseAgent(BaseNode, abc.ABC):
    name: str                                        # must be a valid Python identifier, not "user"
    description: str = ''                            # one-liner used by parent agent routing
    parent_agent: Optional[BaseAgent] = Field(default=None, init=False, exclude=True)
    sub_agents: list[BaseAgent] = Field(default_factory=list)
    before_agent_callback: Optional[BeforeAgentCallback] = None
    after_agent_callback: Optional[AfterAgentCallback] = None
```

`BaseAgent.model_post_init` sets `parent_agent` on every sub-agent; a
sub-agent with an existing parent raises `ValueError` immediately.

### Example 1 — custom agent with `BaseAgentState` and per-turn state persistence

```python
import asyncio
from typing import AsyncGenerator, Optional
from google.adk.agents.base_agent import BaseAgent, BaseAgentState
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.genai import types
from google.adk.features import experimental


@experimental
class CounterState(BaseAgentState):
    """Typed per-agent state tracked across turns."""
    calls: int = 0


class CountingAgent(BaseAgent):
    """Echoes user text and tracks how many times it has been called."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # Load persisted state or create fresh
        state = self._load_agent_state(ctx, CounterState) or CounterState()
        state.calls += 1

        # Persist state delta back to session
        ctx.agent_states[self.name] = state.model_dump()

        last_text = ""
        for event in reversed(ctx.session.events):
            if event.author == "user" and event.content:
                for part in event.content.parts:
                    if part.text:
                        last_text = part.text
                        break
            if last_text:
                break

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(
                role="model",
                parts=[types.Part(text=f"[call #{state.calls}] {last_text}")],
            ),
        )


agent = CountingAgent(name="counter_agent", description="counts invocations")
print(agent.name)        # counter_agent
print(agent.root_agent)  # same object — it has no parent
```

### Example 2 — `clone()` for parallel agent variants

```python
from google.adk.agents import LlmAgent

base = LlmAgent(
    name="analyst",
    model="gemini-2.5-flash",
    instruction="You are a data analyst.",
)

# Clone with a modified instruction — original is untouched
strict_variant = base.clone(update={"name": "strict_analyst"})
print(strict_variant.name)       # strict_analyst
print(base.name)                  # analyst (unchanged)

# Sub-agents are deep-copied too so adding one does not pollute the original
from google.adk.agents import SequentialAgent
pipeline = SequentialAgent(
    name="pipeline",
    sub_agents=[base, strict_variant],
)
print(pipeline.find_agent("strict_analyst").name)  # strict_analyst
print(pipeline.find_sub_agent("analyst").name)     # analyst
```

### Example 3 — `before_agent_callback` list for access control + logging

```python
import asyncio
from typing import Optional
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types


ALLOWED_USERS = {"alice", "bob"}


def check_access(callback_context: CallbackContext) -> Optional[types.Content]:
    user = callback_context.state.get("current_user", "")
    if user not in ALLOWED_USERS:
        return types.Content(
            role="model",
            parts=[types.Part(text=f"Access denied for user: {user!r}")],
        )
    return None  # allow execution to continue


def log_invocation(callback_context: CallbackContext) -> Optional[types.Content]:
    user = callback_context.state.get("current_user", "anonymous")
    print(f"[audit] agent invoked by {user!r}")
    return None  # do not short-circuit


# Multiple callbacks are tried in order; the first non-None result wins.
agent = LlmAgent(
    name="secure_agent",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
    before_agent_callback=[check_access, log_invocation],
)

# check_access fires first; if user is unknown the agent body is skipped.
```

---

## 2 · `AuthCredential` + `AuthCredentialTypes` + `HttpAuth` + `ServiceAccount`

**Source:** `google/adk/auth/auth_credential.py`

`AuthCredential` is the unified data class for every credential shape ADK
understands.  The `auth_type` discriminator selects the credential union
member: `api_key`, `http`, `service_account`, or `oauth2`.

### Key types (verified `auth_credential.py`)

```python
class AuthCredentialTypes(str, Enum):
    API_KEY         = "apiKey"
    HTTP            = "http"
    OAUTH2          = "oauth2"
    OPEN_ID_CONNECT = "openIdConnect"
    SERVICE_ACCOUNT = "serviceAccount"

class HttpCredentials(BaseModelWithConfig):
    username: str | None = None
    password: str | None = None
    token: str | None = None

class OAuth2Auth(BaseModelWithConfig):
    client_id: str | None = None
    client_secret: str | None = None
    auth_uri: str | None = None
    access_token: str | None = None
    refresh_token: str | None = None
    code_verifier: str | None = None          # PKCE
    code_challenge_method: Literal["S256"] | None = None

class ServiceAccount(BaseModelWithConfig):
    service_account_credential: ServiceAccountCredential | None = None
    scopes: List[str] | None = None
    use_default_credential: bool | None = False
    use_id_token: bool | None = False         # for Cloud Run / Cloud Functions
    audience: str | None = None               # required when use_id_token=True

class AuthCredential(BaseModelWithConfig):
    auth_type: AuthCredentialTypes
    resource_ref: str | None = None           # future: Vertex Secret Manager ref
    api_key: str | None = None
    http: HttpAuth | None = None
    service_account: ServiceAccount | None = None
    oauth2: OAuth2Auth | None = None
```

`BaseModelWithConfig` uses `alias_generator=to_camel` so credentials
serialise to camelCase JSON automatically.

### Example 1 — API key and HTTP Bearer credentials

```python
from google.adk.auth.auth_credential import (
    AuthCredential, AuthCredentialTypes,
    HttpAuth, HttpCredentials,
)

# API key (query param / header)
api_key_cred = AuthCredential(
    auth_type=AuthCredentialTypes.API_KEY,
    api_key="AIzaSy...",
)

# HTTP Bearer token (e.g. a static JWT)
bearer_cred = AuthCredential(
    auth_type=AuthCredentialTypes.HTTP,
    http=HttpAuth(
        scheme="bearer",
        credentials=HttpCredentials(token="eyJhbGci..."),
    ),
)

# HTTP Basic
basic_cred = AuthCredential(
    auth_type=AuthCredentialTypes.HTTP,
    http=HttpAuth(
        scheme="basic",
        credentials=HttpCredentials(username="alice", password="s3cret"),
    ),
)

print(api_key_cred.model_dump(by_alias=True))
# {'authType': 'apiKey', 'apiKey': 'AIzaSy...', ...}
```

### Example 2 — OAuth2 Authorization Code with PKCE

```python
from google.adk.auth.auth_credential import (
    AuthCredential, AuthCredentialTypes, OAuth2Auth,
)

# Credential holding the client registration; auth_uri is empty until
# AuthHandler.generate_auth_uri() populates it.
oauth2_cred = AuthCredential(
    auth_type=AuthCredentialTypes.OAUTH2,
    oauth2=OAuth2Auth(
        client_id="your-client-id",
        client_secret="your-client-secret",
        redirect_uri="https://myapp.example.com/oauth/callback",
        code_challenge_method="S256",   # PKCE — verifier generated by AuthHandler
    ),
)

# Simulating a completed OAuth2 exchange (access_token + refresh_token set)
completed_cred = AuthCredential(
    auth_type=AuthCredentialTypes.OAUTH2,
    oauth2=OAuth2Auth(
        client_id="your-client-id",
        client_secret="your-client-secret",
        access_token="ya29.access...",
        refresh_token="1//refresh...",
        expires_at=1_800_000_000,
    ),
)
print(completed_cred.oauth2.access_token[:10])  # ya29.access
```

### Example 3 — Service Account with ID token for Cloud Run

```python
from google.adk.auth.auth_credential import (
    AuthCredential, AuthCredentialTypes,
    ServiceAccount, ServiceAccountCredential,
)

# use_id_token=True + audience is required for Cloud Run service-to-service auth.
# A model_validator in ServiceAccount raises ValueError if audience is omitted.
sa_cred = AuthCredential(
    auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
    service_account=ServiceAccount(
        service_account_credential=ServiceAccountCredential.model_construct(
            **{
                "type": "service_account",
                "project_id": "my-project",
                "private_key_id": "key-id",
                "private_key": "-----BEGIN RSA PRIVATE KEY-----\n...",
                "client_email": "svc@my-project.iam.gserviceaccount.com",
                "client_id": "112233",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/svc",
                "universe_domain": "googleapis.com",
            }
        ),
        use_id_token=True,
        audience="https://my-cloud-run-service-abc123-uc.a.run.app",
    ),
)
print(sa_cred.service_account.use_id_token)   # True

# ADC fallback — no JSON key required
adc_cred = AuthCredential(
    auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
    service_account=ServiceAccount(
        use_default_credential=True,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    ),
)
print(adc_cred.service_account.use_default_credential)  # True
```

---

## 3 · `RunConfig` + `StreamingMode` + `ToolThreadPoolConfig`

**Source:** `google/adk/agents/run_config.py`

`RunConfig` controls how a single `Runner.run_async()` or `Runner.run_live()`
invocation behaves.  It is passed per-call so it can differ between requests
in the same session.

### Key fields (verified `run_config.py`)

```python
class StreamingMode(Enum):
    NONE = None   # single aggregated event per turn
    SSE  = 'sse'  # yields partial streaming chunks + final aggregated event
    BIDI = 'bidi' # reserved; actual bidi uses runner.run_live()

class ToolThreadPoolConfig(BaseModel):
    max_workers: int = Field(default=4, ge=1)

class RunConfig(BaseModel):
    streaming_mode: StreamingMode = StreamingMode.NONE
    max_llm_calls: int = 500          # ≤0 → unbounded (logs a warning)
    support_cfc: bool = False         # Compositional Function Calling via LIVE API
    tool_thread_pool_config: Optional[ToolThreadPoolConfig] = None
    save_live_blob: bool = False      # persist video/audio to artifact service
    output_audio_transcription: Optional[types.AudioTranscriptionConfig] = ...
    input_audio_transcription:  Optional[types.AudioTranscriptionConfig] = ...
    get_session_config: Optional[GetSessionConfig] = None
    telemetry: TelemetryConfig | None = None
    custom_metadata: Optional[dict[str, Any]] = None
```

`max_llm_calls` is validated: `sys.maxsize` raises `ValueError`; `≤0` emits a
warning but is permitted for unbounded execution.

### Example 1 — SSE streaming with typewriter display

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

agent = LlmAgent(name="streamer", model="gemini-2.5-flash",
                 instruction="Answer concisely.")
session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="demo", session_service=session_service)

run_config = RunConfig(streaming_mode=StreamingMode.SSE)

async def stream_response(user_text: str) -> None:
    session = await session_service.create_session(app_name="demo", user_id="u1")
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=user_text)]),
        run_config=run_config,
    ):
        if event.partial and event.content:
            # Print partial chunks in-place (typewriter effect)
            for part in event.content.parts:
                if part.text and not getattr(part, "function_call", None):
                    print(part.text, end="", flush=True)
        elif not event.partial and event.is_final_response():
            print()  # newline after stream ends

# asyncio.run(stream_response("What is 2+2?"))
```

### Example 2 — thread pool for blocking live tools

```python
import time
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, ToolThreadPoolConfig
from google.adk.tools import FunctionTool


def slow_db_lookup(query: str) -> str:
    """A blocking I/O tool that would stall the event loop without thread pool."""
    time.sleep(0.5)           # simulates a blocking DB read
    return f"result for: {query}"


agent = LlmAgent(
    name="live_agent",
    model="gemini-2.5-flash",
    instruction="Use slow_db_lookup to answer questions.",
    tools=[slow_db_lookup],
)

# Thread pool lets the event loop stay responsive to model audio while
# slow_db_lookup blocks in a background thread.
run_config = RunConfig(
    tool_thread_pool_config=ToolThreadPoolConfig(max_workers=8),
)
print(run_config.tool_thread_pool_config.max_workers)  # 8
```

### Example 3 — cap LLM calls and attach per-request custom metadata

```python
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.sessions.base_session_service import GetSessionConfig

agent = LlmAgent(name="capped_agent", model="gemini-2.5-flash",
                 instruction="You are an assistant.")

run_config = RunConfig(
    # Hard-cap at 10 LLM calls per invocation to control costs.
    max_llm_calls=10,
    # Only load the 50 most recent events for large sessions.
    get_session_config=GetSessionConfig(num_recent_events=50),
    # Attach caller metadata available via invocation_context.run_config.custom_metadata
    custom_metadata={
        "request_id": "req-abc-123",
        "tenant_id": "acme-corp",
        "feature_flags": {"experimental_tools": True},
    },
)

print(run_config.max_llm_calls)                           # 10
print(run_config.custom_metadata["tenant_id"])            # acme-corp
print(run_config.get_session_config.num_recent_events)    # 50
```

---

## 4 · `LlmRequest` — LLM request assembly

**Source:** `google/adk/models/llm_request.py`

`LlmRequest` is the internal mutable container that every `BaseLlmRequestProcessor`
populates before ADK calls the model.  Understanding it lets you write processors
or inspect requests in callbacks.

### Key methods (verified `llm_request.py`)

```python
class LlmRequest(BaseModel):
    model: Optional[str] = None
    contents: list[types.Content] = Field(default_factory=list)
    config: types.GenerateContentConfig = Field(default_factory=...)
    tools_dict: dict[str, BaseTool] = Field(default_factory=dict, exclude=True)
    cache_config: Optional[ContextCacheConfig] = None
    previous_interaction_id: Optional[str] = None  # for Interactions API chaining

    def append_instructions(self, instructions: list[str] | types.Content) -> list[types.Content]: ...
    def append_tools(self, tools: list[BaseTool]) -> None: ...
    def set_output_schema(self, output_schema: SchemaType, *, base_model: SchemaType = None) -> None: ...
```

`append_tools` merges new `FunctionDeclaration`s into the **single**
`types.Tool(function_declarations=[...])` object that already exists in
`config.tools`, avoiding duplicate wrapper objects.

### Example 1 — inspecting and extending a request in a plugin

```python
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types


class DebugRequestPlugin(BasePlugin):
    """Logs the assembled system instruction before each LLM call."""

    async def before_model_call(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> None:
        si = llm_request.config.system_instruction
        n_tools = sum(
            len(t.function_declarations or [])
            for t in (llm_request.config.tools or [])
            if isinstance(t, types.Tool)
        )
        print(
            f"[DebugRequestPlugin] system_instruction length={len(si or '')}, "
            f"tools={n_tools}, contents={len(llm_request.contents)}"
        )
```

### Example 2 — `append_instructions` with binary inline data

```python
from google.adk.models.llm_request import LlmRequest
from google.genai import types

req = LlmRequest(model="gemini-2.5-flash")

# Text instructions are concatenated with double newlines
req.append_instructions(["You are a vision expert.", "Be concise."])
assert "vision expert" in req.config.system_instruction
assert "concise" in req.config.system_instruction

# Passing a Content with an inline image: the text parts go to
# system_instruction; the image part becomes a user content reference.
image_bytes = b"\x89PNG\r\n..."   # placeholder PNG bytes
instruction_with_image = types.Content(
    role="user",
    parts=[
        types.Part(text="Use this logo as context for all responses."),
        types.Part(inline_data=types.Blob(mime_type="image/png", data=image_bytes)),
    ],
)
extra_user_contents = req.append_instructions(instruction_with_image)
# extra_user_contents[0] is a user Content with the PNG, now prepended to req.contents
print(len(extra_user_contents))          # 1
print(req.contents[0].parts[0].text)     # "Referenced inline data: inline_data_0"
```

### Example 3 — `set_output_schema` for structured JSON output

```python
from pydantic import BaseModel as PydanticModel
from google.adk.models.llm_request import LlmRequest


class AnalysisResult(PydanticModel):
    sentiment: str
    confidence: float
    keywords: list[str]


req = LlmRequest(model="gemini-2.5-flash")
req.set_output_schema(AnalysisResult)

print(req.config.response_mime_type)      # application/json
# response_schema holds the Pydantic model class; the Gemini SDK converts it
# to a JSON Schema when calling the API.
print(req.config.response_schema)         # <class 'AnalysisResult'>

# list schema is also supported
req2 = LlmRequest(model="gemini-2.5-flash")
req2.set_output_schema(list[AnalysisResult])
print(req2.config.response_schema)        # list[AnalysisResult]
```

---

## 5 · `AuthHandler` — OAuth / OIDC flow orchestration

**Source:** `google/adk/auth/auth_handler.py`

`AuthHandler` is the component that drives the three-phase OAuth flow inside
ADK: (1) build an auth redirect URI, (2) store the credential once the user
completes the browser flow, (3) exchange an auth code for an access token.
It is used internally by `_AuthLlmRequestProcessor` but you can use it
directly in custom auth toolsets or middleware.

### Key methods (verified `auth_handler.py`)

```python
class AuthHandler:
    def __init__(self, auth_config: AuthConfig): ...

    # Phase 1 — build the auth request to show the user
    def generate_auth_request(self) -> AuthConfig: ...

    # Phase 1b — generate the actual redirect URI (PKCE + state)
    def generate_auth_uri(self) -> AuthCredential: ...

    # Phase 2 — store the returned credential in session state
    async def parse_and_store_auth_response(self, state: State) -> None: ...

    # Phase 3 — exchange auth code → access token via OAuth2CredentialExchanger
    async def exchange_auth_token(self) -> AuthCredential: ...
```

`generate_auth_uri` uses `authlib.OAuth2Session.create_authorization_url`.
For PKCE (`code_challenge_method="S256"`), if no `code_verifier` is provided
one is generated with `generate_token(48)`.

### Example 1 — generating an OAuth2 auth URI for a custom tool

```python
from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowAuthorizationCode
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth
from google.adk.auth.auth_handler import AuthHandler
from google.adk.auth.auth_tool import AuthConfig

# Define the OAuth2 scheme (mirrors an OpenAPI securityScheme)
auth_scheme = OAuth2(
    flows=OAuthFlows(
        authorizationCode=OAuthFlowAuthorizationCode(
            authorizationUrl="https://accounts.google.com/o/oauth2/v2/auth",
            tokenUrl="https://oauth2.googleapis.com/token",
            scopes={"https://www.googleapis.com/auth/drive.readonly": "Read Drive"},
        )
    )
)

raw_credential = AuthCredential(
    auth_type=AuthCredentialTypes.OAUTH2,
    oauth2=OAuth2Auth(
        client_id="your-client-id",
        client_secret="your-client-secret",
        redirect_uri="https://myapp.example.com/callback",
        code_challenge_method="S256",  # PKCE
    ),
)

auth_config = AuthConfig(
    auth_scheme=auth_scheme,
    raw_auth_credential=raw_credential,
    credential_key="drive_cred",
)

handler = AuthHandler(auth_config=auth_config)
auth_request = handler.generate_auth_request()

# auth_request.exchanged_auth_credential.oauth2.auth_uri is now populated
print(auth_request.exchanged_auth_credential.oauth2.auth_uri[:40])
# 'https://accounts.google.com/o/oauth2/v2/...'
```

### Example 2 — storing auth response in session state

```python
import asyncio
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth
from google.adk.auth.auth_handler import AuthHandler
from google.adk.auth.auth_tool import AuthConfig
from google.adk.sessions.state import State
from fastapi.openapi.models import OAuth2, OAuthFlows, OAuthFlowAuthorizationCode

# After the user completes the browser OAuth flow, ADK receives the
# auth_response_uri (containing ?code=...&state=...).
exchanged_credential = AuthCredential(
    auth_type=AuthCredentialTypes.OAUTH2,
    oauth2=OAuth2Auth(
        client_id="your-client-id",
        client_secret="your-client-secret",
        redirect_uri="https://myapp.example.com/callback",
        auth_response_uri="https://myapp.example.com/callback?code=auth_code&state=xyz",
        auth_code="auth_code",
    ),
)

auth_scheme = OAuth2(
    flows=OAuthFlows(
        authorizationCode=OAuthFlowAuthorizationCode(
            authorizationUrl="https://accounts.google.com/o/oauth2/v2/auth",
            tokenUrl="https://oauth2.googleapis.com/token",
            scopes={},
        )
    )
)

auth_config = AuthConfig(
    auth_scheme=auth_scheme,
    exchanged_auth_credential=exchanged_credential,
    credential_key="drive_cred",
)
handler = AuthHandler(auth_config=auth_config)

# Creates state["temp:drive_cred"] = <exchanged_token_credential>
# For OAuth2/OIDC schemes, exchange_auth_token() is called automatically
# inside parse_and_store_auth_response.
state = State({}, {})
# asyncio.run(handler.parse_and_store_auth_response(state=state))
print("Credential key prefix:", "temp:" + auth_config.credential_key)
```

### Example 3 — `get_auth_response` to retrieve a stored credential

```python
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Auth
from google.adk.auth.auth_handler import AuthHandler
from google.adk.auth.auth_tool import AuthConfig
from google.adk.sessions.state import State
from fastapi.openapi.models import APIKey, APIKeyIn

# API key example — no token exchange needed
api_key_scheme = APIKey(**{"in": APIKeyIn.header, "name": "X-API-Key"})
api_key_credential = AuthCredential(
    auth_type=AuthCredentialTypes.API_KEY,
    api_key="sk-live-abc123",
)

auth_config = AuthConfig(
    auth_scheme=api_key_scheme,
    exchanged_auth_credential=api_key_credential,
    credential_key="my_api_key",
)
handler = AuthHandler(auth_config=auth_config)

# Simulate state that already has the credential stored
state = State({}, {"temp:my_api_key": api_key_credential})
retrieved = handler.get_auth_response(state)
print(retrieved.api_key)  # sk-live-abc123
```

---

## 6 · `EventsCompactionConfig` + `ResumabilityConfig`

**Source:** `google/adk/apps/_configs.py`

Both configs are passed to `App(...)` to manage long-running sessions.
`EventsCompactionConfig` controls sliding-window summarisation of the event
history to keep prompt sizes manageable. `ResumabilityConfig` enables
best-effort resumption after agent pauses or failures.

### Constructor fields (verified `_configs.py`)

```python
@experimental
class ResumabilityConfig(BaseModel):
    is_resumable: bool = False

@experimental
class EventsCompactionConfig(BaseModel):
    summarizer: Optional[BaseEventsSummarizer] = None
    compaction_interval: int         # trigger after N new user invocations
    overlap_size: int                # preceding invocations kept un-compacted for context
    token_threshold: Optional[int]   # also trigger when prompt tokens ≥ threshold
    event_retention_size: Optional[int]  # raw events retained when token trigger fires
    # model_validator: token_threshold and event_retention_size must be set together
```

### Example 1 — interval-based compaction with a custom summarizer

```python
from google.adk.apps.app import App
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.events.event import Event
from google.adk.agents import LlmAgent
from typing import Optional


class EchoSummarizer(BaseEventsSummarizer):
    """Stub: just returns a placeholder summary event."""

    async def maybe_summarize_events(self, *, events: list[Event]) -> Optional[Event]:
        # A real implementation would call an LLM here.
        if not events:
            return None
        print(f"[EchoSummarizer] compacting {len(events)} events")
        return None   # returning None skips actual compaction


agent = LlmAgent(name="root", model="gemini-2.5-flash", instruction="Answer briefly.")

app = App(
    name="compact_app",
    root_agent=agent,
    events_compaction_config=EventsCompactionConfig(
        summarizer=EchoSummarizer(),
        compaction_interval=5,   # compact after every 5 user turns
        overlap_size=2,          # keep the last 2 turns un-compacted for context
    ),
)
print(app.events_compaction_config.compaction_interval)  # 5
```

### Example 2 — token-threshold based compaction

```python
from google.adk.apps.app import App
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.agents import LlmAgent
from google.adk.models.gemini_llm import GeminiLlm

summarizer_llm = GeminiLlm(model="gemini-2.5-flash")

agent = LlmAgent(name="root", model="gemini-2.5-flash", instruction="Be concise.")

app = App(
    name="token_compact_app",
    root_agent=agent,
    events_compaction_config=EventsCompactionConfig(
        summarizer=LlmEventSummarizer(llm=summarizer_llm),
        compaction_interval=10,
        overlap_size=3,
        # Compact when the prompt hits 100k tokens, retaining the last 20 raw events
        token_threshold=100_000,
        event_retention_size=20,
    ),
)
print(app.events_compaction_config.token_threshold)      # 100000
print(app.events_compaction_config.event_retention_size) # 20
```

### Example 3 — enabling resumability for long-running tasks

```python
from google.adk.apps.app import App
from google.adk.apps._configs import ResumabilityConfig
from google.adk.agents import LlmAgent

# ResumabilityConfig.is_resumable=True allows the Runner to pause and
# later re-enter an in-progress invocation without restarting from scratch.
agent = LlmAgent(
    name="long_runner",
    model="gemini-2.5-flash",
    instruction="Handle multi-step tasks that may take minutes.",
)

app = App(
    name="resumable_app",
    root_agent=agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)
print(app.resumability_config.is_resumable)  # True

# Combining resumability + compaction:
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer

class NoOpSummarizer(BaseEventsSummarizer):
    async def maybe_summarize_events(self, *, events):
        return None

app2 = App(
    name="resilient_app",
    root_agent=agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
    events_compaction_config=EventsCompactionConfig(
        summarizer=NoOpSummarizer(),
        compaction_interval=20,
        overlap_size=5,
    ),
)
```

---

## 7 · `LlmEventSummarizer` — LLM-based sliding window compaction

**Source:** `google/adk/apps/llm_event_summarizer.py`

`LlmEventSummarizer` is the production-ready implementation of
`BaseEventsSummarizer` shipped with ADK.  It formats session events into a
readable dialogue transcript, calls an LLM to summarise them, and packages
the result as an `Event` with an `EventCompaction` action.

### Key internals (verified `llm_event_summarizer.py`)

```python
class LlmEventSummarizer(BaseEventsSummarizer):
    _DEFAULT_PROMPT_TEMPLATE = (
        "The following is a conversation history ... "
        "{conversation_history}"
    )
    _MAX_TOOL_CONTENT_CHARS = 2000   # caps tool args/responses to prevent bloat

    def __init__(self, llm: BaseLlm, prompt_template: Optional[str] = None): ...

    def _format_events_for_prompt(self, events: list[Event]) -> str:
        # Includes: model text, user text, thought parts (non-compaction only),
        # function_call args (truncated), function_response results (truncated)

    async def maybe_summarize_events(self, *, events: list[Event]) -> Optional[Event]:
        # Returns an Event with actions.compaction set; EventCompaction carries
        # start_timestamp, end_timestamp, compacted_content (role="model")
```

### Example 1 — basic setup with the default prompt

```python
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models.gemini_llm import GeminiLlm

llm = GeminiLlm(model="gemini-2.5-flash")
summarizer = LlmEventSummarizer(llm=llm)

# Default prompt template is used when none is supplied
print("{conversation_history}" in summarizer._prompt_template)  # True
print(summarizer._MAX_TOOL_CONTENT_CHARS)                        # 2000
```

### Example 2 — custom domain-specific prompt template

```python
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models.gemini_llm import GeminiLlm

SUPPORT_TEMPLATE = """\
You are summarising a customer support transcript.
Focus on: reported issue, steps taken, resolution status, and open action items.

Transcript:
{conversation_history}

Provide a structured summary with sections: Issue, Steps Taken, Resolution, Next Steps."""

llm = GeminiLlm(model="gemini-2.5-flash")
summarizer = LlmEventSummarizer(llm=llm, prompt_template=SUPPORT_TEMPLATE)

print(summarizer._prompt_template[:50])   # "You are summarising a customer support transcript."
```

### Example 3 — inspecting `_format_events_for_prompt` output

```python
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.models.gemini_llm import GeminiLlm
from google.genai import types

llm = GeminiLlm(model="gemini-2.5-flash")
summarizer = LlmEventSummarizer(llm=llm)

# Build synthetic events to inspect the formatter
events = [
    Event(
        invocation_id="inv1", author="user",
        content=types.Content(role="user", parts=[types.Part(text="What is the weather?")]),
        actions=EventActions(),
    ),
    Event(
        invocation_id="inv1", author="weather_agent",
        content=types.Content(
            role="model",
            parts=[
                types.Part(thought=True, text="I should call the weather tool."),
                types.Part(text="The weather in London is 18°C and cloudy."),
            ],
        ),
        actions=EventActions(),
    ),
]

formatted = summarizer._format_events_for_prompt(events)
print(formatted)
# user: What is the weather?
# weather_agent (thought): I should call the weather tool.
# weather_agent: The weather in London is 18°C and cloudy.
```

---

## 8 · `ToolConfirmation` — human-in-the-loop tool execution gates

**Source:** `google/adk/tools/tool_confirmation.py`

`ToolConfirmation` is a small `@experimental` Pydantic model that a tool's
`require_confirmation` callable returns when the framework should pause and
wait for an explicit user decision before executing the tool.  The agent
emits an `adk_request_confirmation` function call; the user responds with
a `ToolConfirmation` JSON payload; and
`_RequestConfirmationLlmRequestProcessor` re-executes the confirmed tools.

### Model fields (verified `tool_confirmation.py`)

```python
@experimental(FeatureName.TOOL_CONFIRMATION)
class ToolConfirmation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        alias_generator=alias_generators.to_camel,  # serialises to camelCase
        populate_by_name=True,
    )

    hint: str = ""            # shown to the user explaining why confirmation is needed
    confirmed: bool = False   # must be True for the tool to run
    payload: Optional[Any] = None  # extra JSON data the tool needs from the user
```

### Example 1 — `require_confirmation` callable on a `FunctionTool`

```python
from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_confirmation import ToolConfirmation
from typing import Optional


def delete_record(record_id: str) -> str:
    """Permanently deletes the specified record from the database."""
    return f"Record {record_id} deleted."


def confirm_delete(ctx: ReadonlyContext) -> Optional[ToolConfirmation]:
    """Gate expensive / irreversible operations behind a user prompt."""
    return ToolConfirmation(
        hint="This will permanently delete the record. Please confirm.",
        confirmed=False,         # starts unconfirmed; user must set to True
    )


# require_confirmation is a field on FunctionTool, not on LlmAgent.
# Pass a bool (always gate) or a callable (ctx: ReadonlyContext) -> Optional[ToolConfirmation].
delete_tool = FunctionTool(delete_record, require_confirmation=confirm_delete)

agent = LlmAgent(
    name="admin_agent",
    model="gemini-2.5-flash",
    instruction="You manage database records.",
    tools=[delete_tool],
)
```

### Example 2 — `payload` for collecting extra user input before tool execution

```python
from google.adk.tools.tool_confirmation import ToolConfirmation
from typing import Optional
from google.adk.agents.readonly_context import ReadonlyContext


def transfer_funds(amount: float, destination_account: str) -> dict:
    """Transfers funds to the specified account."""
    return {"status": "success", "amount": amount, "to": destination_account}


def confirm_transfer(ctx: ReadonlyContext) -> Optional[ToolConfirmation]:
    """Ask the user for a 2FA code before executing the transfer."""
    return ToolConfirmation(
        hint="Enter your 2FA code to authorise this transfer.",
        confirmed=False,
        payload={"requires_2fa_code": True},  # client renders a 2FA input
    )


# When the user responds with:
#   ToolConfirmation(confirmed=True, payload={"2fa_code": "123456"})
# _RequestConfirmationLlmRequestProcessor re-runs transfer_funds.

confirmed = ToolConfirmation(confirmed=True, payload={"2fa_code": "123456"})
print(confirmed.model_dump(by_alias=True))
# {'hint': '', 'confirmed': True, 'payload': {'2fa_code': '123456'}}
```

### Example 3 — round-trip serialisation (camelCase aliases)

```python
from google.adk.tools.tool_confirmation import ToolConfirmation
import json

# Tools send ToolConfirmation as camelCase JSON to the client
tc = ToolConfirmation(hint="Confirm deletion?", confirmed=False, payload=None)
serialised = tc.model_dump(by_alias=True, mode="json")
print(json.dumps(serialised, indent=2))
# {
#   "hint": "Confirm deletion?",
#   "confirmed": false,
#   "payload": null
# }

# Client responds with camelCase; ADK parses it via populate_by_name=True
response_json = {"hint": "", "confirmed": True, "payload": {"note": "approved"}}
parsed = ToolConfirmation.model_validate(response_json)
print(parsed.confirmed, parsed.payload)   # True {'note': 'approved'}
```

---

## 9 · `LiveRequest` + `LiveRequestQueue` + `ActiveStreamingTool`

**Source:** `google/adk/agents/live_request_queue.py`, `google/adk/agents/active_streaming_tool.py`

These three classes form the bidirectional streaming infrastructure for
`runner.run_live()` sessions.  `LiveRequest` wraps all possible input types
with a clear priority order.  `LiveRequestQueue` is an `asyncio.Queue` with
domain-specific send helpers.  `ActiveStreamingTool` tracks the asyncio task
and input queue for a currently executing streaming tool.

### Key types (verified source)

```python
class LiveRequest(BaseModel):
    """Priority order (highest first): activity_start > activity_end > blob > content."""
    content: Optional[types.Content] = None         # text / structured turn
    blob: Optional[types.Blob] = None               # audio / video realtime chunk
    activity_start: Optional[types.ActivityStart] = None
    activity_end: Optional[types.ActivityEnd] = None
    close: bool = False                             # signals end of stream

class LiveRequestQueue:
    def send_content(self, content: types.Content) -> None: ...
    def send_realtime(self, blob: types.Blob) -> None: ...
    def send_activity_start(self) -> None: ...
    def send_activity_end(self) -> None: ...
    def close(self) -> None: ...
    async def get(self) -> LiveRequest: ...

class ActiveStreamingTool(BaseModel):
    task: Optional[asyncio.Task] = None   # background asyncio.Task
    stream: Optional[LiveRequestQueue] = None  # input feed into the tool
```

### Example 1 — sending text and audio to a live session

```python
import asyncio
from google.adk.agents.live_request_queue import LiveRequestQueue, LiveRequest
from google.genai import types

queue = LiveRequestQueue()

# Send a text turn (e.g. from a client WebSocket message)
queue.send_content(
    types.Content(role="user", parts=[types.Part(text="Hello, can you hear me?")])
)

# Send a realtime audio chunk (e.g. from a microphone stream)
audio_bytes = b"\x00\x01" * 160  # 20ms of 8kHz 16-bit PCM
queue.send_realtime(types.Blob(mime_type="audio/pcm;rate=8000", data=audio_bytes))

# Signal voice activity detection events
queue.send_activity_start()   # user started speaking
queue.send_activity_end()     # user stopped speaking

# Retrieve the first item (would normally happen in the runner's async loop)
async def drain_one():
    item = await queue.get()
    if item.content:
        print("got text:", item.content.parts[0].text)
    elif item.blob:
        print("got audio blob, size:", len(item.blob.data))

asyncio.run(drain_one())   # got text: Hello, can you hear me?
```

### Example 2 — building a streaming tool that reads from its `LiveRequestQueue`

```python
import asyncio
from typing import AsyncGenerator
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.tools.tool_context import ToolContext


async def transcribe_stream(tool_context: ToolContext) -> AsyncGenerator[str, None]:
    """Streaming tool: drains the tool's input queue and yields transcript lines."""
    active_tool = tool_context.active_streaming_tool
    if active_tool is None or active_tool.stream is None:
        yield "No stream available"
        return

    words: list[str] = []
    while True:
        try:
            req = await asyncio.wait_for(active_tool.stream.get(), timeout=0.1)
        except asyncio.TimeoutError:
            break
        if req.close:
            break
        if req.content:
            for part in req.content.parts:
                if part.text:
                    words.append(part.text)

    yield " ".join(words) if words else "(silence)"
```

### Example 3 — `ActiveStreamingTool` lifecycle management in a custom runner

```python
import asyncio
from google.adk.agents.active_streaming_tool import ActiveStreamingTool
from google.adk.agents.live_request_queue import LiveRequestQueue


async def run_streaming_tool_lifecycle():
    """Shows how ActiveStreamingTool wraps a background task + input queue."""
    queue = LiveRequestQueue()

    async def fake_tool_body():
        while True:
            req = await queue.get()
            if req.close:
                break
            if req.content:
                print("tool received:", req.content.parts[0].text)

    task = asyncio.create_task(fake_tool_body())
    active = ActiveStreamingTool(task=task, stream=queue)

    # Feed data to the tool
    from google.genai import types
    active.stream.send_content(
        types.Content(role="user", parts=[types.Part(text="ping")])
    )
    active.stream.close()

    # Wait for the tool to finish
    await active.task
    print("task done:", active.task.done())  # True


asyncio.run(run_streaming_tool_lifecycle())
# tool received: ping
# task done: True
```

---

## 10 · Model name utilities — routing by model family

**Source:** `google/adk/utils/model_name_utils.py`

These module-level functions let code branches on model family without
hard-coding string prefixes.  They normalise both simple names
(`"gemini-2.5-flash"`) and Vertex AI path forms
(`"projects/my-proj/locations/us-central1/publishers/google/models/gemini-2.5-flash"`).

### Key functions (verified `model_name_utils.py`)

```python
def extract_model_name(model_string: str) -> str:
    # Strips Vertex AI path prefix and 'models/' prefix; returns bare model id
    # Patterns: projects/.../publishers/.../models/<name>
    #           apigee/.../<name>
    #           models/<name>

def is_gemini_model(model_string: Optional[str]) -> bool:
    # True if extract_model_name(...).startswith('gemini-')

def is_gemini_1_model(model_string: Optional[str]) -> bool:
    # True if matches ^gemini-1\.\d+

def is_gemini_eap_or_2_or_above(model_string: Optional[str]) -> bool:
    # True for EAP names (gemini-<variant>-early-expN)
    # OR packaging.version.Version(major) >= 2

def is_gemini_3_1_flash_live(model_string: Optional[str]) -> bool:
def is_gemini_3_5_live_translate(model_string: Optional[str]) -> bool:
```

### Example 1 — normalising Vertex AI path-based model IDs

```python
from google.adk.utils.model_name_utils import extract_model_name, is_gemini_model

# Simple name — returned as-is
assert extract_model_name("gemini-2.5-flash") == "gemini-2.5-flash"

# Vertex AI full path
vertex_path = "projects/my-proj/locations/us-central1/publishers/google/models/gemini-2.5-pro"
assert extract_model_name(vertex_path) == "gemini-2.5-pro"

# models/ prefix (used by some SDKs)
assert extract_model_name("models/gemini-2.5-flash") == "gemini-2.5-flash"

# Apigee path
apigee_path = "apigee/gemini-2.5-flash"
assert extract_model_name(apigee_path) == "gemini-2.5-flash"

# is_gemini_model handles all path forms
assert is_gemini_model(vertex_path) is True
assert is_gemini_model("gpt-4o") is False
assert is_gemini_model(None) is False
```

### Example 2 — routing logic by model generation

```python
from google.adk.utils.model_name_utils import (
    is_gemini_model,
    is_gemini_1_model,
    is_gemini_eap_or_2_or_above,
)


def select_output_strategy(model: str) -> str:
    """Return the output schema strategy appropriate for the model."""
    if not is_gemini_model(model):
        # Non-Gemini models (LiteLLM, etc.) handle response_format natively
        return "litellm_response_format"
    if is_gemini_1_model(model):
        # Gemini 1.x does not support output schema with tools simultaneously
        return "set_model_response_tool_workaround"
    if is_gemini_eap_or_2_or_above(model):
        # Gemini 2+ and EAP models support response_schema + tools together on Vertex
        return "native_output_schema"
    return "unknown"


print(select_output_strategy("gemini-1.5-pro"))           # set_model_response_tool_workaround
print(select_output_strategy("gemini-2.5-flash"))         # native_output_schema
print(select_output_strategy("gemini-flash-early-exp"))   # native_output_schema (EAP)
print(select_output_strategy("gpt-4o"))                   # litellm_response_format
```

### Example 3 — bypassing model ID validation for internal model names

```python
import os
from google.adk.utils.model_name_utils import (
    is_gemini_model,
    is_gemini_model_id_check_disabled,
)

# In normal usage the check is enabled
assert is_gemini_model_id_check_disabled() is False

# Internal deployments can set ADK_DISABLE_GEMINI_MODEL_ID_CHECK=true to bypass
# the ^gemini- prefix check for non-public model IDs.
os.environ["ADK_DISABLE_GEMINI_MODEL_ID_CHECK"] = "true"
print(is_gemini_model_id_check_disabled())  # True

# Note: is_gemini_model itself still uses extract_model_name + regex;
# the env var is checked by downstream consumers (e.g. output_schema_utils)
# to skip the model-family guard when routing decisions are made.
os.environ.pop("ADK_DISABLE_GEMINI_MODEL_ID_CHECK", None)
print(is_gemini_model_id_check_disabled())  # False
```

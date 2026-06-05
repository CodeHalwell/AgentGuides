---
title: "Class deep dives — volume 13 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.2.0 classes: ApigeeLlm/ApiType (Apigee proxy routing for enterprise API governance), AudioCacheManager/AudioCacheConfig (live bidirectional audio caching + artifact flush), AudioTranscriber (Cloud Speech-to-Text for live agents), TranscriptionManager (transcription event lifecycle), AgentEngineSandboxComputer/VMaaS (Vertex AI Agent Engine sandbox computer; BYOS + auto-provision), ParameterManagerClient (GCP Parameter Manager config retrieval), SecretManagerClient (GCP Secret Manager credential retrieval), GcpAuthProvider/GcpAuthProviderScheme (Agent Identity IAM Connector Credentials; 2-legged and 3-legged OAuth; polling), AgentRegistry/AgentRegistrySingleMcpToolset (Google Cloud Agent Registry; A2A agent + MCP server discovery), OAuth2CredentialRefresher/BaseCredentialRefresher/InMemoryCredentialService/BaseCredentialService (pluggable credential refresh and storage pipeline)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 13"
  order: 78
---

Source-verified against **google-adk==2.2.0** (installed from PyPI, June 2026). Every field name, signature, and code example is drawn from the installed package source at `/usr/local/lib/python3.11/dist-packages/google/adk/`.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `ApigeeLlm` + `ApiType` | `google.adk.models.apigee_llm` | Stable |
| 2 | `AudioCacheManager` + `AudioCacheConfig` | `google.adk.flows.llm_flows.audio_cache_manager` | Stable |
| 3 | `AudioTranscriber` | `google.adk.flows.llm_flows.audio_transcriber` | Stable |
| 4 | `TranscriptionManager` | `google.adk.flows.llm_flows.transcription_manager` | Stable |
| 5 | `AgentEngineSandboxComputer` | `google.adk.integrations.vmaas.sandbox_computer` | Experimental |
| 6 | `ParameterManagerClient` | `google.adk.integrations.parameter_manager.parameter_client` | Stable |
| 7 | `SecretManagerClient` | `google.adk.integrations.secret_manager.secret_client` | Stable |
| 8 | `GcpAuthProvider` + `GcpAuthProviderScheme` | `google.adk.integrations.agent_identity` | Experimental |
| 9 | `AgentRegistry` + `AgentRegistrySingleMcpToolset` | `google.adk.integrations.agent_registry.agent_registry` | Experimental |
| 10 | `OAuth2CredentialRefresher` + `BaseCredentialRefresher` + `InMemoryCredentialService` + `BaseCredentialService` | `google.adk.auth.refresher`, `.credential_service` | Experimental |

---

## 1 · `ApigeeLlm` + `ApiType`

**Source:** `google.adk.models.apigee_llm`

`ApigeeLlm` routes all Gemini (and OpenAI-compatible) LLM calls through an **Apigee API proxy** instead of calling Google endpoints directly. This gives enterprise teams a central governance layer: rate limiting, usage metering, custom authentication, request/response logging, and traffic shaping — all without changing agent code.

`ApigeeLlm` extends `Gemini` (the standard Vertex AI / Google AI model backend) and overrides `generate_content_async` to construct requests against the proxy URL instead.

### Constructor (source-verified)

```python
from google.adk.models.apigee_llm import ApigeeLlm

ApigeeLlm(
    *,
    model: str,                             # required — see format below
    proxy_url: str | None = None,           # falls back to APIGEE_PROXY_URL env var
    custom_headers: dict[str, str] | None = None,
    retry_options: types.HttpRetryOptions | None = None,
    api_type: ApigeeLlm.ApiType | str = ApigeeLlm.ApiType.UNKNOWN,
    credentials: google.auth.credentials.Credentials | None = None,
)
```

### `ApiType` enum

```python
class ApigeeLlm.ApiType(str, enum.Enum):
    UNKNOWN           = "unknown"          # auto-detected from model string
    CHAT_COMPLETIONS  = "chat_completions" # OpenAI /chat/completions API shape
    GENAI             = "genai"            # Google GenAI API shape
```

The `api_type` is **automatically inferred** from the model string prefix:
- `apigee/openai/…` → `CHAT_COMPLETIONS`
- `apigee/gemini/…` or `apigee/vertex_ai/…` → `GENAI`
- anything else → `GENAI` (default)

### Model string format

```
apigee/[<provider>/][<version>/]<model_id>

provider  : vertex_ai | gemini  (optional; overrides GOOGLE_GENAI_USE_VERTEXAI)
version   : v1 | v1beta         (optional; uses provider default if omitted)
model_id  : gemini-2.5-flash    (required)

Examples:
  apigee/gemini-2.5-flash                 → Gemini AI  default version
  apigee/v1/gemini-2.5-flash              → Gemini AI  v1
  apigee/vertex_ai/gemini-2.5-flash       → Vertex AI  default version
  apigee/gemini/v1/gemini-2.5-flash       → Gemini AI  v1  explicit
  apigee/vertex_ai/v1beta/gemini-2.5-flash→ Vertex AI  v1beta
  apigee/openai/gpt-4o                    → chat_completions mode
```

### Minimal usage

```python
import os
from google.adk.agents import LlmAgent
from google.adk.models.apigee_llm import ApigeeLlm

# The proxy URL can come from the constructor or the APIGEE_PROXY_URL env var.
os.environ["APIGEE_PROXY_URL"] = "https://my-org.apigee.net/gemini-proxy"

agent = LlmAgent(
    name="enterprise_assistant",
    model=ApigeeLlm(model="apigee/gemini-2.5-flash"),
    instruction="You are a helpful enterprise assistant.",
)
```

### Vertex AI backend via Apigee

```python
import os
from google.adk.agents import LlmAgent
from google.adk.models.apigee_llm import ApigeeLlm

# Required for vertex_ai provider
os.environ["GOOGLE_CLOUD_PROJECT"] = "my-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

llm = ApigeeLlm(
    model="apigee/vertex_ai/gemini-2.5-flash",
    proxy_url="https://my-org.apigee.net/vertex-proxy",
    custom_headers={"x-api-key": "my-api-key"},
)

agent = LlmAgent(name="va_agent", model=llm, instruction="Answer concisely.")
```

### OpenAI-compatible model through Apigee

```python
from google.adk.agents import LlmAgent
from google.adk.models.apigee_llm import ApigeeLlm

# Route GPT-4o through an Apigee proxy that normalises OpenAI responses
llm = ApigeeLlm(
    model="apigee/openai/gpt-4o",
    proxy_url="https://my-org.apigee.net/openai-proxy",
    api_type=ApigeeLlm.ApiType.CHAT_COMPLETIONS,
)

agent = LlmAgent(name="gpt_agent", model=llm, instruction="Be concise.")
```

### Custom credentials scope

```python
import google.auth
from google.adk.models.apigee_llm import ApigeeLlm

# Some Apigee proxies use tokeninfo to identify callers —
# pass credentials with the userinfo.email scope.
creds, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/cloud-platform"]
)
llm = ApigeeLlm(
    model="apigee/vertex_ai/gemini-2.5-flash",
    proxy_url="https://my-org.apigee.net/vertex-proxy",
    credentials=creds,
)
```

### Multi-model failover via Apigee

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.apigee_llm import ApigeeLlm
from google.adk.runners import InMemoryRunner

PRIMARY_PROXY  = "https://my-org.apigee.net/gemini-primary"
FALLBACK_PROXY = "https://my-org.apigee.net/gemini-fallback"

primary_agent = LlmAgent(
    name="primary",
    model=ApigeeLlm(model="apigee/gemini-2.5-flash", proxy_url=PRIMARY_PROXY),
    instruction="You are the primary assistant.",
)

# The APIGEE_PROXY_URL fallback env var provides a default, so agents that
# don't specify proxy_url automatically pick up the organisation-wide proxy.
fallback_agent = LlmAgent(
    name="fallback",
    model=ApigeeLlm(model="apigee/gemini-2.5-flash", proxy_url=FALLBACK_PROXY),
    instruction="You are the fallback assistant.",
)
```

### When to use

| Scenario | Use `ApigeeLlm` |
|---|---|
| Central API rate limiting across all teams | Yes |
| Per-request metering / cost allocation | Yes |
| Request/response audit logging via Apigee policies | Yes |
| Custom auth (API key, mTLS) at proxy layer | Yes |
| Direct Gemini or Vertex AI without proxy | No — use `Gemini` / `VertexAi` |

---

## 2 · `AudioCacheManager` + `AudioCacheConfig`

**Source:** `google.adk.flows.llm_flows.audio_cache_manager`

In **live bidirectional audio sessions** the agent receives a continuous stream of small audio blobs from the user and emits blobs back. `AudioCacheManager` buffers these chunks on `InvocationContext` and later **flushes** them as combined audio files to the `ArtifactService`, creating session-persisted events so the audio is queryable after the session ends.

### Constructor (source-verified)

```python
from google.adk.flows.llm_flows.audio_cache_manager import (
    AudioCacheManager,
    AudioCacheConfig,
)

AudioCacheManager(config: AudioCacheConfig | None = None)

AudioCacheConfig(
    max_cache_size_bytes: int   = 10 * 1024 * 1024,  # 10 MB
    max_cache_duration_seconds: float = 300.0,        # 5 minutes
    auto_flush_threshold: int   = 100,                # chunk count
)
```

### Key methods (source-verified)

```python
# Append one audio blob to the in-memory buffer.
manager.cache_audio(
    invocation_context,
    audio_blob: types.Blob,
    cache_type: str,           # "input" (user) or "output" (model)
)

# Combine all buffered blobs → save to ArtifactService → return Events.
events: list[Event] = await manager.flush_caches(
    invocation_context,
    flush_user_audio:  bool = True,
    flush_model_audio: bool = True,
)

# Inspect buffer size without flushing.
stats: dict[str, int] = manager.get_cache_stats(invocation_context)
# keys: input_chunks, output_chunks, input_bytes, output_bytes,
#       total_chunks, total_bytes
```

### Artifact URI convention

When `flush_caches()` saves a buffer it stores the file as:

```
artifact://<app_name>/<user_id>/<session_id>/_adk_live/
    adk_live_audio_storage_<type>_<timestamp_ms>.<ext>#<revision>
```

The `Event` that gets created holds a `FileData` part pointing to this URI — so the conversation history contains a reference to the audio, not the raw bytes.

### Usage: persist live session audio

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.flows.llm_flows.audio_cache_manager import (
    AudioCacheManager,
    AudioCacheConfig,
)
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types


class AudioPersistencePlugin(BasePlugin):
    """Flush audio caches after each turn completes."""

    def __init__(self):
        self._manager = AudioCacheManager(
            config=AudioCacheConfig(
                max_cache_size_bytes=50 * 1024 * 1024,  # 50 MB
                auto_flush_threshold=200,
            )
        )

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        stats = self._manager.get_cache_stats(invocation_context)
        if stats["total_chunks"] > 0:
            events = await self._manager.flush_caches(invocation_context)
            print(f"Flushed {len(events)} audio events to artifact store")


agent = LlmAgent(
    name="voice_agent",
    model="gemini-2.5-flash",
    instruction="You are a helpful voice assistant.",
)

app = App(
    agent=agent,
    name="voice_app",
    artifact_service=InMemoryArtifactService(),
)

runner = Runner(app=app, plugins=[AudioPersistencePlugin()])
```

### Manually caching and flushing audio chunks

```python
from google.genai import types

# Simulate receiving audio from a live session
user_audio = types.Blob(data=b"\x00\x01\x02...", mime_type="audio/pcm")
model_audio = types.Blob(data=b"\x03\x04\x05...", mime_type="audio/pcm")

manager = AudioCacheManager()
manager.cache_audio(ctx, user_audio, cache_type="input")
manager.cache_audio(ctx, model_audio, cache_type="output")

# At end of session, flush to artifact store
events = await manager.flush_caches(ctx)
for ev in events:
    # ev.content.parts[0].file_data.file_uri  → artifact URI
    print(ev.content.parts[0].file_data.file_uri)
```

---

## 3 · `AudioTranscriber`

**Source:** `google.adk.flows.llm_flows.audio_transcriber`

`AudioTranscriber` wraps **Google Cloud Speech-to-Text** to transcribe the audio blobs collected on `InvocationContext.transcription_cache`. It bundles consecutive segments from the same speaker before calling the API to reduce latency and API call count.

> **Install note:** `pip install google-adk[speech]` (or `pip install google-cloud-speech`) is required.

### Constructor (source-verified)

```python
from google.adk.flows.llm_flows.audio_transcriber import AudioTranscriber

AudioTranscriber(init_client: bool = False)
# init_client=True eagerly creates a SpeechClient; False (default) is lazy.
```

### `transcribe_file` method (source-verified)

```python
contents: list[types.Content] = transcriber.transcribe_file(
    invocation_context: InvocationContext,
)
```

Internally the method:
1. Iterates `invocation_context.transcription_cache`
2. **Merges** consecutive `Blob` entries from the same role (`user` or `model`) into a single audio buffer — reducing API calls
3. Sends each merged buffer to the Cloud Speech-to-Text API
4. Returns a list of `types.Content` objects with `role` and transcribed `text`

### Usage: post-session transcription

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.apps import App
from google.adk.flows.llm_flows.audio_transcriber import AudioTranscriber
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.invocation_context import InvocationContext


class TranscriptionPlugin(BasePlugin):
    """Transcribe collected audio after each invocation."""

    def __init__(self):
        # init_client=True eagerly initialises the Speech client at startup
        self._transcriber = AudioTranscriber(init_client=True)

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        if not invocation_context.transcription_cache:
            return

        contents = self._transcriber.transcribe_file(invocation_context)
        for content in contents:
            role = content.role
            text = content.parts[0].text if content.parts else ""
            print(f"[{role}] {text}")

        # Clear cache so next turn starts fresh
        invocation_context.transcription_cache = []


agent = LlmAgent(
    name="voice_agent",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
)

runner = Runner(
    app=App(agent=agent, name="voice_app"),
    plugins=[TranscriptionPlugin()],
)
```

### When `transcription_cache` contains `Content` objects

The transcription cache can hold both raw `Blob` entries (audio bytes) and `Content` objects (already-transcribed text from the model). `AudioTranscriber` handles both:

```python
from google.genai import types

# Already-transcribed model turn stored directly as Content
model_content = types.Content(
    role="model",
    parts=[types.Part(text="Hello, how can I help you?")],
)
# Raw user audio blob
user_audio = types.Blob(data=b"...", mime_type="audio/pcm")

# The transcriber merges blobs for Speech API calls and passes
# Content objects through unchanged.
```

---

## 4 · `TranscriptionManager`

**Source:** `google.adk.flows.llm_flows.transcription_manager`

`TranscriptionManager` converts **`types.Transcription` events** emitted by Gemini Live API into `Event` objects and saves them to the session service. This creates a queryable text transcript alongside the raw audio artifacts.

### Constructor (source-verified)

```python
from google.adk.flows.llm_flows.transcription_manager import TranscriptionManager

manager = TranscriptionManager()  # no constructor args
```

### Key methods (source-verified)

```python
# Handle a transcription emitted by the user's microphone input.
await manager.handle_input_transcription(
    invocation_context: InvocationContext,
    transcription: types.Transcription,
)

# Handle a transcription emitted by the model's audio output.
await manager.handle_output_transcription(
    invocation_context: InvocationContext,
    transcription: types.Transcription,
)
```

Both methods create an `Event` with:
- `input_transcription` / `output_transcription` set to the `types.Transcription` object
- `author` set to `"user"` for input, or `invocation_context.agent.name` for output
- `invocation_id` from the current invocation
- `timestamp` from `platform_time.get_time()`

### Typical live-session pipeline

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner, RunConfig, StreamingMode
from google.adk.apps import App
from google.adk.sessions import InMemorySessionService
from google.adk.flows.llm_flows.transcription_manager import TranscriptionManager
from google.genai import types


async def run_voice_session():
    agent = LlmAgent(
        name="voice_bot",
        model="gemini-2.5-flash-live",
        instruction="You are a helpful voice assistant.",
    )

    session_service = InMemorySessionService()
    app = App(agent=agent, name="voice_app")
    runner = Runner(app=app)

    session = await session_service.create_session(
        app_name="voice_app", user_id="u1"
    )

    manager = TranscriptionManager()

    async for event in runner.run_live(
        user_id="u1",
        session_id=session.id,
        config=RunConfig(streaming_mode=StreamingMode.BIDI),
    ):
        # Real-time display of transcriptions as they arrive
        if event.input_transcription:
            # User spoke this
            print(f"User: {event.input_transcription.text}")
            # Persist to session store
            await manager.handle_input_transcription(
                runner._invocation_context, event.input_transcription
            )
        elif event.output_transcription:
            # Agent spoke this
            print(f"Agent: {event.output_transcription.text}")
            await manager.handle_output_transcription(
                runner._invocation_context, event.output_transcription
            )


asyncio.run(run_voice_session())
```

### Comparison: `TranscriptionManager` vs `AudioTranscriber`

| | `TranscriptionManager` | `AudioTranscriber` |
|---|---|---|
| Input | `types.Transcription` (text from Gemini Live) | `Blob` audio chunks in `transcription_cache` |
| Output | `Event` objects saved to session | `list[types.Content]` (in-memory) |
| Backend | None (text only) | Cloud Speech-to-Text |
| When used | Gemini Live API native transcription | Offline transcription of raw audio |

---

## 5 · `AgentEngineSandboxComputer`

**Source:** `google.adk.integrations.vmaas.sandbox_computer`  
**Extra install:** `pip install google-adk[vertexai]`

`AgentEngineSandboxComputer` is an `@experimental` `BaseComputer` implementation backed by **Vertex AI Agent Engine Computer Use Sandbox** — a cloud-managed remote browser environment. It lets agents take screenshots, click, type, and navigate a real browser without running Chrome locally.

### Constructor (source-verified)

```python
from google.adk.integrations.vmaas import AgentEngineSandboxComputer

AgentEngineSandboxComputer(
    *,
    project_id: str | None = None,
    location: str = "us-central1",
    service_account_email: str | None = None,
    sandbox_name: str | None = None,            # BYOS: existing sandbox resource name
    sandbox_template_name: str | None = None,   # template for faster creation
    sandbox_snapshot_name: str | None = None,   # snapshot for state restore
    sandbox_ttl_seconds: int = 3600,            # auto-created sandbox TTL
    search_engine_url: str = "https://www.google.com",
    vertexai_client: vertexai.Client | None = None,
)
```

### Two initialisation modes

| Mode | When to use |
|---|---|
| **Auto-provision** (default) | No `sandbox_name` — agent engine + sandbox created on first `prepare()` call; names stored in `session_state` for reuse across turns |
| **BYOS** (bring your own sandbox) | Pass `sandbox_name` — agent connects to your pre-existing sandbox; agent engine name extracted from the sandbox resource path |

### State sharing via `session_state`

The computer stores sandbox resource names and access tokens in `tool_context.state` so that:
- Multiple turns within a session reuse the same sandbox
- Multiple agent server instances can pick up an existing sandbox
- Tokens are refreshed automatically 60 seconds before expiry

State keys (source-verified):
```python
_STATE_KEY_AGENT_ENGINE_NAME = "_vmaas_agent_engine_name"
_STATE_KEY_SANDBOX_NAME      = "_vmaas_sandbox_name"
_STATE_KEY_ACCESS_TOKEN      = "_vmaas_access_token"
_STATE_KEY_TOKEN_EXPIRY      = "_vmaas_token_expiry"
```

### Auto-provision example

```python
from google.adk.agents import LlmAgent
from google.adk.tools.computer_use import ComputerUseToolset
from google.adk.integrations.vmaas import AgentEngineSandboxComputer

computer = AgentEngineSandboxComputer(
    project_id="my-project",
    location="us-central1",
    service_account_email="agent-sa@my-project.iam.gserviceaccount.com",
    sandbox_ttl_seconds=1800,          # 30-minute TTL
    search_engine_url="https://www.google.com",
)

toolset = ComputerUseToolset(computer=computer)

agent = LlmAgent(
    name="browser_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a browser automation agent. "
        "Use the computer tools to navigate websites and complete tasks."
    ),
    tools=[toolset],
)
```

### BYOS (bring your own sandbox)

```python
EXISTING_SANDBOX = (
    "projects/my-project/locations/us-central1/"
    "reasoningEngines/123456/sandboxEnvironments/789"
)

computer = AgentEngineSandboxComputer(
    project_id="my-project",
    sandbox_name=EXISTING_SANDBOX,
)

toolset = ComputerUseToolset(computer=computer)
```

### Snapshot-based sandbox (fast restore)

```python
SNAPSHOT = (
    "projects/my-project/locations/us-central1/"
    "reasoningEngines/123456/sandboxEnvironmentSnapshots/snap-001"
)

computer = AgentEngineSandboxComputer(
    project_id="my-project",
    service_account_email="agent-sa@my-project.iam.gserviceaccount.com",
    sandbox_snapshot_name=SNAPSHOT,  # Restore from snapshot for faster boot
)
```

### Required IAM roles

| Principal | Role | Purpose |
|---|---|---|
| `service_account_email` | `roles/iam.serviceAccountTokenCreator` | Generate access tokens |
| Agent SA | `roles/aiplatform.user` | Create/manage agent engines + sandboxes |
| Agent SA | `roles/aiplatform.admin` | Full sandbox lifecycle (if creating new engines) |

---

## 6 · `ParameterManagerClient`

**Source:** `google.adk.integrations.parameter_manager.parameter_client`  
**Extra install:** `pip install google-adk[parameter-manager]` (or `pip install google-cloud-parametermanager`)

`ParameterManagerClient` wraps **Google Cloud Parameter Manager**, letting agents retrieve structured configuration values (database connection strings, feature flags, deployment config) at runtime — without embedding them in code.

### Constructor (source-verified)

```python
from google.adk.integrations.parameter_manager import ParameterManagerClient

ParameterManagerClient(
    service_account_json: str | None = None,  # JSON keyfile content (not path)
    auth_token: str | None = None,            # pre-existing Bearer token
    location: str | None = None,              # regional endpoint; None = global
)
# Exactly one of service_account_json / auth_token must be provided,
# OR neither (uses Application Default Credentials).
```

### `get_parameter` (source-verified)

```python
value: str = client.get_parameter(
    resource_name: str,
    # Format: "projects/*/locations/*/parameters/*/versions/*"
    # Use "latest" for the current live version.
)
```

### Minimal usage

```python
from google.adk.integrations.parameter_manager import ParameterManagerClient

client = ParameterManagerClient()  # uses ADC

db_url = client.get_parameter(
    "projects/my-project/locations/global/parameters/db-url/versions/latest"
)
print(db_url)  # e.g. "postgresql://host:5432/mydb"
```

### Using in a `FunctionTool`

```python
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.integrations.parameter_manager import ParameterManagerClient


_param_client = ParameterManagerClient()

def get_api_endpoint(service: str) -> str:
    """Return the API endpoint URL for a named service."""
    resource = (
        f"projects/my-project/locations/global/"
        f"parameters/{service}-endpoint/versions/latest"
    )
    return _param_client.get_parameter(resource)


agent = LlmAgent(
    name="routing_agent",
    model="gemini-2.5-flash",
    instruction="Use get_api_endpoint to look up service endpoints before calling them.",
    tools=[FunctionTool(func=get_api_endpoint)],
)
```

### Regional Parameter Manager

```python
# Use a regional endpoint for compliance or latency reasons
regional_client = ParameterManagerClient(
    location="us-central1",
)

config = regional_client.get_parameter(
    "projects/my-project/locations/us-central1/parameters/feature-flags/versions/latest"
)
```

### With a service account JSON (non-ADC environments)

```python
import os
from google.adk.integrations.parameter_manager import ParameterManagerClient

SA_JSON = os.environ["SA_KEYFILE_JSON"]  # content, not file path

client = ParameterManagerClient(service_account_json=SA_JSON)

max_retries = int(
    client.get_parameter(
        "projects/my-project/locations/global/"
        "parameters/agent-max-retries/versions/latest"
    )
)
```

### `ParameterManagerClient` vs `SecretManagerClient`

| | `ParameterManagerClient` | `SecretManagerClient` |
|---|---|---|
| Stores | Structured config (templates, JSON, YAML) | Opaque secrets (API keys, passwords) |
| Versioning | Named versions + "latest" alias | Numbered versions + "latest" alias |
| Rendering | Template variable substitution | Byte-for-byte payload |
| IAM | `roles/parametermanager.parameterVersionAccessor` | `roles/secretmanager.secretAccessor` |
| Use for | Database URLs, feature flags, app config | Credentials, tokens, certificates |

---

## 7 · `SecretManagerClient`

**Source:** `google.adk.integrations.secret_manager.secret_client`  
**Extra install:** `pip install google-adk[secretmanager]` (or `pip install google-cloud-secret-manager`)

`SecretManagerClient` wraps **Google Cloud Secret Manager** for retrieving API keys, passwords, certificates, and other sensitive credentials at runtime.

### Constructor (source-verified)

```python
from google.adk.integrations.secret_manager import SecretManagerClient

SecretManagerClient(
    service_account_json: str | None = None,  # JSON keyfile content (not path)
    auth_token: str | None = None,            # pre-existing Bearer token
    location: str | None = None,              # regional endpoint; None = global
)
# Provide service_account_json OR auth_token, not both.
# Provide neither to use Application Default Credentials.
```

### `get_secret` (source-verified)

```python
value: str = client.get_secret(
    resource_name: str,
    # Format: "projects/*/secrets/*/versions/*"
    # Use "latest" for the current active version.
)
```

### Minimal usage

```python
from google.adk.integrations.secret_manager import SecretManagerClient

client = SecretManagerClient()  # uses ADC

api_key = client.get_secret(
    "projects/my-project/secrets/openai-api-key/versions/latest"
)
```

### Injecting secrets into tool execution

```python
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.integrations.secret_manager import SecretManagerClient
import httpx


_secrets = SecretManagerClient()

def call_payment_api(amount: float, currency: str) -> dict:
    """Call the payment API with the stored credentials."""
    api_key = _secrets.get_secret(
        "projects/my-project/secrets/payment-api-key/versions/latest"
    )
    response = httpx.post(
        "https://api.payments.example.com/charge",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"amount": amount, "currency": currency},
    )
    return response.json()


agent = LlmAgent(
    name="payment_agent",
    model="gemini-2.5-flash",
    instruction="Process payment requests using call_payment_api.",
    tools=[FunctionTool(func=call_payment_api)],
)
```

### Caching secrets to avoid per-call latency

```python
import functools
from google.adk.integrations.secret_manager import SecretManagerClient


class CachingSecretClient:
    """Wraps SecretManagerClient with a simple in-memory cache."""

    def __init__(self):
        self._client = SecretManagerClient()
        self._cache: dict[str, str] = {}

    def get_secret(self, resource_name: str) -> str:
        if resource_name not in self._cache:
            self._cache[resource_name] = self._client.get_secret(resource_name)
        return self._cache[resource_name]


_secrets = CachingSecretClient()
```

### Regional Secret Manager (compliance)

```python
# Some organisations mandate that secrets never leave a specific region.
regional_client = SecretManagerClient(
    location="europe-west4",
)

cert_pem = regional_client.get_secret(
    "projects/my-project/secrets/tls-cert/versions/latest"
)
```

---

## 8 · `GcpAuthProvider` + `GcpAuthProviderScheme`

**Source:**  
- `google.adk.integrations.agent_identity.gcp_auth_provider`  
- `google.adk.integrations.agent_identity.gcp_auth_provider_scheme`  
**Extra install:** `pip install google-adk[agent-identity]`

`GcpAuthProvider` implements `BaseAuthProvider` using the **IAM Connector Credentials Service** — a Google Cloud managed service that handles OAuth 2.0 flows (API key, 2-legged, and 3-legged) on behalf of the agent. It is the recommended way to give ADK agents access to Google Workspace, third-party SaaS, or other OAuth-protected APIs without managing tokens yourself.

### `GcpAuthProviderScheme` (source-verified)

```python
from google.adk.integrations.agent_identity import GcpAuthProviderScheme

GcpAuthProviderScheme(
    name: str,                          # resource name of the Auth Provider in GCP
    scopes: list[str] | None = None,    # OAuth2 scopes to request
    continue_uri: str | None = None,    # redirect URI for 3-legged OAuth
    # type_ is always "gcpAuthProviderScheme" — do not override
)
```

### `GcpAuthProvider` (source-verified)

```python
from google.adk.integrations.agent_identity import GcpAuthProvider

GcpAuthProvider(
    client: IAMConnectorCredentialsServiceClient | None = None,
    # None → lazy-initialises using IAM_CONNECTOR_CREDENTIALS_TARGET_HOST env var
)
```

`supported_auth_schemes` returns `(GcpAuthProviderScheme,)`.

### Auth flow summary

| Flow type | Behaviour |
|---|---|
| **API key** | Operation completes immediately; `operation.done == True` on first call |
| **2-legged OAuth** | Operation may be pending; provider polls for up to `NON_INTERACTIVE_TOKEN_POLL_TIMEOUT_SEC` (10 s) |
| **3-legged OAuth** | Operation returns `consent_pending` with an `authorization_uri`; ADK injects a HITL credential-request event so the user can complete the OAuth consent |

### Register and use with `AuthenticatedFunctionTool`

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_provider_registry import AuthProviderRegistry
from google.adk.integrations.agent_identity import (
    GcpAuthProvider,
    GcpAuthProviderScheme,
)
from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool
import httpx


# 1. Register the auth provider globally
registry = AuthProviderRegistry.get_instance()
registry.register_provider(GcpAuthProvider())


# 2. Define an authenticated tool
async def search_gmail(query: str) -> list[dict]:
    """Search the user's Gmail for matching messages."""
    # The framework injects a valid Bearer token via AuthConfig
    ...


gmail_tool = AuthenticatedFunctionTool(
    func=search_gmail,
    auth_config=AuthConfig(
        auth_scheme=GcpAuthProviderScheme(
            name="projects/my-project/locations/global/authProviders/gmail-oauth",
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
            continue_uri="https://my-app.example.com/oauth-callback",
        )
    ),
)

agent = LlmAgent(
    name="gmail_agent",
    model="gemini-2.5-flash",
    instruction="Search the user's email when asked.",
    tools=[gmail_tool],
)
```

### Custom IAM Connector host (non-prod)

```python
import os
from google.adk.integrations.agent_identity import GcpAuthProvider

# Override the service endpoint for testing
os.environ["IAM_CONNECTOR_CREDENTIALS_TARGET_HOST"] = "staging.iamconnector.example.com"

provider = GcpAuthProvider()  # picks up the custom host lazily
```

### Credential response structure

When the IAM Connector Credentials Service returns a token, it comes back as a `(header, token)` pair. `GcpAuthProvider` constructs an `AuthCredential` from it:

- If `header` is `Authorization: Bearer <token>` → `AuthCredentialTypes.HTTP` with `scheme="Bearer"`
- Otherwise → `HttpAuth.additional_headers` with the raw header name and value, plus `X-GOOG-API-KEY: <token>`

---

## 9 · `AgentRegistry` + `AgentRegistrySingleMcpToolset`

**Source:** `google.adk.integrations.agent_registry.agent_registry`  
**Extra install:** `pip install google-adk[a2a]`

`AgentRegistry` is a client for the **Google Cloud Agent Registry** service — a managed catalogue of A2A-protocol agents and MCP servers. Instead of hard-coding connection URLs, your orchestrator agent can discover and connect to registered agents/tools at runtime.

### Constructor (source-verified)

```python
from google.adk.integrations.agent_registry import AgentRegistry

AgentRegistry(
    project_id: str,                                    # required
    location: str,                                      # required
    header_provider: Callable[[ReadonlyContext], dict[str, str]] | None = None,
)
# Uses google.auth.default() for credentials.
```

### MCP server methods

```python
# List all registered MCP servers (paginated)
result = registry.list_mcp_servers(
    filter_str: str | None = None,     # e.g. "displayName='my-server'"
    page_size: int | None = None,
    page_token: str | None = None,
)

# Get one MCP server's metadata
server = registry.get_mcp_server(name: str)  # full resource name

# Get a ready-to-use McpToolset for a registered server
# Auth is resolved automatically from IAM bindings if not specified.
toolset = registry.get_mcp_toolset(
    mcp_server_name: str,
    auth_scheme: AuthScheme | None = None,
    auth_credential: AuthCredential | None = None,
    *,
    continue_uri: str | None = None,
)
```

### A2A agent methods

```python
# List all registered A2A agents
agents = registry.list_agents(
    filter_str: str | None = None,
    page_size: int | None = None,
    page_token: str | None = None,
)

# Get metadata for one agent
info = registry.get_agent_info(name: str)

# Get a ready-to-use RemoteA2aAgent for a registered agent
remote_agent = registry.get_remote_a2a_agent(
    agent_name: str,
    agent_skills: list[AgentSkill] | None = None,
    protocol_binding: A2ATransport | None = None,
)
```

### Endpoint methods

```python
# List registered inference endpoints
endpoints = registry.list_endpoints(...)

# Get one endpoint's metadata
endpoint = registry.get_endpoint(name: str)

# Get just the model name for an endpoint
model = registry.get_model_name(endpoint_name: str)
```

### Full example: orchestrator with dynamic tool discovery

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.integrations.agent_registry import AgentRegistry


async def build_orchestrator():
    registry = AgentRegistry(
        project_id="my-project",
        location="us-central1",
    )

    # Discover and wire up all MCP servers in the registry
    mcp_result = registry.list_mcp_servers(page_size=20)
    toolsets = []
    for server in mcp_result.get("mcpServers", []):
        toolset = registry.get_mcp_toolset(
            mcp_server_name=server["name"],
        )
        toolsets.append(toolset)

    # Discover registered A2A sub-agents
    agent_result = registry.list_agents(page_size=10)
    sub_agents = []
    for a in agent_result.get("agents", []):
        remote = registry.get_remote_a2a_agent(agent_name=a["name"])
        sub_agents.append(remote)

    orchestrator = LlmAgent(
        name="orchestrator",
        model="gemini-2.5-flash",
        instruction=(
            "You are an orchestrator. "
            "Delegate tasks to the appropriate sub-agent or tool."
        ),
        tools=toolsets,
        sub_agents=sub_agents,
    )

    runner = InMemoryRunner(agent=orchestrator, app_name="registry_demo")
    await runner.session_service.create_session(
        app_name="registry_demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Summarise today's calendar events", user_id="u1", session_id="s1"
    )
    print(events[-1].content.parts[0].text)


asyncio.run(build_orchestrator())
```

### Pinning a specific transport protocol

```python
from a2a.types import TransportProtocol as A2ATransport

# Force HTTP/JSON transport (avoid gRPC)
remote = registry.get_remote_a2a_agent(
    agent_name="projects/my-project/locations/us-central1/agents/my-agent",
    protocol_binding=A2ATransport.http_json,
)
```

### `AgentRegistrySingleMcpToolset`

`AgentRegistry.get_mcp_toolset()` returns an `AgentRegistrySingleMcpToolset` — a subclass of `McpToolset` that stamps every tool with a `gcp.mcp.server.destination.id` custom metadata key. This key is picked up by ADK's OpenTelemetry tracing (`google.adk.telemetry.tracing`) to annotate spans with the registered server ID, enabling end-to-end attribution in Cloud Trace.

```python
# You do not typically instantiate this directly;
# registry.get_mcp_toolset() does it for you.
from google.adk.integrations.agent_registry.agent_registry import (
    AgentRegistrySingleMcpToolset,
)
```

---

## 10 · `OAuth2CredentialRefresher` + `BaseCredentialRefresher` + `InMemoryCredentialService` + `BaseCredentialService`

**Source:**  
- `google.adk.auth.refresher.base_credential_refresher`  
- `google.adk.auth.refresher.oauth2_credential_refresher`  
- `google.adk.auth.credential_service.base_credential_service`  
- `google.adk.auth.credential_service.in_memory_credential_service`

These four `@experimental` classes form the **pluggable credential storage and refresh pipeline** for ADK's auth system. Where `AuthConfig`/`AuthHandler` handle the initial credential exchange, these classes handle the **persistence** (where to store exchanged tokens) and **refresh** (detecting and renewing expired tokens).

### `BaseCredentialService` ABC (source-verified)

```python
from google.adk.auth.credential_service.base_credential_service import (
    BaseCredentialService,
)

class BaseCredentialService(ABC):
    @abstractmethod
    async def load_credential(
        self,
        auth_config: AuthConfig,
        callback_context: CallbackContext,
    ) -> Optional[AuthCredential]: ...

    @abstractmethod
    async def save_credential(
        self,
        auth_config: AuthConfig,
        callback_context: CallbackContext,
    ) -> None: ...
```

The storage key is derived from `auth_config.get_credential_key()`, and the scope is `(app_name, user_id)` — so credentials are isolated per user.

### `InMemoryCredentialService` (source-verified)

```python
from google.adk.auth.credential_service.in_memory_credential_service import (
    InMemoryCredentialService,
)

service = InMemoryCredentialService()
# Internal structure: _credentials[app_name][user_id][credential_key] → AuthCredential
```

### `BaseCredentialRefresher` ABC (source-verified)

```python
from google.adk.auth.refresher.base_credential_refresher import (
    BaseCredentialRefresher,
    CredentialRefresherError,
)

class BaseCredentialRefresher(ABC):
    @abstractmethod
    async def is_refresh_needed(
        self,
        auth_credential: AuthCredential,
        auth_scheme: Optional[AuthScheme] = None,
    ) -> bool: ...

    @abstractmethod
    async def refresh(
        self,
        auth_credential: AuthCredential,
        auth_scheme: Optional[AuthScheme] = None,
    ) -> AuthCredential: ...
```

### `OAuth2CredentialRefresher` (source-verified)

```python
from google.adk.auth.refresher.oauth2_credential_refresher import (
    OAuth2CredentialRefresher,
)

refresher = OAuth2CredentialRefresher()
# Requires: pip install authlib (for OAuth2Token.is_expired())
# Also handles google.oauth2.credentials.Credentials (Google OAuth2 JSON format)
```

`is_refresh_needed` checks `auth_credential.oauth2.expires_at` and `expires_in` via `authlib.oauth2.rfc6749.OAuth2Token.is_expired()`.

### Custom credential service (Redis-backed example)

```python
from typing import Optional
from google.adk.auth.credential_service.base_credential_service import BaseCredentialService
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_tool import AuthConfig
from google.adk.agents.callback_context import CallbackContext
import json
import redis


class RedisCredentialService(BaseCredentialService):
    """Store OAuth tokens in Redis for multi-instance deployments."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self._r = redis.from_url(redis_url)

    def _key(self, auth_config: AuthConfig, ctx: CallbackContext) -> str:
        app = ctx._invocation_context.app_name
        user = ctx._invocation_context.user_id
        return f"adk:cred:{app}:{user}:{auth_config.credential_key}"

    async def load_credential(
        self,
        auth_config: AuthConfig,
        callback_context: CallbackContext,
    ) -> Optional[AuthCredential]:
        data = self._r.get(self._key(auth_config, callback_context))
        if data is None:
            return None
        return AuthCredential.model_validate_json(data)

    async def save_credential(
        self,
        auth_config: AuthConfig,
        callback_context: CallbackContext,
    ) -> None:
        key = self._key(auth_config, callback_context)
        cred = auth_config.exchanged_auth_credential
        if cred:
            self._r.setex(key, 3600, cred.model_dump_json())
```

### Custom credential refresher (custom token introspection)

```python
from typing import Optional
from google.adk.auth.refresher.base_credential_refresher import (
    BaseCredentialRefresher,
    CredentialRefresherError,
)
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes
from google.adk.auth.auth_schemes import AuthScheme
import httpx
import time


class IntrospectionRefresher(BaseCredentialRefresher):
    """Refresh by calling a token introspection endpoint."""

    def __init__(self, introspect_url: str, client_id: str, client_secret: str):
        self._url = introspect_url
        self._cid = client_id
        self._csec = client_secret

    async def is_refresh_needed(
        self,
        auth_credential: AuthCredential,
        auth_scheme: Optional[AuthScheme] = None,
    ) -> bool:
        if not auth_credential.http:
            return False
        token = auth_credential.http.credentials.token
        async with httpx.AsyncClient() as client:
            r = await client.post(
                self._url,
                data={"token": token},
                auth=(self._cid, self._csec),
            )
        info = r.json()
        # active=False or expires < now+60s → needs refresh
        if not info.get("active"):
            return True
        exp = info.get("exp", 0)
        return exp - time.time() < 60

    async def refresh(
        self,
        auth_credential: AuthCredential,
        auth_scheme: Optional[AuthScheme] = None,
    ) -> AuthCredential:
        # Implement your token refresh flow here
        raise CredentialRefresherError("Refresh not implemented")
```

### Wiring into `AuthConfig`

```python
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_schemes import OAuthGrantType, OAuth2Auth
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, OAuth2Credential
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from google.adk.auth.refresher.oauth2_credential_refresher import OAuth2CredentialRefresher

# Use InMemoryCredentialService (single-instance) and OAuth2CredentialRefresher
auth_config = AuthConfig(
    auth_scheme=OAuth2Auth(
        flows={
            "clientCredentials": {
                "tokenUrl": "https://auth.example.com/token",
                "scopes": {"read:data": "Read access"},
            }
        }
    ),
    raw_auth_credential=AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Credential(
            client_id="my-client-id",
            client_secret="my-client-secret",
        ),
    ),
    credential_service=InMemoryCredentialService(),
    credential_refresher=OAuth2CredentialRefresher(),
)
```

### `CredentialRefresherRegistry`

```python
from google.adk.auth.refresher.credential_refresher_registry import (
    CredentialRefresherRegistry,
)
from google.adk.auth.refresher.oauth2_credential_refresher import (
    OAuth2CredentialRefresher,
)

# Register a refresher for OAuth2 credentials application-wide
registry = CredentialRefresherRegistry.get_instance()
registry.register(OAuth2CredentialRefresher())

# The auth pipeline will automatically call the registered refresher
# before each tool invocation that requires fresh credentials.
```

### Architecture: where each class fits

```
AuthConfig.raw_auth_credential
       │
       ▼
AuthHandler.exchange()          ← one-time credential exchange (OAuth2 flow)
       │
       ▼
BaseCredentialService.save()    ← persist exchanged token (InMemory / Redis / ...)
       │
  [next turn]
       │
       ▼
BaseCredentialService.load()    ← retrieve persisted token
       │
       ▼
BaseCredentialRefresher.is_refresh_needed()  ← check expiry
       │  if expired
       ▼
BaseCredentialRefresher.refresh()            ← get fresh token
       │
       ▼
BaseTool.run_async()            ← tool executes with fresh credential
```

---

## Version matrix

| Class | Added | Stable in |
|---|---|---|
| `ApigeeLlm` | 2.0.0 | 2.0.0 |
| `AudioCacheManager` / `AudioCacheConfig` | 2.1.0 | 2.1.0 |
| `AudioTranscriber` | 2.1.0 | 2.1.0 |
| `TranscriptionManager` | 2.1.0 | 2.1.0 |
| `AgentEngineSandboxComputer` | 2.2.0 | @experimental |
| `ParameterManagerClient` | 2.1.0 | 2.1.0 |
| `SecretManagerClient` | 2.1.0 | 2.1.0 |
| `GcpAuthProvider` / `GcpAuthProviderScheme` | 2.2.0 | @experimental |
| `AgentRegistry` / `AgentRegistrySingleMcpToolset` | 2.2.0 | @experimental |
| `OAuth2CredentialRefresher` / `BaseCredentialRefresher` | 2.1.0 | @experimental |
| `InMemoryCredentialService` / `BaseCredentialService` | 2.1.0 | @experimental |

All examples verified against **google-adk==2.2.0** installed from PyPI (June 2026).

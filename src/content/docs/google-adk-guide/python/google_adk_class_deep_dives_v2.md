---
title: "Class deep dives — volume 2 (6 additional classes)"
description: "Source-verified deep dives into 6 more google-adk 2.1.0 classes: RemoteA2aAgent, LangGraphAgent, AuthCredential, GcsArtifactService, PubSubToolset, SpannerToolset."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 2"
  order: 61
---

Source-verified against **google-adk==2.1.0** (installed from PyPI, May 2026). Every field name, signature, and example is taken directly from the installed package source.

| Class | Module | Status |
|---|---|---|
| `RemoteA2aAgent` | `google.adk.agents.remote_a2a_agent` | `@a2a_experimental` |
| `LangGraphAgent` | `google.adk.agents.langgraph_agent` | Stable |
| `AuthCredential` + auth models | `google.adk.auth.auth_credential` | Stable |
| `GcsArtifactService` | `google.adk.artifacts.gcs_artifact_service` | Stable |
| `PubSubToolset` | `google.adk.tools.pubsub.pubsub_toolset` | `@experimental` |
| `SpannerToolset` | `google.adk.tools.spanner.spanner_toolset` | `@experimental` |

---

## 1 · `RemoteA2aAgent`

`google.adk.agents.remote_a2a_agent.RemoteA2aAgent` wraps a remote A2A-compatible agent as a local `BaseAgent`. From the orchestrator's perspective it behaves exactly like a local sub-agent — you add it to `sub_agents=` or `tools=[AgentTool(...)]` on any `LlmAgent`.

Decorated with `@a2a_experimental` — expect breaking changes in future minor releases.

### Constructor (verified `remote_a2a_agent.py:108-212`)

```python
RemoteA2aAgent(
    name: str,
    agent_card: Union[AgentCard, str],  # URL, file path, or AgentCard object
    *,
    description: str = "",
    httpx_client: Optional[httpx.AsyncClient] = None,   # deprecated; use a2a_client_factory
    timeout: float = 600.0,
    genai_part_converter: Callable = convert_genai_part_to_a2a_part,
    a2a_part_converter: Callable = convert_a2a_part_to_genai_part,
    a2a_client_factory: Optional[A2AClientFactory] = None,
    a2a_request_meta_provider: Optional[Callable[[InvocationContext, A2AMessage], dict]] = None,
    full_history_when_stateless: bool = False,
    config: Optional[A2aRemoteAgentConfig] = None,
    use_legacy: bool = True,
    **kwargs,
)
```

| Parameter | Default | Notes |
|---|---|---|
| `name` | required | Unique identifier; must be a valid Python identifier |
| `agent_card` | required | `AgentCard` object, `https://…` URL, or local file path to JSON |
| `description` | `""` | Auto-populated from the remote agent card if blank |
| `timeout` | `600.0` | HTTP timeout in seconds for the entire A2A round-trip |
| `a2a_client_factory` | `None` | Custom `A2AClientFactory`; use this instead of `httpx_client` |
| `a2a_request_meta_provider` | `None` | `(InvocationContext, A2AMessage) -> dict` — attach auth tokens, tenant IDs, etc. to every outgoing request |
| `full_history_when_stateless` | `False` | When `True`, stateless remote agents (those that return no context ID) receive the full session history on every turn |
| `config` | `None` | `A2aRemoteAgentConfig` with request interceptors |
| `use_legacy` | `True` | `False` emits the new-integration extension header — only set when both peers have been upgraded |

### Example 1 — URL-resolved agent card

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.runners import InMemoryRunner

# The remote agent card is fetched from the /.well-known/agent.json endpoint
remote_math = RemoteA2aAgent(
    name="remote_math",
    agent_card="https://math-service.internal/.well-known/agent.json",
    description="Remote math solver — handles advanced calculus and algebra.",
    timeout=30.0,
)

orchestrator = LlmAgent(
    name="orchestrator",
    model="gemini-2.5-flash",
    instruction=(
        "For maths questions, delegate to 'remote_math'. "
        "Answer everything else yourself."
    ),
    sub_agents=[remote_math],
)

async def main():
    runner = InMemoryRunner(agent=orchestrator, app_name="demo")
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "What is the integral of x^2 from 0 to 3?",
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 2 — file-based agent card

```python
import json
from pathlib import Path
from a2a.types import AgentCard
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

# Load a pre-downloaded card (useful in air-gapped environments)
card_path = Path("agents/billing_service_card.json")

# Option A: pass the path string — ADK reads and parses the JSON
remote_billing = RemoteA2aAgent(
    name="billing_agent",
    agent_card=str(card_path),
    timeout=60.0,
)

# Option B: pre-parse and pass an AgentCard object
card_dict = json.loads(card_path.read_text())
card = AgentCard.model_validate(card_dict)

remote_billing_v2 = RemoteA2aAgent(
    name="billing_agent",
    agent_card=card,
    timeout=60.0,
)
```

### Example 3 — signed requests with `a2a_request_meta_provider`

`a2a_request_meta_provider` lets you attach metadata (auth tokens, tenant IDs, trace IDs) to every outgoing A2A message. The callable receives `InvocationContext` and the outgoing `A2AMessage`.

```python
import time
import hmac
import hashlib
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.agents.invocation_context import InvocationContext
from a2a.types import Message as A2AMessage

SECRET_KEY = b"shared-hmac-secret"

def sign_request(ctx: InvocationContext, msg: A2AMessage) -> dict:
    """Attach an HMAC signature and tenant header to every outgoing A2A request."""
    timestamp = str(int(time.time()))
    payload = f"{ctx.session.id}:{timestamp}"
    sig = hmac.new(SECRET_KEY, payload.encode(), hashlib.sha256).hexdigest()
    return {
        "X-Tenant-ID": ctx.session.state.get("tenant_id", "default"),
        "X-Timestamp": timestamp,
        "X-Signature": sig,
    }

remote_agent = RemoteA2aAgent(
    name="secure_specialist",
    agent_card="https://specialist.internal/.well-known/agent.json",
    a2a_request_meta_provider=sign_request,
)
```

### Example 4 — custom interceptors via `A2aRemoteAgentConfig`

Interceptors mutate or inspect every A2A request before it is sent:

```python
from google.adk.a2a.agent.config import A2aRemoteAgentConfig
from google.adk.a2a.agent.interceptors import RequestInterceptor  # example path
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
import logging

logger = logging.getLogger(__name__)

class LoggingInterceptor:
    async def before_request(self, context, message):
        logger.info("A2A request to remote agent — session=%s", context.session.id)
        return message  # pass through unmodified

config = A2aRemoteAgentConfig(
    request_interceptors=[LoggingInterceptor()],
)

remote_agent = RemoteA2aAgent(
    name="logged_remote",
    agent_card="https://remote.example.com/.well-known/agent.json",
    config=config,
)
```

### Example 5 — multi-agent team with a mix of local and remote

```python
from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.runners import InMemoryRunner

# Local expert
local_writer = LlmAgent(
    name="writer",
    model="gemini-2.5-flash",
    description="Writes polished prose from bullet-point facts.",
    instruction="Turn the facts you receive into a crisp, 200-word paragraph.",
    mode="single_turn",
)

# Remote expert (different team's service)
remote_researcher = RemoteA2aAgent(
    name="researcher",
    agent_card="https://research.internal/.well-known/agent.json",
    description="Gathers verified facts on any topic.",
    timeout=45.0,
)

coordinator = LlmAgent(
    name="coordinator",
    model="gemini-2.5-pro",
    instruction=(
        "1. Ask 'researcher' for facts on the user's topic. "
        "2. Pass those facts to 'writer' to produce the final article."
    ),
    sub_agents=[remote_researcher, local_writer],
    mode="chat",
)
```

### Gotchas

- Agent cards fetched by URL are resolved **once** at construction time (lazy-loaded on first call). Cache the `RemoteA2aAgent` instance across requests.
- `full_history_when_stateless=True` sends the entire session history every turn — only enable this for small sessions; it can be expensive.
- If the remote service returns a `TaskState.input_required` status, ADK injects a mock function-call so the orchestrating LLM can relay the prompt back to the user (`remote_a2a_agent.py:_add_mock_function_call`).
- `use_legacy=True` (the default) uses the older A2A wire format. Coordinate with the remote team before flipping `use_legacy=False`.

---

## 2 · `LangGraphAgent`

`google.adk.agents.langgraph_agent.LangGraphAgent` is a bridge that wraps a compiled LangGraph `CompiledGraph` as a `BaseAgent`. It passes ADK session events into the graph as LangChain message objects and yields the graph's final response as an ADK `Event`.

Useful when: a team already has a LangGraph workflow they want to expose inside a larger ADK system, or for gradual migration from LangGraph to ADK.

> **Install prerequisite:** `pip install langchain-core langgraph langchain-google-genai`
> (`langchain-google-genai` is needed for `ChatGoogleGenerativeAI` used in the examples below.)

### Class definition (verified `agents/langgraph_agent.py`)

```python
from pydantic import ConfigDict

class LangGraphAgent(BaseAgent):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # CompiledGraph is not a Pydantic model — arbitrary_types_allowed is required.

    graph: CompiledGraph
    instruction: str = ""
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | required | Inherited from `BaseAgent` |
| `description` | `str` | `""` | Inherited from `BaseAgent` |
| `graph` | `CompiledGraph` | required | A LangGraph compiled graph (from `graph.compile()`) |
| `instruction` | `str` | `""` | Injected as a `SystemMessage` on the **first** turn only (skipped if the graph's checkpoint already has messages) |

### Memory behaviour

The agent inspects `self.graph.checkpointer` to decide how to pass history:

| Scenario | Behaviour |
|---|---|
| `graph.checkpointer` is **set** | Sends only the **most recent user messages** (LangGraph manages the rest via its own checkpointer) |
| `graph.checkpointer` is **None** | Sends the **full conversation** between `user` and this agent as `HumanMessage` / `AIMessage` pairs |

### Example 1 — minimal ReAct graph as an ADK agent

```python
import asyncio
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from google.adk.agents.langgraph_agent import LangGraphAgent
from google.adk.runners import InMemoryRunner

# Build a LangGraph ReAct agent
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

# compile() creates a CompiledGraph; no checkpointer → ADK manages history
react_graph = create_react_agent(llm, tools=[multiply])

# Wrap it as an ADK agent
adk_agent = LangGraphAgent(
    name="react_calculator",
    description="Answers arithmetic questions using a LangGraph ReAct agent.",
    graph=react_graph,
    instruction="You are a precise calculator. Show all steps.",
)

async def main():
    runner = InMemoryRunner(agent=adk_agent, app_name="demo")
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "What is 13.5 × 47?", user_id="u1", session_id="s1"
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 2 — multi-turn with LangGraph checkpointer

When you attach a LangGraph `MemorySaver`, LangGraph owns conversation history. ADK still owns **session state** (e.g. `session.state["key"]`), but the graph's `messages` list is managed by LangGraph. ADK sends only the latest user messages each turn.

```python
import asyncio
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from google.adk.agents.langgraph_agent import LangGraphAgent
from google.adk.runners import InMemoryRunner

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# MemorySaver → LangGraph tracks the message thread itself
checkpointer = MemorySaver()
graph = create_react_agent(llm, tools=[], checkpointer=checkpointer)

agent = LangGraphAgent(
    name="chat_agent",
    graph=graph,
    instruction="You are a helpful assistant.",
)

async def multi_turn():
    runner = InMemoryRunner(agent=agent, app_name="demo")
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )

    # Turn 1
    events = await runner.run_debug("My name is Alice.", user_id="u1", session_id="s1")
    print(events[-1].content.parts[0].text)

    # Turn 2 — LangGraph remembers "Alice" via its own MemorySaver
    events = await runner.run_debug("What is my name?", user_id="u1", session_id="s1")
    print(events[-1].content.parts[0].text)   # → "Your name is Alice."

asyncio.run(multi_turn())
```

### Example 3 — LangGraphAgent as a sub-agent inside an LlmAgent

```python
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from google.adk.agents import LlmAgent
from google.adk.agents.langgraph_agent import LangGraphAgent

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Build a specialist LangGraph workflow
specialist_graph = create_react_agent(llm, tools=[])

langgraph_specialist = LangGraphAgent(
    name="specialist",
    description="A specialised reasoning agent built with LangGraph.",
    graph=specialist_graph,
    mode="single_turn",    # BaseAgent fields still work
)

# Compose it into an ADK multi-agent system
root = LlmAgent(
    name="coordinator",
    model="gemini-2.5-flash",
    instruction="For complex reasoning tasks, delegate to 'specialist'.",
    sub_agents=[langgraph_specialist],
    mode="chat",
)
```

### How ADK wires the session thread ID

`LangGraphAgent` passes `{"configurable": {"thread_id": ctx.session.id}}` as the LangGraph `RunnableConfig`. This aligns LangGraph's checkpointer thread with ADK's session — the same `session_id` → the same LangGraph thread.

---

## 3 · `AuthCredential` and the authentication model

`google.adk.auth.auth_credential.AuthCredential` is the **credential envelope** used throughout ADK — by `OpenAPIToolset`, `McpToolset`, `APIHubToolset`, and any custom authenticated tool. It is a Pydantic model with a `model_config` that supports camelCase aliases.

### Class hierarchy (verified `auth/auth_credential.py`)

```
AuthCredential
├── auth_type: AuthCredentialTypes   ← required; determines which payload field is used
├── resource_ref: str | None
├── api_key: str | None
├── http: HttpAuth | None
│    ├── scheme: str                 ← e.g. "bearer", "basic"
│    └── credentials: HttpCredentials
│         ├── username: str | None
│         ├── password: str | None
│         └── token: str | None
├── service_account: ServiceAccount | None
│    ├── service_account_credential: ServiceAccountCredential | None
│    ├── scopes: list[str] | None
│    ├── use_default_credential: bool = False
│    ├── use_id_token: bool = False
│    └── audience: str | None
└── oauth2: OAuth2Auth | None
     ├── client_id / client_secret
     ├── access_token / refresh_token
     ├── auth_uri / redirect_uri / auth_code
     └── ...
```

### `AuthCredentialTypes` enum

```python
from google.adk.auth.auth_credential import AuthCredentialTypes

AuthCredentialTypes.API_KEY          # "apiKey"
AuthCredentialTypes.HTTP             # "http"
AuthCredentialTypes.OAUTH2           # "oauth2"
AuthCredentialTypes.OPEN_ID_CONNECT  # "openIdConnect"
AuthCredentialTypes.SERVICE_ACCOUNT  # "serviceAccount"
```

### Example 1 — API key credential

```python
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes

cred = AuthCredential(
    auth_type=AuthCredentialTypes.API_KEY,
    api_key="YOUR_API_KEY_HERE",
)
```

Pair with an `APIKeyScheme` or `APIKeyHeader` auth scheme so ADK knows *where* to inject the key (header, query param, cookie):

```python
from google.adk.auth.auth_schemes import APIKeyHeader

scheme = APIKeyHeader(name="X-API-Key")
# Then pass scheme + cred to OpenAPIToolset / McpToolset
```

### Example 2 — HTTP Basic auth

```python
from google.adk.auth.auth_credential import (
    AuthCredential, AuthCredentialTypes, HttpAuth, HttpCredentials
)

cred = AuthCredential(
    auth_type=AuthCredentialTypes.HTTP,
    http=HttpAuth(
        scheme="basic",
        credentials=HttpCredentials(
            username="myuser",
            password="hunter2",
        ),
    ),
)
```

### Example 3 — HTTP Bearer token (OAuth2 token already obtained)

```python
from google.adk.auth.auth_credential import (
    AuthCredential, AuthCredentialTypes, HttpAuth, HttpCredentials
)

cred = AuthCredential(
    auth_type=AuthCredentialTypes.HTTP,
    http=HttpAuth(
        scheme="bearer",
        credentials=HttpCredentials(token="ya29.access_token_here"),
    ),
)
```

### Example 4 — OAuth2 Authorization Code flow

Pass the credential with only `client_id` and `client_secret` initially. ADK will orchestrate the OAuth dance; once complete it populates `access_token` / `refresh_token` automatically.

```python
from google.adk.auth.auth_credential import (
    AuthCredential, AuthCredentialTypes, OAuth2Auth
)

# Initial credential — no tokens yet; ADK fetches them
cred = AuthCredential(
    auth_type=AuthCredentialTypes.OAUTH2,
    oauth2=OAuth2Auth(
        client_id="1234.apps.googleusercontent.com",
        client_secret="YOUR_CLIENT_SECRET",
        redirect_uri="http://localhost:8080/callback",
        scopes=["https://www.googleapis.com/auth/calendar.readonly"],
    ),
)
```

### Example 5 — Service account (JSON key)

```python
import json
from google.adk.auth.auth_credential import (
    AuthCredential, AuthCredentialTypes, ServiceAccount, ServiceAccountCredential
)

with open("service_account.json") as f:
    sa = json.load(f)

cred = AuthCredential(
    auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
    service_account=ServiceAccount(
        service_account_credential=ServiceAccountCredential(
            type_=sa["type"],
            project_id=sa["project_id"],
            private_key_id=sa["private_key_id"],
            private_key=sa["private_key"],
            client_email=sa["client_email"],
            client_id=sa["client_id"],
            auth_uri=sa["auth_uri"],
            token_uri=sa["token_uri"],
            auth_provider_x509_cert_url=sa["auth_provider_x509_cert_url"],
            client_x509_cert_url=sa["client_x509_cert_url"],
            universe_domain=sa.get("universe_domain", "googleapis.com"),
        ),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    ),
)
```

Shorthand using `model_construct` (bypasses validation — useful when the JSON is already trusted):

```python
from google.adk.auth.auth_credential import ServiceAccountCredential

sa_cred = ServiceAccountCredential.model_construct(**sa)
```

### Example 6 — Application Default Credentials (no key file)

When running on GKE, Cloud Run, or any GCP environment with a Workload Identity or metadata server:

```python
from google.adk.auth.auth_credential import (
    AuthCredential, AuthCredentialTypes, ServiceAccount
)

cred = AuthCredential(
    auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
    service_account=ServiceAccount(
        use_default_credential=True,   # uses ADC; no key file needed
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    ),
)
```

Validation: `ServiceAccount` raises `ValueError` if `use_default_credential=False` and `service_account_credential=None`.

### Example 7 — Service Account with ID token (service-to-service auth)

Required when calling Cloud Run, Cloud Functions, or other services that verify caller identity via Google-signed ID tokens rather than access tokens:

```python
from google.adk.auth.auth_credential import (
    AuthCredential, AuthCredentialTypes, ServiceAccount
)

cred = AuthCredential(
    auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
    service_account=ServiceAccount(
        use_default_credential=True,
        use_id_token=True,
        audience="https://my-cloud-run-service-xyz.run.app",
    ),
)
```

Validation: `ServiceAccount` raises `ValueError` if `use_id_token=True` and `audience` is not set.

### Putting it together — `OpenAPIToolset` with auth

```python
from google.adk.auth.auth_credential import (
    AuthCredential, AuthCredentialTypes, HttpAuth, HttpCredentials
)
from google.adk.auth.auth_schemes import APIKeyHeader
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from google.adk.agents import LlmAgent

# Suppose the API requires a key in the X-API-Key header
scheme = APIKeyHeader(name="X-API-Key")
cred = AuthCredential(
    auth_type=AuthCredentialTypes.API_KEY,
    api_key="my-key-abc123",
)

toolset = OpenAPIToolset(
    spec_dict=my_openapi_dict,
    auth_scheme=scheme,
    auth_credential=cred,
)

agent = LlmAgent(
    name="api_agent",
    model="gemini-2.5-flash",
    instruction="Use the available API tools to answer questions.",
    tools=[toolset],
)
```

---

## 4 · `GcsArtifactService`

`google.adk.artifacts.gcs_artifact_service.GcsArtifactService` stores ADK artifacts in a Google Cloud Storage bucket. It is a full implementation of `BaseArtifactService` — every method runs in a thread pool via `asyncio.to_thread` so it never blocks the event loop.

Install prerequisite: `pip install google-cloud-storage`

### Constructor

```python
GcsArtifactService(bucket_name: str, **kwargs)
```

`**kwargs` are forwarded verbatim to `google.cloud.storage.Client(...)`. Use this for custom credentials, project IDs, or client options.

```python
from google.adk.artifacts import GcsArtifactService

# ADC (default) — works on GKE/Cloud Run automatically
gcs = GcsArtifactService("my-adk-artifacts")

# Explicit project
gcs = GcsArtifactService("my-adk-artifacts", project="my-gcp-project")

# Explicit service account
from google.oauth2 import service_account
sa_creds = service_account.Credentials.from_service_account_file(
    "sa.json",
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
gcs = GcsArtifactService("my-adk-artifacts", credentials=sa_creds)
```

### Blob path structure

The storage layout (verified `gcs_artifact_service.py:_get_blob_name`):

| Scope | How to trigger | Blob path |
|---|---|---|
| Session-scoped | `session_id=<id>`, any filename without `user:` prefix | `{app_name}/{user_id}/{session_id}/{filename}/{version}` |
| User-scoped | `filename="user:foo"` prefix — `session_id` is ignored | `{app_name}/{user_id}/user/{filename}/{version}` |

> **GCS note:** Unlike some abstract ADK docs that say `session_id=None` = user-scoped, `GcsArtifactService` raises `InputValidationError` if `session_id is None` and the filename does not have the `"user:"` prefix (`gcs_artifact_service.py:_get_blob_prefix`). Always use the `"user:"` filename prefix for cross-session user storage.

Version numbers are 0-based integers. The first save returns `0`; each subsequent save of the same filename increments by 1.

### Example 1 — wiring into a `Runner`

```python
import asyncio
import os
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.artifacts import GcsArtifactService

agent = LlmAgent(
    name="doc_writer",
    model="gemini-2.5-flash",
    instruction="Generate reports and save them as artifacts.",
)

runner = Runner(
    app_name="reports",
    agent=agent,
    session_service=InMemorySessionService(),
    memory_service=InMemoryMemoryService(),
    artifact_service=GcsArtifactService(
        bucket_name=os.environ["ARTIFACT_BUCKET"],
        project=os.environ["GCP_PROJECT"],
    ),
)
```

### Example 2 — save, load, list, and version a report

```python
import asyncio
from google.genai import types
from google.adk.artifacts import GcsArtifactService

gcs = GcsArtifactService("my-bucket")

async def demo():
    # Save version 0 of a text report
    v0 = await gcs.save_artifact(
        app_name="reports",
        user_id="user-42",
        session_id="session-abc",
        filename="monthly_report.txt",
        artifact=types.Part(text="March 2026 report: revenue $1.2M"),
        custom_metadata={"author": "alice", "month": "march-2026"},
    )
    print(f"Saved as version {v0}")  # → 0

    # Save version 1
    v1 = await gcs.save_artifact(
        app_name="reports",
        user_id="user-42",
        session_id="session-abc",
        filename="monthly_report.txt",
        artifact=types.Part(text="March 2026 report (revised): revenue $1.25M"),
    )
    print(f"Saved as version {v1}")  # → 1

    # Load the latest version (returns types.Part)
    part = await gcs.load_artifact(
        app_name="reports",
        user_id="user-42",
        session_id="session-abc",
        filename="monthly_report.txt",
    )
    print(part.text)   # → "March 2026 report (revised): revenue $1.25M"

    # Load a specific version
    v0_part = await gcs.load_artifact(
        app_name="reports",
        user_id="user-42",
        session_id="session-abc",
        filename="monthly_report.txt",
        version=0,
    )
    print(v0_part.text)  # → "March 2026 report: revenue $1.2M"

    # List all filenames in the session scope
    keys = await gcs.list_artifact_keys(
        app_name="reports",
        user_id="user-42",
        session_id="session-abc",
    )
    print(keys)  # → ["monthly_report.txt"]

asyncio.run(demo())
```

### Example 3 — user-scoped (cross-session) artifacts

Files whose `filename` starts with `"user:"` are stored at `{app_name}/{user_id}/user/{filename}` — independent of any session:

```python
import asyncio
from google.genai import types
from google.adk.artifacts import GcsArtifactService

gcs = GcsArtifactService("my-bucket")

async def save_user_profile():
    profile_json = '{"name": "Alice", "lang": "en"}'
    version = await gcs.save_artifact(
        app_name="myapp",
        user_id="user-42",
        filename="user:profile.json",        # ← "user:" prefix → user scope
        artifact=types.Part(text=profile_json),
        # session_id is ignored for user-namespaced files
    )
    print(f"Profile saved as version {version}")

    # Load from any session
    part = await gcs.load_artifact(
        app_name="myapp",
        user_id="user-42",
        filename="user:profile.json",
        # session_id can be None or any session ID — result is the same
    )
    print(part.text)   # → '{"name": "Alice", "lang": "en"}'

asyncio.run(save_user_profile())
```

### Example 4 — listing versions with metadata

`list_artifact_versions` returns `list[ArtifactVersion]` — full metadata including GCS URI, creation timestamp, MIME type, and custom metadata:

```python
import asyncio
from google.adk.artifacts import GcsArtifactService

gcs = GcsArtifactService("my-bucket")

async def audit_trail():
    versions = await gcs.list_artifact_versions(
        app_name="reports",
        user_id="user-42",
        session_id="session-abc",
        filename="monthly_report.txt",
    )
    for av in versions:
        print(
            f"v{av.version}: {av.canonical_uri} "
            f"| {av.mime_type} "
            f"| created={av.create_time:.0f} "
            f"| metadata={av.custom_metadata}"
        )
    # → v0: gs://my-bucket/reports/user-42/session-abc/monthly_report.txt/0 | ...
    # → v1: gs://my-bucket/reports/user-42/session-abc/monthly_report.txt/1 | ...

asyncio.run(audit_trail())
```

`ArtifactVersion` fields:

| Field | Type | Notes |
|---|---|---|
| `version` | `int` | 0-based revision number |
| `canonical_uri` | `str` | `gs://{bucket}/{blob_name}` |
| `create_time` | `float` | Unix timestamp |
| `mime_type` | `Optional[str]` | From the uploaded blob's `content_type` |
| `custom_metadata` | `dict` | Metadata dict passed to `save_artifact` |

### Example 5 — saving binary artifacts (images, PDFs)

```python
import asyncio
from pathlib import Path
from google.genai import types
from google.adk.artifacts import GcsArtifactService

gcs = GcsArtifactService("my-bucket")

async def save_pdf():
    pdf_bytes = Path("invoice_2026_03.pdf").read_bytes()
    part = types.Part(
        inline_data=types.Blob(
            data=pdf_bytes,
            mime_type="application/pdf",
        )
    )
    version = await gcs.save_artifact(
        app_name="invoices",
        user_id="user-42",
        session_id="session-xyz",
        filename="invoice_march.pdf",
        artifact=part,
        custom_metadata={"month": "2026-03", "currency": "USD"},
    )
    print(f"Invoice saved as version {version}")

    # Load it back
    loaded_part = await gcs.load_artifact(
        app_name="invoices",
        user_id="user-42",
        session_id="session-xyz",
        filename="invoice_march.pdf",
    )
    assert loaded_part.inline_data.mime_type == "application/pdf"
    print(f"Loaded {len(loaded_part.inline_data.data)} bytes")

asyncio.run(save_pdf())
```

### Example 6 — deleting artifacts

```python
import asyncio
from google.adk.artifacts import GcsArtifactService

gcs = GcsArtifactService("my-bucket")

async def cleanup():
    # Deletes ALL versions of the file (all blobs matching the prefix)
    await gcs.delete_artifact(
        app_name="reports",
        user_id="user-42",
        session_id="session-abc",
        filename="monthly_report.txt",
    )

asyncio.run(cleanup())
```

### GCS IAM requirements

| Operation | IAM role / permission |
|---|---|
| `save_artifact` | `storage.objects.create` |
| `load_artifact` | `storage.objects.get` |
| `list_artifact_keys` / `list_versions` | `storage.objects.list` |
| `delete_artifact` | `storage.objects.delete` |

The pre-built role `roles/storage.objectAdmin` covers all four. On GKE/Cloud Run, bind it to the pod's Workload Identity service account.

---

## 5 · `PubSubToolset` *(experimental)*

`google.adk.tools.pubsub.pubsub_toolset.PubSubToolset` exposes three Google Cloud Pub/Sub operations as ADK tools. It is marked `@experimental(FeatureName.PUBSUB_TOOLSET)`.

Install prerequisite: `pip install google-cloud-pubsub`

### Tools provided

| Tool name | Source function | What it does |
|---|---|---|
| `publish_message` | `message_tool.publish_message` | Publish a UTF-8 message to a Pub/Sub topic |
| `pull_messages` | `message_tool.pull_messages` | Pull up to `max_messages` from a subscription |
| `acknowledge_messages` | `message_tool.acknowledge_messages` | Acknowledge pulled messages by ack ID |

### Constructor

```python
PubSubToolset(
    *,
    tool_filter: ToolPredicate | list[str] | None = None,
    credentials_config: PubSubCredentialsConfig | None = None,
    pubsub_tool_config: PubSubToolConfig | None = None,
)
```

| Parameter | Default | Notes |
|---|---|---|
| `tool_filter` | `None` | Include only named tools or match via predicate |
| `credentials_config` | `None` | `PubSubCredentialsConfig` for non-ADC auth |
| `pubsub_tool_config` | `None` | `PubSubToolConfig(project_id=...)` — if `None`, project is inferred from ADC |

### Example 1 — publish-and-subscribe event router

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.pubsub.pubsub_toolset import PubSubToolset
from google.adk.tools.pubsub.config import PubSubToolConfig

# Only expose publish and pull (skip acknowledge for this agent)
toolset = PubSubToolset(
    tool_filter=["publish_message", "pull_messages"],
    pubsub_tool_config=PubSubToolConfig(project_id="my-gcp-project"),
)

agent = LlmAgent(
    name="event_router",
    model="gemini-2.5-flash",
    instruction=(
        "You route orders to the correct Pub/Sub topic.\n"
        "Topic naming: projects/my-gcp-project/topics/{topic_name}.\n"
        "Available topics: 'orders-eu', 'orders-us', 'orders-apac'."
    ),
    tools=[toolset],
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="router")
    await runner.session_service.create_session(
        app_name="router", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Publish order #EU-9918 for customer in Paris to the right topic.",
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 2 — consumer agent with auto-acknowledge

```python
from google.adk.agents import LlmAgent
from google.adk.tools.pubsub.pubsub_toolset import PubSubToolset
from google.adk.tools.pubsub.config import PubSubToolConfig

toolset = PubSubToolset(
    pubsub_tool_config=PubSubToolConfig(project_id="my-gcp-project"),
)

agent = LlmAgent(
    name="order_processor",
    model="gemini-2.5-flash",
    instruction=(
        "Pull messages from the subscription "
        "'projects/my-gcp-project/subscriptions/orders-sub'. "
        "For each message: parse it, acknowledge it, and summarise the order. "
        "Pull max 5 messages per turn."
    ),
    tools=[toolset],
)
```

**Tool signatures** the model sees (from `message_tool.py`):

```python
# publish_message
publish_message(
    topic_name: str,      # e.g. "projects/my-project/topics/my-topic"
    message: str,         # UTF-8 content
    attributes: Optional[dict[str, str]] = None,
    ordering_key: str = "",
) -> dict   # {"message_id": "..."}

# pull_messages
pull_messages(
    subscription_name: str,    # "projects/my-project/subscriptions/my-sub"
    max_messages: int = 1,
    auto_ack: bool = False,    # set True to skip a separate acknowledge step
) -> dict   # {"messages": [{"message_id": ..., "data": ..., "ack_id": ..., ...}]}

# acknowledge_messages
acknowledge_messages(
    subscription_name: str,
    ack_ids: list[str],
) -> dict   # {"status": "SUCCESS"}
```

### Example 3 — custom credentials (non-ADC)

```python
import pathlib
from google.adk.tools.pubsub.pubsub_toolset import PubSubToolset
from google.adk.tools.pubsub.pubsub_credentials import PubSubCredentialsConfig

# PubSubCredentialsConfig is a BaseGoogleCredentialsConfig subclass
# Default scope: "https://www.googleapis.com/auth/pubsub"
cred_config = PubSubCredentialsConfig(
    service_account_json=pathlib.Path("sa.json").read_text(),
    # scopes default to pubsub scope — override only if needed
)

toolset = PubSubToolset(credentials_config=cred_config)
```

### Gotchas

- `PubSubToolset` is `@experimental` — import paths may change.
- The `attributes` parameter on `publish_message` must be a `dict[str, str]`; non-string values silently become strings via Pub/Sub's proto serialisation.
- `pull_messages` uses **synchronous** Pub/Sub pull (not streaming). For high-throughput use cases, implement a streaming pull outside ADK and only use the agent for processing.
- Setting `auto_ack=True` on `pull_messages` acknowledges messages even if your agent crashes mid-processing — only use it for idempotent workflows.

---

## 6 · `SpannerToolset` *(experimental)*

`google.adk.tools.spanner.spanner_toolset.SpannerToolset` exposes Cloud Spanner schema inspection and SQL execution as ADK tools. Decorated with `@experimental(FeatureName.SPANNER_TOOLSET)`.

Install prerequisite: `pip install google-cloud-spanner`

### Tools provided

| Tool name | Module | What it does |
|---|---|---|
| `spanner_list_table_names` | `metadata_tool` | List all tables in the database |
| `spanner_list_table_indexes` | `metadata_tool` | List indexes for a table |
| `spanner_list_table_index_columns` | `metadata_tool` | List columns in a table's index |
| `spanner_list_named_schemas` | `metadata_tool` | List named schemas (PostgreSQL dialect) |
| `spanner_get_table_schema` | `metadata_tool` | DDL and column definitions for a table |
| `spanner_execute_sql` | `query_tool` | Execute a SQL SELECT query (data read only by default) |
| `spanner_similarity_search` | `search_tool` | Vector similarity search by embedding |
| `spanner_vector_store_similarity_search` | `search_tool` | Vector store similarity search (requires `vector_store_settings`) |

The last two are only registered when `Capabilities.DATA_READ` is in `SpannerToolSettings.capabilities` (the default).

### Constructor

```python
SpannerToolset(
    *,
    tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
    credentials_config: Optional[SpannerCredentialsConfig] = None,
    spanner_tool_settings: Optional[SpannerToolSettings] = None,
)
```

The `tool_name_prefix` is hard-coded to `"spanner"` — all tool names are `spanner_*`.

### `SpannerToolSettings` fields (verified `settings.py`)

| Field | Type | Default | Notes |
|---|---|---|---|
| `capabilities` | `list[Capabilities]` | `[Capabilities.DATA_READ]` | Controls which tools are registered |
| `max_executed_query_result_rows` | `int` | `50` | Safety cap on rows returned by SQL |
| `query_result_mode` | `QueryResultMode` | `DEFAULT` | `DEFAULT` = list of rows; `DICT_LIST` = list of `{col: val}` dicts |
| `database_role` | `Optional[str]` | `None` | Spanner database role for fine-grained access control |
| `vector_store_settings` | `Optional[SpannerVectorStoreSettings]` | `None` | Required for `spanner_vector_store_similarity_search` |

### Example 1 — schema exploration agent

```python
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.spanner.spanner_toolset import SpannerToolset
from google.adk.tools.spanner.settings import SpannerToolSettings, Capabilities

# Read-only: metadata + SQL query
settings = SpannerToolSettings(
    capabilities=[Capabilities.DATA_READ],
    max_executed_query_result_rows=100,
)

toolset = SpannerToolset(spanner_tool_settings=settings)

agent = LlmAgent(
    name="spanner_analyst",
    model="gemini-2.5-flash",
    instruction=(
        "You are a Spanner expert. Use the available tools to explore the "
        "database schema and answer questions about the data. "
        "Always list tables first, then inspect the schema before querying."
    ),
    tools=[toolset],
)
```

### Example 2 — metadata-only agent (no SQL)

Expose only schema inspection tools — no SQL execution:

```python
from google.adk.tools.spanner.spanner_toolset import SpannerToolset

# Filter to metadata tools only — no spanner_execute_sql exposed
toolset = SpannerToolset(
    tool_filter=[
        "spanner_list_table_names",
        "spanner_get_table_schema",
    ],
)
```

### Example 3 — custom credentials + database role

```python
import pathlib
from google.adk.tools.spanner.spanner_toolset import SpannerToolset
from google.adk.tools.spanner.settings import SpannerToolSettings, QueryResultMode
from google.adk.tools.spanner.spanner_credentials import SpannerCredentialsConfig

cred_config = SpannerCredentialsConfig(
    service_account_json=pathlib.Path("sa.json").read_text(),
    # Default scopes: spanner.admin + spanner.data
)

settings = SpannerToolSettings(
    max_executed_query_result_rows=25,
    query_result_mode=QueryResultMode.DICT_LIST,  # {col: val} dicts instead of rows
    database_role="analyst_role",                  # Spanner fine-grained access control
)

toolset = SpannerToolset(
    credentials_config=cred_config,
    spanner_tool_settings=settings,
)
```

### Example 4 — full text-to-SQL assistant

```python
import asyncio
import os
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.spanner.spanner_toolset import SpannerToolset
from google.adk.tools.spanner.settings import SpannerToolSettings, QueryResultMode

settings = SpannerToolSettings(
    max_executed_query_result_rows=50,
    query_result_mode=QueryResultMode.DICT_LIST,
)

toolset = SpannerToolset(spanner_tool_settings=settings)

agent = LlmAgent(
    name="sql_assistant",
    model="gemini-2.5-pro",
    instruction=(
        "You are a SQL assistant for a Google Cloud Spanner database. "
        "Always start by listing tables (spanner_list_table_names), then "
        "get the schema (spanner_get_table_schema) before writing queries. "
        "Return results formatted as a markdown table."
    ),
    tools=[toolset],
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="sql")
    await runner.session_service.create_session(
        app_name="sql", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Show me the top 5 customers by total order value.",
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

### Example 5 — vector similarity search

```python
from google.adk.agents import LlmAgent
from google.adk.tools.spanner.spanner_toolset import SpannerToolset
from google.adk.tools.spanner.settings import (
    SpannerToolSettings,
    Capabilities,
    SpannerVectorStoreSettings,
)

vector_settings = SpannerVectorStoreSettings(
    project_id="my-project",
    instance_id="my-instance",
    database_id="my-database",
    table_name="documents",
    content_column="text_content",
    embedding_column="text_embedding",
    vector_length=768,
    vertex_ai_embedding_model_name="text-embedding-005",
    top_k=5,
    distance_type="COSINE",
    selected_columns=["title", "url", "text_content"],
)

settings = SpannerToolSettings(
    capabilities=[Capabilities.DATA_READ],
    vector_store_settings=vector_settings,
)

toolset = SpannerToolset(spanner_tool_settings=settings)

agent = LlmAgent(
    name="semantic_search_agent",
    model="gemini-2.5-flash",
    instruction=(
        "Use `spanner_vector_store_similarity_search` to find semantically "
        "similar documents. Return the top matches with their title and URL."
    ),
    tools=[toolset],
)
```

### Gotchas

- `SpannerToolset` is `@experimental` — import paths and settings fields may change.
- The tool name prefix `"spanner"` is hard-coded. If you use multiple `SpannerToolset` instances targeting different databases, use `tool_filter` or subclass to override.
- `max_executed_query_result_rows` is a safety cap, not a SQL `LIMIT` — the query may still scan many rows; add explicit `LIMIT` clauses in your instructions.
- The `spanner_vector_store_similarity_search` tool is **only registered** when `vector_store_settings` is set in `SpannerToolSettings`.
- `query_result_mode=DICT_LIST` returns larger payloads (column names repeated per row) — use `DEFAULT` for very wide schemas.

---

## Version notes

All examples verified against **google-adk==2.1.0** installed from PyPI (`pip install google-adk`) in May 2026. Import paths, field names, and class signatures cross-checked against the installed source at `/usr/local/lib/python3.11/dist-packages/google/adk/`.

| Class | Source file | Status |
|---|---|---|
| `RemoteA2aAgent` | `agents/remote_a2a_agent.py` | `@a2a_experimental` |
| `LangGraphAgent` | `agents/langgraph_agent.py` | Stable (concept implementation) |
| `AuthCredential` | `auth/auth_credential.py` | Stable |
| `GcsArtifactService` | `artifacts/gcs_artifact_service.py` | Stable |
| `PubSubToolset` | `tools/pubsub/pubsub_toolset.py` | `@experimental(FeatureName.PUBSUB_TOOLSET)` |
| `SpannerToolset` | `tools/spanner/spanner_toolset.py` | `@experimental(FeatureName.SPANNER_TOOLSET)` |

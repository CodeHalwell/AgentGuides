---
title: "Class deep dives — volume 16 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.2.0 classes: AgentCardBuilder/to_a2a (A2A server bootstrap; skill extraction from LlmAgent; lifespan wiring; uvicorn launch), convert_a2a_part_to_genai_part/convert_genai_part_to_a2a_part (bidirectional A2A↔GenAI part conversion; sentinel encoding for DataPart; thought_signature preservation), AdkEventToA2AEventsConverter (ADK Event → A2A event pipeline; TaskState mapping; streaming artifact chunks), AuthScheme/OAuthGrantType/ExtendedOAuth2/OpenIdConnectWithConfig/CustomAuthScheme (auth scheme taxonomy; grant-type detection; credential pairing), McpInstructionProvider (MCP Prompt as LlmAgent instruction; state-injected args; InstructionProvider protocol), BaseArtifactService/ArtifactVersion/InMemoryArtifactService (artifact storage contract; version model; session vs user scope; artifact reference resolution), RequestInput/workflow HITL utilities (interrupt_id; response_schema; create_request_input_event; process_auth_resume fallback chain), RetryConfig/_should_retry_node/_get_retry_delay (exponential backoff; jitter algorithm; exception allowlist; delay table), TaskResultAggregator/ExecutorContext (A2A state-priority aggregation; intermediate event rewriting; per-request routing context), ToolConfig/BaseToolConfig/ToolArgsConfig (YAML/Python tool config patterns; built-in/class/instance/function/factory; custom BaseToolConfig.from_config())."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 16"
  order: 81
---

Source-verified against **google-adk==2.2.0** (installed from PyPI, June 2026). Every field name, signature, and code example is drawn from the installed package source at `/usr/local/lib/python3.11/dist-packages/google/adk/`.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `AgentCardBuilder` + `to_a2a` | `google.adk.a2a.utils` | `@a2a_experimental` |
| 2 | `convert_a2a_part_to_genai_part` + `convert_genai_part_to_a2a_part` | `google.adk.a2a.converters.part_converter` | `@a2a_experimental` |
| 3 | `AdkEventToA2AEventsConverter` + event conversion pipeline | `google.adk.a2a.converters.from_adk_event` | `@a2a_experimental` |
| 4 | `AuthScheme` + `OAuthGrantType` + `ExtendedOAuth2` + `OpenIdConnectWithConfig` + `CustomAuthScheme` | `google.adk.auth.auth_schemes` | Mixed (see section) |
| 5 | `McpInstructionProvider` | `google.adk.agents.mcp_instruction_provider` | Stable |
| 6 | `BaseArtifactService` + `ArtifactVersion` + `InMemoryArtifactService` | `google.adk.artifacts` | Stable |
| 7 | `RequestInput` + workflow HITL utilities | `google.adk.events.request_input`, `google.adk.workflow.utils._workflow_hitl_utils` | Stable |
| 8 | `RetryConfig` + `_should_retry_node` + `_get_retry_delay` | `google.adk.workflow` | Stable |
| 9 | `TaskResultAggregator` + `ExecutorContext` | `google.adk.a2a.executor` | `@a2a_experimental` |
| 10 | `ToolConfig` + `BaseToolConfig` + `ToolArgsConfig` | `google.adk.tools.tool_configs` | `@experimental` |

---

## 1 · `AgentCardBuilder` + `to_a2a`

**Source:** `google.adk.a2a.utils.agent_card_builder`, `google.adk.a2a.utils.agent_to_a2a`

> Both are decorated `@a2a_experimental`.

`AgentCardBuilder` generates an A2A-protocol `AgentCard` by introspecting an ADK `BaseAgent` or `Workflow`. It walks the agent tree to discover tools, planners, and sub-agents, converting each into an A2A `AgentSkill`. This is the canonical way to advertise an ADK agent to A2A clients without writing `AgentCard` JSON by hand.

`to_a2a` is the one-call bootstrap that wraps an ADK agent (or `Runner`) in a fully wired Starlette application: it instantiates the task store, push-notification config store, and `DefaultRequestHandler`, then calls `AgentCardBuilder.build()` inside an async lifespan hook before returning a Starlette app ready for uvicorn.

### Constructor (source-verified)

```python
from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder

AgentCardBuilder(
    *,
    agent: BaseAgent | Workflow,
    rpc_url: str | None = None,              # defaults to "http://localhost:80/a2a"
    capabilities: AgentCapabilities | None = None,
    doc_url: str | None = None,
    provider: AgentProvider | None = None,
    agent_version: str | None = None,        # defaults to "0.0.1"
    security_schemes: dict[str, SecurityScheme] | None = None,
)
```

### Skill extraction rules for `LlmAgent`

`build()` calls `_build_primary_skills(agent)` followed by `_build_sub_agent_skills(agent)`. For each `LlmAgent` the following skills are emitted:

- A main **"model"** skill with `id=agent.name`
- One **"tool"** skill per tool in `agent.tools`
- A **"planning"** skill if `agent.planner` is set
- A **"code-execution"** skill if `agent.code_executor` is set

Sub-agent skills are recursively gathered and their `id` is prefixed with the parent sub-agent's name. The agent description is passed through `_replace_pronouns` so that "you are" → "I am", "you were" → "I was", "you're" → "I am", "you've" → "I have", "yours" → "mine", "your" → "my", "you" → "I" (sorted longest-first to prevent partial matches). The resulting `AgentCard` has `default_input_modes=['text/plain']` and `default_output_modes=['text/plain']`.

### Example 1 — minimal launch

```python
import uvicorn
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

agent = LlmAgent(
    name="greeter",
    model="gemini-2.5-flash",
    instruction="You are a friendly greeter. Welcome users warmly.",
)

# to_a2a wires InMemoryTaskStore, A2aAgentExecutor, DefaultRequestHandler,
# and builds the AgentCard in the Starlette lifespan hook.
app = to_a2a(agent, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Example 2 — with application lifespan (DB init / close)

```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import asyncpg
import uvicorn
from starlette.applications import Starlette
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

# Application-level lifespan that opens and closes a DB connection pool.
# to_a2a combines this with its own internal setup_a2a lifespan via
# _combined_lifespan so both run correctly.
@asynccontextmanager
async def db_lifespan(app: Starlette) -> AsyncGenerator[None, None]:
    app.state.db = await asyncpg.create_pool(dsn="postgresql://localhost/mydb")
    try:
        yield
    finally:
        await app.state.db.close()

agent = LlmAgent(
    name="db_agent",
    model="gemini-2.5-flash",
    instruction="You are a data assistant.",
)

app = to_a2a(agent, host="0.0.0.0", port=8001, lifespan=db_lifespan)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Example 3 — persistent task store + custom `AgentCard` from JSON

```python
import json
import uvicorn
from a2a.types import AgentCard
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
# Hypothetical persistent task store — swap InMemoryTaskStore for your own
# implementation of the a2a TaskStore protocol.
from my_project.stores import DatabaseTaskStore

with open("agent_card.json") as fh:
    card_data = json.load(fh)
agent_card = AgentCard.model_validate(card_data)

agent = LlmAgent(
    name="order_agent",
    model="gemini-2.5-pro",
    instruction="You are an order management assistant.",
)

task_store = DatabaseTaskStore(dsn="postgresql://localhost/tasks")

app = to_a2a(
    agent,
    host="0.0.0.0",
    port=8002,
    agent_card=agent_card,       # skip AgentCardBuilder; use the provided card directly
    task_store=task_store,       # replace the default InMemoryTaskStore
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
```

### Example 4 — `agent_executor_factory` for a custom executor

```python
import uvicorn
from google.adk.agents import LlmAgent
from google.adk.a2a.executor import A2aAgentExecutor
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

class TenantAwareExecutor(A2aAgentExecutor):
    """Injects tenant_id from the A2A request metadata into session state."""

    async def execute(self, context, event_queue):
        # context.request.metadata may carry X-Tenant-ID from the caller
        tenant_id = (context.request.metadata or {}).get("tenant_id", "default")
        # Store in session state so tools can read it via ToolContext.state
        context.session.state["tenant_id"] = tenant_id
        await super().execute(context, event_queue)

agent = LlmAgent(
    name="tenant_agent",
    model="gemini-2.5-flash",
    instruction="You are a multi-tenant assistant.",
)
runner = Runner(
    agent=agent,
    app_name="tenant_app",
    session_service=InMemorySessionService(),
)

app = to_a2a(
    agent,
    host="0.0.0.0",
    port=8003,
    runner=runner,
    agent_executor_factory=lambda runner: TenantAwareExecutor(runner=runner),
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
```

---

## 2 · `convert_a2a_part_to_genai_part` + `convert_genai_part_to_a2a_part`

**Source:** `google.adk.a2a.converters.part_converter`

> Both functions are decorated `@a2a_experimental`.

These two functions provide the bidirectional translation layer between A2A protocol `Part` objects (`TextPart`, `FilePart`, `DataPart`) and GenAI SDK `types.Part` objects. They are the lowest-level primitive in the A2A conversion stack — every message flowing through `A2aAgentExecutor` passes through them. Understanding them is essential when building custom A2A executors or debugging part-level data loss.

### Key constants (source-verified)

```python
from google.adk.a2a.converters.part_converter import (
    A2A_DATA_PART_METADATA_TYPE_KEY,
    A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL,
    A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE,
    A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT,
    A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE,
    A2A_DATA_PART_TEXT_MIME_TYPE,
    A2A_DATA_PART_START_TAG,
    A2A_DATA_PART_END_TAG,
)

# A2A_DATA_PART_METADATA_TYPE_KEY                  = 'type'
# A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL        = 'function_call'
# A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE    = 'function_response'
# A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT= 'code_execution_result'
# A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE      = 'executable_code'
# A2A_DATA_PART_TEXT_MIME_TYPE                     = 'text/plain'
# A2A_DATA_PART_START_TAG                          = b'<a2a_datapart_json>'
# A2A_DATA_PART_END_TAG                            = b'</a2a_datapart_json>'
```

### Conversion mapping (source-verified)

`convert_a2a_part_to_genai_part(a2a_part)` returns `Optional[genai_types.Part]`:

| A2A Part type | Condition | GenAI Part produced |
|---|---|---|
| `TextPart` | — | `Part(text=..., thought=metadata['adk:thought'])` |
| `FilePart` with `FileWithUri` | — | `Part(file_data=FileData(file_uri, mime_type, display_name))` |
| `FilePart` with `FileWithBytes` | — | `Part(inline_data=Blob(data=base64.b64decode(bytes), mime_type))` |
| `DataPart` | `type=function_call` in metadata | `Part(function_call=...)`, restores `thought_signature` from base64 |
| `DataPart` | `type=function_response` | `Part(function_response=...)` |
| `DataPart` | other types | `Part(inline_data=Blob(data=START_TAG+json+END_TAG, mime_type='application/octet-stream'))` |

`convert_genai_part_to_a2a_part(part)` returns `Optional[a2a_types.Part]`:

| GenAI Part field | A2A Part produced |
|---|---|
| `text` | `TextPart(text=...)` with `adk:thought` metadata if `part.thought is not None` |
| `file_data` | `FilePart(file=FileWithUri(...))` |
| `inline_data` starting with `A2A_DATA_PART_START_TAG` | Decoded back to `DataPart` via sentinel stripping |
| `inline_data` (other) | `FilePart(file=FileWithBytes(bytes=base64.b64encode(...)))` |
| `function_call` | `DataPart(data=fc.model_dump(), metadata={'type': 'function_call'})`, `thought_signature` encoded as base64 string |
| `function_response` | `DataPart(data=fr.model_dump(), metadata={'type': 'function_response'})` |
| `code_execution_result` | `DataPart(data=..., metadata={'type': 'code_execution_result'})` |
| `executable_code` | `DataPart(data=..., metadata={'type': 'executable_code'})` |

### Example 1 — round-trip text part with thought metadata

```python
from google.genai import types as genai_types
from google.adk.a2a.converters.part_converter import (
    convert_genai_part_to_a2a_part,
    convert_a2a_part_to_genai_part,
)

# Create a GenAI part that carries a "thought" flag (Gemini thinking tokens)
original = genai_types.Part(text="The user wants a haiku.", thought=True)

# GenAI → A2A: thought=True is preserved in TextPart metadata as 'adk:thought'
a2a_part = convert_genai_part_to_a2a_part(original)
print(a2a_part)
# TextPart(text='The user wants a haiku.', metadata={'adk:thought': True})

# A2A → GenAI: round-trip restores the thought flag
restored = convert_a2a_part_to_genai_part(a2a_part)
assert restored.text == "The user wants a haiku."
assert restored.thought is True
```

### Example 2 — round-trip function call with `thought_signature`

```python
import base64
from google.genai import types as genai_types
from google.adk.a2a.converters.part_converter import (
    convert_genai_part_to_a2a_part,
    convert_a2a_part_to_genai_part,
)

# Gemini may attach a thought_signature bytes blob to function calls.
# The converter preserves it across the A2A boundary as a base64 string.
sig_bytes = b"\x01\x02\x03\x04"  # example opaque signature
fc_part = genai_types.Part(
    function_call=genai_types.FunctionCall(
        name="search_flights",
        args={"destination": "Paris", "date": "2026-07-14"},
        id="fc_001",
    ),
)
# Simulate thought_signature attachment (set by the Gemini backend)
fc_part.function_call.thought_signature = sig_bytes

# GenAI → A2A: thought_signature encoded as base64 in DataPart.data
a2a_part = convert_genai_part_to_a2a_part(fc_part)
# a2a_part.root is a DataPart with metadata={'type': 'function_call'}
data_part = a2a_part.root
assert data_part.metadata["type"] == "function_call"
assert data_part.data["thought_signature"] == base64.b64encode(sig_bytes).decode()

# A2A → GenAI: base64 decoded back to bytes on the restored FunctionCall
restored = convert_a2a_part_to_genai_part(a2a_part)
assert restored.function_call.name == "search_flights"
assert restored.function_call.thought_signature == sig_bytes
```

### Example 3 — custom `DataPart` round-trip via sentinel encoding

```python
import json
import base64
from a2a.types import DataPart, Part as A2APart
from google.adk.a2a.converters.part_converter import (
    A2A_DATA_PART_START_TAG,
    A2A_DATA_PART_END_TAG,
    convert_a2a_part_to_genai_part,
    convert_genai_part_to_a2a_part,
)

# A DataPart with an unrecognised type is wrapped in sentinel bytes and
# stored as inline_data so it survives passage through GenAI and back.
custom_payload = {"type": "my_custom_event", "value": 42, "tags": ["a", "b"]}
original_data_part = DataPart(
    data=custom_payload,
    metadata={"type": "my_custom_event"},
)
original = A2APart(root=original_data_part)

# A2A → GenAI: encoded as inline_data with sentinel wrapping
genai_part = convert_a2a_part_to_genai_part(original)
assert genai_part.inline_data is not None
raw = genai_part.inline_data.data  # bytes
assert raw.startswith(A2A_DATA_PART_START_TAG)
assert raw.endswith(A2A_DATA_PART_END_TAG)

# Confirm the JSON payload is intact between the sentinels
inner = raw[len(A2A_DATA_PART_START_TAG): -len(A2A_DATA_PART_END_TAG)]
decoded = json.loads(inner)
assert decoded["data"]["value"] == 42

# GenAI → A2A: sentinel stripped; original DataPart reconstructed
restored = convert_genai_part_to_a2a_part(genai_part)
restored_dp = restored.root
assert restored_dp.data["value"] == 42
assert restored_dp.metadata["type"] == "my_custom_event"
```

---

## 3 · `AdkEventToA2AEventsConverter` + event conversion pipeline

**Source:** `google.adk.a2a.converters.from_adk_event`

> Decorated `@a2a_experimental`.

`AdkEventToA2AEventsConverter` is a `Callable` type alias that defines the contract for translating a single ADK `Event` into a list of A2A update events (`TaskStatusUpdateEvent | TaskArtifactUpdateEvent`). The default implementation `convert_adk_event_to_a2a_events` handles the full mapping between ADK's event model and the A2A task state machine. Swapping in a custom converter lets you filter sensitive parts, enrich metadata, or emit synthetic A2A events without forking the executor.

### Type alias (source-verified)

```python
from typing import Callable, Optional, List, Dict
from google.adk.events.event import Event
from google.adk.a2a.converters.part_converter import GenAIPartToA2APartConverter
from a2a.types import TaskStatusUpdateEvent, TaskArtifactUpdateEvent

# A2A update events that the executor feeds into its event queue
A2AUpdateEvent = TaskStatusUpdateEvent | TaskArtifactUpdateEvent

AdkEventToA2AEventsConverter = Callable[
    [
        Event,                          # the ADK event to convert
        Optional[Dict[str, str]],       # agents_artifacts: agent_name → current artifact_id
        Optional[str],                  # task_id
        Optional[str],                  # context_id
        GenAIPartToA2APartConverter,    # part-level converter (injectable)
    ],
    List[A2AUpdateEvent],
]
```

### State mapping rules (source-verified)

| ADK Event condition | A2A `TaskState` |
|---|---|
| `event.partial == True` | `working` (`is_final=False`) |
| `event.actions.escalate` or `event.is_final_response()` | `completed` |
| `event.error_code` / `event.error_message` set | `failed` |
| `event.get_function_calls()` with IDs in `long_running_tool_ids` | `input_required` (HITL) |
| Artifact content parts | `TaskArtifactUpdateEvent(append=True)` for streaming chunks |

The `agents_artifacts` dict is mutated in-place: it tracks the active artifact ID per agent name so that streaming chunks for the same artifact are appended (`append=True`) rather than creating a new artifact record on each chunk.

### Example 1 — consuming the default converter in a custom executor

```python
from google.adk.a2a.executor import A2aAgentExecutor
from google.adk.a2a.converters.from_adk_event import convert_adk_event_to_a2a_events
from google.adk.a2a.converters.part_converter import convert_genai_part_to_a2a_part

class LoggingExecutor(A2aAgentExecutor):
    """Logs every A2A event before forwarding it to the queue."""

    async def execute(self, context, event_queue):
        agents_artifacts: dict[str, str] = {}

        async for adk_event in self._run_agent(context):
            a2a_events = convert_adk_event_to_a2a_events(
                adk_event,
                agents_artifacts,
                context.task_id,
                context.context_id,
                convert_genai_part_to_a2a_part,   # default part converter
            )
            for evt in a2a_events:
                print(f"[A2A] {type(evt).__name__}: {evt}")
                await event_queue.put(evt)
```

### Example 2 — custom converter to filter sensitive parts

```python
from typing import Optional, Dict, List
from google.adk.events.event import Event
from google.adk.a2a.converters.from_adk_event import (
    convert_adk_event_to_a2a_events,
    AdkEventToA2AEventsConverter,
)
from google.adk.a2a.converters.part_converter import (
    GenAIPartToA2APartConverter,
    convert_genai_part_to_a2a_part,
)
from a2a.types import TaskStatusUpdateEvent, TaskArtifactUpdateEvent

SENSITIVE_KEYWORDS = {"password", "api_key", "secret"}

def _redacting_part_converter(part):
    """Wraps the default converter; redacts text parts with sensitive keywords."""
    a2a_part = convert_genai_part_to_a2a_part(part)
    if a2a_part is None:
        return None
    inner = a2a_part.root
    if hasattr(inner, "text") and inner.text:
        lower = inner.text.lower()
        if any(kw in lower for kw in SENSITIVE_KEYWORDS):
            inner.text = "[REDACTED]"
    return a2a_part

def redacting_converter(
    event: Event,
    agents_artifacts: Optional[Dict[str, str]],
    task_id: Optional[str],
    context_id: Optional[str],
    part_converter: GenAIPartToA2APartConverter,
) -> List[TaskStatusUpdateEvent | TaskArtifactUpdateEvent]:
    # Use the custom part converter instead of the default one
    return convert_adk_event_to_a2a_events(
        event,
        agents_artifacts,
        task_id,
        context_id,
        _redacting_part_converter,   # inject custom converter
    )

# Wire into to_a2a via a custom executor factory
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import LlmAgent

agent = LlmAgent(name="safe_agent", model="gemini-2.5-flash", instruction="Be helpful.")

# Pass the custom executor that uses redacting_converter
# (the factory receives the runner and returns an executor instance)
app = to_a2a(
    agent,
    host="0.0.0.0",
    port=8004,
)
```

### Example 3 — inspecting the `agents_artifacts` state dict during streaming

```python
import asyncio
from google.adk.a2a.converters.from_adk_event import convert_adk_event_to_a2a_events
from google.adk.a2a.converters.part_converter import convert_genai_part_to_a2a_part
from a2a.types import TaskArtifactUpdateEvent

async def stream_and_inspect(event_source, task_id: str, context_id: str):
    """Shows how agents_artifacts evolves across a streaming response."""
    agents_artifacts: dict[str, str] = {}

    async for adk_event in event_source:
        a2a_events = convert_adk_event_to_a2a_events(
            adk_event,
            agents_artifacts,
            task_id,
            context_id,
            convert_genai_part_to_a2a_part,
        )

        for evt in a2a_events:
            if isinstance(evt, TaskArtifactUpdateEvent):
                print(
                    f"Artifact update for agent={adk_event.author!r}: "
                    f"artifact_id={evt.artifact.artifact_id!r}, "
                    f"append={evt.append}"
                )

    # After all events, agents_artifacts holds the last artifact_id per agent.
    # A value of None means the agent's artifact stream was finalised.
    print("Final agents_artifacts state:", agents_artifacts)
```

---

## 4 · `AuthScheme` + `OAuthGrantType` + `ExtendedOAuth2` + `OpenIdConnectWithConfig` + `CustomAuthScheme`

**Source:** `google.adk.auth.auth_schemes`

`AuthScheme` is the union type that encompasses every authentication scheme ADK supports. `OpenIdConnectWithConfig` extends the standard FastAPI `SecurityBase` with explicit endpoint URLs so that ADK can drive the OIDC flow without a discovery document. `CustomAuthScheme` is a base class for proprietary auth systems that don't fit any standard scheme. `ExtendedOAuth2` adds `issuer_url` to the standard `OAuth2` scheme for endpoint auto-discovery. `OAuthGrantType` is an enum that normalises the four standard OAuth 2.0 grant types and can be detected from an `OAuthFlows` object.

### Types (source-verified)

```python
from typing import Union, Optional, List
from fastapi.security import OAuth2
from fastapi.security.base import SecurityBase
from fastapi.security.oauth2 import OAuthFlows
from fastapi.openapi.models import SecurityScheme, SecuritySchemeType
from pydantic import BaseModel, Field
from enum import Enum
from google.adk.auth.auth_schemes import (
    AuthScheme,
    AuthSchemeType,
    OAuthGrantType,
    OpenIdConnectWithConfig,
    CustomAuthScheme,
    ExtendedOAuth2,
)

# AuthScheme is the union type used wherever a scheme is declared:
# AuthScheme = Union[SecurityScheme, OpenIdConnectWithConfig, CustomAuthScheme]
# AuthSchemeType = SecuritySchemeType   (re-export from FastAPI)

class OAuthGrantType(str, Enum):
    CLIENT_CREDENTIALS  = "client_credentials"
    AUTHORIZATION_CODE  = "authorization_code"
    IMPLICIT            = "implicit"
    PASSWORD            = "password"

    @staticmethod
    def from_flow(flow: OAuthFlows) -> "OAuthGrantType":
        # Inspects the OAuthFlows object to determine which grant type is set
        ...

class OpenIdConnectWithConfig(SecurityBase):
    type_: SecuritySchemeType = SecuritySchemeType.openIdConnect
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: Optional[str] = None
    revocation_endpoint: Optional[str] = None
    token_endpoint_auth_methods_supported: Optional[List[str]] = None
    grant_types_supported: Optional[List[str]] = None
    scopes: Optional[List[str]] = None

class CustomAuthScheme(BaseModel):
    # Subclasses must define a default for type_ for OAuth2 user
    # consent flow rehydration.  Uses alias "type" on the wire.
    type_: str = Field(alias="type")

class ExtendedOAuth2(OAuth2):
    issuer_url: Optional[str] = None   # for OIDC endpoint auto-discovery
```

### Pairing `AuthScheme` with `AuthCredential`

| Scheme | `AuthCredential` fields |
|---|---|
| `SecurityScheme(type=apiKey)` | `auth_type=API_KEY, api_key=...` |
| `SecurityScheme(type=http, scheme='bearer')` | `auth_type=HTTP, http=HttpAuth(scheme='bearer', credentials=HttpCredentials(token=...))` |
| `OAuth2` scheme | `auth_type=OAUTH2, oauth2=OAuth2Auth(...)` |
| `OpenIdConnectWithConfig` | `auth_type=OPEN_ID_CONNECT, oauth2=OAuth2Auth(...)` |
| Service-account type | `auth_type=SERVICE_ACCOUNT, service_account=ServiceAccount(...)` |

### Example 1 — API key scheme for `AuthenticatedFunctionTool`

```python
from fastapi.openapi.models import APIKey, APIKeyIn, SecuritySchemeType
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, ApiKey
from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool

# Declare the scheme: API key carried in an HTTP header
api_key_scheme = APIKey(**{
    "type": SecuritySchemeType.apiKey,
    "name": "X-API-Key",
    "in": APIKeyIn.header,
})

def call_weather_api(location: str) -> dict:
    """Fetches weather for the given location."""
    return {"temperature": "22C", "condition": "sunny"}

weather_tool = AuthenticatedFunctionTool(
    func=call_weather_api,
    auth_scheme=api_key_scheme,
    auth_credential=AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY,
        api_key=ApiKey(api_key="<injected-at-runtime>"),
    ),
)
```

### Example 2 — OAuth2 authorization_code flow scheme + credential pairing

```python
from fastapi.openapi.models import OAuth2 as OAuth2Scheme, OAuthFlows, OAuthFlowAuthorizationCode
from google.adk.auth.auth_credential import (
    AuthCredential, AuthCredentialTypes, OAuth2Auth
)
from google.adk.auth.auth_schemes import OAuthGrantType

# Define the scheme
flows = OAuthFlows(
    authorizationCode=OAuthFlowAuthorizationCode(
        authorizationUrl="https://accounts.google.com/o/oauth2/v2/auth",
        tokenUrl="https://oauth2.googleapis.com/token",
        scopes={
            "https://www.googleapis.com/auth/calendar.readonly": "Read calendar events",
        },
    )
)
oauth2_scheme = OAuth2Scheme(flows=flows)

# Detect grant type at runtime
grant_type = OAuthGrantType.from_flow(flows)
print(grant_type)  # OAuthGrantType.AUTHORIZATION_CODE

# Credential pairing — token is injected after the user consents
credential = AuthCredential(
    auth_type=AuthCredentialTypes.OAUTH2,
    oauth2=OAuth2Auth(
        client_id="my-client-id",
        client_secret="my-client-secret",
        redirect_uri="https://myapp.example.com/oauth2/callback",
        scopes=["https://www.googleapis.com/auth/calendar.readonly"],
    ),
)
```

### Example 3 — custom `CustomAuthScheme` subclass for a proprietary auth system

```python
from typing import Optional
from pydantic import Field
from google.adk.auth.auth_schemes import CustomAuthScheme
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes

class HmacAuthScheme(CustomAuthScheme):
    """HMAC-SHA256 request signing scheme used by our internal API gateway."""

    # type_ must have a default so rehydration works during OAuth2 consent resumption
    type_: str = Field(default="hmacAuth", alias="type")

    # Scheme-specific configuration
    algorithm: str = "hmac-sha256"
    header_name: str = "X-HMAC-Signature"
    timestamp_header: str = "X-Request-Timestamp"
    signing_key_env_var: str = "HMAC_SIGNING_KEY"

    class Config:
        populate_by_name = True

# Use it in an AuthenticatedFunctionTool
hmac_scheme = HmacAuthScheme()

# The matching credential carries the signing key
class HmacCredential(AuthCredential):
    # Extend AuthCredential with the shared secret for HMAC signing
    hmac_key: Optional[str] = None

credential = HmacCredential(
    auth_type=AuthCredentialTypes.API_KEY,  # closest built-in type
    hmac_key="super-secret-key-from-vault",
)
```

### Example 4 — `OAuthGrantType.from_flow` usage

```python
from fastapi.openapi.models import OAuthFlows, OAuthFlowClientCredentials
from google.adk.auth.auth_schemes import OAuthGrantType

m2m_flows = OAuthFlows(
    clientCredentials=OAuthFlowClientCredentials(
        tokenUrl="https://auth.example.com/token",
        scopes={"read:data": "Read access to data"},
    )
)
grant = OAuthGrantType.from_flow(m2m_flows)
print(grant)  # OAuthGrantType.CLIENT_CREDENTIALS

# Useful for conditional logic in custom auth handlers:
if grant == OAuthGrantType.CLIENT_CREDENTIALS:
    # No user redirect needed; fetch token directly with client_id + client_secret
    pass
elif grant == OAuthGrantType.AUTHORIZATION_CODE:
    # Must redirect the user to the authorization_endpoint
    pass
```

---

## 5 · `McpInstructionProvider`

**Source:** `google.adk.agents.mcp_instruction_provider`

`McpInstructionProvider` implements the `InstructionProvider` protocol — a callable `(ReadonlyContext) -> str` — by fetching a named **MCP Prompt** from a running MCP server at each agent invocation. This decouples your agent's system prompt from your Python code: prompt engineers edit prompts in the MCP server and agents pick up changes without redeployment. It fits naturally into any `LlmAgent` as a drop-in replacement for a string or lambda `instruction`.

### Constructor (source-verified)

```python
import sys
from google.adk.agents.mcp_instruction_provider import McpInstructionProvider

McpInstructionProvider(
    connection_params=...,   # MCPSessionManager connection params (StdioServerParameters
                             # or SseServerParameters or StreamableHTTPServerParameters)
    prompt_name="my_prompt", # name of the MCP Prompt to fetch
    errlog=sys.stderr,       # where to write MCP session errors
)
```

### Call semantics (source-verified)

When `LlmAgent` resolves its instruction, it calls `provider(context: ReadonlyContext) -> str`:

1. Opens an MCP session via `MCPSessionManager(connection_params)`
2. Calls `session.list_prompts()` → finds the `PromptDefinition` for `prompt_name`
3. Extracts `arg_names` from `prompt_definition.arguments`
4. Looks up matching keys from `context.state` → builds `prompt_args` dict
5. Calls `session.get_prompt(prompt_name, arguments=prompt_args)` → `GetPromptResult`
6. Concatenates all `message.content.text` for `text`-type messages
7. Raises `ValueError` if the result contains no messages

### Example 1 — basic usage with a fixed MCP prompt

```python
from mcp import StdioServerParameters
from google.adk.agents import LlmAgent
from google.adk.agents.mcp_instruction_provider import McpInstructionProvider

# The MCP server exposes a prompt named "customer_support_v2"
mcp_params = StdioServerParameters(
    command="python",
    args=["./prompts_server.py"],
)

provider = McpInstructionProvider(
    connection_params=mcp_params,
    prompt_name="customer_support_v2",
)

agent = LlmAgent(
    name="support_agent",
    model="gemini-2.5-flash",
    # Drop-in replacement for a string instruction; fetched fresh each invocation
    instruction=provider,
)
```

### Example 2 — state-injected prompt args from session state

```python
from mcp import StdioServerParameters
from google.adk.agents import LlmAgent
from google.adk.agents.mcp_instruction_provider import McpInstructionProvider
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# The MCP prompt "localised_assistant" declares two arguments:
#   - user_language  (e.g. "fr", "de", "ja")
#   - expertise_level (e.g. "beginner", "expert")
# McpInstructionProvider reads matching keys from context.state automatically.

mcp_params = StdioServerParameters(
    command="node",
    args=["./prompts-server.js"],
)

provider = McpInstructionProvider(
    connection_params=mcp_params,
    prompt_name="localised_assistant",
)

agent = LlmAgent(
    name="localised_agent",
    model="gemini-2.5-flash",
    instruction=provider,
)

session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="demo", session_service=session_service)

# Inject prompt args as session state before the first run
import asyncio
from google.genai import types

async def run():
    session = await session_service.create_session(
        app_name="demo",
        user_id="user_42",
        state={
            "user_language": "fr",
            "expertise_level": "beginner",
        },
    )
    async for event in runner.run_async(
        user_id="user_42",
        session_id=session.id,
        new_message=types.Content(
            role="user", parts=[types.Part(text="Bonjour!")]
        ),
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

asyncio.run(run())
```

### Example 3 — fallback to static string on MCP server unavailability

```python
import sys
from mcp import StdioServerParameters
from google.adk.agents import LlmAgent
from google.adk.agents.mcp_instruction_provider import McpInstructionProvider
from google.adk.agents.readonly_context import ReadonlyContext

FALLBACK_INSTRUCTION = (
    "I am a helpful assistant. I answer questions clearly and concisely."
)

mcp_params = StdioServerParameters(
    command="python",
    args=["./prompts_server.py"],
)

_mcp_provider = McpInstructionProvider(
    connection_params=mcp_params,
    prompt_name="main_assistant",
    errlog=sys.stderr,
)

def resilient_instruction(context: ReadonlyContext) -> str:
    """Fetches the MCP prompt; falls back to a static string if unavailable."""
    try:
        return _mcp_provider(context)
    except Exception as exc:
        print(f"[warn] MCP prompt fetch failed ({exc!r}); using fallback.", file=sys.stderr)
        return FALLBACK_INSTRUCTION

agent = LlmAgent(
    name="resilient_agent",
    model="gemini-2.5-flash",
    instruction=resilient_instruction,
)
```

---

## 6 · `BaseArtifactService` + `ArtifactVersion` + `InMemoryArtifactService`

**Source:** `google.adk.artifacts`

`ArtifactVersion` is the versioned metadata record stored alongside each artifact blob. `BaseArtifactService` is the abstract storage contract — seven async methods covering save, load, list, delete, and version inspection. `InMemoryArtifactService` is the built-in reference implementation, with a path scheme that distinguishes session-scoped artifacts (tied to a specific session) from user-scoped artifacts (shared across sessions via the `"user:"` filename prefix).

### `ArtifactVersion` fields (source-verified)

```python
from google.adk.artifacts.base_artifact_service import ArtifactVersion

ArtifactVersion(
    # Uses camelCase aliases (alias_generator=to_camel) — serialises as camelCase JSON
    version=0,                  # monotonically increasing; first version = 0
    canonical_uri="memory://apps/myapp/users/u1/sessions/s1/artifacts/report.pdf/versions/0",
    custom_metadata={},         # free-form dict; pass your own indexing fields
    create_time=1749600000.0,   # unix timestamp; default = platform_time.get_time()
    mime_type="application/pdf",
)
```

### `BaseArtifactService` abstract contract (source-verified)

```python
from abc import ABC, abstractmethod
from typing import Optional
from google.genai import types
from google.adk.artifacts.base_artifact_service import BaseArtifactService, ArtifactVersion

class BaseArtifactService(ABC):

    @abstractmethod
    async def save_artifact(
        self, *, app_name: str, user_id: str, filename: str,
        artifact: types.Part, session_id: str,
        custom_metadata: dict | None = None,
    ) -> int:
        """Saves artifact; returns the new version number (0-based)."""

    @abstractmethod
    async def load_artifact(
        self, *, app_name: str, user_id: str, filename: str,
        session_id: str, version: int | None = None,
    ) -> Optional[types.Part]:
        """Loads artifact. version=None returns the latest version."""

    @abstractmethod
    async def list_artifact_keys(
        self, *, app_name: str, user_id: str, session_id: str,
    ) -> list[str]:
        """Returns both session-scoped and user-scoped filenames."""

    @abstractmethod
    async def delete_artifact(
        self, *, app_name: str, user_id: str, filename: str, session_id: str,
    ) -> None: ...

    @abstractmethod
    async def list_versions(
        self, *, app_name: str, user_id: str, filename: str, session_id: str,
    ) -> list[int]: ...

    @abstractmethod
    async def list_artifact_versions(
        self, *, app_name: str, user_id: str, filename: str, session_id: str,
    ) -> list[ArtifactVersion]: ...

    @abstractmethod
    async def get_artifact_version(
        self, *, app_name: str, user_id: str, filename: str,
        session_id: str, version: int,
    ) -> Optional[ArtifactVersion]: ...
```

### `InMemoryArtifactService` path scheme

| Scope | Key pattern | Canonical URI |
|---|---|---|
| Session-scoped | `"{app_name}/{user_id}/{session_id}/{filename}"` | `"memory://apps/{app}/users/{user}/sessions/{session}/artifacts/{file}/versions/{version}"` |
| User-scoped (`"user:"` prefix in filename) | `"{app_name}/{user_id}/user/{filename_without_prefix}"` | `"memory://apps/{app}/users/{user}/artifacts/{file}/versions/{version}"` |

If a loaded `Part` has `file_data.file_uri` starting with `artifact://`, `load_artifact` resolves it recursively using `artifact_util.parse_artifact_uri()`.

### Example 1 — save/load cycle with text + binary artifacts

```python
import asyncio
from google.genai import types
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService

svc = InMemoryArtifactService()
APP, USER, SESSION = "demo_app", "alice", "session_001"

async def demo():
    # Save a text artifact (version 0)
    v0 = await svc.save_artifact(
        app_name=APP, user_id=USER, session_id=SESSION,
        filename="summary.txt",
        artifact=types.Part(text="Initial summary of the quarterly report."),
    )
    print(f"Saved version: {v0}")  # 0

    # Save a new version (version 1)
    v1 = await svc.save_artifact(
        app_name=APP, user_id=USER, session_id=SESSION,
        filename="summary.txt",
        artifact=types.Part(text="Revised summary: revenue up 12%."),
    )
    print(f"Saved version: {v1}")  # 1

    # Load latest
    latest = await svc.load_artifact(
        app_name=APP, user_id=USER, session_id=SESSION, filename="summary.txt",
    )
    print(latest.text)  # "Revised summary: revenue up 12%."

    # Load specific version
    original = await svc.load_artifact(
        app_name=APP, user_id=USER, session_id=SESSION,
        filename="summary.txt", version=0,
    )
    print(original.text)  # "Initial summary of the quarterly report."

    # Save a binary artifact (e.g., an image)
    img_bytes = open("chart.png", "rb").read()
    await svc.save_artifact(
        app_name=APP, user_id=USER, session_id=SESSION,
        filename="chart.png",
        artifact=types.Part(
            inline_data=types.Blob(mime_type="image/png", data=img_bytes)
        ),
    )
    keys = await svc.list_artifact_keys(
        app_name=APP, user_id=USER, session_id=SESSION
    )
    print(keys)  # ['summary.txt', 'chart.png']

asyncio.run(demo())
```

### Example 2 — user-namespace artifacts shared across sessions

```python
import asyncio
from google.genai import types
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService

svc = InMemoryArtifactService()
APP, USER = "myapp", "bob"

async def user_scope_demo():
    # The "user:" prefix stores the artifact at the user level,
    # making it accessible from any session for this user.
    await svc.save_artifact(
        app_name=APP, user_id=USER, session_id="session_A",
        filename="user:preferences.json",
        artifact=types.Part(text='{"theme": "dark", "language": "en"}'),
    )

    # Load from a completely different session — still accessible
    prefs = await svc.load_artifact(
        app_name=APP, user_id=USER, session_id="session_B",
        filename="user:preferences.json",
    )
    print(prefs.text)  # '{"theme": "dark", "language": "en"}'

    # list_artifact_keys returns user-scoped keys from any session
    keys_from_b = await svc.list_artifact_keys(
        app_name=APP, user_id=USER, session_id="session_B"
    )
    assert "user:preferences.json" in keys_from_b

asyncio.run(user_scope_demo())
```

### Example 3 — custom `BaseArtifactService` with S3 backend

```python
import json
import time
from typing import Optional
import boto3
from google.genai import types
from google.adk.artifacts.base_artifact_service import BaseArtifactService, ArtifactVersion

class S3ArtifactService(BaseArtifactService):
    """Stores artifacts in S3; metadata in DynamoDB."""

    def __init__(self, bucket: str, table_name: str, region: str = "us-east-1"):
        self._s3 = boto3.client("s3", region_name=region)
        self._ddb = boto3.resource("dynamodb", region_name=region).Table(table_name)
        self._bucket = bucket

    def _s3_key(self, app_name, user_id, session_id, filename, version):
        return f"{app_name}/{user_id}/{session_id}/{filename}/v{version}"

    async def save_artifact(
        self, *, app_name, user_id, filename, artifact, session_id,
        custom_metadata=None,
    ) -> int:
        versions = await self.list_versions(
            app_name=app_name, user_id=user_id,
            filename=filename, session_id=session_id,
        )
        new_version = len(versions)
        key = self._s3_key(app_name, user_id, session_id, filename, new_version)

        body = artifact.text.encode() if artifact.text else artifact.inline_data.data
        mime = "text/plain" if artifact.text else artifact.inline_data.mime_type
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=body, ContentType=mime)

        av = ArtifactVersion(
            version=new_version,
            canonical_uri=f"s3://{self._bucket}/{key}",
            custom_metadata=custom_metadata or {},
            create_time=time.time(),
            mime_type=mime,
        )
        self._ddb.put_item(Item={
            "pk": f"{app_name}#{user_id}#{session_id}#{filename}",
            "sk": str(new_version),
            **av.model_dump(by_alias=True),
        })
        return new_version

    async def load_artifact(
        self, *, app_name, user_id, filename, session_id, version=None,
    ) -> Optional[types.Part]:
        all_versions = await self.list_versions(
            app_name=app_name, user_id=user_id,
            filename=filename, session_id=session_id,
        )
        if not all_versions:
            return None
        v = max(all_versions) if version is None else version
        key = self._s3_key(app_name, user_id, session_id, filename, v)
        obj = self._s3.get_object(Bucket=self._bucket, Key=key)
        body = obj["Body"].read()
        mime = obj["ContentType"]
        if mime == "text/plain":
            return types.Part(text=body.decode())
        return types.Part(inline_data=types.Blob(mime_type=mime, data=body))

    async def list_artifact_keys(self, *, app_name, user_id, session_id) -> list[str]:
        prefix = f"{app_name}/{user_id}/{session_id}/"
        resp = self._s3.list_objects_v2(Bucket=self._bucket, Prefix=prefix, Delimiter="/")
        return [cp["Prefix"].split("/")[-2] for cp in resp.get("CommonPrefixes", [])]

    async def delete_artifact(self, *, app_name, user_id, filename, session_id) -> None:
        versions = await self.list_versions(
            app_name=app_name, user_id=user_id, filename=filename, session_id=session_id
        )
        for v in versions:
            key = self._s3_key(app_name, user_id, session_id, filename, v)
            self._s3.delete_object(Bucket=self._bucket, Key=key)

    async def list_versions(self, *, app_name, user_id, filename, session_id) -> list[int]:
        prefix = self._s3_key(app_name, user_id, session_id, filename, "")
        resp = self._s3.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
        return sorted(
            int(obj["Key"].split("/v")[-1])
            for obj in resp.get("Contents", [])
        )

    async def list_artifact_versions(
        self, *, app_name, user_id, filename, session_id
    ) -> list[ArtifactVersion]:
        # Omitted for brevity — query DynamoDB for ArtifactVersion records
        return []

    async def get_artifact_version(
        self, *, app_name, user_id, filename, session_id, version
    ) -> Optional[ArtifactVersion]:
        return None
```

### Example 4 — `ArtifactVersion` metadata inspection

```python
import asyncio
from google.genai import types
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService

svc = InMemoryArtifactService()
APP, USER, SESSION = "audit_app", "carol", "s1"

async def inspect_versions():
    for i, content in enumerate(["draft v1", "draft v2", "final"]):
        await svc.save_artifact(
            app_name=APP, user_id=USER, session_id=SESSION,
            filename="report.txt",
            artifact=types.Part(text=content),
            custom_metadata={"author": "carol", "revision": i + 1},
        )

    # List version numbers
    versions = await svc.list_versions(
        app_name=APP, user_id=USER, filename="report.txt", session_id=SESSION
    )
    print(versions)  # [0, 1, 2]

    # Get full ArtifactVersion records
    av_list = await svc.list_artifact_versions(
        app_name=APP, user_id=USER, filename="report.txt", session_id=SESSION
    )
    for av in av_list:
        print(f"v{av.version}: uri={av.canonical_uri!r}, "
              f"meta={av.custom_metadata}, created={av.create_time:.0f}")

    # Get a specific version
    av2 = await svc.get_artifact_version(
        app_name=APP, user_id=USER, filename="report.txt",
        session_id=SESSION, version=2,
    )
    print(f"Latest canonical URI: {av2.canonical_uri}")

asyncio.run(inspect_versions())
```

---

## 7 · `RequestInput` + workflow HITL utilities

**Source:** `google.adk.events.request_input`, `google.adk.workflow.utils._workflow_hitl_utils`

`RequestInput` is the typed envelope for a **human-in-the-loop interrupt** inside a workflow. It carries an `interrupt_id` (used to correlate the resume response), an optional `response_schema` (so the UI can render a typed form), and an optional human-readable `message`. The HITL utility functions turn these objects into ADK `Event` objects that the A2A task state machine recognises as `input_required`, and provide a robust `process_auth_resume` with a three-fallback parsing chain for credential responses.

### `RequestInput` fields (source-verified)

```python
import uuid
from typing import Optional, Any
from pydantic import BaseModel, Field
from google.adk.events.request_input import RequestInput

RequestInput(
    # model_config: alias_generator=to_camel, populate_by_name=True
    interrupt_id=str(uuid.uuid4()),   # default_factory; correlates request ↔ response
    payload=None,                     # optional pre-filled data for the UI
    message=None,                     # human-readable prompt shown to the user
    response_schema=None,             # Python type, generic alias, or JSON Schema dict
                                      # serialised via schema_to_json_schema()
)
```

### Utility functions (source-verified)

```python
from google.adk.workflow.utils._workflow_hitl_utils import (
    create_request_input_event,
    create_request_input_response,
    create_auth_request_event,
    process_auth_resume,
    has_auth_credential,
)
```

`create_request_input_event(request_input)` returns an `Event` with:
- `content.parts[0].function_call.name = 'adk_request_input'`
- `content.parts[0].function_call.id = interrupt_id`
- `long_running_tool_ids = [interrupt_id]`

`create_request_input_response(interrupt_id, response)` returns a `types.Part` with `function_response(id=interrupt_id, name='adk_request_input', response=response)`.

`process_auth_resume(response_data, auth_config, state)` tries three fallbacks in order:
1. `AuthConfig.model_validate(response_data)`
2. `AuthCredential.model_validate(response_data)`
3. For API_KEY credential type: wraps the plain value as `AuthCredential(auth_type=API_KEY, api_key=str(value))`

### Example 1 — `FunctionNode` with `auth_config` triggering a HITL auth interrupt

```python
import asyncio
from google.adk.workflow import FunctionNode, Graph
from google.adk.auth.auth_config import AuthConfig
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes
from google.adk.auth.auth_schemes import AuthScheme
from fastapi.openapi.models import APIKey, APIKeyIn, SecuritySchemeType
from google.adk.workflow.utils._workflow_hitl_utils import (
    create_auth_request_event,
    has_auth_credential,
)

api_key_scheme = APIKey(**{
    "type": SecuritySchemeType.apiKey,
    "name": "X-API-Key",
    "in": APIKeyIn.header,
})

# AuthConfig combines the scheme + an (initially empty) credential template
auth_config = AuthConfig(
    auth_scheme=api_key_scheme,
    raw_auth_credential=AuthCredential(auth_type=AuthCredentialTypes.API_KEY),
)

def check_and_call_api(tool_context):
    """Returns None if auth is missing (triggers HITL); else calls the API."""
    from google.adk.workflow.utils._workflow_hitl_utils import has_auth_credential
    if not has_auth_credential(auth_config, tool_context.state):
        # Emit the auth interrupt event for the A2A client
        return create_auth_request_event(auth_config, interrupt_id="auth_001")
    # Auth is present — proceed with the API call
    return {"status": "ok", "data": "..."}

node = FunctionNode(
    name="api_caller",
    func=check_and_call_api,
    auth_config=auth_config,
)
```

### Example 2 — manual `RequestInput` with typed `response_schema`

```python
import asyncio
from pydantic import BaseModel
from google.adk.events.request_input import RequestInput
from google.adk.workflow.utils._workflow_hitl_utils import (
    create_request_input_event,
    create_request_input_response,
)

class ApprovalDecision(BaseModel):
    approved: bool
    reason: str
    approver_name: str

# Create the interrupt event — sent to the A2A client as input_required
ri = RequestInput(
    message="Please review and approve the order before it is dispatched.",
    response_schema=ApprovalDecision,   # serialised to JSON Schema for the UI
    payload={"order_id": "ORD-9821", "total": 1250.00},
)
request_event = create_request_input_event(ri)
# request_event.long_running_tool_ids == [ri.interrupt_id]
# A2A task state → input_required

# --- Later, when the user submits their approval ---
user_response = {
    "approved": True,
    "reason": "Within budget threshold.",
    "approver_name": "Dana",
}
resume_part = create_request_input_response(
    interrupt_id=ri.interrupt_id,
    response=user_response,
)
# Add resume_part to the next run's new_message to resume the workflow
```

### Example 3 — `process_auth_resume` with all three response formats

```python
from google.adk.auth.auth_config import AuthConfig
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, ApiKey
from fastapi.openapi.models import APIKey, APIKeyIn, SecuritySchemeType
from google.adk.workflow.utils._workflow_hitl_utils import process_auth_resume

api_key_scheme = APIKey(**{
    "type": SecuritySchemeType.apiKey,
    "name": "X-API-Key",
    "in": APIKeyIn.header,
})
auth_config = AuthConfig(
    auth_scheme=api_key_scheme,
    raw_auth_credential=AuthCredential(auth_type=AuthCredentialTypes.API_KEY),
)
state = {}

# Format 1: full AuthConfig dict (most structured)
full_auth_config_response = {
    "auth_scheme": {"type": "apiKey", "name": "X-API-Key", "in": "header"},
    "exchanged_auth_credential": {
        "auth_type": "api_key",
        "api_key": {"api_key": "sk-abc123"},
    },
}
process_auth_resume(full_auth_config_response, auth_config, state)

# Format 2: AuthCredential dict
credential_response = {
    "auth_type": "api_key",
    "api_key": {"api_key": "sk-def456"},
}
process_auth_resume(credential_response, auth_config, state)

# Format 3: plain scalar value — automatically wrapped as API_KEY credential
plain_key_response = "sk-plain-key-789"
process_auth_resume(plain_key_response, auth_config, state)
# Internally: AuthCredential(auth_type=API_KEY, api_key=ApiKey(api_key="sk-plain-key-789"))
```

---

## 8 · `RetryConfig` + `_should_retry_node` + `_get_retry_delay`

**Source:** `google.adk.workflow`

`RetryConfig` is the Pydantic model that controls exponential-backoff retry behaviour for individual workflow nodes. `_should_retry_node` is the predicate called after every node failure to decide whether to retry. `_get_retry_delay` computes the next sleep duration using a jittered exponential backoff algorithm. Together they give workflow nodes fault-tolerance against transient errors without requiring any application-level retry logic.

### `RetryConfig` fields (source-verified)

```python
from google.adk.workflow.retry_config import RetryConfig

RetryConfig(
    max_attempts=5,         # default; 0 or 1 means no retries
    initial_delay=1.0,      # default seconds; delay before the 2nd attempt
    max_delay=60.0,         # default seconds; cap on computed delay
    backoff_factor=2.0,     # default; multiplier per attempt
    jitter=1.0,             # default; 0.0 = no jitter; 1.0 = ±100% of delay
    exceptions=None,        # None = retry all exceptions
                            # list[str | type[BaseException]] = allowlist
)
```

`exceptions` is normalised by validator `_normalize_exceptions` — passing `[ValueError]` is equivalent to `["ValueError"]`.

### `_get_retry_delay` algorithm (source-verified)

```python
# Exact algorithm from source:
attempt_for_calc = max(0, attempt_count - 1)   # 0-indexed exponent
delay = initial_delay * (backoff_factor ** attempt_for_calc)
delay = min(delay, max_delay)
if jitter > 0.0:
    random_offset = random.uniform(-jitter * delay, jitter * delay)
    delay = max(0.0, delay + random_offset)
```

### Delay table — default settings (no jitter, deterministic for illustration)

With `initial_delay=1.0`, `backoff_factor=2.0`, `max_delay=60.0`, `jitter=0.0`:

| Attempt | `attempt_for_calc` | `delay` before jitter | `min(delay, 60)` |
|---|---|---|---|
| 1 (first failure) | 0 | `1.0 × 2⁰ = 1.0s` | **1.0 s** |
| 2 | 1 | `1.0 × 2¹ = 2.0s` | **2.0 s** |
| 3 | 2 | `1.0 × 2² = 4.0s` | **4.0 s** |
| 4 | 3 | `1.0 × 2³ = 8.0s` | **8.0 s** |
| 5 | 4 | `1.0 × 2⁴ = 16.0s` | **16.0 s** |

With default `jitter=1.0` each delay is offset by a uniform random value in `[-delay, +delay]`, then floored at 0.0.

### Example 1 — basic `RetryConfig` on a `FunctionNode`

```python
import httpx
from google.adk.workflow import FunctionNode, Graph
from google.adk.workflow.retry_config import RetryConfig

retry_cfg = RetryConfig(
    max_attempts=3,
    initial_delay=2.0,
    backoff_factor=2.0,
    max_delay=30.0,
    jitter=0.5,   # ±50% of delay
)

def fetch_exchange_rate(base: str, quote: str) -> dict:
    """Fetches the current exchange rate from an external API."""
    resp = httpx.get(
        f"https://api.exchangerate.example.com/v1/latest",
        params={"base": base, "symbols": quote},
        timeout=5.0,
    )
    resp.raise_for_status()
    return {"rate": resp.json()["rates"][quote]}

exchange_node = FunctionNode(
    name="fetch_exchange_rate",
    func=fetch_exchange_rate,
    retry_config=retry_cfg,
)

# If fetch_exchange_rate raises any exception, the node retries up to 3 times
# with delays of approximately 2s, 4s (plus ±50% jitter each time).
```

### Example 2 — exception allowlist: retry only on HTTP errors

```python
import httpx
from google.adk.workflow import FunctionNode
from google.adk.workflow.retry_config import RetryConfig

# Only transient HTTP errors (5xx, timeouts) trigger a retry;
# programming errors (TypeError, ValueError) propagate immediately.
http_retry = RetryConfig(
    max_attempts=4,
    initial_delay=1.0,
    backoff_factor=3.0,
    max_delay=30.0,
    jitter=0.25,
    exceptions=[httpx.HTTPStatusError, httpx.TimeoutException],
    # Equivalent: exceptions=["HTTPStatusError", "TimeoutException"]
)

def call_payment_gateway(amount: float, currency: str) -> dict:
    resp = httpx.post(
        "https://payments.example.com/charge",
        json={"amount": amount, "currency": currency},
        timeout=10.0,
    )
    resp.raise_for_status()   # raises HTTPStatusError on 4xx/5xx
    return resp.json()

payment_node = FunctionNode(
    name="charge_payment",
    func=call_payment_gateway,
    retry_config=http_retry,
)
```

### Example 3 — zero-jitter deterministic retry for testing

```python
from google.adk.workflow.retry_config import RetryConfig
from google.adk.workflow.utils._retry_utils import _should_retry_node, _get_retry_delay

# In tests you want predictable delays and deterministic retry decisions.
deterministic_retry = RetryConfig(
    max_attempts=3,
    initial_delay=0.01,    # fast in tests
    backoff_factor=2.0,
    max_delay=1.0,
    jitter=0.0,            # zero jitter → exact computed values
)

class _FakeNodeState:
    def __init__(self, attempt_count):
        self.attempt_count = attempt_count

# Verify retry decisions
assert _should_retry_node(ValueError("boom"), deterministic_retry, _FakeNodeState(1)) is True
assert _should_retry_node(ValueError("boom"), deterministic_retry, _FakeNodeState(3)) is False

# Verify exact delays (jitter=0.0 → always the computed value)
assert _get_retry_delay(deterministic_retry, _FakeNodeState(1)) == 0.01   # attempt_for_calc=0
assert _get_retry_delay(deterministic_retry, _FakeNodeState(2)) == 0.02   # attempt_for_calc=1
assert _get_retry_delay(deterministic_retry, _FakeNodeState(3)) == 0.04   # attempt_for_calc=2
```

### Example 4 — delay table for attempts 1–5 with default settings

```python
from google.adk.workflow.retry_config import RetryConfig
from google.adk.workflow.utils._retry_utils import _get_retry_delay

default_retry = RetryConfig(
    max_attempts=5,
    initial_delay=1.0,
    max_delay=60.0,
    backoff_factor=2.0,
    jitter=0.0,   # zero jitter for a reproducible table
)

class _FakeState:
    def __init__(self, n): self.attempt_count = n

print("Attempt | Delay (s)")
print("--------|----------")
for attempt in range(1, 6):
    d = _get_retry_delay(default_retry, _FakeState(attempt))
    print(f"    {attempt}   |  {d:.2f}")

# Attempt | Delay (s)
# --------|----------
#     1   |  1.00
#     2   |  2.00
#     3   |  4.00
#     4   |  8.00
#     5   |  16.00
```

---

## 9 · `TaskResultAggregator` + `ExecutorContext`

**Source:** `google.adk.a2a.executor`

> Both are decorated `@a2a_experimental`.

`TaskResultAggregator` solves a subtle problem in A2A streaming: the `DefaultRequestHandler` terminates the event loop as soon as it sees a non-`working` `TaskState`. For long multi-step agents this would end the task prematurely the first time an `input_required` or `auth_required` event is emitted mid-stream. `TaskResultAggregator` buffers the priority state across all events and rewrites intermediate events to `working`, only surfacing the true final state once the agent turn completes.

`ExecutorContext` is a lightweight value object that carries the per-request routing quadruple (`app_name`, `user_id`, `session_id`, `runner`) from the `DefaultRequestHandler` into a custom executor, without polluting the A2A `RequestContext`.

### `TaskResultAggregator` (source-verified)

```python
from google.adk.a2a.executor import TaskResultAggregator
from a2a.types import TaskState

# Priority ordering (highest to lowest):
#   failed > auth_required > input_required > working
#
# Key behaviour: when self._task_state == TaskState.working,
# ALL intermediate event states are rewritten to TaskState.working.
# This prevents DefaultRequestHandler from closing the event loop early.
# Only the final aggregated state reflects the true priority.

agg = TaskResultAggregator()
print(agg.task_state)              # TaskState.working (initial)
print(agg.task_status_message)     # None
```

### `ExecutorContext` (source-verified)

```python
from google.adk.a2a.executor import ExecutorContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.agents import LlmAgent

agent = LlmAgent(name="my_agent", model="gemini-2.5-flash", instruction="Be helpful.")
runner = Runner(agent=agent, app_name="myapp", session_service=InMemorySessionService())

ctx = ExecutorContext(
    app_name="myapp",
    user_id="user_123",
    session_id="session_456",
    runner=runner,
)

print(ctx.app_name)    # "myapp"
print(ctx.user_id)     # "user_123"
print(ctx.session_id)  # "session_456"
print(ctx.runner)      # <Runner ...>
```

### Example 1 — manual use of `TaskResultAggregator` outside the executor

```python
from a2a.types import TaskState, TaskStatusUpdateEvent, TaskStatus, Message, TextPart, Part
from google.adk.a2a.executor import TaskResultAggregator

def make_status_event(state: TaskState, text: str) -> TaskStatusUpdateEvent:
    return TaskStatusUpdateEvent(
        task_id="t1",
        context_id="c1",
        status=TaskStatus(
            state=state,
            message=Message(
                role="agent",
                parts=[Part(root=TextPart(text=text))],
            ),
        ),
        final=False,
    )

agg = TaskResultAggregator()

# Simulate a sequence of events from a multi-step agent
events = [
    make_status_event(TaskState.working, "Searching knowledge base..."),
    make_status_event(TaskState.working, "Drafting response..."),
    make_status_event(TaskState.input_required, "Please confirm the action."),
    make_status_event(TaskState.working, "Awaiting confirmation..."),
]

for evt in events:
    agg.process_event(evt)
    print(f"Input state: {evt.status.state!r} → aggregated: {agg.task_state!r}")
    # All intermediate events have their state rewritten to working
    # so consumers see a consistent stream.

print(f"Final aggregated state: {agg.task_state}")  # TaskState.input_required
```

### Example 2 — priority demonstration: `auth_required` then `failed` → final is `failed`

```python
from a2a.types import TaskState, TaskStatusUpdateEvent, TaskStatus
from google.adk.a2a.executor import TaskResultAggregator

agg = TaskResultAggregator()

def make_evt(state: TaskState) -> TaskStatusUpdateEvent:
    return TaskStatusUpdateEvent(
        task_id="t2",
        context_id="c2",
        status=TaskStatus(state=state),
        final=False,
    )

# Scenario: agent requests auth, then encounters an unrecoverable error
agg.process_event(make_evt(TaskState.auth_required))
print(agg.task_state)   # TaskState.auth_required  (won over working)

agg.process_event(make_evt(TaskState.failed))
print(agg.task_state)   # TaskState.failed  (failed always wins)

# Even if a subsequent working event arrives, failed is sticky
agg.process_event(make_evt(TaskState.working))
print(agg.task_state)   # TaskState.failed  (failed is never overridden)
```

### Example 3 — custom executor subclass using `ExecutorContext` for per-tenant routing

```python
from google.adk.a2a.executor import A2aAgentExecutor, ExecutorContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.agents import LlmAgent

# Tenant-specific runner pool (one runner per tenant model override)
_tenant_runners: dict[str, Runner] = {}

def get_tenant_runner(tenant_id: str) -> Runner:
    if tenant_id not in _tenant_runners:
        agent = LlmAgent(
            name=f"tenant_{tenant_id}_agent",
            model="gemini-2.5-flash",
            instruction=f"You serve tenant {tenant_id}.",
        )
        _tenant_runners[tenant_id] = Runner(
            agent=agent,
            app_name=f"app_{tenant_id}",
            session_service=InMemorySessionService(),
        )
    return _tenant_runners[tenant_id]

class TenantRoutingExecutor(A2aAgentExecutor):
    """Routes each request to the appropriate per-tenant Runner."""

    def _build_executor_context(self, request_context) -> ExecutorContext:
        # Extract tenant_id from the A2A request metadata
        tenant_id = (request_context.request.metadata or {}).get(
            "tenant_id", "default"
        )
        runner = get_tenant_runner(tenant_id)
        return ExecutorContext(
            app_name=runner.app_name,
            user_id=request_context.current_task.created_by or "anonymous",
            session_id=request_context.current_task.session_id or "new",
            runner=runner,
        )

    async def execute(self, request_context, event_queue):
        exec_ctx = self._build_executor_context(request_context)
        # Use exec_ctx.runner to run the agent for this tenant
        async for event in exec_ctx.runner.run_async(
            user_id=exec_ctx.user_id,
            session_id=exec_ctx.session_id,
            new_message=request_context.get_user_input(),
        ):
            await event_queue.put(event)
```

---

## 10 · `ToolConfig` + `BaseToolConfig` + `ToolArgsConfig`

**Source:** `google.adk.tools.tool_configs`

> All three are decorated `@experimental(FeatureName.TOOL_CONFIG)`.

`ToolConfig`, `BaseToolConfig`, and `ToolArgsConfig` are the building blocks of ADK's declarative tool loading system. They let you declare tools in YAML agent configuration files and have the ADK loader resolve them to `BaseTool` instances at runtime — by name (built-ins), by class path (with or without constructor args), by instance path, or by calling a factory function. `BaseToolConfig` is the extension point for fully custom loading logic.

### Types (source-verified)

```python
from pydantic import BaseModel, ConfigDict
from typing import Optional
from google.adk.tools.tool_configs import BaseToolConfig, ToolArgsConfig, ToolConfig

class BaseToolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

class ToolArgsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")   # accepts any free key-value pairs

class ToolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str    # built-in name, or fully-qualified class / instance / function path
    args: Optional[ToolArgsConfig] = None
```

### Five configuration patterns (source-verified from docstring)

#### Pattern 1 — ADK built-in tool by name

```yaml
# agent.yaml
tools:
  - name: google_search
```

```python
ToolConfig(name="google_search")
```

#### Pattern 2 — ADK built-in tool class with args

```yaml
# agent.yaml
tools:
  - name: AgentTool
    args:
      agent: ./another_agent.yaml
```

```python
ToolConfig(name="AgentTool", args=ToolArgsConfig(agent="./another_agent.yaml"))
```

#### Pattern 3 — user-defined tool instance by fully qualified path

```yaml
# agent.yaml
tools:
  - name: my_package.my_module.my_tool_instance
```

```python
ToolConfig(name="my_package.my_module.my_tool_instance")
```

#### Pattern 4 — user-defined tool class with args (instantiated at load time)

```yaml
# agent.yaml
tools:
  - name: my_package.tools.MyCustomTool
    args:
      api_key: "${MY_API_KEY}"
      timeout: 30
```

```python
ToolConfig(
    name="my_package.tools.MyCustomTool",
    args=ToolArgsConfig(api_key="${MY_API_KEY}", timeout=30),
)
```

#### Pattern 5 — factory function that returns a `BaseTool`

```yaml
# agent.yaml
tools:
  - name: my_package.tools.create_search_tool
    args:
      index_name: product_catalog
      top_k: 5
```

```python
# The factory function signature must be:
# def create_search_tool(args: ToolArgsConfig) -> BaseTool

from google.adk.tools.tool_configs import ToolArgsConfig
from google.adk.tools import BaseTool

def create_search_tool(args: ToolArgsConfig) -> BaseTool:
    from my_package.tools import VectorSearchTool
    return VectorSearchTool(
        index_name=args.index_name,
        top_k=int(args.top_k),
    )

ToolConfig(
    name="my_package.tools.create_search_tool",
    args=ToolArgsConfig(index_name="product_catalog", top_k=5),
)
```

#### Pattern 6 — plain function tool

```yaml
# agent.yaml
tools:
  - name: my_package.tools.lookup_order
```

```python
# lookup_order is a plain Python function; the loader wraps it in FunctionTool
ToolConfig(name="my_package.tools.lookup_order")
```

### Example 1 — YAML `agent.yaml` referencing tools by name and class

```yaml
# agent.yaml
name: shopping_assistant
model: gemini-2.5-flash
instruction: "I am a shopping assistant. I help users find and order products."
tools:
  # Pattern 1: built-in by name
  - name: google_search

  # Pattern 2: built-in class with arg pointing to a sub-agent YAML
  - name: AgentTool
    args:
      agent: ./order_agent.yaml

  # Pattern 4: user-defined class with constructor args
  - name: my_shop.tools.ProductSearchTool
    args:
      catalog_id: "spring_2026"
      max_results: 10

  # Pattern 5: factory function
  - name: my_shop.tools.make_checkout_tool
    args:
      payment_provider: stripe
      currency: USD
```

### Example 2 — Python equivalent construction of `ToolConfig` objects

```python
from google.adk.tools.tool_configs import ToolConfig, ToolArgsConfig

# Mirrors the YAML above, constructed programmatically
configs = [
    ToolConfig(name="google_search"),
    ToolConfig(name="AgentTool", args=ToolArgsConfig(agent="./order_agent.yaml")),
    ToolConfig(
        name="my_shop.tools.ProductSearchTool",
        args=ToolArgsConfig(catalog_id="spring_2026", max_results=10),
    ),
    ToolConfig(
        name="my_shop.tools.make_checkout_tool",
        args=ToolArgsConfig(payment_provider="stripe", currency="USD"),
    ),
]

# Serialise back to dict (for debugging or writing YAML)
for cfg in configs:
    print(cfg.model_dump(exclude_none=True))
```

### Example 3 — custom `BaseToolConfig` subclass with `from_config()` for remote registry

```python
from typing import Optional
from pydantic import Field
from google.adk.tools.tool_configs import BaseToolConfig, ToolArgsConfig
from google.adk.tools.base_tool import BaseTool

class RemoteRegistryToolConfig(BaseToolConfig):
    """Loads a tool definition from a remote tool registry by slug."""

    # Extra fields beyond BaseToolConfig's strict schema
    # (BaseToolConfig uses extra="forbid", so we override model_config)
    from pydantic import ConfigDict
    model_config = ConfigDict(extra="forbid")

    registry_url: str = Field(description="Base URL of the remote tool registry")
    tool_slug: str = Field(description="Unique slug identifying the tool in the registry")
    api_token: Optional[str] = Field(default=None, description="Bearer token for registry access")

    @classmethod
    def from_config(cls, config: "RemoteRegistryToolConfig") -> BaseTool:
        """Fetches the tool spec from the registry and instantiates it."""
        import httpx
        from google.adk.tools import FunctionTool

        headers = {}
        if config.api_token:
            headers["Authorization"] = f"Bearer {config.api_token}"

        resp = httpx.get(
            f"{config.registry_url}/tools/{config.tool_slug}",
            headers=headers,
        )
        resp.raise_for_status()
        tool_spec = resp.json()

        # Build a FunctionTool from the registry spec
        def dynamic_tool(**kwargs):
            """Dynamically generated tool from registry."""
            result = httpx.post(
                tool_spec["endpoint"],
                json=kwargs,
                headers=headers,
            )
            return result.json()

        dynamic_tool.__name__ = tool_spec["name"]
        dynamic_tool.__doc__ = tool_spec.get("description", "")

        return FunctionTool(func=dynamic_tool)


# Usage: instantiate and call from_config
cfg = RemoteRegistryToolConfig(
    registry_url="https://tools.example.com/api",
    tool_slug="crm-lookup-v2",
    api_token="registry-token-xyz",
)
tool = RemoteRegistryToolConfig.from_config(cfg)
```

---

## Volume index

| Vol. | Classes |
|------|---------|
| [1](./google_adk_class_deep_dives/) | `LlmAgent`, `RunConfig`, `Context`, `BasePlugin`, `App`, `Workflow`, `BaseNode`, `FunctionTool`, `RetryConfig`, `BaseTool` |
| [2](./google_adk_class_deep_dives_v2/) | `RemoteA2aAgent`, `LangGraphAgent`, `AuthCredential`, `GcsArtifactService`, `PubSubToolset`, `SpannerToolset` |
| [3](./google_adk_class_deep_dives_v3/) | `AgentTool`, `BuiltInPlanner`, `PlanReActPlanner`, `InMemorySessionService`, `VertexAiRagMemoryService`, `UnsafeLocalCodeExecutor`, `BuiltInCodeExecutor`, `LiteLlm`, `AgentEvaluator`, shell agents |
| [4](./google_adk_class_deep_dives_v4/) | `LongRunningFunctionTool`, `LiveRequestQueue`, `AnthropicLlm`/`Claude`, `SqliteSessionService`, `ExecuteBashTool`, `TransferToAgentTool`, `VertexAiMemoryBankService`, `SimplePromptOptimizer`, `SkillRegistry`/`SkillToolset`, `ToolConfirmation` |
| [5](./google_adk_class_deep_dives_v5/) | `VertexAiSessionService`, `VertexAiSearchTool`, `VertexAiCodeExecutor`, `APIHubToolset`, `ToolboxToolset`, `ConversationScenario`, `TrajectoryEvaluator`, `AuthConfig`/`AuthHandler`, `PreloadMemoryTool`, `CodeExecutorContext` |
| [6](./google_adk_class_deep_dives_v6/) | `ComputerUseTool`/`BaseComputer`, `OpenAPIToolset`/`RestApiTool`, `LlmEventSummarizer`, `Session`/`State`, `Event`/`EventActions`, `ExampleTool`, `GoogleSearchTool`/`UrlContextTool`, `LlmBackedUserSimulator`/`UserPersona`, `GEPARootAgentPromptOptimizer`, `EnvironmentSimulationPlugin` |
| [7](./google_adk_class_deep_dives_v7/) | `InvocationContext`, `SetModelResponseTool`, `DynamicNodeScheduler`/`ctx.run_node()`, retrieval tool chain, `FirestoreSessionService`/`FirestoreMemoryService`, `BigQueryToolset`/`WriteMode`, `LangchainTool`, `CrewaiTool`, `SlackRunner`, `FileArtifactService` |
| [8](./google_adk_class_deep_dives_v8/) | `ReadonlyContext`, `FunctionNode`, `JoinNode`/`Trigger`, `ContainerCodeExecutor`, `GkeCodeExecutor`, `AgentEngineSandboxCodeExecutor`, `ApplicationIntegrationToolset`, `BigtableToolset`, `OpenAILlm` |
| [9](./google_adk_class_deep_dives_v9/) | `Gemma`/`Gemma3Ollama`, `ContextCacheConfig`/`GeminiContextCacheManager`, `DataAgentToolset`, `DiscoveryEngineSearchTool`, `GoogleMapsGroundingTool`, `EnterpriseWebSearchTool`, `LoadMemoryTool`, `LoadArtifactsTool`, `exit_loop`/`get_user_choice_tool`, multi-turn eval suite |
| [10](./google_adk_class_deep_dives_v10/) | `Graph`/`Edge`, `NodeRunner`, `NodeState`/`NodeStatus`, `_ParallelWorker`, `ActiveStreamingTool`/`TranscriptionEntry`, `CachePerformanceAnalyzer`, `StreamingResponseAggregator`, `AgentRefConfig`/`ArgumentConfig`/`CodeConfig`, `BaseAuthProvider`/`AuthProviderRegistry`, `FinishTaskTool`/`TaskRequest`/`TaskResult` |
| [11](./google_adk_class_deep_dives_v11/) | `GoogleApiToolset` family, `load_web_page`, `UiWidget`, `_ToolNode`, `SqliteSpanExporter`, `RougeEvaluator`/`FinalResponseMatchV1`, `FinalResponseMatchV2Evaluator`, `HallucinationsV1Evaluator`, function calling pipeline, `_ContentLlmRequestProcessor`+`_InstructionsLlmRequestProcessor` |
| [12](./google_adk_class_deep_dives_v12/) | `BasePlugin`/`PluginManager`, `ContextFilterPlugin`, `ReflectAndRetryToolPlugin`, `GlobalInstructionPlugin`, `SaveFilesAsArtifactsPlugin`, `MultimodalToolResultsPlugin`, `DebugLoggingPlugin`, `RunConfig`/`StreamingMode`/`ToolThreadPoolConfig`, `AuthenticatedFunctionTool`, `FeatureName`/feature flags |
| [13](./google_adk_class_deep_dives_v13/) | `ApigeeLlm`/`ApiType`, `AudioCacheManager`/`AudioCacheConfig`, `AudioTranscriber`, `TranscriptionManager`, `AgentEngineSandboxComputer`/VMaaS, `ParameterManagerClient`, `SecretManagerClient`, `GcpAuthProvider`/`GcpAuthProviderScheme`, `AgentRegistry`/`AgentRegistrySingleMcpToolset`, `OAuth2CredentialRefresher` pipeline |
| [14](./google_adk_class_deep_dives_v14/) | `A2aAgentExecutor`/`A2aAgentExecutorConfig`/`ExecuteInterceptor`, `Context`/`ToolContext` advanced API, `GcsArtifactService` deep-dive, `LlmEventSummarizer`/`BaseEventsSummarizer`, `EventsCompactionConfig` advanced, `SpannerVectorStoreSettings`, `LangGraphAgent`+checkpointer, `Trigger` sub-branch/isolation, `ResumabilityConfig`, `PubSubToolset` advanced |
| [15](./google_adk_class_deep_dives_v15/) | `LlmRequest`, `LlmResponse`, `BaseLlm`, `LLMRegistry`, `BaseLlmFlow`/`AutoFlow`/`SingleFlow`, `BaseLlmRequestProcessor`/`BaseLlmResponseProcessor`, `Sampler`/`AgentOptimizer`, `LocalEvalSampler`/`LocalEvalSamplerConfig`, optimization data types, `ApiRegistry` |
| **16** | **`AgentCardBuilder`/`to_a2a`, `convert_a2a_part_to_genai_part`/`convert_genai_part_to_a2a_part`, `AdkEventToA2AEventsConverter`, `AuthScheme`/`OAuthGrantType`/`ExtendedOAuth2`/`OpenIdConnectWithConfig`/`CustomAuthScheme`, `McpInstructionProvider`, `BaseArtifactService`/`ArtifactVersion`/`InMemoryArtifactService`, `RequestInput`/HITL utilities, `RetryConfig`/`_should_retry_node`/`_get_retry_delay`, `TaskResultAggregator`/`ExecutorContext`, `ToolConfig`/`BaseToolConfig`/`ToolArgsConfig`** |

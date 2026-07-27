---
title: "Class deep dives — volume 45 (to_mcp_server, AutoAuthCredentialExchanger, FeatureName/temporary_feature_override/@experimental decorators, PROGRESSIVE_SSE_STREAMING in StreamingResponseAggregator, parse_edge_items chain syntax, ScenarioGenerator, FunctionNode parameter_binding, _RequestIntercepterPlugin, print_event verbose, LlmAgent.set_default_model)"
description: "10 source-verified deep dives for google-adk 2.5.0: to_mcp_server (expose any ADK agent as an MCP server via FastMCP), AutoAuthCredentialExchanger (auto-dispatch by auth type with custom_exchangers override), FeatureName/FeatureConfig/temporary_feature_override/@experimental/@stable/@working_in_progress complete feature-flag system, PROGRESSIVE_SSE_STREAMING mode in StreamingResponseAggregator (ordered-parts accumulation, streaming FC partial-args via JSONPath), parse_edge_items + workflow chain syntax (chain tuples, routing maps, fan-out), ScenarioGenerator facade (Vertex AI AI-powered eval scenario generation), FunctionNode parameter_binding='node_input' + type coercions (dict→BaseModel, Content→str, generator support), _RequestIntercepterPlugin (LLM-request interception via custom_metadata UUID bridge for eval), print_event + verbose debug output, LlmAgent.set_default_model + set_default_live_model class-level model defaults."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 45"
  order: 114
---

import { Aside } from "@astrojs/starlight/components";

<Aside type="note">
All signatures, constants, and behaviours on this page were verified directly
against the installed package source (locate yours with
`python -c 'import google.adk; print(google.adk.__file__)'`) on
**google-adk == 2.5.0**. No documentation or blog posts were used as primary
sources.
</Aside>

---

## 1 · `to_mcp_server` — expose any ADK agent as an MCP server

**Sources:** `google/adk/tools/mcp_tool/_agent_to_mcp.py`

### Why it matters

`to_mcp_server` (new in 2.5.0, `@experimental(FeatureName.MCP_AGENT_SERVER)`) is
the MCP counterpart of `to_a2a`. It wraps any `BaseAgent` in a `FastMCP` server
so that MCP hosts — Claude Code, OpenAI Codex, IDE extensions, or any MCP client
— can drive an ADK agent using the standard Model Context Protocol.

### Internals

```python
from mcp.server.fastmcp import FastMCP
import weakref

@experimental(FeatureName.MCP_AGENT_SERVER)
def to_mcp_server(
    agent: BaseAgent,
    *,
    name: Optional[str] = None,
    instructions: Optional[str] = None,
    runner: Optional[Runner] = None,
) -> FastMCP:
    tool_name = name or agent.name or "adk_agent"
    server = FastMCP(name=tool_name, instructions=instructions)
    agent_runner = runner if runner is not None else _build_runner(agent)
    # WeakKeyDictionary: entry dropped when MCP connection is GC'd
    sessions: MutableMapping[object, str] = weakref.WeakKeyDictionary()

    async def call_agent(request: str, ctx: Context) -> list[ContentBlock]:
        return await _run_agent(agent_runner, request, ctx, sessions)

    server.add_tool(
        call_agent,
        name=tool_name,
        description=agent.description or f"Run the {tool_name} agent.",
        structured_output=False,
    )
    return server
```

**Session lifecycle.** One ADK session is kept per MCP connection — the
`ctx.session` object is the key in a `WeakKeyDictionary`. When the connection is
garbage-collected the entry disappears automatically; no explicit cleanup needed.

**Part→ContentBlock mapping.**

| ADK part field | MCP content block |
|---|---|
| `part.text` | `TextContent(type="text", text=...)` |
| `part.inline_data` with `image/*` | `ImageContent(type="image", data=b64, mimeType=...)` |
| `part.inline_data` with `audio/*` | `AudioContent(type="audio", data=b64, mimeType=...)` |
| other `inline_data` | `EmbeddedResource` with `BlobResourceContents` |
| function calls, thoughts | `None` (skipped) |

Intermediate (non-final) text events are forwarded as **MCP progress
notifications** (`ctx.report_progress`) so the host can stream partial output
while waiting for the final response.

### Example 1 — stdio server (CLI host)

```python
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool._agent_to_mcp import to_mcp_server

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
)

server = to_mcp_server(agent, instructions="An AI assistant powered by Gemini.")
# Launch with stdio transport (Claude Code adds it via mcp settings)
server.run(transport="stdio")
```

### Example 2 — HTTP server (networked host)

```python
import uvicorn
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool._agent_to_mcp import to_mcp_server

research_agent = LlmAgent(
    name="researcher",
    model="gemini-2.5-pro",
    instruction="Search and summarise research topics.",
    description="Research assistant that summarises topics on demand.",
)

server = to_mcp_server(research_agent, name="research")
# Exposes POST /mcp  (Streamable-HTTP transport)
app = server.streamable_http_app()   # returns a Starlette ASGI app
uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Example 3 — custom Runner for persistent storage

```python
from google.adk.runners import Runner
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.artifacts.gcs_artifact_service import GcsArtifactService
from google.adk.tools.mcp_tool._agent_to_mcp import to_mcp_server

persistent_runner = Runner(
    app_name="production_agent",
    agent=agent,
    session_service=DatabaseSessionService("postgresql+asyncpg://..."),
    artifact_service=GcsArtifactService("my-bucket"),
)

server = to_mcp_server(agent, runner=persistent_runner)
server.run(transport="streamable-http")
```

---

## 2 · `AutoAuthCredentialExchanger` — dispatch by auth type

**Sources:** `google/adk/tools/openapi_tool/auth/credential_exchangers/auto_auth_credential_exchanger.py`

### What it does

`AutoAuthCredentialExchanger` (`@experimental`) is a convenience wrapper that
selects the right `BaseAuthCredentialExchanger` based on
`auth_credential.auth_type`. The built-in dispatch table:

| `auth_type` | Exchanger used |
|---|---|
| `AuthCredentialTypes.OAUTH2` | `OAuth2CredentialExchanger` |
| `AuthCredentialTypes.OPEN_ID_CONNECT` | `OAuth2CredentialExchanger` |
| `AuthCredentialTypes.SERVICE_ACCOUNT` | `ServiceAccountCredentialExchanger` |
| anything else | returns credential unchanged |
| `None` / no credential | returns `None` |

### Constructor

```python
class AutoAuthCredentialExchanger(BaseAuthCredentialExchanger):
    def __init__(
        self,
        custom_exchangers: Optional[
            Dict[str, Type[BaseAuthCredentialExchanger]]
        ] = None,
    ):
        self.exchangers = {
            AuthCredentialTypes.OAUTH2: OAuth2CredentialExchanger,
            AuthCredentialTypes.OPEN_ID_CONNECT: OAuth2CredentialExchanger,
            AuthCredentialTypes.SERVICE_ACCOUNT: ServiceAccountCredentialExchanger,
        }
        if custom_exchangers:
            self.exchangers.update(custom_exchangers)
```

`custom_exchangers` is a plain `dict[str, Type[BaseAuthCredentialExchanger]]`
keyed by `AuthCredentialTypes` string values. Pass it to add new types or
override existing ones.

### Example 1 — service account → bearer token

```python
from google.adk.auth.auth_credential import (
    AuthCredential,
    AuthCredentialTypes,
    ServiceAccount,
    ServiceAccountCredential,
)
from google.adk.auth.auth_schemes import HttpAuthScheme, HttpCredentials
from google.adk.tools.openapi_tool.auth.credential_exchangers.auto_auth_credential_exchanger import (
    AutoAuthCredentialExchanger,
)

sa_credential = AuthCredential(
    auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
    service_account=ServiceAccount(
        service_account_credential=ServiceAccountCredential(
            type_="service_account",
            project_id="my-project",
            private_key_id="key-id",
            private_key="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
            client_email="sa@my-project.iam.gserviceaccount.com",
            client_id="123456",
            auth_uri="https://accounts.google.com/o/oauth2/auth",
            token_uri="https://oauth2.googleapis.com/token",
            auth_provider_x509_cert_url="https://www.googleapis.com/oauth2/v1/certs",
            client_x509_cert_url="https://www.googleapis.com/robot/v1/metadata/x509/...",
            universe_domain="googleapis.com",
        ),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    ),
)

bearer_scheme = HttpAuthScheme(http=HttpCredentials(scheme="bearer"))

exchanger = AutoAuthCredentialExchanger()
result = exchanger.exchange_credential(
    auth_scheme=bearer_scheme,
    auth_credential=sa_credential,
)
# result.http.credentials.token contains the access token
print(result.http.credentials.token)
```

### Example 2 — custom exchanger for API keys

```python
from google.adk.tools.openapi_tool.auth.credential_exchangers.base_credential_exchanger import (
    BaseAuthCredentialExchanger,
)

class ApiKeyRefresher(BaseAuthCredentialExchanger):
    def exchange_credential(self, auth_scheme, auth_credential=None):
        # Fetch a fresh API key from a secrets vault
        new_key = fetch_from_vault(auth_credential.api_key.value)
        return AuthCredential(
            auth_type=AuthCredentialTypes.API_KEY,
            api_key=ApiKey(value=new_key),
        )

exchanger = AutoAuthCredentialExchanger(
    custom_exchangers={AuthCredentialTypes.API_KEY: ApiKeyRefresher}
)
```

### Example 3 — OpenID Connect flow

```python
from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes
from google.adk.auth.auth_schemes import OpenIdConnectScheme

oidc_credential = AuthCredential(
    auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
    oauth2=OAuth2Credential(
        client_id="...",
        client_secret="...",
        auth_response_uri="https://auth.example.com/callback?code=abc",
        redirect_uri="https://my-app.example.com/callback",
    ),
)

oidc_scheme = OpenIdConnectScheme(openIdConnectUrl="https://auth.example.com/.well-known/openid-configuration")
exchanger = AutoAuthCredentialExchanger()
token_credential = exchanger.exchange_credential(oidc_scheme, oidc_credential)
# OAuth2CredentialExchanger handles OIDC by exchanging the auth code
```

---

## 3 · `FeatureName` + `FeatureConfig` + `temporary_feature_override` + decorators

**Sources:** `google/adk/features/_feature_registry.py`, `google/adk/features/_feature_decorator.py`

### Feature lifecycle stages

```python
class FeatureStage(Enum):
    WIP          = "wip"           # internal only, default_on=False
    EXPERIMENTAL = "experimental"  # API may change, default_on varies
    STABLE       = "stable"        # production-ready, default_on=True
```

A `FeatureConfig` holds the stage and the default-enabled flag:

```python
@dataclass
class FeatureConfig:
    stage: FeatureStage
    default_on: bool = False
```

### Priority order for `is_feature_enabled`

1. **Programmatic overrides** (`override_feature_enabled`) — highest priority
2. **Environment variables** `ADK_ENABLE_<NAME>` / `ADK_DISABLE_<NAME>`
3. **Registry defaults** (`FeatureConfig.default_on`)

### `temporary_feature_override` — scoped override for tests

```python
@contextmanager
def temporary_feature_override(
    feature_name: FeatureName,
    enabled: bool,
) -> Generator[None, None, None]:
    had_override = feature_name in _FEATURE_OVERRIDES
    original_value = _FEATURE_OVERRIDES.get(feature_name)
    _FEATURE_OVERRIDES[feature_name] = enabled
    try:
        yield
    finally:
        if had_override:
            _FEATURE_OVERRIDES[feature_name] = original_value
        else:
            _FEATURE_OVERRIDES.pop(feature_name, None)
```

The context manager saves and restores the override state for **non-overlapping
(nested or sequential) uses**. Because `_FEATURE_OVERRIDES` is a plain `dict`
mutated without a lock, two async tasks or threads overriding the *same* feature
simultaneously can restore each other's values out of order. Use it in
single-threaded tests or ensure no two concurrent tasks touch the same flag.

### Example 1 — check feature at runtime

```python
from google.adk.features import FeatureName, is_feature_enabled

if is_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING):
    print("Progressive SSE streaming is active")
```

### Example 2 — programmatic permanent override

```python
from google.adk.features import FeatureName, override_feature_enabled

# Force progressive streaming on regardless of env vars
override_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING, True)

# Disable MCP graceful error handling (internal kill-switch)
override_feature_enabled(FeatureName._MCP_GRACEFUL_ERROR_HANDLING, False)
```

### Example 3 — scoped test override

```python
import pytest
from google.adk.features import FeatureName, temporary_feature_override

@pytest.fixture
def with_progressive_streaming():
    with temporary_feature_override(FeatureName.PROGRESSIVE_SSE_STREAMING, True):
        yield

def test_streaming_order(with_progressive_streaming):
    # PROGRESSIVE_SSE_STREAMING is on only for this test
    assert is_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING)

def test_default_behavior():
    # Back to the registry default after fixture teardown
    ...
```

### Example 4 — env-var toggle (no code change)

```bash
# Enable progressive streaming in production
ADK_ENABLE_PROGRESSIVE_SSE_STREAMING=1 python my_agent.py

# Disable it temporarily without redeployment
ADK_DISABLE_PROGRESSIVE_SSE_STREAMING=1 python my_agent.py
```

### The `@experimental`, `@stable`, `@working_in_progress` decorators

These class/function decorators register a `FeatureConfig` in `_FEATURE_REGISTRY`
(on first use) and **wrap `__init__` / the function** to call
`check_feature_enabled()` at invocation time — raising `RuntimeError` if the
feature is disabled. They are used internally by ADK on its own classes; the
`feature_name` argument must be a real `FeatureName` enum member.

```python
from google.adk.features._feature_decorator import experimental, stable, working_in_progress
from google.adk.features._feature_registry import FeatureName

# @experimental wraps __init__: raises RuntimeError if feature is disabled
# (TOOL_CONFIRMATION defaults to default_on=False)
@experimental(FeatureName.TOOL_CONFIRMATION)
class ToolConfirmationVariant:
    def __init__(self):
        ...  # RuntimeError raised here when TOOL_CONFIRMATION is disabled

# @stable wraps a function: raises RuntimeError only if explicitly disabled
@stable(FeatureName.SKILL_TOOLSET)
def create_skill_toolset_instance(): ...

# @working_in_progress: same guard, always default_on=False
@working_in_progress(FeatureName.IN_MEMORY_SESSION_SERVICE_LIGHT_COPY)
class _LightCopyVariant: ...
```

`@experimental` sets `default_on=False`; `@stable` sets `default_on=True`.
`@working_in_progress` always starts disabled.

---

## 4 · `PROGRESSIVE_SSE_STREAMING` in `StreamingResponseAggregator`

**Sources:** `google/adk/utils/streaming_utils.py`

### Two streaming modes

`StreamingResponseAggregator` has two code paths, selected by
`is_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING)` (default `True`
in 2.5.0):

| Mode | Behaviour |
|---|---|
| **Progressive** (new, default) | Accumulates parts in arrival order; text/FC/other parts interleave correctly; `close()` flushes all buffers |
| **Non-progressive** (legacy) | Concatenates text chunks; emits a merged text event only when a non-text chunk arrives |

### Progressive mode internals

```python
# In __init__:
self._parts_sequence: list[types.Part] = []
self._current_text_buffer: str = ''
self._current_text_is_thought: Optional[bool] = None
self._current_fc_name: Optional[str] = None
self._current_fc_args: dict[str, Any] = {}
self._current_fc_id: Optional[str] = None
self._current_thought_signature: Optional[bytes] = None
```

**Text accumulation:** Consecutive text chunks of the same type (thought vs
non-thought) are merged into `_current_text_buffer`. A type switch flushes the
buffer first via `_flush_text_buffer_to_sequence()`.

**Streaming function-call args via JSONPath:**

```python
def _process_streaming_function_call(self, fc: types.FunctionCall) -> None:
    # fc.partial_args: list of PartialArg with json_path + value
    for partial_arg in fc.partial_args or []:
        value, has_value = self._get_value_from_partial_arg(partial_arg, partial_arg.json_path)
        if has_value:
            self._set_value_by_json_path(partial_arg.json_path, value)
    if not fc.will_continue:
        self._flush_text_buffer_to_sequence()
        self._flush_function_call_to_sequence()
```

JSONPath strings like `$.location.latitude` navigate `_current_fc_args`
incrementally. String chunks are appended; other types overwrite.

**Final aggregation via `close()`:**

```python
def close(self) -> Optional[LlmResponse]:
    if is_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING):
        self._flush_text_buffer_to_sequence()
        self._flush_function_call_to_sequence()
        content = types.ModelContent(parts=self._parts_sequence) if self._parts_sequence else None
        return LlmResponse(content=content, ...)
```

### Example 1 — verify progressive ordering

```python
import asyncio
from unittest.mock import MagicMock
from google.adk.utils.streaming_utils import StreamingResponseAggregator
from google.adk.features import FeatureName, temporary_feature_override
from google.genai import types

async def collect_partial_responses(chunks):
    agg = StreamingResponseAggregator()
    partials = []
    for chunk in chunks:
        async for r in agg.process_response(chunk):
            partials.append(r)
    final = agg.close()
    return partials, final

with temporary_feature_override(FeatureName.PROGRESSIVE_SSE_STREAMING, True):
    # Build fake streaming chunks: thought → text → FC
    thought_chunk = types.GenerateContentResponse(candidates=[...])  # thought part
    text_chunk = types.GenerateContentResponse(candidates=[...])     # text part
    fc_chunk = types.GenerateContentResponse(candidates=[...])       # FC part
    partials, final = asyncio.run(collect_partial_responses([thought_chunk, text_chunk, fc_chunk]))

# final.content.parts preserves insertion order: [thought, text, fc]
assert [p.thought for p in final.content.parts] == [True, None, None]
```

### Example 2 — disable progressive mode (legacy text-merge)

```python
from google.adk.features import FeatureName, override_feature_enabled

# Revert to old concatenation behaviour (for migration testing)
override_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING, False)
```

### Example 3 — streaming FC with partial args

```python
# Simulate a streaming FC: two chunks building up {"location": "New York"}
from google.genai import types

fc_start = types.FunctionCall(
    name="get_weather",
    id="fc-123",
    partial_args=[
        types.PartialArg(json_path="$.location", string_value="New"),
    ],
    will_continue=True,
)
fc_end = types.FunctionCall(
    name="get_weather",
    id="fc-123",
    partial_args=[
        types.PartialArg(json_path="$.location", string_value=" York"),
    ],
    will_continue=False,
)
# After processing both: _current_fc_args == {"location": "New York"}
# _flush_function_call_to_sequence() creates Part.from_function_call(name="get_weather", args={"location": "New York"})
```

---

## 5 · `parse_edge_items` + workflow chain syntax

**Sources:** `google/adk/workflow/utils/_graph_parser.py`

### What it solves

`parse_edge_items` is the parser that converts the high-level chain syntax
supported by `Workflow` into a flat list of `Edge` objects. It handles three
`EdgeItem` types:

| Input type | Behaviour |
|---|---|
| `Edge(from_node, to_node, route)` | Passed through unchanged |
| `tuple` | Treated as a linear or branching chain |
| Anything else | `ValueError` |

### Chain syntax

A chain is a `tuple` of `ChainElement` values. Consecutive elements become
unconditional edges. A `RoutingMap` (plain `dict`) in position `i+1` creates
**conditional fan-out** edges from element `i`:

```
(node_a, node_b, {route_key: node_c})
```

`RouteValue` can be `str | int | bool`. Fan-out tuples `(node_c, node_d)` as
dict values create parallel edges from the same source.

### Example 1 — simple linear chain

```python
from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow

fetch = LlmAgent(name="fetch", model="gemini-2.5-flash", instruction="Fetch data")
process = LlmAgent(name="process", model="gemini-2.5-flash", instruction="Process")
store = LlmAgent(name="store", model="gemini-2.5-flash", instruction="Store")

wf = Workflow(
    name="pipeline",
    edges=[(fetch, process, store)],   # linear chain
)
```

This expands to:
- `Edge(START → fetch)`
- `Edge(fetch → process)`
- `Edge(process → store)`

### Example 2 — conditional routing

```python
classifier = LlmAgent(name="classifier", ...)
path_a = LlmAgent(name="path_a", ...)
path_b = LlmAgent(name="path_b", ...)
merge = LlmAgent(name="merge", ...)

wf = Workflow(
    name="branching",
    edges=[
        (classifier, {"A": path_a, "B": path_b}),
        (path_a, merge),
        (path_b, merge),
    ],
)
```

`classifier` must set `event.actions.route = "A"` or `"B"` to pick the branch.

### Example 3 — fan-out to multiple nodes on one route

```python
from google.adk.workflow._graph import Edge

# Both validator_1 and validator_2 run when classifier routes "VALIDATE"
wf = Workflow(
    name="fan_out",
    edges=[
        (classifier, {"VALIDATE": (validator_1, validator_2)}),
        (validator_1, aggregator),
        (validator_2, aggregator),
    ],
)
```

### Example 4 — explicit `Edge` objects mixed with chains

```python
from google.adk.workflow._graph import Edge
from google.adk.workflow._base_node import START

wf = Workflow(
    name="mixed",
    edges=[
        (fetch, process),             # chain
        Edge(from_node=process, to_node=store, route="ok"),    # explicit with route
        Edge(from_node=process, to_node=error_handler, route="error"),
    ],
)
```

---

## 6 · `ScenarioGenerator` — Vertex AI-powered eval scenario generation

**Sources:** `google/adk/evaluation/_vertex_ai_scenario_generation_facade.py`, `google/adk/evaluation/conversation_scenarios.py`

### What it does

`ScenarioGenerator` is a facade over the Vertex Gen AI Eval SDK that uses an LLM
to automatically generate `ConversationScenario` objects from your agent's
topology. It eliminates the need to hand-craft `starting_prompt` /
`conversation_plan` pairs for every test case.

### Auth

```python
class ScenarioGenerator:
    def __init__(self) -> None:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location   = os.environ.get("GOOGLE_CLOUD_LOCATION")
        api_key    = os.environ.get("GOOGLE_API_KEY")

        if api_key:
            self._client = vertexai.Client(api_key=api_key)
        elif project_id or location:
            # Both required; raises ValueError if either is missing
            self._client = vertexai.Client(project=project_id, location=location)
        else:
            raise ValueError("Either API Key or Google Cloud project+location required.")
```

### `generate_scenarios`

```python
def generate_scenarios(
    self,
    agent: BaseAgent,
    config: ConversationGenerationConfig,
) -> list[ConversationScenario]:
    agent_info = types.evals.AgentInfo.load_from_agent(agent=agent)
    vertex_config = types.evals.UserScenarioGenerationConfig(
        count=config.count,
        generation_instruction=config.generation_instruction,
        environment_context=config.environment_context,
        model_name=config.model_name,
    )
    eval_dataset = self._client.evals.generate_conversation_scenarios(
        agent_info=agent_info,
        config=vertex_config,
    )
    return [
        ConversationScenario(
            starting_prompt=case.user_scenario.starting_prompt,
            conversation_plan=case.user_scenario.conversation_plan,
        )
        for case in eval_dataset.eval_cases
        if case.user_scenario
    ]
```

`AgentInfo.load_from_agent` inspects the agent's instruction, description, and
tool declarations to give the LLM context for generating realistic scenarios.

### Example 1 — basic scenario generation

```python
import os
from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.evaluation._vertex_ai_scenario_generation_facade import ScenarioGenerator
from google.adk.evaluation.conversation_scenarios import ConversationGenerationConfig

os.environ["GOOGLE_CLOUD_PROJECT"] = "my-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

research_agent = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction="You are a research assistant. Use google_search to answer questions.",
    tools=[google_search],
)

generator = ScenarioGenerator()
config = ConversationGenerationConfig(
    count=5,
    generation_instruction="Generate diverse research questions about climate change.",
    model_name="gemini-2.5-pro",
)

scenarios = generator.generate_scenarios(agent=research_agent, config=config)
for s in scenarios:
    print(f"Prompt: {s.starting_prompt}")
    print(f"Plan:   {s.conversation_plan}\n")
```

### Example 2 — with environment context

```python
config = ConversationGenerationConfig(
    count=10,
    generation_instruction="Generate customer support scenarios for a banking app.",
    environment_context=(
        "The agent has access to account_lookup, transaction_history, "
        "and escalate_to_human tools."
    ),
    model_name="gemini-2.5-flash",
)
scenarios = generator.generate_scenarios(agent=bank_agent, config=config)
```

### Example 3 — wire scenarios into `evaluate_eval_set`

```python
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_set import EvalSet

# Pass the ConversationScenario directly so the multi-turn plan is preserved.
# Use reference-free multi-turn metrics — response_match_score requires a
# golden final_response, which generated scenarios don't provide.
eval_cases = [
    EvalCase(
        eval_id=f"auto_{i}",
        conversation_scenario=s,  # keeps conversation_plan intact
    )
    for i, s in enumerate(scenarios)
]

eval_set = EvalSet(
    eval_set_id="generated_banking_scenarios",
    name="Generated banking support scenarios",
    eval_cases=eval_cases,
)

# evaluate_eval_set accepts an in-memory EvalSet and an importable agent module path
AgentEvaluator.evaluate_eval_set(
    agent_module="my_package.agents.bank_agent",  # dotted import path to agent module
    eval_set=eval_set,
    num_runs=1,
)
```

---

## 7 · `FunctionNode` `parameter_binding='node_input'` + type coercions

**Sources:** `google/adk/workflow/_function_node.py`

### `parameter_binding` field

```python
class FunctionNode(BaseNode):
    parameter_binding: Literal['state', 'node_input'] = 'state'
```

| Value | Source of function arguments |
|---|---|
| `'state'` (default) | `ctx.state` — parameters read directly from session state by name |
| `'node_input'` | `node_input` dict passed to the node; ADK infers `input_schema`/`output_schema` from the function signature |

`'node_input'` makes the node behave like a tool: the caller passes a typed
dict and the function receives strongly-typed Pydantic arguments.

### Type coercions applied automatically

When `FunctionNode` binds parameters it applies these coercions via `TypeAdapter`:

| Received type | Annotation | Coerced to |
|---|---|---|
| `dict` | `BaseModel` subclass | `BaseModel.model_validate(dict)` |
| `list[dict]` | `list[BaseModel]` | list of validated model instances |
| `dict[K, dict]` | `dict[K, BaseModel]` | dict of validated model instances |
| `types.Content` | `str` | `" ".join(p.text for p in content.parts)` — non-text parts logged and dropped |
| anything else | any annotation | `TypeAdapter(annotation).validate_python(value)` |

`_PASSTHROUGH_OUTPUT_TYPES = (types.Content, Event, RequestInput)` — output
values of these types are emitted as-is without schema validation.

### Generator support

`FunctionNode` handles both sync generators and async generators:

```python
# Sync generator → wrapped with _sync_to_async_gen()
def streaming_processor(items: list[str]):
    for item in items:
        yield Event(author="processor", content=...)

# Async generator → used directly
async def async_pipeline(query: str):
    async for chunk in fetch_chunks(query):
        yield Event(author="pipeline", content=...)
```

### Example 1 — `parameter_binding='state'` (default)

```python
from google.adk.workflow._function_node import FunctionNode
from google.adk.workflow import Workflow
from google.adk.agents.context import Context

async def enrich_data(ctx: Context, item_id: str, count: int):
    """Reads item_id and count from ctx.state."""
    items = await fetch_items(item_id, count)
    ctx.state["enriched_items"] = items

enricher = FunctionNode(func=enrich_data, name="enricher")
```

`item_id` and `count` are read from `ctx.state["item_id"]` and
`ctx.state["count"]` automatically.

### Example 2 — `parameter_binding='node_input'` (tool-like)

```python
from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str
    max_results: int = 10

class SearchResult(BaseModel):
    hits: list[str]
    total: int

async def search(request: SearchRequest) -> SearchResult:
    results = await run_search(request.query, request.max_results)
    return SearchResult(hits=results, total=len(results))

search_node = FunctionNode(
    func=search,
    name="search",
    parameter_binding="node_input",
    # input_schema / output_schema auto-inferred from type hints
)
```

When `parameter_binding='node_input'` the dict `{"query": "ADK tutorial", "max_results": 5}`
is coerced to `SearchRequest` before calling `search`.

### Example 3 — auth gate

```python
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_schemes import OAuthGrantType, ExtendedOAuth2

oauth_config = AuthConfig(
    auth_scheme=ExtendedOAuth2(
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl="https://accounts.google.com/o/oauth2/auth",
                tokenUrl="https://oauth2.googleapis.com/token",
                scopes={"https://www.googleapis.com/auth/drive.readonly": "Read Drive"},
            )
        )
    ),
    raw_auth_credential=AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Credential(client_id="...", client_secret="..."),
    ),
)

drive_node = FunctionNode(
    func=list_drive_files,
    name="drive_lister",
    auth_config=oauth_config,   # interrupts first call if no credential in state
)
```

On first run, `drive_node` yields `adk_request_credential` and suspends.
After the user completes OAuth, it resumes with the credential available at
`AuthHandler(oauth_config).get_auth_response(ctx.state)`.

---

## 8 · `_RequestIntercepterPlugin` — LLM-request capture for eval

**Sources:** `google/adk/evaluation/request_intercepter_plugin.py`

<Aside type="caution">
This class is for ADK eval-system internal use only. Do not take a direct
dependency on it — it may change without notice. It is documented here to help
you understand the mechanism if you need to build similar instrumentation.
</Aside>

### How it works

The plugin intercepts every call to the model by coupling the `LlmRequest` with
the `LlmResponse` via a UUID in `custom_metadata`:

```python
class _RequestIntercepterPlugin(BasePlugin):
    def __init__(self, name: str):
        super().__init__(name=name)
        self._llm_requests_cache: dict[str, LlmRequest] = {}

    async def before_model_callback(self, *, callback_context, llm_request):
        request_id = str(uuid.uuid4())
        self._llm_requests_cache[request_id] = llm_request
        callback_context.state[_LLM_REQUEST_ID_KEY] = request_id  # "__llm_request_key__"
        return None   # do not short-circuit

    async def after_model_callback(self, *, callback_context, llm_response):
        if _LLM_REQUEST_ID_KEY in callback_context.state:
            if llm_response.custom_metadata is None:
                llm_response.custom_metadata = {}
            llm_response.custom_metadata[_LLM_REQUEST_ID_KEY] = (
                callback_context.state[_LLM_REQUEST_ID_KEY]
            )
        return None   # do not replace response

    def get_model_request(self, llm_response: LlmResponse) -> Optional[LlmRequest]:
        if llm_response.custom_metadata and _LLM_REQUEST_ID_KEY in llm_response.custom_metadata:
            request_id = llm_response.custom_metadata[_LLM_REQUEST_ID_KEY]
            return self._llm_requests_cache.get(request_id)
        return None
```

### Example 1 — build your own request-tracing plugin

```python
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
import uuid

class RequestTracingPlugin(BasePlugin):
    """Captures LlmRequest alongside each LlmResponse for debugging."""

    def __init__(self):
        super().__init__(name="request_tracer")
        self._cache: dict[str, LlmRequest] = {}
        self._REQUEST_KEY = "__trace_id__"

    async def before_model_callback(self, *, callback_context, llm_request):
        trace_id = str(uuid.uuid4())
        self._cache[trace_id] = llm_request
        callback_context.state[self._REQUEST_KEY] = trace_id
        return None

    async def after_model_callback(self, *, callback_context, llm_response):
        if self._REQUEST_KEY in callback_context.state:
            llm_response.custom_metadata = llm_response.custom_metadata or {}
            llm_response.custom_metadata[self._REQUEST_KEY] = (
                callback_context.state[self._REQUEST_KEY]
            )
        return None

    def lookup_request(self, llm_response: LlmResponse) -> LlmRequest | None:
        meta = llm_response.custom_metadata or {}
        return self._cache.get(meta.get(self._REQUEST_KEY))

# Plugins are registered on Runner, not LlmAgent
tracer = RequestTracingPlugin()
agent = LlmAgent(name="agent", model="gemini-2.5-flash")
runner = Runner(
    app_name="my_app",
    agent=agent,
    plugins=[tracer],
)

# After a run, retrieve the request for any response
async for event in runner.run_async(...):
    if event.is_final_response():
        req = tracer.lookup_request(event.llm_response)
        if req:
            print("System instruction:", req.config.system_instruction)
```

### Example 2 — inspect tools presented to the model

```python
from google.adk.runners import Runner

async def audit_tool_declarations(agent, user_message):
    """Log every tool declaration sent to the model."""
    tracer = RequestTracingPlugin()
    runner = Runner(
        app_name="audit_app",
        agent=agent,
        plugins=[tracer],
    )

    async for event in runner.run_async(
        user_id="audit_user",
        session_id="audit_session",
        new_message=user_message,
    ):
        if event.llm_response:
            req = tracer.lookup_request(event.llm_response)
            if req and req.tools:
                for tool_decl in req.tools:
                    print(f"Tool presented to model: {tool_decl.name}")
```

---

## 9 · `print_event` + verbose debug output

**Sources:** `google/adk/utils/_debug_output.py`

### Overview

`print_event` is the quick CLI debugging utility that formats an ADK `Event`
to stdout. Its design choices:

- **Default (`verbose=False`):** prints only text parts, prefixed with
  `{event.author} > `. Clean for watching the agent's final replies.
- **Verbose (`verbose=True`):** also prints tool calls, tool results, code
  execution, and file data — each on its own line with a bracket label.

### Truncation constants

```python
_ARGS_MAX_LEN     = 50   # tool call argument previews
_RESPONSE_MAX_LEN = 100  # tool response previews
_CODE_OUTPUT_MAX_LEN = 100
```

### Internal text-buffering pattern

The function accumulates consecutive text parts into a `text_buffer` list and
flushes it in one `print()` call before any non-text part, avoiding the
`author > ` prefix being printed multiple times for a single multi-part response:

```python
def flush_text() -> None:
    if text_buffer:
        combined_text = ''.join(text_buffer)
        print(f'{event.author} > {combined_text}')
        text_buffer.clear()
```

### Example 1 — quiet monitoring loop

```python
from google.adk.utils._debug_output import print_event

async for event in runner.run_async(
    user_id="u1",
    session_id="s1",
    new_message=user_content,
):
    print_event(event)           # shows only text (agent replies)
```

Output:
```
assistant > Here is your answer: climate change refers to ...
```

### Example 2 — verbose debugging

```python
async for event in runner.run_async(...):
    print_event(event, verbose=True)
```

Output:
```
assistant > [Calling tool: google_search({"query": "climate change 20...})]
assistant > [Tool result: {"results": [{"title": "Climate Ch...}]}]
assistant > Climate change refers to long-term shifts in temperatures and weather patterns.
```

### Example 3 — integration with multi-turn sessions

```python
import asyncio
from google.genai import types
from google.adk.utils._debug_output import print_event

questions = [
    "What is climate change?",
    "What are the main causes?",
    "What can individuals do?",
]

async def interactive_session():
    session = await runner.session_service.create_session(
        app_name="demo", user_id="user"
    )
    for q in questions:
        print(f"\nUser: {q}")
        async for event in runner.run_async(
            user_id="user",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=q)]),
        ):
            print_event(event, verbose=False)

asyncio.run(interactive_session())
```

### Example 4 — filter only final responses

```python
async for event in runner.run_async(...):
    if event.is_final_response():
        print_event(event)       # skip intermediate streaming chunks
```

---

## 10 · `LlmAgent.set_default_model` + `set_default_live_model`

**Sources:** `google/adk/agents/llm_agent.py`

### What they do

`set_default_model` and `set_default_live_model` are `@classmethod`s that
override the class-level fallback model used when an `LlmAgent` is constructed
without an explicit `model=` argument. The built-in default is
`gemini-3.5-flash`.

```python
@classmethod
def set_default_model(cls, model: Union[str, BaseLlm]) -> None:
    """Overrides the default model used when an agent has no model set."""
    if not isinstance(model, (str, BaseLlm)):
        raise TypeError('Default model must be a model name or BaseLlm.')
    if isinstance(model, str) and not model:
        raise ValueError('Default model must be a non-empty string.')
    cls._default_model = model

@classmethod
def set_default_live_model(cls, model: Union[str, BaseLlm]) -> None:
    """Overrides the default model used for live mode when an agent has no model set."""
    ...  # same validation
    cls._default_live_model = model
```

The stored value is resolved via `LLMRegistry.new_llm(model_string)` on first
access, making both string and `BaseLlm` instance forms equivalent at runtime.

### When to use

- **Multi-agent apps where all agents use the same model** — set once at
  startup rather than repeating `model=` on every `LlmAgent`.
- **Cost tier switching** — flip all agents to a cheaper model for testing
  without changing agent definitions.
- **Live mode separation** — use `gemini-2.5-flash-live` for streaming audio
  while keeping `gemini-2.5-pro` for text reasoning.

### Example 1 — app-wide default

```python
from google.adk.agents import LlmAgent

# Set once at app startup
LlmAgent.set_default_model("gemini-2.5-pro")

# All subsequent agents without explicit model= use gemini-2.5-pro
orchestrator = LlmAgent(name="orchestrator", instruction="Route requests.")
summariser  = LlmAgent(name="summariser",   instruction="Summarise results.")
validator   = LlmAgent(name="validator",    instruction="Validate outputs.")
```

### Example 2 — per-environment switching

```python
import os

if os.environ.get("ENV") == "production":
    LlmAgent.set_default_model("gemini-2.5-pro")
else:
    LlmAgent.set_default_model("gemini-2.5-flash")  # cheaper for dev/test

# Agent definitions stay unchanged across environments
agent = LlmAgent(name="worker", instruction="Do the task.")
```

### Example 3 — live vs text default split

```python
from google.adk.agents import LlmAgent

LlmAgent.set_default_model("gemini-2.5-pro")
LlmAgent.set_default_live_model("gemini-2.5-flash-live")

# Text agent uses gemini-2.5-pro
text_agent = LlmAgent(name="text_worker", instruction="Answer questions.")

# Live agent uses gemini-2.5-flash-live (for real-time audio)
live_agent = LlmAgent(name="voice_worker", instruction="Handle voice input.")
```

### Example 4 — pass a `BaseLlm` instance

```python
from google.adk.models.lite_llm import LiteLlm

LlmAgent.set_default_model(LiteLlm(model="openai/gpt-4o"))

# All agents without explicit model= now route through GPT-4o via LiteLLM
agent = LlmAgent(name="openai_agent", instruction="Use GPT-4o.")
```

### Example 5 — reset to ADK default

```python
# Restore the built-in default (gemini-3.5-flash)
LlmAgent.set_default_model("gemini-3.5-flash")
LlmAgent.set_default_live_model("gemini-2.0-flash-live-001")
```

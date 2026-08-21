---
title: "Class deep dives — volume 48 (ReadonlyContext, ActiveStreamingTool, TranscriptionEntry, JoinNode, Trigger, DynamicNodeScheduler, ToolboxToolset, UrlContextTool, OpenAPIToolset, A2aAgentExecutor + A2aAgentExecutorConfig + ExecuteInterceptor)"
description: "10 source-verified deep dives for google-adk 2.7.0: ReadonlyContext (MappingProxyType state; get_credential by key; custom_metadata read-only view; TYPE_CHECKING-only imports), ActiveStreamingTool (Pydantic BaseModel; asyncio.Task + LiveRequestQueue pair; arbitrary_types_allowed; extra='forbid'; streaming tool lifecycle), TranscriptionEntry (role nullable for function calls; Blob|Content union; arbitrary_types_allowed), JoinNode (_requires_all_predecessors=True; dict-keyed input_schema validation; pass-through yield; barrier semantics in Workflow), Trigger (input/use_sub_branch/branch/isolation_scope; ser_json_bytes base64; routing data model for node edges), DynamicNodeScheduler (ScheduleDynamicNode protocol; DynamicNodeRun/DynamicNodeState; dedup/resume/fresh three-phase logic; ReplayManager chronological barrier), ToolboxToolset (toolbox-adk delegate; server_url + toolset_name + tool_names + auth_token_getters + bound_params + credentials + additional_headers; lazy get_tools), UrlContextTool (Gemini 2 built-in; process_llm_request injects types.Tool(url_context); model-check guard; singleton url_context), OpenAPIToolset (spec_dict/spec_str/yaml parsing; ssl_verify + header_provider + httpx_client_factory; preserve_property_names; credential_key; configure_ssl_verify_all), A2aAgentExecutor + A2aAgentExecutorConfig + ExecuteInterceptor (@a2a_experimental; Runner callable resolution; legacy/new version selection via extension; execute_before/after hooks; TaskResultAggregator; ExecuteInterceptor before_agent/after_event/after_agent)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 48"
  order: 117
---

import { Aside } from "@astrojs/starlight/components";

<Aside type="note">
All signatures, constants, and behaviours on this page were verified directly
against the installed package source (locate yours with
`python -c 'import google.adk; print(google.adk.__file__)'`) on
**google-adk == 2.7.0**. No documentation or blog posts were used as primary
sources.
</Aside>

---

## 1 · `ReadonlyContext` — safe read-only view of invocation state

**Source:** `google/adk/agents/readonly_context.py`

### Why it matters

Callbacks, `InstructionProvider` implementations, and toolset `get_tools()`
methods receive a `ReadonlyContext` rather than the full
`InvocationContext`. The wrapper enforces read-only access at the API level —
`state` is exposed as a `MappingProxyType` so mutations raise `TypeError`
rather than silently corrupting session state.

### Internals

```python
class ReadonlyContext:
    def __init__(self, invocation_context: InvocationContext) -> None:
        self._invocation_context = invocation_context

    @property
    def user_content(self) -> types.Content | None: ...   # original user turn
    @property
    def invocation_id(self) -> str: ...                   # unique per run_async() call
    @property
    def agent_name(self) -> str: ...                      # "unknown" when agent is None
    @property
    def state(self) -> MappingProxyType[str, Any]: ...    # READ-ONLY session state
    @property
    def session(self) -> Session: ...                     # full session object
    @property
    def user_id(self) -> str: ...
    @property
    def run_config(self) -> RunConfig | None: ...
    @property
    def custom_metadata(self) -> Mapping[str, Any]: ...   # read-only view of _custom_metadata

    def get_credential(self, key: str) -> AuthCredential | None: ...
```

All imports of heavier types (`InvocationContext`, `Session`, `RunConfig`,
`AuthCredential`) live inside `if TYPE_CHECKING:` blocks — no circular-import
risk and zero runtime cost for types that are not used.

### Example 1 — dynamic tool filtering based on session state

```python
from typing import Optional
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.base_tool import BaseTool
from google.adk.agents.readonly_context import ReadonlyContext

class RoleBasedToolset(BaseToolset):
    """Only expose admin tools when the session marks the user as an admin."""

    def __init__(self, all_tools: list[BaseTool], admin_tools: list[BaseTool]):
        super().__init__()   # required: initialises invocation cache and tool filter
        self._all_tools = all_tools
        self._admin_tools = admin_tools

    async def get_tools(
        self, readonly_context: Optional[ReadonlyContext] = None
    ) -> list[BaseTool]:
        if readonly_context is None:
            return self._all_tools
        # state is MappingProxyType — reads work, writes raise TypeError
        is_admin = readonly_context.state.get("role") == "admin"
        return self._all_tools + (self._admin_tools if is_admin else [])

    async def close(self) -> None:
        pass
```

### Example 2 — credential lookup in an `InstructionProvider`

`InstructionProvider` is a type alias for
`Callable[[ReadonlyContext], str | Awaitable[str]]` — any async (or sync)
function that accepts a `ReadonlyContext` and returns a string. The HTTP
credential token lives at `cred.http.credentials.token` (not `cred.http.token`).

```python
from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext

async def api_key_instruction(ctx: ReadonlyContext) -> str:
    """Confirm whether a credential is configured — without exposing any token value."""
    cred = ctx.get_credential("my_api_service")
    configured = (
        cred is not None
        and cred.http is not None
        and cred.http.credentials is not None
        and bool(cred.http.credentials.token)
    )
    status = "configured" if configured else "not yet resolved"
    return (
        f"You are a helpful assistant. "
        f"The API service credential is {status}. "
        "Use it for any external calls that require authentication."
    )

# Wire the callable directly as instruction= — it is called each turn.
agent = LlmAgent(
    name="api_agent",
    model="gemini-2.5-flash",
    instruction=api_key_instruction,
)
```

### Example 3 — reading custom metadata in a `before_tool_callback`

ADK calls `before_tool_callback` with three arguments: the tool being invoked,
its resolved argument dict, and a `ToolContext`. `ToolContext` inherits
`ReadonlyContext` and exposes `custom_metadata`.

```python
from typing import Any, Optional
from google.adk.agents import LlmAgent
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

def log_invocation(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> Optional[dict[str, Any]]:
    meta = tool_context.custom_metadata   # Mapping[str, Any], not mutable
    request_id = meta.get("request_id", "unknown")
    print(f"[{tool_context.invocation_id}] tool={tool.name} request_id={request_id}")
    return None  # returning None lets the tool run normally

agent = LlmAgent(
    name="logged_agent",
    model="gemini-2.5-flash",
    instruction="Answer helpfully.",
    before_tool_callback=log_invocation,
)
```

---

## 2 · `ActiveStreamingTool` — streaming tool resource container

**Source:** `google/adk/agents/active_streaming_tool.py`

### Why it matters

When ADK runs a streaming tool alongside a live agent session (via
`Runner.run_live()`), it needs to track two resources per active invocation:
the background `asyncio.Task` that runs the tool coroutine, and the
`LiveRequestQueue` through which the agent feeds realtime input to that tool.
`ActiveStreamingTool` is that container.

### Internals

```python
class ActiveStreamingTool(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,   # needed for asyncio.Task
        extra='forbid',
    )

    task: Optional[asyncio.Task[Any]] = None
    """Background asyncio task executing the streaming tool."""

    stream: Optional[LiveRequestQueue] = None
    """Queue through which the live agent sends data to the tool."""
```

Both fields are optional because the container may be created before the
task or queue is ready, and `extra='forbid'` prevents accidental field
additions that would silently be ignored.

### Example 1 — building a streaming tool that accepts realtime input

ADK passes the `LiveRequestQueue` for input-streaming tools via a parameter
named `input_stream` (detected by type annotation). The runner creates and
registers the `ActiveStreamingTool` automatically; your function just declares
the parameter.

```python
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.tools.tool_context import ToolContext

async def live_transcription_tool(
    input_stream: LiveRequestQueue,
    tool_context: ToolContext,
):
    """A streaming tool that receives audio blobs and yields transcriptions.

    ADK inspects the signature, detects `input_stream: LiveRequestQueue`,
    creates a dedicated queue, registers an ActiveStreamingTool(task=...,
    stream=queue), and starts feeding data into the queue from the live
    model stream.
    """
    transcript_parts = []

    while True:
        req = await input_stream.get()
        if req.close:
            break
        if req.blob:
            # In a real app: call a speech-to-text API here.
            transcript_parts.append(f"[audio:{len(req.blob.data)}bytes]")

    return {"transcript": " ".join(transcript_parts)}
```

### Example 2 — checking task liveness before cancellation

```python
from google.adk.agents.active_streaming_tool import ActiveStreamingTool

def cancel_if_running(active: ActiveStreamingTool) -> None:
    """Cancel the background task only if it is still running."""
    if active.task and not active.task.done():
        active.task.cancel()
    if active.stream:
        active.stream.close()  # enqueues the sentinel LiveRequest(close=True)
```

---

## 3 · `TranscriptionEntry` — audio/video transcription data record

**Source:** `google/adk/agents/transcription_entry.py`

### Why it matters

Live agents that process audio or video need a typed container to accumulate
raw blobs alongside model-generated `Content` objects for later transcription.
`TranscriptionEntry` provides that container, keeping `role` nullable for
function-call contributions where no conversational role applies.

### Internals

```python
class TranscriptionEntry(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='forbid',
    )

    role: Optional[str] = None
    # "user" / "model" for speech turns; None for function calls.

    data: Union[types.Blob, types.Content]
    # Raw audio/video blob OR a structured Content object.
```

The `Union[Blob, Content]` discriminator is resolved by Pydantic's standard
left-to-right matching — pass a `Blob` for raw media, a `Content` for text or
mixed-modal content.

### Example 1 — accumulating a live session transcript

```python
from google.genai import types
from google.adk.agents.transcription_entry import TranscriptionEntry

transcript: list[TranscriptionEntry] = []

# User audio chunk received from microphone
transcript.append(TranscriptionEntry(
    role="user",
    data=types.Blob(mime_type="audio/pcm;rate=16000", data=b"\x00" * 3200),
))

# Model text response
transcript.append(TranscriptionEntry(
    role="model",
    data=types.Content(
        role="model",
        parts=[types.Part(text="I heard you say hello.")],
    ),
))

# Function call — role is None
transcript.append(TranscriptionEntry(
    role=None,
    data=types.Content(
        parts=[types.Part(function_call=types.FunctionCall(
            name="search_web", args={"query": "ADK docs"}
        ))],
    ),
))

print(f"Entries: {len(transcript)}")  # 3
```

### Example 2 — serialising and deserialising a transcript

```python
import json
from google.genai import types
from google.adk.agents.transcription_entry import TranscriptionEntry

entry = TranscriptionEntry(
    role="user",
    data=types.Blob(mime_type="audio/pcm", data=b"\x01\x02\x03"),
)

# model_dump_json serialises the entry; how binary data is encoded depends
# on google.genai's Blob implementation — test with your own payload.
payload = entry.model_dump_json()

# Round-trip
restored = TranscriptionEntry.model_validate_json(payload)
assert restored.role == "user"
```

---

## 4 · `JoinNode` — all-predecessors barrier node

**Source:** `google/adk/workflow/_join_node.py`

### Why it matters

In a Workflow graph, a `JoinNode` acts as a synchronisation barrier: it only
runs after **every** predecessor edge has delivered its output. This is the
natural counterpart to fan-out — fan back in with `JoinNode`.

### Internals

```python
class JoinNode(BaseNode):
    @property
    @override
    def _requires_all_predecessors(self) -> bool:
        return True          # overrides BaseNode's default of False

    @override
    def _validate_input_data(self, data: Any) -> Any:
        # When input_schema is set, validates EACH predecessor's contribution
        # (keyed by branch string) independently.
        if self.input_schema and isinstance(data, dict):
            return {
                k: self._validate_schema(v, self.input_schema)
                for k, v in data.items()
            }
        return super()._validate_input_data(data)

    @override
    async def _run_impl(self, *, ctx: Context, node_input: Any) -> AsyncGenerator:
        # Simply yields the aggregated dict of predecessor outputs.
        yield Event(output=node_input, branch=ctx._invocation_context.branch)
```

The aggregated `node_input` is a `dict` keyed by **branch identifiers** when
multiple predecessors are present. In the common case (no custom routing), a
branch identifier equals the predecessor node's name, which is why the example
below uses keys like `"fetch_weather"`. If a node uses `Trigger(branch=…)` or
sub-branch routing the key will differ from the node name, so always inspect
the actual branch strings in production code rather than assuming they match
names. The `JoinNode` itself does no transformation — it passes the collected
dict straight through for downstream nodes to consume.

### Example 1 — fan-out / fan-in workflow

```python
from google.adk.workflow import Workflow
from google.adk.workflow._base_node import START
from google.adk.workflow._join_node import JoinNode
from google.adk.workflow._function_node import FunctionNode

async def fetch_weather(node_input):
    return {"weather": "sunny", "city": node_input["city"]}

async def fetch_news(node_input):
    return {"headlines": ["ADK 2.7 ships", "AI news"], "city": node_input["city"]}

async def combine_results(node_input):
    # node_input is a dict keyed by branch identifier (usually node name).
    return {
        "weather": node_input.get("fetch_weather", {}).get("weather"),
        "headlines": node_input.get("fetch_news", {}).get("headlines"),
    }

# parameter_binding='node_input' binds input-dict *keys* to matching
# function-parameter names (e.g. city= from {"city": "Paris"}).
# Functions that accept a single `node_input` arg use the default 'state'
# binding, where the entire incoming payload is passed through directly.
fetch_weather_node = FunctionNode(name="fetch_weather", func=fetch_weather)
fetch_news_node    = FunctionNode(name="fetch_news",    func=fetch_news)
join               = JoinNode(name="join")
combine_node       = FunctionNode(name="combine",       func=combine_results)

workflow = Workflow(
    name="briefing",
    edges=[
        # Fan out from START to both fetch nodes in parallel.
        (START, (fetch_weather_node, fetch_news_node)),
        (fetch_weather_node, join),
        (fetch_news_node,    join),
        (join,               combine_node),
    ],
)
```

### Example 2 — typed join with input_schema validation

```python
from pydantic import BaseModel
from google.adk.workflow._join_node import JoinNode

class BranchResult(BaseModel):
    score: float
    label: str

# JoinNode validates each branch's dict value against BranchResult
typed_join = JoinNode(
    name="typed_join",
    input_schema=BranchResult,
)
```

---

## 5 · `Trigger` — routing data model for workflow edges

**Source:** `google/adk/workflow/_trigger.py`

### Why it matters

`Trigger` is the internal data model that the workflow orchestrator passes
along edges between nodes. It carries the output payload, routing metadata
(`branch`), and isolation hints (`use_sub_branch`, `isolation_scope`).
Understanding `Trigger` explains how `ctx.route` and sub-branch behaviour work.

### Internals

```python
class Trigger(BaseModel):
    model_config = ConfigDict(ser_json_bytes='base64')

    input: Any = None
    """Payload forwarded to the downstream node."""

    use_sub_branch: bool = False
    """When True the downstream node executes in its own sub-branch,
    isolating its event history from the parent branch."""

    branch: str | None = None
    """Branch string inherited from the predecessor node."""

    isolation_scope: str | None = None
    """Scope tag explicitly propagated to this trigger for partitioned
    state isolation."""
```

Bytes embedded in `input` are base64-serialised by `ser_json_bytes='base64'`
during session persistence.

### Example 1 — sub-branch isolation for parallel fan-out

```python
from google.adk.workflow._trigger import Trigger

# The orchestrator builds triggers internally, but you can inspect them
# from within a FunctionNode via ctx.route:
from google.adk.agents.context import Context

async def router_node(node_input, ctx: Context):
    queries = node_input["queries"]
    # Return a list of Triggers to fan out to N sub-branches
    return [
        Trigger(input={"query": q}, use_sub_branch=True)
        for q in queries
    ]
```

### Example 2 — isolation scope for partitioned state

```python
from google.adk.workflow._trigger import Trigger

# Scope keeps each tenant's state partition separate during parallel runs.
def make_tenant_trigger(tenant_id: str, payload: dict) -> Trigger:
    return Trigger(
        input=payload,
        use_sub_branch=True,
        isolation_scope=f"tenant:{tenant_id}",
    )
```

---

## 6 · `DynamicNodeScheduler` — `ctx.run_node()` implementation

**Source:** `google/adk/workflow/_dynamic_node_scheduler.py`

### Why it matters

`ctx.run_node()` lets a `FunctionNode` or `ToolNode` schedule child nodes at
runtime rather than through static edges. `DynamicNodeScheduler` is the
implementation behind that call. It handles three cases:

1. **Fresh** — no prior events; run the node normally.
2. **Completed** — prior session events show the node already finished;
   return cached output instantly (deduplication).
3. **Waiting** — prior events show the node interrupted; resolve or propagate
   those interrupts.

### Key types

```python
@dataclass(kw_only=True)
class DynamicNodeRun:
    state: NodeState          # status, interrupts, run_id
    output: Any = None        # populated on completion
    task: asyncio.Task | None = None
    transfer_to_agent: str | None = None
    recovered_state: _ChildScanState | None = None  # from session events

@dataclass(kw_only=True)
class DynamicNodeState:
    runs: dict[str, DynamicNodeRun] = field(default_factory=dict)
    # keyed by full node_path, e.g. "/my_workflow@1/child_node@1"

    interrupt_ids: set[str] = field(default_factory=set)
    replay_manager: ReplayManager = field(default_factory=ReplayManager)

class DynamicNodeScheduler(ScheduleDynamicNode):
    def __init__(self, *, state: DynamicNodeState) -> None: ...

    async def __call__(
        self,
        ctx: Context,
        node: BaseNode,
        node_input: Any,
        *,
        node_name: str | None = None,
        use_as_output: bool = False,
        run_id: str,
        use_sub_branch: bool = False,
        override_branch: str | None = None,
        override_isolation_scope: str | None = None,
    ) -> Context: ...
```

### Example 1 — dynamic child node from a `FunctionNode`

```python
import asyncio
from google.adk.workflow import Workflow
from google.adk.workflow._base_node import START
from google.adk.workflow._function_node import FunctionNode
from google.adk.agents.context import Context

async def search(node_input):
    return {"results": [f"result for {node_input['query']}"]}

search_node = FunctionNode(name="search", func=search)

async def orchestrate(node_input, ctx: Context):
    queries = node_input.get("queries", [])
    results = []
    for i, q in enumerate(queries):
        child_ctx = await ctx.run_node(
            search_node,
            node_input={"query": q},
            run_id=str(i),   # deterministic for resume
        )
        results.append(child_ctx)
    return {"all_results": results}

# rerun_on_resume=True is required when a FunctionNode calls ctx.run_node().
orchestrator = FunctionNode(
    name="orchestrate", func=orchestrate, rerun_on_resume=True,
)
# Every workflow must have a START edge to seed the initial nodes.
workflow = Workflow(name="multi_search", edges=[(START, orchestrator)])
```

### Example 2 — `use_as_output` to propagate child output

```python
async def delegate(node_input, ctx: Context):
    # Child node's output REPLACES this node's output (use_as_output=True).
    await ctx.run_node(
        some_llm_node,
        node_input=node_input,
        run_id="primary",
        use_as_output=True,
    )
    # No explicit return needed — the child's output is already registered.
```

### Example 3 — deduplication on resume

```python
# On resume (e.g. after a HITL interrupt), DynamicNodeScheduler scans
# session events and finds the completed child. It fast-forwards by
# returning a mock Context with the cached output — no re-execution.
# This is completely automatic; your FunctionNode code is unchanged.

async def safe_idempotent_node(node_input, ctx: Context):
    # Calling run_node with the same run_id on resume = cached result.
    child = await ctx.run_node(
        expensive_node,
        node_input=node_input,
        run_id="step-1",
    )
    return child
```

---

## 7 · `ToolboxToolset` — MCP Toolbox bridge

**Source:** `google/adk/tools/toolbox_toolset.py`

### Why it matters

[MCP Toolbox for Databases](https://github.com/googleapis/mcp-toolbox-sdk-python)
is Google's server-side tool proxy that manages connection pools, auth, and
schema validation for database tools. `ToolboxToolset` is the thin ADK adapter
that wraps `toolbox_adk.ToolboxToolset` so you can add Toolbox tools to any
ADK agent with a single call.

Install the extras: `pip install google-adk[toolbox]`

### Internals

```python
class ToolboxToolset(BaseToolset):
    def __init__(
        self,
        server_url: str,
        toolset_name: Optional[str] = None,
        tool_names: Optional[List[str]] = None,
        auth_token_getters: Optional[Mapping[str, Callable[[], str]]] = None,
        bound_params: Optional[Mapping[str, Union[Callable[[], Any], Any]]] = None,
        credentials: Optional[CredentialConfig] = None,
        additional_headers: Optional[Mapping[str, str]] = None,
        **kwargs,
    ): ...

    async def get_tools(
        self, readonly_context: Optional[ReadonlyContext] = None
    ) -> list[BaseTool]: ...

    async def close(self): ...
```

`get_tools()` and `close()` are fully delegated to the underlying
`toolbox_adk.ToolboxToolset`. If the extra is not installed an `ImportError`
with a clear message is raised at construction time, not at call time.

When both `toolset_name` and `tool_names` are omitted, **all** tools are
loaded from the server.

### Example 1 — basic database toolset

```python
from google.adk.agents import LlmAgent
from google.adk.tools.toolbox_toolset import ToolboxToolset
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

toolset = ToolboxToolset(
    server_url="http://127.0.0.1:5000",
    toolset_name="my-postgres-toolset",
)

agent = LlmAgent(
    name="db_agent",
    model="gemini-2.5-flash",
    instruction="Answer questions about the database.",
    tools=[toolset],
)

runner = Runner(
    app_name="db_app",
    agent=agent,
    session_service=InMemorySessionService(),
)
```

### Example 2 — specific tools with auth token getters

```python
from google.adk.tools.toolbox_toolset import ToolboxToolset
import google.auth.transport.requests
import google.oauth2.id_token

TOOLBOX_URL = "https://toolbox.example.com"

def get_google_id_token() -> str:
    # Mint an audience-bound OIDC ID token — not the OAuth access token from
    # google.auth.default(), which Toolbox OIDC auth will reject.
    auth_req = google.auth.transport.requests.Request()
    return google.oauth2.id_token.fetch_id_token(auth_req, audience=TOOLBOX_URL)

toolset = ToolboxToolset(
    server_url=TOOLBOX_URL,
    tool_names=["search_orders", "get_inventory"],
    auth_token_getters={"google": get_google_id_token},
)
```

### Example 3 — bound parameters for context injection

```python
from google.adk.tools.toolbox_toolset import ToolboxToolset

def get_current_user_id() -> str:
    # In a real app this would read from session state.
    return "user_42"

toolset = ToolboxToolset(
    server_url="http://127.0.0.1:5000",
    bound_params={
        "user_id": get_current_user_id,   # called fresh on each tool invoke
        "region": "us-central1",          # static value
    },
)
```

---

## 8 · `UrlContextTool` — Gemini 2 URL grounding built-in

**Source:** `google/adk/tools/url_context_tool.py`

### Why it matters

`UrlContextTool` activates Gemini 2's built-in URL context retrieval: the
model can fetch and read URLs it encounters in a conversation without any
local code execution. Add it to an agent's `tools` list and Gemini will
automatically call it when it determines a URL should be read.

### Internals

```python
class UrlContextTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(name='url_context', description='url_context')
        # name/description unused — this is a model built-in

    async def process_llm_request(
        self,
        *,
        tool_context: ToolContext,
        llm_request: LlmRequest,
    ) -> None:
        llm_request.config = llm_request.config or types.GenerateContentConfig()
        llm_request.config.tools = llm_request.config.tools or []
        if is_gemini_model(llm_request.model) or _is_managed_agent(llm_request):
            llm_request.config.tools.append(
                types.Tool(url_context=types.UrlContext())
            )
        else:
            raise ValueError(
                f'Url context tool is not supported for model {llm_request.model}'
            )

# Module-level singleton — import and use directly:
url_context = UrlContextTool()
```

The model check (`is_gemini_model`) can be bypassed via the
`GOOGLE_ADK_DISABLE_GEMINI_MODEL_ID_CHECK` environment variable for testing
with custom model IDs.

### Example 1 — agent that reads URLs on demand

```python
from google.adk.agents import LlmAgent
from google.adk.tools.url_context_tool import url_context

agent = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction=(
        "You are a research assistant. When the user provides URLs, "
        "read and analyse their content to give accurate, detailed answers."
    ),
    tools=[url_context],
)
```

### Example 2 — combining with Google Search

```python
from google.adk.agents import LlmAgent
from google.adk.tools.url_context_tool import url_context
from google.adk.tools.google_search_tool import google_search

agent = LlmAgent(
    name="web_agent",
    model="gemini-2.5-flash",
    instruction="Search the web and read linked pages to answer questions.",
    tools=[google_search, url_context],
)
```

### Example 3 — non-Gemini model guard

`UrlContextTool` checks the model inside `process_llm_request`, not at agent
construction time. Calling the method directly lets you observe the guard
without needing a registered LLM backend:

```python
import asyncio
from google.adk.models import LlmRequest
from google.adk.tools.url_context_tool import UrlContextTool

tool = UrlContextTool()

async def check_guard():
    req = LlmRequest(model="gpt-4o")   # any non-Gemini model string
    try:
        # tool_context is not used by this check, so None is fine here.
        await tool.process_llm_request(tool_context=None, llm_request=req)
    except ValueError as e:
        print(e)
        # "Url context tool is not supported for model gpt-4o"

asyncio.run(check_guard())
```

---

## 9 · `OpenAPIToolset` — turn any OpenAPI spec into ADK tools

**Source:** `google/adk/tools/openapi_tool/openapi_spec_parser/openapi_toolset.py`

### Why it matters

`OpenAPIToolset` parses an OpenAPI 3.x specification (JSON or YAML) and
generates one `RestApiTool` per operation. Each generated tool is a fully
functional ADK tool with auth, SSL, and dynamic headers baked in. This lets
you expose any REST API to an agent without writing individual tool functions.

### Internals

```python
class OpenAPIToolset(BaseToolset):
    def __init__(
        self,
        *,
        spec_dict: Optional[Dict[str, Any]] = None,
        spec_str: Optional[str] = None,
        spec_str_type: Literal["json", "yaml"] = "json",
        auth_scheme: Optional[AuthScheme] = None,
        auth_credential: Optional[AuthCredential] = None,
        credential_key: Optional[str] = None,
        tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
        tool_name_prefix: Optional[str] = None,
        ssl_verify: Optional[Union[bool, str, ssl.SSLContext]] = None,
        header_provider: Optional[Callable[[ReadonlyContext], Dict[str, str]]] = None,
        httpx_client_factory: Optional[HttpxClientFactory] = None,
        preserve_property_names: bool = False,
    ): ...

    async def get_tools(
        self, readonly_context: Optional[ReadonlyContext] = None
    ) -> List[RestApiTool]: ...

    def get_tool(self, tool_name: str) -> Optional[RestApiTool]: ...

    def configure_ssl_verify_all(
        self, ssl_verify: Optional[Union[bool, str, ssl.SSLContext]] = None
    ) -> None: ...
```

Tools are parsed eagerly at construction time — `get_tools()` only applies the
`tool_filter` predicate. Property names are converted to `snake_case` by
default; set `preserve_property_names=True` to keep the original names (useful
for camelCase APIs).

### Example 1 — parse a JSON spec

```python
import json
from google.adk.agents import LlmAgent
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

with open("petstore.json") as f:
    spec_str = f.read()

toolset = OpenAPIToolset(spec_str=spec_str, spec_str_type="json")
agent = LlmAgent(
    name="pet_agent",
    model="gemini-2.5-flash",
    instruction="Help users manage pets in the pet store.",
    tools=[toolset],
)
```

### Example 2 — YAML spec with bearer-token auth

```python
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from google.adk.tools.openapi_tool.auth.auth_helpers import token_to_scheme_credential

# token_type is the only positional arg; pass the rest as keyword arguments.
scheme, credential = token_to_scheme_credential(
    "oauth2Token",
    location="header",
    name="Authorization",
    credential_value="my-bearer-token",
)

toolset = OpenAPIToolset(
    spec_str=open("api.yaml").read(),
    spec_str_type="yaml",
    auth_scheme=scheme,
    auth_credential=credential,
)
```

### Example 3 — corporate TLS proxy with custom CA

```python
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

toolset = OpenAPIToolset(
    spec_dict=my_spec_dict,
    ssl_verify="/etc/ssl/certs/corporate-ca.pem",  # path to CA bundle
)

# Or swap the CA after construction:
toolset.configure_ssl_verify_all("/etc/ssl/certs/new-ca.pem")
```

### Example 4 — dynamic headers from session context

```python
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

def correlation_header_provider(ctx: ReadonlyContext) -> dict[str, str]:
    return {
        "X-Request-ID": ctx.invocation_id,
        "X-User-ID": ctx.user_id,
    }

toolset = OpenAPIToolset(
    spec_dict=my_spec_dict,
    header_provider=correlation_header_provider,
)
```

### Example 5 — subset of operations via `tool_filter`

```python
toolset = OpenAPIToolset(
    spec_dict=my_spec_dict,
    tool_filter=["list_pets", "create_pet"],   # only expose these two operations
    tool_name_prefix="petstore",              # avoid name collisions
)
# get_tools_with_prefix adds "_" automatically: "petstore_list_pets", "petstore_create_pet"
```

---

## 10 · `A2aAgentExecutor` + `A2aAgentExecutorConfig` + `ExecuteInterceptor` — ADK over A2A protocol

**Sources:** `google/adk/a2a/executor/a2a_agent_executor.py`,
`google/adk/a2a/executor/config.py`

### Why it matters

The [Agent-to-Agent (A2A) protocol](https://github.com/google/a2a) defines
how autonomous agents discover and call each other. `A2aAgentExecutor` adapts
any ADK `Runner` to the `a2a.server.agent_execution.AgentExecutor` interface
so it can be hosted in an A2A server, while `A2aAgentExecutorConfig` lets you
swap converter functions and attach lifecycle interceptors without subclassing.

Both classes are decorated with `@a2a_experimental`.

### Internals — `A2aAgentExecutor`

```python
@a2a_experimental
class A2aAgentExecutor(AgentExecutor):
    def __init__(
        self,
        *,
        runner: Runner | Callable[..., Runner | Awaitable[Runner]],
        config: A2aAgentExecutorConfig | None = None,
        use_legacy: bool = False,
        force_new_version: bool = False,
    ): ...

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None: ...

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None: ...
```

`runner` may be a live `Runner` instance **or** a zero/one-argument callable
(sync or async) that returns one. The callable form is cached after first
resolution — useful for deferred or async initialisation. The
`use_legacy` / `force_new_version` flags control which internal executor
implementation is activated; normally leave both as `False` and let the
`_NEW_A2A_ADK_INTEGRATION_EXTENSION` header decide.

The execution flow:

1. Resolve the runner (cache on first call).
2. Convert the A2A `RequestContext` → `AgentRunRequest` via `config.request_converter`.
3. Ensure the session exists; create if absent.
4. Publish `working` status to the event queue.
5. Drain `runner.run_async()` and convert each ADK event → A2A events via
   `config.event_converter`.
6. Publish the final `completed` / `failed` / `input_required` status.

### Internals — `A2aAgentExecutorConfig` and `ExecuteInterceptor`

```python
@a2a_experimental
class A2aAgentExecutorConfig(BaseModel):
    a2a_part_converter: A2APartToGenAIPartConverter = convert_a2a_part_to_genai_part
    gen_ai_part_converter: GenAIPartToA2APartConverter = convert_genai_part_to_a2a_part
    request_converter: A2ARequestToAgentRunRequestConverter = (
        convert_a2a_request_to_agent_run_request
    )
    event_converter: AdkEventToA2AEventsConverter = legacy_convert_event_to_a2a_events
    adk_event_converter: AdkEventToA2AEventsConverterImpl = convert_event_to_a2a_events_impl
    execute_interceptors: Optional[list[ExecuteInterceptor]] = None

@dataclasses.dataclass
class ExecuteInterceptor:
    before_agent: Optional[
        Callable[[RequestContext], Awaitable[RequestContext]]
    ] = None
    # Inspect/mutate the incoming request before the agent runs.

    after_event: Optional[
        Callable[
            [ExecutorContext, A2AEvent, Event],
            Awaitable[Union[A2AEvent, list[A2AEvent], None]],
        ]
    ] = None
    # Mutate or drop (return None) each outgoing A2A event.

    after_agent: Optional[
        Callable[
            [ExecutorContext, TaskStatusUpdateEvent],
            Awaitable[TaskStatusUpdateEvent],
        ]
    ] = None
    # Inspect/mutate the final terminal status event.
```

### Example 1 — minimal A2A server setup

```python
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
import uvicorn

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
)
runner = Runner(
    app_name="my_a2a_app",
    agent=agent,
    session_service=InMemorySessionService(),
)

executor = A2aAgentExecutor(runner=runner)

# AgentCard uses protobuf — no top-level 'url' field.
# supported_interfaces declares the endpoint and protocol binding; without it
# card validation fails and the server cannot start.
# AgentSkill requires at minimum id, name, description, and tags.
card = AgentCard(
    name="assistant",
    description="A helpful assistant exposed via A2A.",
    version="1.0.0",
    supported_interfaces=[AgentInterface(
        url="http://localhost:8080/",   # client-reachable address, not the bind wildcard
        protocol_binding="JSONRPC",
    )],
    capabilities=AgentCapabilities(streaming=True),
    default_input_modes=["text"],
    default_output_modes=["text"],
    skills=[AgentSkill(
        id="general",
        name="General",
        description="General-purpose conversational assistant.",
        tags=["general"],
    )],
)

# DefaultRequestHandler requires a concrete TaskStore — not None.
handler = DefaultRequestHandler(
    agent_executor=executor,
    task_store=InMemoryTaskStore(),
)
app = A2AStarletteApplication(agent_card=card, http_handler=handler)

if __name__ == "__main__":
    uvicorn.run(app.build(), host="0.0.0.0", port=8080)
```

### Example 2 — deferred async runner initialisation

```python
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.runners import Runner

async def build_runner() -> Runner:
    # Expensive async setup: load config, open other connections, etc.
    # DatabaseSessionService is constructed directly (no async .create() method).
    from google.adk.sessions import DatabaseSessionService
    session_service = DatabaseSessionService(
        db_url="postgresql+asyncpg://user:pass@localhost/mydb"
    )
    agent = ...
    return Runner(app_name="app", agent=agent, session_service=session_service)

# Pass the coroutine function — it is awaited and cached on the first execute() call.
executor = A2aAgentExecutor(runner=build_runner)
```

### Example 3 — lifecycle interceptors

```python
from google.adk.a2a.executor.config import A2aAgentExecutorConfig, ExecuteInterceptor
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.types import TaskStatusUpdateEvent
import logging

logger = logging.getLogger(__name__)

async def log_before(ctx: RequestContext) -> RequestContext:
    logger.info("A2A request received: task_id=%s", ctx.task_id)
    return ctx

async def log_after(executor_ctx, final_event: TaskStatusUpdateEvent) -> TaskStatusUpdateEvent:
    logger.info(
        "A2A task completed: task_id=%s state=%s",
        final_event.task_id,
        final_event.status.state if final_event.status else "unknown",
    )
    return final_event

config = A2aAgentExecutorConfig(
    execute_interceptors=[
        ExecuteInterceptor(
            before_agent=log_before,
            after_agent=log_after,
        ),
    ]
)
executor = A2aAgentExecutor(runner=my_runner, config=config)
```

### Example 4 — filtering internal tool events from the A2A stream

```python
from google.adk.a2a.executor.config import A2aAgentExecutorConfig, ExecuteInterceptor
from typing import Union
from a2a.server.events import Event as A2AEvent
from google.adk.events.event import Event as AdkEvent

async def filter_internal_events(
    executor_ctx, a2a_event: A2AEvent, adk_event: AdkEvent
) -> Union[A2AEvent, None]:
    # Drop function-call and function-response events from the A2A stream.
    # Detect them by inspecting the event content, not event.author (which
    # ADK does not set to a sentinel value for tool events).
    if adk_event.get_function_calls() or adk_event.get_function_responses():
        return None   # hide tool-call events from the A2A client
    return a2a_event

config = A2aAgentExecutorConfig(
    execute_interceptors=[
        ExecuteInterceptor(after_event=filter_internal_events),
    ]
)
```

---
title: "Class deep dives — volume 16 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.2.0 classes: MCPSessionManager + connection params (session pooling, sampling callbacks, graceful reconnection), EventsCompactionConfig (token-threshold + sliding-window context management), App advanced configuration (compaction, resumability, context-cache), BaseLlmFlow + SingleFlow + AutoFlow (LLM pipeline internals), SpannerToolset full details (all 8 tools, SpannerToolSettings, SpannerVectorStoreSettings, ANN indexes), LlamaIndexRetrieval + custom BaseRetrievalTool patterns, A2A part converters (bidirectional GenAI↔A2A conversion, AdkEventToA2AEventsConverter), GoogleTool + BaseGoogleCredentialsConfig + GoogleCredentialsManager, Workflow rehydration (InterceptionResult, check_interception, _ChildScanState), AgentLoader + AgentConfig (all 4 YAML/module loading patterns)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 16"
  order: 85
---

Source-verified against **google-adk==2.2.0** (installed from PyPI, June 2026). Every field name, signature, and code example is drawn from the installed package source at `/usr/local/lib/python3.11/dist-packages/google/adk/`.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `MCPSessionManager` + `StdioConnectionParams` + `SseConnectionParams` + `StreamableHTTPConnectionParams` | `google.adk.tools.mcp_tool.mcp_session_manager` | Stable |
| 2 | `EventsCompactionConfig` + compaction internals | `google.adk.apps._configs`, `google.adk.apps.compaction` | `@experimental` |
| 3 | `App` advanced (compaction, resumability, context-cache) | `google.adk.apps.app` | Stable |
| 4 | `BaseLlmFlow` + `SingleFlow` + `AutoFlow` | `google.adk.flows.llm_flows` | Stable |
| 5 | `SpannerToolset` + `SpannerToolSettings` + `SpannerVectorStoreSettings` | `google.adk.tools.spanner` | `@experimental` |
| 6 | `LlamaIndexRetrieval` + custom `BaseRetrievalTool` | `google.adk.tools.retrieval` | Stable |
| 7 | A2A part converters (`convert_genai_part_to_a2a_part`, `convert_a2a_part_to_genai_part`, `AdkEventToA2AEventsConverter`) | `google.adk.a2a.converters` | `@experimental` |
| 8 | `GoogleTool` + `BaseGoogleCredentialsConfig` + credential management | `google.adk.tools`, `google.adk.tools._google_credentials` | `@experimental` |
| 9 | Workflow rehydration (`InterceptionResult`, `check_interception`, `_ChildScanState`) | `google.adk.workflow.utils._replay_interceptor` | Stable (internal) |
| 10 | `AgentLoader` + `AgentConfig` (all 4 loading patterns) | `google.adk.cli.utils.agent_loader`, `google.adk.agents.agent_config` | `@experimental` |

---

## 1 · `MCPSessionManager` + connection parameter classes

**Sources:** `google.adk.tools.mcp_tool.mcp_session_manager`, `google.adk.tools.mcp_tool.mcp_toolset`

`MCPSessionManager` is the internal session pool that `McpToolset` uses to manage persistent MCP connections. It creates, reuses, and automatically reconnects sessions keyed by connection type and headers. You rarely construct it directly — `McpToolset` builds one for you — but understanding its internals is essential for production deployments with custom authentication, sampling, and reconnection requirements.

### Connection parameter classes (source-verified)

```python
from mcp import StdioServerParameters
from google.adk.tools.mcp_tool.mcp_session_manager import (
    StdioConnectionParams,
    SseConnectionParams,
    StreamableHTTPConnectionParams,
)
```

#### `StdioConnectionParams`

```python
StdioConnectionParams(
    server_params: StdioServerParameters,   # command, args, env
    timeout: float = 5.0,                   # seconds to wait for stdio handshake
)
```

`StdioServerParameters` comes from the MCP SDK (`mcp.StdioServerParameters`) and accepts `command`, `args`, and `env`.

> **Prefer `StdioConnectionParams` over the bare `StdioServerParameters`** because it adds a `timeout` — bare `StdioServerParameters` passed directly to `McpToolset` has no connection timeout.

#### `SseConnectionParams`

```python
SseConnectionParams(
    url: str,
    headers: dict[str, Any] | None = None,
    timeout: float = 5.0,                  # connection timeout
    sse_read_timeout: float = 300.0,       # 5 min: max silence before close
    httpx_client_factory: CheckableMcpHttpClientFactory = create_mcp_http_client,
)
```

#### `StreamableHTTPConnectionParams`

```python
StreamableHTTPConnectionParams(
    url: str,
    headers: dict[str, Any] | None = None,
    timeout: float = 5.0,
    sse_read_timeout: float = 300.0,
    terminate_on_close: bool = True,       # send DELETE when closing
    httpx_client_factory: CheckableMcpHttpClientFactory = create_mcp_http_client,
)
```

`terminate_on_close=True` sends an HTTP DELETE to the MCP endpoint when the session is closed — the correct behaviour for HTTP Streamable connections that manage server-side state.

### `McpToolset` constructor (full source-verified signature)

```python
from mcp import StdioServerParameters
from mcp.client.session import SamplingFnT
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import (
    StdioConnectionParams,
    SseConnectionParams,
    StreamableHTTPConnectionParams,
)

McpToolset(
    connection_params,          # StdioConnectionParams | SseConnectionParams | StreamableHTTPConnectionParams | StdioServerParameters
    tool_filter=None,           # list[str] | ToolPredicate
    tool_name_prefix=None,      # str — prepended to every tool name
    errlog=sys.stderr,
    auth_scheme=None,           # AuthScheme
    auth_credential=None,       # AuthCredential
    require_confirmation=False, # bool | Callable[..., bool]
    header_provider=None,       # Callable[[ReadonlyContext], dict[str, str]]
    progress_callback=None,     # ProgressFnT | ProgressCallbackFactory
    use_mcp_resources=False,    # expose MCP resources as LoadMcpResourceTool
    sampling_callback=None,     # SamplingFnT — handle model-sampling requests from server
    sampling_capabilities=None, # SamplingCapability
    credential_key=None,        # str — cache key for auth credentials
)
```

### Example 1 — stdio server with connection timeout

```python
from mcp import StdioServerParameters
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.agents.llm_agent import LlmAgent

toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        ),
        timeout=10.0,   # 10-second stdio handshake timeout
    ),
    tool_filter=["read_file", "write_file", "list_directory"],
    tool_name_prefix="fs",
)

agent = LlmAgent(
    name="file_agent",
    model="gemini-2.0-flash",
    instruction="You help users manage files. Use the fs_ tools.",
    tools=[toolset],
)
```

### Example 2 — SSE server with dynamic auth headers

`header_provider` is called at each tool invocation with a `ReadonlyContext`, so it can return fresh tokens from session state:

```python
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

def _auth_headers(ctx: ReadonlyContext) -> dict[str, str]:
    token = ctx.state.get("bearer_token", "")
    return {"Authorization": f"Bearer {token}"}

toolset = McpToolset(
    connection_params=SseConnectionParams(
        url="https://mcp.example.com/sse",
        timeout=5.0,
        sse_read_timeout=120.0,
    ),
    header_provider=_auth_headers,
)
```

### Example 3 — Streamable HTTP with custom HTTPX client factory

```python
import httpx
from mcp.client.streamable_http import create_mcp_http_client
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

def _mtls_factory(**kwargs) -> httpx.AsyncClient:
    """Custom factory that adds mTLS certificates."""
    return httpx.AsyncClient(
        cert=("/path/to/client.pem", "/path/to/client.key"),
        verify="/path/to/ca.pem",
        **{k: v for k, v in kwargs.items() if k in ("timeout", "headers")},
    )

toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://secure-mcp.internal/mcp",
        timeout=10.0,
        terminate_on_close=True,
        httpx_client_factory=_mtls_factory,
    ),
)
```

### Example 4 — MCP sampling callback (server-initiated LLM calls)

Some MCP servers request LLM sampling to complete sub-tasks. Provide a `sampling_callback` to handle those requests:

```python
from mcp import ClientSession
from mcp.types import CreateMessageRequest, CreateMessageResult, TextContent
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

async def _sampling_handler(
    session: ClientSession,
    request: CreateMessageRequest,
) -> CreateMessageResult:
    """Route sampling requests to the project's internal LLM."""
    prompt = request.params.messages[-1].content
    if isinstance(prompt, TextContent):
        response_text = f"[sampled response for: {prompt.text[:80]}]"
    else:
        response_text = "[sampled response]"
    return CreateMessageResult(
        model="gemini-2.0-flash",
        role="assistant",
        content=TextContent(type="text", text=response_text),
        stopReason="endTurn",
    )

toolset = McpToolset(
    connection_params=SseConnectionParams(url="https://mcp.example.com/sse"),
    sampling_callback=_sampling_handler,
)
```

### Example 5 — tool name prefix + per-tool confirmation

```python
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.base_tool import BaseTool
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

DANGEROUS_TOOLS = {"delete_file", "execute_command"}

def _confirm_dangerous(tool: BaseTool, ctx: ReadonlyContext) -> bool:
    return tool.name.removeprefix("sys_") in DANGEROUS_TOOLS

toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(command="python3", args=["mcp_server.py"]),
    ),
    tool_name_prefix="sys",           # tools become sys_read_file, sys_execute_command, …
    require_confirmation=_confirm_dangerous,
)
```

### Session pooling internals

`MCPSessionManager` maintains a dict of `(session, AsyncExitStack, event_loop)` triples keyed by a SHA-256 hash of the serialised merged headers. On each `create_session()` call it:
1. Checks whether an existing session for the header key is still connected **and** bound to the current event loop.
2. If either check fails, cleans up the old session and creates a fresh one.
3. Under `FeatureName._MCP_GRACEFUL_ERROR_HANDLING`, also inspects the `SessionContext._is_task_alive` flag to detect crashed transports that appear open at the stream level.

The manager supports pickling (`__getstate__` / `__setstate__`) so it can survive serialisation in multi-process deployments — sessions are not preserved across pickle/unpickle boundaries but the configuration is.

---

## 2 · `EventsCompactionConfig` + compaction internals

**Sources:** `google.adk.apps._configs`, `google.adk.apps.compaction`

Long-running ADK sessions accumulate events that eventually overflow the LLM context window. `EventsCompactionConfig` controls two compaction strategies that periodically summarise older events into a single compacted event, keeping the session healthy.

### Constructor (source-verified)

```python
from google.adk.apps._configs import EventsCompactionConfig

EventsCompactionConfig(
    summarizer=None,             # BaseEventsSummarizer | None (auto-created from root LlmAgent)
    compaction_interval: int,    # REQUIRED — new invocations between sliding-window sweeps
    overlap_size: int,           # REQUIRED — prior invocations to re-include for context
    token_threshold=None,        # int | None — trigger threshold (chars/4 ≈ tokens)
    event_retention_size=None,   # int | None — raw events kept un-compacted
)
```

`token_threshold` and `event_retention_size` **must be set together** or both left as `None` — a `ValidationError` is raised otherwise.

### Sliding-window strategy

Compaction fires every `compaction_interval` new user invocations. When triggered, it summarises from `overlap_size` invocations before the new block to the end of the new block — creating overlapping summaries that preserve cross-turn context.

```
compaction_interval=3, overlap_size=1

After inv 1,2,3:  compact [1,2,3]
After inv 5,6:    (only 2 new, not triggered)
After inv 7:      compact [3,4,5,6,7]  ← inv 3 re-included (overlap_size=1)
```

### Token-threshold strategy

When `token_threshold` is set, ADK checks the prompt token count after every invocation. If it meets or exceeds the threshold, it immediately compacts events, keeping the last `event_retention_size` raw events uncompacted. This takes **priority** over the sliding-window sweep when both are configured.

### Example 1 — sliding-window compaction

```python
from google.adk.apps.app import App
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer

root = LlmAgent(
    name="assistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant.",
)

app = App(
    name="long_session_app",
    root_agent=root,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=5,   # summarise every 5 new invocations
        overlap_size=2,          # re-include last 2 invocations for context
    ),
)
```

### Example 2 — token-threshold compaction

```python
from google.adk.apps.app import App
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.agents.llm_agent import LlmAgent

root = LlmAgent(
    name="assistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant.",
)

app = App(
    name="token_managed_app",
    root_agent=root,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=10,    # sliding window as fallback
        overlap_size=2,
        token_threshold=50_000,    # trigger when prompt ≥ ~50k tokens
        event_retention_size=20,   # keep last 20 events raw
    ),
)
```

### Example 3 — custom summarizer

Provide a custom `BaseEventsSummarizer` to control how events are condensed:

```python
from typing import Optional
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.events.event import Event
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.apps.app import App
from google.adk.agents.llm_agent import LlmAgent


class BulletSummarizer(BaseEventsSummarizer):
    """Condenses events into a bullet-point list."""

    async def maybe_summarize_events(
        self,
        events: list[Event],
    ) -> Optional[Event]:
        # Gather all model text responses
        texts = []
        for ev in events:
            if ev.author == "model" and ev.content:
                for part in (ev.content.parts or []):
                    if part.text:
                        texts.append(part.text[:200])
        if not texts:
            return None
        summary = "Previous conversation summary:\n" + "\n".join(
            f"• {t}" for t in texts[-10:]
        )
        from google.genai import types
        import time
        start_ts = events[0].timestamp
        end_ts = events[-1].timestamp
        return Event(
            invocation_id=Event.new_id(),
            author="model",
            timestamp=end_ts + 0.001,
            content=types.Content(
                role="model",
                parts=[types.Part(text=summary)],
            ),
            actions=type("A", (), {"compaction": type("C", (), {
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                "compacted_content": types.Content(
                    role="model", parts=[types.Part(text=summary)]
                ),
            })()})(),
        )


root = LlmAgent(name="assistant", model="gemini-2.0-flash", instruction="Help users.")

app = App(
    name="custom_compact_app",
    root_agent=root,
    events_compaction_config=EventsCompactionConfig(
        summarizer=BulletSummarizer(),
        compaction_interval=4,
        overlap_size=1,
    ),
)
```

### Compaction safety guarantees

The compaction logic in `google.adk.apps.compaction` enforces several invariants before committing a compaction:
- **Pending function calls** — events containing function calls that have no matching response in the session are never compacted. The split point is shifted earlier to keep call/response pairs together.
- **HITL signals** — events with unresolved human-in-the-loop `requested_tool_confirmations` or `requested_auth_configs` are never compacted mid-flow.
- **Subsumption** — if two compaction ranges overlap, the narrower one is considered subsumed and ignored when building the effective context.

---

## 3 · `App` advanced configuration

**Source:** `google.adk.apps.app`

`App` is the top-level container for an ADK agentic application. Beyond the basics of `name` and `root_agent`, it wires together four cross-cutting configurations that apply to all agents in the app.

### Constructor (source-verified)

```python
from google.adk.apps.app import App

App(
    name: str,                                  # REQUIRED; letters, digits, underscores, hyphens; not "user"
    root_agent,                                 # REQUIRED; BaseAgent or BaseNode
    plugins: list[BasePlugin] = [],             # app-wide plugins
    events_compaction_config=None,              # EventsCompactionConfig
    context_cache_config=None,                  # ContextCacheConfig
    resumability_config=None,                   # ResumabilityConfig
)
```

`validate_app_name` enforces `^[a-zA-Z][a-zA-Z0-9_-]*$` and rejects the reserved name `"user"`.

### `ResumabilityConfig` (source-verified)

```python
from google.adk.apps._configs import ResumabilityConfig

ResumabilityConfig(
    is_resumable: bool = False,
)
```

When `is_resumable=True`, ADK can pause a long-running tool invocation and resume it from the last persisted event after a crash or timeout. Resumption is best-effort (at-least-once semantics) — ensure your tools are idempotent.

### Example 1 — fully configured App

```python
from google.adk.apps.app import App
from google.adk.apps._configs import EventsCompactionConfig, ResumabilityConfig
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.plugins.auto_tracing_plugin import AutoTracingPlugin

root = LlmAgent(
    name="main_agent",
    model="gemini-2.0-flash",
    instruction="You are a production assistant.",
)

app = App(
    name="production-app",
    root_agent=root,
    plugins=[AutoTracingPlugin()],
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=8,
        overlap_size=2,
        token_threshold=60_000,
        event_retention_size=15,
    ),
    resumability_config=ResumabilityConfig(is_resumable=True),
)
```

### Example 2 — multiple plugins with ordering

Plugins run in the order they appear in the `plugins` list. Place tracing and logging first so they capture the full invocation lifecycle:

```python
from google.adk.apps.app import App
from google.adk.agents.llm_agent import LlmAgent
from google.adk.plugins.auto_tracing_plugin import AutoTracingPlugin
from google.adk.plugins.debug_logging_plugin import DebugLoggingPlugin
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin

root = LlmAgent(
    name="agent",
    model="gemini-2.0-flash",
    instruction="Help users.",
)

app = App(
    name="multi-plugin-app",
    root_agent=root,
    plugins=[
        AutoTracingPlugin(),              # 1st: sets up OTel trace context
        DebugLoggingPlugin(),             # 2nd: logs all LLM calls
        ReflectAndRetryToolPlugin(max_retries=2),  # 3rd: auto-retries failed tools
    ],
)
```

### Example 3 — validating app names

```python
from google.adk.apps.app import validate_app_name

# Valid names
validate_app_name("my-app")        # OK
validate_app_name("agent_v2")      # OK

# Invalid names — will raise ValueError
try:
    validate_app_name("user")          # reserved
except ValueError as e:
    print(e)  # App name cannot be 'user'; reserved for end-user input.

try:
    validate_app_name("123app")        # must start with letter
except ValueError as e:
    print(e)  # Invalid app name '123app': must start with a letter...
```

### Example 4 — App with Workflow root node

`App.root_agent` accepts either a `BaseAgent` or a `BaseNode` (Workflow node):

```python
from google.adk.apps.app import App
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._node import node
from google.adk.agents.llm_agent import LlmAgent

@node
async def step_one(ctx):
    return "step one result"

@node
async def step_two(ctx, step_one_result: str):
    return f"step two got: {step_one_result}"

wf = Workflow(name="pipeline")
wf.add_node(step_one)
wf.add_node(step_two, dependencies=["step_one"])

app = App(
    name="workflow-app",
    root_agent=wf,   # BaseNode accepted
)
```

---

## 4 · `BaseLlmFlow` + `SingleFlow` + `AutoFlow`

**Sources:** `google.adk.flows.llm_flows.base_llm_flow`, `google.adk.flows.llm_flows.single_flow`, `google.adk.flows.llm_flows.auto_flow`

The LLM flow classes form the backbone of how `LlmAgent` interacts with language models. Every `LlmAgent` is backed by one of these flows — `SingleFlow` for agents that do not auto-transfer to sub-agents, `AutoFlow` for agents that do.

### Hierarchy

```
BaseLlmFlow (ABC)
  └── SingleFlow          ← base flow + request/response processor chain
        └── AutoFlow      ← SingleFlow + agent_transfer request processor
```

### Flow selection

`LlmAgent` selects its flow based on whether agent transfers are enabled:
- `disallow_transfer_to_parent=True` **and** `disallow_transfer_to_peers=True` → `SingleFlow`
- Otherwise → `AutoFlow`

### `BaseLlmFlow` processor chain

`BaseLlmFlow` maintains two ordered processor lists:

| List | Type | Purpose |
|---|---|---|
| `request_processors` | `list[BaseLlmRequestProcessor]` | Transform the `LlmRequest` before sending to the model |
| `response_processors` | `list[BaseLlmResponseProcessor]` | Handle the `LlmResponse` after receiving from the model |

`SingleFlow` populates these with the standard ADK processors for instructions, contents, tools, function calls, and output schemas.

### Example 1 — reading the flow of a live LlmAgent

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.flows.llm_flows.auto_flow import AutoFlow
from google.adk.flows.llm_flows.single_flow import SingleFlow

agent = LlmAgent(
    name="my_agent",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant.",
)

# Inspect the agent's flow (requires instantiated agent)
flow = agent._get_flow()
print(type(flow).__name__)           # "AutoFlow" (default)
print([type(p).__name__ for p in flow.request_processors])
# ['InstructionsLlmRequestProcessor', 'ContentsLlmRequestProcessor', ...]
```

### Example 2 — custom `BaseLlmRequestProcessor` to inject metadata

Implement `BaseLlmRequestProcessor` to modify the LLM request before it is sent:

```python
from google.adk.flows.llm_flows._base_llm_processor import BaseLlmRequestProcessor
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.agents.llm_agent import LlmAgent
import datetime


class DateInjectorProcessor(BaseLlmRequestProcessor):
    """Prepends today's date to the system instruction."""

    async def process(
        self,
        invocation_context: InvocationContext,
        llm_request: LlmRequest,
    ) -> None:
        today = datetime.date.today().isoformat()
        date_note = f"\n\n[Current date: {today}]"
        if llm_request.config and llm_request.config.system_instruction:
            si = llm_request.config.system_instruction
            if isinstance(si, str):
                llm_request.config.system_instruction = si + date_note
        return None   # yield nothing → continue pipeline


# Attach to an agent's flow by subclassing LlmAgent
class DateAwareLlmAgent(LlmAgent):
    def _build_flow(self):
        flow = super()._build_flow()
        flow.request_processors.append(DateInjectorProcessor())
        return flow
```

### Example 3 — custom `BaseLlmResponseProcessor` for response auditing

```python
from google.adk.flows.llm_flows._base_llm_processor import BaseLlmResponseProcessor
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_response import LlmResponse
from google.adk.events.event import Event
from typing import AsyncGenerator
import logging

logger = logging.getLogger(__name__)


class AuditResponseProcessor(BaseLlmResponseProcessor):
    """Logs every model response for audit purposes."""

    async def process(
        self,
        invocation_context: InvocationContext,
        llm_response: LlmResponse,
        model_response_event: Event,
    ) -> AsyncGenerator[Event, None]:
        if llm_response.content:
            for part in (llm_response.content.parts or []):
                if part.text:
                    logger.info(
                        "MODEL RESPONSE [session=%s]: %.200s",
                        invocation_context.session.id,
                        part.text,
                    )
        return   # must be a generator — return without yield
        yield    # make Python treat this as an async generator
```

### Example 4 — forcing `SingleFlow` (no agent transfers)

```python
from google.adk.agents.llm_agent import LlmAgent

# Setting both flags forces SingleFlow, disabling all auto-transfers
isolated_agent = LlmAgent(
    name="isolated",
    model="gemini-2.0-flash",
    instruction="Answer only from your own knowledge.",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
```

### `AutoFlow` agent-transfer mechanics

`AutoFlow` adds `agent_transfer.request_processor` to `SingleFlow`'s chain. This processor:
1. Injects descriptions of available sub-agents and peers into the system instruction.
2. Parses `transfer_to_agent` tool calls from model responses.
3. Emits an `Event` with `actions.transfer_to_agent = target_name`, which the `Runner` picks up to activate the target agent for the next turn.

Peer transfers are only enabled when **all** of these are true:
- Parent is also an `LlmAgent`
- `disallow_transfer_to_peers` is `False` (the default)

---

## 5 · `SpannerToolset` + `SpannerToolSettings` + `SpannerVectorStoreSettings`

**Sources:** `google.adk.tools.spanner`

`SpannerToolset` is marked `@experimental` and exposes up to 8 read-oriented tools for querying Cloud Spanner databases, including semantic vector similarity search powered by Vertex AI embeddings.

### All 8 tools (source-verified)

| Tool name (with prefix) | Function | Category |
|---|---|---|
| `spanner_list_table_names` | List all table names in the database | Metadata |
| `spanner_list_table_indexes` | List indexes for a given table | Metadata |
| `spanner_list_table_index_columns` | List indexed columns | Metadata |
| `spanner_list_named_schemas` | List named schemas | Metadata |
| `spanner_get_table_schema` | Get full DDL for a table | Metadata |
| `spanner_execute_sql` | Run a read-only SQL query | Data |
| `spanner_similarity_search` | Text similarity search via ML.PREDICT / embedding | Vector |
| `spanner_vector_store_similarity_search` | Vector store search (managed table) | Vector |

### `SpannerToolSettings` (source-verified)

```python
from google.adk.tools.spanner.settings import SpannerToolSettings, Capabilities, QueryResultMode

SpannerToolSettings(
    capabilities=[Capabilities.DATA_READ],      # list[Capabilities]; only DATA_READ currently
    max_executed_query_result_rows=50,          # max rows returned by execute_sql
    query_result_mode=QueryResultMode.DEFAULT,  # DEFAULT (list of rows) | DICT_LIST (list of dicts)
    database_role=None,                         # str | None — Spanner database role
    vector_store_settings=None,                 # SpannerVectorStoreSettings | None
)
```

`QueryResultMode.DICT_LIST` returns rows as `[{"col_name": value, ...}, ...]` — much easier for LLMs to parse than unlabeled arrays.

### `SpannerVectorStoreSettings` (source-verified)

```python
from google.adk.tools.spanner.settings import SpannerVectorStoreSettings

SpannerVectorStoreSettings(
    project_id: str,
    instance_id: str,
    database_id: str,
    table_name: str,
    content_column: str,
    embedding_column: str,
    vector_length: int,
    vertex_ai_embedding_model_name: str,        # e.g. "text-embedding-005"
    selected_columns: list[str] = [],           # defaults to [content_column]
    nearest_neighbors_algorithm="EXACT_NEAREST_NEIGHBORS",  # or "APPROXIMATE_NEAREST_NEIGHBORS"
    top_k: int = 4,
    distance_type: str = "COSINE",             # "COSINE" | "DOT_PRODUCT" | "EUCLIDEAN"
    num_leaves_to_search=None,                 # int | None — ANN only
    additional_filter=None,                    # str | None — extra WHERE clause
    vector_search_index_settings=None,         # VectorSearchIndexSettings | None — ANN only
    additional_columns_to_setup=None,          # list[TableColumn] | None
    primary_key_columns=None,                  # list[str] | None
)
```

### Example 1 — basic read-only SQL agent

```python
from google.adk.tools.spanner.spanner_toolset import SpannerToolset
from google.adk.tools.spanner.settings import SpannerToolSettings, QueryResultMode
from google.adk.agents.llm_agent import LlmAgent

settings = SpannerToolSettings(
    max_executed_query_result_rows=100,
    query_result_mode=QueryResultMode.DICT_LIST,
)

toolset = SpannerToolset(
    spanner_tool_settings=settings,
)

agent = LlmAgent(
    name="db_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You answer questions about the sales database. "
        "Use spanner_execute_sql for queries. "
        "Always call spanner_list_table_names first if you are unsure about schema."
    ),
    tools=[toolset],
)
```

### Example 2 — vector similarity search (exact nearest neighbours)

```python
from google.adk.tools.spanner.spanner_toolset import SpannerToolset
from google.adk.tools.spanner.settings import (
    SpannerToolSettings,
    SpannerVectorStoreSettings,
    QueryResultMode,
)
from google.adk.agents.llm_agent import LlmAgent

vector_settings = SpannerVectorStoreSettings(
    project_id="my-project",
    instance_id="my-instance",
    database_id="docs_db",
    table_name="document_embeddings",
    content_column="content",
    embedding_column="embedding",
    vector_length=768,
    vertex_ai_embedding_model_name="text-embedding-005",
    selected_columns=["content", "title", "url"],
    top_k=5,
    distance_type="COSINE",
)

settings = SpannerToolSettings(
    query_result_mode=QueryResultMode.DICT_LIST,
    vector_store_settings=vector_settings,
)

toolset = SpannerToolset(spanner_tool_settings=settings)

agent = LlmAgent(
    name="rag_agent",
    model="gemini-2.0-flash",
    instruction="Answer questions by searching the document store first.",
    tools=[toolset],
)
```

### Example 3 — ANN (approximate nearest neighbours) with index

```python
from google.adk.tools.spanner.settings import (
    SpannerVectorStoreSettings,
    SpannerToolSettings,
    VectorSearchIndexSettings,
)

index_cfg = VectorSearchIndexSettings(
    index_name="doc_embedding_idx",
    num_leaves=2000,        # ~rows/1000
    tree_depth=2,
    additional_storing_columns=["category"],
)

vector_settings = SpannerVectorStoreSettings(
    project_id="my-project",
    instance_id="my-instance",
    database_id="docs_db",
    table_name="document_embeddings",
    content_column="content",
    embedding_column="embedding",
    vector_length=768,
    vertex_ai_embedding_model_name="text-embedding-005",
    nearest_neighbors_algorithm="APPROXIMATE_NEAREST_NEIGHBORS",
    num_leaves_to_search=50,
    vector_search_index_settings=index_cfg,
    additional_filter="category = 'technical'",
    top_k=10,
)

settings = SpannerToolSettings(vector_store_settings=vector_settings)
toolset = SpannerToolset(spanner_tool_settings=settings)
```

### Example 4 — limiting tools with a filter

```python
from google.adk.tools.spanner.spanner_toolset import SpannerToolset

# Only expose read tools, hide vector search
toolset = SpannerToolset(
    tool_filter=["spanner_list_table_names", "spanner_execute_sql"],
)
```

---

## 6 · `LlamaIndexRetrieval` + custom `BaseRetrievalTool`

**Sources:** `google.adk.tools.retrieval.llama_index_retrieval`, `google.adk.tools.retrieval.base_retrieval_tool`

`LlamaIndexRetrieval` bridges LlamaIndex retrievers into the ADK tool system. It is one of the few ADK tools that takes an external framework object (`BaseRetriever`) rather than wrapping a plain function.

### `LlamaIndexRetrieval` constructor (source-verified)

```python
from google.adk.tools.retrieval.llama_index_retrieval import LlamaIndexRetrieval

LlamaIndexRetrieval(
    name: str,
    description: str,
    retriever: BaseRetriever,   # any llama_index.core.base.base_retriever.BaseRetriever
)
```

`run_async` calls `retriever.retrieve(args['query'])` (synchronously, inside the async method) and returns the text of the first result node.

> **Note:** `retrieve()` returns a list of `NodeWithScore`. The current ADK implementation returns only the **first** result's text. For multi-result responses, subclass and override `run_async`.

### Example 1 — LlamaIndex VectorStoreIndex retriever

```python
# pip install llama-index-core llama-index-embeddings-google
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from google.adk.tools.retrieval.llama_index_retrieval import LlamaIndexRetrieval
from google.adk.agents.llm_agent import LlmAgent

# Build the index
documents = SimpleDirectoryReader("./docs").load_data()
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=3)

# Wrap in ADK tool
retrieval_tool = LlamaIndexRetrieval(
    name="search_docs",
    description="Search the company documentation for relevant information.",
    retriever=retriever,
)

agent = LlmAgent(
    name="rag_agent",
    model="gemini-2.0-flash",
    instruction="Answer questions by searching the documentation first.",
    tools=[retrieval_tool],
)
```

### Example 2 — multi-result override

```python
from typing import Any
from google.adk.tools.retrieval.llama_index_retrieval import LlamaIndexRetrieval
from google.adk.tools.tool_context import ToolContext


class MultiResultRetrieval(LlamaIndexRetrieval):
    """Returns the top-N results as a formatted string."""

    def __init__(self, *, name: str, description: str, retriever, top_n: int = 3):
        super().__init__(name=name, description=description, retriever=retriever)
        self.top_n = top_n

    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> Any:
        results = self.retriever.retrieve(args["query"])[:self.top_n]
        if not results:
            return "No results found."
        return "\n\n---\n\n".join(
            f"[Result {i+1}] {r.text}" for i, r in enumerate(results)
        )
```

### Building a custom `BaseRetrievalTool`

`BaseRetrievalTool` is the base class for retrieval tools. It inherits from `BaseTool` and adds a standard `query: str` parameter to the function declaration.

```python
from typing import Any
from typing_extensions import override
from google.adk.tools.retrieval.base_retrieval_tool import BaseRetrievalTool
from google.adk.tools.tool_context import ToolContext
import httpx


class SemanticSearchTool(BaseRetrievalTool):
    """Retrieval tool backed by a custom REST search endpoint."""

    def __init__(self, *, endpoint: str, api_key: str):
        super().__init__(
            name="semantic_search",
            description="Search the knowledge base using semantic similarity.",
        )
        self._endpoint = endpoint
        self._api_key = api_key

    @override
    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> Any:
        query = args.get("query", "")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self._endpoint,
                json={"query": query, "top_k": 5},
                headers={"X-Api-Key": self._api_key},
                timeout=10.0,
            )
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
        if not hits:
            return "No results found."
        return "\n\n".join(
            f"[Score: {h['score']:.3f}] {h['text']}" for h in hits
        )
```

### Example 3 — combining retrieval tools with regular tools

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.retrieval.llama_index_retrieval import LlamaIndexRetrieval

def get_current_time() -> str:
    """Returns the current UTC time."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

retrieval_tool = LlamaIndexRetrieval(
    name="search_docs",
    description="Search internal documentation.",
    retriever=retriever,  # from Example 1
)

agent = LlmAgent(
    name="hybrid_agent",
    model="gemini-2.0-flash",
    instruction="Answer questions using documentation and real-time data.",
    tools=[retrieval_tool, get_current_time],
)
```

---

## 7 · A2A part converters

**Sources:** `google.adk.a2a.converters.part_converter`, `google.adk.a2a.converters.from_adk_event`, `google.adk.a2a.converters.to_adk_event`

The A2A converter layer translates between **Google GenAI `types.Part`** objects (used internally by ADK) and **A2A `a2a.types.Part`** objects (used in multi-agent A2A protocol traffic). These are marked `@experimental` but are essential for building custom A2A integrations.

### Key constants (source-verified)

```python
from google.adk.a2a.converters.part_converter import (
    A2A_DATA_PART_METADATA_TYPE_KEY,           # "type"
    A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY, # "is_long_running"
    A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL,  # "function_call"
    A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE, # "function_response"
    A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT, # "code_execution_result"
    A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE,      # "executable_code"
    A2A_DATA_PART_TEXT_MIME_TYPE,              # "text/plain"
    A2A_DATA_PART_START_TAG,                   # b'<a2a_datapart_json>'
    A2A_DATA_PART_END_TAG,                     # b'</a2a_datapart_json>'
)
```

### GenAI→A2A conversion

`convert_genai_part_to_a2a_part` converts a `google.genai.types.Part` to an `a2a.types.Part`:

- `text` part → `TextPart`
- `inline_data` part → `FilePart` with `FileWithBytes` (base64 encoded)
- `file_data` part → `FilePart` with `FileWithUri`
- `function_call` part → `DataPart` with `metadata={"type": "function_call"}` and JSON payload
- `function_response` part → `DataPart` with `metadata={"type": "function_response"}`
- `executable_code` / `code_execution_result` → `DataPart` with corresponding type tag

### A2A→GenAI conversion

`convert_a2a_part_to_genai_part` reverses the above:

- `TextPart` → `types.Part(text=..., thought=metadata.get("thought"))`
- `FilePart` with URI → `types.Part(file_data=...)`
- `FilePart` with bytes → `types.Part(inline_data=...)`
- `DataPart` with `type=function_call` → `types.Part(function_call=...)` (JSON-decoded)
- `DataPart` with `type=function_response` → `types.Part(function_response=...)`

### `AdkEventToA2AEventsConverter` type alias

```python
from google.adk.a2a.converters.from_adk_event import AdkEventToA2AEventsConverter

# Type signature (source-verified):
# Callable[
#     [Event, Optional[Dict[str, str]], Optional[str], Optional[str], GenAIPartToA2APartConverter],
#     List[A2AUpdateEvent],
# ]
```

This callable is passed to the `A2aAgentExecutor` and called for every ADK event to produce `TaskStatusUpdateEvent` or `TaskArtifactUpdateEvent` objects for the A2A client.

### Example 1 — converting a text Part

```python
from google.genai import types as genai_types
from google.adk.a2a.converters.part_converter import convert_genai_part_to_a2a_part

text_part = genai_types.Part(text="Hello from the model!")
a2a_part = convert_genai_part_to_a2a_part(text_part)
print(type(a2a_part.root).__name__)  # TextPart
print(a2a_part.root.text)           # "Hello from the model!"
```

### Example 2 — round-trip function call conversion

```python
from google.genai import types as genai_types
from google.adk.a2a.converters.part_converter import (
    convert_genai_part_to_a2a_part,
    convert_a2a_part_to_genai_part,
)

# Simulate a function call part
fc_part = genai_types.Part(
    function_call=genai_types.FunctionCall(
        id="call_001",
        name="search_web",
        args={"query": "latest ADK news"},
    )
)

# GenAI → A2A
a2a_part = convert_genai_part_to_a2a_part(fc_part)
print(type(a2a_part.root).__name__)  # DataPart
print(a2a_part.root.metadata)        # {"type": "function_call"}

# A2A → GenAI
restored_part = convert_a2a_part_to_genai_part(a2a_part)
print(restored_part.function_call.name)  # "search_web"
print(restored_part.function_call.args)  # {"query": "latest ADK news"}
```

### Example 3 — custom `AdkEventToA2AEventsConverter`

Override the default conversion to add custom metadata to outgoing A2A events:

```python
from typing import Optional, Dict, List
from google.adk.events.event import Event
from google.adk.a2a.converters.from_adk_event import (
    AdkEventToA2AEventsConverter,
    _convert_adk_event_to_a2a_events,  # default implementation
)
from google.adk.a2a.converters.part_converter import (
    GenAIPartToA2APartConverter,
    convert_genai_part_to_a2a_part,
)
from a2a.server.events import Event as A2AEvent


def my_converter(
    event: Event,
    agents_artifacts: Optional[Dict[str, str]],
    task_id: Optional[str],
    context_id: Optional[str],
    part_converter: GenAIPartToA2APartConverter = convert_genai_part_to_a2a_part,
) -> List:
    # Delegate to the default converter
    a2a_events = _convert_adk_event_to_a2a_events(
        event, agents_artifacts, task_id, context_id, part_converter
    )
    # Add custom metadata to every status update
    for a2a_event in a2a_events:
        if hasattr(a2a_event, "status") and a2a_event.status:
            if a2a_event.status.message:
                for part in (a2a_event.status.message.parts or []):
                    if hasattr(part.root, "metadata") and part.root.metadata is None:
                        part.root.metadata = {}
                    if hasattr(part.root, "metadata") and part.root.metadata is not None:
                        part.root.metadata["app_version"] = "1.0.0"
    return a2a_events
```

---

## 8 · `GoogleTool` + `BaseGoogleCredentialsConfig`

**Sources:** `google.adk.tools.google_tool`, `google.adk.tools._google_credentials`

`GoogleTool` (marked `@experimental(FeatureName.GOOGLE_TOOL)`) is a credential-aware `FunctionTool` subclass for building tools that call Google APIs. It handles the full OAuth2 flow — presenting an auth URL to the user, exchanging the code, caching the token, and refreshing it on expiry — so your tool function only needs to call the API.

### Constructor (source-verified)

```python
from google.adk.tools.google_tool import GoogleTool

GoogleTool(
    func: Callable[..., Any],
    credentials_config: Optional[BaseGoogleCredentialsConfig] = None,
    tool_settings: Optional[BaseModel] = None,
)
```

Parameters named `credentials` and `settings` are automatically stripped from the function's LLM-visible schema (`_ignore_params`). The framework injects fresh `google.auth.credentials.Credentials` into `credentials` and the `tool_settings` object into `settings` at call time.

### `BaseGoogleCredentialsConfig` (source-verified)

```python
from google.adk.tools._google_credentials import BaseGoogleCredentialsConfig

# Auth option 1: pre-built credentials (service account / ADC)
BaseGoogleCredentialsConfig(
    credentials=google_credentials_object,
)

# Auth option 2: token from session state
BaseGoogleCredentialsConfig(
    external_access_token_key="user_access_token",
)

# Auth option 3: OAuth2 client credentials → user consent flow
BaseGoogleCredentialsConfig(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    scopes=["https://www.googleapis.com/auth/gmail.readonly"],
)
```

All three options are mutually exclusive. Set exactly one.

### Example 1 — Gmail read tool with OAuth2 flow

```python
import google.auth.credentials
from typing import Optional
from google.adk.tools.google_tool import GoogleTool
from google.adk.tools._google_credentials import BaseGoogleCredentialsConfig
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.llm_agent import LlmAgent


async def list_gmail_labels(
    credentials: google.auth.credentials.Credentials,  # injected by GoogleTool
    tool_context: ToolContext,                          # injected by ADK
    max_results: int = 20,
) -> dict:
    """List Gmail labels for the authenticated user."""
    import httpx
    headers = {"Authorization": f"Bearer {credentials.token}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://gmail.googleapis.com/gmail/v1/users/me/labels",
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
    labels = [l["name"] for l in data.get("labels", [])[:max_results]]
    return {"labels": labels}


gmail_tool = GoogleTool(
    func=list_gmail_labels,
    credentials_config=BaseGoogleCredentialsConfig(
        client_id="YOUR_CLIENT_ID.apps.googleusercontent.com",
        client_secret="YOUR_CLIENT_SECRET",
        scopes=["https://www.googleapis.com/auth/gmail.readonly"],
    ),
)

agent = LlmAgent(
    name="gmail_agent",
    model="gemini-2.0-flash",
    instruction="Help users manage their Gmail. List labels when asked.",
    tools=[gmail_tool],
)
```

### Example 2 — service account (ADC) credentials

```python
import google.auth
from google.adk.tools.google_tool import GoogleTool
from google.adk.tools._google_credentials import BaseGoogleCredentialsConfig

# Use Application Default Credentials (e.g. in Cloud Run)
adc_credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

async def call_vertex_api(
    credentials: google.auth.credentials.Credentials,
    endpoint: str,
) -> dict:
    """Call a Vertex AI endpoint."""
    import httpx
    headers = {"Authorization": f"Bearer {credentials.token}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://us-central1-aiplatform.googleapis.com/v1/{endpoint}",
                                headers=headers)
        return resp.json()

vertex_tool = GoogleTool(
    func=call_vertex_api,
    credentials_config=BaseGoogleCredentialsConfig(credentials=adc_credentials),
)
```

### Example 3 — token from session state

Use `external_access_token_key` when a frontend has already obtained an access token and stored it in session state:

```python
from google.adk.tools.google_tool import GoogleTool
from google.adk.tools._google_credentials import BaseGoogleCredentialsConfig
import google.auth.credentials

async def read_calendar_events(
    credentials: google.auth.credentials.Credentials,
    tool_context,
    days_ahead: int = 7,
) -> list[dict]:
    """Return calendar events for the next N days."""
    import httpx
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    time_min = now.isoformat()
    time_max = (now + timedelta(days=days_ahead)).isoformat()
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://www.googleapis.com/calendar/v3/calendars/primary/events",
            headers={"Authorization": f"Bearer {credentials.token}"},
            params={"timeMin": time_min, "timeMax": time_max, "singleEvents": "true"},
        )
        return resp.json().get("items", [])

calendar_tool = GoogleTool(
    func=read_calendar_events,
    credentials_config=BaseGoogleCredentialsConfig(
        external_access_token_key="google_calendar_token",  # read from session state
    ),
)
```

At runtime the framework reads `tool_context.state["google_calendar_token"]` and injects it as `credentials.token`.

---

## 9 · Workflow rehydration (`InterceptionResult`, `check_interception`, `_ChildScanState`)

**Source:** `google.adk.workflow.utils._replay_interceptor`, `google.adk.workflow.utils._rehydration_utils`

When a Workflow session is replayed (e.g. after a crash or HITL pause), ADK reconstructs the execution state from persisted events. `check_interception` is the gating function called at the start of each node execution to decide whether to re-run it or fast-forward from the cached result.

### `InterceptionResult` (source-verified)

```python
from google.adk.workflow.utils._replay_interceptor import InterceptionResult

@dataclass(kw_only=True)
class InterceptionResult:
    should_run: bool                # True → execute node natively
    output: Any = None              # cached output for fast-forward
    route: Any = None               # cached routing decision
    interrupts: set[str] = field(default_factory=set)   # unresolved HITL IDs
    resume_inputs: dict[str, Any] | None = None          # resolved responses for re-run
    transfer_to_agent: str | None = None                 # agent name for fast-forward transfer
```

### `_ChildScanState` (source-verified)

`_ChildScanState` accumulates per-child-node state while scanning historical events:

```python
from google.adk.workflow.utils._rehydration_utils import _ChildScanState

@dataclass
class _ChildScanState:
    run_id: str | None = None
    output: Any = None
    route: str | None = None
    branch: str | None = None
    isolation_scope: str | None = None
    transfer_to_agent: str | None = None
    interrupt_ids: set[str] = field(default_factory=set)
    resolved_ids: set[str] = field(default_factory=set)
    resolved_responses: dict[str, Any] = field(default_factory=dict)
```

`interrupt_ids - resolved_ids` gives the set of **unresolved** interrupts — if non-empty, the node stays in `WAITING` state.

### `check_interception` decision tree

```
check_interception(node_path, node, recovered, current_run, curr_parent_ctx)
    ├─ current_run.status == COMPLETED  → InterceptionResult(should_run=False, output=cached)
    ├─ current_run.status == WAITING    → InterceptionResult(should_run=False, interrupts=unresolved)
    ├─ recovered is None                → InterceptionResult(should_run=True)
    ├─ recovered has unresolved interrupts → stay WAITING
    ├─ recovered has output (completed) → fast-forward (should_run=False, output=cached)
    └─ recovered has resume_inputs      → re-run with injected inputs (should_run=True)
```

### Example 1 — understanding rehydration in a Workflow

```python
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._node import node
from google.adk.agents.context import Context
from google.adk.apps.app import App
from google.adk.apps._configs import ResumabilityConfig


@node
async def fetch_data(ctx: Context) -> dict:
    """Step 1: fetch data from an external API."""
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://api.example.com/data", timeout=30.0)
        return resp.json()


@node
async def process_data(ctx: Context, fetch_data_result: dict) -> str:
    """Step 2: process the fetched data."""
    records = fetch_data_result.get("records", [])
    return f"Processed {len(records)} records."


wf = Workflow(name="etl_pipeline")
wf.add_node(fetch_data)
wf.add_node(process_data, dependencies=["fetch_data"])

app = App(
    name="etl-app",
    root_agent=wf,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

# If the app crashes between step 1 and step 2:
# On resume, check_interception sees fetch_data's output in session events
# and fast-forwards it (should_run=False) — no duplicate API call.
# process_data has no historical output, so it re-runs (should_run=True).
```

### Example 2 — HITL (human-in-the-loop) with rehydration

```python
from google.adk.workflow._workflow import Workflow
from google.adk.workflow._node import node
from google.adk.agents.context import Context
from typing import Any


@node
async def request_approval(ctx: Context, action: str) -> bool:
    """Pause and ask a human to approve an action."""
    # This node uses workflow interrupts to pause until human input arrives.
    # On resume, check_interception injects the human's response via resume_inputs.
    return ctx.get_input("approved", schema=bool)  # blocks until resolved


@node
async def execute_action(ctx: Context, request_approval_result: bool, action: str) -> str:
    if not request_approval_result:
        return "Action cancelled by human reviewer."
    return f"Executed: {action}"


wf = Workflow(name="approval_workflow")
wf.add_node(request_approval)
wf.add_node(execute_action, dependencies=["request_approval"])
```

### Example 3 — inspecting rehydration state (debugging)

```python
from google.adk.workflow.utils._rehydration_utils import (
    _ChildScanState,
    _unwrap_response,
    _wrap_response,
)

# _wrap_response / _unwrap_response normalise FunctionResponse payloads
wrapped = _wrap_response("hello")
print(wrapped)                    # {"result": "hello"}
print(_unwrap_response(wrapped))  # "hello"

# Dict values are passed through unchanged
wrapped_dict = _wrap_response({"key": "value"})
print(wrapped_dict)               # {"key": "value"}
print(_unwrap_response(wrapped_dict))  # {"key": "value"}
```

---

## 10 · `AgentLoader` + `AgentConfig` (YAML / module loading)

**Sources:** `google.adk.cli.utils.agent_loader`, `google.adk.agents.agent_config`

`AgentLoader` is the ADK CLI's unified agent discovery engine. It supports four loading patterns, auto-detects `.env` files, and caches loaded agents in-process. `AgentConfig` is the Pydantic discriminated union that powers YAML-based agent definitions.

### `AgentLoader` constructor (source-verified)

```python
from google.adk.cli.utils.agent_loader import AgentLoader

loader = AgentLoader(agents_dir: str)
# Accepts either:
#   agents_dir/ (multi-agent directory)
#   agents_dir/agent_name/ (single-agent directory with agent.py or root_agent.yaml)
```

When `agents_dir` points directly at a single-agent directory, the loader enters **single-agent mode** and exposes only that one agent.

### 4 Loading patterns

| Pattern | File structure | Trigger |
|---|---|---|
| **a) `.agent` submodule** | `agents_dir/{name}/agent.py` | `root_agent` var in `agent.py` |
| **b) Module** | `agents_dir/{name}.py` | `root_agent` var in module |
| **c) Package** | `agents_dir/{name}/__init__.py` | `root_agent` in package |
| **d) YAML config** | `agents_dir/{name}/root_agent.yaml` | `root_agent.yaml` parsed via `AgentConfig` |

`AgentLoader` tries patterns in order a → b → c → d and returns on the first match.

### `AgentConfig` + discriminated union (source-verified)

```python
from google.adk.agents.agent_config import AgentConfig, ConfigsUnion

# AgentConfig is a RootModel[ConfigsUnion]
# ConfigsUnion discriminates on the "agent_class" field:
#   "LlmAgent"        → LlmAgentConfig
#   "LoopAgent"       → LoopAgentConfig
#   "ParallelAgent"   → ParallelAgentConfig
#   "SequentialAgent" → SequentialAgentConfig
#   anything else     → BaseAgentConfig
```

### Example 1 — loading an agent from Python code (pattern a)

Directory structure:
```
agents/
  weather_agent/
    agent.py         ← defines root_agent
    .env             ← auto-loaded by AgentLoader
```

`agents/weather_agent/agent.py`:
```python
from google.adk.agents.llm_agent import LlmAgent

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: Sunny, 22°C"

root_agent = LlmAgent(
    name="weather_agent",
    model="gemini-2.0-flash",
    instruction="Answer weather queries using the get_weather tool.",
    tools=[get_weather],
)
```

Loading:
```python
from google.adk.cli.utils.agent_loader import AgentLoader
import asyncio

loader = AgentLoader("agents/")
agent = loader.load_agent("weather_agent")
print(type(agent).__name__)   # LlmAgent
```

### Example 2 — YAML-based agent definition (pattern d)

`agents/support_bot/root_agent.yaml`:
```yaml
agent_class: LlmAgent
name: support_bot
model: gemini-2.0-flash
instruction: |
  You are a customer support agent for Acme Corp.
  Be helpful, concise, and professional.
  Escalate to a human if the user is upset.
output_key: last_response
disallow_transfer_to_parent: false
disallow_transfer_to_peers: false
```

Loading:
```python
from google.adk.cli.utils.agent_loader import AgentLoader

loader = AgentLoader("agents/")
agent = loader.load_agent("support_bot")
print(agent.instruction[:50])  # "You are a customer support agent..."
```

### Example 3 — YAML with sub-agents (SequentialAgent)

`agents/pipeline/root_agent.yaml`:
```yaml
agent_class: SequentialAgent
name: pipeline
sub_agents:
  - agent_class: LlmAgent
    name: extractor
    model: gemini-2.0-flash
    instruction: "Extract key facts from the user's text."
    output_key: extracted_facts
  - agent_class: LlmAgent
    name: summariser
    model: gemini-2.0-flash
    instruction: "Summarise the extracted facts into 3 bullet points."
    output_key: summary
```

```python
from google.adk.cli.utils.agent_loader import AgentLoader

loader = AgentLoader("agents/")
pipeline = loader.load_agent("pipeline")
print([a.name for a in pipeline.sub_agents])  # ["extractor", "summariser"]
```

### Example 4 — single-agent mode

Useful when deploying a single agent as a Cloud Run service:

```python
from google.adk.cli.utils.agent_loader import AgentLoader

# agents/my_agent/agent.py exists with root_agent defined
loader = AgentLoader("agents/my_agent/")
print(loader.is_single_agent)          # True
print(loader.single_agent_name)        # "my_agent"
agent = loader.load_agent("my_agent")  # equivalent to loader.load_agent(loader.single_agent_name)
```

### Example 5 — programmatic YAML parsing with `AgentConfig`

```python
import yaml
from google.adk.agents.agent_config import AgentConfig

yaml_str = """
agent_class: LlmAgent
name: classifier
model: gemini-2.0-flash
instruction: "Classify the user's intent as: order, support, or other."
output_schema:
  code: "from pydantic import BaseModel\nclass Intent(BaseModel):\n    category: str\n    confidence: float"
"""

config = AgentConfig.model_validate(yaml.safe_load(yaml_str))
print(type(config.root).__name__)   # LlmAgentConfig
print(config.root.name)             # "classifier"
print(config.root.model)            # "gemini-2.0-flash"

# Convert config back to a live agent
from google.adk.agents.config_agent_utils import build_agent_from_config
agent = build_agent_from_config(config.root)
```

### Agent name validation

`AgentLoader._validate_agent_name` enforces `^[a-zA-Z0-9_]+$` and rejects directory traversal and reserved names starting with `__` (unless loading from the special built-in agents directory):

```python
from google.adk.cli.utils.agent_loader import AgentLoader

loader = AgentLoader("agents/")

# Safe names
loader._validate_agent_name("my_agent")     # OK
loader._validate_agent_name("agent123")     # OK

# Rejected names
try:
    loader._validate_agent_name("../etc/passwd")
except ValueError as e:
    print(e)   # Invalid agent name: ...

try:
    loader._validate_agent_name("agent with spaces")
except ValueError as e:
    print(e)   # Invalid agent name: ...
```

---

## Quick reference

| Class | Module | Key feature |
|---|---|---|
| `MCPSessionManager` | `google.adk.tools.mcp_tool.mcp_session_manager` | Session pool + auto-reconnect keyed by header hash |
| `StdioConnectionParams` | same | Adds `timeout` to bare `StdioServerParameters` |
| `SseConnectionParams` | same | SSE with `sse_read_timeout` + custom httpx factory |
| `StreamableHTTPConnectionParams` | same | HTTP Streamable + `terminate_on_close` DELETE |
| `EventsCompactionConfig` | `google.adk.apps._configs` | Token-threshold + sliding-window compaction |
| `ResumabilityConfig` | same | At-least-once long-running tool resumption |
| `App` | `google.adk.apps.app` | Top-level container wiring all cross-cutting configs |
| `BaseLlmFlow` | `google.adk.flows.llm_flows.base_llm_flow` | Request + response processor chains |
| `SingleFlow` | same | Standard ADK processor pipeline |
| `AutoFlow` | `google.adk.flows.llm_flows.auto_flow` | `SingleFlow` + agent-transfer support |
| `SpannerToolset` | `google.adk.tools.spanner.spanner_toolset` | 8 read-only Spanner tools (`@experimental`) |
| `SpannerToolSettings` | `google.adk.tools.spanner.settings` | Row limits, dict mode, database role, vector settings |
| `SpannerVectorStoreSettings` | same | Managed vector store: model, columns, ANN config |
| `LlamaIndexRetrieval` | `google.adk.tools.retrieval.llama_index_retrieval` | Bridges LlamaIndex `BaseRetriever` into ADK |
| `BaseRetrievalTool` | `google.adk.tools.retrieval.base_retrieval_tool` | Base for custom retrieval tools |
| `convert_genai_part_to_a2a_part` | `google.adk.a2a.converters.part_converter` | GenAI Part → A2A Part |
| `convert_a2a_part_to_genai_part` | same | A2A Part → GenAI Part |
| `AdkEventToA2AEventsConverter` | `google.adk.a2a.converters.from_adk_event` | Type alias for custom ADK→A2A event converters |
| `GoogleTool` | `google.adk.tools.google_tool` | OAuth2-aware Google API tool (`@experimental`) |
| `BaseGoogleCredentialsConfig` | `google.adk.tools._google_credentials` | 3-way credential config: ADC / token-from-state / OAuth2 |
| `InterceptionResult` | `google.adk.workflow.utils._replay_interceptor` | Fast-forward / re-run decision for workflow replay |
| `_ChildScanState` | `google.adk.workflow.utils._rehydration_utils` | Per-node state accumulated during event scanning |
| `AgentLoader` | `google.adk.cli.utils.agent_loader` | Unified agent discovery (4 patterns + caching) |
| `AgentConfig` | `google.adk.agents.agent_config` | Pydantic discriminated union for YAML agent configs |

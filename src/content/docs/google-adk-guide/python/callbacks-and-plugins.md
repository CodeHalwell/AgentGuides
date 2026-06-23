---
title: "Callbacks and Plugins"
description: "Intercept agent, model, and tool execution — per-agent callbacks and runner-wide plugins."
framework: google-adk
language: python
sidebar:
  order: 40
---

Verified against google-adk==2.3.0 (`google/adk/agents/llm_agent.py`, `google/adk/plugins/`).

Callbacks and plugins are the two interception surfaces in ADK. **Callbacks** are configured per-agent. **Plugins** are configured per-runner and apply globally. Plugins run **before** agent callbacks at each hook point and short-circuit the chain if any one returns a non-`None` value (`plugins/base_plugin.py:41-71`).

## Minimal example

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.plugins import LoggingPlugin, BasePlugin
from google.adk.runners import InMemoryRunner

async def redact_secrets(tool, tool_args, tool_context):
    if tool_args.get("query", "").lower().startswith("password"):
        return {"error": "query rejected by policy"}
    return None

agent = LlmAgent(
    name="policy_aware",
    model="gemini-2.5-flash",
    instruction="Be helpful.",
    before_tool_callback=redact_secrets,   # per-agent
)

app = App(name="demo", root_agent=agent, plugins=[LoggingPlugin()])  # runner-wide
runner = InMemoryRunner(app=app)
```

## Per-agent callbacks

Set these as fields on `LlmAgent` (or `BaseAgent` for agent-level hooks). Each accepts **a single callable or a list** — the list is called in order until one returns non-`None` (`llm_agent.py:77-484`).

| Field | Signature | Return meaning |
|---|---|---|
| `before_agent_callback` | `(callback_context)` | `types.Content` → replace agent reply; `None` → proceed |
| `after_agent_callback` | `(callback_context)` | `types.Content` → override output; `None` → proceed |
| `before_model_callback` | `(callback_context, llm_request)` | `LlmResponse` → skip the LLM call; `None` → proceed |
| `after_model_callback` | `(callback_context, llm_response)` | `LlmResponse` → replace response; `None` → proceed |
| `on_model_error_callback` | `(callback_context, llm_request, error)` | `LlmResponse` → swallow the error; `None` → re-raise |
| `before_tool_callback` | `(tool, args, tool_context)` | `dict` → skip the tool, use dict as result; `None` → proceed |
| `after_tool_callback` | `(tool, args, tool_context, result)` | `dict` → replace result; `None` → proceed |
| `on_tool_error_callback` | `(tool, args, tool_context, error)` | `dict` → swallow the error; `None` → re-raise |

All callbacks may be sync or async.

```python
async def inject_context(callback_context, llm_request):
    user = callback_context.state.get("user_name", "anon")
    llm_request.config.system_instruction = f"User: {user}. {llm_request.config.system_instruction or ''}"
    return None

agent = LlmAgent(
    name="personalised",
    before_model_callback=inject_context,
)
```

### `CallbackContext` vs `ToolContext`

- `CallbackContext` — passed to agent- and model-level callbacks. Exposes `state`, `agent_name`, `invocation_id`, `session`, and read-only `user_content`.
- `ToolContext` — passed to tool callbacks. Extends `CallbackContext` with `function_call_id`, `actions`, `request_confirmation()`, and artifact helpers (`load_artifact`, `save_artifact`).

Both read and mutate **session state**. State keys with reserved prefixes behave differently:

| Prefix | Scope | Persisted |
|---|---|---|
| (none) | Session | yes |
| `app:` | All sessions in the app | yes |
| `user:` | All sessions of that user | yes |
| `temp:` | Current invocation only | no (stripped before commit) |

### `ReadonlyContext` — for instruction providers and toolsets

`ReadonlyContext` is the lightest context variant. It is passed to:
- Dynamic instruction providers (`instruction=callable` on `LlmAgent`)
- `BaseToolset.get_tools(readonly_context=...)`

It exposes only reads — no `actions`, no mutation of state.

```python
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents import LlmAgent
from google.adk.tools.base_toolset import BaseToolset

# --- Dynamic instruction from session state -----------------------------------
async def personalised_instruction(ctx: ReadonlyContext) -> str:
    name = ctx.state.get("user:display_name", "there")
    lang = ctx.state.get("user:lang", "en")
    return (
        f"Hello {name}! Always respond in language '{lang}'. "
        f"Invocation: {ctx.invocation_id[:8]}."
    )

agent = LlmAgent(
    name="greeter",
    model="gemini-2.5-flash",
    instruction=personalised_instruction,  # called every turn
)

# --- Role-gated toolset -------------------------------------------------------
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from typing import Optional

class RoleGatedToolset(BaseToolset):
    async def get_tools(
        self, readonly_context: Optional[ReadonlyContext] = None
    ) -> list[BaseTool]:
        role = "guest"
        if readonly_context:
            role = readonly_context.state.get("user:role", "guest")
        tools: list[BaseTool] = [FunctionTool(func=self._read)]
        if role in ("editor", "admin"):
            tools.append(FunctionTool(func=self._write))
        return tools

    async def _read(self, key: str) -> str:
        """Read a shared key."""
        return _store.get(key, "")

    async def _write(self, key: str, value: str) -> dict:
        """Write a shared key."""
        _store[key] = value
        return {"ok": True}

    async def close(self) -> None:
        pass

_store: dict = {}
```

**Available on `ReadonlyContext`:**

| Member | Type | Purpose |
|---|---|---|
| `ctx.state` | `MappingProxyType` | Read-only session state snapshot |
| `ctx.user_content` | `types.Content \| None` | The original user message for this turn |
| `ctx.invocation_id` | `str` | ID of the current invocation |
| `ctx.agent_name` | `str` | Name of the currently running agent |
| `ctx.user_id` | `str` | Current user ID |
| `ctx.session` | `Session` | Full session object (read-only use) |
| `ctx.run_config` | `RunConfig \| None` | Per-run config |
| `ctx.get_credential(key)` | `AuthCredential \| None` | Retrieve a resolved credential by key |

## Runner-wide plugins

Subclass `BasePlugin` and register via `App(plugins=[...])`.

```python
from google.adk.plugins import BasePlugin
from google.genai import types

class BudgetPlugin(BasePlugin):
    def __init__(self, max_tokens: int):
        super().__init__(name="budget")
        self.max_tokens = max_tokens
        self.spent = 0

    async def after_model_callback(self, *, callback_context, llm_response):
        if llm_response.usage_metadata:
            self.spent += llm_response.usage_metadata.total_token_count or 0
        if self.spent >= self.max_tokens:
            return llm_response.__class__(
                content=types.Content(role="model", parts=[types.Part(text="Budget hit.")])
            )
        return None
```

### Full lifecycle example

A plugin that uses the full hook surface to log every invocation with timing and block a disallowed tool (`drop_table`). Derived from the `BasePlugin` source in `plugins/base_plugin.py`:

```python
import asyncio
import logging
import time
from typing import Any, Optional

from google.adk.agents import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

logger = logging.getLogger(__name__)

BLOCKED_TOOLS = {"drop_table", "delete_database", "wipe_storage"}

class AuditPlugin(BasePlugin):
    """Logs invocations and blocks dangerous tools."""

    def __init__(self):
        super().__init__(name="audit")
        self._start_times: dict[str, float] = {}
        self._lock = asyncio.Lock()

    # ── Invocation lifecycle ─────────────────────────────────────────────────

    async def on_user_message_callback(
        self, *, invocation_context: InvocationContext, user_message: types.Content
    ) -> Optional[types.Content]:
        text = "".join(p.text or "" for p in (user_message.parts or []))
        logger.info("[audit] user(%s) → %s", invocation_context.session.id, text[:120])
        return None  # proceed normally

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> Optional[types.Content]:
        iid = invocation_context.invocation_id
        async with self._lock:
            self._start_times[iid] = time.monotonic()
        return None

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Event
    ) -> Optional[Event]:
        return None  # let the original event through

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        iid = invocation_context.invocation_id
        async with self._lock:
            elapsed = time.monotonic() - self._start_times.pop(iid, time.monotonic())
        logger.info("[audit] invocation %s finished in %.2fs", iid, elapsed)

    async def close(self) -> None:
        logger.info("[audit] plugin closed — flushing logs")

    # ── Tool policy ──────────────────────────────────────────────────────────

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict]:
        if tool.name in BLOCKED_TOOLS:
            logger.warning("[audit] BLOCKED tool call: %s(%s)", tool.name, tool_args)
            return {"error": f"Tool '{tool.name}' is blocked by policy."}
        logger.info("[audit] tool call: %s(%s)", tool.name, tool_args)
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> Optional[dict]:
        logger.info("[audit] tool result: %s → %s", tool.name, str(result)[:200])
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> Optional[dict]:
        logger.error("[audit] tool error: %s → %s: %s", tool.name, type(error).__name__, error)
        return None  # re-raise by returning None

    # ── Model monitoring ─────────────────────────────────────────────────────

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        if llm_response.usage_metadata:
            logger.info(
                "[audit] tokens — in: %d, out: %d",
                llm_response.usage_metadata.prompt_token_count or 0,
                llm_response.usage_metadata.candidates_token_count or 0,
            )
        return None


# Register on App
from google.adk.apps import App
from google.adk.plugins import LoggingPlugin

app = App(
    name="my_app",
    root_agent=my_agent,
    plugins=[AuditPlugin(), LoggingPlugin()],  # AuditPlugin runs first
)
```

`BasePlugin` methods default to `pass` (returning `None`). Only implement the hooks you need.

Plugins can implement any subset of the hooks below (`plugins/base_plugin.py`):

| Hook | Fires |
|---|---|
| `on_user_message_callback(*, invocation_context, user_message)` | When the runner receives the user message, before anything else |
| `before_run_callback(*, invocation_context)` | Once per invocation, first hook after the user message is appended |
| `on_event_callback(*, invocation_context, event)` | For every event before it is persisted and yielded |
| `after_run_callback(*, invocation_context)` | Last hook — for cleanup/metrics |
| `before_agent_callback(*, agent, callback_context)` / `after_agent_callback` | Wraps each agent |
| `before_model_callback(*, callback_context, llm_request)` / `after_model_callback` / `on_model_error_callback` | Wraps each model call |
| `before_tool_callback(*, tool, tool_args, tool_context)` / `after_tool_callback` / `on_tool_error_callback` | Wraps each tool call |
| `close()` | When the runner is closed (`runner.close()`) |

All are `async def`. All return `Optional[<relevant type>]` — non-`None` short-circuits the chain.

## Built-in plugins

ADK ships the following plugins in `google.adk.plugins` (verified from `plugins/__init__.py` and each module file for google-adk==2.3.0):

### `LoggingPlugin`

Emits structured logs for every model/tool/agent event via the `google_adk` logger hierarchy. Drop-in for observability with no configuration.

```python
from google.adk.plugins import LoggingPlugin
from google.adk.apps import App

app = App(name="demo", root_agent=agent, plugins=[LoggingPlugin()])
```

Configure the log level via standard Python logging:

```python
import logging
logging.getLogger("google_adk").setLevel(logging.DEBUG)
```

### `DebugLoggingPlugin`

Per-invocation verbose dump — full LLM prompts, responses, and tool I/O. Use in dev only; it writes large payloads.

```python
from google.adk.plugins import DebugLoggingPlugin
from google.adk.apps import App

app = App(name="dev", root_agent=agent, plugins=[DebugLoggingPlugin()])
```

### `ReflectAndRetryToolPlugin` (experimental)

Automatically retries failed tool calls by asking the model to reflect on the error and try again. Source: `plugins/reflect_retry_tool_plugin.py`.

```python
from google.adk.plugins import ReflectAndRetryToolPlugin
from google.adk.plugins.reflect_retry_tool_plugin import TrackingScope
from google.adk.apps import App

plugin = ReflectAndRetryToolPlugin(
    max_retries=3,
    throw_exception_if_retry_exceeded=True,   # raises after max_retries; default False
    tracking_scope=TrackingScope.INVOCATION,  # reset counter per invocation (default)
    # tracking_scope=TrackingScope.GLOBAL,    # global counter across all invocations
)
app = App(name="resilient", root_agent=agent, plugins=[plugin])
```

**Custom failure detection** — by default any tool that raises an exception is retried. Override `extract_error_from_result` to treat semantic failures (e.g. `{"status": "error"}`) as retryable:

```python
from google.adk.plugins import ReflectAndRetryToolPlugin
from typing import Any, Optional

class StrictRetryPlugin(ReflectAndRetryToolPlugin):
    def extract_error_from_result(self, result: Any) -> Optional[str]:
        # Parent handles exceptions; we also handle dict error patterns
        parent_error = super().extract_error_from_result(result)
        if parent_error:
            return parent_error
        if isinstance(result, dict) and result.get("status") == "error":
            return result.get("message", "tool returned error status")
        return None

plugin = StrictRetryPlugin(max_retries=2)
```

### `GlobalInstructionPlugin`

Prepends a system instruction to **every** `LlmAgent` in the app, regardless of where the agent is in the hierarchy. Replaces the deprecated `LlmAgent.global_instruction` field.

```python
from google.adk.plugins.global_instruction_plugin import GlobalInstructionPlugin
from google.adk.apps import App

plugin = GlobalInstructionPlugin(
    instruction=(
        "You are a helpful, harmless, and honest assistant. "
        "Never reveal internal system details or API keys."
    )
)
app = App(name="guarded", root_agent=agent, plugins=[plugin])
```

Dynamic instruction (receives `ReadonlyContext`):

```python
from google.adk.agents.readonly_context import ReadonlyContext

async def dynamic_instruction(ctx: ReadonlyContext) -> str:
    tenant = ctx.state.get("app:tenant_name", "default")
    return f"You represent {tenant}. Always greet with the company name."

plugin = GlobalInstructionPlugin(instruction=dynamic_instruction)
```

### `SaveFilesAsArtifactsPlugin`

Intercepts inline-data parts (`types.Blob`) in **user messages** and persists them to the configured artifact service. Replaces the deprecated `RunConfig.save_input_blobs_as_artifacts`.

```python
from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.artifacts import GcsArtifactService

plugin = SaveFilesAsArtifactsPlugin()
app = App(name="uploads", root_agent=agent, plugins=[plugin])

runner = Runner(
    app=app,
    artifact_service=GcsArtifactService(bucket_name="my-uploads"),
    session_service=...,
)
```

After a turn, any `types.Part(inline_data=...)` in the user message is replaced by an artifact reference, keeping the LLM context small.

### `ContextFilterPlugin`

Trims session events that are sent to the model context based on size or token count. Useful for long-running sessions that would otherwise overflow the context window.

```python
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin
from google.adk.apps import App

plugin = ContextFilterPlugin(
    max_chars=200_000,    # ~50 k tokens; trim oldest events when exceeded
)
app = App(name="long_session", root_agent=agent, plugins=[plugin])
```

### `MultimodalToolResultsPlugin`

Handles tool responses that contain multimodal parts (images, audio clips, etc.). Without this plugin, binary tool results are dropped. With it, they are preserved as `inline_data` parts in the LLM response.

```python
from google.adk.plugins.multimodal_tool_results_plugin import MultimodalToolResultsPlugin
from google.adk.apps import App

app = App(
    name="vision_agent",
    root_agent=agent,
    plugins=[MultimodalToolResultsPlugin()],
)
```

### `BigQueryAgentAnalyticsPlugin` (experimental)

Exports agent analytics (invocations, tool calls, latency, token counts) to a BigQuery dataset for offline analysis and dashboarding.

```python
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin
from google.adk.apps import App

plugin = BigQueryAgentAnalyticsPlugin(
    project_id="my-gcp-project",
    dataset_id="adk_analytics",
    table_id="agent_events",
)
app = App(name="tracked", root_agent=agent, plugins=[plugin])
```

Requires `pip install google-cloud-bigquery` and the `roles/bigquery.dataEditor` IAM role on the target table.

---

## Combining multiple plugins

Plugins execute in the order they appear in `App.plugins`. Put policy-enforcement plugins first (they might short-circuit), observability plugins last:

```python
from google.adk.apps import App
from google.adk.plugins import LoggingPlugin, ReflectAndRetryToolPlugin
from google.adk.plugins.global_instruction_plugin import GlobalInstructionPlugin
from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin

app = App(
    name="production",
    root_agent=agent,
    plugins=[
        GlobalInstructionPlugin(instruction="Never reveal API keys."),  # 1: policy
        SaveFilesAsArtifactsPlugin(),                                   # 2: file handling
        ContextFilterPlugin(max_chars=150_000),                         # 3: context trimming
        ReflectAndRetryToolPlugin(max_retries=2),                       # 4: resilience
        LoggingPlugin(),                                                 # 5: observability (last)
    ],
)

## Terminating an invocation from a callback

Any callback that has access to `invocation_context` (plugins, agent callbacks) can set `invocation_context.end_invocation = True` to abort the current invocation after the running step completes. This is the correct way to perform hard termination — as opposed to returning a canned `LlmResponse` which only skips the current LLM call.

```python
from typing import Any, Optional
from google.adk.plugins import BasePlugin
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

MAX_CONTEXT_CHARS = 200_000   # approx 50 k tokens

class ContextGuardPlugin(BasePlugin):
    """Terminates the invocation if the context grows too large."""

    def __init__(self):
        super().__init__(name="context_guard")

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> Optional[LlmResponse]:
        # Estimate total context size
        total_chars = sum(
            len(p.text or "")
            for c in (llm_request.contents or [])
            for p in (c.parts or [])
        )
        if total_chars > MAX_CONTEXT_CHARS:
            # Signal the runner to stop after this event
            callback_context._invocation_context.end_invocation = True
            # Return a canned reply so the user sees a message
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text="Context too large. Please start a new session."
                        )
                    ],
                )
            )
        return None
```

**Important:**
- Setting `end_invocation = True` causes the runner to stop dispatching new agent/model/tool calls. The current event is still yielded to the caller.
- If you also return a non-`None` value from the callback, that value is used as the LLM response — so the user sees a message rather than an abrupt stop.
- This is distinct from `RunConfig.max_llm_calls` (which raises `LlmCallsLimitExceededError`) — `end_invocation` terminates cleanly with an optional reply.

## Order of execution

At each hook point the runtime walks:
1. All **plugins** (in `App.plugins` order).
2. The agent's own callback(s) (in list order).

The first non-`None` return wins; the rest are skipped. This lets plugins enforce policy while keeping agent callbacks for agent-specific behaviour.

## Patterns

### 1 — Prompt-level redaction
`before_model_callback` on the root agent. Strip PII from `llm_request.contents`. Keep the plugin version of the same logic for a secondary net.

### 2 — Response budgeting
`after_model_callback` plugin accumulates `llm_response.usage_metadata`. When budget is blown, return a canned `LlmResponse` to short-circuit the session.

### 3 — Tool policy
`before_tool_callback` plugin validates `tool_args` (e.g. forbid `DROP TABLE`). Return `{"error": "..."}` to block without raising.

### 4 — Self-healing tools
Register `ReflectAndRetryToolPlugin` at the app level. Flaky tools get up to `max_retries` automatic retries with reflection prompts.

### 5 — Observability
`on_event_callback` on a plugin → push every event to OpenTelemetry / Cloud Trace. Use `after_run_callback` to flush.

## Gotchas

- A callback that returns **any non-None** value short-circuits the chain — `return None` is required to let the chain continue. Returning `False`, `0`, or `{}` also short-circuits because the runtime checks `is not None`, not truthiness.
- Plugins execute before agent callbacks at the same hook. A plugin returning non-`None` prevents the agent's own callback from running.
- `LlmAgent.global_instruction` is deprecated — migrate to `GlobalInstructionPlugin`.
- `RunConfig.save_input_blobs_as_artifacts` is deprecated — migrate to `SaveFilesAsArtifactsPlugin`.
- Stateful plugins should guard mutation with `asyncio.Lock` — concurrent invocations share the plugin instance.
- `callback_context.state[...] = value` is persisted to session state on the next `append_event`. Use `temp:` prefix for scratch values that must not persist.

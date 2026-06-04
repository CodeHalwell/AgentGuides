---
title: "Class deep dives — volume 12 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.1.0 classes: BasePlugin/PluginManager (12-callback lifecycle + early-exit chain), ContextFilterPlugin (sliding-window context pruning), ReflectAndRetryToolPlugin (self-healing tool retry), GlobalInstructionPlugin (app-level system instructions), SaveFilesAsArtifactsPlugin (auto-upload to artifact service), MultimodalToolResultsPlugin (raw Part tool returns), DebugLoggingPlugin (YAML trace capture), RunConfig/StreamingMode (complete runtime config with SSE patterns + ToolThreadPoolConfig), AuthenticatedFunctionTool (transparent per-tool auth), FeatureName/override_feature_enabled/temporary_feature_override (feature flag system)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 12"
  order: 77
---

Source-verified against **google-adk==2.1.0** (installed from PyPI, June 2026). Every field name, signature, and code example is drawn from the installed package source at `/usr/local/lib/python3.11/dist-packages/google/adk/`.

| # | Class / group | Module | Status |
|---|---|---|---|
| 1 | `BasePlugin` + `PluginManager` | `google.adk.plugins.base_plugin`, `.plugin_manager` | Stable |
| 2 | `ContextFilterPlugin` | `google.adk.plugins.context_filter_plugin` | Stable |
| 3 | `ReflectAndRetryToolPlugin` + `TrackingScope` + `ToolFailureResponse` | `google.adk.plugins.reflect_retry_tool_plugin` | Experimental |
| 4 | `GlobalInstructionPlugin` | `google.adk.plugins.global_instruction_plugin` | Stable |
| 5 | `SaveFilesAsArtifactsPlugin` | `google.adk.plugins.save_files_as_artifacts_plugin` | Stable |
| 6 | `MultimodalToolResultsPlugin` | `google.adk.plugins.multimodal_tool_results_plugin` | Stable |
| 7 | `DebugLoggingPlugin` | `google.adk.plugins.debug_logging_plugin` | Stable |
| 8 | `RunConfig` + `StreamingMode` + `ToolThreadPoolConfig` | `google.adk.agents.run_config` | Stable |
| 9 | `AuthenticatedFunctionTool` | `google.adk.tools.authenticated_function_tool` | Experimental |
| 10 | `FeatureName` + `override_feature_enabled` + `temporary_feature_override` | `google.adk.features._feature_registry` | Stable |

---

## 1 · `BasePlugin` + `PluginManager`

**Source:** `google.adk.plugins.base_plugin`, `google.adk.plugins.plugin_manager`

Plugins are the **application-level** counterpart to agent callbacks. Where callbacks are attached to a single `LlmAgent`, plugins are registered on the `Runner` and fire for every agent in the entire application. ADK executes all plugin callbacks before the corresponding agent callbacks; if any plugin returns a non-`None` value the remaining plugins *and* the agent callback are skipped (early-exit).

### The 12 lifecycle callbacks + `close()` (source-verified)

```python
class BasePlugin(ABC):
    # ── Invocation lifecycle ─────────────────────────────────
    async def on_user_message_callback(
        self, *, invocation_context: InvocationContext, user_message: types.Content
    ) -> Optional[types.Content]: ...
    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> Optional[types.Content]: ...
    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None: ...
    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Event
    ) -> Optional[Event]: ...

    # ── Agent lifecycle ──────────────────────────────────────
    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]: ...
    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]: ...

    # ── LLM lifecycle ────────────────────────────────────────
    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]: ...
    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]: ...
    async def on_model_error_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest, error: Exception
    ) -> Optional[LlmResponse]: ...

    # ── Tool lifecycle ───────────────────────────────────────
    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> Optional[dict]: ...
    async def after_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext, result: dict
    ) -> Optional[dict]: ...
    async def on_tool_error_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext, error: Exception
    ) -> Optional[dict]: ...

    # ── Resource management ──────────────────────────────────
    async def close(self) -> None: ...
```

### `PluginManager` — registration and execution

`PluginManager` is an internal class (instantiated by `Runner`) that orchestrates plugin execution. You interact with it indirectly by passing a list of plugins to `Runner`.

```python
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.plugins.base_plugin import BasePlugin

# Internal wiring (shown for understanding — you don't call PluginManager directly)
manager = PluginManager(plugins=[plugin_a, plugin_b], close_timeout=5.0)

# Duplicate names raise ValueError immediately
manager.register_plugin(plugin_a)  # OK
manager.register_plugin(plugin_a)  # raises ValueError: already registered

# Retrieval
found = manager.get_plugin("my_plugin_name")  # None if not found
```

### Minimal custom plugin

Only implement the callbacks you need — the base class provides `pass` defaults for all others.

```python
from typing import Any, Optional
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext


class AuditPlugin(BasePlugin):
    def __init__(self):
        super().__init__(name="audit")
        self._calls: list[str] = []

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> Optional[dict]:
        self._calls.append(f"{tool.name}({tool_args})")
        return None  # don't short-circuit

    async def close(self) -> None:
        print(f"[audit] Total tool calls this session: {len(self._calls)}")
```

### Registering plugins on the `Runner`

```python
from google.adk.runners import Runner
from google.adk.agents import LlmAgent

agent = LlmAgent(name="assistant", model="gemini-2.0-flash")
runner = Runner(
    agent=agent,
    app_name="my_app",
    session_service=...,
    plugins=[AuditPlugin()],
)
```

### Early-exit semantics

If `before_tool_callback` on the first plugin returns a dict, that dict becomes the tool result immediately. `PluginManager._run_callbacks` stops iterating and logs:

> `Plugin 'X' returned a value for callback 'before_tool_callback', exiting early.`

This lets you implement tool caching, policy enforcement, or test stubs at the plugin layer without touching agent code.

---

## 2 · `ContextFilterPlugin`

**Source:** `google.adk.plugins.context_filter_plugin`

`ContextFilterPlugin` trims the conversation history sent to the LLM before every model call, keeping only the most recent *N invocations*. It is essential for long-running agents that would otherwise exceed the model context window.

### Constructor (source-verified)

```python
class ContextFilterPlugin(BasePlugin):
    def __init__(
        self,
        num_invocations_to_keep: Optional[int] = None,
        custom_filter: Optional[Callable[[list[types.Content]], list[types.Content]]] = None,
        name: str = "context_filter_plugin",
    ): ...
```

| Parameter | Purpose |
|---|---|
| `num_invocations_to_keep` | Positive integer: keep only the last N user-initiated turns |
| `custom_filter` | `Callable[[list[Content]], list[Content]]` — full control over history |
| `name` | Plugin instance name (must be unique per Runner) |

An **invocation** is defined as one or more consecutive human user messages (role=`"user"`) that are not function responses. Tool outputs (`role="user"` + `function_response` parts) are transparent to the boundary detection.

### Orphan-safe splitting

The plugin contains `_adjust_split_index_to_avoid_orphaned_function_responses` which walks backwards from the candidate cut point to ensure every retained `function_response` part has its matching `function_call` — preventing the model from seeing an answer without the corresponding question.

### Usage patterns

**Sliding-window (keep last 5 conversations):**

```python
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin
from google.adk.runners import Runner

runner = Runner(
    agent=agent,
    app_name="my_app",
    session_service=session_service,
    plugins=[ContextFilterPlugin(num_invocations_to_keep=5)],
)
```

**Custom filter — strip large tool payloads before sending to model:**

```python
from google.genai import types

def strip_large_tool_payloads(contents: list[types.Content]) -> list[types.Content]:
    """Replace function_response payloads > 2 KB with a summary stub."""
    result = []
    for content in contents:
        if not content.parts:
            result.append(content)
            continue
        new_parts = []
        for part in content.parts:
            if part.function_response:
                resp = part.function_response.response or {}
                raw = str(resp)
                if len(raw) > 2048:
                    new_parts.append(types.Part(
                        function_response=types.FunctionResponse(
                            id=part.function_response.id,
                            name=part.function_response.name,
                            response={"summary": f"[truncated — {len(raw)} chars]"},
                        )
                    ))
                    continue
            new_parts.append(part)
        result.append(types.Content(role=content.role, parts=new_parts))
    return result

runner = Runner(
    agent=agent,
    app_name="my_app",
    session_service=session_service,
    plugins=[ContextFilterPlugin(custom_filter=strip_large_tool_payloads)],
)
```

**Combining both — sliding window then payload trim:**

```python
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin

runner = Runner(
    agent=agent,
    app_name="my_app",
    session_service=session_service,
    plugins=[
        ContextFilterPlugin(
            num_invocations_to_keep=10,
            custom_filter=strip_large_tool_payloads,
        )
    ],
)
```

Both transformations are applied in sequence inside the single `before_model_callback`.

---

## 3 · `ReflectAndRetryToolPlugin` + `TrackingScope` + `ToolFailureResponse`

**Source:** `google.adk.plugins.reflect_retry_tool_plugin`

`ReflectAndRetryToolPlugin` intercepts tool failures and returns a structured `ToolFailureResponse` to the LLM containing the error details, the arguments that were used, and explicit reflection guidance. This lets the model self-correct without the entire invocation failing.

### Constructor (source-verified)

```python
@experimental  # from google.adk.utils.feature_decorator — warns on instantiation
class ReflectAndRetryToolPlugin(BasePlugin):
    def __init__(
        self,
        name: str = "reflect_retry_tool_plugin",
        max_retries: int = 3,
        throw_exception_if_retry_exceeded: bool = True,
        tracking_scope: TrackingScope = TrackingScope.INVOCATION,
    ): ...
```

| Parameter | Default | Purpose |
|---|---|---|
| `max_retries` | `3` | Consecutive failures allowed before giving up (`0` = no retries) |
| `throw_exception_if_retry_exceeded` | `True` | Raise the final exception, or return a "give-up" guidance dict |
| `tracking_scope` | `INVOCATION` | Lifetime of failure counters |

### `TrackingScope` enum

```python
class TrackingScope(Enum):
    INVOCATION = "invocation"  # reset counters per invocation_id (default)
    GLOBAL = "global"          # counters accumulate for the runner's lifetime
```

### `ToolFailureResponse` model

```python
class ToolFailureResponse(BaseModel):
    response_type: str = REFLECT_AND_RETRY_RESPONSE_TYPE  # sentinel
    error_type: str = ""
    error_details: str = ""
    retry_count: int = 0
    reflection_guidance: str = ""
```

The LLM receives this as the tool result and is expected to read `reflection_guidance` before retrying.

### Basic usage

```python
from google.adk.plugins.reflect_retry_tool_plugin import (
    ReflectAndRetryToolPlugin,
    TrackingScope,
)

runner = Runner(
    agent=agent,
    app_name="my_app",
    session_service=session_service,
    plugins=[
        ReflectAndRetryToolPlugin(
            max_retries=3,
            throw_exception_if_retry_exceeded=False,  # return guidance instead of raising
        )
    ],
)
```

### Detecting errors in *successful* tool responses

Some APIs return HTTP 200 with a JSON error body. Override `extract_error_from_result` to trigger retry logic in those cases:

```python
from typing import Any, Optional
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext


class ApiErrorRetryPlugin(ReflectAndRetryToolPlugin):
    async def extract_error_from_result(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: Any,
    ) -> Optional[dict]:
        # API returns {"status": "error", "message": "..."} on failure
        if isinstance(result, dict) and result.get("status") == "error":
            return result
        return None  # no error detected

runner = Runner(
    agent=agent,
    app_name="my_app",
    session_service=session_service,
    plugins=[ApiErrorRetryPlugin(max_retries=2)],
)
```

### Custom per-user scoping

Override `_get_scope_key` to track failures on a per-user or per-session basis rather than per-invocation:

```python
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin
from google.adk.tools.tool_context import ToolContext


class PerUserRetryPlugin(ReflectAndRetryToolPlugin):
    def _get_scope_key(self, tool_context: ToolContext) -> str:
        # Use user_id as the scope — reset only when that user succeeds
        return tool_context.user_id or tool_context.invocation_id
```

### Concurrency safety

Failure counters are protected by `asyncio.Lock`. When multiple tools run in parallel the lock ensures each counter increment is atomic:

```python
async with self._lock:
    current_retries = tool_failure_counter.get(tool.name, 0) + 1
    tool_failure_counter[tool.name] = current_retries
```

---

## 4 · `GlobalInstructionPlugin`

**Source:** `google.adk.plugins.global_instruction_plugin`

`GlobalInstructionPlugin` injects a system instruction **in front of** every agent's own `system_instruction`, regardless of how many agents are in the graph. It is the successor to the deprecated `global_instruction` field on `LlmAgent`.

### Constructor (source-verified)

```python
class GlobalInstructionPlugin(BasePlugin):
    def __init__(
        self,
        global_instruction: Union[str, InstructionProvider] = "",
        name: str = "global_instruction",
    ) -> None: ...
```

`InstructionProvider` is `Callable[[ReadonlyContext], Union[str, Awaitable[str]]]` — the same signature as the per-agent `instruction` callable.

### Static string instruction

```python
from google.adk.plugins.global_instruction_plugin import GlobalInstructionPlugin

runner = Runner(
    agent=root_agent,
    app_name="enterprise_app",
    session_service=session_service,
    plugins=[
        GlobalInstructionPlugin(
            global_instruction=(
                "You are an enterprise assistant. Always respond in formal English. "
                "Do not reveal internal system names or credentials. "
                "If you cannot help, escalate to the support team."
            )
        )
    ],
)
```

### Dynamic instruction with session state injection

String instructions support `{state_key}` substitution via `instructions_utils.inject_session_state`:

```python
plugin = GlobalInstructionPlugin(
    global_instruction=(
        "The current user is {user_name} with role {user_role}. "
        "Always address them by name and respect their access level."
    )
)
# At runtime, {user_name} and {user_role} are resolved from session.state
```

### Callable (async) instruction provider

```python
from google.adk.agents.readonly_context import ReadonlyContext

async def tenant_instruction(ctx: ReadonlyContext) -> str:
    tenant_id = ctx.state.get("tenant_id", "default")
    # Could fetch from a config service
    config = await fetch_tenant_config(tenant_id)
    return f"You are operating for tenant '{config.name}'. Locale: {config.locale}."

plugin = GlobalInstructionPlugin(global_instruction=tenant_instruction)
```

### Prepend semantics (source-verified)

The plugin prepends `global_instruction` before any existing `system_instruction`:

```python
# Inside before_model_callback:
if isinstance(existing_instruction, str):
    llm_request.config.system_instruction = (
        f"{final_global_instruction}\n\n{existing_instruction}"
    )
else:  # Iterable[str]
    new_instruction_list = [final_global_instruction]
    new_instruction_list.extend(list(existing_instruction))
    llm_request.config.system_instruction = new_instruction_list
```

---

## 5 · `SaveFilesAsArtifactsPlugin`

**Source:** `google.adk.plugins.save_files_as_artifacts_plugin`

When users upload files in a chat UI, those files arrive as `inline_data` blobs inside `types.Content`. `SaveFilesAsArtifactsPlugin` intercepts these blobs, saves them to the configured `ArtifactService`, and replaces the inline data with either a placeholder text part or a GCS/HTTPS file reference that the model can access directly.

### Constructor (source-verified)

```python
class SaveFilesAsArtifactsPlugin(BasePlugin):
    def __init__(
        self,
        name: str = "save_files_as_artifacts_plugin",
        *,
        attach_file_reference: bool = True,
    ): ...
```

| Parameter | Default | Purpose |
|---|---|---|
| `attach_file_reference` | `True` | Append a `FileData` part pointing to the saved artifact URI so the model can read it |

### Session vs user-scoped artifacts

The scoping follows `Blob.display_name`:

- `"report.pdf"` → session-scoped (deleted when session ends)
- `"user:resume.pdf"` → user-scoped (persists across sessions)

### Placeholder mechanics

For every `inline_data` part the plugin:
1. Saves the blob to `artifact_service.save_artifact(...)` → returns a `version` int.
2. Appends `types.Part(text='[Uploaded Artifact: "report.pdf"]')` to the message.
3. If `attach_file_reference=True` and the artifact's `canonical_uri` is model-accessible (`gs://`, `https://`, `http://`), appends a `types.Part(file_data=FileData(file_uri=...))`.

The model sees the placeholder text and the file reference, not the raw bytes — keeping the context window compact.

### Setup with `GcsArtifactService`

```python
from google.adk.artifacts import GcsArtifactService
from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
from google.adk.runners import Runner

artifact_service = GcsArtifactService(bucket_name="my-agent-artifacts")

runner = Runner(
    agent=agent,
    app_name="chat_app",
    session_service=session_service,
    artifact_service=artifact_service,
    plugins=[SaveFilesAsArtifactsPlugin()],
)
```

### Receiving files in tool code

After the plugin runs, tool code retrieves the artifact by name:

```python
from google.adk.tools.tool_context import ToolContext

async def process_document(filename: str, tool_context: ToolContext) -> dict:
    artifact = await tool_context.load_artifact(filename)
    if artifact is None:
        return {"error": f"Artifact '{filename}' not found."}
    # artifact.inline_data.data  — raw bytes
    # artifact.inline_data.mime_type
    return {"size_bytes": len(artifact.inline_data.data)}
```

### Suppressing the file reference (privacy mode)

Set `attach_file_reference=False` when the artifact URI must not be exposed to the model (e.g. for compliance reasons). The model knows the file was uploaded via the placeholder text but cannot read it directly.

```python
plugins=[SaveFilesAsArtifactsPlugin(attach_file_reference=False)]
```

---

## 6 · `MultimodalToolResultsPlugin`

**Source:** `google.adk.plugins.multimodal_tool_results_plugin`

Normally, an ADK tool must return a `dict`. `MultimodalToolResultsPlugin` relaxes this: a tool can return a `types.Part` or a `list[types.Part]` directly, and the plugin bridges these into the LLM context via session state.

### How it works (source-verified)

`after_tool_callback` detects non-dict returns:

```python
if not (
    isinstance(result, types.Part)
    or isinstance(result, list) and result and isinstance(result[0], types.Part)
):
    return result  # passthrough unchanged

parts = [result] if isinstance(result, types.Part) else result[:]
# Accumulate in state under the sentinel key
tool_context.state[PARTS_RETURNED_BY_TOOLS_ID] += parts
return None  # suppress the original dict result
```

`before_model_callback` then appends the saved parts to the last content in `llm_request.contents`:

```python
if saved_parts := callback_context.state.get(PARTS_RETURNED_BY_TOOLS_ID, None):
    llm_request.contents[-1].parts += saved_parts
    callback_context.state.update({PARTS_RETURNED_BY_TOOLS_ID: []})
```

### Tool returning an image part

```python
from google.genai import types
from google.adk.tools.tool_context import ToolContext
from google.adk.plugins.multimodal_tool_results_plugin import MultimodalToolResultsPlugin

async def capture_screenshot(url: str, tool_context: ToolContext) -> types.Part:
    image_bytes = await render_page_to_png(url)
    return types.Part(
        inline_data=types.Blob(mime_type="image/png", data=image_bytes)
    )

runner = Runner(
    agent=agent,
    app_name="vision_app",
    session_service=session_service,
    plugins=[MultimodalToolResultsPlugin()],
)
```

### Tool returning multiple parts

```python
async def extract_frames(video_url: str, tool_context: ToolContext) -> list[types.Part]:
    frames = await sample_video_frames(video_url, n=5)
    return [
        types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=frame))
        for frame in frames
    ]
```

> **Note:** This plugin is a temporary bridge while the Gemini API's function-response protocol catches up to supporting `FunctionResponsePart` natively (tracked in [adk-python#3064](https://github.com/google/adk-python/issues/3064)). It will eventually be removed in favour of first-class multimodal tool responses.

---

## 7 · `DebugLoggingPlugin`

**Source:** `google.adk.plugins.debug_logging_plugin`

`DebugLoggingPlugin` implements **all 12 lifecycle callbacks** (every `BasePlugin` method except `close()`) to produce a comprehensive YAML trace of every agent invocation. Each invocation is appended as a separate YAML document (delimited by `---`) to the configured output file.

### Constructor (source-verified)

```python
class DebugLoggingPlugin(BasePlugin):
    def __init__(
        self,
        *,
        name: str = "debug_logging_plugin",
        output_path: str = "adk_debug.yaml",
        include_session_state: bool = True,
        include_system_instruction: bool = True,
    ): ...
```

| Parameter | Default | Purpose |
|---|---|---|
| `output_path` | `"adk_debug.yaml"` | File to append trace YAML to |
| `include_session_state` | `True` | Snapshot of `session.state` at end of each invocation |
| `include_system_instruction` | `True` | Full system instruction in `llm_request` entries |

### What each callback captures

| Callback | Entry type written | Key fields |
|---|---|---|
| `on_user_message_callback` | `user_message` | serialized `Content` |
| `before_run_callback` | `invocation_start` | agent name, branch |
| `before_agent_callback` | `agent_start` | agent name, branch |
| `before_model_callback` | `llm_request` | model, content count, full contents, tools, config |
| `after_model_callback` | `llm_response` | content, partial, usage metadata, finish_reason |
| `on_model_error_callback` | `llm_error` | error_type, error_message, model |
| `before_tool_callback` | `tool_call` | tool_name, function_call_id, args |
| `after_tool_callback` | `tool_response` | tool_name, function_call_id, result |
| `on_tool_error_callback` | `tool_error` | tool_name, args, error_type, error_message |
| `on_event_callback` | `event` | event_id, author, content, actions (state_delta, artifact_delta, transfer_to_agent), usage_metadata |
| `after_agent_callback` | `agent_end` | agent name |
| `after_run_callback` | `session_state_snapshot` + `invocation_end` | full state dict, event count |

### Basic usage

```python
from google.adk.plugins.debug_logging_plugin import DebugLoggingPlugin
from google.adk.runners import Runner

runner = Runner(
    agent=agent,
    app_name="dev_app",
    session_service=session_service,
    plugins=[DebugLoggingPlugin(output_path="/tmp/adk_debug.yaml")],
)
```

### Reading the output

Each invocation produces a top-level YAML document:

```yaml
---
invocation_id: inv-abc123
session_id: sess-xyz
app_name: dev_app
user_id: user-42
start_time: "2026-06-04T10:00:00.123456"
entries:
  - timestamp: "2026-06-04T10:00:00.124"
    entry_type: invocation_start
    invocation_id: inv-abc123
    agent_name: root_agent
  - timestamp: "2026-06-04T10:00:00.130"
    entry_type: llm_request
    agent_name: root_agent
    data:
      model: gemini-2.0-flash
      content_count: 3
      tools: [search_web, lookup_db]
      config:
        temperature: 0.7
  # ... more entries ...
  - timestamp: "2026-06-04T10:00:01.200"
    entry_type: invocation_end
```

### Combining with other plugins

`DebugLoggingPlugin` always returns `None` — it never short-circuits. Place it first in the plugin list to capture the unmodified request before any filtering or rewriting plugins run, or last to capture the final modified state:

```python
plugins=[
    ContextFilterPlugin(num_invocations_to_keep=5),
    DebugLoggingPlugin(output_path="/tmp/trace.yaml"),  # sees filtered contents
]
```

---

## 8 · `RunConfig` + `StreamingMode` + `ToolThreadPoolConfig`

**Source:** `google.adk.agents.run_config`

`RunConfig` is the Pydantic model passed as `run_config=` to `runner.run_async()` or `runner.run_live()`. It controls streaming, audio, context compression, thread pools, and safety limits.

### `StreamingMode` enum

```python
class StreamingMode(Enum):
    NONE = None   # single aggregated response per turn (default)
    SSE  = "sse"  # server-sent events: partial chunks + final aggregated
    BIDI = "bidi" # reserved; run_live() uses a separate path
```

### `RunConfig` fields (source-verified)

```python
class RunConfig(BaseModel):
    streaming_mode: StreamingMode = StreamingMode.NONE
    max_llm_calls: int = 500            # ≤ 0 means unbounded (warns)
    support_cfc: bool = False           # experimental CFC via LIVE API
    context_window_compression: Optional[types.ContextWindowCompressionConfig] = None
    get_session_config: Optional[GetSessionConfig] = None
    tool_thread_pool_config: Optional[ToolThreadPoolConfig] = None
    custom_metadata: Optional[dict[str, Any]] = None
    # Live-mode fields:
    speech_config: Optional[types.SpeechConfig] = None
    response_modalities: Optional[list[str]] = None
    output_audio_transcription: Optional[types.AudioTranscriptionConfig] = ...
    input_audio_transcription: Optional[types.AudioTranscriptionConfig] = ...
    realtime_input_config: Optional[types.RealtimeInputConfig] = None
    session_resumption: Optional[types.SessionResumptionConfig] = None
    save_live_blob: bool = False
    enable_affective_dialog: Optional[bool] = None
    proactivity: Optional[types.ProactivityConfig] = None
```

### SSE streaming — event filtering patterns

```python
from google.adk.agents.run_config import RunConfig, StreamingMode

run_config = RunConfig(streaming_mode=StreamingMode.SSE)

# Pattern A: typewriter text + final tool calls
displayed = ""
async for event in runner.run_async(user_id="u1", session_id="s1",
                                    new_message=msg, run_config=run_config):
    if event.partial and event.content and event.content.parts:
        has_text = any(p.text for p in event.content.parts)
        has_fc   = any(p.function_call for p in event.content.parts)
        if has_text and not has_fc:
            chunk = "".join(p.text or "" for p in event.content.parts)
            print(chunk, end="", flush=True)
            displayed += chunk
    elif not event.partial and event.get_function_calls():
        for fc in event.get_function_calls():
            print(f"\n[tool] {fc.name}({fc.args})")

# Pattern B: final events only (no streaming effect)
async for event in runner.run_async(..., run_config=run_config):
    if not event.partial and event.is_final_response():
        print("".join(p.text or "" for p in (event.content.parts or [])))
```

### Capping LLM calls to prevent runaway loops

```python
run_config = RunConfig(max_llm_calls=20)
# After 20 model calls the runner stops regardless of agent state
```

### `ToolThreadPoolConfig` — non-blocking tool execution in live mode

```python
from google.adk.agents.run_config import RunConfig, ToolThreadPoolConfig

run_config = RunConfig(
    tool_thread_pool_config=ToolThreadPoolConfig(max_workers=8),
)
```

Tools run in a `ThreadPoolExecutor` so blocking I/O (network calls, `time.sleep`, database queries) doesn't stall the event loop. Both sync and async tools are supported — async tools get their own event loop inside the worker thread.

> **GIL note:** Thread pools help with blocking I/O and C-extension work. For CPU-bound pure Python, they provide no parallelism benefit.

### Loading only recent session events

```python
from google.adk.agents.run_config import RunConfig
from google.adk.sessions.base_session_service import GetSessionConfig

run_config = RunConfig(
    get_session_config=GetSessionConfig(num_recent_events=50),
)
# Runner passes this to session_service.get_session(); only the last 50 events load
```

### Context window compression

```python
from google.genai import types

run_config = RunConfig(
    context_window_compression=types.ContextWindowCompressionConfig(
        sliding_window=types.SlidingWindow(target_tokens=16_000),
    )
)
```

### Custom metadata for observability

```python
run_config = RunConfig(
    custom_metadata={
        "request_id": "req-abc123",
        "feature_flag": "v2_agent",
        "ab_group": "treatment",
    }
)
```

`custom_metadata` is attached to `InvocationContext` and accessible in plugins via `invocation_context.run_config.custom_metadata`.

---

## 9 · `AuthenticatedFunctionTool`

**Source:** `google.adk.tools.authenticated_function_tool`

`AuthenticatedFunctionTool` is a subclass of `FunctionTool` that runs a `CredentialManager` before invoking the user function. If credentials are not yet available, it pauses and asks the client to complete an auth flow (e.g. OAuth2 redirect). Once credentials are available, it injects them into the function via a `credential` parameter.

### Constructor (source-verified)

```python
@experimental(FeatureName.AUTHENTICATED_FUNCTION_TOOL)
class AuthenticatedFunctionTool(FunctionTool):
    def __init__(
        self,
        *,
        func: Callable[..., Any],
        auth_config: AuthConfig = None,
        response_for_auth_required: Optional[Union[dict[str, Any], str]] = None,
    ): ...
```

| Parameter | Purpose |
|---|---|
| `func` | The tool function; may include a `credential` parameter |
| `auth_config` | `AuthConfig(auth_scheme=..., raw_auth_credential=...)` |
| `response_for_auth_required` | Returned to the LLM when credentials are missing/insufficient |

### OAuth2 tool with credential injection

```python
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool
from google.adk.auth.auth_credential import AuthCredential, OAuth2Auth


async def fetch_user_emails(
    max_results: int,
    credential: AuthCredential,  # injected by AuthenticatedFunctionTool
) -> dict:
    """Fetch emails from Gmail using the authenticated user's credentials."""
    token = credential.oauth2.access_token
    headers = {"Authorization": f"Bearer {token}"}
    response = await http_client.get(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages",
        headers=headers,
        params={"maxResults": max_results},
    )
    return response.json()


gmail_tool = AuthenticatedFunctionTool(
    func=fetch_user_emails,
    auth_config=AuthConfig(
        auth_scheme=OpenIdConnectWithConfig(
            # OpenIdConnectWithConfig requires explicit endpoint URLs (no discovery)
            authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
            token_endpoint="https://oauth2.googleapis.com/token",
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        ),
        raw_auth_credential=AuthCredential(
            auth_type="oauth2",
            oauth2=OAuth2Auth(
                client_id="123-xxx.apps.googleusercontent.com",
                client_secret="GOCSPX-...",
            ),
        ),
    ),
    response_for_auth_required="Please complete the Google login flow to access Gmail.",
)
```

### Auth flow mechanics

`run_async` follows this sequence:

```
1. credential_manager.get_auth_credential(tool_context)
   ├─ credentials cached → inject into func → return result
   └─ no credentials → credential_manager.request_credential(tool_context)
                        └─ returns response_for_auth_required to LLM
                           (LLM tells user to complete OAuth)
```

After the user completes the OAuth flow, the next invocation finds the cached token and calls the function normally.

### Function without `credential` parameter

If `func` does not declare a `credential` parameter, the credential is still resolved but simply not passed. This is useful for tools that read credentials from a shared context or environment:

```python
async def list_calendar_events(calendar_id: str) -> dict:
    # credentials loaded from environment / shared context
    ...

tool = AuthenticatedFunctionTool(func=list_calendar_events, auth_config=...)
```

---

## 10 · `FeatureName` + `override_feature_enabled` + `temporary_feature_override`

**Source:** `google.adk.features._feature_registry`

ADK uses a three-stage feature flag system to gate experimental, in-progress, and stable capabilities. Every non-trivial optional feature has a `FeatureName` entry and can be controlled at three levels: registry defaults → environment variables → programmatic overrides.

### `FeatureStage` enum

```python
class FeatureStage(Enum):
    WIP          = "wip"          # internal dev only, disabled by default
    EXPERIMENTAL = "experimental" # works, API may change, enabled by default in 2.1.0
    STABLE       = "stable"       # production-ready, no breaking changes without MAJOR bump
```

### Selected `FeatureName` entries (2.1.0)

| FeatureName | Stage | Default on |
|---|---|---|
| `AUTHENTICATED_FUNCTION_TOOL` | EXPERIMENTAL | True |
| `COMPUTER_USE` | EXPERIMENTAL | True |
| `DATA_AGENT_TOOLSET` | EXPERIMENTAL | True |
| `JSON_SCHEMA_FOR_FUNC_DECL` | EXPERIMENTAL | True |
| `PROGRESSIVE_SSE_STREAMING` | EXPERIMENTAL | True |
| `PUBSUB_TOOLSET` | EXPERIMENTAL | True |
| `SKILL_TOOLSET` | EXPERIMENTAL | True |
| `SPANNER_TOOLSET` | EXPERIMENTAL | True |
| `TOOL_CONFIRMATION` | EXPERIMENTAL | True |
| `SNAKE_CASE_SKILL_NAME` | EXPERIMENTAL | **False** |
| `IN_MEMORY_SESSION_SERVICE_LIGHT_COPY` | WIP | **False** |
| `BIG_QUERY_TOOLSET` | STABLE | True |

### Priority order (highest to lowest)

1. **Programmatic override** via `override_feature_enabled()`
2. **Environment variable** `ADK_ENABLE_<FEATURE_NAME>` or `ADK_DISABLE_<FEATURE_NAME>`
3. **Registry default** (`FeatureConfig.default_on`)

### `override_feature_enabled` — programmatic control

```python
from google.adk.features._feature_registry import (
    FeatureName,
    override_feature_enabled,
    is_feature_enabled,
)

# Disable progressive SSE streaming (e.g. to test legacy path)
override_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING, False)

# Enable snake_case skill names (opt-in experimental)
override_feature_enabled(FeatureName.SNAKE_CASE_SKILL_NAME, True)

print(is_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING))  # False
```

### `temporary_feature_override` — scoped context manager

Designed for testing: restores the original state when the context exits.

```python
from google.adk.features._feature_registry import (
    FeatureName,
    temporary_feature_override,
)
import pytest

@pytest.mark.asyncio
async def test_with_snake_case_skills():
    with temporary_feature_override(FeatureName.SNAKE_CASE_SKILL_NAME, True):
        skill_toolset = SkillToolset(skills=[my_skill])
        # snake_case naming is active here
        assert "my_skill_tool" in [t.name for t in await skill_toolset.get_tools()]
    # outside the block: original state restored
```

### Environment variable control

```bash
# Disable JSON schema for function declarations (revert to legacy schema)
export ADK_DISABLE_JSON_SCHEMA_FOR_FUNC_DECL=1

# Enable in-memory session light copy (WIP feature)
export ADK_ENABLE_IN_MEMORY_SESSION_SERVICE_LIGHT_COPY=1
```

### The `@experimental` decorator

Classes and functions decorated with `@experimental(FeatureName.X)` check `is_feature_enabled(X)` at instantiation/call time and raise `RuntimeError` if it is disabled:

```python
from google.adk.features import experimental, FeatureName

@experimental(FeatureName.AUTHENTICATED_FUNCTION_TOOL)
class AuthenticatedFunctionTool(FunctionTool):
    def __init__(self, ...):
        # check happens in the generated __init__ wrapper
        ...
```

To use an experimental class in an environment where it might be disabled:

```python
from google.adk.features._feature_registry import override_feature_enabled, FeatureName

override_feature_enabled(FeatureName.AUTHENTICATED_FUNCTION_TOOL, True)

# Now safe to instantiate
tool = AuthenticatedFunctionTool(func=my_func, auth_config=...)
```

### `is_feature_enabled` in your own code

Use `is_feature_enabled` to write conditional paths that respect the ADK feature gate:

```python
from google.adk.features._feature_registry import is_feature_enabled, FeatureName

def build_toolset(config):
    tools = [base_tool_a, base_tool_b]
    if is_feature_enabled(FeatureName.COMPUTER_USE):
        from google.adk.tools.computer_use import ComputerUseToolset
        tools.append(ComputerUseToolset())
    return tools
```

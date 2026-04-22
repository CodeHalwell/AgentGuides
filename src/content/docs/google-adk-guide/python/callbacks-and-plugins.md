---
title: "Callbacks and Plugins"
description: "Intercept agent, model, and tool execution — per-agent callbacks and runner-wide plugins."
framework: google-adk
language: python
sidebar:
  order: 40
---

Verified against google-adk==2.0.0b1 (`google/adk/agents/llm_agent.py`, `google/adk/plugins/`).

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

Plugins can implement any subset of the 11 hooks below (`plugins/base_plugin.py`):

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

### `LoggingPlugin`
```python
from google.adk.plugins import LoggingPlugin
app = App(name="demo", root_agent=agent, plugins=[LoggingPlugin()])
```
Emits structured logs for every model/tool/agent event via the `google_adk` logger. Drop-in when you want to see what ADK is doing.

### `DebugLoggingPlugin`
Per-invocation verbose dump — full prompts, responses, tool I/O. Use in dev, never in prod (it writes large payloads to logs).

### `ReflectAndRetryToolPlugin` (experimental)
```python
from google.adk.plugins import ReflectAndRetryToolPlugin
from google.adk.plugins.reflect_retry_tool_plugin import TrackingScope

plugin = ReflectAndRetryToolPlugin(
    max_retries=3,
    throw_exception_if_retry_exceeded=True,
    tracking_scope=TrackingScope.INVOCATION,   # or GLOBAL
)
```
Catches tool failures, asks the model to reflect on the error, and retries up to `max_retries` times. Override `extract_error_from_result` to treat `{"status": "error"}` shapes as failures even when the tool didn't raise.

### `GlobalInstructionPlugin`
```python
from google.adk.plugins.global_instruction_plugin import GlobalInstructionPlugin
plugin = GlobalInstructionPlugin(instruction="You are a safety-first assistant.")
```
Prepends a system instruction to **every** agent in the app. Replaces the deprecated `LlmAgent.global_instruction` field.

### `SaveFilesAsArtifactsPlugin`
Intercepts inline-data parts in user messages and persists them to the artifact service, replacing the blob with a reference. Replaces the deprecated `RunConfig.save_input_blobs_as_artifacts`.

### Other built-ins
- `BigQueryAgentAnalyticsPlugin` — exports agent analytics to BigQuery (experimental).
- `ContextFilterPlugin` — trims session events sent to the model based on size/token budgets.
- `MultimodalToolResultsPlugin` — handles tool responses that contain multimodal parts.

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

- A callback that returns **any** truthy value short-circuits the chain — `return None` is required to let the chain continue.
- Plugins execute before agent callbacks at the same hook. A plugin returning non-`None` prevents the agent's own callback from running.
- `LlmAgent.global_instruction` is deprecated — migrate to `GlobalInstructionPlugin`.
- `RunConfig.save_input_blobs_as_artifacts` is deprecated — migrate to `SaveFilesAsArtifactsPlugin`.
- Stateful plugins should guard mutation with `asyncio.Lock` — concurrent invocations share the plugin instance.
- `callback_context.state[...] = value` is persisted to session state on the next `append_event`. Use `temp:` prefix for scratch values that must not persist.

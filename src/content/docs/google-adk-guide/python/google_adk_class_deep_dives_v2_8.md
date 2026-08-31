---
title: "Class Deep Dives — v2.8.0"
description: "Source-verified deep dives for 10 classes new or enhanced in google-adk 2.8.0: LiveRequestQueue, ExecuteBashTool, ToolConfirmation, App, EventsCompactionConfig, LlmEventSummarizer, PreloadMemoryTool, ToolboxToolset, GcsArtifactService, and the exit_loop / get_user_choice_tool built-ins."
framework: google-adk
language: python
sidebar:
  order: 120
---

All examples and field tables on this page are source-verified against **google-adk==2.8.0** (installed, introspected with `inspect.getsource`). The ten classes were sampled to cover areas that either arrived new in 2.8.0 or remained thin in earlier documentation.

| # | Class / Symbol | Module | Subject |
|---|---|---|---|
| 1 | `LiveRequestQueue` | `google.adk.agents.live_request_queue` | Bidirectional / live streaming |
| 2 | `ExecuteBashTool` + `BashToolPolicy` | `google.adk.tools.bash_tool` | Bash execution with HITL gate |
| 3 | `ToolConfirmation` | `google.adk.tools.tool_confirmation` | Human-in-the-loop gate (any tool) |
| 4 | `App` | `google.adk.apps.app` | Top-level application container |
| 5 | `EventsCompactionConfig` | `google.adk.apps.app` | Token-threshold + sliding-window compaction |
| 6 | `LlmEventSummarizer` | `google.adk.apps.llm_event_summarizer` | LLM-driven session summarizer |
| 7 | `PreloadMemoryTool` | `google.adk.tools.preload_memory_tool` | Proactive memory injection |
| 8 | `ToolboxToolset` | `google.adk.tools.toolbox_toolset` | Google Cloud Toolbox / MCP Toolbox |
| 9 | `GcsArtifactService` | `google.adk.artifacts.gcs_artifact_service` | Production GCS artifact storage |
| 10 | `exit_loop` + `get_user_choice_tool` | `google.adk.tools` | Control flow built-ins |

---

## 1 — `LiveRequestQueue`

**Module:** `google.adk.agents.live_request_queue`

`LiveRequestQueue` is the send-side of a bidirectional streaming session. You create one, pass it to `runner.run_live()`, and push `LiveRequest` packets into it from your application. The agent drains the queue asynchronously on the receive side.

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.runners import InMemoryRunner
from google.genai import types

agent = LlmAgent(
    name="voice_assistant",
    model="gemini-2.5-flash",
    instruction="You are a helpful voice assistant. Respond briefly.",
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="live_demo")
    session = await runner.session_service.create_session(
        app_name="live_demo", user_id="u1"
    )

    queue = LiveRequestQueue()

    async def send_messages():
        await asyncio.sleep(0.1)
        # Send a text message
        queue.send_content(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="What time is it?")]
            )
        )
        await asyncio.sleep(0.5)
        # Close the stream when done
        queue.close()

    async def receive_responses():
        async for event in runner.run_live(
            user_id="u1",
            session_id=session.id,
            live_request_queue=queue,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"Agent: {part.text}")

    await asyncio.gather(send_messages(), receive_responses())

asyncio.run(main())
```

### Method reference

Source-verified from `google/adk/agents/live_request_queue.py`:

| Method | Signature | What it does |
|---|---|---|
| `close()` | `() -> None` | Enqueues a terminal `LiveRequest(close=True)` — agent drains, then stops |
| `send_content(content, partial=False)` | `(Content, bool) -> None` | Send a text/multimodal turn; `partial=True` streams an incomplete message |
| `send_realtime(blob)` | `(Blob) -> None` | Send a raw audio/image blob for real-time input (voice mode) |
| `send_activity_start()` | `() -> None` | Mark beginning of a discrete user activity (voice VAD signal) |
| `send_activity_end()` | `() -> None` | Mark end of a discrete user activity |
| `send_audio_stream_end()` | `() -> None` | Force-flush audio — use when an audio chunk stream ends mid-utterance |
| `send(req)` | `(LiveRequest) -> None` | Send an arbitrary pre-built `LiveRequest` |
| `get()` | `() -> LiveRequest` (async coroutine) | Internal: drain one item asynchronously (used by the runner — don't call directly) |

### Audio streaming pattern

```python
import asyncio
import wave
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.genai import types

async def stream_audio(queue: LiveRequestQueue, wav_path: str) -> None:
    """Stream a WAV file in 20 ms chunks — mirrors real-time microphone input."""
    CHUNK_DURATION_MS = 20
    with wave.open(wav_path, "rb") as wf:
        # Live audio endpoint requires mono, 16-bit, linear PCM.
        assert wf.getnchannels() == 1, "WAV must be mono"
        assert wf.getsampwidth() == 2, "WAV must be 16-bit"
        assert wf.getcomptype() == "NONE", "WAV must be uncompressed PCM"
        sample_rate = wf.getframerate()
        frames_per_chunk = int(sample_rate * CHUNK_DURATION_MS / 1000)

        queue.send_activity_start()
        while True:
            chunk = wf.readframes(frames_per_chunk)
            if not chunk:
                break
            queue.send_realtime(
                types.Blob(data=chunk, mime_type=f"audio/pcm;rate={sample_rate}")
            )
            await asyncio.sleep(CHUNK_DURATION_MS / 1000)
        queue.send_activity_end()
    queue.close()
```

### Gotchas

- `LiveRequestQueue` uses `asyncio.Queue` internally. All `send_*` methods call `put_nowait` — they are safe to call from sync code but must be called from the **same event loop** thread as the running runner.
- `send_audio_stream_end()` forces a flush; call it before `close()` if you're in voice mode to avoid dropped trailing audio.
- `partial=True` on `send_content` lets you stream a message incrementally (like a user typing). Only the last `partial=False` chunk triggers a model response.
- `close()` is idempotent in terms of the agent's behaviour — it only enqueues one sentinel — but calling it multiple times enqueues multiple sentinels. Call it once.

---

## 2 — `ExecuteBashTool` + `BashToolPolicy`

**Module:** `google.adk.tools.bash_tool`

`ExecuteBashTool` gives an agent the ability to run shell commands inside a workspace directory. Every invocation requires user confirmation (via ADK's `ToolConfirmation` protocol — see §3) before the command executes. `BashToolPolicy` controls which commands are permitted and what resource limits are applied to the subprocess.

### `BashToolPolicy` — field reference

Source-verified from `google/adk/tools/bash_tool.py`:

| Field | Type | Default | Notes |
|---|---|---|---|
| `allowed_command_prefixes` | `tuple[str, ...]` | `("*",)` | `"*"` allows everything; list explicit prefixes like `("git", "ls", "cat")` to allowlist |
| `blocked_operators` | `tuple[str, ...]` | `()` | Shell operators to block in the command string, e.g. `(";", "&&", "&#124;&#124;")` |
| `timeout_seconds` | `int \| None` | `30` | Seconds before the subprocess is SIGKILL'd; `None` = no limit |
| `max_memory_bytes` | `int \| None` | `None` | `RLIMIT_AS` on the spawned process; `None` = unlimited |
| `max_file_size_bytes` | `int \| None` | `None` | `RLIMIT_FSIZE` on the spawned process |
| `max_child_processes` | `int \| None` | `None` | `RLIMIT_NPROC`; set to prevent fork bombs |

`BashToolPolicy` is a frozen dataclass — create a new one to change settings.

### `ExecuteBashTool` — constructor

```python
ExecuteBashTool(
    *,
    workspace: pathlib.Path | None = None,   # defaults to cwd
    policy: BashToolPolicy | None = None,    # defaults to permissive policy
)
```

The tool registers itself as `name="execute_bash"` and writes a dynamic description that names the allowed command prefixes so the model knows upfront what it can ask for.

### Basic usage

```python
import asyncio
import pathlib
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.bash_tool import ExecuteBashTool, BashToolPolicy
from google.genai import types

# Example policy — reduces accidental misuse, but is NOT a security sandbox.
#
# Limitations:
#   - allowed_command_prefixes is a startswith check: `echo $(rm -rf ...)` passes
#     because the string starts with "echo". Adding "$(" and "`" to
#     blocked_operators closes the most obvious bypasses but cannot cover every
#     vector — e.g. `git -c alias.x='!rm -rf /' x` starts with "git" and
#     contains none of the blocked tokens, yet executes a shell command via
#     git's alias feature.
#   - For security-sensitive workloads, run the agent inside a real OS-level
#     sandbox (Docker, gVisor, Firecracker, etc.) rather than relying on this
#     filter as your primary containment boundary.
policy = BashToolPolicy(
    allowed_command_prefixes=("git", "ls", "cat", "echo"),
    blocked_operators=(
        ";", "&&", "||", "|", ">", ">>",
        "$(", "`",          # command substitution
        "\n", "\r",         # newline injection
    ),
    timeout_seconds=10,
    max_memory_bytes=128 * 1024 * 1024,   # 128 MB
    max_child_processes=10,
)

bash_tool = ExecuteBashTool(
    workspace=pathlib.Path("/workspace/my_repo"),
    policy=policy,
)

agent = LlmAgent(
    name="repo_analyst",
    model="gemini-2.5-flash",
    instruction=(
        "You are a repository analyst. "
        "Use execute_bash to inspect the repo — git log, ls, cat files. "
        "Always explain what you are about to run before requesting confirmation."
    ),
    tools=[bash_tool],
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="bash_demo")
    session = await runner.session_service.create_session(
        app_name="bash_demo", user_id="u1"
    )

    # Step 1 — initial invocation: model calls execute_bash, tool suspends
    # and emits an adk_request_confirmation function call asking for approval.
    confirmation_fc_id = None
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part.from_text(text="Show me the last 5 commits.")],
        ),
    ):
        if event.content:
            for part in event.content.parts:
                if part.text:
                    print(part.text)
                # Capture the confirmation function-call ID so we can approve it.
                if (part.function_call
                        and part.function_call.name == "adk_request_confirmation"):
                    confirmation_fc_id = part.function_call.id

    # Step 2 — send approval: respond to the pending adk_request_confirmation
    # with confirmed=True so the tool actually executes the bash command.
    if confirmation_fc_id:
        async for event in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=confirmation_fc_id,
                            name="adk_request_confirmation",
                            response={"confirmed": True},
                        )
                    )
                ],
            ),
        ):
            if event.content:
                for part in event.content.parts:
                    if part.text:
                        print(part.text)

asyncio.run(main())
```

### How confirmation works

1. On first invocation the tool calls `tool_context.request_confirmation(hint="Please approve or reject the bash command: <cmd>")` and returns an error string asking for confirmation.
2. The model pauses. The client presents the confirmation to the user.
3. On the next turn the client sends back a `ToolConfirmation(confirmed=True/False)` (see §3). If `confirmed=False`, the tool returns `{"error": "This tool call is rejected."}`.
4. Only after `confirmed=True` does the subprocess actually run.

### Response shape

```python
# Success
{"stdout": "...", "stderr": "...", "returncode": 0}

# Timeout
{"error": "Command timed out after 10 seconds.", "stdout": "...", "stderr": "...", "returncode": -9}

# Rejected by user
{"error": "This tool call is rejected."}

# Blocked by policy
{"error": "Command prefix 'rm' is not allowed."}
```

### Gotchas

- `ExecuteBashTool` is POSIX-only — it returns `{"error": "ExecuteBashTool is only supported on POSIX systems."}` on Windows.
- The subprocess runs in a new session (`start_new_session=True`) and is SIGKILL'd on timeout via `os.killpg`. The `finally` block always kills the process group, even on success, to avoid zombie children.
- `allowed_command_prefixes` is a `startswith` check on the raw command string — it is **not** a security boundary on its own. A command such as `echo $(rm -rf /workspace)` passes the prefix check because the string starts with `"echo"`, while the nested `$(...)` still executes in the shell. Adding `"$("` and backtick to `blocked_operators` closes that vector, but there are further bypasses for tools that interpret their own flags: for example, `git -c alias.x='!rm -rf /' x` starts with the allowed prefix `"git"` and contains no blocked token, yet git's alias feature executes the shell command. For security-sensitive workloads, run inside a real OS-level sandbox rather than relying on `BashToolPolicy` as your primary containment boundary.
- `blocked_operators` performs a string-contains check on the raw command. Shell injection via encoded operators (e.g. `%26%26`) can bypass it — pair with a strict `allowed_command_prefixes` allowlist for security-sensitive environments.
- `max_memory_bytes`, `max_file_size_bytes`, and `max_child_processes` are applied via `resource.setrlimit` in `preexec_fn` — they only take effect on Linux/macOS that support POSIX resource limits.

---

## 3 — `ToolConfirmation`

**Module:** `google.adk.tools.tool_confirmation`

`ToolConfirmation` is the structured response the agent framework exchanges with a client to implement human-in-the-loop approval for any tool. It is `@experimental`.

### Field reference

Source-verified from `google/adk/tools/tool_confirmation.py`:

| Field | Type | Default | Notes |
|---|---|---|---|
| `hint` | `str` | `""` | The prompt text shown to the user when requesting confirmation |
| `confirmed` | `bool` | `False` | `True` = user approved; `False` = user rejected |
| `payload` | `Any \| None` | `None` | Optional JSON-serializable extra data from the user (e.g. edited arguments) |

The model config sets `alias_generator=to_camel` and `populate_by_name=True` — both snake_case and camelCase field names are accepted in payloads.

### Requesting confirmation from inside a tool

```python
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.tool_confirmation import ToolConfirmation

async def delete_database_record(record_id: str, tool_context: ToolContext) -> dict:
    """Delete a record from the database. Requires confirmation."""
    # First call: no confirmation yet — request it
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"About to delete record '{record_id}'. This cannot be undone. Approve?"
        )
        tool_context.actions.skip_summarization = True
        return {"status": "awaiting_confirmation"}

    # Second call: confirmation arrived
    if not tool_context.tool_confirmation.confirmed:
        return {"status": "rejected", "record_id": record_id}

    # User approved — proceed
    # ... actual deletion logic ...
    return {"status": "deleted", "record_id": record_id}
```

### Parsing confirmation from a client response

```python
from google.adk.tools.tool_confirmation import ToolConfirmation

# Direct dict (ADK web UI format)
conf = ToolConfirmation.from_response_dict({"confirmed": True, "hint": "Approved"})
print(conf.confirmed)  # True

# ADK client wrapper format: {"response": "<json string>"}
wrapped = {"response": '{"confirmed": false, "hint": ""}'}
conf = ToolConfirmation.from_response_dict(wrapped)
print(conf.confirmed)  # False
```

`from_response_dict` handles both the direct dict format and the ADK client's `{"response": json_string}` wrapper — use it whenever you deserialize confirmation responses from the wire.

### Pattern: confirmation with amended arguments

```python
async def send_email(to: str, subject: str, body: str, tool_context: ToolContext) -> dict:
    """Send an email — user can amend subject/body before confirming."""
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"Send email to {to}? Subject: {subject}",
        )
        tool_context.actions.skip_summarization = True
        return {"status": "awaiting_confirmation"}

    conf = tool_context.tool_confirmation
    if not conf.confirmed:
        return {"status": "rejected"}

    # Payload may carry user-amended fields
    if conf.payload and isinstance(conf.payload, dict):
        subject = conf.payload.get("subject", subject)
        body = conf.payload.get("body", body)

    # ... send logic ...
    return {"status": "sent", "to": to, "subject": subject}
```

### Gotchas

- `ToolConfirmation` is `@experimental` — the `feature_name` is `FeatureName.TOOL_CONFIRMATION`. The API may change in future releases.
- `tool_context.request_confirmation()` does **not** set `skip_summarization` automatically — set it explicitly in your tool body (as shown in the examples above) if you don't want the "awaiting_confirmation" response fed back to the model as a completion.
- If a tool calls `request_confirmation` but the client never sends back a `ToolConfirmation`, the tool will keep returning "awaiting_confirmation" on every subsequent model call until the session ends. Guard with a max-retries counter in the tool body if needed.

---

## 4 — `App`

**Module:** `google.adk.apps.app`

`App` is the top-level container for an agentic application. It wraps a `root_agent` (or `root_node`) together with app-wide `plugins`, `events_compaction_config`, `context_cache_config`, and `resumability_config`. Prefer `App` over passing `agent=` directly to `Runner` — it keeps application-level concerns separate from runtime concerns.

### Constructor

```python
App(
    name: str,                                          # required; must be a valid app_name
    root_agent: BaseAgent | BaseNode,                   # required
    plugins: list[BasePlugin] = [],                     # app-wide plugins
    events_compaction_config: EventsCompactionConfig | None = None,
    context_cache_config: ContextCacheConfig | None = None,
    resumability_config: ResumabilityConfig | None = None,
)
```

Validation rules (from `_validate`):
- `root_agent` must be a `BaseAgent` or `BaseNode` instance; `None` raises `ValueError`.
- `name` is passed through `validate_app_name` — must match `^[a-zA-Z][a-zA-Z0-9_-]*$` (start with a letter; letters, digits, underscores, and hyphens only). The reserved name `"user"` is also rejected with `ValueError`.

### Minimal example

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="Be helpful.",
)

app = App(name="my_app", root_agent=agent)

runner = Runner(
    app=app,
    session_service=InMemorySessionService(),
)
```

### App with plugins and compaction

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService

agent = LlmAgent(
    name="long_chat_agent",
    model="gemini-2.5-flash",
    instruction="Assist with multi-turn conversations.",
)

compaction_cfg = EventsCompactionConfig(
    token_threshold=40_000,       # compact when prompt exceeds 40k tokens
    event_retention_size=30,      # keep last 30 raw events uncompacted
)

app = App(
    name="long_chat",
    root_agent=agent,
    plugins=[LoggingPlugin()],
    events_compaction_config=compaction_cfg,
)

runner = Runner(
    app=app,
    session_service=InMemorySessionService(),
    memory_service=InMemoryMemoryService(),
)
```

### Gotchas

- `Runner(plugins=...)` is deprecated — move plugins to `App(plugins=...)`.
- `global_instruction` on `LlmAgent` is deprecated — use `GlobalInstructionPlugin` in `App(plugins=[GlobalInstructionPlugin(...)])`.
- `App.root_agent` accepts a `BaseNode` (a `Workflow` node) as well as a `BaseAgent`; the type hint says `Union[BaseAgent, Any, None]` due to a circular-import workaround.
- The validator checks `root_agent is not None` — passing `root_agent=None` raises `ValueError("root_agent must be provided.")`.

---

## 5 — `EventsCompactionConfig`

**Module:** `google.adk.apps.app`

`EventsCompactionConfig` configures automatic event compaction to prevent unbounded session growth in long conversations. It is `@experimental`.

Two compaction modes are available and can be combined:

| Mode | Fields | Trigger |
|---|---|---|
| **Token-threshold** | `token_threshold` + `event_retention_size` | Fires after any invocation where the prompt token count ≥ `token_threshold` |
| **Sliding-window** | `compaction_interval` + `overlap_size` | Fires after every N new user invocations since the last compaction |

### Field reference

| Field | Type | Constraint | Notes |
|---|---|---|---|
| `summarizer` | `BaseEventsSummarizer \| None` | — | Defaults to `LlmEventSummarizer` built from the root agent's model if `None` |
| `token_threshold` | `int \| None` | `> 0` | Must be set alongside `event_retention_size` |
| `event_retention_size` | `int \| None` | `>= 0` | Number of raw events to keep after compaction; `0` = compact everything |
| `compaction_interval` | `int \| None` | `> 0` | Number of new invocations between sliding-window compactions |
| `overlap_size` | `int \| None` | `>= 0` | Invocations from the previous window included in each new summary |

A `model_validator` enforces three rules:

- **Token pair:** `token_threshold` and `event_retention_size` must be set together — setting only one raises `ValueError`.
- **Sliding-window pair:** `compaction_interval` and `overlap_size` must be set together — setting only one raises `ValueError`.
- **At least one mode:** at least one complete trigger pair must be configured — an instance with no fields set (other than `summarizer`) raises `ValueError`.

### Token-threshold compaction

```python
from google.adk.apps.app import EventsCompactionConfig

# Compact when prompt hits 32k tokens; keep the last 20 raw events
cfg = EventsCompactionConfig(
    token_threshold=32_000,
    event_retention_size=20,
)
```

**What happens:** After each invocation the runner checks the last known prompt token count. If it meets `token_threshold`, all events except the most recent `event_retention_size` are summarised into a single compaction event by the `summarizer`. The summary is prepended to the retained events, so the model always has a condensed history plus recent raw turns.

### Sliding-window compaction

```python
from google.adk.apps.app import EventsCompactionConfig

# Compact every 5 user invocations; include 2 invocations of overlap
cfg = EventsCompactionConfig(
    compaction_interval=5,
    overlap_size=2,
)
```

**What happens:** After every 5th new user invocation, the runner generates a summary of all events from the previous compaction boundary (minus `overlap_size` invocations of overlap). The overlap keeps adjacent windows contextually connected.

### Combined mode

```python
from google.adk.apps.app import EventsCompactionConfig

# Try token-threshold first; fall back to sliding-window if token count unavailable
cfg = EventsCompactionConfig(
    token_threshold=50_000,
    event_retention_size=50,
    compaction_interval=10,
    overlap_size=2,
)
```

When both modes are configured, the runner tries token-threshold compaction first. If it fires, the sliding-window pass is skipped for that invocation. If token count is unavailable, the sliding-window mode handles compaction independently.

### Custom summarizer

```python
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models.lite_llm import LiteLlm

fast_llm = LiteLlm(model="gemini-2.0-flash")

cfg = EventsCompactionConfig(
    summarizer=LlmEventSummarizer(
        llm=fast_llm,
        prompt_template=(
            "Summarise the following agent conversation in bullet points, "
            "preserving all tool calls and their results. "
            "Conversation:\n\n{conversation_history}"
        ),
    ),
    token_threshold=40_000,
    event_retention_size=25,
)
```

### Gotchas

- Token count is read from `event.usage_metadata.prompt_token_count` (the last event with usage data). If no event has usage metadata (e.g. very short sessions), ADK estimates token count as `total_chars // 4`.
- Rewound invocations (via `runner.rewind_async`) are excluded from the compaction candidate set — only live events are summarised.
- `event_retention_size=0` compacts everything, including the most recent turn. Use with caution in interactive apps — if the LLM hallucinates the summary, all raw context is gone.
- The compaction event is persisted via `session_service.append_event`. If your session service is read-only or has write failures, compaction events will be silently lost.

---

## 6 — `LlmEventSummarizer`

**Module:** `google.adk.apps.llm_event_summarizer`

`LlmEventSummarizer` is the default compaction engine. It formats session events into a transcript, sends it to an LLM with a structured prompt, and wraps the response in an `Event` carrying an `EventCompaction` payload.

### Constructor

```python
LlmEventSummarizer(
    llm: BaseLlm,
    prompt_template: str | None = None,   # see default below
)
```

### Default prompt template

```
The following is a conversation history between a user and an AI agent.
It may or may not start from a compacted history. Please identify and
reiterate the user request, summarize the context so far, focusing on
key decisions made and information obtained, as well as any unresolved
questions or tasks.
CRITICAL INSTRUCTIONS:
1. Explicitly identify and state the primary language used by the user
   at the top of your summary (e.g., "Conversation Language: English").
2. If the agent called any tools, accurately list the exact tool names
   used to maintain tool grounding.
The rest of the summary should be concise and capture the essence of
the interaction.

{conversation_history}
```

The `{conversation_history}` placeholder is required. If your template omits it, `str.format()` silently discards the conversation data (unused keyword arguments are not an error in Python), so the summarizer sends the model a prompt with no event history and may persist an ungrounded summary. Always include the placeholder.

### What gets included in the transcript

Tool call args and responses are truncated at **2 000 characters** each to prevent the compaction prompt from exceeding the model's context window itself. Thoughts emitted by previous compaction events are excluded (they're the summariser's own internal state, not conversation content).

### Minimal usage

```python
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps import App
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

agent = LlmAgent(
    name="support_bot",
    model="gemini-2.5-flash",
    instruction="Provide customer support.",
)

# Use a cheaper model for summarisation
summarizer = LlmEventSummarizer(llm=Gemini(model="gemini-2.0-flash"))

app = App(
    name="support",
    root_agent=agent,
    events_compaction_config=EventsCompactionConfig(
        summarizer=summarizer,
        compaction_interval=8,
        overlap_size=1,
    ),
)
```

### Custom summarizer subclass

```python
import json
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events.event import Event

class JsonSummarizer(LlmEventSummarizer):
    """Produces a structured JSON summary instead of prose."""

    _JSON_TEMPLATE = (
        "Summarise this agent conversation as a JSON object with keys: "
        "'language', 'user_goal', 'steps_taken' (list), 'unresolved' (list), 'outcome'.\n\n"
        "Conversation:\n{conversation_history}\n\n"
        "Return valid JSON only, no markdown fencing."
    )

    def __init__(self, llm):
        super().__init__(llm=llm, prompt_template=self._JSON_TEMPLATE)

    async def maybe_summarize_events(self, *, events: list[Event]):
        event = await super().maybe_summarize_events(events=events)
        if event and event.actions.compaction:
            # Optionally validate the JSON
            compacted = event.actions.compaction.compacted_content
            if compacted and compacted.parts:
                try:
                    json.loads(compacted.parts[0].text or "{}")
                except json.JSONDecodeError:
                    pass  # summary arrived as prose — still usable
        return event
```

### Gotchas

- `LlmEventSummarizer` uses `llm.generate_content_async` with `stream=False`. If the LLM call fails or returns no content, `maybe_summarize_events` returns `None` (no compaction event is written).
- Tool content is truncated at 2 000 characters per call/response (`_MAX_TOOL_CONTENT_CHARS`). For tool-heavy agents that rely on full search results in their summaries, consider raising this limit in a subclass.
- The summary event's author is always `"user"` (not `"model"`), even though `compacted_content.role` is set to `"model"`. This is by design — the compaction event acts as a history anchor, not a model response.
- `LlmEventSummarizer` is instantiated automatically by `EventsCompactionConfig` if `summarizer=None` is left as the default and the root agent is an `LlmAgent`. The auto-selected model is `agent.canonical_model`.

---

## 7 — `PreloadMemoryTool`

**Module:** `google.adk.tools.preload_memory_tool`

`PreloadMemoryTool` runs automatically before every LLM call — the model never explicitly invokes it. It queries the memory service with the current user message, then injects any matching past memories as a transient system context. Unlike `LoadMemoryTool` (which the model calls on demand), `PreloadMemoryTool` always preloads.

### Usage

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

agent = LlmAgent(
    name="personalised_assistant",
    model="gemini-2.5-flash",
    instruction=(
        "You are a personal assistant. Use past context to personalise responses. "
        "Do not mention that you have a memory system unless asked."
    ),
    tools=[PreloadMemoryTool()],
)

app = App(name="personal_demo", root_agent=agent)

runner = Runner(
    app=app,
    session_service=InMemorySessionService(),
    memory_service=InMemoryMemoryService(),
)
```

### What gets injected

If `search_memory` returns any results, each memory entry is formatted as:

```
Time: <timestamp>          ← only if memory.timestamp is set
<author>: <text>           ← author prefix only if memory.author is set
```

All entries are joined with newlines, then wrapped in:

```
The following content is from your previous conversations with the user.
They may be useful for answering the user's current query.
<PAST_CONVERSATIONS>
...memories...
</PAST_CONVERSATIONS>
```

This block is injected as a transient `role="user"` `Content` that is **not persisted** to the session — it only appears in the outgoing LLM request for the current turn.

### Comparing `PreloadMemoryTool` vs `LoadMemoryTool`

| Behaviour | `PreloadMemoryTool` | `LoadMemoryTool` |
|---|---|---|
| Called by | Framework (automatic) | Model (on demand) |
| Injection point | `process_llm_request` — transient, not in session | `run_async` — appears as tool response in session |
| Model awareness | Model does not know memory was injected | Model explicitly calls the tool |
| Use case | Always-on personalisation | Model-controlled recall on complex queries |
| Missing memory service | Silent no-op (catches exception) | Raises `ValueError` at call time |

### Combining both tools

```python
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.adk.tools.load_memory_tool import load_memory
from google.adk.agents import LlmAgent

agent = LlmAgent(
    name="smart_recall",
    model="gemini-2.5-flash",
    instruction=(
        "You have automatic memory preloading for personalisation. "
        "For deeper research, you can also call load_memory explicitly."
    ),
    tools=[PreloadMemoryTool(), load_memory],
)
```

### Gotchas

- `PreloadMemoryTool` only extracts text parts from memory entries (`extract_text` helper). Embedded images or binary content in memories are ignored.
- If the first part of the user's message has no text (e.g. a voice blob), `PreloadMemoryTool` returns immediately without querying memory. Ensure text is always present in the first part for effective recall.
- `search_memory` exceptions are swallowed with a `logging.warning`. If memory search fails silently in production, check your memory service logs separately.
- `PreloadMemoryTool` is **not visible to the model** in the function declarations — it has no `_get_declaration`. Do not include it in instructions as a callable tool.

---

## 8 — `ToolboxToolset`

**Module:** `google.adk.tools.toolbox_toolset`

`ToolboxToolset` connects an ADK agent to a [Google Cloud MCP Toolbox](https://github.com/googleapis/mcp-toolbox-sdk-python) server, exposing its registered tools as regular ADK tools. Requires `pip install google-adk[toolbox]`.

### Constructor

```python
ToolboxToolset(
    server_url: str,                                         # required
    toolset_name: str | None = None,                         # load a named toolset subset
    tool_names: list[str] | None = None,                     # load specific tools by name
    auth_token_getters: Mapping[str, Callable[[], str]] | None = None,
    bound_params: Mapping[str, Callable[[], Any] | Any] | None = None,
    credentials: CredentialConfig | None = None,
    additional_headers: Mapping[str, str] | None = None,
    **kwargs,                                                # forwarded to toolbox_adk.ToolboxToolset
)
```

If both `toolset_name` and `tool_names` are omitted, all tools from the server are loaded.

### Basic example

```python
from google.adk.agents import LlmAgent
from google.adk.tools.toolbox_toolset import ToolboxToolset
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# Connect to a locally-running toolbox server
toolbox = ToolboxToolset("http://localhost:5000")

agent = LlmAgent(
    name="data_agent",
    model="gemini-2.5-flash",
    instruction="Use the available tools to answer data questions.",
    tools=[toolbox],
)

app = App(name="toolbox_demo", root_agent=agent)
runner = Runner(app=app, session_service=InMemorySessionService())
```

### Authenticated toolbox connection

```python
import google.auth
from google.auth.transport.requests import Request as AuthRequest
from google.adk.tools.toolbox_toolset import ToolboxToolset

def get_id_token() -> str:
    """Fetches a Google-signed ID token for Toolbox authentication."""
    import google.oauth2.id_token
    auth_req = AuthRequest()
    target_audience = "https://my-toolbox.run.app"
    return google.oauth2.id_token.fetch_id_token(auth_req, target_audience)

toolbox = ToolboxToolset(
    server_url="https://my-toolbox.run.app",
    toolset_name="bigquery_tools",           # only load this named subset
    auth_token_getters={"google": get_id_token},
    additional_headers={"X-Tenant-Id": "acme"},
)
```

### Binding parameters (session state injection)

```python
from google.adk.tools.toolbox_toolset import ToolboxToolset

# Bind dynamic values that are resolved at call time
toolbox = ToolboxToolset(
    server_url="http://localhost:5000",
    bound_params={
        "user_id": lambda: "current_user",     # callable → evaluated each call
        "region": "us-central1",               # static value
    },
)
```

Bound parameters are injected into every tool call by the underlying `toolbox_adk` library before the request reaches the server. The model does not see them in the function schema.

### Cleanup

`ToolboxToolset` implements `close()`. The `Runner` calls it automatically on `runner.close()`. If you manage the runner manually, call `await toolbox.close()` to release connections:

```python
async with runner:
    ...  # runner.__aexit__ calls toolbox.close()
```

### Gotchas

- `ToolboxToolset` raises `ImportError` with a clear message if `toolbox-adk` is not installed — unlike silent fallbacks in other toolsets.
- `get_tools` delegates entirely to the underlying `toolbox_adk.ToolboxToolset` — ADK's tool-name prefix and tool-filter features apply after the delegate returns its list.
- If the toolbox server is unreachable at agent startup, `get_tools` raises a network error at the first invocation (lazy-loaded). Validate connectivity before deploying.
- `toolset_name` and `tool_names` can be combined — the resulting toolset is the union of tools from the named toolset and the explicitly listed tools.

---

## 9 — `GcsArtifactService`

**Module:** `google.adk.artifacts.gcs_artifact_service`

`GcsArtifactService` persists agent artifacts to Google Cloud Storage. It is the production alternative to `InMemoryArtifactService` and `FileArtifactService`. It supports session-scoped and user-scoped (cross-session) artifacts, versioning, and custom metadata.

### Constructor

```python
GcsArtifactService(
    bucket_name: str,   # required; GCS bucket must exist
    **kwargs,           # forwarded to google.cloud.storage.Client()
)
```

`kwargs` can include `project=`, `credentials=`, `client_options=` — anything accepted by `google.cloud.storage.Client`.

### GCS key structure

Artifacts are stored under predictable path prefixes:

| Scope | GCS path |
|---|---|
| Session-scoped | `{app_name}/{user_id}/{session_id}/{filename}/{version}` |
| User-scoped (`user:` prefix) | `{app_name}/{user_id}/user/{filename}/{version}` |

`filename` may contain slashes (e.g. `reports/monthly/jan.pdf`) — they are preserved in the blob path.

### Basic usage

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.gcs_artifact_service import GcsArtifactService
from google.genai import types

artifact_service = GcsArtifactService(
    bucket_name="my-agent-artifacts",
    project="my-gcp-project",
)

agent = LlmAgent(
    name="file_processor",
    model="gemini-2.5-flash",
    instruction="Process files and save results as artifacts.",
)

app = App(name="file_app", root_agent=agent)
runner = Runner(
    app=app,
    session_service=InMemorySessionService(),
    artifact_service=artifact_service,
)
```

### Saving and loading artifacts

```python
from google.genai import types

# Save a text artifact (session-scoped)
version = await runner.artifact_service.save_artifact(
    app_name="file_app",
    user_id="u1",
    session_id="s1",
    filename="report.txt",
    artifact=types.Part(text="Monthly report content here."),
    custom_metadata={"generated_by": "report_agent", "quarter": "Q3"},
)
print(f"Saved as version {version}")  # → 0 (first version)

# Save a binary artifact
with open("chart.png", "rb") as f:
    data = f.read()

version2 = await runner.artifact_service.save_artifact(
    app_name="file_app",
    user_id="u1",
    session_id="s1",
    filename="chart.png",
    artifact=types.Part(
        inline_data=types.Blob(data=data, mime_type="image/png")
    ),
)

# Load latest version
part = await runner.artifact_service.load_artifact(
    app_name="file_app",
    user_id="u1",
    session_id="s1",
    filename="report.txt",
)
print(part.text)

# Load a specific version
part_v0 = await runner.artifact_service.load_artifact(
    app_name="file_app",
    user_id="u1",
    session_id="s1",
    filename="report.txt",
    version=0,
)
```

### User-scoped artifacts (cross-session)

```python
# Prefix filename with "user:" to make it cross-session
version = await runner.artifact_service.save_artifact(
    app_name="file_app",
    user_id="u1",
    session_id=None,   # session_id ignored for user-scoped artifacts
    filename="user:preferences.json",
    artifact=types.Part(text='{"theme": "dark"}'),
)

# Load from any session
prefs = await runner.artifact_service.load_artifact(
    app_name="file_app",
    user_id="u1",
    filename="user:preferences.json",
)
```

### Listing and deleting

```python
# List all artifact filenames for a session
keys = await runner.artifact_service.list_artifact_keys(
    app_name="file_app", user_id="u1", session_id="s1"
)
print(keys)  # ["chart.png", "report.txt"]

# List all versions with metadata
versions = await runner.artifact_service.list_artifact_versions(
    app_name="file_app", user_id="u1", session_id="s1", filename="report.txt"
)
for av in versions:
    print(av.version, av.canonical_uri, av.mime_type, av.create_time)

# Delete all versions of an artifact
await runner.artifact_service.delete_artifact(
    app_name="file_app", user_id="u1", session_id="s1", filename="report.txt"
)
```

### Workload Identity authentication

```python
from google.adk.artifacts.gcs_artifact_service import GcsArtifactService
import google.auth

# On GKE / Cloud Run — Application Default Credentials are picked up automatically
artifact_service = GcsArtifactService(bucket_name="prod-artifacts")
# No credentials= kwarg needed; google.cloud.storage.Client() calls google.auth.default()
```

### Gotchas

- All `save_artifact` / `load_artifact` / `list_*` / `delete_artifact` calls use `asyncio.to_thread` — they are non-blocking but run GCS operations in a thread pool. Under high concurrency, monitor the thread pool executor queue depth.
- Versioning is 0-indexed, monotonically increasing. `save_artifact` reads all existing versions and increments `max(versions) + 1`. Under concurrent writes for the same file, two callers may both compute the same next version and overwrite each other — use a coordinated write lock for concurrent agents sharing an artifact name.
- Session-scoped `save_artifact` raises `InputValidationError("Session ID must be provided for session-scoped artifacts.")` if `session_id=None` and the filename does not start with `user:`.
- `delete_artifact` deletes every version — it's all-or-nothing. There is no API to delete a single version.
- `custom_metadata` values are coerced to strings before storage as GCS blob metadata. Retrieve them from `ArtifactVersion.custom_metadata` (a `dict[str, str]`).
- `GcsArtifactService` requires `google-cloud-storage` — it's a lazy import (`from google.cloud import storage`) so the `ImportError` only fires at instantiation.

---

## 10 — `exit_loop` + `get_user_choice_tool`

**Module:** `google.adk.tools` (`google.adk.tools.exit_loop_tool`, `google.adk.tools.get_user_choice_tool`)

These are lightweight built-in control-flow tools. Neither is a class — both are module-level callables (or a `LongRunningFunctionTool` instance) that you drop directly into `tools=[]`.

### `exit_loop`

Source-verified from `google/adk/tools/exit_loop_tool.py`:

```python
def exit_loop(tool_context: ToolContext) -> None:
    """Exits the loop. Call only when instructed to do so."""
    tool_context.actions.escalate = True
    tool_context.actions.skip_summarization = True
```

Setting `escalate=True` causes the enclosing `LoopAgent` (or a `Workflow` loop node with `end_condition`) to terminate on the next iteration check. `skip_summarization=True` prevents the exit turn from being fed back to the model as a completion.

**Usage — model-controlled loop termination:**

```python
from google.adk.agents import LlmAgent, LoopAgent
from google.adk.tools.exit_loop_tool import exit_loop

refiner = LlmAgent(
    name="refiner",
    model="gemini-2.5-flash",
    instruction=(
        "Review the draft in state['draft']. "
        "If it meets quality criteria, call exit_loop. "
        "Otherwise, rewrite it and update state['draft']."
    ),
    tools=[exit_loop],
)

loop = LoopAgent(
    name="refine_loop",
    sub_agents=[refiner],
    max_iterations=5,
)
```

**Workflow note:**

`exit_loop` is designed for `LoopAgent`. In a `Workflow`, every cycle must include at least one conditional (route-based) `Edge` — an unconditional self-loop raises `ValidationError: Unconditional cycle detected` at construction time, even if the agent would call `exit_loop` at runtime. For model-controlled loop termination inside a `Workflow`, wire the loop with a conditional edge (the agent sets `context.route` to a specific value) and route to a terminal node on the exit condition. For simple Python-predicate termination, use the `end_condition` parameter on the loop node instead.

### `get_user_choice_tool`

Source-verified from `google/adk/tools/get_user_choice_tool.py`:

```python
# The callable (used as a long-running tool):
def get_user_choice(options: list[str], tool_context: ToolContext) -> Optional[str]:
    """Provides options to the user and asks them to choose one."""
    tool_context.actions.skip_summarization = True
    return None

# The pre-built LongRunningFunctionTool instance:
get_user_choice_tool = LongRunningFunctionTool(func=get_user_choice)
```

`get_user_choice` is a long-running tool — it returns `None` immediately and suspends the agent. The client is expected to present the `options` list to the user, collect a choice, and resume the tool with the selected value.

**Usage — branching dialog:**

```python
from google.adk.agents import LlmAgent
from google.adk.tools.get_user_choice_tool import get_user_choice_tool

agent = LlmAgent(
    name="booking_agent",
    model="gemini-2.5-flash",
    instruction=(
        "Help the user book a flight. "
        "When you need to ask the user to choose between options, "
        "use get_user_choice with a list of options."
    ),
    tools=[get_user_choice_tool],
)
```

**Usage — wiring the client-side response:**

When `get_user_choice` is called the first time, the tool returns `None` and the runner yields a pending function-call event. The client must capture the function-call ID from that event and resume by sending a `types.FunctionResponse` — not a plain user message — so the model receives the choice as the tool result.

```python
from google.genai import types

# Step 1 — first run: capture the pending function-call ID
pending_call_id = None
async for event in runner.run_async(
    user_id="u1",
    session_id="s1",
    new_message=types.Content(
        role="user",
        parts=[types.Part.from_text(text="Book me a flight to London")],
    ),
):
    if event.content and event.content.parts:
        for part in event.content.parts:
            if part.function_call and part.function_call.name == "get_user_choice":
                pending_call_id = part.function_call.id

# Step 2 — present options to the user, collect their answer, then resume
# by sending a FunctionResponse that matches the pending call's ID:
user_answer = "Morning flight"   # collected from UI / stdin
async for event in runner.run_async(
    user_id="u1",
    session_id="s1",
    new_message=types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=pending_call_id,
                    name="get_user_choice",
                    response={"result": user_answer},
                )
            )
        ],
    ),
):
    ...
```

### Comparison

| Aspect | `exit_loop` | `get_user_choice_tool` |
|---|---|---|
| Type | Plain function | `LongRunningFunctionTool` |
| Import | `from google.adk.tools.exit_loop_tool import exit_loop` | `from google.adk.tools.get_user_choice_tool import get_user_choice_tool` |
| Purpose | Terminate a loop from inside the model | Present options and wait for user selection |
| Suspends agent? | No — fires and completes | Yes — suspends until resumed |
| skip_summarization | Set to `True` | Set to `True` |
| Use with | `LoopAgent`, `Workflow` loop nodes | Any agent that needs interactive branching |

### Gotchas

- `exit_loop` must be in the `tools=[]` of the agent inside the loop. Adding it only to a parent agent has no effect.
- Calling `exit_loop` outside a loop context sets `escalate=True` globally — the invocation ends but the agent is not technically broken.
- `get_user_choice_tool` is a module-level singleton — import it directly. Do not instantiate a new `LongRunningFunctionTool(func=get_user_choice)` unless you need a renamed version.
- The model's response after `get_user_choice` fires will include the selected option as the tool result. Ensure your `instruction` tells the model how to interpret and act on that result.

---

## Version note

All content on this page is verified against **google-adk==2.8.0** (August 2026). Classes and modules may have changed in later releases. Check `inspect.getsource` against your installed version before using field names or constructor signatures in production.

| Area | Revision |
|---|---|
| 2026-08-31 | Initial page — 10 classes sourced from google-adk==2.8.0. |

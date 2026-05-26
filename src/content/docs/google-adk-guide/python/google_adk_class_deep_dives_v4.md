---
title: "Class deep dives — volume 4 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.1.0 classes: LongRunningFunctionTool, LiveRequestQueue/LiveRequest, AnthropicLlm/Claude, SqliteSessionService, ExecuteBashTool/BashToolPolicy, TransferToAgentTool, VertexAiMemoryBankService, SimplePromptOptimizer, SkillRegistry/Skill/SkillToolset, and ToolConfirmation."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 4"
  order: 63
---

Source-verified against **google-adk==2.1.0** (installed from PyPI, May 2026). Every field name, signature, and code example is taken directly from the installed package source.

| # | Class | Module | Status |
|---|---|---|---|
| 1 | `LongRunningFunctionTool` | `google.adk.tools.long_running_tool` | Stable |
| 2 | `LiveRequestQueue` + `LiveRequest` | `google.adk.agents.live_request_queue` | Stable |
| 3 | `AnthropicLlm` + `Claude` | `google.adk.models.anthropic_llm` | Stable |
| 4 | `SqliteSessionService` | `google.adk.sessions.sqlite_session_service` | Stable |
| 5 | `ExecuteBashTool` + `BashToolPolicy` | `google.adk.tools.bash_tool` | `@experimental` |
| 6 | `TransferToAgentTool` | `google.adk.tools.transfer_to_agent_tool` | Stable |
| 7 | `VertexAiMemoryBankService` | `google.adk.memory.vertex_ai_memory_bank_service` | Stable |
| 8 | `SimplePromptOptimizer` | `google.adk.optimization.simple_prompt_optimizer` | Stable |
| 9 | `SkillRegistry` + `Skill` + `SkillToolset` | `google.adk.skills` + `google.adk.tools.skill_toolset` | `@experimental` |
| 10 | `ToolConfirmation` | `google.adk.tools.tool_confirmation` | `@experimental` |

---

## 1 · `LongRunningFunctionTool`

`google.adk.tools.long_running_tool.LongRunningFunctionTool` is a thin subclass of `FunctionTool` designed for operations that may take significant wall-clock time (report generation, batch processing, long database queries). Wrapping a callable with this class does **two** things:

1. Sets `self.is_long_running = True` — the framework identifies this and handles the response asynchronously by function-call ID rather than expecting an immediate reply.
2. Appends a note to the function declaration's description instructing the model **not** to call the tool again if it has already received an intermediate or pending status — preventing double-invocations.

### Constructor (verified `long_running_tool.py`)

```python
LongRunningFunctionTool(func: Callable)
```

The tool inherits all `FunctionTool` schema inference — type annotations, docstrings, and `google.genai.types.Schema` are all honoured.

### Example 1 — batch report that returns a job ID first

```python
import asyncio
import uuid
from google.adk.agents import LlmAgent
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.runners import InMemoryRunner

# Simulate a long-running job store
_jobs: dict[str, str] = {}

async def generate_sales_report(region: str, quarter: str) -> dict:
    """Generate a quarterly sales report for the specified region.

    This operation takes 30–120 seconds. Returns job status immediately
    and the report when complete.

    Args:
        region: Sales region (e.g. "EMEA", "APAC", "AMER").
        quarter: Quarter identifier (e.g. "Q1-2026").

    Returns:
        A dict with 'job_id' and 'status'. When status is 'complete'
        the 'report_url' key is also present.
    """
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = "pending"

    # Simulate async work (would be a real DB query, BigQuery job, etc.)
    asyncio.create_task(_run_report_job(job_id, region, quarter))

    return {"job_id": job_id, "status": "pending",
            "message": f"Report started. Call check_report_status('{job_id}') to poll."}


async def _run_report_job(job_id: str, region: str, quarter: str):
    await asyncio.sleep(2)          # real work goes here
    _jobs[job_id] = "complete"


def check_report_status(job_id: str) -> dict:
    """Check the status of a report generation job.

    Args:
        job_id: The job ID returned by generate_sales_report.

    Returns:
        Dict with 'status' ('pending' | 'complete' | 'error') and, when
        complete, a 'report_url'.
    """
    status = _jobs.get(job_id, "not_found")
    if status == "complete":
        return {"job_id": job_id, "status": "complete",
                "report_url": f"gs://reports/{job_id}.pdf"}
    return {"job_id": job_id, "status": status}


agent = LlmAgent(
    name="report_agent",
    model="gemini-2.5-flash",
    description="Generates and retrieves sales reports.",
    instruction=(
        "You generate sales reports. When asked, call generate_sales_report. "
        "Then poll check_report_status until status is 'complete', then "
        "return the report_url to the user."
    ),
    tools=[
        LongRunningFunctionTool(generate_sales_report),  # marked long-running
        check_report_status,                              # normal FunctionTool
    ],
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="reports")
    await runner.session_service.create_session(
        app_name="reports", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Please generate a Q1-2026 EMEA sales report.", user_id="u1", session_id="s1"
    )
    for e in events:
        if e.content and e.content.role == "model":
            for part in e.content.parts:
                if part.text and not getattr(part, "thought", False):
                    print(part.text)

asyncio.run(main())
```

### Example 2 — data export with progress updates via state

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.runners import InMemoryRunner

async def export_dataset(
    dataset_id: str,
    format: str,
    tool_context: ToolContext,
) -> dict:
    """Export a dataset to a file. Runs in the background.

    Args:
        dataset_id: The ID of the dataset to export.
        format: Output format — 'csv', 'parquet', or 'json'.
    """
    # Write progress into session state so other tools/callbacks can read it
    tool_context.state["export_progress"] = 0
    tool_context.state["export_status"] = "starting"

    # In production, kick off a background task and return immediately
    asyncio.create_task(_do_export(dataset_id, format, tool_context))
    return {
        "status": "started",
        "dataset_id": dataset_id,
        "format": format,
        "message": "Export started. Check state['export_status'] for updates.",
    }

async def _do_export(dataset_id: str, format: str, tool_context: ToolContext):
    for pct in [25, 50, 75, 100]:
        await asyncio.sleep(0.5)
        tool_context.state["export_progress"] = pct
    tool_context.state["export_status"] = "complete"
    tool_context.state["export_url"] = f"gs://exports/{dataset_id}.{format}"


def get_export_status(tool_context: ToolContext) -> dict:
    """Return the current export progress from session state."""
    return {
        "status": tool_context.state.get("export_status", "idle"),
        "progress_pct": tool_context.state.get("export_progress", 0),
        "url": tool_context.state.get("export_url"),
    }


agent = LlmAgent(
    name="exporter",
    model="gemini-2.5-flash",
    instruction="Help users export datasets. Report progress until complete.",
    tools=[LongRunningFunctionTool(export_dataset), get_export_status],
)
```

### Key differences: `FunctionTool` vs `LongRunningFunctionTool`

| Aspect | `FunctionTool` | `LongRunningFunctionTool` |
|---|---|---|
| `is_long_running` | `False` | `True` |
| Model hint in description | None | "Do not call again if pending status seen" |
| Framework response handling | Synchronous — result returned in same turn | Async — result returned by `function_call_id` in a later turn |
| Best for | Sub-second operations | DB queries, external jobs, file generation (seconds to minutes) |

---

## 2 · `LiveRequestQueue` + `LiveRequest`

`google.adk.agents.live_request_queue.LiveRequestQueue` is the input channel for **bidirectional streaming (live) agents**. Instead of calling `runner.run()` with a single message, you feed an agent a `LiveRequestQueue` so you can stream audio blobs, text turns, and activity signals while the agent processes in parallel.

### `LiveRequest` fields (verified `live_request_queue.py`)

```python
class LiveRequest(BaseModel):
    content: Optional[types.Content] = None       # text/parts turn
    blob: Optional[types.Blob] = None             # audio/image blob (realtime)
    activity_start: Optional[types.ActivityStart] = None  # user started speaking
    activity_end: Optional[types.ActivityEnd] = None      # user finished speaking
    close: bool = False                            # shut down the queue
```

Priority when multiple fields set: `activity_start > activity_end > blob > content`.

### `LiveRequestQueue` API

```python
queue = LiveRequestQueue()

# Feed a typed text message
queue.send_content(types.Content(
    role="user",
    parts=[types.Part.from_text("Hello, what can you do?")]
))

# Feed a raw audio blob (16-bit PCM, 16 kHz)
queue.send_realtime(types.Blob(mime_type="audio/pcm", data=pcm_bytes))

# VAD signals for manual turn detection
queue.send_activity_start()   # user started speaking
queue.send_activity_end()     # user finished speaking (triggers model response)

# Send a pre-built LiveRequest
queue.send(LiveRequest(content=..., blob=...))

# Shut down (signals the runner to stop reading)
queue.close()

# Internal — the runner calls this to dequeue
request: LiveRequest = await queue.get()
```

### Example 1 — text-based live chat loop

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.agents.live_request_queue import LiveRequest, LiveRequestQueue
from google.adk.runners import InMemoryRunner
from google.genai import types

agent = LlmAgent(
    name="live_assistant",
    model="gemini-2.5-flash",
    instruction="You are a helpful real-time assistant.",
)

async def live_text_session():
    runner = InMemoryRunner(agent=agent, app_name="live_demo")
    session = await runner.session_service.create_session(
        app_name="live_demo", user_id="u1"
    )

    queue = LiveRequestQueue()

    async def send_messages():
        """Producer: feed messages into the queue."""
        for msg in ["Hi!", "What's 12 * 9?", "Thanks, bye!"]:
            await asyncio.sleep(0.5)
            queue.send_content(
                types.Content(role="user", parts=[types.Part.from_text(msg)])
            )
        await asyncio.sleep(1)
        queue.close()

    async def receive_events():
        """Consumer: process events emitted by the agent."""
        async for event in runner.run_live(
            user_id="u1",
            session_id=session.id,
            live_request_queue=queue,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text and not getattr(part, "thought", False):
                        print(f"Agent: {part.text}", flush=True)

    await asyncio.gather(send_messages(), receive_events())

asyncio.run(live_text_session())
```

### Example 2 — audio streaming with VAD signals

```python
import asyncio
import wave
from google.adk.agents import LlmAgent
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner
from google.genai import types

agent = LlmAgent(
    name="voice_agent",
    model="gemini-2.5-flash",   # use a model that supports native audio
    instruction="You are a voice assistant. Respond concisely.",
)

async def stream_audio_file(wav_path: str, queue: LiveRequestQueue, chunk_ms: int = 100):
    """Stream a WAV file in chunks to the live queue."""
    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        chunk_frames = int(sample_rate * chunk_ms / 1000)

        queue.send_activity_start()
        while True:
            frames = wf.readframes(chunk_frames)
            if not frames:
                break
            queue.send_realtime(types.Blob(
                mime_type=f"audio/pcm;rate={sample_rate}",
                data=frames,
            ))
            await asyncio.sleep(chunk_ms / 1000)  # real-time pacing
        queue.send_activity_end()

    await asyncio.sleep(3)  # allow agent to finish responding
    queue.close()


async def voice_session(wav_path: str):
    runner = InMemoryRunner(agent=agent, app_name="voice")
    session = await runner.session_service.create_session(
        app_name="voice", user_id="u1"
    )
    queue = LiveRequestQueue()

    async def producer():
        await stream_audio_file(wav_path, queue)

    async def consumer():
        async for event in runner.run_live(
            user_id="u1",
            session_id=session.id,
            live_request_queue=queue,
            run_config=RunConfig(
                response_modalities=["AUDIO"],  # ask for audio back
            ),
        ):
            if event.content:
                for part in event.content.parts:
                    if getattr(part, "inline_data", None):
                        # Write audio response to file
                        with open("response.pcm", "ab") as f:
                            f.write(part.inline_data.data)
                    elif part.text:
                        print(f"[transcript] {part.text}")

    await asyncio.gather(producer(), consumer())
```

### Example 3 — multi-turn live session with dynamic injection

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.runners import InMemoryRunner
from google.genai import types

agent = LlmAgent(
    name="dynamic_agent",
    model="gemini-2.5-flash",
    instruction="You are an assistant. The user may update context mid-conversation.",
)

async def dynamic_session():
    runner = InMemoryRunner(agent=agent, app_name="dynamic")
    session = await runner.session_service.create_session(
        app_name="dynamic", user_id="u1"
    )
    queue = LiveRequestQueue()

    async def producer():
        # Turn 1
        queue.send_content(types.Content(
            role="user", parts=[types.Part.from_text("Summarise the meeting notes.")]
        ))
        await asyncio.sleep(2)

        # Inject tool result / context mid-stream
        queue.send_content(types.Content(
            role="user",
            parts=[types.Part.from_text(
                "Additional context: Meeting was about Q2 budget. "
                "Key decisions: 10% headcount increase, new ML infra budget €2M."
            )],
        ))
        await asyncio.sleep(3)
        queue.close()

    async def consumer():
        async for event in runner.run_live(
            user_id="u1",
            session_id=session.id,
            live_request_queue=queue,
        ):
            if event.content and event.content.role == "model":
                for part in event.content.parts:
                    if part.text and not getattr(part, "thought", False):
                        print(part.text, end="", flush=True)
        print()

    await asyncio.gather(producer(), consumer())
```

---

## 3 · `AnthropicLlm` + `Claude`

`google.adk.models.anthropic_llm.AnthropicLlm` integrates Claude models (claude-3.5-sonnet, claude-opus-4, etc.) into ADK via the direct **Anthropic API**. `Claude` is a subclass that routes requests through **Vertex AI** instead (for GCP-managed deployments with IAM controls).

### Class hierarchy

```
BaseLlm
└── AnthropicLlm          ← direct Anthropic API
    └── Claude            ← Vertex AI endpoint (requires GOOGLE_CLOUD_PROJECT + LOCATION)
```

### Constructor fields (verified `anthropic_llm.py`)

```python
class AnthropicLlm(BaseLlm):
    model: str = "claude-sonnet-4-20250514"   # default Claude model
    max_tokens: int = 8192                    # max output tokens

class Claude(AnthropicLlm):
    model: str = "claude-3-5-sonnet-v2@20241022"  # default Vertex model
```

`supported_models()` matches patterns `r"claude-3-.*"` and `r"claude-.*-4.*"` — any model matching these regexes is auto-routed to `AnthropicLlm`.

### ThinkingConfig mapping (verified `_build_anthropic_thinking_param`)

| `thinking_budget` value | Anthropic thinking type |
|---|---|
| `None` | `ValueError` — must be explicit |
| `0` | `"disabled"` |
| `-1` (or any negative, e.g. `AUTOMATIC`) | `"adaptive"` (model chooses depth) — **required for Opus 4.7+** |
| `≥ 1024` (positive int) | `"enabled"` with `budget_tokens` — **deprecated for Opus 4.6+** |

### Example 1 — direct Anthropic API with extended thinking

```python
import asyncio
import os
from google.adk.agents import LlmAgent
from google.adk.models.anthropic_llm import AnthropicLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

os.environ["ANTHROPIC_API_KEY"] = "your-key-here"  # or set in environment

agent = LlmAgent(
    name="claude_thinker",
    model=AnthropicLlm(
        model="claude-opus-4-7",
        max_tokens=16000,
    ),
    instruction="You are an expert reasoning engine. Think carefully before answering.",
    generate_content_config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=-1,   # adaptive — required for claude-opus-4-7
        )
    ),
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="claude_demo")
    await runner.session_service.create_session(
        app_name="claude_demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Prove that there are infinitely many prime numbers.",
        user_id="u1", session_id="s1"
    )
    for e in events:
        if e.content and e.content.role == "model":
            for part in e.content.parts:
                if getattr(part, "thought", False):
                    print(f"[thinking] {part.text[:200]}...")
                elif part.text:
                    print(f"[answer] {part.text}")

asyncio.run(main())
```

### Example 2 — Vertex AI (GCP-managed Claude)

```python
import asyncio
import os
from google.adk.agents import LlmAgent
from google.adk.models.anthropic_llm import Claude
from google.adk.runners import InMemoryRunner
from google.genai import types

# Vertex AI requires these environment variables
os.environ["GOOGLE_CLOUD_PROJECT"] = "my-gcp-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-east5"  # Claude on Vertex regions

agent = LlmAgent(
    name="vertex_claude",
    # Use a Vertex resource path — Claude class strips to model ID automatically
    model=Claude(model="claude-3-5-sonnet-v2@20241022", max_tokens=4096),
    instruction="You are a helpful assistant deployed on Google Cloud.",
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="vertex_claude_demo")
    await runner.session_service.create_session(
        app_name="vertex_claude_demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "What is the capital of France?",
        user_id="u1", session_id="s1"
    )
    for e in events:
        if e.content and e.content.role == "model":
            for part in e.content.parts:
                if part.text and not getattr(part, "thought", False):
                    print(part.text)

asyncio.run(main())
```

### Example 3 — multi-model agent team (Gemini + Claude)

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.anthropic_llm import AnthropicLlm
from google.adk.tools.agent_tool import AgentTool
from google.adk.runners import InMemoryRunner

# Claude handles creative writing
writer_agent = LlmAgent(
    name="writer",
    model=AnthropicLlm(model="claude-sonnet-4-20250514"),
    description="Writes polished, creative prose and marketing copy.",
    instruction="Write engaging, creative content. Be concise and vivid.",
    mode="single_turn",
)

# Gemini handles orchestration and fact-checking
orchestrator = LlmAgent(
    name="orchestrator",
    model="gemini-2.5-flash",
    instruction=(
        "You coordinate content creation. For writing tasks, delegate to the "
        "writer sub-agent. For factual questions, answer directly."
    ),
    tools=[AgentTool(agent=writer_agent)],
)

async def main():
    runner = InMemoryRunner(agent=orchestrator, app_name="team")
    await runner.session_service.create_session(
        app_name="team", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Write a 3-sentence tagline for a new AI-powered coffee machine.",
        user_id="u1", session_id="s1"
    )
    for e in events:
        if e.content and e.content.role == "model":
            for part in e.content.parts:
                if part.text and not getattr(part, "thought", False):
                    print(part.text)

asyncio.run(main())
```

### `content_block_to_part` and `content_to_message_param` utility functions

These public helpers let you interop between ADK's `google.genai.types` world and Anthropic's message format:

```python
from google.adk.models.anthropic_llm import (
    content_to_message_param,
    content_block_to_part,
    function_declaration_to_tool_param,
)
from google.genai import types
import anthropic

# Convert ADK content → Anthropic MessageParam
adk_content = types.Content(
    role="user",
    parts=[types.Part.from_text("What is machine learning?")]
)
anthropic_msg = content_to_message_param(adk_content)
# → {"role": "user", "content": [{"type": "text", "text": "What is machine learning?"}]}

# Convert Anthropic ContentBlock → ADK Part
text_block = anthropic.types.TextBlock(type="text", text="It's a field of AI...")
adk_part = content_block_to_part(text_block)
# → types.Part with .text set

# Convert FunctionDeclaration → Anthropic ToolParam
fn_decl = types.FunctionDeclaration(
    name="get_weather",
    description="Get weather for a city.",
    parameters=types.Schema(
        type="OBJECT",
        properties={"city": types.Schema(type="STRING")},
        required=["city"],
    ),
)
tool_param = function_declaration_to_tool_param(fn_decl)
```

---

## 4 · `SqliteSessionService`

`google.adk.sessions.sqlite_session_service.SqliteSessionService` persists sessions (state, events) to a local SQLite database via `aiosqlite`. It is the recommended choice for **local development**, **CLI tools**, and single-node servers where you want session history to survive process restarts without a cloud dependency.

### Constructor

```python
SqliteSessionService(db_path: str)
```

`db_path` accepts:
- A plain filesystem path: `"sessions.db"` or `"/var/data/sessions.db"`
- A SQLAlchemy-style URL: `"sqlite:///relative.db"` or `"sqlite:////absolute.db"`
- A URI with query params (for WAL mode, etc.): `"sqlite:///sessions.db?mode=wal"`

The service auto-creates 4 tables on first connect: `app_states`, `user_states`, `sessions`, `events`. States are stored as JSON and updated atomically with SQLite's `json_patch` on each delta — no full-row overwrites.

### Schema overview

```sql
-- App-scoped state (shared across all users/sessions of an app)
app_states(app_name PK, state JSON, update_time REAL)

-- User-scoped state (shared across all sessions for app+user)
user_states(app_name, user_id PK, state JSON, update_time REAL)

-- Session metadata + session-local state
sessions(app_name, user_id, id PK, state JSON, create_time, update_time)

-- Event log with foreign-key to sessions (CASCADE DELETE)
events(id, app_name, user_id, session_id FK, invocation_id, timestamp, event_data JSON)
```

State prefixes mirror `InMemorySessionService`:
- `app:<key>` → written to `app_states`
- `user:<key>` → written to `user_states`
- bare `<key>` → written to `sessions.state`

### Example 1 — basic persistent session

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory import InMemoryMemoryService
from google.adk.apps import App

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant. Remember what the user tells you.",
)

session_service = SqliteSessionService("conversations.db")

app = App(
    name="my_app",
    agent=agent,
    session_service=session_service,
    artifact_service=InMemoryArtifactService(),
    memory_service=InMemoryMemoryService(),
)

async def main():
    runner = app.build()

    # Create a session (persistent across restarts)
    session = await session_service.create_session(
        app_name="my_app",
        user_id="alice",
        session_id="conv-001",
        state={"user:display_name": "Alice"},  # user-scoped, persisted to user_states
    )

    # First run
    events = await runner.run_debug(
        "My name is Alice and I love hiking.", user_id="alice", session_id="conv-001"
    )

    # --- Restart simulation: create a fresh runner with the same DB ---
    session2 = await session_service.get_session(
        app_name="my_app", user_id="alice", session_id="conv-001"
    )
    print("Restored session state:", session2.state)
    print("Event count:", len(session2.events))

asyncio.run(main())
```

### Example 2 — listing and cleaning up sessions

```python
import asyncio
from google.adk.sessions.sqlite_session_service import SqliteSessionService

session_service = SqliteSessionService("conversations.db")

async def admin_tasks():
    # List all sessions for a user
    result = await session_service.list_sessions(app_name="my_app", user_id="alice")
    for s in result.sessions:
        print(f"session {s.id}: last_update={s.last_update_time:.0f}, "
              f"state_keys={list(s.state.keys())}")

    # Paginated event fetch (last 10 events only)
    from google.adk.sessions.base_session_service import GetSessionConfig
    recent = await session_service.get_session(
        app_name="my_app",
        user_id="alice",
        session_id="conv-001",
        config=GetSessionConfig(num_recent_events=10),
    )
    print(f"Fetched {len(recent.events)} recent events")

    # Delete a session (cascades to events via FK)
    await session_service.delete_session(
        app_name="my_app", user_id="alice", session_id="conv-001"
    )
    print("Session deleted.")

asyncio.run(admin_tasks())
```

### Example 3 — WAL mode + custom connection options for high-throughput servers

```python
from google.adk.sessions.sqlite_session_service import SqliteSessionService

# WAL mode survives concurrent readers + one writer without blocking
# Pass URI query params — aiosqlite forwards them via sqlite3's URI API
session_service = SqliteSessionService(
    "sqlite:////var/data/sessions.db?mode=wal&cache=shared"
)
```

### Example 4 — state scoping (app / user / session)

```python
import asyncio
from google.adk.sessions.sqlite_session_service import SqliteSessionService

svc = SqliteSessionService(":memory:")   # in-memory SQLite for tests

async def demo():
    # Create session with all three scope levels
    session = await svc.create_session(
        app_name="shop",
        user_id="bob",
        state={
            "app:feature_flags": {"new_checkout": True},  # → app_states
            "user:tier": "premium",                        # → user_states
            "cart_items": [],                              # → sessions.state
        },
    )

    # Retrieve: merged view (app + user + session)
    s = await svc.get_session(app_name="shop", user_id="bob", session_id=session.id)
    assert s.state["app:feature_flags"]["new_checkout"] is True
    assert s.state["user:tier"] == "premium"
    assert s.state["cart_items"] == []

asyncio.run(demo())
```

### Migration note

If you have a database created with an older ADK version (before `event_data` column), the constructor raises `RuntimeError` with migration instructions:

```bash
python -m google.adk.sessions.migration.migrate_from_sqlalchemy_sqlite \
    --source_db_path old.db --dest_db_path old.db.new
```

---

## 5 · `ExecuteBashTool` + `BashToolPolicy`

`google.adk.tools.bash_tool.ExecuteBashTool` lets an agent run shell commands in a sandboxed workspace. It is decorated `@experimental(FeatureName.SKILL_TOOLSET)` — enable the feature flag before using it.

Every invocation **always** requests user confirmation via `ToolConfirmation` before executing — the tool cannot be used without this confirmation step.

### `BashToolPolicy` fields (verified `bash_tool.py`)

```python
@dataclasses.dataclass(frozen=True)
class BashToolPolicy:
    allowed_command_prefixes: tuple[str, ...] = ("*",)    # ("*",) = allow all
    blocked_operators: tuple[str, ...] = ()                # e.g. ("|", ";", "&&")
    timeout_seconds: Optional[int] = 30
    max_memory_bytes: Optional[int] = None                 # subprocess RLIMIT_AS
    max_file_size_bytes: Optional[int] = None              # RLIMIT_FSIZE
    max_child_processes: Optional[int] = None              # RLIMIT_NPROC
```

### `ExecuteBashTool` constructor

```python
ExecuteBashTool(
    workspace: pathlib.Path | None = None,   # defaults to cwd()
    policy: Optional[BashToolPolicy] = None, # defaults to BashToolPolicy()
)
```

### Example 1 — restricted bash tool for a code agent

```python
import asyncio
import os
import pathlib
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.bash_tool import ExecuteBashTool, BashToolPolicy
from google.adk import features

# Enable the experimental flag
os.environ["GOOGLE_ADK_FEATURE_SKILL_TOOLSET"] = "true"
features.enable(features.FeatureName.SKILL_TOOLSET)

workspace = pathlib.Path("/tmp/agent_workspace")
workspace.mkdir(exist_ok=True)

policy = BashToolPolicy(
    allowed_command_prefixes=(
        "ls", "cat", "head", "tail", "grep",
        "python", "pip", "pytest",
    ),
    blocked_operators=("|", ";", "&&", "||", "`", "$("),
    timeout_seconds=60,
    max_memory_bytes=512 * 1024 * 1024,   # 512 MB
    max_file_size_bytes=100 * 1024 * 1024, # 100 MB
    max_child_processes=4,
)

agent = LlmAgent(
    name="code_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a code assistant. You can run safe bash commands in the "
        "workspace. Always ask for confirmation before executing. "
        "Allowed commands: ls, cat, head, tail, grep, python, pip, pytest."
    ),
    tools=[ExecuteBashTool(workspace=workspace, policy=policy)],
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="code")
    await runner.session_service.create_session(
        app_name="code", user_id="dev", session_id="s1"
    )
    events = await runner.run_debug(
        "List the Python files in the workspace.",
        user_id="dev", session_id="s1"
    )
    for e in events:
        if e.content and e.content.role == "model":
            for part in e.content.parts:
                if part.text:
                    print(part.text)

asyncio.run(main())
```

### Example 2 — `BashToolPolicy` command validation

```python
from google.adk.tools.bash_tool import BashToolPolicy, _validate_command

policy = BashToolPolicy(
    allowed_command_prefixes=("ls", "cat", "grep"),
    blocked_operators=("|", ";"),
)

# These pass:
assert _validate_command("ls -la /tmp", policy) is None
assert _validate_command("cat README.md", policy) is None

# These fail:
print(_validate_command("rm -rf /", policy))
# "Command blocked. Permitted prefixes are: ls, cat, grep"

print(_validate_command("ls | grep .py", policy))
# "Command contains blocked operator: |"
```

### Tool response structure

On success:
```python
{
    "stdout": "file1.py\nfile2.py\n",
    "stderr": "",
    "returncode": 0,
}
```

On timeout:
```python
{
    "error": "Command timed out after 30 seconds.",
    "stdout": "...",
    "stderr": "...",
    "returncode": -9,
}
```

On rejected confirmation:
```python
{"error": "This tool call is rejected."}
```

---

## 6 · `TransferToAgentTool`

`google.adk.tools.transfer_to_agent_tool.TransferToAgentTool` is a `FunctionTool` subclass that provides **enum-constrained agent handoff**. Unlike the low-level `transfer_to_agent()` function which accepts any string, `TransferToAgentTool` bakes the valid agent names into the JSON Schema as an `enum` — preventing the model from hallucinating invalid agent names.

### Constructor (verified `transfer_to_agent_tool.py`)

```python
TransferToAgentTool(agent_names: list[str])
```

Internally calls `tool_context.actions.transfer_to_agent = agent_name`, which the runner intercepts to route the conversation.

### Example 1 — hub-and-spoke routing with enum safety

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool
from google.adk.runners import InMemoryRunner

billing_agent = LlmAgent(
    name="billing_agent",
    model="gemini-2.5-flash",
    description="Handles billing queries, invoices, and payment issues.",
    instruction="You specialise in billing. Help with invoices, payments, and subscriptions.",
    mode="single_turn",
)

tech_agent = LlmAgent(
    name="tech_support",
    model="gemini-2.5-flash",
    description="Handles technical issues, bug reports, and product errors.",
    instruction="You are technical support. Debug issues, provide workarounds.",
    mode="single_turn",
)

sales_agent = LlmAgent(
    name="sales_agent",
    model="gemini-2.5-flash",
    description="Handles sales enquiries, pricing, and new account setup.",
    instruction="You handle sales enquiries. Provide pricing and upsell.",
    mode="single_turn",
)

router = LlmAgent(
    name="router",
    model="gemini-2.5-flash",
    description="Routes customer enquiries to the correct specialist.",
    instruction=(
        "You are a customer service router. Determine which specialist the "
        "customer needs: billing_agent, tech_support, or sales_agent. "
        "Transfer immediately — do not answer yourself."
    ),
    sub_agents=[billing_agent, tech_agent, sales_agent],
    tools=[
        # Enum constraint prevents hallucinated agent names
        TransferToAgentTool(
            agent_names=["billing_agent", "tech_support", "sales_agent"]
        )
    ],
    disallow_transfer_to_parent=True,
)

async def main():
    runner = InMemoryRunner(agent=router, app_name="support")
    await runner.session_service.create_session(
        app_name="support", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "I was charged twice for my subscription this month!",
        user_id="u1", session_id="s1"
    )
    for e in events:
        if e.content and e.content.role == "model":
            for part in e.content.parts:
                if part.text and not getattr(part, "thought", False):
                    print(part.text)

asyncio.run(main())
```

### Example 2 — the low-level `transfer_to_agent` function

When you need to transfer imperatively inside a tool (not via the model), use the underlying function directly:

```python
from google.adk.tools.transfer_to_agent_tool import transfer_to_agent
from google.adk.tools.tool_context import ToolContext

def smart_router(query: str, tool_context: ToolContext) -> str:
    """Route queries programmatically without relying on the LLM."""
    q = query.lower()
    if any(w in q for w in ["bill", "invoice", "payment", "charge"]):
        transfer_to_agent("billing_agent", tool_context)
        return "Transferring to billing..."
    elif any(w in q for w in ["error", "bug", "crash", "broken"]):
        transfer_to_agent("tech_support", tool_context)
        return "Transferring to tech support..."
    else:
        transfer_to_agent("sales_agent", tool_context)
        return "Transferring to sales..."
```

### `TransferToAgentTool` vs `sub_agents` vs `AgentTool`

| Method | Transfer style | Model constrained? | Notes |
|---|---|---|---|
| `sub_agents=[...]` | Implicit, auto-routed | No — model can hallucinate | Framework injects routing instructions |
| `TransferToAgentTool(agent_names)` | Explicit tool call | **Yes** — enum constraint | Recommended for large agent teams |
| `AgentTool(agent)` | Sub-call, result returned | N/A | Keeps conversation in parent context |
| `transfer_to_agent()` function | Programmatic | N/A | Use inside tool functions for imperative routing |

---

## 7 · `VertexAiMemoryBankService`

`google.adk.memory.vertex_ai_memory_bank_service.VertexAiMemoryBankService` persists and searches long-term memory using **Vertex AI Memory Bank** (part of Agent Engine). Memories are extracted from session events, stored with semantic indexing, and recalled automatically via `PreloadMemoryTool` / `LoadMemoryTool`, or on-demand via `search_memory`.

### Constructor (verified `vertex_ai_memory_bank_service.py`)

```python
VertexAiMemoryBankService(
    project: Optional[str] = None,
    location: Optional[str] = None,
    agent_engine_id: str,      # REQUIRED — e.g. "456"
    *,
    express_mode_api_key: Optional[str] = None,
)
```

`agent_engine_id` is the **numeric ID** of your Reasoning Engine resource. Extract it from a full resource name like this:

```python
agent_engine_id = agent_engine.api_resource.name.split("/")[-1]  # "456"
```

Requires `google-cloud-aiplatform` (install with `pip install google-adk[gcp]`).

### `add_session_to_memory` vs direct `add_memory`

The service supports two ingestion paths:
1. **`add_session_to_memory(session)`** — extracts memories from conversation events using the Vertex AI `ingest_events` API (default) or `generate_memories` API (when `custom_metadata` contains `GenerateMemories`-specific keys like `disable_consolidation`).
2. **`context.add_memory(memories=[...])`** — (ADK 2.1.0+) inject explicit `MemoryEntry` objects directly from agent code.

### Example 1 — persistent memory across sessions

```python
import asyncio
import os
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.memory.vertex_ai_memory_bank_service import VertexAiMemoryBankService
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

os.environ["GOOGLE_CLOUD_PROJECT"] = "my-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

memory_service = VertexAiMemoryBankService(
    project="my-project",
    location="us-central1",
    agent_engine_id="123456789",   # your Reasoning Engine numeric ID
)

agent = LlmAgent(
    name="personal_assistant",
    model="gemini-2.5-flash",
    instruction=(
        "You are a personal assistant with long-term memory. "
        "Use the preloaded memories to personalise your responses."
    ),
    tools=[PreloadMemoryTool()],   # auto-injects recalled memories into context
)

app = App(
    name="assistant_app",
    agent=agent,
    session_service=SqliteSessionService("sessions.db"),
    memory_service=memory_service,
)

async def main():
    runner = app.build()

    # Session 1: user tells assistant their preferences
    session1 = await app.session_service.create_session(
        app_name="assistant_app", user_id="alice", session_id="sess-1"
    )
    await runner.run_debug(
        "I prefer concise responses and I work in fintech.",
        user_id="alice", session_id="sess-1"
    )
    # Save this session's events to memory
    await memory_service.add_session_to_memory(session1)

    # Session 2: new session — memories are recalled automatically via PreloadMemoryTool
    session2 = await app.session_service.create_session(
        app_name="assistant_app", user_id="alice", session_id="sess-2"
    )
    events = await runner.run_debug(
        "Summarise what you know about my preferences.",
        user_id="alice", session_id="sess-2"
    )
    for e in events:
        if e.content and e.content.role == "model":
            for part in e.content.parts:
                if part.text and not getattr(part, "thought", False):
                    print(part.text)

asyncio.run(main())
```

### Example 2 — direct memory search

```python
import asyncio
from google.adk.memory.vertex_ai_memory_bank_service import VertexAiMemoryBankService

memory_service = VertexAiMemoryBankService(
    project="my-project",
    location="us-central1",
    agent_engine_id="123456789",
)

async def search():
    results = await memory_service.search_memory(
        app_name="assistant_app",
        user_id="alice",
        query="What is Alice's job industry?",
    )
    for entry in results.memories:
        print(f"[{entry.score:.2f}] {entry.content}")

asyncio.run(search())
```

### Example 3 — add explicit memories via `context.add_memory` (ADK 2.1.0+)

```python
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.apps import App

def before_agent_callback(context: Context) -> None:
    """Inject explicit memories from an external knowledge base."""
    context.add_memory(memories=[
        MemoryEntry(
            content="User prefers dark mode in all applications.",
            author="onboarding_system",
        ),
        MemoryEntry(
            content="User is an expert in Python and TypeScript.",
            author="profile_service",
        ),
    ])

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="Use the injected memories to personalise responses.",
    before_agent_callback=before_agent_callback,
)
```

---

## 8 · `SimplePromptOptimizer`

`google.adk.optimization.simple_prompt_optimizer.SimplePromptOptimizer` implements iterative **automated prompt optimisation**. Given an agent and evaluation samples, it repeatedly generates candidate prompts using an LLM meta-optimizer, scores them on a random batch, and keeps the best performer. The `Sampler` abstraction handles evaluation execution.

### `SimplePromptOptimizerConfig` fields (verified `simple_prompt_optimizer.py`)

```python
class SimplePromptOptimizerConfig(BaseModel):
    optimizer_model: str = "gemini-2.5-flash"     # LLM that generates new prompts
    model_configuration: GenerateContentConfig     # defaults to thinking_budget=10240
    num_iterations: int = 10                       # optimization rounds
    batch_size: int = 5                            # training examples per round
```

### How it works (verified source)

1. **Baseline**: Scores the initial agent on a random batch of `batch_size` training examples.
2. **Loop** (`num_iterations` times):
   a. Calls the meta-optimizer LLM with the current prompt + current score → generates a new candidate prompt.
   b. Clones the agent (`best_agent.clone(update={"instruction": new_prompt})`) — preserving all other fields.
   c. Scores the candidate on a fresh random batch.
   d. Keeps the candidate only if its score is strictly better.
3. **Final validation**: Runs the best agent over the full validation split.

### Example 1 — optimize a customer support agent

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.optimization.simple_prompt_optimizer import (
    SimplePromptOptimizer,
    SimplePromptOptimizerConfig,
)

# --- Define the agent to optimise ---
initial_agent = LlmAgent(
    name="support_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a customer support agent. "
        "Help users with their questions."  # deliberately weak starting prompt
    ),
)

# --- Config: 5 iterations, batch of 3, using Flash for meta-optimisation ---
config = SimplePromptOptimizerConfig(
    optimizer_model="gemini-2.5-flash",
    num_iterations=5,
    batch_size=3,
)
optimizer = SimplePromptOptimizer(config=config)

# --- Sampler implementation (pseudocode — implement for your eval framework) ---
# In practice, Sampler calls your evaluation pipeline.
# See google.adk.optimization.sampler.Sampler for the interface.

# async def run_optimization():
#     result = await optimizer.optimize(
#         initial_agent=initial_agent,
#         sampler=my_sampler,
#     )
#     best = result.optimized_agents[0]
#     print(f"Final validation score: {best.overall_score:.3f}")
#     print(f"Optimised prompt:\n{best.optimized_agent.instruction}")
```

### Example 2 — using the optimizer meta-prompt template

The optimizer uses a fixed template (source-verified):

```
You are an expert prompt engineer. Your task is to improve the system prompt
for an AI agent. The agent's current prompt achieved an average score of
{current_score:.2f} on a set of evaluation tasks. A higher score is better.

<current_prompt>
{current_prompt_text}
</current_prompt>

Based on the current prompt, rewrite it to create a new, improved version...
Output only the new, full, improved agent prompt.
```

You can influence optimisation direction by seeding the initial prompt with structural elements the meta-optimizer tends to preserve and extend:

```python
initial_agent = LlmAgent(
    name="support_agent",
    model="gemini-2.5-flash",
    instruction="""
    ## Role
    You are a customer support specialist for [Product].

    ## Responsibilities
    - Answer billing questions
    - Troubleshoot technical issues
    - Escalate complex cases

    ## Constraints
    - Never share internal policies
    - Respond in < 3 sentences unless detail is explicitly requested
    """,
)
```

### `SimplePromptOptimizer` vs `AgentOptimizer` base class

```python
from google.adk.optimization.agent_optimizer import AgentOptimizer

# AgentOptimizer is the abstract base — implement your own optimization strategy:
class MyCustomOptimizer(AgentOptimizer):
    async def optimize(self, initial_agent, sampler):
        # your optimization logic
        ...
```

`SimplePromptOptimizer` is the only built-in implementation. For `GEPA` (Generative Evolutionary Prompt Architecture), see `gepa_root_agent_prompt_optimizer.py` in the same package.

---

## 9 · `SkillRegistry` + `Skill` + `SkillToolset`

ADK's **Skills** system (all `@experimental(FeatureName.SKILL_TOOLSET)`) provides a structured way to package reusable agent capabilities as markdown-based skill bundles that agents can discover, load, and execute.

### Three-layer skill structure

| Layer | Content | When loaded |
|---|---|---|
| **L1** `Frontmatter` | `name`, `description`, `allowed_tools`, `metadata` | Always — for discovery/search |
| **L2** `instructions` | SKILL.md body — detailed step-by-step instructions | When the skill is triggered |
| **L3** `Resources` | `references`, `assets`, `scripts` | On demand |

### `Skill` model fields (verified `skills/models.py`)

```python
class Skill(BaseModel):
    frontmatter: Frontmatter
    instructions: str
    resources: Resources = Resources()

    @property
    def name(self) -> str: return self.frontmatter.name
    @property
    def description(self) -> str: return self.frontmatter.description

class Frontmatter(BaseModel):
    name: str            # kebab-case, max 64 chars
    description: str     # max 1024 chars
    license: Optional[str] = None
    compatibility: Optional[str] = None
    allowed_tools: Optional[str] = None   # space-delimited pre-approved tools
    metadata: dict[str, Any] = {}         # adk_additional_tools supported

class Resources(BaseModel):
    references: dict[str, str | bytes] = {}  # markdown instructions
    assets: dict[str, str | bytes] = {}      # schemas, templates, examples
    scripts: dict[str, Script] = {}          # executable bash scripts
```

### `SkillRegistry` interface (verified `skill_registry.py`)

```python
class SkillRegistry(ABC):
    @abstractmethod
    async def get_skill(self, *, name: str) -> Skill: ...

    @abstractmethod
    async def search_skills(self, *, query: str) -> list[Frontmatter]: ...

    def search_tool_description(self) -> str | None: ...  # custom search hint
```

### Example 1 — building and loading skills from a directory

The `load_skill_from_dir` and `list_skills_in_dir` helpers read skill bundles from the filesystem:

```python
import asyncio
import pathlib
from google.adk.skills import (
    load_skill_from_dir,
    list_skills_in_dir,
    Skill,
    Frontmatter,
    Resources,
    Script,
)

# Skills are stored as directories:
# skills/
#   data-analysis/
#     SKILL.md        ← frontmatter (YAML front matter) + L2 instructions
#     schema.sql      ← asset
#     analyse.py      ← script

async def explore_skills():
    skills_dir = pathlib.Path("./skills")
    skills_dir.mkdir(exist_ok=True)

    # Create a minimal skill directory for demo
    skill_dir = skills_dir / "sql-query"
    skill_dir.mkdir(exist_ok=True)
    (skill_dir / "SKILL.md").write_text("""\
---
name: sql-query
description: >
  Execute SQL queries against the connected database.
  Use this skill when the user asks data questions that require SQL.
allowed-tools: execute_bash
---

## Instructions

1. Parse the user's natural-language request into a SQL query.
2. Validate the query against the schema in `schema.sql`.
3. Execute using the `execute_bash` tool with `sqlite3 data.db`.
4. Format the results as a markdown table.

### Safety rules
- Only SELECT queries are allowed.
- Always add LIMIT 1000 unless the user explicitly asks for all rows.
""")
    (skill_dir / "schema.sql").write_text(
        "CREATE TABLE orders (id INT, user_id INT, amount DECIMAL, created_at DATE);\n"
    )

    # Load skill
    skill = await load_skill_from_dir(skill_dir)
    print(f"Skill: {skill.name}")
    print(f"Description: {skill.description}")
    print(f"Instructions length: {len(skill.instructions)} chars")

    # List all skills
    frontmatters = await list_skills_in_dir(skills_dir)
    for fm in frontmatters:
        print(f"  • {fm.name}: {fm.description[:60]}")

asyncio.run(explore_skills())
```

### Example 2 — custom `SkillRegistry` implementation

```python
import asyncio
from typing import Any
from google.adk.skills import SkillRegistry, Skill, Frontmatter, Resources

class InMemorySkillRegistry(SkillRegistry):
    """Simple in-memory registry for testing and demos."""

    def __init__(self, skills: list[Skill]):
        self._skills = {s.name: s for s in skills}

    async def get_skill(self, *, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' not found")
        return self._skills[name]

    async def search_skills(self, *, query: str) -> list[Frontmatter]:
        q = query.lower()
        return [
            s.frontmatter
            for s in self._skills.values()
            if q in s.name.lower() or q in s.description.lower()
        ]

    def search_tool_description(self) -> str:
        return "Search for available skills by keyword. Returns skill names and descriptions."


# Build a registry with two skills
registry = InMemorySkillRegistry(skills=[
    Skill(
        frontmatter=Frontmatter(
            name="summarise-document",
            description="Summarise a long document into bullet points.",
        ),
        instructions=(
            "1. Extract the main sections.\n"
            "2. For each section, write 1-3 bullet points.\n"
            "3. Limit total summary to 200 words.\n"
        ),
    ),
    Skill(
        frontmatter=Frontmatter(
            name="translate-text",
            description="Translate text between languages.",
        ),
        instructions=(
            "1. Detect the source language.\n"
            "2. Translate to the requested target language.\n"
            "3. Preserve formatting and tone.\n"
        ),
    ),
])

async def demo():
    results = await registry.search_skills(query="translate")
    print([fm.name for fm in results])  # ['translate-text']

    skill = await registry.get_skill(name="summarise-document")
    print(skill.instructions[:80])

asyncio.run(demo())
```

### Example 3 — attaching `SkillToolset` to an agent

```python
import asyncio
import os
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.skill_toolset import SkillToolset
from google.adk import features

os.environ["GOOGLE_ADK_FEATURE_SKILL_TOOLSET"] = "true"
features.enable(features.FeatureName.SKILL_TOOLSET)

# Assuming `registry` from the previous example
skill_toolset = SkillToolset(skill_registry=registry)

agent = LlmAgent(
    name="versatile_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You can discover and use skills to help with tasks. "
        "Search for relevant skills before attempting complex tasks."
    ),
    tools=[skill_toolset],
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="skills_demo")
    await runner.session_service.create_session(
        app_name="skills_demo", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Please summarise the following document: [long document text here]",
        user_id="u1", session_id="s1"
    )
    for e in events:
        if e.content and e.content.role == "model":
            for part in e.content.parts:
                if part.text and not getattr(part, "thought", False):
                    print(part.text)

asyncio.run(main())
```

### `allowed_tools` in `Frontmatter`

The `allowed_tools` field (YAML key: `allowed-tools`) contains a space-delimited list of tools that are pre-approved for this skill — bypassing normal `ToolConfirmation` flow. E.g.:

```yaml
---
name: file-ops
description: Perform safe file operations.
allowed-tools: read_file write_file list_directory
---
```

---

## 10 · `ToolConfirmation`

`google.adk.tools.tool_confirmation.ToolConfirmation` is the data model for **Human-in-the-Loop (HITL) tool approval**. It is decorated `@experimental(FeatureName.TOOL_CONFIRMATION)`. When a tool needs explicit human approval before executing, it returns early with a `ToolConfirmation` request; the framework pauses, surfaces the request to the UI, and re-invokes the tool with the user's decision.

### Fields (verified `tool_confirmation.py`)

```python
class ToolConfirmation(BaseModel):
    hint: str = ""            # explains WHY confirmation is needed
    confirmed: bool = False   # True = approved, False = pending/rejected
    payload: Optional[Any] = None   # extra data needed from the user (JSON-serialisable)
```

The `model_config` uses `alias_generator=alias_generators.to_camel` and `populate_by_name=True` — fields can be accessed as snake_case in Python but are serialised camelCase in JSON (`isConfirmed`, etc. — but note fields here are short enough that camel ≈ snake).

### How `ToolContext.request_confirmation` works

When a tool calls `tool_context.request_confirmation(hint="...")`:

1. `tool_context.tool_confirmation` is set to a `ToolConfirmation(hint=..., confirmed=False)`.
2. `tool_context.actions.skip_summarization = True` — prevents premature summarisation.
3. The tool returns an error message.
4. The framework surfaces the `ToolConfirmation` to the caller (UI, test harness, etc.).
5. The caller re-invokes with the tool's function-call args **plus** a `ToolConfirmation(confirmed=True, payload=...)` attached.
6. On re-invocation, `tool_context.tool_confirmation.confirmed` is `True` → the tool proceeds.

### Example 1 — database write with approval gate

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.runners import InMemoryRunner
from google.adk import features
import os

os.environ["GOOGLE_ADK_FEATURE_TOOL_CONFIRMATION"] = "true"
features.enable(features.FeatureName.TOOL_CONFIRMATION)


async def delete_customer_record(
    customer_id: str,
    tool_context: ToolContext,
) -> dict:
    """Permanently delete a customer record from the database.

    Args:
        customer_id: The ID of the customer to delete.

    Returns:
        Success or error dict.
    """
    # First invocation: request confirmation
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=(
                f"You are about to permanently delete customer '{customer_id}'. "
                "This action CANNOT be undone. Type 'CONFIRM' to proceed."
            ),
        )
        tool_context.actions.skip_summarization = True
        return {
            "status": "awaiting_confirmation",
            "message": "This action requires explicit approval before execution.",
        }

    # Second invocation: check approval
    if not tool_context.tool_confirmation.confirmed:
        return {"status": "rejected", "message": "Deletion cancelled by user."}

    # Approved — proceed with deletion
    # db.delete_customer(customer_id)  # your actual DB call
    return {
        "status": "success",
        "message": f"Customer {customer_id} has been permanently deleted.",
    }


agent = LlmAgent(
    name="admin_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a database admin assistant. You can delete customer records "
        "but ALWAYS request confirmation first."
    ),
    tools=[delete_customer_record],
)
```

### Example 2 — financial transaction with amount-aware confirmation

```python
import asyncio
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.adk import features
import os

os.environ["GOOGLE_ADK_FEATURE_TOOL_CONFIRMATION"] = "true"
features.enable(features.FeatureName.TOOL_CONFIRMATION)


async def transfer_funds(
    from_account: str,
    to_account: str,
    amount_gbp: float,
    tool_context: ToolContext,
) -> dict:
    """Transfer funds between accounts.

    Args:
        from_account: Source account number.
        to_account: Destination account number.
        amount_gbp: Amount to transfer in GBP.

    Returns:
        Transaction result or confirmation request.
    """
    # Low-value transfers auto-approve; high-value require confirmation
    CONFIRMATION_THRESHOLD = 1000.0

    if amount_gbp >= CONFIRMATION_THRESHOLD and not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=(
                f"High-value transfer requested: £{amount_gbp:.2f} "
                f"from {from_account} → {to_account}. "
                "Provide your 2FA code to authorise."
            ),
        )
        return {
            "status": "awaiting_confirmation",
            "amount": amount_gbp,
            "requires_2fa": True,
        }

    if tool_context.tool_confirmation and not tool_context.tool_confirmation.confirmed:
        return {"status": "rejected", "message": "Transfer rejected or 2FA failed."}

    # Validate 2FA payload if present
    if tool_context.tool_confirmation and tool_context.tool_confirmation.payload:
        otp = tool_context.tool_confirmation.payload.get("otp_code")
        if otp != "123456":   # your real OTP validation here
            return {"status": "error", "message": "Invalid 2FA code."}

    # Execute transfer
    tx_id = "TX" + str(int(amount_gbp * 100))
    return {
        "status": "success",
        "transaction_id": tx_id,
        "amount_gbp": amount_gbp,
        "from": from_account,
        "to": to_account,
    }
```

### Example 3 — testing HITL flow without a real UI

```python
import asyncio
from unittest.mock import patch, AsyncMock
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.tool_context import ToolContext

async def risky_tool(action: str, tool_context: ToolContext) -> dict:
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(hint=f"Confirm action: {action}")
        return {"status": "awaiting_confirmation"}
    if not tool_context.tool_confirmation.confirmed:
        return {"status": "rejected"}
    return {"status": "executed", "action": action}

agent = LlmAgent(
    name="test_agent",
    model="gemini-2.5-flash",
    instruction="Execute the requested action after confirmation.",
    tools=[risky_tool],
)

async def test_approved():
    """Simulate a UI that auto-approves all confirmations."""
    runner = InMemoryRunner(agent=agent, app_name="test")
    await runner.session_service.create_session(
        app_name="test", user_id="u1", session_id="s1"
    )

    # First call — triggers confirmation request
    events1 = await runner.run_debug(
        "Execute action: deploy-to-production", user_id="u1", session_id="s1"
    )

    # Simulate user approval by calling again with confirmed=True
    # (In a real UI, the harness injects ToolConfirmation into the next tool call)
    # This is framework-internal; for integration tests, mock the tool_context.

asyncio.run(test_approved())
```

### `ToolConfirmation` in `ExecuteBashTool`

The bash tool (`ExecuteBashTool`) always calls `tool_context.request_confirmation()` before executing any command — it is the canonical example of confirmation-gated execution in the ADK codebase. Every command, regardless of policy, requires a confirmed `ToolConfirmation` to run. This makes it safe to give agents broad command prefixes while still keeping a human in the loop.

---

## Quick-reference: which class for which job?

| Need | Class |
|---|---|
| Tool that takes > a few seconds | `LongRunningFunctionTool` |
| Voice/audio/bidirectional streaming | `LiveRequestQueue` + `LiveRequest` |
| Use Claude models (Anthropic or Vertex) | `AnthropicLlm` / `Claude` |
| Persistent local sessions (dev / single-node) | `SqliteSessionService` |
| Let agent run shell commands safely | `ExecuteBashTool` + `BashToolPolicy` |
| Constrained multi-agent routing | `TransferToAgentTool` |
| Long-term semantic memory (GCP/Vertex) | `VertexAiMemoryBankService` |
| Iterative automatic prompt improvement | `SimplePromptOptimizer` |
| Reusable capability bundles | `SkillRegistry` + `Skill` + `SkillToolset` |
| Human-in-the-loop approval before action | `ToolConfirmation` |

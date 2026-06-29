---
title: "Memory and Artifacts"
description: "Long-term memory (InMemory, Vertex Memory Bank, Vertex RAG) and artifact services (file, GCS, in-memory)."
framework: google-adk
language: python
sidebar:
  order: 60
---

Verified against google-adk==2.3.0 (`google/adk/memory/`, `google/adk/artifacts/`).

Both memory and artifacts are **per-runner services**: you pass an instance when constructing the `Runner` (or rely on `InMemoryRunner`'s built-in in-memory pair). Memory is for searchable long-term context across sessions; artifacts are versioned file storage tied to sessions or users.

## Minimal example

```python
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.artifacts import FileArtifactService
from google.adk.tools import load_memory, load_artifacts

agent = LlmAgent(
    name="librarian",
    model="gemini-2.5-flash",
    instruction="Use load_memory for recall and load_artifacts for files.",
    tools=[load_memory, load_artifacts],
)

runner = Runner(
    app_name="demo",
    agent=agent,
    session_service=InMemorySessionService(),
    memory_service=InMemoryMemoryService(),
    artifact_service=FileArtifactService(root_dir="./artifacts"),
)
```

## Memory service landscape

`BaseMemoryService` (`memory/base_memory_service.py:44`) defines the following operations:

| Method | Purpose |
|---|---|
| `add_session_to_memory(session)` | Ingest the full session's events |
| `add_events_to_memory(*, app_name, user_id, events, session_id=None, custom_metadata=None)` | Incremental add — last turn only, for example |
| `add_memory(*, app_name, user_id, memories, custom_metadata=None)` | Write explicit `MemoryEntry` items directly |
| `search_memory(*, app_name, user_id, query)` | Semantic / keyword search — returns `SearchMemoryResponse` |

Not every service implements every method — `InMemoryMemoryService` supports search via word-set matching; Vertex services delegate to GCP APIs.

### Implementations

| Service | Storage | Search | Extra setup |
|---|---|---|---|
| `InMemoryMemoryService()` | Python dict, keyed by `(app_name, user_id)` | Case-insensitive word overlap | None |
| `VertexAiMemoryBankService(project, location, agent_engine_id, *, express_mode_api_key=None)` | GCP Agent Engine Memory Bank | Semantic (Vertex-side) | Enable Agent Engine in your GCP project; pass the engine's numeric ID (not the full resource path) |
| `VertexAiRagMemoryService(rag_corpus, similarity_top_k=None, vector_distance_threshold=10)` | Vertex AI RAG corpus | Vector similarity | Create a RAG corpus in advance; format `projects/.../ragCorpora/{id}` or just the id |

`VertexAiRagMemoryService` is imported lazily — install `google-cloud-aiplatform` to enable it (`memory/__init__.py:30-37`).

### Wiring memory into agent turns

Two built-in tools consume the memory service:

- `load_memory` — a regular `FunctionTool`. The LLM decides when to call it and passes a query string; results come back as a list of `MemoryEntry`.
- `preload_memory` — invisible to the model. Runs automatically each turn, prepends the top-k matches to the system context. Good when the model would forget to call `load_memory` on its own.

```python
from google.adk.tools import load_memory, preload_memory

agent = LlmAgent(
    name="remembering",
    model="gemini-2.5-flash",
    tools=[load_memory, preload_memory],   # both can be used together
)
```

Both tools call `memory_service.search_memory(app_name=..., user_id=..., query=...)`. Without a memory service configured, they return an empty result set (no error).

### Writing to memory

Memory is not automatically populated — you decide when to ingest a session:

```python
# End of a chat: add the whole session to memory for future recall
await runner.memory_service.add_session_to_memory(session)

# Or just the latest turn
await runner.memory_service.add_events_to_memory(
    app_name="demo",
    user_id="u1",
    events=session.events[-2:],
)

# Or explicit facts
from google.adk.memory.memory_entry import MemoryEntry
await runner.memory_service.add_memory(
    app_name="demo",
    user_id="u1",
    memories=[MemoryEntry(content="User prefers metric units.")],
)
```

`VertexAiMemoryBankService.add_events_to_memory` uses `memories.ingest_events` by default; it switches to `memories.generate` if `custom_metadata` includes Vertex-specific keys (`ttl`, `revision_ttl`, `metadata`, `wait_for_completion`). See `vertex_ai_memory_bank_service.py:229-250`.

## MemoryEntry

```python
from google.adk.memory.memory_entry import MemoryEntry

entry = MemoryEntry(
    content="User's favourite language is Rust.",
    timestamp=1_747_000_000.0,   # float seconds, optional
    custom_metadata={"source": "self-report"},
)
```

`search_memory` returns a `SearchMemoryResponse` whose `memories: list[MemoryEntry]` is what the model sees via `load_memory`.

## Artifact service landscape

`BaseArtifactService` (`artifacts/base_artifact_service.py:88`) abstracts versioned file storage. Key methods:

| Method | Purpose |
|---|---|
| `save_artifact(*, app_name, user_id, filename, artifact, session_id=None, custom_metadata=None) -> int` | Save a new version. Returns the 0-based revision id |
| `load_artifact(*, app_name, user_id, filename, session_id=None, version=None) -> types.Part` | Load latest (or specific version) |
| `list_artifact_keys(*, app_name, user_id, session_id=None) -> list[str]` | List filenames in scope |
| `delete_artifact(*, app_name, user_id, filename, session_id=None)` | Remove an artifact |
| `list_versions(*, app_name, user_id, filename, session_id=None) -> list[int]` | Version numbers only |
| `list_artifact_versions(*, app_name, user_id, filename, session_id=None) -> list[ArtifactVersion]` | Version metadata (uri, mime, timestamp) |
| `get_artifact_version(*, app_name, user_id, filename, session_id=None, version=None) -> Optional[ArtifactVersion]` | Metadata for a single version |

### Scoping rules

- `session_id=<id>` — session-scoped artifact. Lost when the session is deleted.
- `session_id=None` — user-scoped artifact. Lives across sessions.
- `filename="user:foo.pdf"` — explicit user scope even inside a session (filename prefix convention; see `file_artifact_service.py:60-85`).

### Implementations

| Service | Storage | Notes |
|---|---|---|
| `InMemoryArtifactService()` | Python dict; keeps full blobs in memory | Dev/testing |
| `FileArtifactService(root_dir)` | Local filesystem under `root_dir/` | Versions stored as `artifacts/<session>/<filename>/v<N>`; metadata JSON alongside. Thread-safe via per-file locks |
| `GcsArtifactService(bucket_name, **kwargs)` | Google Cloud Storage bucket | `kwargs` forwarded to `google.cloud.storage.Client` |

```python
from google.adk.artifacts import FileArtifactService, GcsArtifactService

# Local
local = FileArtifactService(root_dir="/var/adk/artifacts")

# GCS (defaults to ADC credentials)
gcs = GcsArtifactService(bucket_name="my-adk-artifacts")

# GCS with explicit project
gcs_proj = GcsArtifactService(bucket_name="my-adk-artifacts", project="my-gcp-project")

# GCS with service account key file
from google.oauth2 import service_account
sa_creds = service_account.Credentials.from_service_account_file(
    "sa.json",
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
gcs_sa = GcsArtifactService("my-adk-artifacts", credentials=sa_creds)
```

**GCS blob path structure** (verified `gcs_artifact_service.py:_get_blob_name`):

| Scope | How to trigger | Blob path |
|---|---|---|
| Session-scoped | `session_id=<id>`, filename without `user:` prefix | `{app_name}/{user_id}/{session_id}/{filename}/{version}` |
| User-scoped | `filename="user:foo"` prefix — `session_id` is ignored | `{app_name}/{user_id}/user/{filename}/{version}` |

> **GCS note:** `GcsArtifactService` raises `InputValidationError` when `session_id=None` and the filename does not start with `"user:"`. Use the `"user:"` filename prefix (not `session_id=None` alone) for cross-session user storage.

Version numbers start at **0** and increment by 1 on each save.

**GCS IAM requirements:** `storage.objects.create` (save), `storage.objects.get` (load), `storage.objects.list` (list/versions), `storage.objects.delete` (delete). The `roles/storage.objectAdmin` role covers all four.

> For more detailed `GcsArtifactService` examples — binary artifacts, user-scoped files, listing version metadata — see [Class deep dives — vol. 2](./google_adk_class_deep_dives_v2/#4--gcsartifactservice).

### Saving and loading from tools

```python
from google.genai import types
from google.adk.tools import FunctionTool

async def make_report(topic: str, tool_context) -> dict:
    pdf_bytes = render_pdf(topic)
    part = types.Part(inline_data=types.Blob(mime_type="application/pdf", data=pdf_bytes))
    version = await tool_context.save_artifact(filename=f"{topic}.pdf", artifact=part)
    return {"saved": True, "version": version}

report_tool = FunctionTool(func=make_report)
```

Inside a callback/tool:

- `tool_context.save_artifact(filename, artifact, *, custom_metadata=None)` — returns the new version int.
- `tool_context.load_artifact(filename, *, version=None)` — returns `types.Part` or `None`.
- `tool_context.list_artifacts()` — returns a list of filenames in scope.
- `tool_context.get_artifact_version(filename, version=None)` — returns metadata only.

### The `load_artifacts` tool

`load_artifacts` is a singleton `FunctionTool` the model can call to fetch an artifact by name and have its content injected as a `types.Part`. Include it in `tools=[load_artifacts]` when the agent should reference past files.

## ArtifactVersion

```python
class ArtifactVersion(BaseModel):
    version: int
    canonical_uri: str
    custom_metadata: dict
    create_time: float        # unix seconds
    mime_type: Optional[str]
```

`canonical_uri` is the back-end-specific reference (file path, `gs://...`, in-memory key). Use `list_artifact_versions` to get metadata without downloading blobs.

## Patterns

### 1 — End-of-chat memory ingest
Append a plugin that overrides `after_run_callback` and calls `memory_service.add_session_to_memory(session)`. All future chats for the same user gain recall.

### 2 — Selective memory
Use `custom_metadata={"ttl": ...}` on `add_events_to_memory` with `VertexAiMemoryBankService` to auto-expire short-lived memories (e.g. session-specific preferences that shouldn't leak to future users).

### 3 — User-scoped long-term file cache
Save with `session_id=None` (or `filename="user:history.json"`). The artifact survives session deletion and is available to every future session of the same user.

### 4 — Versioned reports
Each run saves a new version of `report.pdf`. The UI lists `list_artifact_versions(...)` with timestamps so a reviewer can diff outputs turn-by-turn.

### 5 — RAG corpus-backed memory
`VertexAiRagMemoryService(rag_corpus="...")` plus `load_memory` in `tools=`. The corpus is updated by a separate ingestion job (files, web pages, BigQuery). Agents retrieve only — they never mutate the corpus.

## Complete examples

### Example A — cross-session memory recall

This shows the full lifecycle: chat, ingest the session, start a new session, and recall what was said.

```python
import asyncio
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.apps import App
from google.adk.tools import load_memory

APP_NAME = "memory_demo"
USER_ID = "alice"

agent = LlmAgent(
    name="assistant",
    model="gemini-2.0-flash",
    instruction=(
        "You are a helpful assistant. Use load_memory to recall "
        "things the user told you in previous conversations."
    ),
    tools=[load_memory],
)

async def main():
    session_service = InMemorySessionService()
    memory_service = InMemoryMemoryService()
    app = App(name=APP_NAME, root_agent=agent)
    runner = Runner(app=app, session_service=session_service, memory_service=memory_service)

    # --- Session 1: tell the agent a fact ---
    s1 = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    async for event in runner.run_async(
        user_id=USER_ID, session_id=s1.id,
        new_message=types.Content(role="user", parts=[types.Part(text="My favourite programming language is Rust.")]),
    ):
        if event.is_final_response():
            print("Session 1:", event.content.parts[0].text)

    # Ingest the session into memory so future sessions can recall it
    session_1 = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=s1.id
    )
    await memory_service.add_session_to_memory(session_1)

    # --- Session 2: recall the fact ---
    s2 = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    async for event in runner.run_async(
        user_id=USER_ID, session_id=s2.id,
        new_message=types.Content(role="user", parts=[types.Part(text="What is my favourite programming language?")]),
    ):
        if event.is_final_response():
            print("Session 2:", event.content.parts[0].text)
            # → "Your favourite programming language is Rust."

asyncio.run(main())
```

### Example B — saving and retrieving artifacts across an agent turn

```python
import asyncio
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import FileArtifactService
from google.adk.apps import App
from google.adk.tools import FunctionTool, load_artifacts
from google.adk.tools.tool_context import ToolContext

APP_NAME = "artifact_demo"
USER_ID = "bob"

async def generate_report(topic: str, tool_context: ToolContext) -> dict:
    """Generates a text report and saves it as an artifact."""
    content = f"# Report: {topic}\n\nThis is an auto-generated report about {topic}."
    part = types.Part(
        inline_data=types.Blob(
            mime_type="text/plain",
            data=content.encode(),
        )
    )
    version = await tool_context.save_artifact(
        filename=f"{topic.lower().replace(' ', '_')}_report.txt",
        artifact=part,
    )
    return {"saved": True, "filename": f"{topic.lower().replace(' ', '_')}_report.txt", "version": version}

report_tool = FunctionTool(func=generate_report)

agent = LlmAgent(
    name="report_agent",
    model="gemini-2.0-flash",
    instruction=(
        "Generate reports when asked. Use generate_report to save them. "
        "Use load_artifacts to retrieve them when asked."
    ),
    tools=[report_tool, load_artifacts],
)

async def main():
    session_service = InMemorySessionService()
    artifact_service = FileArtifactService(root_dir="/tmp/adk_artifacts")
    app = App(name=APP_NAME, root_agent=agent)
    runner = Runner(app=app, session_service=session_service, artifact_service=artifact_service)
    session = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID
    )

    # Turn 1: generate and save a report
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="Generate a report about climate change.")]),
    ):
        if event.is_final_response():
            print("Turn 1:", event.content.parts[0].text)

    # Turn 2: retrieve the report by asking the agent to load it
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="Show me the climate change report you generated.")]),
    ):
        if event.is_final_response():
            print("Turn 2:", event.content.parts[0].text)

asyncio.run(main())
```

### Example C — VertexAiMemoryBankService for semantic recall

```python
import asyncio
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import VertexAiMemoryBankService
from google.adk.apps import App
from google.adk.tools import load_memory, preload_memory

# Replace with your GCP project, location, and Agent Engine ID (numeric string)
PROJECT = "my-gcp-project"
LOCATION = "us-central1"
AGENT_ENGINE_ID = "1234567890"   # Numeric ID, not the full resource path
APP_NAME = "vertex_memory_demo"
USER_ID = "charlie"

agent = LlmAgent(
    name="vertex_assistant",
    model="gemini-2.0-flash",
    instruction=(
        "You are a helpful assistant. Memories from past conversations "
        "are automatically available to you via preload_memory. "
        "Use load_memory to search for specific information."
    ),
    # preload_memory runs automatically every turn — model sees memories without calling a tool
    # load_memory lets the model explicitly query memories when needed
    tools=[preload_memory, load_memory],
)

async def main():
    memory_service = VertexAiMemoryBankService(
        project=PROJECT,
        location=LOCATION,
        agent_engine_id=AGENT_ENGINE_ID,
        # Optional: express_mode_api_key="..." for express mode
    )
    session_service = InMemorySessionService()
    app = App(name=APP_NAME, root_agent=agent)
    runner = Runner(app=app, session_service=session_service, memory_service=memory_service)

    # Session 1 — establish user preferences
    s1 = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    async for event in runner.run_async(
        user_id=USER_ID, session_id=s1.id,
        new_message=types.Content(role="user", parts=[types.Part(text="I'm a vegetarian and allergic to nuts. Keep this in mind.")]),
    ):
        if event.is_final_response():
            print("S1:", event.content.parts[0].text)

    # Ingest into Vertex AI Memory Bank for semantic retrieval
    s1_obj = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=s1.id
    )
    await memory_service.add_session_to_memory(s1_obj)

    # Session 2 — the agent recalls dietary requirements via semantic search
    s2 = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    async for event in runner.run_async(
        user_id=USER_ID, session_id=s2.id,
        new_message=types.Content(role="user", parts=[types.Part(text="Suggest a recipe for dinner tonight.")]),
    ):
        if event.is_final_response():
            # Agent should suggest a vegetarian, nut-free recipe
            print("S2:", event.content.parts[0].text)

asyncio.run(main())
```

### Example D — versioned user-scoped GCS artifacts

```python
import asyncio
from google.genai import types
from google.adk.artifacts import GcsArtifactService
from google.adk.sessions import InMemorySessionService

APP_NAME = "gcs_demo"
USER_ID = "diana"

async def demo_gcs_artifacts():
    artifact_service = GcsArtifactService(bucket_name="my-adk-artifacts")
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)

    # Save version 0 — user-scoped (survives session deletion)
    # Use the "user:" prefix to scope to user rather than session
    v0 = await artifact_service.save_artifact(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session.id,
        filename="user:preferences.json",
        artifact=types.Part(
            inline_data=types.Blob(
                mime_type="application/json",
                data=b'{"theme": "dark", "language": "en"}',
            )
        ),
    )
    print(f"Saved version {v0}")  # → 0

    # Save version 1 — updated preferences
    v1 = await artifact_service.save_artifact(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session.id,
        filename="user:preferences.json",
        artifact=types.Part(
            inline_data=types.Blob(
                mime_type="application/json",
                data=b'{"theme": "light", "language": "fr"}',
            )
        ),
    )
    print(f"Saved version {v1}")  # → 1

    # List all versions
    versions = await artifact_service.list_artifact_versions(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session.id,
        filename="user:preferences.json",
    )
    for v in versions:
        print(f"Version {v.version}: {v.canonical_uri} at {v.create_time}")

    # Load a specific version
    old_prefs = await artifact_service.load_artifact(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session.id,
        filename="user:preferences.json",
        version=0,
    )
    print("Old prefs:", old_prefs.inline_data.data)

asyncio.run(demo_gcs_artifacts())
```

## Gotchas

- The memory tools (`load_memory`, `preload_memory`) silently no-op when no `memory_service` is configured on the runner. Wire one explicitly, or you'll never see memories.
- `VertexAiMemoryBankService` requires the **numeric** `agent_engine_id` (`"456"`), not the full resource path. The constructor warns if it detects a `/`.
- `SaveFilesAsArtifactsPlugin` replaced the deprecated `RunConfig.save_input_blobs_as_artifacts`. Install the plugin on the `App` instead of toggling the flag.
- Artifact `save_artifact` accepts a `types.Part` OR a plain dict (camelCase or snake_case); `ensure_part` normalises via Pydantic validation (`base_artifact_service.py:68-85`).
- `FileArtifactService` creates the root directory lazily on first save; make sure the process has write permissions.
- `GcsArtifactService` uses Application Default Credentials by default — on GKE/Cloud Run make sure the service account has `storage.objects.create` and `storage.objects.get`.
- An artifact's `version` starts at **0**, not 1 (`base_artifact_service.py:122`).

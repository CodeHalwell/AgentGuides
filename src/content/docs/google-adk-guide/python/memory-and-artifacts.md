---
title: "Memory and Artifacts"
description: "Long-term memory (InMemory, Vertex Memory Bank, Vertex RAG) and artifact services (file, GCS, in-memory)."
framework: google-adk
language: python
sidebar:
  order: 60
---

Verified against google-adk==2.0.0 (`google/adk/memory/`, `google/adk/artifacts/`).

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
```

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

## Gotchas

- The memory tools (`load_memory`, `preload_memory`) silently no-op when no `memory_service` is configured on the runner. Wire one explicitly, or you'll never see memories.
- `VertexAiMemoryBankService` requires the **numeric** `agent_engine_id` (`"456"`), not the full resource path. The constructor warns if it detects a `/`.
- `SaveFilesAsArtifactsPlugin` replaced the deprecated `RunConfig.save_input_blobs_as_artifacts`. Install the plugin on the `App` instead of toggling the flag.
- Artifact `save_artifact` accepts a `types.Part` OR a plain dict (camelCase or snake_case); `ensure_part` normalises via Pydantic validation (`base_artifact_service.py:68-85`).
- `FileArtifactService` creates the root directory lazily on first save; make sure the process has write permissions.
- `GcsArtifactService` uses Application Default Credentials by default — on GKE/Cloud Run make sure the service account has `storage.objects.create` and `storage.objects.get`.
- An artifact's `version` starts at **0**, not 1 (`base_artifact_service.py:122`).

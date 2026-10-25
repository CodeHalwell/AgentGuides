---
title: "Unified Memory"
description: "Standalone Memory class — remember/recall/forget, scope and slice views, LanceDB vs Qdrant Edge storage, and how crew-level memory wires in."
framework: "crewai"
language: "python"
sidebar:
  label: "Memory"
  order: 22
---

> **Verified against `crewai` %%version=crewai%%** (checked %%versions-checked%%; source: `crewai/memory/unified_memory.py`, `crewai/memory/memory_scope.py`, `crewai/memory/storage/*.py`).

CrewAI's memory system centres on a single class — `Memory` — that works either attached to a crew or on its own. Older docs mention "short-term", "long-term", and "entity" memory; in 1.14 those collapse into scoped views over the unified store.

## Minimal runnable example

```python
from crewai import Memory

mem = Memory()  # LanceDB under $CREWAI_STORAGE_DIR/memory, else the platform data dir

mem.remember(
    "Ada prefers Postgres over MongoDB for transactional workloads.",
    scope="/users/ada",
    categories=["preferences", "db"],
    importance=0.8,
)

matches = mem.recall("what database does ada like?", limit=3)
for m in matches:
    print(f"{m.score:.2f}  {m.record.content}")
```

`recall()` is `depth="deep"` by default, which can call the analysis LLM. Pass `depth="shallow"` for a single embedding lookup with no LLM call.

### Running without OpenAI

`embedder` also accepts any callable that takes a list of strings and returns one vector per string. That makes `Memory` usable offline, in tests, or with an embedding service CrewAI has no provider for. Supplying `scope`, `categories` and `importance` yourself and recalling with `depth="shallow"` means no LLM is involved at all:

```python
import hashlib
import re

from crewai import Memory


def embed(texts: list[str]) -> list[list[float]]:
    """Toy bag-of-words embedder. Replace it with a call to a real embedding model."""
    vectors = []
    for text in texts:
        vec = [0.0] * 64
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        vectors.append([x / norm for x in vec])
    return vectors


mem = Memory(storage="./demo-memory", embedder=embed)   # a path string becomes a LanceDB directory
mem.remember("Ada prefers Postgres for transactional workloads.", scope="/users/ada",
             categories=["preferences"], importance=0.8)
mem.remember("Ada works remotely from Toronto.", scope="/users/ada",
             categories=["bio"], importance=0.5)

for m in mem.recall("which database does Ada prefer", scope="/users/ada", depth="shallow", limit=2):
    print(f"{m.score:.2f}  {m.record.content}")
print(mem.tree())          # the scopes and record counts under "/"
mem.close()                # waits for background saves and releases the store
```

The analysis model is only needed when you leave `scope`, `categories` or `importance` unset on `remember()`, or use deep recall.

The default configuration picks up `OPENAI_API_KEY` for both analysis (`gpt-5.4-mini`) and embedding (`text-embedding-3-large`, 3072 dimensions). Both are pluggable — see below.

> **Upgrading an old local store?** The default embedder used to be `text-embedding-3-small` (1536 dimensions). A LanceDB store written with the old default raises `EmbeddingDimensionMismatchError` under the new one. Either run `crewai reset-memories --memory`, or keep the old vectors by pinning `embedder={"provider": "openai", "config": {"model_name": "text-embedding-3-small"}}`. The key is `model_name`; `model` raises `TypeError`.

## Constructor fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `llm` | `str \| BaseLLM` | `"gpt-5.4-mini"` | Analysis model: infers scope, categories and importance on save, extracts memories from text, and runs the deep-recall steps. Built lazily, so it is only needed when one of those happens. |
| `storage` | `StorageBackend \| str` | `"lancedb"` | `"lancedb"`, `"qdrant-edge"`, a path string (becomes a LanceDB path), or a custom backend. |
| `embedder` | `callable \| dict \| None` | `None` | Embedder callable, provider-config dict (e.g. `{"provider": "google", "config": {...}}`), or `None` for default OpenAI. |
| `recency_weight` | `float` | `0.3` | Composite score weighting. Must sum with the other two weights to ~1.0. |
| `semantic_weight` | `float` | `0.5` | |
| `importance_weight` | `float` | `0.2` | |
| `recency_half_life_days` | `int` | `30` | Recency score halves every N days. |
| `consolidation_threshold` | `float` | `0.85` | Similarity above which the LLM tries to merge near-duplicates on save. |
| `consolidation_limit` | `int` | `5` | Max candidates compared during consolidation. |
| `default_importance` | `float` | `0.5` | Used when `importance=None` and the LLM can't infer one. |
| `confidence_threshold_high` | `float` | `0.8` | Deep-recall early-exit threshold. |
| `confidence_threshold_low` | `float` | `0.5` | Below this, deep-recall spawns another round. |
| `complex_query_threshold` | `float` | `0.7` | For queries the analyser rates "complex", explore further when confidence is below this. |
| `query_analysis_threshold` | `int` | `200` | Queries shorter than this many characters skip LLM query analysis during deep recall. |
| `exploration_budget` | `int` | `1` | Max LLM-driven deep-recall rounds. |
| `read_only` | `bool` | `False` | If `True`, `remember()` returns `None` and `remember_many()` returns `[]` without storing anything; `update()` returns the record unchanged; `recall()` does not refresh access times. `forget()` and `reset()` still delete. |
| `root_scope` | `str \| None` | `None` | All operations are implicitly nested under this path. |

## Storage backends

| Backend | Value | Install | Notes |
|---|---|---|---|
| LanceDB (default) | `"lancedb"` | included | File-backed, in-process. Default location is `$CREWAI_STORAGE_DIR/memory` if set, otherwise the per-project data directory. |
| LanceDB at a path | `"./some/dir"` | included | Any string without the special markers becomes a LanceDB path. |
| Qdrant Edge | `"qdrant-edge"` | `pip install "crewai[qdrant-edge]"` | In-process Qdrant Edge (`qdrant-edge-py`) with payload indexing, better filtering. |
| Custom | `MyBackend()` | — | Any class implementing the `StorageBackend` protocol from `crewai.memory.storage.backend`. To make every `Memory` in a process use one, register it with `crewai.memory.storage.factory.set_memory_storage_factory(...)` at start-up. |

```python
from crewai import Memory
from crewai.memory.storage.lancedb_storage import LanceDBStorage

mem = Memory(storage=LanceDBStorage(path="./prod-memory"))
```

## Core operations

### `remember`

```python
record = mem.remember(
    "Ada prefers Postgres.",
    scope="/users/ada",              # optional; LLM infers if None
    categories=["preferences"],       # optional; LLM infers
    metadata={"source": "slack"},
    importance=0.8,                  # 0-1; LLM infers if None
    source="slack-msg-1234",         # provenance; used for private-record filtering
    private=False,
    root_scope=None,                 # per-call override of instance-level root_scope
)
```

- Returns the saved `MemoryRecord` (or `None` on a read-only memory).
- Synchronous: it waits for the save, which runs on the memory's single-worker thread pool.
- Fields you leave out (`scope`, `categories`, `importance`) are filled in by the analysis LLM. Fields you pass are used as given.
- Near-duplicates (similarity above `consolidation_threshold`) can be merged by the LLM instead of stored twice.

### `remember_many`

```python
mem.remember_many([
    "Ada is a backend engineer.",
    "Ada works remote from Toronto.",
], scope="/users/ada", categories=["bio"])
```

- **Fires-and-forgets** — returns an empty list immediately. The save runs in the background.
- The next `recall()` waits for pending saves (read barrier).

### `recall`

```python
matches = mem.recall(
    "tell me about ada",
    scope="/users/ada",      # optional prefix filter
    categories=["bio"],       # optional filter
    limit=10,
    depth="deep",            # or "shallow"
    source="slack-msg-1234", # only if you store private records
    include_private=False,
)
```

- `depth="shallow"` — single embed + vector search. Fast, no LLM calls.
- `depth="deep"` (default) — runs a small recall flow. Queries of 200 characters or more (`query_analysis_threshold`) are first distilled by the LLM into up to three sub-queries and candidate scopes; shorter queries skip that step and are embedded as they are. If the best score is below `confidence_threshold_low` (0.5) and `exploration_budget` allows, one more LLM round extracts and checks the top excerpts. Complex long queries explore deeper when confidence is under `complex_query_threshold`. So a short query with a confident match costs no LLM call even at `"deep"`.
- Results are ranked by a composite score = `semantic_weight * similarity + recency_weight * recency + importance_weight * importance`.
- `recall()` first waits for pending background saves, so it always sees what `remember_many()` queued.

### `forget` and `update`

```python
# Delete one user's old records
from datetime import datetime, timedelta

cutoff = datetime.utcnow() - timedelta(days=365)
old_ids = [r.id for r in mem.list_records(scope="/users/ada", limit=1000) if r.created_at < cutoff]
# Never call forget(record_ids=[]): an empty list is ignored and the call deletes everything (see below).
deleted = mem.forget(record_ids=old_ids) if old_ids else 0

# Edit one record (re-embeds the new content)
mem.update(record_id, content="Ada now prefers DuckDB for analytics.", importance=0.9)
```

`forget()` takes `scope`, `categories`, `older_than`, `metadata_filter` and `record_ids` and returns the number of records deleted. `update()` also accepts `scope`, `categories` and `metadata`, returns the updated record, and raises `ValueError` if the id does not exist.

> **`older_than` with only a scope is unsafe in 1.15.28.** On the default LanceDB backend, `mem.forget(scope="/users/ada", older_than=cutoff)` deleted records created seconds earlier in our test, because the scope and date conditions are combined without parentheses (`scope LIKE ... OR scope = '/' AND created_at < ...`) and the scope clause alone matches. `forget(older_than=cutoff)` with no scope, `forget(scope=..., categories=[...], older_than=cutoff)` and `forget(record_ids=[...])` behaved correctly. The example above selects ids in Python for that reason.
>
> **An empty `record_ids` list deletes everything.** The storage layer only filters by id when the list is non-empty; `forget(record_ids=[])` with no other criteria falls through to `delete("id != ''")` and removes every record (or, with `root_scope` set, everything under it). Always check the list first, as the example does. Try your own call against a scratch store before relying on it.

## Scope and slice views

Most apps want a bounded view rather than the whole memory:

```python
# Scope: everything under /projects/phoenix
phoenix = mem.scope("/projects/phoenix")
phoenix.remember("Design doc frozen on 2026-04-01.")

# Slice: read-only view across multiple scopes
shared = mem.slice(
    scopes=["/users/ada", "/teams/platform"],
    categories=["decisions"],
    read_only=True,
)
matches = shared.recall("who owns auth?")
```

- `scope(path)` is a two-way view: it can read and write under `path`. Its `remember()` takes a scope relative to that root and defaults to the root itself. It also has `remember_many`, `recall`, `forget`, `list_scopes`, `info`, `tree`, `list_categories`, `reset` and `subscope(path)`.
- `slice(scopes=[...])` is read-only by default and can span many paths. A read-only slice's `remember()` returns `None`. A writable slice (`read_only=False`) needs an explicit `scope=` argument on `remember()`. A slice has `recall`, `list_scopes`, `info` and `list_categories`, but no `forget`.
- `Memory` itself also offers inspection helpers: `list_scopes(path)`, `list_records(scope, limit, offset)`, `info(path)`, `tree(path, max_depth)`, `list_categories(path)`, `reset(scope)` and `reset_all()`.

## Attaching memory to a crew

```python
from crewai import Crew, Memory

mem = Memory(root_scope="/crew/research")

crew = Crew(
    agents=[a, b],
    tasks=[t1, t2],
    memory=mem,            # or memory=True for a default Memory()
)
```

- `memory=True` builds a `Memory` for you with root scope `/crew/<crew name>` (`/crew/crew` for an unnamed crew), uses the crew's `embedder` if set, and takes its analysis LLM from `chat_llm` or, failing that, the first agent's `llm`. The embeddings still default to OpenAI.
- `memory=mem` uses your instance as it is, with no automatic root scope. Set `root_scope` yourself to keep crews apart.
- `memory=mem.scope("/some/path")` or `memory=mem.slice([...])` attaches a view.
- An agent can have its own `Agent(memory=...)`; `Agent(memory=True)` builds a `Memory` that uses the agent's LLM for analysis. Without one, the agent falls back to the crew's memory.

What a task run actually does with memory (we traced the prompts of a crew with a scripted model):

1. **Before the task**, the agent calls `recall(task.description, limit=5)` (deep) and appends the matches to the task prompt under "Memories from past conversations", with a reminder that the selection may be incomplete.
2. **During the task**, the agent has two extra tools, `search_memory` (a `RecallMemoryTool`) and `save_to_memory` (a `RememberTool`). A read-only memory gets only the search tool.
3. **After the task**, the task description, expected output and result are passed to the analysis LLM, which extracts short memory statements; they are saved with `remember_many`. When the memory has a `root_scope`, they go under `<root_scope>/agent/<agent role>`; otherwise the LLM chooses the scope. Delegation steps are not saved.
4. The crew drains pending saves before `kickoff()` returns.

That is three kinds of LLM call per task (query analysis for long task descriptions, extraction, and scope/category analysis on save) on top of your agents' own calls. Budget for them, or pass a cheaper `llm=` to `Memory`.

## Patterns

### 1. Per-user memory slice

```python
shared = Memory(storage="lancedb")
ada_view = shared.scope("/users/ada")

ada_crew = Crew(agents=[...], tasks=[...], memory=ada_view)
```

Each user gets an isolated prefix without needing a separate DB.

### 2. Import/export for dev vs prod

```python
prod = Memory(storage=LanceDBStorage(path="/mnt/memory/prod"))
dev  = Memory(storage=LanceDBStorage(path="./dev-memory"))

for rec in prod.list_records(limit=5000):
    dev.remember(rec.content, scope=rec.scope, categories=rec.categories,
                 metadata=rec.metadata, importance=rec.importance)
```

### 3. Memory as a tool for an agent

```python
from crewai.tools.memory_tools import create_memory_tools

tools = create_memory_tools(mem)
agent = Agent(role="Analyst", goal="...", backstory="...", tools=tools)
```

`create_memory_tools` returns a `RecallMemoryTool` and, unless the memory is read-only, a `RememberTool`, wrapping `recall` / `remember` as agent-callable tools. Agents on a crew with memory already get these automatically; you only build them yourself for an agent that is outside such a crew or when you want a different `Memory` than the crew's.

### 4. Read-only prod, writable staging

```python
prod_view = Memory(storage=prod_storage, read_only=True)
```

Safer to pass to untrusted agents. `remember()` returns `None`, nothing is stored, and agents are given only the search tool. `forget()` and `reset()` still delete, so do not hand those to untrusted code.

### 5. Custom embedder

```python
from crewai.rag.embeddings.types import EmbedderConfig

mem = Memory(embedder={"provider": "google-generativeai", "config": {"model_name": "gemini-embedding-001"}})
```

Any provider config that `build_embedder` accepts works here; provider SDKs are not installed by default (this one needs `google-generativeai`).

## Gotchas

- **`depth="deep"` costs LLM calls.** For fast one-off lookups, pass `depth="shallow"` — it skips query analysis entirely.
- **`remember_many` is async.** It returns `[]` immediately. If you need the records back, call `remember` in a loop, or call `drain_writes()` and then `list_records()`. `close()` drains and shuts the store down.
- **Three weights should sum to ~1.0.** They're not normalised automatically; wildly off numbers make the ranking useless.
- **Default embedder needs OpenAI.** `Memory()` with no arguments uses OpenAI for the analysis LLM and the embedder, both created on first use. Pass `llm=` and `embedder=` (a provider dict or a callable) to avoid the dependency.
- **Private records** only show up in `recall` when `source=` matches the record's source, unless `include_private=True`.
- **Concurrent writers need a shared lock.** LanceDB writes are guarded by a file lock in the system temp directory, which covers processes on one machine, or by a Redis lock when `REDIS_URL` is set (`crewai_core.lock_store`). For several hosts writing one store, set `REDIS_URL` or use a custom `StorageBackend` built for concurrent writers.
- **`entity memory` is not a separate class.** The legacy `EntityMemory` was folded into scoped views — use `categories=["entity:<name>"]` or distinct scopes.

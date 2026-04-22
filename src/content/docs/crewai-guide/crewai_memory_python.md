---
title: "Unified Memory"
description: "Standalone Memory class — remember/recall/forget, scope and slice views, LanceDB vs Qdrant Edge storage, and how crew-level memory wires in."
framework: crewai
language: python
sidebar:
  label: "Memory"
  order: 22
---

> **Verified against crewai==1.14.3a2** (source: `crewai/memory/unified_memory.py`, `crewai/memory/memory_scope.py`, `crewai/memory/storage/*.py`).

CrewAI's memory system centres on a single class — `Memory` — that works either attached to a crew or on its own. Older docs mention "short-term", "long-term", and "entity" memory; in 1.14 those collapse into scoped views over the unified store.

## Minimal runnable example

```python
from crewai import Memory

mem = Memory()  # lancedb under ./memory by default

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

The default configuration picks up `OPENAI_API_KEY` for both analysis (`gpt-4o-mini`) and embedding (`text-embedding-3-small`). Both are pluggable — see below.

## Constructor fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `llm` | `str \| BaseLLM` | `"gpt-4o-mini"` | Used for query analysis and consolidation (deep-recall flow only). |
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
| `exploration_budget` | `int` | `1` | Max LLM-driven deep-recall rounds. |
| `read_only` | `bool` | `False` | If `True`, `remember()` is a no-op and returns `None`. |
| `root_scope` | `str \| None` | `None` | All operations are implicitly nested under this path. |

## Storage backends

| Backend | Value | Install | Notes |
|---|---|---|---|
| LanceDB (default) | `"lancedb"` | included | File-backed; fast for up to ~100k records. |
| LanceDB at a path | `"./some/dir"` | included | Any string without the special markers becomes a LanceDB path. |
| Qdrant Edge | `"qdrant-edge"` | `pip install qdrant-client` | In-process Qdrant with payload indexing, better filtering. |
| Custom | `MyBackend()` | — | Any class implementing the `StorageBackend` protocol from `crewai.memory.storage.backend`. |

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

- Returns the saved `MemoryRecord`.
- Synchronous; the save goes through the single-worker thread pool.
- Triggers consolidation: if a very similar record exists the LLM may merge them.

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
- `depth="deep"` (default) — the LLM rewrites the query into sub-queries, selects scopes, and iterates using `exploration_budget`.
- Results are ranked by a composite score = `semantic_weight * similarity + recency_weight * recency + importance_weight * importance`.

### `forget` and `update`

```python
# Delete everything older than a cutoff for a user
from datetime import datetime, timedelta
deleted = mem.forget(
    scope="/users/ada",
    older_than=datetime.utcnow() - timedelta(days=365),
)

# Edit one record
mem.update(record_id, content="Ada now prefers DuckDB for analytics.", importance=0.9)
```

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

- `scope(path)` is a two-way view — can read and write under `path`.
- `slice(scopes=[...])` is read-only by default and can span many paths.

## Attaching memory to a crew

```python
from crewai import Crew, Memory

mem = Memory(root_scope="/crew/research")

crew = Crew(
    agents=[a, b],
    tasks=[t1, t2],
    memory=mem,            # or memory=True for a fresh Memory() with defaults
)
```

- `memory=True` — CrewAI spins up a default `Memory()`; fine for exploration but ties you to `OPENAI_API_KEY`.
- `memory=mem` — full control; set `root_scope` to avoid cross-crew leakage.
- `memory=mem.scope("/some/path")` or `memory=mem.slice([...])` — scoped view.

During kickoff the agent saves observations with `remember_many` and recalls with `recall(..., depth="deep")`. The crew calls `drain_writes()` before returning so every save has finished persisting.

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
from crewai.tools.memory_tools import memory_tools_for

tools = memory_tools_for(memory=mem)
agent = Agent(role="Analyst", goal="...", backstory="...", tools=tools)
```

`crewai.tools.memory_tools` wraps `remember` / `recall` as agent-callable tools.

### 4. Read-only prod, writable staging

```python
prod_view = Memory(storage=prod_storage, read_only=True)
```

Safer to pass to untrusted agents; writes silently succeed as no-ops so you can still attach to a crew.

### 5. Custom embedder

```python
from crewai.rag.embeddings.types import EmbedderConfig

mem = Memory(embedder={"provider": "google", "config": {"model": "text-embedding-004"}})
```

Any provider config that `build_embedder` accepts works here.

## Gotchas

- **`depth="deep"` costs LLM calls.** For fast one-off lookups, pass `depth="shallow"` — it skips query analysis entirely.
- **`remember_many` is async.** If you need the records back, call `remember` in a loop or call `drain_writes()` first.
- **Three weights should sum to ~1.0.** They're not normalised automatically; wildly off numbers make the ranking useless.
- **Default embedder needs OpenAI.** `Memory()` with no args tries to reach OpenAI for both the LLM and embedder. Pass your own to avoid the dependency.
- **Private records** only show up in `recall` when `source=` matches the record's source, unless `include_private=True`.
- **Two processes writing the same LanceDB path will corrupt it.** Use Qdrant Edge or a remote store if you need multi-process writes.
- **`entity memory` is not a separate class.** The legacy `EntityMemory` was folded into scoped views — use `categories=["entity:<name>"]` or distinct scopes.

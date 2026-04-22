---
title: "Knowledge sources"
description: "Every built-in BaseKnowledgeSource — PDF, CSV, JSON, Excel, Docling, string, text file — plus custom sources, embedder config, and crew vs agent attachment."
framework: crewai
language: python
sidebar:
  label: "Knowledge"
  order: 23
---

> **Verified against crewai==1.14.3a2** (source: `crewai/knowledge/knowledge.py`, `crewai/knowledge/source/*.py`, `crewai/knowledge/knowledge_config.py`).

Knowledge is a read-mostly vector store attached to an `Agent` or `Crew`. Unlike [Memory](./crewai_memory_python/) it's populated ahead of time and isn't automatically written to during a run.

## Minimal runnable example

```python
from crewai import Agent, Crew, Task
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

faq = StringKnowledgeSource(
    content=(
        "Returns: accepted within 30 days, original packaging only.\n"
        "Shipping: free over $50 to the continental US.\n"
    ),
)

support = Agent(
    role="Customer Support",
    goal="Answer customer questions using company policy",
    backstory="Careful agent who quotes policy verbatim.",
    knowledge_sources=[faq],
)

task = Task(
    description="Customer asks: {question}",
    expected_output="Short answer citing policy.",
    agent=support,
)

out = Crew(agents=[support], tasks=[task]).kickoff(
    inputs={"question": "Can I return an item after 6 weeks?"}
)
print(out.raw)
```

## Built-in sources

All live under `crewai.knowledge.source.*`. They subclass `BaseKnowledgeSource` (or `BaseFileKnowledgeSource` for file-backed ones).

| Source | Import | Takes | Notes |
|---|---|---|---|
| **String** | `StringKnowledgeSource` | `content: str` | In-memory text. |
| **Text file** | `TextFileKnowledgeSource` | `file_paths=[...]` | Plain `.txt`, `.md`, etc. |
| **PDF** | `PDFKnowledgeSource` | `file_paths=[...]` | Needs `pdfplumber` (`pip install pdfplumber`). |
| **CSV** | `CSVKnowledgeSource` | `file_paths=[...]` | Concatenates rows as text. |
| **JSON** | `JSONKnowledgeSource` | `file_paths=[...]` | Recursively flattens to text. |
| **Excel** | `ExcelKnowledgeSource` | `file_paths=[...]` | Reads each sheet; needs `openpyxl`. |
| **Docling** | `CrewDoclingSource` | `file_paths=[...]` or URLs | Universal — PDF/DOCX/HTML/Markdown/AsciiDoc/images. Needs `docling`. |

Common fields on every source:

| Field | Type | Default | Notes |
|---|---|---|---|
| `chunk_size` | `int` | `4000` | Character size per chunk. |
| `chunk_overlap` | `int` | `200` | Overlap between chunks. |
| `metadata` | `dict` | `{}` | Currently unused but preserved on the source. |
| `collection_name` | `str \| None` | `None` | Explicit collection name. Defaults to the agent `role`. |

File sources additionally accept `file_paths=[...]`. The singular `file_path` still works but is deprecated.

## String source

```python
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

policy = StringKnowledgeSource(content=open("policy.txt").read(), chunk_size=1200, chunk_overlap=100)
```

## PDF / Text / CSV / JSON

```python
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

pdfs  = PDFKnowledgeSource(file_paths=["contracts/2024.pdf", "contracts/2025.pdf"])
csvs  = CSVKnowledgeSource(file_paths=["sales/q1.csv"])
jsons = JSONKnowledgeSource(file_paths=["config/routes.json"])
texts = TextFileKnowledgeSource(file_paths=["README.md"])
```

Paths are resolved relative to `knowledge/` in the project root unless absolute — CrewAI will look there first.

## Excel source

```python
from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource

xl = ExcelKnowledgeSource(file_paths=["quarterly.xlsx"])
```

Each sheet becomes an inner dict keyed by sheet name; chunks tag the sheet in the metadata when available.

## Docling — universal source

```python
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource

docs = CrewDoclingSource(file_paths=[
    "https://arxiv.org/pdf/2305.15334",        # URL
    "internal/handbook.pdf",
    "internal/handbook.docx",
])
```

Accepted formats per source: `MD`, `ASCIIDOC`, `PDF`, `DOCX`, `HTML`, images. If `docling` isn't installed, importing the class raises immediately.

## Custom sources

Subclass `BaseKnowledgeSource` (or `BaseFileKnowledgeSource` for file I/O). Two methods to implement:

```python
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource

class NotionSource(BaseKnowledgeSource):
    database_id: str

    def validate_content(self) -> None:
        if not self.database_id:
            raise ValueError("database_id required")

    def add(self) -> None:
        pages = fetch_notion_pages(self.database_id)   # your code
        for p in pages:
            self.chunks.extend(self._chunk_text(p.text))
        self._save_documents()
```

`_chunk_text` and `_save_documents` are inherited — you rarely need to override them.

## Attaching knowledge

### To an agent

```python
from crewai import Agent
agent = Agent(
    role="Legal Researcher",
    goal="Find relevant clauses",
    backstory="GC with 20 years of experience.",
    knowledge_sources=[pdfs, faq],
)
```

The collection name defaults to the agent's `role`; storage lives at `./knowledge/<collection>` (LanceDB).

### To a whole crew

```python
from crewai import Crew
crew = Crew(
    agents=[a, b],
    tasks=[t1, t2],
    knowledge_sources=[pdfs],            # shared across all agents
    embedder={"provider": "openai", "config": {"model": "text-embedding-3-large"}},
)
```

Agents inherit the crew embedder if they don't set their own.

## Tuning retrieval — `KnowledgeConfig`

```python
from crewai import Agent
from crewai.knowledge.knowledge_config import KnowledgeConfig

agent = Agent(
    role="Researcher",
    goal="...",
    backstory="...",
    knowledge_sources=[pdfs],
    knowledge_config=KnowledgeConfig(results_limit=8, score_threshold=0.5),
)
```

- `results_limit` (default `5`) — how many chunks the retriever returns.
- `score_threshold` (default `0.6`) — similarity floor; increase for precision, lower for recall.

## Querying directly

```python
from crewai import Knowledge
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

k = Knowledge(
    collection_name="kb",
    sources=[StringKnowledgeSource(content="Python was released in 1991.")],
)
k.add_sources()

hits = k.query(["when was python released?"], results_limit=3, score_threshold=0.6)
for h in hits:
    print(h)
```

- `query(queries, results_limit=5, score_threshold=0.6)` — synchronous.
- `aquery(...)` — async equivalent.
- `reset()` — wipe the collection.

## Patterns

### 1. Versioned company KB

```python
kb = CrewDoclingSource(file_paths=["knowledge/handbook-2026-q1.pdf"])
crew = Crew(agents=[...], tasks=[...], knowledge_sources=[kb])
```

New quarter → new file, rebuild the collection (call `Knowledge.reset()` then `add_sources()`).

### 2. Per-agent specialised sources

```python
legal  = Agent(role="Legal", ..., knowledge_sources=[legal_pdfs])
finops = Agent(role="FinOps", ..., knowledge_sources=[sql_schema_txt, runbooks_md])
```

Collections don't collide — they're isolated by `role`.

### 3. Tighten retrieval for a precise task

```python
Agent(
    role="Compliance",
    goal="Quote the exact paragraph",
    backstory="Paranoid reader.",
    knowledge_sources=[pdfs],
    knowledge_config=KnowledgeConfig(results_limit=3, score_threshold=0.75),
)
```

### 4. Reuse one Knowledge across crews

```python
from crewai import Knowledge

k = Knowledge(collection_name="shared-kb", sources=[pdfs, docs])
k.add_sources()   # build once

crew_a = Crew(..., knowledge=k)
crew_b = Crew(..., knowledge=k)
```

Pass `knowledge=` (a built `Knowledge` instance) instead of `knowledge_sources=` to skip re-indexing.

### 5. Mixed URL + local ingestion

```python
CrewDoclingSource(file_paths=[
    "https://developer.mozilla.org/en-US/docs/Web/API",
    "internal/onboarding.pdf",
])
```

Docling treats URLs and local paths the same as long as the format is supported.

## Gotchas

- **`knowledge/` is the default root.** Relative paths are resolved there; absolute paths work everywhere.
- **Embedder needed.** Either pass `embedder=` on the agent/crew, or set `OPENAI_API_KEY` for the default.
- **`PDFKnowledgeSource` uses pdfplumber**, not the Docling path. If you need tables/images extracted, use `CrewDoclingSource`.
- **Chunking is naive by default** — a sliding window over characters. Override `_chunk_text` in a subclass if you need sentence-aware splits.
- **`file_path` is deprecated.** Always use `file_paths=[...]`, even for a single file.
- **Knowledge is read-heavy.** It doesn't auto-update. For dynamic content, use Memory or rebuild the collection.
- **`knowledge_storage=` on an agent overrides the default LanceDB store.** Pass a custom `BaseKnowledgeStorage` if you need Qdrant/Pinecone/etc.

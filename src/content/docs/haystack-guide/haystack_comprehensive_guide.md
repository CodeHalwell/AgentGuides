---
title: "Haystack Comprehensive Technical Guide"
description: "Components, pipelines, the Agent class, tools, state, hooks and multi-agent patterns in Haystack 3."
framework: "haystack"
language: "python"
sidebar:
  label: "Comprehensive guide"
  order: 2
---

> Verified against `haystack-ai` %%version=haystack-ai%% (checked %%versions-checked%%). This guide was first written for Haystack 2.x; every example has been reworked for the 3.x API. If you are upgrading existing code, start with [Migrating from Haystack 2.x](#migrating-from-haystack-2x).

Haystack is deepset's open-source framework for building LLM applications out of typed **components** wired into **pipelines**. Retrieval, ranking, prompt building and generation are all components; so is the `Agent`, which runs a tool-calling loop and can sit inside a pipeline like anything else.

Haystack 3 is a consolidation release rather than a rewrite. The component and pipeline model is unchanged, but a number of legacy pieces were removed or moved out of the core package:

- Text **Generators** (`OpenAIGenerator` and friends) are gone; use the **ChatGenerators**, which now also accept a plain `str`.
- `ToolInvoker` is gone; tool execution belongs to `Agent`.
- `AsyncPipeline` has been merged into `Pipeline` (`run`, `run_async`, `stream`).
- Hugging Face, sentence-transformers, web search, Whisper, OpenTelemetry and Datadog components now live in separate integration packages.
- Prompt template variables are **required by default** in `PromptBuilder`, `ChatPromptBuilder`, `LLM` and `Agent`.
- Agent extension points are expressed as **hooks** (`before_run`, `before_llm`, `before_tool`, `after_tool`, `on_exit`, `after_run`); human-in-the-loop confirmation is one such hook.

## Table of Contents

1. [Part I: Core Fundamentals](#part-i-core-fundamentals)
   - [Installation and Setup](#installation-and-setup)
   - [Haystack 3 Architecture Overview](#haystack-3-architecture-overview)
   - [Components and Pipelines](#components-and-pipelines)
   - [Document Stores and Retrievers](#document-stores-and-retrievers)
   - [Agent Concepts in Haystack](#agent-concepts-in-haystack)
   - [Provider-Agnostic Design Philosophy](#provider-agnostic-design-philosophy)
   - [Configuration Patterns and Best Practices](#configuration-patterns-and-best-practices)
2. [Part II: Simple Agents](#part-ii-simple-agents)
   - [Creating Agents with the Agent Class](#creating-agents-with-the-agent-class)
   - [Tool Integration and Function Calling](#tool-integration-and-function-calling)
   - [Single-Purpose Agents](#single-purpose-agents)
   - [Conversational Agents with Memory](#conversational-agents-with-memory)
   - [Context Management](#context-management)
   - [Agent Configuration and Customisation](#agent-configuration-and-customisation)
   - [Error Handling and Resilience](#error-handling-and-resilience)
3. [Part III: Agentic Workflows and Multi-Agent Systems](#part-iii-agentic-workflows-and-multi-agent-systems)
   - [Agentic AI Workflows: Production-First Design](#agentic-ai-workflows-production-first-design)
   - [Multi-Agent Pipeline Design Patterns](#multi-agent-pipeline-design-patterns)
   - [SearchableToolset and SkillToolset](#searchabletoolset-and-skilltoolset)
   - [LLMRanker](#llmranker)
4. [Migrating from Haystack 2.x](#migrating-from-haystack-2x)

**Continue elsewhere in this guide:** [Multi-agent systems](/haystack-guide/haystack_multi_agent_guide/) for coordination patterns · [Advanced agents](/haystack-guide/haystack_advanced_agents_python/) for hooks, budgets, state and custom components · [Observability](/haystack-guide/haystack_observability_python/) for tracing, logging, retries and fallbacks · [Production guide](/haystack-guide/haystack_production_guide/) for deployment, security, testing and evaluation · [Recipes](/haystack-guide/haystack_recipes/) for end-to-end examples.

---

## Part I: Core fundamentals

### Installation and Setup

#### Basic Installation

The core package is `haystack-ai`. It requires Python 3.11 or later (3.10 support was dropped in 3.4.0).

```bash
# Latest release
pip install haystack-ai

# Pinned to the version this guide was verified against
pip install haystack-ai==%%version=haystack-ai%%
```

The core package ships the OpenAI and Azure OpenAI generators and embedders, the in-memory document store, converters, preprocessors, routers, joiners, builders, [evaluators](/haystack-guide/haystack_production_guide/#testing-and-evaluation) and the `Agent`. Everything else is an **integration package** that installs alongside it and is imported from the `haystack_integrations` namespace:

```bash
pip install anthropic-haystack            # AnthropicChatGenerator
pip install elasticsearch-haystack        # ElasticsearchDocumentStore + retrievers
pip install qdrant-haystack               # QdrantDocumentStore + retrievers
pip install sentence-transformers-haystack  # local embedders and rankers
pip install huggingface-api-haystack      # HF Inference API / TGI / TEI components
pip install serperdev-haystack            # SerperDevWebSearch
pip install mcp-haystack                  # MCPTool / MCPToolset
pip install opentelemetry-haystack        # OpenTelemetry tracer + connector
pip install langfuse-haystack             # LangfuseConnector
```

There are no `haystack-ai[...]` extras for these; install the integration packages by name.

Haystack sends anonymous usage statistics (component types, and for agents the generator, model and tool names; never prompts, documents or tool descriptions). Set `HAYSTACK_TELEMETRY_ENABLED=False` in the environment to opt out. The variable is read when `haystack` is imported, so set it in the shell or deployment rather than in code that runs after the import.

#### Virtual Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install haystack-ai
```

#### Verifying Installation

```python
from haystack import Pipeline, component
from haystack.version import __version__

print(f"Haystack version: {__version__}")


@component
class Upper:
    @component.output_types(text=str)
    def run(self, text: str):
        return {"text": text.upper()}


pipe = Pipeline()
pipe.add_component("upper", Upper())
print(pipe.run({"upper": {"text": "installation verified"}}))
# {'upper': {'text': 'INSTALLATION VERIFIED'}}
```

#### Docker Setup

```dockerfile
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir haystack-ai==%%version=haystack-ai%%

COPY . .
CMD ["python", "app.py"]
```

### Haystack 3 Architecture Overview

#### Architectural Principles

**1. Components.** A component is a class decorated with `@component` whose `run()` method declares typed inputs (its parameters) and typed outputs (via `@component.output_types`). Components can also implement `run_async()`, `warm_up()` and `close()`.

**2. Pipelines.** A `Pipeline` is a directed graph of components. Connections are validated by type when you call `connect()`. Pipelines may contain branches and loops; `max_runs_per_component` guards against runaway loops.

**3. One pipeline class, three ways to run it.** `Pipeline.run()` runs components sequentially. `await Pipeline.run_async()` runs independent branches concurrently (bounded by `concurrency_limit`). `Pipeline.stream()` returns a handle you iterate with `async for` to receive `StreamingChunk`s from every streaming-capable component.

**4. Provider-agnostic integrations.** Every chat generator implements the same `run(messages=...) -> {"replies": [...]}` contract, so you can swap OpenAI for Anthropic, Ollama or a mock without touching the rest of the pipeline.

**5. Agents are components.** `Agent` wraps a chat generator and a set of tools, runs the tool-calling loop, and exposes `messages` and `last_message` (plus any state keys) as outputs.

**6. Explicit resource lifecycle.** Components that talk to external services create their clients in `warm_up()` and release them in `close()`. Pipelines warm components up automatically on first run; call `pipeline.close()` (or `await pipeline.close_async()`) on shutdown.

#### Architectural Layers

```
┌─────────────────────────────────────────────────┐
│ Application layer                               │
│   Agents, RAG applications, services            │
├─────────────────────────────────────────────────┤
│ Pipeline layer                                  │
│   Pipeline.run / run_async / stream, routing,   │
│   loops, SuperComponent, serialisation          │
├─────────────────────────────────────────────────┤
│ Component layer                                 │
│   Builders, ChatGenerators, retrievers,         │
│   rankers, routers, joiners, Agent, tools       │
├─────────────────────────────────────────────────┤
│ Integration layer (haystack_integrations.*)     │
│   Anthropic, Ollama, Elasticsearch, Qdrant,     │
│   MCP, tracing connectors, ...                  │
├─────────────────────────────────────────────────┤
│ Data layer                                      │
│   Document, ChatMessage, ToolCall, State        │
└─────────────────────────────────────────────────┘
```

### Components and Pipelines

#### Understanding Components

```python
from haystack import component


@component
class TextStats:
    """Strips text and reports simple statistics."""

    def __init__(self, min_length: int = 1):
        self.min_length = min_length

    @component.output_types(text=str, word_count=int, char_count=int)
    def run(self, text: str):
        cleaned = text.strip()
        if len(cleaned) < self.min_length:
            raise ValueError(f"Text shorter than {self.min_length} characters")
        return {
            "text": cleaned,
            "word_count": len(cleaned.split()),
            "char_count": len(cleaned),
        }


stats = TextStats(min_length=3)
print(stats.run(text="  Hello world  "))
# {'text': 'Hello world', 'word_count': 2, 'char_count': 11}
```

Rules worth remembering:

- Inputs are the parameters of `run()`. A parameter with a default is optional.
- Outputs must be declared with `@component.output_types(...)` and returned as a dict with exactly those keys.
- Serialisation reads each `__init__` parameter back from the attribute of the same name (`self.min_length` for `min_length`) and falls back to the parameter's default if there is no such attribute. So store every parameter under its own name: a component with `def __init__(self, n=2): self.limit = n` built with `n=5` is saved with `n: 2`, with no error. Values must be JSON-serialisable; if a parameter is not (a client object, a callable), implement `to_dict()` / `from_dict()` yourself, or `Pipeline.dumps()` raises `SerializationError`.
- For async support, add `async def run_async(...)` with the same signature.

#### Creating Pipelines

```python
from haystack import Pipeline, component


@component
class Formatter:
    @component.output_types(formatted=str)
    def run(self, text: str, word_count: int):
        return {"formatted": f"{text} ({word_count} words)"}


pipeline = Pipeline()
pipeline.add_component("stats", TextStats())
pipeline.add_component("formatter", Formatter())

# "sender.output_name" -> "receiver.input_name"
pipeline.connect("stats.text", "formatter.text")
pipeline.connect("stats.word_count", "formatter.word_count")

result = pipeline.run({"stats": {"text": "sample text here"}})
print(result["formatter"]["formatted"])  # sample text here (3 words)
```

`Pipeline.run()` returns the outputs of leaf components only. Pass `include_outputs_from={"stats"}` to also get intermediate outputs.

#### A Minimal RAG Pipeline

```python
from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore

store = InMemoryDocumentStore()
store.write_documents([
    Document(content="Haystack 3 merged AsyncPipeline into Pipeline."),
    Document(content="Haystack 3 removed the legacy text Generators."),
])

template = [
    ChatMessage.from_system("Answer using only the supplied documents."),
    ChatMessage.from_user(
        "Documents:\n{% for doc in documents %}- {{ doc.content }}\n{% endfor %}\n"
        "Question: {{ question }}"
    ),
]

rag = Pipeline()
rag.add_component("retriever", InMemoryBM25Retriever(document_store=store, top_k=3))
rag.add_component("prompt", ChatPromptBuilder(template=template))
rag.add_component("llm", OpenAIChatGenerator(model="%%model=openai-small%%"))

rag.connect("retriever.documents", "prompt.documents")
rag.connect("prompt.prompt", "llm.messages")

question = "What happened to AsyncPipeline?"
result = rag.run({"retriever": {"query": question}, "prompt": {"question": question}})
print(result["llm"]["replies"][0].text)
```

#### Branching with ConditionalRouter

`ConditionalRouter` routes are Jinja2 expressions. Each route needs a `condition`, an `output`, an `output_name` and an `output_type`; the first route whose condition is true wins.

```python
from haystack.components.routers import ConditionalRouter

routes = [
    {
        "condition": "{{ 'refund' in query | lower }}",
        "output": "{{ query }}",
        "output_name": "billing",
        "output_type": str,
    },
    {
        "condition": "{{ True }}",  # fallback
        "output": "{{ query }}",
        "output_name": "general",
        "output_type": str,
    },
]

router = ConditionalRouter(routes=routes)
print(router.run(query="I want a refund"))  # {'billing': 'I want a refund'}
```

In a pipeline, connect `router.billing` and `router.general` to different downstream components; only the branch that receives input runs. Use `BranchJoiner` to merge branches back into one stream, and `DocumentJoiner`, `AnswerJoiner`, `ListJoiner` or `StringJoiner` to combine outputs that arrive together.

#### Branching inside a pipeline

A component runs once all of its mandatory inputs have arrived. A router sends its value to exactly one output socket, so the components wired to the other sockets never receive input and never run. This example routes short and long questions to different handlers and merges them again with a `BranchJoiner`:

```python
from haystack import Pipeline, component
from haystack.components.joiners import BranchJoiner
from haystack.components.routers import ConditionalRouter


@component
class Tag:
    def __init__(self, label: str):
        self.label = label

    @component.output_types(text=str)
    def run(self, text: str):
        return {"text": f"[{self.label}] {text}"}


routes = [
    {"condition": "{{ query | length > 20 }}", "output": "{{ query }}", "output_name": "long", "output_type": str},
    {"condition": "{{ query | length <= 20 }}", "output": "{{ query }}", "output_name": "short", "output_type": str},
]

branching = Pipeline()
branching.add_component("router", ConditionalRouter(routes=routes))
branching.add_component("quick", Tag("quick path"))
branching.add_component("thorough", Tag("thorough path"))
branching.add_component("joiner", BranchJoiner(str))

branching.connect("router.short", "quick.text")
branching.connect("router.long", "thorough.text")
branching.connect("quick.text", "joiner")
branching.connect("thorough.text", "joiner")

print(branching.run({"router": {"query": "hi"}}))
# {'joiner': {'value': '[quick path] hi'}}
print(branching.run({"router": {"query": "a considerably longer question than twenty characters"}}))
# {'joiner': {'value': '[thorough path] a considerably longer question than twenty characters'}}
```

Without the joiner, `run()` returns the output of whichever branch ran, keyed by that component's name (for example `{"quick": {"text": "[quick path] hi"}}`).

#### Merging inputs and running branches concurrently

A component whose `run()` parameter is typed `Variadic[...]` accepts any number of incoming connections and receives their values as one iterable. Independent branches overlap when you call `run_async()`; `concurrency_limit` (default 4) bounds how many components run at once. A component with an `async def run_async()` is awaited directly, and one without it is run in a worker thread, so blocking code still overlaps but a real coroutine is cheaper.

```python
import asyncio
import time

from haystack import Pipeline, component
from haystack.core.component.types import Variadic


@component
class SlowLookup:
    """Stands in for a call to a slow service; has a sync and an async implementation."""

    def __init__(self, source: str, delay: float = 0.3):
        self.source, self.delay = source, delay

    @component.output_types(result=str)
    def run(self, query: str):
        time.sleep(self.delay)
        return {"result": f"{self.source}: {query}"}

    @component.output_types(result=str)
    async def run_async(self, query: str):
        await asyncio.sleep(self.delay)
        return {"result": f"{self.source}: {query}"}


@component
class Collect:
    @component.output_types(results=list[str])
    def run(self, results: Variadic[str]):
        return {"results": sorted(results)}


fan_out = Pipeline()
for name in ("docs", "wiki", "tickets"):
    fan_out.add_component(name, SlowLookup(name))
fan_out.add_component("collect", Collect())
for name in ("docs", "wiki", "tickets"):
    fan_out.connect(f"{name}.result", "collect.results")

data = {name: {"query": "refund policy"} for name in ("docs", "wiki", "tickets")}

start = time.perf_counter()
fan_out.run(data)                                    # one after another: about 0.9 s (the first call adds start-up cost)
print(f"run:       {time.perf_counter() - start:.1f}s")

start = time.perf_counter()
result = asyncio.run(fan_out.run_async(data))        # all three at once: about 0.3 s
print(f"run_async: {time.perf_counter() - start:.1f}s")
print(result["collect"]["results"])
# ['docs: refund policy', 'tickets: refund policy', 'wiki: refund policy']
```

`concurrency_limit=1` makes `run_async()` sequential again, which is useful when a downstream service has a strict rate limit.

### Document Stores and Retrievers

A **document store** holds `Document`s (content, `meta`, an optional embedding). A **retriever** is a component that queries one store. Every store implements the same small protocol (`write_documents`, `filter_documents`, `count_documents`, `delete_documents`); searching is done by store-specific retrievers, so a BM25 retriever and an embedding retriever can share one store. `InMemoryDocumentStore` needs no server and is the right choice for tests, notebooks and small corpora. Production deployments swap in an integration store (Elasticsearch, Qdrant and others) and its retrievers without changing the rest of the pipeline.

Embedding retrieval needs two components: a **document embedder** at indexing time and a **text embedder** at query time, both using the same model. The example below uses a deliberately crude hashing embedder so that it runs offline; replace `HashDocumentEmbedder` and `HashTextEmbedder` with `OpenAIDocumentEmbedder` and `OpenAITextEmbedder` (or an integration embedder) for real use, and keep the rest as is.

```python
import hashlib
import math

from haystack import Document, Pipeline, component
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

DIM = 256


def hash_embed(text: str) -> list[float]:
    """Toy bag-of-words embedding: hash each word into one of DIM buckets, then normalise."""
    vec = [0.0] * DIM
    for word in text.lower().split():
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@component
class HashTextEmbedder:
    @component.output_types(embedding=list[float])
    def run(self, text: str):
        return {"embedding": hash_embed(text)}


@component
class HashDocumentEmbedder:
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        return {"documents": [
            Document(content=d.content, meta=d.meta, embedding=hash_embed(d.content or "")) for d in documents
        ]}


store = InMemoryDocumentStore(embedding_similarity_function="cosine")

indexing = Pipeline()
indexing.add_component("embed", HashDocumentEmbedder())
indexing.add_component("write", DocumentWriter(document_store=store, policy=DuplicatePolicy.OVERWRITE))
indexing.connect("embed.documents", "write.documents")
indexing.run({"embed": {"documents": [
    Document(content="Returns are accepted within 30 days", meta={"lang": "en", "year": 2024}),
    Document(content="Shipping takes five to seven days", meta={"lang": "en", "year": 2023}),
    Document(content="Les retours sont acceptes sous 30 jours", meta={"lang": "fr", "year": 2024}),
]}})
print(store.count_documents())   # 3

# Hybrid retrieval: BM25 and embeddings over the same store, fused with reciprocal rank fusion
hybrid = Pipeline()
hybrid.add_component("embed", HashTextEmbedder())
hybrid.add_component("bm25", InMemoryBM25Retriever(document_store=store, top_k=3))
hybrid.add_component("dense", InMemoryEmbeddingRetriever(document_store=store, top_k=3))
hybrid.add_component("join", DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=2))
hybrid.connect("embed.embedding", "dense.query_embedding")
hybrid.connect("bm25.documents", "join.documents")
hybrid.connect("dense.documents", "join.documents")

question = "how many days for returns"
result = hybrid.run({"embed": {"text": question}, "bm25": {"query": question}})
print([d.content for d in result["join"]["documents"]])
# ['Returns are accepted within 30 days', 'Shipping takes five to seven days']

# Metadata filters can be passed to each retriever at run time
french = {"field": "meta.lang", "operator": "==", "value": "fr"}
result = hybrid.run({
    "embed": {"text": question},
    "bm25": {"query": "retours jours", "filters": french},
    "dense": {"filters": french},
})
print([d.content for d in result["join"]["documents"]])
# ['Les retours sont acceptes sous 30 jours']
```

The same filters work directly on the store:

```python
recent_english = {
    "operator": "AND",
    "conditions": [
        {"field": "meta.lang", "operator": "==", "value": "en"},
        {"field": "meta.year", "operator": ">=", "value": 2024},
    ],
}
print([d.content for d in store.filter_documents(filters=recent_english)])
# ['Returns are accepted within 30 days']
```

#### Things worth knowing

- **Filter syntax.** A filter is either a comparison, `{"field": "meta.year", "operator": ">=", "value": 2024}`, or a logical group, `{"operator": "AND", "conditions": [...]}`. Comparison operators are `==`, `!=`, `>`, `>=`, `<`, `<=`, `in` and `not in`; logical operators are `AND`, `OR` and `NOT`. Fields inside `meta` are addressed as `meta.<name>`.
- **Duplicates.** `DocumentWriter` takes a `DuplicatePolicy`: `NONE` (the default) uses the store's default, which for `InMemoryDocumentStore` is to raise `DuplicateDocumentError`; `SKIP` ignores documents whose ID already exists; `OVERWRITE` replaces them; `FAIL` raises. A document's ID is a hash of its content, `meta`, blob and embeddings, so re-indexing identical documents is detected automatically (and a document that gains an embedding gets a new ID).
- **Filter policy.** Retrievers accept `filters` at construction and at run time. By default (`FilterPolicy.REPLACE`) the run-time filters replace the constructor's; pass `filter_policy=FilterPolicy.MERGE` (from `haystack.document_stores.types`) to combine them instead.
- **Scores.** `InMemoryBM25Retriever` uses the `BM25L` variant by default (`bm25_algorithm` on the store also accepts `BM25Okapi` and `BM25Plus`). `InMemoryEmbeddingRetriever` scores with the store's `embedding_similarity_function`, `"dot_product"` unless you choose `"cosine"`. Scores from different retrievers are not comparable, which is why the example fuses by rank.
- **Persistence.** The in-memory store lives in the process. `store.save_to_disk(path)` and `InMemoryDocumentStore.load_from_disk(path)` write and restore it as JSON, embeddings included; use an integration store when several processes need the same data.
- **Experimental helpers.** `TextEmbeddingRetriever` (wraps a text embedder and an embedding retriever into one component) and `MultiRetriever` (runs several text retrievers in parallel and merges their results) are in `haystack.components.retrievers`. `MultiRetriever` is marked experimental in the source and may change without a deprecation cycle.

### Agent Concepts in Haystack

The `Agent` component (`haystack.components.agents.Agent`) runs this loop:

```
messages (+ rendered system_prompt / user_prompt)
        │
        ▼
 ┌──────────────┐  before_llm hooks
 │ chat_generator│◄───────────────────────────┐
 └──────┬───────┘                             │
        │ reply                               │
        ▼                                     │
  tool calls? ── no ──► exit condition "text" │
        │ yes                                 │
        ▼  before_tool hooks                  │
 ┌──────────────┐                             │
 │ execute tools │ (concurrently, bounded by  │
 │               │  tool_concurrency_limit)   │
 └──────┬───────┘  after_tool hooks           │
        │ tool results appended to messages   │
        ├── tool name in exit_conditions? ──► stop
        └─────────────────────────────────────┘
   on_exit hooks may set continue_run = True to keep going
```

Key points:

- The loop stops when the model replies with text and `"text"` is in `exit_conditions` (the default), when a tool listed in `exit_conditions` is called, when a hook sets `stop_run`, or when `max_agent_steps` (default 100) is reached.
- Everything the agent knows during a run lives in a `State` object. `messages`, `step_count`, `token_usage` and `tool_call_counts` are always present; you add your own keys with `state_schema`.
- `Agent.run()` returns a dict with `messages`, `last_message`, `step_count`, `token_usage`, `tool_call_counts`, `exit_reason` and any custom state keys that were set during the run.

### Provider-Agnostic Design Philosophy

Every chat generator takes `messages` and returns `{"replies": list[ChatMessage]}`, and accepts the same `tools` argument. Swapping providers is a one-line change:

```python
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator  # pip install anthropic-haystack
from haystack.dataclasses import ChatMessage


def make_generator(provider: str):
    if provider == "openai":
        return OpenAIChatGenerator(model="%%model=openai-small%%")          # reads OPENAI_API_KEY
    if provider == "anthropic":
        return AnthropicChatGenerator(model="%%model=anthropic%%")  # reads ANTHROPIC_API_KEY
    raise ValueError(f"Unknown provider: {provider}")


for provider in ["openai", "anthropic"]:
    generator = make_generator(provider)
    reply = generator.run(messages=[ChatMessage.from_user("Say hello in five words.")])["replies"][0]
    print(provider, reply.text, reply.meta.get("usage"))
```

Model names change faster than this guide; check your provider's current model list.

API keys are passed as `Secret` objects, never plain strings. The default is to read a named environment variable:

```python
from haystack.utils import Secret

generator = OpenAIChatGenerator(
    model="%%model=openai-small%%",
    api_key=Secret.from_env_var("OPENAI_API_KEY"),  # this is already the default
    generation_kwargs={"max_completion_tokens": 512},
    timeout=30,
    max_retries=3,
)
```

`OpenAIChatGenerator` also works with OpenAI-compatible servers (vLLM, LM Studio, many hosted gateways) via `api_base_url`. For OpenAI's Responses API use `OpenAIResponsesChatGenerator`.

#### Document stores

All document stores implement the `DocumentStore` protocol (`write_documents`, `filter_documents`, `count_documents`, `delete_documents`). Retrieval is done by store-specific **retriever components**, not by methods on the store. [Document Stores and Retrievers](#document-stores-and-retrievers) covers filters, hybrid retrieval and an offline example; this snippet shows how integration stores slot in:

```python
# pip install elasticsearch-haystack qdrant-haystack
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever


def make_bm25_retriever(kind: str):
    if kind == "memory":
        return InMemoryBM25Retriever(document_store=InMemoryDocumentStore())
    if kind == "elasticsearch":
        store = ElasticsearchDocumentStore(hosts="http://localhost:9200", index="documents")
        return ElasticsearchBM25Retriever(document_store=store)
    raise ValueError(kind)


qdrant = QdrantDocumentStore(url="http://localhost:6333", index="documents", embedding_dim=1536)
embedding_retriever = QdrantEmbeddingRetriever(document_store=qdrant, top_k=5)
```

Note that in Haystack 3, auto-generated `Document.id` values for documents with non-empty `meta` differ from those produced by 2.x. If you rely on IDs for deduplication against an index written by 2.x, see the [migration notes](#migrating-from-haystack-2x).

### Configuration Patterns and Best Practices

#### Environment-based configuration

Keep secrets in the environment and everything else in a small typed config object:

```python
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AppConfig:
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "%%model=openai-small%%"))
    max_agent_steps: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_STEPS", "10")))
    retriever_top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVER_TOP_K", "5")))

    def validate(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        if self.max_agent_steps < 1:
            raise ValueError("AGENT_MAX_STEPS must be at least 1")


config = AppConfig()
```

Remember that from 3.0 a missing API key is reported when the component warms up or first runs, not when it is constructed, so call `config.validate()` (or `pipeline.warm_up()`) at start-up if you want to fail fast.

#### YAML pipeline definitions

Pipelines serialise to YAML and back:

```python
import tempfile
from pathlib import Path

from haystack import Pipeline

yaml_text = rag.dumps()                 # a string; rag.dump(fp) writes to a file

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "rag.yaml"
    with open(path, "w") as fp:
        rag.dump(fp)
    with open(path) as fp:
        restored = Pipeline.load(fp)    # or Pipeline.loads(yaml_text)
```

From 3.0, loading is gated by a module allowlist. Classes from `haystack`, `haystack_integrations` and `haystack_experimental` load out of the box; your own components and tool functions must be allowed explicitly:

```python
from haystack.core.serialization import allow_deserialization_module

with open("agent.yaml") as fp:
    pipe = Pipeline.load(fp, allowed_modules=["myapp.*"])   # per call

allow_deserialization_module("myapp.*")                     # process-wide
# or: export HAYSTACK_DESERIALIZATION_ALLOWLIST="myapp.*"
```

`Pipeline.load(fp, unsafe=True)` bypasses the allowlist. Only use it for YAML you wrote yourself.

---

## Part II: Simple agents

### Creating Agents with the Agent Class

#### Basic agent with tools

```python
from typing import Annotated

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import tool


@tool
def get_weather(city: Annotated[str, "Name of the city"]) -> str:
    """Get the current weather for a city."""
    data = {"London": "15°C, rain", "Tokyo": "20°C, cloudy"}
    return data.get(city, "No data for that city")


@tool
def search(query: Annotated[str, "Search query"]) -> str:
    """Search the web for recent information."""
    return f"(stub) top results for {query!r}"


agent = Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    tools=[get_weather, search],
    system_prompt="You are a concise assistant. Use tools when they help.",
    max_agent_steps=10,
)

result = agent.run(messages=[ChatMessage.from_user("What's the weather in London?")])
print(result["last_message"].text)
print(result["tool_call_counts"], result["token_usage"])
```

The `@tool` decorator builds the JSON schema from the type hints, using `Annotated[...]` strings as parameter descriptions and the docstring as the tool description.

#### Using a `user_prompt` template

Instead of building `ChatMessage`s yourself, give the agent a Jinja2 `user_prompt`. Its variables become run-time inputs (and pipeline input sockets):

```python
agent = Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    system_prompt="You are a careful analyst.",
    user_prompt="Summarise the following in {{ language }}:\n\n{{ text }}",
)

result = agent.run(messages=[], text="Haystack is a framework ...", language="French")
print(result["last_message"].text)
```

Template variables are required by default. Pass `required_variables=["text"]` to make the rest optional, or `required_variables=None` to make them all optional. Prompts are fixed at construction; to vary the prompt per run, build the `ChatMessage`s yourself and pass them as `messages`.

#### Async and streaming

```python
import asyncio
from haystack.components.generators.utils import print_streaming_chunk

# Print tokens and tool activity as they arrive
agent.run(
    messages=[],
    text="Haystack is a framework ...",
    language="English",
    streaming_callback=print_streaming_chunk,
)


async def main() -> str:
    result = await agent.run_async(messages=[], text="Haystack is a framework ...", language="German")
    return result["last_message"].text

print(asyncio.run(main()))
```

#### Runtime tool selection

`Agent.run()` accepts `tools=` to override the tool set for one run. Pass either tool objects or a list of tool names drawn from the tools given at construction:

```python
agent = Agent(chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"), tools=[get_weather, search])
agent.run(messages=[ChatMessage.from_user("Weather in Tokyo?")], tools=["get_weather"])
```

#### Images and files

A user message can carry images alongside text. Build it with `content_parts`, and use a vision-capable model:

```python
from haystack.dataclasses import ChatMessage, ImageContent

image = ImageContent.from_file_path("receipt.png", detail="low")   # or ImageContent.from_url(...)
message = ChatMessage.from_user(content_parts=["What is the total on this receipt?", image])

result = agent.run(messages=[message])
print(result["last_message"].text)
```

Documents such as PDFs travel as `FileContent.from_file_path(...)`; `OpenAIResponsesChatGenerator` and `AnthropicChatGenerator` accept them, the Chat Completions-based `OpenAIChatGenerator` does not. In pipelines, `ImageFileToImageContent` and `PDFToImageContent` (from `haystack.components.converters.image`) turn files into `ImageContent`, and a string template in `ChatPromptBuilder` can place one in a message with `{{ image | templatize_part }}`.

### Tool Integration and Function Calling

Haystack has several tool types, all subclasses of `haystack.tools.Tool`:

| Type | Use it for |
|------|------------|
| `@tool` / `create_tool_from_function` | A plain Python function with type hints |
| `Tool(...)` | Full control over name, description and JSON schema |
| `ComponentTool` | Exposing any Haystack component (retriever, web search, ...) as a tool |
| `PipelineTool` | Exposing a whole pipeline as a tool |
| `AgentTool` | Exposing another `Agent` as a tool (multi-agent delegation) |
| `MCPTool` / `MCPToolset` | Tools served by an MCP server (`mcp-haystack`) |
| `Toolset` / `SearchableToolset` | Grouping tools; letting the agent search a large catalogue |
| `SkillToolset` | Instruction "skills" (`SKILL.md` files) the agent loads on demand |

#### Function tools

```python
from typing import Annotated, Literal

from haystack.tools import tool


@tool
def compound_interest(
    principal: Annotated[float, "Initial amount"],
    rate_percent: Annotated[float, "Annual interest rate, e.g. 5 for 5%"],
    years: Annotated[int, "Number of years"],
    compounding: Annotated[Literal["yearly", "quarterly", "monthly"], "Compounding frequency"] = "yearly",
) -> dict:
    """Calculate compound interest."""
    n = {"yearly": 1, "quarterly": 4, "monthly": 12}[compounding]
    amount = principal * (1 + rate_percent / 100 / n) ** (n * years)
    return {"final_amount": round(amount, 2), "interest": round(amount - principal, 2)}
```

Non-string results are serialised with `json.dumps` before being sent back to the model. Use `outputs_to_string={"handler": my_fn}` (or `{"source": "key"}`) to control that.

#### Explicit `Tool` definitions

Use `Tool` directly when the schema should differ from the Python signature, or the function is not yours:

```python
from haystack.tools import Tool

def run_sql(sql: str) -> list[dict]:
    # Replace with a read-only database call
    return [{"id": 1, "name": "Example"}]

sql_tool = Tool(
    name="run_sql",
    description="Run a read-only SQL query against the analytics database.",
    parameters={
        "type": "object",
        "properties": {"sql": {"type": "string", "description": "A single SELECT statement"}},
        "required": ["sql"],
    },
    function=run_sql,
)
```

#### Components as tools

```python
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import ComponentTool

store = InMemoryDocumentStore()
doc_search = ComponentTool(
    component=InMemoryBM25Retriever(document_store=store, top_k=3),
    name="search_docs",
    description="Search the internal knowledge base.",
)

# Web search lives in an integration package from 3.0: pip install serperdev-haystack
from haystack_integrations.components.websearch.serperdev import SerperDevWebSearch

web_search = ComponentTool(
    component=SerperDevWebSearch(top_k=3),   # reads SERPERDEV_API_KEY
    name="web_search",
    description="Search the web for current information.",
)
```

#### MCP tools

`mcp-haystack` connects an Agent to any [Model Context Protocol](https://modelcontextprotocol.io) server. `MCPToolset` loads every tool the server offers (or the ones named in `tool_names`) and `MCPTool` wraps a single named tool. The server is described by `StdioServerInfo` (a subprocess), `StreamableHttpServerInfo` (a URL) or `SSEServerInfo` (legacy SSE endpoints).

```python
# pip install mcp-haystack
from haystack_integrations.tools.mcp import MCPToolset, StdioServerInfo, StreamableHttpServerInfo

fs_tools = MCPToolset(
    server_info=StdioServerInfo(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/data"]),
    tool_names=["read_file", "list_directory"],   # optional allow-list
)

remote_tools = MCPToolset(server_info=StreamableHttpServerInfo(url="http://localhost:8000/mcp"))

agent = Agent(chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"), tools=fs_tools)
```

To try MCP without Node or a network service, write a small server with the MCP Python SDK (installed with `mcp-haystack`). This one is `calc_server.py`:

```python
# calc_server.py
from mcp.server.mcpserver import MCPServer   # in mcp 1.x: from mcp.server.fastmcp import FastMCP

server = MCPServer("calc")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@server.tool()
def shout(text: str) -> str:
    """Upper-case some text."""
    return text.upper()


if __name__ == "__main__":
    server.run(transport="stdio")
```

and the client runs it as a subprocess. The reply here is scripted with `MockChatGenerator`, so the example needs no API key:

```python
import json
import sys

from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack_integrations.tools.mcp import MCPToolset, StdioServerInfo

server = StdioServerInfo(command=sys.executable, args=["calc_server.py"])


def text_only(result: str) -> str:
    # By default the tool result sent to the model is the whole MCP CallToolResult as JSON
    return json.loads(result)["content"][0]["text"]


tools = MCPToolset(
    server_info=server,
    tool_names=["add"],
    eager_connect=True,                               # connect now instead of at warm-up
    outputs_to_string={"add": {"handler": text_only}},  # keyed by tool name
)
print([t.name for t in tools])   # ['add']

llm = MockChatGenerator(responses=[
    ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="add", arguments={"a": 2, "b": 3})]),
    "2 + 3 = 5",
])
agent = Agent(chat_generator=llm, tools=tools)
result = agent.run(messages=[ChatMessage.from_user("What is 2 + 3?")])

print(result["messages"][-2].tool_call_result.result)   # 5
print(result["last_message"].text)                      # 2 + 3 = 5
tools.close()
```

Points that catch people out:

- **Connection is lazy by default.** Without `eager_connect=True` the toolset connects when the Agent warms up. Until then, iterating over it yields a single placeholder tool called `mcp_not_connected_placeholder_...`, so list or filter tools only after connecting.
- **Results are JSON.** The default string passed to the model is the full `CallToolResult` (content blocks, `structuredContent`, `isError`). Use `outputs_to_string` to pass only what the model needs.
- **Close what you open.** Call `close()` on the toolset or tool when you have finished with it, or the subprocess and connection stay alive.
- **Timeouts.** `connection_timeout` and `invocation_timeout` default to 30 seconds. `StdioServerInfo` and the HTTP server infos retry connections (`max_retries`, `base_delay`, `max_delay`).
- **Trust.** A stdio server runs with your privileges, and an MCP server's tool descriptions and outputs are untrusted input. Prefer an allow-list in `tool_names` and see [Security](/haystack-guide/haystack_production_guide/#security).

#### Tool errors

By default a tool that raises is reported back to the model as an error result so it can recover. Set `raise_on_tool_invocation_failure=True` on the `Agent` to raise instead. Validate inputs inside the tool and return a helpful message rather than relying on exceptions for expected failures:

```python
import ast
import operator

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


@tool
def calculate(expression: Annotated[str, "Arithmetic expression using + - * / and brackets"]) -> str:
    """Evaluate a simple arithmetic expression safely."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        raise ValueError("unsupported expression")
    try:
        return str(ev(ast.parse(expression, mode="eval")))
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        return f"Error: {exc}"
```

### Single-Purpose Agents

A single-purpose agent is just an `Agent` with a narrow system prompt and a small, well-described tool set. Wrap it in a function or class if you need extra bookkeeping.

#### Customer support agent

```python
from typing import Annotated, Literal

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import tool

KNOWLEDGE_BASE = {
    "returns": "Items can be returned within 30 days for a full refund.",
    "shipping": "Standard shipping takes 5-7 business days.",
    "warranty": "All products come with a one-year warranty.",
}


@tool
def lookup_policy(topic: Annotated[Literal["returns", "shipping", "warranty"], "Policy topic"]) -> str:
    """Look up a customer-support policy."""
    return KNOWLEDGE_BASE[topic]


@tool
def check_order_status(order_id: Annotated[str, "Order ID such as ORD-12345"]) -> dict:
    """Check the status of an order."""
    return {"order_id": order_id, "status": "Shipped", "tracking": "TRK123456789"}


@tool
def create_ticket(
    issue: Annotated[str, "Short description of the issue"],
    priority: Annotated[Literal["low", "normal", "high"], "Ticket priority"] = "normal",
) -> dict:
    """Create a support ticket for issues the assistant cannot resolve."""
    return {"ticket_id": "TICKET-001", "issue": issue, "priority": priority}


support_agent = Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    tools=[lookup_policy, check_order_status, create_ticket],
    system_prompt=(
        "You are a polite customer support specialist. Resolve issues with the tools available. "
        "If you cannot resolve an issue, create a ticket and give the customer its ID."
    ),
    max_agent_steps=8,
)

result = support_agent.run(messages=[ChatMessage.from_user("Can I return order ORD-12345?")])
print(result["last_message"].text)
```

#### Data analysis agent

Closures are a tidy way to give tools access to an object without exposing it to the model:

```python
import pandas as pd
from haystack.tools import tool


def make_analysis_agent(df: pd.DataFrame) -> Agent:
    @tool
    def describe_data() -> str:
        """Return column names, dtypes and summary statistics for the dataset."""
        return df.describe(include="all").to_json()

    @tool
    def correlate(col1: Annotated[str, "First column"], col2: Annotated[str, "Second column"]) -> str:
        """Return the Pearson correlation between two numeric columns."""
        if col1 not in df or col2 not in df:
            return f"Unknown column. Available: {list(df.columns)}"
        return f"{df[col1].corr(df[col2]):.3f}"

    return Agent(
        chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
        tools=[describe_data, correlate],
        system_prompt="You are a data analyst. Use the tools, then explain the result plainly.",
    )


data = pd.DataFrame({"sales": [100, 150, 200, 180, 220], "marketing": [10, 15, 20, 18, 25]})
analyst = make_analysis_agent(data)
print(analyst.run(messages=[ChatMessage.from_user("How do marketing and sales relate?")])["last_message"].text)
```

### Conversational Agents with Memory

`Agent` is stateless between runs: memory is simply the list of messages you pass in. The `messages` output contains the full transcript of the run (system prompt, user turn, tool calls, tool results and the final reply), so the simplest memory is to keep that list and feed it back.

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole


class Conversation:
    def __init__(self, agent: Agent, max_messages: int = 40):
        self.agent = agent
        self.max_messages = max_messages
        self.history: list[ChatMessage] = []

    def chat(self, text: str) -> str:
        self.history.append(ChatMessage.from_user(text))
        result = self.agent.run(messages=self.history)
        # Drop the system prompt the agent prepends; it is re-added on every run.
        self.history = [m for m in result["messages"] if not m.is_from(ChatRole.SYSTEM)]
        self.history = self.history[-self.max_messages:]
        return result["last_message"].text


chat = Conversation(Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    system_prompt="You are a friendly assistant. Remember what the user tells you.",
))
chat.chat("Hi, my name is Alice and my favourite colour is blue.")
print(chat.chat("What is my favourite colour?"))
```

Truncating by message count can split a tool call from its result. In production, trim on turn boundaries (start the kept window at a user message), or let a [compaction hook](#context-management) do it inside the run.

For persistence across processes, serialise messages with `ChatMessage.to_dict()` / `ChatMessage.from_dict()` and store them in Redis, a database or a document store keyed by session ID. The [recipes](/haystack-guide/haystack_recipes/) page has a Redis-backed example.

### Context Management

Long runs fill the context window with tool output and old turns. Haystack ships three hooks for this. `CompactionHook`, its compactors and `TokenBudgetHook` are experimental: they emit an `ExperimentalWarning` and may change without a deprecation cycle.

| Hook | Hook point | What it does |
|------|------------|--------------|
| `TokenBudgetHook(max_total_tokens=...)` | `before_llm` | Ends the run with `exit_reason="token_budget_exceeded"` (see [token budgets](/haystack-guide/haystack_advanced_agents_python/#token-budgets)) |
| `CompactionHook(compactor, context_window=..., compact_at=0.7, compact_to=0.4)` | `before_llm` | Once the conversation reaches `compact_at` of the window, a compactor shrinks it towards `compact_to` |
| `ToolResultOffloadHook(store, offload_strategies)` | `after_tool` | Writes selected tool results to a store and leaves a pointer with a short preview in the conversation |

The compactors, all in `haystack.hooks.compaction`:

- `SlidingWindowCompactor()` drops the oldest complete turns, then the oldest agent steps of the current task, and leaves an omission note. It keeps the system messages and the latest user message, and never separates a tool call from its result.
- `SummarizationCompactor(chat_generator=...)` replaces old turns with summaries written by a (usually cheaper) model.
- `ToolResultPruningCompactor()` replaces the content of older tool results with a placeholder, keeping every call and result pair.
- `ToolResultOffloadCompactor(store=...)` moves older tool results to a store, like the offload hook but only once compaction is triggered.

Implement the `Compactor` protocol for your own strategy. Compaction is lossy: the agent continues from the shorter record.

```python
import tempfile
from typing import Annotated

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.hooks.compaction import CompactionHook, SlidingWindowCompactor
from haystack.hooks.tool_result_offloading import (
    FileSystemToolResultStore,
    OffloadOverChars,
    ToolResultOffloadHook,
)
from haystack.tools import tool

store = FileSystemToolResultStore(root=tempfile.mkdtemp())   # one directory per session in a service


@tool
def fetch_report(name: Annotated[str, "Report name"]) -> str:
    """Fetch the full text of a report."""
    return f"{name}: " + "quarterly figures ... " * 1000


@tool
def read_offloaded_result(path: Annotated[str, "Path of an offloaded tool result"]) -> str:
    """Read the full text of a tool result that was moved to storage."""
    content = store.read(path)
    return content if isinstance(content, str) else f"{len(content)} bytes of binary content"


agent = Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    tools=[fetch_report, read_offloaded_result],
    hooks={
        "after_tool": [ToolResultOffloadHook(store=store, offload_strategies={"fetch_report": OffloadOverChars(4000)})],
        "before_llm": [CompactionHook(compactor=SlidingWindowCompactor(), context_window=400_000)],
    },
    max_agent_steps=50,
)
```

Set `context_window` to your model's window in tokens, and leave headroom above `compact_at` for the next reply and its tool results. `offload_strategies` maps a tool name, a tuple of names or `"*"` to `AlwaysOffload()`, `NeverOffload()` or `OffloadOverChars(n)`; only successful text results are offloaded. In a multi-user service, give each run its own store with `agent.run(..., hook_context={RESULT_STORE_CONTEXT_KEY: per_run_store})` (the key is importable from `haystack.hooks.tool_result_offloading`) so users never read each other's offloaded results.

### Agent Configuration and Customisation

#### Constructor reference

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `chat_generator` | required | Any chat generator that supports tools |
| `tools` | `None` | List of `Tool`/`Toolset`, or a single `Toolset` |
| `system_prompt` | `None` | Plain string or Jinja2 template |
| `user_prompt` | `None` | Jinja2 template appended as a user message; variables become inputs |
| `required_variables` | `"*"` | Which template variables are mandatory (`"*"`, a list, or `None`) |
| `exit_conditions` | `["text"]` | `"text"` and/or tool names that end the run |
| `state_schema` | `None` | Extra state keys: `{"name": {"type": T, "handler": fn}}` |
| `max_agent_steps` | `100` | Hard cap on loop iterations |
| `streaming_callback` | `None` | Called for each `StreamingChunk` |
| `raise_on_tool_invocation_failure` | `False` | Raise instead of reporting tool errors to the model |
| `tool_concurrency_limit` | `4` | Max tool calls executed in parallel within one step |
| `tool_streaming_callback_passthrough` | `False` | Pass the streaming callback into tools that accept one (e.g. `AgentTool`) |
| `hooks` | `None` | `{hook_point: [hook, ...]}` |

`generation_kwargs` (temperature, max tokens and so on) belong to the chat generator, either at construction or per run via `agent.run(..., generation_kwargs={...})`.

#### State and tools

Tools can read from and write to the agent's state. In 3.x, reading by name requires an explicit `inputs_from_state` mapping:

```python
from haystack.tools import tool


@tool(outputs_to_state={"notes": {"source": "notes"}})
def take_note(text: Annotated[str, "The note to save"]) -> dict:
    """Save a note for later."""
    return {"notes": [text]}


@tool(inputs_from_state={"notes": "notes"})
def summarise_notes(notes: list[str]) -> str:
    """Summarise the notes saved so far."""
    return "; ".join(notes) if notes else "No notes yet."


agent = Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    tools=[take_note, summarise_notes],
    state_schema={"notes": {"type": list[str]}},   # lists are merged by default
)

result = agent.run(messages=[ChatMessage.from_user("Note that the launch is on Friday, then summarise my notes.")])
print(result.get("notes", []))
```

A tool can also receive the whole `State` by declaring a parameter typed as `haystack.components.agents.State`. That parameter is left out of the tool's schema and always injected by the Agent.

`inputs_from_state` is a convenience, not a security boundary: if the model supplies an argument with the same name, the model's value wins and the state lookup is skipped. Read privileged values such as tenant or user IDs from a `State`-typed parameter instead (`state.get("tenant_id")`); the [multi-tenant recipe](/haystack-guide/haystack_recipes/#multi-tenant-customer-support-agent) shows the pattern.

The Agent reserves nine state names, which cannot appear in `state_schema`: `step_count`, `token_usage`, `tool_call_counts` and `exit_reason` (run metadata, returned as outputs), and `continue_run`, `stop_run`, `tools`, `hook_context` and `context_tokens` (internal keys that hooks read or set).

#### Exit conditions

```python
@tool
def final_answer(answer: Annotated[str, "The final answer"], confidence: Annotated[float, "0-1"]) -> dict:
    """Submit the final answer."""
    return {"answer": answer, "confidence": confidence}


agent = Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    tools=[search, final_answer],
    exit_conditions=["final_answer"],   # stop as soon as this tool has run
)
result = agent.run(messages=[ChatMessage.from_user("Research X and submit a final answer.")])
print(result["last_message"].tool_call_result.result)
```

#### Hooks

Hooks are callables that receive the live `State` and influence the run by mutating it. The `@hook` decorator turns a `(state) -> None` function into one:

```python
from haystack.components.agents import State
from haystack.hooks import hook


@hook
def require_save(state: State) -> None:
    """Refuse to finish until the `save` tool has been called."""
    if state.get("tool_call_counts", {}).get("save", 0) == 0:
        state.set("messages", [ChatMessage.from_system("Call `save` before finishing.")])
        state.set("continue_run", True)


@tool
def save(content: str) -> str:
    """Save content to durable storage."""
    return "Saved."


agent = Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    tools=[save],
    hooks={"on_exit": [require_save]},
)
```

Hook points are `before_run`, `before_llm`, `before_tool`, `after_tool`, `on_exit` and `after_run`. Request-scoped resources (a websocket, a user ID) go in `agent.run(..., hook_context={...})` and are read inside hooks with `state.get("hook_context")`.

#### Human-in-the-loop confirmation

Tool confirmation is a `before_tool` hook built from confirmation strategies:

```python
from haystack.hooks.human_in_the_loop import (
    AlwaysAskPolicy,
    BlockingConfirmationStrategy,
    ConfirmationHook,
    RichConsoleUI,     # needs `pip install rich`; SimpleConsoleUI has no extra dependency
)

confirm = ConfirmationHook(
    confirmation_strategies={
        "create_ticket": BlockingConfirmationStrategy(
            confirmation_policy=AlwaysAskPolicy(),
            confirmation_ui=RichConsoleUI(),
        )
    }
)

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    tools=[create_ticket],
    hooks={"before_tool": [confirm]},
)
```

`AskOncePolicy` (ask the first time a tool is used, then remember the answer) and `NeverAskPolicy` are also available.

For a web front end or an automated reviewer, write your own UI. A `ConfirmationUI` is any object with a `get_user_confirmation(tool_name, tool_description, tool_params)` method that returns a `ConfirmationUIResult` whose `action` is `"confirm"`, `"reject"` or `"modify"`. This one approves small refunds, trims medium ones to a limit and rejects large ones; the model is scripted with `MockChatGenerator` so the example runs offline:

```python
from typing import Annotated, Any

from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.hooks.human_in_the_loop import (
    AlwaysAskPolicy,
    BlockingConfirmationStrategy,
    ConfirmationHook,
    ConfirmationUIResult,
)
from haystack.tools import tool


@tool
def issue_refund(order_id: Annotated[str, "Order ID"], amount: Annotated[float, "Amount in GBP"]) -> str:
    """Issue a refund. This moves money."""
    return f"Refunded £{amount:.2f} for {order_id}"


class LimitUI:
    """Approves small refunds, trims medium ones to the limit and rejects large ones."""

    def __init__(self, limit: float = 100.0):
        self.limit = limit

    def get_user_confirmation(
        self, tool_name: str, tool_description: str, tool_params: dict[str, Any]
    ) -> ConfirmationUIResult:
        amount = tool_params["amount"]
        if amount <= self.limit:
            return ConfirmationUIResult(action="confirm")
        if amount <= 3 * self.limit:
            return ConfirmationUIResult(action="modify", new_tool_params={**tool_params, "amount": self.limit})
        return ConfirmationUIResult(action="reject", feedback="Refunds above 300 need a manager.")


hook = ConfirmationHook(confirmation_strategies={
    "issue_refund": BlockingConfirmationStrategy(confirmation_policy=AlwaysAskPolicy(), confirmation_ui=LimitUI()),
})

for amount in (50, 200, 900):
    llm = MockChatGenerator(responses=[
        ChatMessage.from_assistant(tool_calls=[
            ToolCall(tool_name="issue_refund", arguments={"order_id": "ORD-1", "amount": amount})
        ]),
        "Done.",
    ])
    agent = Agent(chat_generator=llm, tools=[issue_refund], hooks={"before_tool": [hook]})
    result = agent.run(messages=[ChatMessage.from_user("Please refund ORD-1.")])
    tool_message = next(m for m in result["messages"] if m.tool_call_result)
    print(amount, "->", tool_message.tool_call_result.result)
# 50 -> Refunded £50.00 for ORD-1
# 200 -> Refunded £100.00 for ORD-1
# 900 -> Tool execution for 'issue_refund' was rejected by the user. With user feedback: Refunds above 300 need a manager.
```

On a reject the tool does not run and `tool_call_counts` stays at zero for it; the model receives the rejection text as the tool result and can explain or try something else. On a modify the tool runs with `new_tool_params`. The protocol classes (`ConfirmationUI`, `ConfirmationPolicy`, `ConfirmationStrategy`) live in `haystack.hooks.human_in_the_loop.types` if you want to type-annotate against them. In a service, pass request-scoped objects such as a websocket through `hook_context`; a strategy receives them as `confirmation_strategy_context`, and `BlockingConfirmationStrategy` also has an async path for `Agent.run_async`.

### Error Handling and Resilience

Inside the agent loop:

- **Tool errors** are reported back to the model as error results so it can recover; set `raise_on_tool_invocation_failure=True` on the `Agent` to raise instead. For expected failures, return a helpful message from the tool (see `calculate` above).
- **`exit_reason`** in the result says why the run stopped: `"text"`, `"length"`, `"content_filter"`, `"max_agent_steps"`, the name of an exit-condition tool, or a reason a hook set with `stop_run` (for example `"token_budget_exceeded"`). Log it, and route on it downstream if you need to.

Timeouts and retries, provider fallback with `FallbackChatGenerator`, retrying around a run and dead-letter handling for pipelines are covered in one place: [Retries and fallbacks](/haystack-guide/haystack_observability_python/#retries-and-fallbacks).

---

## Part III: Agentic workflows and multi-agent systems

### Agentic AI Workflows: Production-First Design

Haystack's strength is that agents and deterministic components compose in the same graph. A common shape is *retrieve → agent → validate*, with the agent's `user_prompt` variables wired to upstream outputs:

```python
from haystack import Document, Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

store = InMemoryDocumentStore()
store.write_documents([Document(content="Q3 revenue grew 12% year on year.")])

analyst = Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    tools=[calculate],
    system_prompt="You are a financial analyst. Cite the documents you use.",
    user_prompt=(
        "Question: {{ query }}\n\nDocuments:\n"
        "{% for d in documents %}- {{ d.content }}\n{% endfor %}"
    ),
)

pipe = Pipeline()
pipe.add_component("retriever", InMemoryBM25Retriever(document_store=store, top_k=5))
pipe.add_component("analyst", analyst)
pipe.connect("retriever.documents", "analyst.documents")

query = "How fast did revenue grow in Q3?"
result = pipe.run({"retriever": {"query": query}, "analyst": {"query": query, "messages": []}})
print(result["analyst"]["last_message"].text)
```

`messages` is a mandatory input of `Agent`, so pass an empty list when the `user_prompt` supplies the turn.

#### Wrapping a pipeline as a component

`SuperComponent` hides a pipeline behind a simple interface, which is handy for reuse and for exposing the pipeline as a tool:

```python
from haystack import SuperComponent

qa = SuperComponent(
    pipeline=pipe,
    input_mapping={"query": ["retriever.query", "analyst.query"], "messages": ["analyst.messages"]},
    output_mapping={"analyst.last_message": "answer"},
)
print(qa.run(query="How fast did revenue grow?", messages=[])["answer"].text)
```

`PipelineTool` does the same but produces a `Tool` directly, with `input_mapping`/`output_mapping` and an explicit `name` and `description`.

#### Running concurrently and streaming

```python
import asyncio

async def main():
    # Independent branches run concurrently
    result = await pipe.run_async(
        {"retriever": {"query": query}, "analyst": {"query": query, "messages": []}},
        concurrency_limit=4,
    )

    # Stream chunks from every streaming-capable component
    handle = pipe.stream({"retriever": {"query": query}, "analyst": {"query": query, "messages": []}})
    async for chunk in handle:
        print(chunk.content, end="", flush=True)
    return handle.result

asyncio.run(main())
```

#### Iterative refinement loop

Pipelines may contain cycles. This one regenerates until a validator is satisfied, bounded by `max_runs_per_component`:

```python
from haystack import Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import BranchJoiner
from haystack.dataclasses import ChatMessage


@component
class LengthValidator:
    def __init__(self, max_words: int = 50):
        self.max_words = max_words

    @component.output_types(valid=str, feedback=list[ChatMessage])
    def run(self, replies: list[ChatMessage]):
        text = replies[0].text or ""
        if len(text.split()) <= self.max_words:
            return {"valid": text}
        return {"feedback": [
            ChatMessage.from_assistant(text),
            ChatMessage.from_user(f"Too long ({len(text.split())} words). Rewrite in under {self.max_words} words."),
        ]}


loop = Pipeline(max_runs_per_component=3)
loop.add_component("prompt", ChatPromptBuilder(template=[ChatMessage.from_user("Explain {{ topic }}.")]))
loop.add_component("joiner", BranchJoiner(list[ChatMessage]))
loop.add_component("llm", OpenAIChatGenerator(model="%%model=openai-small%%"))
loop.add_component("validator", LengthValidator(max_words=50))

loop.connect("prompt.prompt", "joiner")
loop.connect("joiner", "llm.messages")
loop.connect("llm.replies", "validator.replies")
loop.connect("validator.feedback", "joiner")      # the loop

print(loop.run({"prompt": {"topic": "vector databases"}})["validator"]["valid"])
```

If the validator keeps rejecting, the run does not end gracefully: once a component exceeds `max_runs_per_component`, `Pipeline.run` raises `PipelineMaxComponentRuns` (from `haystack.core.errors`). Catch it, or add a counter to the validator that emits `valid` after the last attempt.

### Multi-Agent Pipeline Design Patterns

The [multi-agent guide](/haystack-guide/haystack_multi_agent_guide/) covers these patterns in full; in brief:

- **Coordinator with agents as tools.** `AgentTool` turns an agent into a tool another agent can call, and the coordinator decides at run time when to delegate (example below).
- **Agents as pipeline stages.** When the order is fixed, connect one agent's `last_message` to a `user_prompt` variable of the next: [sequential collaboration](/haystack-guide/haystack_multi_agent_guide/#pattern-1-sequential-collaboration).
- **Routing to specialists.** `ConditionalRouter` (rules) or `LLMMessagesRouter` (a model classifies): [dynamic agent selection](/haystack-guide/haystack_multi_agent_guide/#dynamic-agent-selection).
- **Parallel fan-out.** `asyncio.gather` over `run_async`, or parallel pipeline branches with `Pipeline.run_async`: [parallel processing](/haystack-guide/haystack_multi_agent_guide/#pattern-2-parallel-processing).

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import AgentTool

researcher = Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    tools=[search],
    system_prompt="You research topics thoroughly and report findings with sources.",
)
writer = Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    system_prompt="You turn research notes into clear, well-structured prose.",
)

coordinator = Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    tools=[
        AgentTool(agent=researcher, name="researcher", description="Researches a topic and returns findings."),
        AgentTool(agent=writer, name="writer", description="Writes polished text from notes."),
    ],
    system_prompt="Plan the work, delegate to your specialists, then return the final article.",
)

result = coordinator.run(messages=[ChatMessage.from_user("Write a short briefing on solid-state batteries.")])
print(result["last_message"].text)
```

### SearchableToolset and SkillToolset

When an agent has dozens or hundreds of tools, sending every schema on every call wastes tokens and confuses the model. `SearchableToolset` wraps a catalogue and exposes a single `search_tools` tool; the agent searches by keyword and the matching tools are loaded for subsequent steps.

```python
from haystack.tools import SearchableToolset

catalog = [get_weather, search, compound_interest, calculate, create_ticket]   # any Tools/Toolsets

toolset = SearchableToolset(
    catalog=catalog,
    top_k=3,               # tools returned per search
    search_threshold=8,    # catalogues smaller than this are passed through unchanged
)

agent = Agent(chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"), tools=toolset)
```

`SkillToolset` does the same for instructions rather than tools. A skill is a directory with a `SKILL.md` file (YAML frontmatter with `name` and `description`, then markdown instructions) and optional bundled files. When the toolset warms up, every skill's name and description is written into the description of its `load_skill` tool, so the model knows what exists without a longer system prompt; `load_skill` returns a skill's full instructions and `read_skill_file` returns a bundled file.

```python
import tempfile
from pathlib import Path

from haystack.skill_stores.file_system import FileSystemSkillStore
from haystack.tools import SkillToolset

skills_dir = Path(tempfile.mkdtemp())          # normally a skills/ folder in your project
(skills_dir / "release-notes").mkdir()
(skills_dir / "release-notes" / "SKILL.md").write_text(
    "---\nname: release-notes\ndescription: How to write release notes for this project.\n---\n"
    "Group changes under Features, Fixes and Breaking changes, one line per change.\n"
)

skills = SkillToolset(FileSystemSkillStore(skills_dir))
agent = Agent(chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"), tools=[skills, search])
```

The tool names are fixed, so an agent can have only one `SkillToolset`; implement the `SkillStore` protocol to serve skills from a database or several sources. Bundled images and PDFs come back as content parts, which need a multimodal generator such as `OpenAIResponsesChatGenerator`.

### LLMRanker

`LLMRanker` re-ranks retrieved documents with a chat model, returning only the relevant ones in order:

```python
from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.rankers import LLMRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

pipeline = Pipeline()
pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store, top_k=20))
pipeline.add_component("ranker", LLMRanker(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    top_k=5,
))
pipeline.connect("retriever.documents", "ranker.documents")

result = pipeline.run({"retriever": {"query": "What is RAG?"}, "ranker": {"query": "What is RAG?"}})
for doc in result["ranker"]["documents"]:
    print(doc.content[:80])
```

The prompt is a Jinja2 template with `query` and `documents`; pass `prompt=` to customise it. For cross-encoder ranking, use `SentenceTransformersSimilarityRanker` from `sentence-transformers-haystack`.

---

## Migrating from Haystack 2.x

The upstream [migration guide](https://github.com/deepset-ai/haystack/blob/main/MIGRATION.md) is authoritative; the changes most likely to affect code in this guide's style are below.

| 2.x | 3.x |
|-----|-----|
| `from haystack.components.generators import OpenAIGenerator` | `OpenAIChatGenerator` (accepts a `str`; reply is a `ChatMessage`, use `.text`; metadata in `.meta`) |
| `llm.meta` output socket | Removed; metadata is on each `ChatMessage` |
| `from haystack.components.tools import ToolInvoker` | Removed; pass tools to `Agent` |
| `Agent(tool_invoker_kwargs={"max_workers": 4, ...})` | `tool_concurrency_limit=4`, `tool_streaming_callback_passthrough=True` |
| `agent.run(..., system_prompt=..., user_prompt=...)` | Prompts set only at construction |
| Template variables optional by default | Required by default (`required_variables="*"`) |
| Tool parameters silently filled from matching `State` keys | Declare `inputs_from_state` explicitly |
| `Agent(confirmation_strategies=...)`, `haystack.human_in_the_loop` | `hooks={"before_tool": [ConfirmationHook(...)]}`, `haystack.hooks.human_in_the_loop` |
| `AgentBreakpoint`, `ToolBreakpoint`, `Agent.run(break_point=...)` | Removed; use tracing. Pipeline-level `Breakpoint` remains |
| `from haystack import AsyncPipeline` | `Pipeline` (`run_async`, `run_async_generator`, `stream`) |
| `DALLEImageGenerator` | `OpenAIImageGenerator` |
| `TransformersSimilarityRanker` | `SentenceTransformersSimilarityRanker` (`sentence-transformers-haystack`) |
| `HuggingFaceLocalChatGenerator` | `TransformersChatGenerator` (`transformers-haystack`) |
| `HuggingFaceAPIChatGenerator`, `SentenceTransformers*Embedder`, `SerperDevWebSearch`, Whisper, `OpenAPIConnector` ... | Moved to integration packages; import from `haystack_integrations.*` |
| `haystack.tracing.OpenTelemetryTracer`, `DatadogTracer`; auto-enabled tracing | `opentelemetry-haystack` / `datadog-haystack`; enable explicitly |
| `haystack-experimental` installed with core | Install separately if you import `haystack_experimental` |
| `Pipeline.load()` imports any class | Allowlist: `allowed_modules=`, `allow_deserialization_module()`, `HAYSTACK_DESERIALIZATION_ALLOWLIST` |
| Importing Haystack reconfigured root logging | Scoped to Haystack loggers; `logging.configure_logging(logger_name="")` restores the old behaviour |
| `Document.id` hash depended on `meta` key order | Canonical JSON; IDs for documents with non-empty `meta` change |
| Missing API key raised at construction | Raised at `warm_up()` / first run |
| `GeneratedAnswer` / `ExtractedAnswer` `.to_dict()` wrapped fields in `{"type": ..., "init_parameters": {...}}` | Flat dict of fields; `from_dict()` accepts both formats |
| Confirmation strategies saw the fully prepared tool arguments | They see only the arguments the model requested; `inputs_from_state` values and `State` parameters are injected at execution |
| Python 3.10 supported | Python 3.11 or later from 3.4.0 |

### Before and after

```python
# Haystack 2.x
from haystack.components.generators import OpenAIGenerator
from haystack.components.tools import ToolInvoker

gen = OpenAIGenerator()
text = gen.run("What is NLP?")["replies"][0]
```

```python
# Haystack 3.x
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.agents import Agent

gen = OpenAIChatGenerator()
text = gen.run("What is NLP?")["replies"][0].text

agent = Agent(chat_generator=OpenAIChatGenerator(), tools=[get_weather])
```

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| %%versions-checked%% | %%version=haystack-ai%% | Rewritten for Haystack 3: every example now uses `Agent(chat_generator=..., ...)` and `agent.run(messages=...)`; removed fabricated APIs (`Agent(llm=...)`, `run(query=..., max_iterations=...)`, `Pipeline.from_yaml/to_yaml`, `haystack.cloud`, `.output`/`.input` sockets); integrations moved to `haystack_integrations.*`; `ToolInvoker` section replaced by migration notes; added state, hooks, HITL, `AgentTool`, `SuperComponent`, `stream()`; examples checked against the installed package with `MockChatGenerator`. Re-audited: added sections on document stores and retrievers, branching and concurrent pipelines, a runnable MCP server example and a custom human-in-the-loop UI; documented component serialisation behaviour. |
| 2026-10-21 | 3.4.1 | Python 3.11+; context management (`CompactionHook`, `ToolResultOffloadHook`, `TokenBudgetHook`), `SkillToolset`, images and files; `inputs_from_state` security note and the full reserved-keys list; multi-agent and resilience material each moved to one home; three migration rows added. |
| June 2026 | 2.30.0 | Last 2.x revision. |
| November 2025 | 2.20.0 | First documented version. |

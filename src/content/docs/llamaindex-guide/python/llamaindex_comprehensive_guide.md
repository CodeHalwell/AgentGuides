---
title: "LlamaIndex Comprehensive Technical Guide"
description: "Complete LlamaIndex (Python) reference — ingestion, indexing, retrieval, query engines, workflow-based agents, tools, memory, structured output, multi-agent workflows and evaluation."
framework: "llamaindex"
language: "python"
sidebar:
  label: "Comprehensive guide"
---

> Verified against `llama-index` / `llama-index-core` %%version=llama-index%% (versions checked %%versions-checked%%). Every import and constructor on this page was checked against an installed copy of that release.

LlamaIndex has two halves that are worth keeping separate in your head:

- **The data half** — readers, node parsers, ingestion pipelines, indexes, retrievers, post-processors and query engines. This is the part LlamaIndex is best known for, and it has been stable for a long time.
- **The agent half** — everything is now built on **Workflows** (the event-driven runtime in the `llama-index-workflows` package, re-exported as `llama_index.core.workflow`). The agent classes `FunctionAgent`, `ReActAgent`, `CodeActAgent` and the multi-agent `AgentWorkflow` all live in `llama_index.core.agent.workflow`.

If you learnt LlamaIndex before mid-2025, note that the older agent stack (`OpenAIAgent`, `FunctionCallingAgent`, `AgentRunner`, step workers, the old `ReActAgent.from_tools(...)` / `agent.chat(...)` API) and `QueryPipeline` were removed in `llama-index-core` 0.13.0. See [section 22](#22-migrating-from-the-legacy-agent-api).

---

## Table of contents

1. [Installation and setup](#1-installation-and-setup)
2. [Architecture overview](#2-architecture-overview)
3. [Core concepts](#3-core-concepts)
4. [Agent classes and types](#4-agent-classes-and-types)
5. [LLM and embedding configuration](#5-llm-and-embedding-configuration)
6. [Creating basic agents](#6-creating-basic-agents)
7. [Tool integration with agents](#7-tool-integration-with-agents)
8. [Query engine tools](#8-query-engine-tools)
9. [Context, state and memory](#9-context-state-and-memory)
10. [Structured output](#10-structured-output)
11. [Streaming responses](#11-streaming-responses)
12. [Multi-agent systems with AgentWorkflow](#12-multi-agent-systems-with-agentworkflow)
13. [Custom workflows](#13-custom-workflows)
14. [Serving and deploying workflows (LlamaAgents)](#14-serving-and-deploying-workflows-llamaagents)
15. [Human-in-the-loop](#15-human-in-the-loop)
16. [FunctionTool in detail](#16-functiontool-in-detail)
17. [LlamaHub tools and MCP](#17-llamahub-tools-and-mcp)
18. [Data connectors](#18-data-connectors)
19. [Advanced retrieval](#19-advanced-retrieval)
20. [Prompts](#20-prompts)
21. [Evaluation](#21-evaluation)
22. [Migrating from the legacy agent API](#22-migrating-from-the-legacy-agent-api)

---

## 1. Installation and setup

The `llama-index` package is a small convenience bundle: it installs `llama-index-core` plus the OpenAI LLM and embedding integrations. Everything else is a separate integration package (`llama-index-llms-*`, `llama-index-embeddings-*`, `llama-index-vector-stores-*`, `llama-index-readers-*`, `llama-index-tools-*`).

```bash
# Starter bundle (core + OpenAI LLM and embeddings)
pip install "llama-index==%%version=llama-index%%"

# Or build up from core yourself
pip install "llama-index-core==%%version=llama-index%%" llama-index-llms-openai llama-index-embeddings-openai
```

`llama-index` and `llama-index-core` are released in lockstep and share a version number. Integration packages have their own version numbers; let pip resolve them against the core you pinned rather than pinning them by hand.

Python 3.10 or newer is required.

**Typical extras by use case**

```bash
# PDF/DOCX/etc. support for SimpleDirectoryReader
pip install llama-index-readers-file

# RAG with a persistent vector store
pip install llama-index-vector-stores-chroma chromadb

# Other model providers
pip install llama-index-llms-anthropic llama-index-llms-ollama llama-index-embeddings-huggingface

# Hybrid (BM25 + vector) retrieval
pip install llama-index-retrievers-bm25

# Use MCP servers as agent tools
pip install llama-index-tools-mcp

# Draw workflow graphs
pip install llama-index-utils-workflow
```

**Verify the install**

```python
import llama_index.core
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.llms.openai import OpenAI

print(llama_index.core.__version__)
```

Set `OPENAI_API_KEY` in the environment (or pass `api_key=` to the LLM and embedding classes).

---

## 2. Architecture overview

```
┌─────────────────────────────────────────────────────────┐
│ Application layer                                       │
│   FastAPI / CLI / notebooks / llamactl-served apps      │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│ Orchestration: Workflows                                │
│   Workflow, @step, Context, events                      │
│   FunctionAgent · ReActAgent · CodeActAgent             │
│   AgentWorkflow (multi-agent hand-off)                  │
└────────────────────┬────────────────────────────────────┘
                     │ tools
┌────────────────────┴────────────────────────────────────┐
│ Query layer                                             │
│   Query engines · chat engines · routers                │
│   Retrievers · node post-processors · synthesisers      │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│ Index & storage                                         │
│   VectorStoreIndex · SummaryIndex · PropertyGraphIndex  │
│   StorageContext: docstore, index store, vector store   │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│ Ingestion                                               │
│   Readers → Documents → node parsers → Nodes            │
│   IngestionPipeline (transformations, caching, dedupe)  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│ Models (global Settings or per component)               │
│   LLMs · embedding models · tokenisers                  │
└─────────────────────────────────────────────────────────┘
```

| Concept | What it is |
|---|---|
| `Document` | A unit of source data (a file, a page, a row) with text and metadata |
| `Node` (`TextNode`) | A chunk of a document — the unit that gets embedded and retrieved |
| Index | A structure over nodes; `VectorStoreIndex` is the default |
| Retriever | Given a query, returns `NodeWithScore` objects |
| Query engine | Retriever + post-processors + response synthesiser |
| Tool | A callable an agent may choose to invoke (`FunctionTool`, `QueryEngineTool`) |
| Workflow | An event-driven async program made of `@step` methods |
| Agent | A prebuilt workflow that loops LLM → tool calls → LLM |

---

## 3. Core concepts

### 3.1 Documents

```python
from llama_index.core import Document, SimpleDirectoryReader

# Build documents by hand
doc = Document(
    text="LlamaIndex is a framework for building LLM applications over your data.",
    metadata={"source": "manual", "topic": "overview"},
)

# Load a folder (txt, md, pdf, docx, csv, ... depending on installed readers)
documents = SimpleDirectoryReader("data", recursive=True).load_data()

# Control what the LLM and the embedding model see
doc.excluded_llm_metadata_keys = ["source"]
doc.excluded_embed_metadata_keys = ["source"]
```

### 3.2 Nodes

Nodes are chunks of documents. You rarely build them by hand, but it is useful to know their shape.

```python
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

node1 = TextNode(text="First paragraph.", id_="node-1", metadata={"page": 1})
node2 = TextNode(text="Second paragraph.", id_="node-2", metadata={"page": 1})

node1.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=node2.node_id)
node2.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=node1.node_id)

print(node1.node_id, node1.metadata, node1.get_content())
```

### 3.3 Node parsers and the ingestion pipeline

```python
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        OpenAIEmbedding(model="text-embedding-3-small"),
    ],
)

nodes = pipeline.run(documents=[Document(text="A long document ...")])
print(len(nodes), nodes[0].embedding is not None)
```

Attach a `docstore` and a `vector_store` to the pipeline to get de-duplication and upserts across runs; attach a `cache` to avoid re-embedding unchanged chunks.

### 3.4 Indexes

```python
from llama_index.core import VectorStoreIndex, SummaryIndex, Document

documents = [
    Document(text="LlamaIndex is a data framework for LLM applications."),
    Document(text="RAG grounds an LLM's answer in retrieved context."),
]

# Embedding-based semantic search — the default choice
vector_index = VectorStoreIndex.from_documents(documents)

# Returns every node (optionally LLM-filtered) — good for "summarise all of this"
summary_index = SummaryIndex.from_documents(documents)

# Build directly from nodes you produced with an IngestionPipeline
# vector_index = VectorStoreIndex(nodes)
```

`ListIndex` was the old name for `SummaryIndex`; use `SummaryIndex`. `TreeIndex` and `KeywordTableIndex` still exist but are rarely the right choice today. For graph-shaped data use `PropertyGraphIndex` ([section 3.9](#39-property-graphs)); for images alongside text, `MultiModalVectorStoreIndex` ([section 3.10](#310-multimodal-documents-and-queries)).

### 3.5 Persistence

```python
from llama_index.core import StorageContext, load_index_from_storage

vector_index.storage_context.persist(persist_dir="./storage")

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

With an external vector store the embeddings live in the store, and you reconnect with `VectorStoreIndex.from_vector_store(vector_store)`:

```python
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("docs")
vector_store = ChromaVectorStore(chroma_collection=collection)

# First run: build
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Later runs: reconnect
index = VectorStoreIndex.from_vector_store(vector_store)
```

### 3.6 Retrievers

```python
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

retriever = vector_index.as_retriever(
    similarity_top_k=5,
    filters=MetadataFilters(filters=[ExactMatchFilter(key="topic", value="overview")]),
)

for hit in retriever.retrieve("What is LlamaIndex?"):
    print(f"{hit.score:.3f}  {hit.node.get_content()[:80]}")
```

### 3.7 Query engines

```python
query_engine = vector_index.as_query_engine(
    similarity_top_k=3,
    response_mode="compact",   # also: "refine", "tree_summarize", "accumulate", ...
)

response = query_engine.query("What is LlamaIndex?")
print(response.response)
for source in response.source_nodes:
    print("-", source.node.get_content()[:60], source.score)

# Async variant
# response = await query_engine.aquery("What is LlamaIndex?")
```

Token streaming from a query engine:

```python
streaming_engine = vector_index.as_query_engine(streaming=True)
streaming_response = streaming_engine.query("Summarise the documents")
for token in streaming_response.response_gen:
    print(token, end="", flush=True)
```

### 3.8 Chat engines

For a conversational RAG bot that does not need tools, a chat engine is simpler than an agent. Since 0.13.0 the default `as_chat_engine()` returns a `CondensePlusContextChatEngine`:

```python
chat_engine = vector_index.as_chat_engine(chat_mode="condense_plus_context")
print(chat_engine.chat("What is LlamaIndex?"))
print(chat_engine.chat("And how does it relate to RAG?"))
chat_engine.reset()
```

### 3.9 Property graphs

`PropertyGraphIndex` extracts entities and relations from each chunk with "knowledge graph extractors", stores them as a labelled property graph next to the text, and retrieves by following graph paths as well as by vector similarity. Use it when questions are about relationships ("who worked with whom", "which systems depend on this service"), not only about passages.

```python
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import ImplicitPathExtractor, SimpleLLMPathExtractor
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="%%model=openai-small%%")
graph_index = PropertyGraphIndex.from_documents(
    documents,
    llm=llm,                          # used by the LLM-based extractor
    kg_extractors=[
        SimpleLLMPathExtractor(llm=llm, max_paths_per_chunk=10),   # (subject, relation, object) triples
        ImplicitPathExtractor(),                                   # links from node relationships, no LLM
    ],
)

retriever = graph_index.as_retriever(include_text=True)   # matched triples plus their source text
for hit in retriever.retrieve("Who did Ada Lovelace work with?"):
    print(hit.text[:80])

graph_engine = graph_index.as_query_engine(include_text=True)
graph_index.storage_context.persist(persist_dir="./graph_storage")   # reload with load_index_from_storage
```

`SchemaLLMPathExtractor` restricts extraction to entity and relation types you list, which gives a cleaner graph than free-form triples. The default store is in-memory (`SimplePropertyGraphStore`); for real data use a graph database integration such as `Neo4jPropertyGraphStore` (`llama-index-graph-stores-neo4j`), passed as `property_graph_store=`. Extraction makes one LLM call per chunk, so building the index costs far more than a vector index over the same documents.

### 3.10 Multimodal documents and queries

Since 0.14.23 the standard query engines can synthesise answers from images as well as text: pass `multimodal=True` and an LLM that accepts image content. `MultiModalVectorStoreIndex` embeds images with a separate image embedding model and stores them beside the text.

```python
# pip install llama-index-readers-file llama-index-embeddings-clip git+https://github.com/openai/CLIP.git
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.clip import ClipEmbedding

documents = SimpleDirectoryReader("./report").load_data()   # .png/.jpg files become ImageDocuments
mm_index = MultiModalVectorStoreIndex.from_documents(documents, image_embed_model=ClipEmbedding())

engine = RetrieverQueryEngine.from_args(
    retriever=mm_index.as_retriever(similarity_top_k=3, image_similarity_top_k=2),
    llm=OpenAI(model="%%model=openai%%"),                    # must accept image input
    multimodal=True,
)
print(engine.query("What does the revenue chart show?"))
```

Calling `mm_index.as_query_engine()` instead returns `SimpleMultiModalQueryEngine`, which is deprecated in favour of this form. Any `MultiModalEmbedding` works as `image_embed_model`; CLIP runs locally and needs PyTorch. To send an image to an LLM directly, put it in a message as a content block:

```python
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock

msg = ChatMessage(
    role="user",
    blocks=[ImageBlock(path="./report/chart.png"), TextBlock(text="Describe this chart.")],
)
print(llm.chat([msg]).message.content)
```

For PDFs and slides where the images are embedded in pages, parsing with LlamaParse ([section 18.2](#182-llamaparse-and-llamacloud)) usually works better than extracting the images yourself.

---

## 4. Agent classes and types

All agents are workflows. They share one calling convention: `handler = agent.run(user_msg=..., ctx=..., memory=...)` returns a `WorkflowHandler`; `await handler` gives an `AgentOutput`, and `handler.stream_events()` yields intermediate events.

| Class | Use it when |
|---|---|
| `FunctionAgent` | The LLM supports native tool/function calling (OpenAI, Anthropic, Gemini, Mistral, most hosted models). The default choice. |
| `ReActAgent` | The LLM has no native tool calling, or you want an explicit Thought/Action/Observation trace. Works with any LLM. |
| `CodeActAgent` | The agent should write and execute Python to solve the task. You supply the `code_execute_fn` sandbox. |
| `AgentWorkflow` | Several agents hand control to each other, sharing state. |

All live in `llama_index.core.agent.workflow` (and are re-exported from `llama_index.core.agent`).

### 4.1 FunctionAgent

```python
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

agent = FunctionAgent(
    tools=[multiply, add],                 # plain functions are wrapped as FunctionTools
    llm=OpenAI(model="%%model=openai-small%%"),
    system_prompt="You are a careful calculator. Always use the tools.",
)

async def main():
    response = await agent.run(user_msg="What is 7 * 8 + 12?")
    print(str(response))

asyncio.run(main())
```

Model names in this guide (`%%model=openai-small%%`, `text-embedding-3-small`) are examples; use whatever your provider currently offers.

### 4.2 ReActAgent

```python
from llama_index.core.agent.workflow import ReActAgent

react_agent = ReActAgent(
    tools=[multiply, add],
    llm=OpenAI(model="%%model=openai-small%%"),
)

# inside an async function
response = await react_agent.run(user_msg="What is 7 * 8 + 12?")
```

`ReActAgent` is **not** removed — only the pre-0.13 implementation (`ReActAgent.from_tools(...)`, `.chat()`) is. The current class is constructed directly and run with `.run()`.

### 4.3 CodeActAgent

```python
from llama_index.core.agent.workflow import CodeActAgent

async def execute_code(code: str) -> str:
    """Run code in YOUR sandbox (container, Jupyter kernel, remote executor) and return stdout."""
    raise NotImplementedError("Wire this to a real sandbox; never exec() untrusted code in-process.")

code_agent = CodeActAgent(
    code_execute_fn=execute_code,
    tools=[multiply, add],
    llm=OpenAI(model="%%model=openai-small%%"),
)
```

### 4.4 AgentWorkflow

`AgentWorkflow` coordinates several named agents. Each agent declares which others it may hand off to; see [section 12](#12-multi-agent-systems-with-agentworkflow). For a single agent you can also write:

```python
from llama_index.core.agent.workflow import AgentWorkflow

workflow = AgentWorkflow.from_tools_or_functions(
    [multiply, add],
    llm=OpenAI(model="%%model=openai-small%%"),
    system_prompt="You are a careful calculator.",
)
response = await workflow.run(user_msg="What is 3 * 4?")
```

There is no `AgentWorkflow.from_tools(...)` and `AgentWorkflow` is not importable from `llama_index.core.workflow`.

---

## 5. LLM and embedding configuration

### 5.1 LLMs

```python
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic      # pip install llama-index-llms-anthropic
from llama_index.llms.google_genai import GoogleGenAI  # pip install llama-index-llms-google-genai
from llama_index.llms.ollama import Ollama            # pip install llama-index-llms-ollama

openai_llm = OpenAI(model="%%model=openai-small%%")
claude_llm = Anthropic(model="%%model=anthropic%%")
gemini_llm = GoogleGenAI(model="%%model=gemini%%")
local_llm = Ollama(model="llama3.1", request_timeout=120.0)

print(openai_llm.complete("Say hello in one word."))
```

Chat-style calls use `ChatMessage`:

```python
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(role="system", content="You are terse."),
    ChatMessage(role="user", content="What is RAG?"),
]
print(openai_llm.chat(messages))

# Async and streaming variants: achat, astream_chat, stream_complete, ...
```

### 5.2 Embeddings

```python
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # pip install llama-index-embeddings-huggingface

openai_embed = OpenAIEmbedding(model="text-embedding-3-small")
local_embed = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

vector = openai_embed.get_text_embedding("hello")
batch = openai_embed.get_text_embedding_batch(["a", "b", "c"])
```

### 5.3 Global `Settings`

`Settings` replaced the old `ServiceContext`. Anything you set here is the default for indexes, query engines and agents that are not given an explicit model.

```python
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

Settings.llm = OpenAI(model="%%model=openai-small%%")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
```

Prefer passing models explicitly in libraries and tests; use `Settings` in applications and notebooks.

### 5.4 Offline testing with mock models

```python
from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.llms.mock import MockFunctionCallingLLM

Settings.embed_model = MockEmbedding(embed_dim=8)
Settings.llm = MockLLM()

# A tool-calling mock suitable for exercising FunctionAgent plumbing
test_llm = MockFunctionCallingLLM()
```

---

## 6. Creating basic agents

### 6.1 Minimal agent

```python
import asyncio
from datetime import datetime
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

def get_current_time() -> str:
    """Return the current local time as an ISO-8601 string."""
    return datetime.now().isoformat(timespec="seconds")

def add_hours(iso_time: str, hours: int) -> str:
    """Add a number of hours to an ISO-8601 timestamp."""
    from datetime import timedelta
    return (datetime.fromisoformat(iso_time) + timedelta(hours=hours)).isoformat()

agent = FunctionAgent(
    tools=[get_current_time, add_hours],
    llm=OpenAI(model="%%model=openai-small%%"),
    system_prompt="You answer questions about time. Use the tools; do not guess.",
)

async def main():
    response = await agent.run(user_msg="What time will it be in 5 hours?")
    print(response)

asyncio.run(main())
```

The docstring and type hints become the tool description and JSON schema, so write them for the model.

### 6.2 Multi-turn conversations

An agent keeps no memory between `run()` calls unless you give it a `Context` (or a `Memory`):

```python
from llama_index.core.workflow import Context

ctx = Context(agent)

async def chat():
    print(await agent.run(user_msg="My name is Ada.", ctx=ctx))
    print(await agent.run(user_msg="What is my name?", ctx=ctx))  # remembers "Ada"
```

### 6.3 Iteration limits and timeouts

```python
response = await agent.run(user_msg="...", max_iterations=10)

agent = FunctionAgent(tools=[multiply, add], llm=llm, timeout=120)   # workflow timeout in seconds
```

`early_stopping_method="generate"` asks the LLM for a final answer when the limit is reached instead of raising.

### 6.4 Inspecting the result

```python
result = await agent.run(user_msg="What is 6 * 7?")
print(result.response.content)     # final assistant message
for call in result.tool_calls:      # ToolCallResult objects
    print(call.tool_name, call.tool_kwargs, call.tool_output.content)
```

---

## 7. Tool integration with agents

### 7.1 Kinds of tool

```python
from llama_index.core.tools import FunctionTool, QueryEngineTool

def web_search(query: str) -> str:
    """Search the web and return a short list of results."""
    return f"Results for {query!r}: ..."

search_tool = FunctionTool.from_defaults(fn=web_search)

kb_tool = QueryEngineTool.from_defaults(
    query_engine=vector_index.as_query_engine(),
    name="knowledge_base",
    description="Answers questions about the internal product documentation.",
)

agent = FunctionAgent(
    tools=[search_tool, kb_tool, multiply],   # FunctionTool, QueryEngineTool and a bare function
    llm=OpenAI(model="%%model=openai-small%%"),
)
```

There is no generic `Tool` class in `llama_index.core.tools`; use `FunctionTool`, `QueryEngineTool`, `RetrieverTool`, or a `ToolSpec`'s `.to_tool_list()`.

### 7.2 Tool chaining

The agent loop chains tools for you: it calls one, reads the result, and decides on the next call. Keep each tool small and deterministic, and let the LLM do the sequencing.

```python
def extract_entities(text: str) -> list[str]:
    """Extract organisation names from text."""
    return ["Acme Ltd", "Globex"]

def lookup_company(name: str) -> dict:
    """Look up basic facts about a company by exact name."""
    return {"name": name, "employees": 1200}

agent = FunctionAgent(
    tools=[extract_entities, lookup_company],
    llm=OpenAI(model="%%model=openai-small%%"),
    system_prompt="Extract entities first, then look each one up, then summarise.",
)
```

### 7.3 Parallel tool calls

`FunctionAgent` allows the LLM to emit several tool calls in one turn (`allow_parallel_tool_calls=True` by default). Set it to `False` when tools have side effects that must be ordered.

### 7.4 Returning directly

`FunctionTool.from_defaults(fn=..., return_direct=True)` ends the agent loop with that tool's output — useful for "lookup and return verbatim" tools.

---

## 8. Query engine tools

### 8.1 One knowledge base

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.tools import QueryEngineTool

docs = [
    Document(text="Machine learning is a subset of AI ..."),
    Document(text="Deep learning uses multi-layer neural networks ..."),
]
index = VectorStoreIndex.from_documents(docs)

kb_tool = QueryEngineTool.from_defaults(
    query_engine=index.as_query_engine(similarity_top_k=3),
    name="ai_knowledge_base",
    description="Definitions and background on AI, ML and deep learning.",
)

agent = FunctionAgent(tools=[kb_tool], llm=OpenAI(model="%%model=openai-small%%"))
response = await agent.run(user_msg="How does deep learning relate to ML?")
```

### 8.2 Several knowledge bases

Give each tool a precise name and a description that says what is *in* it — that is what the LLM routes on.

```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata

tools = [
    QueryEngineTool(
        query_engine=hr_index.as_query_engine(),
        metadata=ToolMetadata(name="hr_policies", description="Leave, benefits and HR policy documents."),
    ),
    QueryEngineTool(
        query_engine=eng_index.as_query_engine(),
        metadata=ToolMetadata(name="engineering_handbook", description="Coding standards, on-call, deployment process."),
    ),
]
agent = FunctionAgent(tools=tools, llm=OpenAI(model="%%model=openai-small%%"))
```

### 8.3 Retriever tools

When you want the agent to see raw chunks rather than a synthesised answer, wrap a retriever:

```python
from llama_index.core.tools import RetrieverTool

retriever_tool = RetrieverTool.from_defaults(
    retriever=index.as_retriever(similarity_top_k=5),
    name="search_docs",
    description="Return the most relevant passages from the documentation.",
)
```

---

## 9. Context, state and memory

### 9.1 `Context`

`Context` holds a workflow run's state: the agent's chat memory, any values you store, and the event stream. Reuse one `Context` per conversation while the process is alive:

```python
from llama_index.core.workflow import Context

ctx = Context(agent)
await agent.run(user_msg="Remember that my favourite colour is green.", ctx=ctx)
await agent.run(user_msg="What is my favourite colour?", ctx=ctx)   # remembers
```

A context can be serialised with `ctx.to_dict(serializer=JsonSerializer())` and restored with `Context.from_dict(agent, data, serializer=JsonSerializer())`, which keeps your own state keys. **It does not keep the prebuilt agents' chat history**: the `memory` entry is skipped during serialisation (you will see an `UnserializableKeyWarning`). To persist conversations across processes, use a database-backed `Memory` (section 9.3) and pass it on every `run()`.

### 9.2 Reading and writing state from tools

A tool whose first parameter is annotated `ctx: Context` receives the running context automatically; the parameter is hidden from the LLM.

```python
from llama_index.core.workflow import Context

async def add_to_cart(ctx: Context, item: str) -> str:
    """Add an item to the user's shopping cart."""
    async with ctx.store.edit_state() as state:
        state.setdefault("cart", []).append(item)
    return f"Added {item}."

async def view_cart(ctx: Context) -> str:
    """Show the shopping cart."""
    cart = await ctx.store.get("cart", default=[])
    return ", ".join(cart) or "Cart is empty."

shop_agent = FunctionAgent(tools=[add_to_cart, view_cart], llm=OpenAI(model="%%model=openai-small%%"))
ctx = Context(shop_agent)
```

Inside an `AgentWorkflow`, shared state seeded with `initial_state=` lives under the `"state"` key (`state["state"]`) and is also rendered into the prompt via `state_prompt`.

### 9.3 `Memory`

`Memory` (in `llama_index.core.memory`) is the current memory class: a token-bounded short-term chat history plus optional long-term **memory blocks**. `ChatMemoryBuffer` still exists but `Memory` is preferred.

```python
from llama_index.core.memory import Memory, StaticMemoryBlock, FactExtractionMemoryBlock, VectorMemoryBlock
from llama_index.core.llms import ChatMessage

memory = Memory.from_defaults(
    session_id="user-123",
    token_limit=40_000,
    memory_blocks=[
        StaticMemoryBlock(name="profile", static_content="The user is a data engineer in London."),
        FactExtractionMemoryBlock(name="facts", llm=OpenAI(model="%%model=openai-small%%"), max_facts=50),
    ],
)

response = await agent.run(user_msg="Hi!", memory=memory)
```

By default the chat history lives in an in-memory SQLite database. Pass `async_database_uri=` (for example `postgresql+asyncpg://...` or `sqlite+aiosqlite:///memory.db`) and a stable `session_id` to persist it; a new `Memory` object built with the same two values on the next request picks the conversation up where it left off:

```python
async def handle_turn(user_id: str, message: str) -> str:
    memory = Memory.from_defaults(
        session_id=user_id,
        async_database_uri="postgresql+asyncpg://app:secret@db/agents",
    )
    return str(await agent.run(user_msg=message, memory=memory))
```

---

## 10. Structured output

### 10.1 From an agent

```python
from pydantic import BaseModel, Field

class Calculation(BaseModel):
    expression: str
    result: float
    explanation: str = Field(description="One sentence explaining the steps")

agent = FunctionAgent(
    tools=[multiply, add],
    llm=OpenAI(model="%%model=openai-small%%"),
    output_cls=Calculation,
)

result = await agent.run(user_msg="Compute (15 * 3) + 9")
calc = result.get_pydantic_model(Calculation)   # or result.structured_response (a dict)
print(calc.result)
```

### 10.2 From an LLM directly

```python
from pydantic import BaseModel
from llama_index.core.prompts import PromptTemplate

class Employee(BaseModel):
    name: str
    title: str
    start_year: int

llm = OpenAI(model="%%model=openai-small%%")

employee = llm.structured_predict(
    Employee,
    PromptTemplate("Extract the employee from: {text}"),
    text="Jane Doe joined as Staff Engineer in 2021.",
)

# Or wrap the LLM so every chat/complete call returns the model
sllm = llm.as_structured_llm(Employee)
resp = sllm.complete("Jane Doe joined as Staff Engineer in 2021.")
print(resp.raw)   # Employee(...)
```

### 10.3 Programs and output parsers

For LLMs without tool calling, `LLMTextCompletionProgram` uses a prompt plus a Pydantic output parser:

```python
from llama_index.core.program import LLMTextCompletionProgram

program = LLMTextCompletionProgram.from_defaults(
    output_cls=Employee,
    prompt_template_str="Extract the employee as JSON from: {text}",
    llm=llm,
)
employee = program(text="Jane Doe joined as Staff Engineer in 2021.")
```

### 10.4 From a query engine

```python
query_engine = index.as_query_engine(output_cls=Employee, response_mode="compact")
structured = query_engine.query("Who is the staff engineer?").response  # an Employee
```

---

## 11. Streaming responses

### 11.1 Token streaming from an agent

```python
from llama_index.core.agent.workflow import AgentStream

handler = agent.run(user_msg="Explain RAG in three sentences.")
async for event in handler.stream_events():
    if isinstance(event, AgentStream):
        print(event.delta, end="", flush=True)

final = await handler
```

### 11.2 Event-based streaming

```python
from llama_index.core.agent.workflow import AgentInput, AgentOutput, ToolCall, ToolCallResult

handler = agent.run(user_msg="What is (3 * 4) + 5?")
async for event in handler.stream_events():
    if isinstance(event, ToolCall):
        print(f"→ calling {event.tool_name}({event.tool_kwargs})")
    elif isinstance(event, ToolCallResult):
        print(f"← {event.tool_name} returned {event.tool_output.content}")
    elif isinstance(event, AgentOutput):
        print(f"[{event.current_agent_name}] {event.response.content}")

result = await handler
```

For logging and tracing rather than UI updates, use the instrumentation module instead of the event stream — see the [observability page](/llamaindex-guide/python/llamaindex_observability_python/).

---

## 12. Multi-agent systems with AgentWorkflow

Earlier editions of this guide described the 2024 `llama-agents` package (`ControlPlaneServer`, `AgentService`, message queues). That package was retired; its successor `llama-deploy` has in turn been folded into **LlamaAgents** and the `llamactl` CLI (see [section 14](#14-serving-and-deploying-workflows-llamaagents)). In-process multi-agent orchestration is done with `AgentWorkflow`.

### 12.1 Hand-off between specialists

```python
import asyncio
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent, AgentOutput, ToolCallResult
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="%%model=openai-small%%")

async def search_web(query: str) -> str:
    """Search the web for information on a topic."""
    return f"(search results for {query})"

async def record_notes(ctx: Context, notes: str, title: str) -> str:
    """Save research notes under a title for the other agents."""
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["research_notes"][title] = notes
    return "Notes recorded."

async def write_report(ctx: Context, report: str) -> str:
    """Write the final report in markdown."""
    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["report"] = report
    return "Report written."

research_agent = FunctionAgent(
    name="ResearchAgent",
    description="Searches the web and records notes on a topic.",
    system_prompt="Research the topic, record notes, then hand off to WriteAgent.",
    tools=[search_web, record_notes],
    llm=llm,
    can_handoff_to=["WriteAgent"],
)

write_agent = FunctionAgent(
    name="WriteAgent",
    description="Writes a markdown report from the research notes.",
    system_prompt="Write a concise report from the notes in the state.",
    tools=[write_report],
    llm=llm,
    can_handoff_to=["ResearchAgent"],
)

workflow = AgentWorkflow(
    agents=[research_agent, write_agent],
    root_agent=research_agent.name,
    initial_state={"research_notes": {}, "report": None},
)

async def main():
    handler = workflow.run(user_msg="Write a short report on retrieval-augmented generation.")
    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):
            print(f"[tool] {event.tool_name}")
        elif isinstance(event, AgentOutput) and event.response.content:
            print(f"[{event.current_agent_name}] {event.response.content[:80]}")
    await handler
    state = await handler.ctx.store.get("state")
    print(state["report"])

asyncio.run(main())
```

How hand-off works: `AgentWorkflow` gives every agent a `handoff` tool listing the agents in its `can_handoff_to`. When the LLM calls it, control (and the shared memory and state) passes to the named agent.

### 12.2 Orchestrator pattern: agents as tools

When you want one agent firmly in charge, wrap the others as tools instead of using hand-off:

```python
async def call_research_agent(ctx: Context, prompt: str) -> str:
    """Ask the research agent to investigate something and return its findings."""
    result = await research_agent.run(user_msg=prompt)
    return str(result)

orchestrator = FunctionAgent(
    system_prompt="Plan the work and delegate research before answering.",
    tools=[call_research_agent],
    llm=llm,
)
```

### 12.3 Choosing a pattern

| Pattern | Control | Good for |
|---|---|---|
| `AgentWorkflow` hand-off | Decentralised: the active agent decides who goes next | Conversational assistants with specialist modes |
| Orchestrator with agents-as-tools | One agent plans and delegates | Research/report tasks with a clear owner |
| Custom `Workflow` | You write the control flow in Python | Fixed pipelines, fan-out/fan-in, strict ordering |

---

## 13. Custom workflows

When the control flow is yours rather than the model's — a fixed pipeline, fan-out and fan-in, strict ordering, approval gates — write a `Workflow`. A workflow is a class whose `@step` methods consume and emit typed events; the runtime routes events by the types in the step signatures and validates the graph before running. The agent classes above are themselves workflows.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.llms.openai import OpenAI

class RetrievedEvent(Event):
    query: str
    context: str

class RAGWorkflow(Workflow):
    def __init__(self, index: VectorStoreIndex, **kwargs):
        super().__init__(**kwargs)
        self.retriever = index.as_retriever(similarity_top_k=3)
        self.llm = OpenAI(model="%%model=openai-small%%")

    @step
    async def retrieve(self, ev: StartEvent) -> RetrievedEvent:
        nodes = await self.retriever.aretrieve(ev.query)
        return RetrievedEvent(query=ev.query, context="\n\n".join(n.get_content() for n in nodes))

    @step
    async def synthesise(self, ev: RetrievedEvent) -> StopEvent:
        prompt = f"Answer using only this context.\n\n{ev.context}\n\nQuestion: {ev.query}"
        return StopEvent(result=str(await self.llm.acomplete(prompt)))

async def main():
    wf = RAGWorkflow(index=vector_index, timeout=60)
    print(await wf.run(query="What is LlamaIndex?"))
```

Workflows come from the separate `llama-index-workflows` package (imported as `workflows`), which `llama_index.core.workflow` re-exports. The **[Workflows page](/llamaindex-guide/python/llamaindex_workflows/)** covers it in full: events and typed start/stop events, `Context` and typed state, `send_event` / `collect_events` and list fan-out, streaming, human-in-the-loop, retry policies and `@catch_error`, timeouts, resources, checkpointing and resuming, nested workflows, drawing and testing. [Workflow patterns](/llamaindex-guide/python/llamaindex_advanced_implementations/) has longer worked examples.

---

## 14. Serving and deploying workflows (LlamaAgents)

A workflow or agent is an ordinary async object, so you can call it from any web framework (the [FastAPI streaming page](/llamaindex-guide/python/llamaindex_streaming_server_fastapi/) does that). LlamaIndex also ships its own serving layer, **LlamaAgents**:

- **`WorkflowServer`** (`pip install llama-agents-server`, `from llama_agents.server import WorkflowServer`) wraps registered workflows in a REST API: start runs, stream their events, send events in (for example a human approval), fetch results, cancel. With a durable workflow store, unfinished runs resume after a restart.
- **`WorkflowClient`** (`pip install llama-agents-client`, `from llama_agents.client import WorkflowClient`) calls that API from Python.
- **`llamactl`** scaffolds a project from templates, serves its `WorkflowServer` locally with reload, packages it as a container, and deploys it to LlamaCloud.

The older import paths `workflows.server` and `workflows.client` (the `[server]` and `[client]` extras of `llama-index-workflows`) still work but emit a deprecation warning pointing to the packages above.

```python
from llama_agents.server import WorkflowServer

server = WorkflowServer()                    # in-memory store; see the production guide for SQLite
server.add_workflow("calculator", agent)     # any Workflow, including FunctionAgent / AgentWorkflow
server.add_workflow("rag", RAGWorkflow(index=vector_index, timeout=60))

if __name__ == "__main__":
    import asyncio
    asyncio.run(server.serve(host="0.0.0.0", port=8080))
```

```bash
curl -X POST localhost:8080/workflows/calculator/run \
  -H 'Content-Type: application/json' \
  -d '{"start_event": {"user_msg": "What is 7 * 8?", "chat_history": [], "max_iterations": 20}}'
```

For agents, send `chat_history` and `max_iterations` as well as `user_msg`; with `user_msg` alone the agent's start event cannot be serialised and the run does not complete.

With `llamactl`:

```bash
uv tool install llamactl
llamactl init                     # choose a template; creates a new project directory
cd my-app                         # the directory you named during init
llamactl serve                    # serves the project's workflows locally, with reload
llamactl auth login               # once, before deploying to LlamaCloud
llamactl deployments create
```

A project tells `llamactl` which server to run in `pyproject.toml`, under `[tool.llamaagents]`, for example `app = "my_app.server:server"`. Durable stores, authentication, the serializer setting agents need, and containers are covered in the [production guide](/llamaindex-guide/python/llamaindex_production_guide/#4-serving-agents-and-workflows-with-workflowserver).

What survives from the old `llama-agents` architecture is the idea, not the API: if you need agents in separate processes, run each as its own service and call it from a tool.

---

## 15. Human-in-the-loop

A step or a tool can pause the run and wait for a human. The run emits an `InputRequiredEvent`; your application answers by sending a `HumanResponseEvent` into the context.

```python
from llama_index.core.workflow import Context, InputRequiredEvent, HumanResponseEvent

async def send_email(ctx: Context, to: str, body: str) -> str:
    """Send an email. Requires human approval."""
    answer = await ctx.wait_for_event(
        HumanResponseEvent,
        waiter_event=InputRequiredEvent(prefix=f"Send email to {to}? (yes/no) "),
    )
    if answer.response.strip().lower() != "yes":
        return "The user declined; the email was not sent."
    # ... actually send it ...
    return f"Email sent to {to}."

agent = FunctionAgent(tools=[send_email], llm=OpenAI(model="%%model=openai-small%%"))

handler = agent.run(user_msg="Email bob@example.com to say the build is green.")
async for event in handler.stream_events():
    if isinstance(event, InputRequiredEvent):
        reply = input(event.prefix)
        handler.ctx.send_event(HumanResponseEvent(response=reply))

print(await handler)
```

Within one process you can keep the handler alive and call `handler.ctx.send_event(...)` whenever the answer arrives. Approval that must survive a restart is easier with a small custom workflow whose state is all serialisable — see [durable approval](/llamaindex-guide/python/llamaindex_advanced_agents_python/#42-approval-that-outlives-the-process); a prebuilt agent paused inside a tool cannot currently be resumed from a serialised context, because its memory is not serialised.

---

## 16. FunctionTool in detail

```python
from typing import Annotated
from pydantic import BaseModel, Field
from llama_index.core.tools import FunctionTool

def search_orders(
    customer_id: Annotated[str, "The customer's ID, e.g. C-1042"],
    limit: int = 5,
) -> list[dict]:
    """Return the customer's most recent orders, newest first."""
    return [{"id": "O-1", "total": 42.0}][:limit]

tool = FunctionTool.from_defaults(fn=search_orders)
print(tool.metadata.name, tool.metadata.description)
print(tool.metadata.fn_schema.model_json_schema())
```

**Explicit schema, name and description**

```python
class SearchInput(BaseModel):
    query: str = Field(description="Full-text query")
    max_results: int = Field(default=10, ge=1, le=50)

def search_database(query: str, max_results: int = 10) -> str:
    return f"{max_results} results for {query!r}"

db_tool = FunctionTool.from_defaults(
    fn=search_database,
    name="database_search",
    description="Full-text search over the product database.",
    fn_schema=SearchInput,
)
```

**Async tools and sync/async pairs**

```python
import httpx

async def fetch_status(url: str) -> int:
    """Return the HTTP status code for a URL."""
    async with httpx.AsyncClient(timeout=10) as client:
        return (await client.get(url)).status_code

status_tool = FunctionTool.from_defaults(async_fn=fetch_status)
```

**Binding parameters the LLM should not set**

```python
def get_user_orders(user_id: str, limit: int = 5) -> list[dict]:
    """Return the current user's orders."""
    ...

orders_tool = FunctionTool.from_defaults(
    fn=get_user_orders,
    partial_params={"user_id": "C-1042"},  # injected, hidden from the schema
)
```

**Errors.** If a tool raises, the exception text is returned to the LLM as the tool output so it can recover. Put retries for transient failures inside the tool itself (for example with `tenacity`).

---

## 17. LlamaHub tools and MCP

### 17.1 Tool specs from LlamaHub

Tool integrations are packages named `llama-index-tools-<name>`; browse them at [llamahub.ai](https://llamahub.ai). A `ToolSpec` turns into a list of tools:

```python
# pip install llama-index-tools-wikipedia
from llama_index.tools.wikipedia import WikipediaToolSpec

wiki_tools = WikipediaToolSpec().to_tool_list()
agent = FunctionAgent(tools=wiki_tools, llm=OpenAI(model="%%model=openai-small%%"))
```

Check each spec's README for the exact tool names and constructor arguments; they vary by package.

### 17.2 MCP servers as tools

```python
# pip install llama-index-tools-mcp
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

mcp_client = BasicMCPClient("http://127.0.0.1:8000/mcp")       # or a command for stdio servers
mcp_tools = await McpToolSpec(client=mcp_client).to_tool_list_async()

agent = FunctionAgent(tools=mcp_tools, llm=OpenAI(model="%%model=openai-small%%"))
```

The same package can expose a LlamaIndex workflow *as* an MCP server (`workflow_as_mcp`).

---

## 18. Data connectors

### 18.1 Files

```python
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(
    input_dir="./docs",
    required_exts=[".md", ".pdf"],
    recursive=True,
    filename_as_id=True,
)
documents = reader.load_data()          # or: await reader.aload_data()
```

`SimpleDirectoryReader` picks a reader per extension; plain-text formats work out of the box, while PDF, DOCX, images and other formats need `llama-index-readers-file` (it is not part of the `llama-index` bundle). For difficult PDFs (tables, scans, slides), a layout-aware parser gives much better chunks; see the next section, or the open-source LiteParse parser (`liteparse`) if documents must not leave your infrastructure.

### 18.2 LlamaParse and LlamaCloud

LlamaParse is LlamaIndex's hosted document parser, part of LlamaCloud (which also offers extraction, classification and managed indexes). Use the `llama-cloud` SDK; the older `llama-cloud-services` package is deprecated and was maintained only until May 2026.

```python
# pip install llama-cloud        (reads LLAMA_CLOUD_API_KEY from the environment)
from llama_cloud import LlamaCloud
from llama_index.core import Document

client = LlamaCloud()
with open("annual_report.pdf", "rb") as f:
    result = client.parsing.parse(
        tier="agentic",            # also "fast", "cost_effective", "agentic_plus"
        version="latest",
        upload_file=f,
        expand=["markdown"],
    )

documents = [
    Document(text=page.markdown, metadata={"file_name": "annual_report.pdf", "page": page.page_number})
    for page in result.markdown.pages
    if page.success
]
```

`parse()` uploads the file, polls until the job finishes and returns the result; `AsyncLlamaCloud` has the same methods for async code. The page-level Markdown keeps tables as Markdown tables, which `MarkdownNodeParser` or `SentenceSplitter` then chunk as usual. See the [LlamaParse documentation](https://developers.llamaindex.ai/) for tiers and options.

### 18.3 Other sources

Readers for databases, SaaS tools and the web are separate packages, for example:

| Source | Package | Class |
|---|---|---|
| Web pages | `llama-index-readers-web` | `SimpleWebPageReader`, `TrafilaturaWebReader` |
| SQL databases | `llama-index-readers-database` | `DatabaseReader` |
| GitHub | `llama-index-readers-github` | `GithubRepositoryReader` |
| Notion | `llama-index-readers-notion` | `NotionPageReader` |
| Confluence | `llama-index-readers-confluence` | `ConfluenceReader` |
| Slack | `llama-index-readers-slack` | `SlackReader` |
| Google Drive/Docs | `llama-index-readers-google` | `GoogleDriveReader`, `GoogleDocsReader` |

```python
# pip install llama-index-readers-web
from llama_index.readers.web import SimpleWebPageReader

pages = SimpleWebPageReader(html_to_text=True).load_data(["https://example.com"])
```

### 18.4 Writing a reader

```python
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

class TicketReader(BaseReader):
    """Load support tickets from an internal API."""

    def __init__(self, client):
        self.client = client

    def load_data(self, project: str) -> list[Document]:
        return [
            Document(text=t["body"], metadata={"ticket_id": t["id"], "project": project})
            for t in self.client.list_tickets(project)
        ]
```

---

## 19. Advanced retrieval

### 19.1 Hybrid retrieval (vector + BM25) with reciprocal-rank fusion

```python
# pip install llama-index-retrievers-bm25
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

vector_retriever = index.as_retriever(similarity_top_k=5)
bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=5)

hybrid = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=5,
    num_queries=1,              # >1 also generates query variations with the LLM
    mode="reciprocal_rerank",
    use_async=True,
)
nodes = hybrid.retrieve("How do I rotate API keys?")
```

`BM25Retriever` is **not** in `llama_index.core.retrievers`; it comes from the separate package above.

### 19.2 Node post-processors and reranking

```python
from llama_index.core.postprocessor import SimilarityPostprocessor, LLMRerank, MetadataReplacementPostProcessor

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.5),
        LLMRerank(top_n=3, llm=OpenAI(model="%%model=openai-small%%")),
    ],
)
```

Other useful post-processors: `SentenceTransformerRerank` (local cross-encoder), `LongContextReorder`, `PrevNextNodePostprocessor`, `MetadataReplacementPostProcessor` (pair with `SentenceWindowNodeParser`), and provider rerankers such as `llama-index-postprocessor-cohere-rerank`.

A custom post-processor:

```python
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

class RecentOnly(BaseNodePostprocessor):
    min_year: int = 2024

    def _postprocess_nodes(self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None):
        return [n for n in nodes if n.node.metadata.get("year", 0) >= self.min_year]
```

### 19.3 Router query engine

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

router = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        QueryEngineTool.from_defaults(
            query_engine=vector_index.as_query_engine(),
            description="Specific facts and details from the documents.",
        ),
        QueryEngineTool.from_defaults(
            query_engine=summary_index.as_query_engine(response_mode="tree_summarize"),
            description="Summaries of the documents as a whole.",
        ),
    ],
)
print(router.query("Give me an overview of the whole corpus."))
```

`LLMMultiSelector` lets the router pick several engines and combine their answers.

### 19.4 Sub-question decomposition

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen import LLMQuestionGenerator

llm = OpenAI(model="%%model=openai-small%%")
engine = SubQuestionQueryEngine.from_defaults(
    question_gen=LLMQuestionGenerator.from_defaults(llm=llm),
    llm=llm,
    query_engine_tools=[
        QueryEngineTool.from_defaults(query_engine=tesla_index.as_query_engine(), name="tesla_10k",
                                      description="Tesla annual report"),
        QueryEngineTool.from_defaults(query_engine=ford_index.as_query_engine(), name="ford_10k",
                                      description="Ford annual report"),
    ],
)
print(engine.query("Compare Tesla's and Ford's revenue growth."))
```

Pass `question_gen` explicitly. Without it, `from_defaults` tries to import the OpenAI function-calling question generator from `llama-index-question-gen-openai`, which is not part of the `llama-index` bundle and cannot be installed alongside current releases (it pins an old `llama-index-llms-openai`). `LLMQuestionGenerator` works with any LLM.

### 19.5 Custom query engine

```python
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.llms import LLM
from llama_index.core import PromptTemplate

QA_PROMPT = PromptTemplate("Context:\n{context}\n\nAnswer the question: {query}\n")

class SimpleRAG(CustomQueryEngine):
    retriever: BaseRetriever
    llm: LLM

    def custom_query(self, query_str: str) -> str:
        nodes = self.retriever.retrieve(query_str)
        context = "\n\n".join(n.get_content() for n in nodes)
        return str(self.llm.complete(QA_PROMPT.format(context=context, query=query_str)))

engine = SimpleRAG(retriever=index.as_retriever(), llm=OpenAI(model="%%model=openai-small%%"))
```

---

## 20. Prompts

```python
from llama_index.core import PromptTemplate, ChatPromptTemplate
from llama_index.core.llms import ChatMessage

qa_template = PromptTemplate(
    "Use the context to answer.\n---\n{context_str}\n---\nQuestion: {query_str}\nAnswer: "
)

chat_template = ChatPromptTemplate([
    ChatMessage(role="system", content="You are a precise technical writer."),
    ChatMessage(role="user", content="Explain {topic} in two sentences."),
])
messages = chat_template.format_messages(topic="vector databases")
```

**Inspecting and overriding a query engine's prompts**

```python
query_engine = index.as_query_engine()
for key in query_engine.get_prompts():
    print(key)        # e.g. response_synthesizer:text_qa_template

query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_template})
```

Agents take a `system_prompt`; the ReAct prompt itself can be replaced with `react_agent.update_prompts({"react_header": ...})`.

**Few-shot examples** are just text in the template — or, for many examples, retrieve the most similar ones per query with a small vector index and format them into the prompt via `function_mappings`.

---

## 21. Evaluation

### 21.1 Evaluating responses

```python
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

judge = OpenAI(model="%%model=openai-small%%")
faithfulness = FaithfulnessEvaluator(llm=judge)
relevancy = RelevancyEvaluator(llm=judge)

query = "What is LlamaIndex?"
response = query_engine.query(query)

f = faithfulness.evaluate_response(response=response)
r = relevancy.evaluate_response(query=query, response=response)
print(f.passing, r.passing, r.feedback)
```

Also available: `CorrectnessEvaluator` (against a reference answer), `AnswerRelevancyEvaluator`, `ContextRelevancyEvaluator`, `SemanticSimilarityEvaluator`, and `BatchEvalRunner` to run many evaluators over many questions concurrently. Each of these calls the judge LLM per question.

### 21.2 Evaluation datasets

An evaluation set built from your own documents is worth more than any generic benchmark. The cheapest useful one is a retrieval set: for each chunk, ask an LLM for questions that the chunk answers, and record that chunk as the expected hit. `DatasetGenerator` generates the questions; `RetrieverEvaluator` scores a retriever against them.

```python
import json

from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import DatasetGenerator, RetrieverEvaluator
from llama_index.core.node_parser import SentenceSplitter

nodes = SentenceSplitter(chunk_size=512).get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)

async def build_retrieval_set(nodes, llm, per_node=2):
    """Generate questions per chunk; the chunk that produced a question is the expected hit."""
    pairs = []
    for node in nodes:
        gen = DatasetGenerator([node], llm=llm, num_questions_per_chunk=per_node)
        for question in await gen.agenerate_questions_from_nodes():
            pairs.append({"query": question, "expected_ids": [node.node_id]})
    return pairs

async def main():
    pairs = await build_retrieval_set(nodes, llm)
    with open("retrieval_eval.json", "w") as f:
        json.dump(pairs, f, indent=2)          # review by hand, then commit it

    evaluator = RetrieverEvaluator.from_metric_names(
        ["hit_rate", "mrr"], retriever=index.as_retriever(similarity_top_k=3)
    )
    results = [await evaluator.aevaluate(**p) for p in pairs]
    hit_rate = sum(r.metric_vals_dict["hit_rate"] for r in results) / len(results)
    mrr = sum(r.metric_vals_dict["mrr"] for r in results) / len(results)
    print(f"hit rate {hit_rate:.2f}  MRR {mrr:.2f}  ({len(results)} questions)")
```

Notes:

- The expected IDs are node IDs, so they must stay stable when you re-ingest. Give the node parser a deterministic `id_func`, as the [production guide](/llamaindex-guide/python/llamaindex_production_guide/#1-ingestion-is-a-job-not-a-request) does.
- Generated questions are often too easy (they reuse the chunk's wording). Read through the set, delete weak questions and add real user questions from your logs.
- For a question-and-reference-answer set, `DatasetGenerator(nodes, llm=llm).agenerate_dataset_from_nodes()` returns a `QueryResponseDataset` (`save_json` / `from_json`) to use with `CorrectnessEvaluator`. It emits a deprecation warning that points to `LabelledRagDataset`, but the `llama_dataset` module (`LabelledRagDataset`, `RagDatasetGenerator`) is not included in `llama-index-core` %%version=llama-index%%, so `DatasetGenerator` is what the current release offers. Its `from_documents()` constructor needs spaCy installed; building it from nodes, as above, does not.

---

## 22. Migrating from the legacy agent API

`llama-index-core` 0.13.0 (July 2025) removed the old agent stack; 0.14.0 (September 2025) moved to `llama-index-workflows` 2.x, which removed workflow checkpointers, sub-workflows and the deprecated `Workflow.send_event` / `Workflow.stream_events` methods (both remain on `Context` / the handler).

| Old (≤ 0.12) | Current |
|---|---|
| `from llama_index.core.agent import ReActAgent`; `ReActAgent.from_tools(tools, llm=llm)` | `from llama_index.core.agent.workflow import ReActAgent`; `ReActAgent(tools=tools, llm=llm)` |
| `OpenAIAgent.from_tools(...)`, `FunctionCallingAgent.from_tools(...)` | `FunctionAgent(tools=..., llm=...)` |
| `AgentRunner`, step workers, `StructuredPlanningAgent` | `FunctionAgent` / `AgentWorkflow` / a custom `Workflow` |
| `agent.chat("...")` (sync) | `await agent.run(user_msg="...", ctx=ctx)` |
| `agent.stream_chat(...)` | `handler = agent.run(...)`; `async for ev in handler.stream_events()` |
| `memory=ChatMemoryBuffer(...)` on the agent | `Context(agent)` per conversation, or `memory=Memory.from_defaults(...)` on `run()` |
| `index.as_chat_engine()` returning an agent | Returns `CondensePlusContextChatEngine`; build an agent explicitly if you need tools |
| `ServiceContext` | `Settings` |
| `QueryPipeline` | A custom `Workflow` |
| `llama-agents` (`ControlPlaneServer`, `AgentService`) | `AgentWorkflow` in-process; `llamactl` / LlamaAgents for deployment |

`run()` must be called while an event loop is running — `asyncio.run(agent.run(...))` fails with `RuntimeError: no running event loop`. From synchronous code, wrap the call in a coroutine: `async def main(): return await agent.run(user_msg=...)` and then `asyncio.run(main())`.

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-24 | 0.14.25 / workflows 2.26.1 | Section 13 cut to an introduction; Workflows now have their own [page](/llamaindex-guide/python/llamaindex_workflows/). Its retry example used the deprecated `ConstantDelayRetryPolicy`. |
| 2026-10-21 | 0.14.25 | Added property graphs (3.9), multimodal (3.10), `WorkflowServer`/`WorkflowClient`/`llamactl` serving (14), LlamaParse via the `llama-cloud` SDK (18.2) and evaluation datasets (21.2). `SubQuestionQueryEngine` example now passes `question_gen` (the default needs a package that cannot be installed with current releases). |
| 2026-10-21 | 0.14.25 | Rewritten. Previous edition used APIs that do not exist (`AgentWorkflow.from_tools`, `from llama_index.core.workflow import AgentWorkflow`, `llama_index.core.tools.Tool`, `BM25Retriever` in core, `llama_agents`) and wrongly stated that `ReActAgent` raises `ImportError`. All examples now use the workflow-based agents and were checked against an installed `llama-index-core`. Version banner now tracks the site's version registry. |
| 2026-05-19 | 0.14.22 | Patch-version bump. |
| 2026-04-03 | 0.14.20 | Agent sections moved off the removed legacy classes (incompletely). |
| November 2025 | 0.14.8 | First documented version. |

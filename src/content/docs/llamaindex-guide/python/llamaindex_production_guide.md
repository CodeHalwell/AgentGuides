---
title: "LlamaIndex Production Guide"
description: "Running LlamaIndex in production: ingestion jobs, index persistence, vector store choice, serving agents and workflows, observability, evaluation gates and testing."
framework: "llamaindex"
language: "python"
sidebar:
  label: "Production"
---

> Verified against `llama-index-core` %%version=llama-index%%, `llama-agents-server` 0.8.0 and `llamactl` 0.10.4 (checked %%versions-checked%%). Every Python block on this page was run, with mock models standing in for OpenAI; the tests shown pass.

This page is about the parts of a production deployment that are specific to LlamaIndex: keeping ingestion out of the request path, where the index lives, how to serve agents and workflows, and how to observe, evaluate and test them. Generic web-service concerns (TLS, autoscaling, secrets management, load balancers) are the same as for any Python service and are left to your platform's documentation.

The running example is a staff-handbook assistant: documents are ingested into Qdrant by a scheduled job, and an API answers questions over them, scoped to the caller's tenant.

---

## 1. Ingestion is a job, not a request

Building an index means reading files, chunking and embedding them, which is slow and costs money. Do it in a separate job (a cron job, a queue worker, a CI step), and have the API only connect to an index that already exists.

`IngestionPipeline` with a docstore and `DocstoreStrategy.UPSERTS` makes the job incremental. It hashes each document, skips the ones that have not changed, and for a changed document deletes its old chunks from the vector store before writing the new ones. This only works if document IDs are stable, which is what `filename_as_id=True` gives you.

```bash
pip install "llama-index-core==%%version=llama-index%%" llama-index-llms-openai llama-index-embeddings-openai \
    llama-index-vector-stores-qdrant llama-index-readers-file
```

```python
# config.py
import os

COLLECTION = os.environ.get("QDRANT_COLLECTION", "handbook")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "%%model=openai-small%%")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
```

```python
# ingest.py
"""Ingestion job: run on a schedule or on document change, never inside the API process."""
import sys
import uuid

from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config import COLLECTION, EMBED_MODEL, QDRANT_API_KEY, QDRANT_URL

DOCSTORE_PATH = "./pipeline_storage"


def stable_node_id(i: int, doc) -> str:
    # same file + same chunk position -> same ID, so evaluation sets keep matching
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.doc_id}#{i}"))


def build_pipeline(client: QdrantClient) -> IngestionPipeline:
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=64, id_func=stable_node_id),
            OpenAIEmbedding(model=EMBED_MODEL),
        ],
        vector_store=QdrantVectorStore(collection_name=COLLECTION, client=client),
        docstore=SimpleDocumentStore(),
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )
    try:
        pipeline.load(DOCSTORE_PATH)        # document hashes and transformation cache
    except FileNotFoundError:
        pass                                # first run
    return pipeline


def ingest(source_dir: str, tenant: str, client: QdrantClient) -> int:
    documents = SimpleDirectoryReader(source_dir, recursive=True, filename_as_id=True).load_data()
    for doc in documents:
        doc.metadata["tenant"] = tenant
        doc.excluded_embed_metadata_keys.append("tenant")
        doc.excluded_llm_metadata_keys.append("tenant")
    pipeline = build_pipeline(client)
    nodes = pipeline.run(documents=documents)
    pipeline.persist(DOCSTORE_PATH)
    return len(nodes)


if __name__ == "__main__":
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print(ingest(sys.argv[1], tenant=sys.argv[2], client=client), "nodes written")
```

Notes:

- **Stable node IDs.** By default each chunk gets a random UUID on every run. The `id_func` above derives the ID from the document ID and chunk position, so the evaluation set in [section 6](#6-evaluation-as-a-release-gate) keeps matching after re-ingestion. Qdrant requires point IDs to be UUIDs or integers, hence `uuid5`.
- **Where the docstore lives.** `SimpleDocumentStore` persisted to a local folder is fine for a single job host. If the job runs on ephemeral machines, use a shared docstore (`llama-index-storage-docstore-redis`, `-postgres` or `-mongodb`) so the hashes survive between runs.
- **Tenant metadata.** The `tenant` key is stored with every chunk so the API can filter on it, and it is excluded from the embedding and the LLM prompt.
- **Hard documents.** For scanned PDFs, complex tables and slides, parse with LlamaParse before chunking (see [data connectors](/llamaindex-guide/python/llamaindex_comprehensive_guide/#18-data-connectors)).

## 2. Index persistence and vector store choice

There are two ways to keep an index between processes:

- **Local persistence**: `index.storage_context.persist("./storage")` writes the docstore, index metadata and (with the default `SimpleVectorStore`) the vectors as JSON files; `load_index_from_storage` reads them back. Everything is loaded into memory. This suits small, read-only corpora that ship with the container, and tests.
- **External vector store**: the vectors and the chunk text live in a database, and the API reconnects with `VectorStoreIndex.from_vector_store(store)` without re-embedding anything. Use this when several replicas share one index or the corpus changes while the service runs.

What to check when picking a store, from LlamaIndex's point of view:

| Need | What to look for in the integration |
|---|---|
| Async serving | An async client (`aquery`/`async_add`). Qdrant needs `aclient=AsyncQdrantClient(...)` or `url=`; without it `aquery` raises. |
| Incremental ingestion | `delete_ref_doc` support, which `DocstoreStrategy.UPSERTS` relies on. |
| Tenant or permission filtering | `MetadataFilters` support, ideally with payload indexes on the filtered keys. |
| Hybrid (dense + sparse) search | Built-in hybrid mode (Qdrant `enable_hybrid=True`, pgvector `hybrid_search=True`), or fuse a BM25 retriever yourself ([advanced retrieval](/llamaindex-guide/python/llamaindex_comprehensive_guide/#19-advanced-retrieval)). |
| Fewer moving parts | pgvector (`llama-index-vector-stores-postgres`) if you already run Postgres. |

Pick the embedding model once per collection: vectors from different models cannot be mixed, so changing the model means re-ingesting into a new collection.

## 3. Serving a query engine with FastAPI

When the service is mostly retrieval and synthesis (no long-running agent loops or human approval steps), a small FastAPI app is the simplest option. The app below has one `/query` route that requires an API key, is rate-limited per key, filters retrieval to the caller's tenant and enforces a timeout. It also defines `/health` (liveness) and `/ready` (index loaded and vector store reachable) for container probes.

```bash
pip install fastapi "uvicorn[standard]" slowapi
```

```python
# app.py
import asyncio
import hashlib
import logging
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, QdrantClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore

from config import COLLECTION, EMBED_MODEL, LLM_MODEL, QDRANT_API_KEY, QDRANT_URL

log = logging.getLogger("rag-api")
QUERY_TIMEOUT_S = float(os.environ.get("QUERY_TIMEOUT_S", "30"))
RATE_LIMIT = os.environ.get("RATE_LIMIT", "30/minute")

# "key1:acme,key2:globex" -> {"key1": "acme", ...}; use a secret store in practice
API_KEYS = dict(
    pair.split(":", 1) for pair in os.environ.get("API_KEYS", "").split(",") if pair
)


def make_clients() -> tuple[QdrantClient, AsyncQdrantClient]:
    return (
        QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY),
        AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    Settings.llm = OpenAI(model=LLM_MODEL, timeout=60, max_retries=3)
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    client, aclient = make_clients()
    store = QdrantVectorStore(collection_name=COLLECTION, client=client, aclient=aclient)
    app.state.qdrant = aclient
    app.state.index = VectorStoreIndex.from_vector_store(store)   # no re-embedding at start-up
    yield
    await aclient.close()
    client.close()


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def tenant_for_key(api_key: str | None = Security(api_key_header)) -> str:
    for known, tenant in API_KEYS.items():
        if api_key and secrets.compare_digest(api_key, known):
            return tenant
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


limiter = Limiter(key_func=lambda request: request.headers.get("X-API-Key", "anonymous"))
app = FastAPI(title="Handbook RAG API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal error"})


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=20)


class Source(BaseModel):
    file_name: str | None
    score: float | None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}                     # liveness: the process is serving


@app.get("/ready")
async def ready(request: Request) -> dict:
    # readiness: index object built and the vector store reachable
    if getattr(request.app.state, "index", None) is None:
        raise HTTPException(status_code=503, detail="index not loaded")
    try:
        exists = await request.app.state.qdrant.collection_exists(COLLECTION)
    except Exception:
        exists = False
    if not exists:
        raise HTTPException(status_code=503, detail="vector store unavailable")
    return {"status": "ready"}


@app.post("/query", response_model=QueryResponse)
@limiter.limit(RATE_LIMIT)
async def query(
    request: Request, body: QueryRequest, tenant: str = Depends(tenant_for_key)
) -> QueryResponse:
    engine = request.app.state.index.as_query_engine(
        similarity_top_k=body.top_k,
        filters=MetadataFilters(
            filters=[MetadataFilter(key="tenant", value=tenant, operator=FilterOperator.EQ)]
        ),
    )
    try:
        response = await asyncio.wait_for(engine.aquery(body.question), QUERY_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Query timed out")

    log.info(
        "query tenant=%s q=%s sources=%s",
        tenant,
        hashlib.sha256(body.question.encode()).hexdigest()[:12],   # no raw user text in logs
        [n.node_id for n in response.source_nodes],
    )
    return QueryResponse(
        answer=str(response),
        sources=[Source(file_name=n.metadata.get("file_name"), score=n.score) for n in response.source_nodes],
    )
```

Run it with `uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2`. Each worker loads its own `VectorStoreIndex` object, which is cheap with an external store because nothing is embedded at start-up.

Design points:

- **One query engine per request** is fine: `as_query_engine()` only wires objects together. It is the cleanest way to apply per-request settings such as `filters` and `similarity_top_k`.
- **Enforce tenancy in the retriever** (`MetadataFilters`), not by asking the model to ignore other tenants' text.
- **Timeouts.** The `OpenAI` client's `timeout` and `max_retries` cover single calls; `asyncio.wait_for` bounds the whole query, including retries and multiple LLM calls in `refine` mode.
- **Streaming** answers over Server-Sent Events are covered on the [streaming server page](/llamaindex-guide/python/llamaindex_streaming_server_fastapi/).

The tests use LlamaIndex's mock models and an in-memory Qdrant, so they run in CI without network access or API keys:

```python
# test_app.py
import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient, QdrantClient

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.vector_stores.qdrant import QdrantVectorStore

import app as app_module
from config import COLLECTION


@pytest.fixture
def client(monkeypatch):
    aclient = AsyncQdrantClient(location=":memory:")
    monkeypatch.setattr(app_module, "make_clients", lambda: (QdrantClient(location=":memory:"), aclient))
    monkeypatch.setattr(app_module, "API_KEYS", {"test-key": "acme"})
    # Mock models instead of OpenAI: no network, no cost
    monkeypatch.setattr(app_module, "OpenAI", lambda **kw: MockLLM(max_tokens=16))
    monkeypatch.setattr(app_module, "OpenAIEmbedding", lambda **kw: MockEmbedding(embed_dim=8))

    with TestClient(app_module.app) as c:
        # seed through the same async client the app queries
        store = QdrantVectorStore(collection_name=COLLECTION, aclient=aclient)
        index = VectorStoreIndex.from_vector_store(store, embed_model=Settings.embed_model)
        docs = [
            Document(text="Holiday allowance is 25 days.", metadata={"tenant": "acme", "file_name": "leave.md"}),
            Document(text="Globex staff get 20 days.", metadata={"tenant": "globex", "file_name": "globex.md"}),
        ]
        c.portal.call(index.ainsert_nodes, docs)
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").status_code == 200


def test_query_requires_key(client):
    assert client.post("/query", json={"question": "Holiday?"}).status_code == 401


def test_query_is_tenant_scoped(client):
    r = client.post("/query", json={"question": "How much holiday?"}, headers={"X-API-Key": "test-key"})
    assert r.status_code == 200
    assert {s["file_name"] for s in r.json()["sources"]} == {"leave.md"}


def test_validation(client):
    r = client.post("/query", json={"question": ""}, headers={"X-API-Key": "test-key"})
    assert r.status_code == 422
```

```python
# test_ingest.py
import os, shutil
from qdrant_client import QdrantClient
from llama_index.core.embeddings import MockEmbedding
import ingest

def test_upserts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ingest, "OpenAIEmbedding", lambda **kw: MockEmbedding(embed_dim=8))
    os.makedirs("docs"); open("docs/leave.md", "w").write("Holiday allowance is 25 days.")
    client = QdrantClient(location=":memory:")
    assert ingest.ingest("docs", "acme", client) == 1
    assert ingest.ingest("docs", "acme", client) == 0          # unchanged: skipped
    open("docs/leave.md", "w").write("Holiday allowance is 27 days.")
    assert ingest.ingest("docs", "acme", client) == 1          # changed: replaced
    assert client.count("handbook").count == 1
    ids1 = [p.id for p in client.scroll("handbook")[0]]
    open("docs/leave.md", "w").write("Holiday allowance is 28 days.")
    ingest.ingest("docs", "acme", client)
    assert [p.id for p in client.scroll("handbook")[0]] == ids1
```

## 4. Serving agents and workflows with WorkflowServer

For agents and custom workflows, LlamaIndex ships its own server. `WorkflowServer` (package `llama-agents-server`, imported from `llama_agents.server`) exposes each registered workflow over REST: start a run and wait (`POST /workflows/{name}/run`), start one in the background (`/run-nowait`), stream its events (`GET /events/{handler_id}`), send events into it, for example a human approval (`POST /events/{handler_id}`), fetch results and cancel. It also serves `/health`. Runs are recorded in a workflow store, and with a durable store such as `SqliteWorkflowStore`, unfinished runs are resumed when the server restarts. The comprehensive guide has a [minimal example](/llamaindex-guide/python/llamaindex_comprehensive_guide/#14-serving-and-deploying-workflows-llamaagents); this is the production shape.

```bash
pip install llama-agents-server llama-agents-client
```

```python
# wfserver.py
import os
import secrets

from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_agents.server import SqliteWorkflowStore, WorkflowServer
from workflows.context.serializers import JsonSerializer

from config import COLLECTION, LLM_MODEL, QDRANT_API_KEY, QDRANT_URL

API_KEY = os.environ["WORKFLOW_API_KEY"]


class RequireApiKey(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path != "/health" and not secrets.compare_digest(
            request.headers.get("X-API-Key", ""), API_KEY
        ):
            return JSONResponse({"detail": "unauthorised"}, status_code=401)
        return await call_next(request)


Settings.llm = OpenAI(model=LLM_MODEL)
index = VectorStoreIndex.from_vector_store(
    QdrantVectorStore(collection_name=COLLECTION, url=QDRANT_URL, api_key=QDRANT_API_KEY)
)
agent = FunctionAgent(
    name="handbook",
    tools=[
        QueryEngineTool.from_defaults(
            query_engine=index.as_query_engine(similarity_top_k=4),
            name="handbook",
            description="Search the staff handbook.",
        )
    ],
    system_prompt="Answer from the handbook. Say so when it has no answer.",
    timeout=120,
)

server = WorkflowServer(
    middleware=[Middleware(RequireApiKey)],
    workflow_store=SqliteWorkflowStore(os.environ.get("WORKFLOW_DB", "workflows.db")),
    serializer=JsonSerializer(),    # restores agent memory/state from the store; see note
)
server.add_workflow("handbook", agent)
```

Serve it with `python -c "import asyncio, wfserver; asyncio.run(wfserver.server.serve(host='0.0.0.0', port=8080))"`, or mount `server.app` (a Starlette app) in a larger ASGI application.

Two things that are easy to get wrong with the current releases:

- **Agent start events.** When you start an agent (`FunctionAgent`, `ReActAgent`, `AgentWorkflow`) over HTTP, send `user_msg`, `chat_history` and `max_iterations`. With only `user_msg`, the agent's start event fails to serialise and the run never completes.
- **The serializer.** With a durable store, agent state (memory, the scratchpad of chat messages) is written to the store and read back between steps. The default serializer refuses to rebuild those classes, and the run fails with `... is not in the serializer's allowed types`. `JsonSerializer()` resolves classes by import path, which is safe as long as only your server writes to the store. To be stricter, list the classes instead: `JsonSerializer(allowed_types=[...])` with the classes named in the error messages.

Call the server from Python with `WorkflowClient`:

```python
import httpx
from llama_agents.client import WorkflowClient

async def ask(question: str) -> dict:
    async with httpx.AsyncClient(
        base_url="https://agents.internal.example.com",
        headers={"X-API-Key": "..."},
        timeout=60,
    ) as http:
        client = WorkflowClient(httpx_client=http)
        handler = await client.run_workflow_nowait(
            "handbook",
            start_event={"user_msg": question, "chat_history": [], "max_iterations": 20},
        )
        done = await client.wait_for_handler(handler.handler_id, timeout=120)
        return done.result.value["result"]   # the serialised AgentOutput
```

### Deploying with llamactl

`llamactl` (the LlamaAgents CLI) runs the same `WorkflowServer` for you. Point it at the server object in `pyproject.toml`:

```toml
[tool.llamaagents]
app = "handbook_agent.wfserver:server"
required_env_vars = ["OPENAI_API_KEY", "WORKFLOW_API_KEY", "QDRANT_URL"]
```

```bash
llamactl serve --persistence local      # local development, runs stored in SQLite
llamactl pkg container                  # writes a Dockerfile that runs `llamactl serve`
llamactl auth login                     # then `llamactl deployments create` to deploy to LlamaCloud
```

`llamactl pkg container` expects a `uv` project with a lock file (`uv lock`). Use it when you want to self-host the LlamaAgents runtime; use your own Dockerfile (section 9) when the workflow server is one part of a larger app.

## 5. Observability and cost

Turn on LlamaIndex's OpenTelemetry integration once at start-up, before any index or agent is built. Every retrieval, LLM call, tool call and workflow step then becomes a span:

```python
# pip install llama-index-observability-otel opentelemetry-exporter-otlp-proto-http
from llama_index.observability.otel import LlamaIndexOpenTelemetry
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

LlamaIndexOpenTelemetry(
    service_name_or_resource="handbook-api",
    span_exporter=OTLPSpanExporter(endpoint="http://otel-collector:4318/v1/traces"),
).start_registering()
```

Spans can carry prompt and retrieved text, so treat the trace store as holding user data, or filter attributes in the collector. Hosted tools (Phoenix, Langfuse and others), custom event handlers and the metrics worth exporting are on the [observability page](/llamaindex-guide/python/llamaindex_observability_python/).

For cost, count tokens with `TokenCountingHandler` and price them from configuration rather than constants in code, since prices change:

```python
import json
import os

import tiktoken
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

token_counter = TokenCountingHandler(tokenizer=tiktoken.get_encoding("o200k_base").encode)
Settings.callback_manager = CallbackManager([token_counter])

# USD per million tokens for the model you deploy, from your provider's price list
PRICES = json.loads(os.environ["LLM_PRICES_PER_MTOK"])        # {"input": ..., "output": ...}


def spend_usd() -> float:
    return (
        token_counter.prompt_llm_token_count * PRICES["input"]
        + token_counter.completion_llm_token_count * PRICES["output"]
    ) / 1_000_000
```

The counter is process-wide and counts every call since the last `token_counter.reset_counts()`. Export the totals as metrics, or read the provider's own usage block from `LLMChatEndEvent.response.raw` in an [event handler](/llamaindex-guide/python/llamaindex_observability_python/#3-custom-event-handlers) for billing-grade numbers.

## 6. Evaluation as a release gate

Changes to chunking, embedding model, `top_k` or prompts can quietly make answers worse. Keep a retrieval evaluation set in the repository and fail the release when hit rate or MRR drops. The comprehensive guide shows how to [generate the set from your own chunks](/llamaindex-guide/python/llamaindex_comprehensive_guide/#212-evaluation-datasets); the gate is a test that runs against the staging index after ingestion:

```python
# test_retrieval_quality.py
import json
import os

import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.vector_stores.qdrant import QdrantVectorStore

from config import COLLECTION, QDRANT_API_KEY, QDRANT_URL

MIN_HIT_RATE = float(os.environ.get("MIN_HIT_RATE", "0.85"))
MIN_MRR = float(os.environ.get("MIN_MRR", "0.6"))


@pytest.mark.asyncio
async def test_retrieval_quality():
    store = QdrantVectorStore(collection_name=COLLECTION, url=QDRANT_URL, api_key=QDRANT_API_KEY)
    retriever = VectorStoreIndex.from_vector_store(store).as_retriever(similarity_top_k=4)
    evaluator = RetrieverEvaluator.from_metric_names(["hit_rate", "mrr"], retriever=retriever)

    with open("eval/retrieval_eval.json") as f:
        pairs = json.load(f)                    # [{"query": ..., "expected_ids": [...]}, ...]
    results = [await evaluator.aevaluate(**pair) for pair in pairs]

    hit_rate = sum(r.metric_vals_dict["hit_rate"] for r in results) / len(results)
    mrr = sum(r.metric_vals_dict["mrr"] for r in results) / len(results)
    assert hit_rate >= MIN_HIT_RATE, f"hit rate {hit_rate:.2f}"
    assert mrr >= MIN_MRR, f"MRR {mrr:.2f}"
```

Run the cheap retrieval gate on every change. Response-level evaluators (`FaithfulnessEvaluator`, `RelevancyEvaluator`) call an LLM judge per question, so run them on a sample, nightly or before a release.

## 7. Testing without a model

- `MockLLM` and `MockEmbedding` (`llama_index.core.llms`, `llama_index.core.embeddings`) make query engines run offline, as in `test_app.py` above.
- `MockFunctionCallingLLM` (`llama_index.core.llms.mock`) can be scripted with `response_generator` to return tool calls, so agent logic is testable without an API key.
- `WorkflowTestRunner` (`workflows.testing`) runs a workflow or agent and collects every event it emitted.

```python
# test_agent.py
import pytest
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.tools import ToolSelection
from workflows.testing import WorkflowTestRunner


def lookup_policy(topic: str) -> str:
    """Look up an HR policy by topic."""
    return "25 days" if topic == "holiday" else "unknown"


@pytest.mark.asyncio
async def test_agent_calls_tool():
    calls = iter([
        ChatMessage(role=MessageRole.ASSISTANT, content="",
                    additional_kwargs={"tool_calls": [ToolSelection(tool_id="1", tool_name="lookup_policy", tool_kwargs={"topic": "holiday"})]}),
        ChatMessage(role=MessageRole.ASSISTANT, content="You get 25 days."),
    ])
    llm = MockFunctionCallingLLM(response_generator=lambda *a, **k: next(calls))
    agent = FunctionAgent(tools=[lookup_policy], llm=llm)
    result = await WorkflowTestRunner(agent).run(
        start_event=agent.start_event_class(user_msg="How much holiday do I get?")
    )
    tool_results = [e for e in result.collected if isinstance(e, ToolCallResult)]
    assert tool_results and tool_results[0].tool_output.content == "25 days"
    assert "25 days" in str(result.result)
```

## 8. Security

- **Retrieved text is untrusted input.** A document can contain instructions aimed at the model (indirect prompt injection). Do not give an agent that reads such documents tools with side effects (sending email, writing records) unless a human approves each call; the [human-in-the-loop](/llamaindex-guide/python/llamaindex_comprehensive_guide/#15-human-in-the-loop) pattern pauses the workflow for that.
- **Filter by permission in the retriever**, with `MetadataFilters` on keys written at ingestion time (section 3). Never rely on the prompt for access control.
- **`CodeActAgent` executes model-written code.** Give it a `code_execute_fn` that runs in a sandbox (a separate container or a hosted sandbox), never `exec` in the API process.
- **Limit what agents can spend**: `max_iterations` on `run()`, `timeout=` on the agent or workflow, and per-key rate limits at the HTTP layer.
- **Logs**: log a hash of the question and the source node IDs, not raw questions or document text.

## 9. Containers and CI

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://localhost:8000/health').status != 200)"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

`urlopen` raises on any 4xx or 5xx status, so the check fails when the endpoint is missing or unhealthy. In Kubernetes, point the liveness probe at `/health` and the readiness probe at `/ready`, so a pod only receives traffic once it can reach the vector store.

Local development with Qdrant and the ingestion job:

```yaml
# compose.yaml
services:
  qdrant:
    image: qdrant/qdrant
    volumes: ["qdrant_data:/qdrant/storage"]
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      QDRANT_URL: http://qdrant:6333
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      API_KEYS: ${API_KEYS}
    depends_on: [qdrant]
  ingest:
    build: .
    command: ["python", "ingest.py", "/data", "acme"]
    environment:
      QDRANT_URL: http://qdrant:6333
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    volumes: ["./docs:/data:ro", "pipeline_storage:/app/pipeline_storage"]
    depends_on: [qdrant]
    profiles: ["jobs"]          # run with: docker compose run --rm ingest
volumes:
  qdrant_data:
  pipeline_storage:
```

A CI pipeline that runs the offline tests on every push, and builds and pushes the image from `main`:

```yaml
# .github/workflows/ci.yml
name: ci
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: "3.13"
      - run: pip install -r requirements.txt pytest pytest-asyncio httpx
      - run: pytest test_app.py test_ingest.py test_agent.py    # mock models: no secrets needed

  image:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v7
      - uses: docker/setup-buildx-action@v4
      - uses: docker/login-action@v4
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v7
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
```

Run `test_retrieval_quality.py` as a separate step after the ingestion job has updated the staging index, with the staging `QDRANT_URL` and an embedding API key supplied as secrets.

`requirements.txt` for the examples on this page:

```text
llama-index-core==%%version=llama-index%%
llama-index-llms-openai
llama-index-embeddings-openai
llama-index-vector-stores-qdrant
llama-index-readers-file
llama-index-observability-otel
opentelemetry-exporter-otlp-proto-http
llama-agents-server
llama-agents-client
fastapi
uvicorn[standard]
slowapi
```

Integration packages are versioned separately from core. Let pip resolve them against the core pin, then freeze the result in a lock file (`uv lock` or `pip-compile`).

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-21 | 0.14.25 | Rewritten around one working app. The previous edition registered three `/query` handlers (so authentication and rate limiting never ran), referenced `/health` and `/ready` routes that did not exist, used `JSONResponse` without importing it, had a cost tracker that never recorded usage, and was mostly generic FastAPI, Kubernetes, nginx and sharding material. New: incremental ingestion with stable node IDs, vector store selection, tenant filtering, `WorkflowServer` and `llamactl`, evaluation gate, offline tests (all run), current GitHub Actions. |

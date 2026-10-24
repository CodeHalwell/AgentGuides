---
title: "LangGraph (Python): production deployment"
description: "Deploying LangGraph graphs: LangSmith Deployment, the Agent Server and langgraph.json, the langgraph CLI, standalone self-hosting, authentication, calling a deployment, and running a graph inside your own service with Postgres persistence."
framework: "langgraph"
language: "python"
sidebar:
  label: "Production deployment"
---

Checked against `langgraph` %%version=langgraph%%, `langgraph-sdk` %%version=langgraph-sdk%%, `langgraph-checkpoint-postgres` %%version=langgraph-checkpoint-postgres%% and `langgraph-cli` 0.4.38 (%%versions-checked%%). The SDK and `RemoteGraph` examples were run against a local `langgraph dev` server. [Chapter 10](/langgraph-guide/python/chapter-10-production/) is the tutorial introduction to this page: checkpointer choice, durability, loop limits and testing are covered there.

## Deployment options

| Option | What runs your graph | Persistence | You manage |
|---|---|---|---|
| **LangSmith Deployment (Cloud)** | LangChain's Agent Server, hosted by LangChain | Managed Postgres (checkpoints, store, threads, runs) | Your code and `langgraph.json` |
| **Hybrid / self-hosted LangSmith** | Agent Server in your cloud, managed through a LangSmith control plane | Your Postgres and Redis | The data plane |
| **Standalone Agent Server** | The Docker image from `langgraph build`, on Kubernetes (Helm chart) or Docker | Your Postgres and Redis | Everything; needs a licence key for production |
| **Your own service** | Your FastAPI/Flask/worker process calling `graph.ainvoke` | `AsyncPostgresSaver` / `AsyncPostgresStore` you configure | Everything, including the HTTP API, queueing and streaming |

The first three run the same server and share one configuration file, `langgraph.json`. The server adds threads, assistants (versioned graph configurations), background runs with a task queue, streaming endpoints, cron jobs, webhooks, double-texting strategies, and LangSmith Studio support. In a server deployment you compile graphs **without** a checkpointer or store; the server supplies them.

Choose "your own service" when you need a different HTTP API or must run inside existing infrastructure, and accept that you then build the run management yourself.

## Project layout and `langgraph.json`

```text
my-app/
├── langgraph.json
├── pyproject.toml          # or requirements.txt
├── .env
└── src/agent/
    ├── graph.py            # defines `graph`
    └── auth.py             # optional custom auth
```

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agent/graph.py:graph"
  },
  "env": ".env",
  "python_version": "3.12",
  "image_distro": "wolfi",
  "store": {
    "index": {"embed": "openai:text-embedding-3-small", "dims": 1536, "fields": ["$"]},
    "ttl": {"refresh_on_read": true, "default_ttl": 10080, "sweep_interval_minutes": 60}
  },
  "checkpointer": {
    "ttl": {"strategy": "delete", "default_ttl": 43200, "sweep_interval_minutes": 60}
  },
  "auth": {"path": "./src/agent/auth.py:auth"},
  "http": {"cors": {"allow_origins": ["https://app.example.com"]}}
}
```

| Key | Purpose |
|---|---|
| `dependencies` | Packages to install: `"."` (your project), local paths or package names |
| `graphs` | Graph ID → `path:variable`. The variable is a compiled graph, or a function that returns one (it can take a `RunnableConfig` to build the graph per run). |
| `env` | A `.env` path or a mapping of environment variables |
| `python_version` | `3.11`, `3.12` or `3.13` (default `3.11`) |
| `image_distro` | Base image distribution: `debian` (default), `wolfi` or `bookworm`. `bullseye` was removed in CLI 0.4.13 and is now rejected. |
| `store.index` | Semantic search over the store (embedding model, dimensions, fields to embed) |
| `store.ttl`, `checkpointer.ttl` | Expiry for store items and checkpoints. `default_ttl` is in **minutes**. |
| `checkpointer.path` | Import path to an `@asynccontextmanager` function that yields your own `BaseCheckpointSaver`, replacing the server's Postgres checkpointer. `checkpointer.ttl.strategy` is `"delete"` or `"keep_latest"`. |
| `auth.path` | A `langgraph_sdk.Auth` instance for custom authentication |
| `http` | CORS, a custom Starlette/FastAPI `app` mounted next to the server's routes, and flags to disable route groups |
| `dockerfile_lines`, `pip_config_file` | Extra Dockerfile lines; a pip config for private indexes |
| `ui`, `webhooks`, `encryption` | Generative UI: a name → JS/TS file mapping for the components that `push_ui_message` refers to ([Chapter 6](/langgraph-guide/python/chapter-06-streaming-and-debugging/#connecting-to-a-react-front-end)); webhook settings; custom encryption for stored data |

The full list is in the [CLI configuration reference](https://docs.langchain.com/langsmith/cli). Check the file with `langgraph validate`.

## The `langgraph` CLI

Install with `pip install -U "langgraph-cli[inmem]"` (the `inmem` extra is needed for `langgraph dev`).

| Command | Use |
|---|---|
| `langgraph new --template new-langgraph-project-python my-app` | New project from a template (also `agent-python`, `deep-agent-python` and JS templates) |
| `langgraph dev` | Development server in-process, no Docker, hot reload, default port 2024; opens LangSmith Studio. `--tunnel` exposes it for a remote front end. Persistence is local and not for production. |
| `langgraph validate` | Validate `langgraph.json` |
| `langgraph build -t my-agent:1.4.0` | Build the server image with your code |
| `langgraph up` | Run the image with Postgres and Redis in Docker, default port 8123. Needs a LangSmith API key for local use and a licence key for production. `--postgres-uri` points it at an existing database. |
| `langgraph dockerfile Dockerfile --add-docker-compose` | Write a Dockerfile (and compose files) to build yourself |
| `langgraph deploy` | (Beta) Build and deploy to LangSmith Deployment |
| `langgraph deploy list`, `deploy logs`, `deploy revisions list`, `deploy delete` | (Beta) Inspect and manage deployments |

`langgraph deploy` reads the API key from `--api-key`, `LANGSMITH_API_KEY` or `.env`, and the deployment name from `--name` (default: the directory name). `--deployment-type` is `serverless` (default, scales to zero) or `dedicated`. For self-hosted or hybrid LangSmith, `--push-to` pushes the image to your own registry and `--image-uri` deploys an image that is already there.

You can also deploy from a GitHub repository through the LangSmith UI; see the [LangSmith Deployment docs](https://docs.langchain.com/langsmith/deployment).

## Standalone Agent Server

For a server without the LangSmith control plane, build the image with `langgraph build` and run it with:

| Variable | Purpose |
|---|---|
| `DATABASE_URI` | Postgres: threads, runs, assistants, checkpoints, store, and the task queue |
| `REDIS_URI` | Redis: pub/sub for streaming output from background runs |
| `LANGGRAPH_CLOUD_LICENSE_KEY` | Licence key, checked at start-up |
| `LANGSMITH_API_KEY`, `LANGSMITH_ENDPOINT` | Tracing to LangSmith (Cloud or self-hosted) |

LangChain recommends Kubernetes with its Helm chart for production standalone servers; plain Docker leaves queue autoscaling, run draining on shutdown and split API/queue services to you. Do not run the standalone server on scale-to-zero platforms. Details: [Self-host standalone servers](https://docs.langchain.com/langsmith/deploy-standalone-server).

## Calling a deployment

With the SDK (`pip install langgraph-sdk`):

```python
import asyncio
from langgraph_sdk import get_client


async def main() -> None:
    client = get_client(url="http://localhost:2024")      # api_key= for a LangSmith deployment
    thread = await client.threads.create()
    async for part in client.runs.stream(
        thread["thread_id"],
        "agent",                                          # graph ID from langgraph.json
        input={"messages": [{"role": "user", "content": "Hello"}]},
        stream_mode="updates",
    ):
        print(part.event, part.data)


asyncio.run(main())
```

Each `part` has `.event` (`"metadata"` first, then the stream mode) and `.data`, which is plain JSON: messages are dicts. `client.runs.wait(...)` returns the final state instead; `client.runs.create(...)` starts a background run; `client.threads.get_state(thread_id)` reads state; `client.runs.stream(..., command={"resume": ...})` resumes an interrupt.

To use a deployed graph like a local one (including as a node in another graph), use [`RemoteGraph`](/langgraph-guide/python/reference-remote-graph/).

## Authentication

The server's default auth is a LangSmith API key in the `x-api-key` header. For your own users, point `auth.path` at a `langgraph_sdk.Auth` object:

```python
# src/agent/auth.py
from langgraph_sdk import Auth

auth = Auth()

VALID_TOKENS = {"token-alice": "alice"}      # replace with a call to your identity provider


@auth.authenticate
async def authenticate(authorization: str | None) -> Auth.types.MinimalUserDict:
    token = (authorization or "").removeprefix("Bearer ")
    if token not in VALID_TOKENS:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token")
    return {"identity": VALID_TOKENS[token]}


@auth.on
async def owner_only(ctx: Auth.types.AuthContext, value: dict) -> dict:
    """Tag every resource with its owner and only return the caller's own."""
    value.setdefault("metadata", {})["owner"] = ctx.user.identity
    return {"owner": ctx.user.identity}
```

The handler returned by `@auth.on` is a filter applied to threads, runs, assistants and store items. See the [authentication guide](https://docs.langchain.com/langsmith/auth) for per-resource handlers.

## Self-hosting in your own service

When you run graphs inside your own web service, open the Postgres saver and store once at start-up and close them at shutdown. Compile the graph with them and pass a `thread_id` per conversation.

```python
# app.py: pip install fastapi uvicorn langgraph-checkpoint-postgres "psycopg[binary,pool]"
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain.agents import create_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

DB_URI = os.environ["DATABASE_URL"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as saver, \
               AsyncPostgresStore.from_conn_string(DB_URI) as store:
        await saver.setup()          # idempotent; creates or migrates tables
        await store.setup()
        app.state.agent = create_agent(
            "anthropic:%%model=anthropic%%", tools=[],
            checkpointer=saver, store=store,
        )
        yield


app = FastAPI(lifespan=lifespan)


@app.post("/chat/{thread_id}")
async def chat(thread_id: str, message: str) -> dict:
    result = await app.state.agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        {"configurable": {"thread_id": thread_id}, "recursion_limit": 25},
    )
    return {"reply": result["messages"][-1].content}
```

`from_conn_string` opens a single connection. Under concurrent load, create an `AsyncConnectionPool` from `psycopg_pool` (with `autocommit=True, prepare_threshold=0, row_factory=dict_row` in its `kwargs`) and pass it to `AsyncPostgresSaver(pool)`; see the [checkpointers reference](/langgraph-guide/python/reference-checkpointers/). Token streaming over SSE is shown in the [FastAPI streaming server](/langgraph-guide/python/langgraph_streaming_server_fastapi/) page. For clean shutdown between steps, see [graceful shutdown](/langgraph-guide/python/reference-graceful-shutdown/).

## Data management

- **Growth.** With the default `durability="async"`, every super-step writes a checkpoint. Use `checkpointer.ttl` (server) or delete old threads (`saver.adelete_thread(thread_id)`) in your own service. `durability="exit"` writes far fewer checkpoints for runs that do not need mid-run recovery.
- **Store expiry.** `PostgresStore(..., ttl={"default_ttl": minutes, "sweep_interval_minutes": 60})`, then `store.start_ttl_sweeper()` in your own service; the server runs the sweeper for you when `store.ttl` is set. `InMemoryStore` does not support TTL.
- **Encryption at rest.** `EncryptedSerializer.from_pycryptodome_aes()` (reads a 16-, 24- or 32-byte key from `LANGGRAPH_AES_KEY`) passed as `serde=` to the saver encrypts checkpoint payloads. Details in the [checkpointers reference](/langgraph-guide/python/reference-checkpointers/).
- **Backups.** Everything lives in Postgres; back it up as you would any application database.

## Capacity, latency and cost

This guide does not quote throughput or latency figures: they depend on the model, the number of model calls per run, tool latency and the database. Measure with your own graph: run a representative set of inputs against `langgraph up` or a staging deployment and read per-node timings from LangSmith traces.

What usually matters, in order:

1. **Model calls per run.** Cap them (`recursion_limit`, `ModelCallLimitMiddleware`) and keep context short (`SummarizationMiddleware`, `ContextEditingMiddleware`).
2. **Parallelism.** Independent steps run in the same super-step when they share a parent; `Send` fans out work. See [Performance](/langgraph-guide/python/langgraph_performance_optimization/).
3. **Checkpoint writes.** `durability="exit"` for short runs that need no mid-run recovery.
4. **Worker concurrency.** On the server, `--n-jobs-per-worker` (`langgraph dev`) and the Helm chart's `numberOfJobsPerWorker` set how many runs a worker executes at once.

For prices, use the providers' own pages ([Anthropic](https://www.anthropic.com/pricing), [OpenAI](https://openai.com/api/pricing/)); LangSmith shows token use and cost per trace.

## Release checklist

- `langgraph validate` passes; the image builds with `langgraph build`.
- Tests run against a scripted model ([Chapter 10](/langgraph-guide/python/chapter-10-production/#testing-with-scripted-models)).
- State schema changes are backward-compatible with existing threads: add fields with defaults rather than renaming or removing them, because old checkpoints are loaded into the new schema.
- `recursion_limit` and node policies set on long-running agents.
- Tracing on (`LANGSMITH_TRACING=true`) and sensitive node inputs trimmed with [`TracePolicy`](/langgraph-guide/python/reference-trace-policy/).
- On LangSmith Deployment, `langgraph deploy revisions list` shows earlier revisions to roll back to.

---
title: "Haystack Production Deployment and Operations Guide"
description: "Running Haystack 3 in production — readiness checks, serialised pipelines, serving with FastAPI or Hayhooks, security, cost controls, testing and evaluation, scaling and caching."
framework: "haystack"
language: "python"
sidebar:
  label: "Production"
  order: 7
---

> Verified against `haystack-ai` %%version=haystack-ai%% (checked %%versions-checked%%).

## Table of Contents

1. [Production Readiness Checklist](#production-readiness-checklist)
2. [Deployment Strategies](#deployment-strategies)
3. [Containers and Kubernetes](#containers-and-kubernetes)
4. [API Service Development](#api-service-development)
5. [Security](#security)
6. [Cost Controls](#cost-controls)
7. [Testing and Evaluation](#testing-and-evaluation)
8. [Scaling Strategies](#scaling-strategies)
9. [Caching and Performance](#caching-and-performance)

For tracing, logging, retries, fallbacks and dead letters, continue in [Observability](/haystack-guide/haystack_observability_python/).

---

## Production Readiness Checklist

### Before you deploy

| Area | Check |
|------|-------|
| Versions | Pin `haystack-ai` and every integration package; Haystack 3 integrations are released independently of the core |
| Secrets | API keys passed as `Secret.from_env_var(...)`, never as strings in code or YAML |
| Start-up | `pipeline.warm_up()` at start-up so missing keys and unreachable stores fail fast (from 3.0, keys are resolved at warm-up, not construction) |
| Shutdown | `pipeline.close()` / `await pipeline.close_async()` to release HTTP clients and connections |
| Serialisation | Pipelines loaded from YAML use an explicit `allowed_modules` list, never `unsafe=True` for files you did not write |
| Prompts | Template variables are required by default; integration tests cover every input path |
| Agents | `max_agent_steps` set deliberately; `exit_reason` logged; tools with side effects behind a confirmation hook |
| Security | Privileged IDs reach tools through `State` parameters, not `inputs_from_state` or the prompt (see [Security](#security)) |
| Cost | A `TokenBudgetHook` per agent; token usage logged per request (see [Cost controls](#cost-controls)) |
| Tests | Pipelines and agents exercised in CI with `MockChatGenerator`; evaluators run on a golden set (see [Testing and evaluation](#testing-and-evaluation)) |
| Tracing | A tracer is enabled explicitly (nothing is auto-enabled in 3.x); content tracing only where PII rules allow |
| Telemetry | `HAYSTACK_TELEMETRY_ENABLED=False` set in the environment if your policy forbids anonymous usage statistics |
| Logging | Decide whether Haystack's handler should propagate to your root logger (see [Observability](/haystack-guide/haystack_observability_python/#logging)) |
| Data | If you index with auto-generated IDs and upgraded from 2.x, re-index documents with non-empty `meta` (their IDs changed) |

### A start-up check

```python
import logging
import os

from haystack import Pipeline

logger = logging.getLogger(__name__)

REQUIRED_ENV = ["OPENAI_API_KEY"]


def check_ready(pipeline: Pipeline) -> None:
    missing = [v for v in REQUIRED_ENV if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

    # Creates API clients and connects to stores; raises on bad credentials or config
    pipeline.warm_up()

    # Connections are type-checked by connect(); log the entry points and outputs for the record
    logger.info("Pipeline ready: inputs=%s outputs=%s", pipeline.inputs(), pipeline.outputs())
```

For a deeper smoke test, run the pipeline once in CI with `MockChatGenerator` in place of the real generator; see [Testing and evaluation](#testing-and-evaluation).

---

## Deployment Strategies

### Pipelines as configuration

Build pipelines in code, serialise them to YAML, and ship the YAML with the release. The same artefact then runs in every environment:

```python
from haystack import Pipeline

# build.py — run in CI
pipeline = build_rag_pipeline()          # your factory function
with open("pipelines/rag.yaml", "w") as fp:
    pipeline.dump(fp)
```

```python
# service start-up
from haystack import Pipeline

with open("pipelines/rag.yaml") as fp:
    pipeline = Pipeline.load(fp, allowed_modules=["myapp.components.*"])
```

Components from `haystack`, `haystack_integrations` and `haystack_experimental` are trusted by default. Your own components and tool functions must be allowed per call (as above), process-wide with `haystack.core.serialization.allow_deserialization_module("myapp.*")`, or through the `HAYSTACK_DESERIALIZATION_ALLOWLIST` environment variable.

### Releasing changes

Keep the service stateless (conversation history and documents in external stores) and any blue-green or canary set-up your platform offers will work. When comparing a canary with the current version, look at the `exit_reason` distribution, steps per run and token usage as well as error rate and latency. Treat a model change, a prompt change and a Haystack upgrade as three separate releases; each can change behaviour on its own.

---

## Containers and Kubernetes

Nothing about a Haystack service is unusual to containerise. A minimal image:

```dockerfile
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 HAYSTACK_TELEMETRY_ENABLED=False
RUN useradd -m -u 1000 haystack
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=haystack:haystack . .
USER haystack
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`requirements.txt` pins `haystack-ai==%%version=haystack-ai%%` and the exact versions of every integration package you tested (`elasticsearch-haystack`, `opentelemetry-haystack` and so on), since integrations are released independently of the core. The base image must be Python 3.11 or later.

The Haystack-specific points for Kubernetes or any other orchestrator:

- **Configuration.** Put `HAYSTACK_DESERIALIZATION_ALLOWLIST` (for pipelines loaded from YAML), `HAYSTACK_TELEMETRY_ENABLED` and, only if you have cleared it for PII, `HAYSTACK_CONTENT_TRACING_ENABLED` in the environment; all three are read at start-up or import. API keys come from a secret and are read by `Secret.from_env_var`.
- **Readiness.** Call `pipeline.warm_up()` (or `agent.warm_up()`) during start-up, before the readiness probe passes, so a bad key or unreachable store keeps the pod out of rotation.
- **Shutdown.** Call `pipeline.close()` / `await pipeline.close_async()` on termination to release HTTP clients and store connections.
- **Scaling signal.** LLM-bound services spend most of their time waiting on the network, so CPU is a weak signal. Scale on in-flight requests or latency if you can.
- **Ingestion.** Run indexing pipelines as batch jobs, separate from the query service.

For SSE streaming behind a proxy, see the [streaming server](/haystack-guide/haystack_streaming_server_fastapi/) page.

---

## API Service Development

### FastAPI Integration

Build the agent once at start-up, warm it up, and call `run_async` per request so one worker can serve many concurrent requests.

```python
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import tool

logger = logging.getLogger(__name__)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}"


class Query(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None


class AgentResponse(BaseModel):
    answer: str
    exit_reason: str | None
    steps: int
    tool_calls: dict[str, int]
    total_tokens: int | None
    seconds: float


agent: Agent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%", timeout=60, max_retries=2),
        tools=[get_weather],
        system_prompt="You are a helpful assistant.",
        max_agent_steps=10,
    )
    agent.warm_up()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=AgentResponse)
async def query(q: Query) -> AgentResponse:
    start = time.perf_counter()
    try:
        result = await agent.run_async(messages=[ChatMessage.from_user(q.text)])
    except Exception:
        logger.exception("Agent run failed")
        raise HTTPException(status_code=502, detail="Upstream model error")
    return AgentResponse(
        answer=result["last_message"].text or "",
        exit_reason=result.get("exit_reason"),
        steps=result["step_count"],
        tool_calls=result["tool_call_counts"],
        total_tokens=result["token_usage"].get("total_tokens"),
        seconds=round(time.perf_counter() - start, 3),
    )
```

For token streaming over server-sent events, see the [streaming server](/haystack-guide/haystack_streaming_server_fastapi/) page.

### Serving with Hayhooks

deepset's **Hayhooks** (`pip install hayhooks`) serves pipelines and agents without a hand-written FastAPI layer. Each deployment is a folder with a `pipeline_wrapper.py` that defines a `PipelineWrapper`:

```python
# pipelines/weather_agent/pipeline_wrapper.py
from collections.abc import AsyncGenerator

from hayhooks import BasePipelineWrapper, async_streaming_generator, get_last_user_message
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}"


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
            tools=[get_weather],
            system_prompt="You are a helpful assistant.",
            max_agent_steps=10,
        )

    async def run_api_async(self, question: str) -> str:
        """Answer a question, calling tools where they help."""
        result = await self.agent.run_async(messages=[ChatMessage.from_user(question)])
        return result["last_message"].text

    async def run_chat_completion_async(self, model: str, messages: list[dict], body: dict) -> AsyncGenerator:
        question = get_last_user_message(messages)
        return async_streaming_generator(
            pipeline=self.agent,
            pipeline_run_args={"messages": [ChatMessage.from_user(question)]},
        )
```

`hayhooks run --pipelines-dir pipelines` starts the server (port 1416 by default) and deploys every folder. The request model for `POST /weather_agent/run` is generated from the `run_api_async` signature (`{"question": "..."}`), and `run_chat_completion_async` adds an OpenAI-compatible `POST /v1/chat/completions` endpoint (use the folder name as `model`) that streams tokens and tool calls. With `pip install "hayhooks[mcp]"`, `hayhooks mcp run` exposes the same wrappers as MCP tools, using the docstring of `run_api_async` (or `run_api`) as the tool description. The endpoints are not authenticated by default, so put the server behind your gateway.

---

## Security

Haystack gives you the pieces; how strict to be depends on your platform. The framework-specific points:

- **Tenant and user IDs.** Pass them to `agent.run(...)` as state values and read them in tools through a parameter typed as `State`, which the Agent always injects. Do not rely on `inputs_from_state` for this: arguments the model supplies override it, so a prompt-injected call can name another tenant. Hooks read request-scoped values from `hook_context`. The [multi-tenant recipe](/haystack-guide/haystack_recipes/#multi-tenant-customer-support-agent) shows the pattern.
- **Side effects.** Put a `ConfirmationHook` (`haystack.hooks.human_in_the_loop`) in front of tools that move money, send messages or delete data, with a `ConfirmationUI` that asks through your front end. From 3.0 the confirmation sees only the arguments the model asked for, which is what the user should approve.
- **Tool surface.** Give each agent only the tools it needs. `MCPToolset(..., tool_names=[...])` limits what an MCP server exposes; tools that read files, run SQL or fetch URLs should enforce their own allow-lists, because retrieved documents and web pages can carry injected instructions.
- **Deserialisation.** Load YAML pipelines with an explicit `allowed_modules` list (or `HAYSTACK_DESERIALIZATION_ALLOWLIST`), never `unsafe=True` for files you did not write.
- **Offloaded tool results.** A `ToolResultOffloadHook` store holds full tool output; give each run its own store through `hook_context` so users cannot read each other's results.
- **Traces.** Content tracing puts prompts, documents and tool results into your tracing backend. Leave `HAYSTACK_CONTENT_TRACING_ENABLED` off where that data is personal, or redact in the exporter.
- **Telemetry.** Haystack sends anonymous usage statistics by default: component types and, for agents, the generator, model and tool names (never prompts, documents or tool descriptions). Set `HAYSTACK_TELEMETRY_ENABLED=False` in the environment to opt out; it is read when `haystack` is imported.
- **Secrets.** Use `Secret.from_env_var(...)`. `Pipeline.dumps()` writes the variable name, not the key, so serialised pipelines are safe to commit.

Authentication, rate limiting and network policy belong to your API gateway or service layer.

## Cost Controls

- **Bound every agent.** `max_agent_steps` limits iterations; [`TokenBudgetHook`](/haystack-guide/haystack_advanced_agents_python/#token-budgets) limits tokens and ends the run with `exit_reason="token_budget_exceeded"`. Sub-agents behind `AgentTool` need their own budget, because a coordinator's `token_usage` counts only its own model calls.
- **Keep the context small.** [`CompactionHook` and `ToolResultOffloadHook`](/haystack-guide/haystack_comprehensive_guide/#context-management) stop long runs re-sending large tool results and old turns; `SearchableToolset` stops large tool catalogues being sent on every call.
- **Use the cheap model where it is enough.** Routing, classification (`LLMMessagesRouter`), narrow specialists and summarisation compactors rarely need the flagship model.
- **Measure.** Log `token_usage`, `step_count`, `tool_call_counts` and `exit_reason` per request (the FastAPI example above returns them), and alert on drift after each release.
- **Cache** repeated questions and unchanged documents (below).

## Testing and Evaluation

### Unit tests with MockChatGenerator

`MockChatGenerator` (`haystack.components.generators.chat`) is a drop-in chat generator that never calls an API. Give it a list of replies, including `ChatMessage`s with tool calls, to drive an agent through a known path, then assert on tools, state and `exit_reason`:

```python
# test_agent.py — run with pytest
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}"


def build_agent(chat_generator) -> Agent:          # the factory your service uses
    return Agent(chat_generator=chat_generator, tools=[get_weather], max_agent_steps=5)


def test_agent_calls_weather_tool():
    llm = MockChatGenerator(responses=[
        ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="get_weather", arguments={"city": "Paris"})]),
        "It is sunny in Paris.",
    ])
    result = build_agent(llm).run(messages=[ChatMessage.from_user("Weather in Paris?")])

    assert result["exit_reason"] == "text"
    assert result["tool_call_counts"] == {"get_weather": 1}
    tool_result = next(m.tool_call_result for m in result["messages"] if m.tool_call_result)
    assert tool_result.result == "Sunny in Paris"
```

`MockChatGenerator(response_fn=...)` computes each reply from the input messages instead, and with no arguments it echoes the last message, which is enough to smoke-test that every connection and prompt variable in a pipeline is wired. Set `HAYSTACK_TELEMETRY_ENABLED=False` in the CI environment.

### Evaluating quality

The core package ships evaluators in `haystack.components.evaluators`: retrieval metrics (`DocumentMRREvaluator`, `DocumentRecallEvaluator`, `DocumentMAPEvaluator`, `DocumentNDCGEvaluator`), answer metrics (`AnswerExactMatchEvaluator`, `SASEvaluator`) and model-judged metrics (`FaithfulnessEvaluator`, `ContextRelevanceEvaluator`, and `LLMEvaluator` for your own rubric). Run them over a small golden set of questions with known relevant documents:

```python
from haystack import Document
from haystack.components.evaluators import DocumentMRREvaluator, FaithfulnessEvaluator
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.evaluation import EvaluationRunResult

docs = [Document(content="Returns are accepted within 30 days."), Document(content="Shipping takes 5-7 business days.")]
store = InMemoryDocumentStore()
store.write_documents(docs)
retriever = InMemoryBM25Retriever(document_store=store, top_k=2)

golden = [  # question and the document that should be retrieved for it
    ("How many days do I have for returns?", docs[0]),
    ("How long does shipping take?", docs[1]),
]
questions = [q for q, _ in golden]
retrieved = [retriever.run(query=q)["documents"] for q in questions]
answers = ["You have 30 days.", "Five to seven business days."]   # your pipeline's answers

mrr = DocumentMRREvaluator().run(
    ground_truth_documents=[[d] for _, d in golden], retrieved_documents=retrieved
)
faithfulness = FaithfulnessEvaluator(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"), progress_bar=False
).run(questions=questions, contexts=[[d.content for d in ds] for ds in retrieved], predicted_answers=answers)

report = EvaluationRunResult(
    run_name="rag-baseline",
    inputs={"question": questions, "answer": answers},
    results={"mrr": mrr, "faithfulness": faithfulness},
)
print(report.aggregated_report())          # one score per metric; detailed_report() gives per-question rows
```

Each evaluator returns an overall `score` and `individual_scores`. In CI, run the golden set on every change to prompts, models or retrieval settings and fail the job when a score drops below your baseline. The model-judged evaluators make one model call per item, so keep the golden set small for per-commit runs and run a larger one nightly.

---

## Scaling Strategies

### Horizontal Scaling with Load Balancing

- **Keep workers stateless.** Store conversation history (`ChatMessage.to_dict()`), user state and documents outside the process so any replica can serve any request.
- **Prefer async.** `Agent.run_async` and `Pipeline.run_async` let one Uvicorn worker overlap many LLM calls. Use a few workers per pod (`--workers 2-4`) for CPU-bound pre-processing.
- **Bound concurrency.** `Pipeline.run_async(..., concurrency_limit=N)` limits parallel components per request; `tool_concurrency_limit` limits parallel tool calls per agent step. Add a semaphore at the API layer to protect provider rate limits.
- **Separate indexing from querying.** Run indexing pipelines as batch jobs (Kubernetes `Job`/`CronJob`) so ingestion spikes do not affect query latency.
- **Spread provider risk.** `FallbackChatGenerator` fails over between providers or regions; see [Retries and fallbacks](/haystack-guide/haystack_observability_python/#retries-and-fallbacks).

```python
import asyncio

MAX_IN_FLIGHT = asyncio.Semaphore(32)   # per process

async def run_limited(messages):
    async with MAX_IN_FLIGHT:
        return await agent.run_async(messages=messages)
```

---

## Caching and Performance

### Multi-Layer Caching Strategy

1. **Response cache** for repeated identical questions (Redis, short TTL).
2. **Embedding / indexing cache** so unchanged documents are not re-embedded: `CacheChecker` looks up documents by a metadata field and splits inputs into hits and misses.
3. **Provider-side prompt caching**: keep long system prompts and tool schemas stable and at the start of the prompt so providers can reuse them.

```python
import hashlib
import json

from redis.asyncio import Redis

redis = Redis.from_url("redis://localhost:6379/0", decode_responses=True)
TTL_SECONDS = 3600


def cache_key(text: str, model: str = "%%model=openai-small%%", prompt_version: str = "v3") -> str:
    digest = hashlib.sha256(f"{model}|{prompt_version}|{text.strip().lower()}".encode()).hexdigest()
    return f"answer:{digest}"


async def cached_answer(text: str) -> dict:
    key = cache_key(text)
    if (hit := await redis.get(key)) is not None:
        return {**json.loads(hit), "from_cache": True}

    result = await run_limited([ChatMessage.from_user(text)])
    payload = {"answer": result["last_message"].text, "exit_reason": result.get("exit_reason")}
    if result.get("exit_reason") == "text":          # only cache clean completions
        await redis.set(key, json.dumps(payload), ex=TTL_SECONDS)
    return {**payload, "from_cache": False}
```

Include the model name and a prompt version in the key so a deployment that changes either does not serve stale answers. Do not cache answers that depend on user-specific tool results.

Skipping already-indexed files with `CacheChecker`:

```python
from haystack import Document
from haystack.components.caching import CacheChecker
from haystack.document_stores.in_memory import InMemoryDocumentStore

store = InMemoryDocumentStore()
store.write_documents([Document(content="old", meta={"url": "https://example.com/a"})])

checker = CacheChecker(document_store=store, cache_field="url")
result = checker.run(items=["https://example.com/a", "https://example.com/b"])
print(result["misses"])   # ['https://example.com/b'] — only these need fetching and indexing
```

---

**Not covered here.** Rate limiting, database operations, governance and disaster recovery depend on your platform more than on Haystack. For tracing and logging see [Observability](/haystack-guide/haystack_observability_python/).

---
title: "Agno: production guide"
description: "Running Agno in production: deploying AgentOS, choosing and migrating databases, authentication, tracing and metrics, limits and fallbacks, keeping context small, evals, and testing agents offline with a scripted model."
framework: "agno"
language: "python"
sidebar:
  label: "Production"
  order: 50
---

Checked against **agno %%version=agno%%** (checked %%versions-checked%%). The tracing, eval and testing code ran offline with SQLite and a scripted model. Postgres, container deployment and the third-party tracing exporters were not run; those sections describe the documented API and say so.

## Deploying AgentOS

An AgentOS app is a FastAPI app, so it deploys like one. The usual shape:

```python
# app.py
from os import getenv

from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.models.openai import OpenAIResponses
from agno.os import AgentOS

db = PostgresDb(db_url=getenv("DATABASE_URL", "postgresql+psycopg://ai:ai@localhost:5432/ai"))

support = Agent(
    id="support",
    name="Support",
    model=OpenAIResponses(id="%%model=openai%%"),
    add_history_to_context=True,
    num_history_runs=5,
    telemetry=False,
)

agent_os = AgentOS(id="support-os", db=db, agents=[support], tracing=True, telemetry=False)
app = agent_os.get_app()
```

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "agno[os,postgres,openai]"
COPY app.py .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7777", "--workers", "2"]
```

The Dockerfile above was not built for this guide. Agno publishes starter repositories with Postgres, Docker Compose and deploy scripts (`agno-agi/agentos-docker`, `agentos-railway`, `agentos-aws`, `agentos-gcp`, `agentos-azure`, `agentos-fly`, `agentos-render`, `agentos-modal`, `agentos-helm`); `agno create` scaffolds one and `agno up` starts it with Docker Compose. Pin the `agno` version in your requirements file so a deploy does not pick up a new release unexpectedly.

Things to settle before going live:

- **Workers and state.** Runs are stored in the database, so any worker can serve any session. Background runs, cancellation and SSE reconnection are per process unless you configure `AgentOS(queue=QueueConfig(durable=True, ...))` and/or `redis=` (see [AgentOS](/agno-guide/agno_agentos/#background-runs-and-the-queue)).
- **MCP toolkits** are connected in the app's lifespan when served by AgentOS; a slow or unreachable MCP server delays start-up.
- **Health checks**: `GET /health` is never behind authentication.
- **Telemetry**: `AGNO_TELEMETRY=false` turns off Agno's usage event everywhere.

## Databases

Use Postgres (or another server database) in production; SQLite is fine for development and single-process tools.

| Need | Choice |
|------|--------|
| Default | `PostgresDb(db_url="postgresql+psycopg://...")` from `agno.db.postgres` (`pip install "agno[postgres]"`) |
| Async app using `arun()` everywhere | `AsyncPostgresDb` (same module); `run()` and sync `continue_run()` raise with an async db |
| Separate schema | `PostgresDb(db_url=..., db_schema="agents")` |
| Own table names | `session_table=`, `runs_table=`, `memory_table=`, `eval_table=`, `knowledge_table=`, `traces_table=`, … |
| Shared engine | `PostgresDb(db_engine=engine)` with an engine from `create_postgres_engine` |

One database object can serve every agent, team and workflow (`AgentOS(db=...)` hands it to those without their own). Tables are created on first use (`AgentOS(auto_provision_dbs=True)` is the default).

### Migrations

Agno tracks a schema version per table. After upgrading `agno`, especially across a major version, run the migrations once with the application stopped and a backup taken:

```python
import asyncio

from agno.db.migrations.manager import MigrationManager
from agno.db.sqlite import SqliteDb

db = SqliteDb(db_file="agno.db")   # use the same configuration as the application
asyncio.run(MigrationManager(db).up())
print(db.get_runs(limit=5))
```

AgentOS exposes the same as `POST /databases/{db_id}/migrate` and `POST /databases/all/migrate`. The 2 → 3 migration moves runs out of the session rows into `agno_runs` and adds `user_id` columns; it is non-destructive (the old column is kept until you call the adapter's `cleanup_legacy_runs_column()`), and per-user vector tables need a separate script from the Agno repository. Read the [v3 migration guide](https://docs.agno.com/other/v3-migration) before upgrading a production database.

## Authentication and data isolation

Covered with tested examples in [AgentOS → Authentication and isolation](/agno-guide/agno_agentos/#authentication-and-isolation). For production:

- Verify JWTs from your identity provider with asymmetric keys (`Authorization(jwks_file=...)` or `verification_keys=[public_key]`, algorithm `RS256`), and set `audience=` with `verify_audience=True`.
- Turn on `user_isolation=True` so sessions, memories, knowledge, evals, metrics and schedules are scoped to the token's `sub`.
- Never use `create_dev_token` or a hard-coded HS256 secret outside development.
- If you serve MCP, set `MCPConfig(allowed_hosts=[...])` and an `authorize=` function.
- Destructive tools should use `requires_confirmation=True` (agents) or `HumanReview` (workflows), and guardrail failures must be handled by checking `run.status` (an output-check failure leaves the model's text in `run.content`; see [Hooks and guardrails](/agno-guide/agno_comprehensive_guide/#hooks-and-guardrails)).

## Observability

### Tracing into your database

Agno instruments agents, teams, workflows, model calls and tool calls with OpenTelemetry (through `openinference-instrumentation-agno`, included in `agno[os]`) and can store the spans in its own database:

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.tracing import setup_tracing

db = SqliteDb(db_file="agno.db")
setup_tracing(db=db)   # once, at start-up

agent = Agent(name="Shop", model=OpenAIResponses(id="%%model=openai%%"), db=db)
agent.run("What does the A1 kettle cost?")

traces, total = db.get_traces(limit=5)
print(total, [t.name for t in traces])   # 1 ['Shop.run']
```

`AgentOS(tracing=True)` does the same for everything it serves and adds `GET /traces`, `GET /traces/{trace_id}`, `POST /traces/search` and `GET /trace_session_stats`. The default exports each span as it ends. `batch_processing=True` exports in batches (`max_queue_size`, `max_export_batch_size`, `schedule_delay_millis`, 5 seconds by default), which is cheaper under load, but spans then appear in the database only after the delay, so the `get_traces` call above would return nothing yet.

Because the instrumentation is standard OpenTelemetry, you can also send spans to an external backend by configuring an OTLP exporter yourself; the `agno[opentelemetry]`, `agno[langfuse]`, `agno[arize]`, `agno[weave]` and `agno[openlit]` extras install the client libraries. Those exporters were not exercised here.

### Metrics and logs

- Every run has `run.metrics` (input, output, cached and reasoning tokens, `cost` where the provider reports it, `duration`, `time_to_first_token`); `agent.get_session_metrics(session_id=...)` totals a session.
- AgentOS aggregates daily metrics: `GET /metrics`, `POST /metrics/refresh`.
- `debug_mode=True` (or `AGNO_DEBUG=true`) logs each request and response; `debug_level=2` logs more. Debug output includes prompts and tool results, so keep it out of production logs.
- `store_events=True` keeps the streamed events on the stored run, for auditing what the user saw.

## Limits, retries and fallbacks

| Risk | Setting |
|------|---------|
| Provider errors and rate limits | `Model(retries=..., delay_between_retries=..., exponential_backoff=True)`; `fallback_models=[...]` or `fallback_config=FallbackConfig(on_rate_limit=[...], on_context_overflow=[...], on_error=[...])` on the agent |
| Runaway tool loops | `tool_call_limit=` on the agent or team; `max_iterations=` on a tasks-mode team; `max_iterations=` on a workflow `Loop` |
| Slow tools | `Toolkit(timeout=...)`, `MCPTools(timeout_seconds=...)` |
| Cancelling | `Agent.cancel_run(run_id)`, or `POST /agents/{id}/runs/{run_id}/cancel` |
| Losing work mid-run | `checkpoint="tool-batch"` writes the run to the database after each model turn, not only at the end |

## Keeping context and cost under control

- `num_history_runs` (or `num_history_messages`) and `max_tool_calls_from_history` bound what history is replayed.
- `compress_tool_results=True` (or a `CompressionManager`) has a model shorten long tool results.
- `offload_tool_results=True` stores results longer than 16,000 characters as files and leaves a short reference the model can read back in pages.
- `compaction=True` replaces old messages with a summary once a session grows past a threshold.
- `store_history_messages` is `False` by default, so each stored run holds only its own messages and history is rebuilt from earlier runs (storage grows linearly; `True` stores the full context in every run and grows quadratically). `store_tool_messages=False` (default `True`) additionally leaves tool results out of stored runs.
- `Model(cache_response=True)` caches identical requests on disk: useful in development and tests, not for live traffic.

These options were checked to exist on `Agent`; their effect on real conversations was not measured for this guide.

## Evals

`agno.eval` has four evaluators and a suite runner. Results can be stored in the `db` (and appear under `/eval-runs` in AgentOS).

| Class | Checks | Needs a judge model |
|-------|--------|---------------------|
| `ReliabilityEval` | The expected tools were called (optionally with the expected arguments) | No |
| `AccuracyEval` | The answer matches an expected output (score 1–10) | Yes |
| `AgentAsJudgeEval` | The answer meets free-text criteria (pass/fail or score) | Yes |
| `PerformanceEval` | Run time and memory of a function over many iterations | No |

```python
from agno.agent import Agent
from agno.eval import Case, ReliabilityEval, run_cases
from agno.models.openai import OpenAIResponses


def get_price(sku: str) -> str:
    """Look up the price of a product."""
    return "£4.00"


shop = Agent(name="Shop", model=OpenAIResponses(id="%%model=openai%%"), tools=[get_price])

# One check on one run
result = ReliabilityEval(agent_response=shop.run("Price of A1?"), expected_tool_calls=["get_price"]).run()
print(result.eval_status)   # PASSED

# A suite: each case runs the agent, then applies its checks
suite = run_cases(
    [
        Case(
            name="price",
            input="How much is product A1?",
            agent=shop,
            expected_tool_calls=("get_price",),
            criteria="States the price in pounds",
            judge_model=OpenAIResponses(id="%%model=openai-small%%"),
        ),
    ]
)
print(suite.status, suite.passed, suite.failed)
for case in suite.results:
    print(case.name, case.judge_passed, case.reliability_passed, case.tools_called)
```

`SuiteResult.to_dict()` is meant for CI. `agno.eval.cli(cases)` wraps a suite in a command-line runner with tag filtering. Treat judged evals as a regression signal, not a proof: run them on a fixed set of cases before each release and watch for changes.

## Testing without a provider

Agno has no built-in fake model, but `agno.models.base.Model` is small to subclass. This scripted model is what every example in this guide was run against. Each call pops the next reply; a reply is text, a tool call, or a function of the messages:

```python
# tests/scripted_model.py
import json
from dataclasses import dataclass, field
from typing import Any, Callable, List, Union

from agno.models.base import Model
from agno.models.response import ModelResponse

Reply = Union[str, dict, Callable[..., Union[str, dict]]]


def tool_call(name: str, call_id: str = "call_1", **args: Any) -> dict:
    """A reply asking Agno to run one tool."""
    return {"tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}


@dataclass
class ScriptedModel(Model):
    id: str = "scripted"
    name: str = "ScriptedModel"
    provider: str = "Test"
    replies: List[Reply] = field(default_factory=list)
    calls: List[dict] = field(default_factory=list)   # what Agno sent, for assertions

    def _next(self, messages, **kwargs) -> ModelResponse:
        self.calls.append({"messages": list(messages), **kwargs})
        reply = self.replies.pop(0) if self.replies else "done"
        if callable(reply):
            reply = reply(messages, **kwargs)
        if isinstance(reply, dict):
            return ModelResponse(role="assistant", content=reply.get("content"), tool_calls=reply.get("tool_calls", []))
        return ModelResponse(role="assistant", content=reply)

    def invoke(self, messages, assistant_message=None, **kwargs):
        return self._next(messages, **kwargs)

    async def ainvoke(self, messages, assistant_message=None, **kwargs):
        return self._next(messages, **kwargs)

    def invoke_stream(self, messages, assistant_message=None, **kwargs):
        yield self._next(messages, **kwargs)

    async def ainvoke_stream(self, messages, assistant_message=None, **kwargs):
        yield self._next(messages, **kwargs)

    def _parse_provider_response(self, response, **kwargs):
        return response

    def _parse_provider_response_delta(self, response):
        return response
```

`invoke` and friends receive the message list plus `tools` (the JSON schemas), `response_format` and `tool_choice`, so tests can assert on exactly what Agno sent. Tool calls use the OpenAI shape (`{"id", "type": "function", "function": {"name", "arguments"}}`), which is what the tool loop in the `Model` base class consumes.

Tests with it, using `InMemoryDb` for anything that needs storage:

```python
# tests/test_agent.py
from pydantic import BaseModel

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.run import RunStatus
from agno.tools import tool

from scripted_model import ScriptedModel, tool_call


def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"Sunny in {city}"


def test_tool_is_called_and_result_used():
    model = ScriptedModel(replies=[tool_call("get_weather", city="Leeds"), "It is sunny in Leeds."])
    agent = Agent(model=model, tools=[get_weather], telemetry=False)

    run = agent.run("Weather in Leeds?")

    assert run.content == "It is sunny in Leeds."
    assert [(t.tool_name, t.result) for t in run.tools] == [("get_weather", "Sunny in Leeds")]
    assert model.calls[0]["tools"][0]["function"]["name"] == "get_weather"


class City(BaseModel):
    name: str
    country: str


def test_structured_output():
    agent = Agent(model=ScriptedModel(replies=['{"name": "Leeds", "country": "UK"}']), output_schema=City, telemetry=False)
    assert agent.run("Leeds").content == City(name="Leeds", country="UK")


@tool(requires_confirmation=True)
def delete_file(path: str) -> str:
    """Delete a file."""
    return f"deleted {path}"


def test_confirmation_pauses_and_resumes():
    model = ScriptedModel(replies=[tool_call("delete_file", path="/tmp/x"), "Deleted."])
    agent = Agent(model=model, tools=[delete_file], db=InMemoryDb(), telemetry=False)

    run = agent.run("Delete /tmp/x", session_id="t")
    assert run.status == RunStatus.paused
    run.active_requirements[0].confirm()
    run = agent.continue_run(run_id=run.run_id, requirements=run.requirements, session_id="t")

    assert run.status == RunStatus.completed
    assert run.tools[0].result == "deleted /tmp/x"


def test_history_is_sent():
    model = ScriptedModel(replies=["Hi Ada.", "Ada."])
    agent = Agent(model=model, db=InMemoryDb(), add_history_to_context=True, telemetry=False)
    agent.run("I am Ada", session_id="s")
    agent.run("Who am I?", session_id="s")
    assert [m.role for m in model.calls[1]["messages"]] == ["user", "assistant", "user"]
```

The same model drives teams (script the leader's `delegate_task_to_member(member_id=..., task=...)` call and give each member its own `ScriptedModel`) and workflows. For AgentOS, wrap the app in FastAPI's `TestClient` (as a context manager, so the lifespan runs) and post form data:

```python
from fastapi.testclient import TestClient

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.os import AgentOS

from scripted_model import ScriptedModel


def test_agentos_run(tmp_path):
    agent = Agent(id="assistant", model=ScriptedModel(replies=["Hello."]), telemetry=False)
    app = AgentOS(agents=[agent], db=SqliteDb(db_file=str(tmp_path / "os.db")), telemetry=False).get_app()
    with TestClient(app) as client:
        response = client.post("/agents/assistant/runs", data={"message": "hi", "stream": "false"})
    assert response.status_code == 200
    assert response.json()["content"] == "Hello."
```

Test tools as plain functions too: a function passed in `tools=` is unchanged and can be called directly. A function decorated with `@tool` becomes a `Function` object; call the original through `my_tool.entrypoint(...)`.

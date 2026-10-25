---
title: "CrewAI in production"
description: "CrewAI's own production tooling — limits and guardrails, event listeners, OpenTelemetry, tracing, hooks, token accounting, checkpoints, offline tests with a stub LLM, and deploying with crewai deploy or your own container."
framework: "crewai"
language: "python"
sidebar:
  label: "Production"
  order: 50
---

> **Verified against `crewai` %%version=crewai%%** (checked %%versions-checked%%; source: `crewai/events/`, `crewai/hooks/`, `crewai/execution.py`, `crewai/state/`, `crewai_cli/deploy/`). The Python examples were run against that release with a stub LLM; the pytest files pass offline.

This page covers what CrewAI itself gives you for running crews and flows in production. Generic service plumbing (web frameworks, Kubernetes, queues) is not CrewAI-specific and is only touched on where CrewAI changes the advice.

## Checklist

| Concern | CrewAI feature | Section |
|---|---|---|
| Runaway loops and cost | `max_iter`, `max_rpm`, `max_execution_time`, tool `max_usage_count` | [Limits](#limits-guardrails-and-failure-policy) |
| Bad output | Task `guardrail` / `guardrails`, `output_pydantic` / `response_model` | [Limits](#limits-guardrails-and-failure-policy) |
| Tools that fail quietly | `tool_failure_policy` | [Limits](#limits-guardrails-and-failure-policy) |
| Logs and metrics | Event listeners on `crewai_event_bus` | [Event listeners](#event-listeners) |
| Traces | Your own OpenTelemetry spans from listeners; CrewAI AMP tracing | [Tracing](#tracing) |
| Policy enforcement | `before_tool_call` / `after_llm_call` hooks | [Hooks](#hooks-block-or-rewrite-calls) |
| Cost reporting | `CrewOutput.token_usage`, `LLMCallCompletedEvent.usage` | [Tokens](#token-usage) |
| Crashes mid-run | `Crew(checkpoint=...)`, `@persist` for flows | [Checkpoints](#recovery-with-checkpoints) |
| Regressions | pytest with a stub `BaseLLM`; `crewai test`, `crewai eval` | [Testing](#testing-without-a-model) |
| Shipping | `crewai deploy validate/create/push`, or your own container | [Deploying](#deploying) |

## Configuration and secrets

- Keep provider keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …) in the environment. CrewAI and the provider SDKs read them from there; `LLM(api_key=...)` is only needed for non-standard setups.
- The project scaffold puts a `.env` in the project root and lists it in `.gitignore`. It is for local runs. Never `COPY` it into an image or commit it; inject secrets at runtime (`docker run --env-file`, Kubernetes Secrets, your platform's secret store).
- `crewai deploy create` **reads your local `.env` and sends its variables to CrewAI AMP** as deployment environment variables, after showing you the list to confirm. Check that file before you deploy, or keep production values out of it and set them on the platform.
- Pin the model per agent (`llm=LLM(model="openai/%%model=openai-small%%")`) rather than relying on the default. An agent without `llm` reads the `MODEL` environment variable, then falls back to CrewAI's built-in default (`gpt-4.1-mini` in 1.15.28), which can change between releases. Provider SDKs other than OpenAI are extras: `pip install "crewai[anthropic]"`, `crewai[google-genai]`, `crewai[bedrock]`, `crewai[litellm]`.
- CrewAI sends anonymous usage telemetry by default. Turn it off with `CREWAI_DISABLE_TELEMETRY=true` (or `CREWAI_DISABLE_TRACKING=true`). `OTEL_SDK_DISABLED=true` also works but disables **every** OpenTelemetry SDK in the process, including your own tracing. Leave `Crew(share_crew=False)` (the default) unless you intend to share crew data with CrewAI.

## Limits, guardrails and failure policy

```python
from crewai import Agent, Crew, Task, LLM
from crewai.tasks.task_output import TaskOutput

llm = LLM(model="openai/%%model=openai-small%%", timeout=60)


def under_300_words(output: TaskOutput) -> tuple[bool, str]:
    words = len(output.raw.split())
    if words > 300:
        return False, f"{words} words; cut it to 300 or fewer."
    return True, output.raw


writer = Agent(
    role="Writer",
    goal="Write accurate product summaries",
    backstory="Technical writer for a hardware company.",
    llm=llm,
    max_iter=10,                 # reasoning iterations per task (default 25)
    max_execution_time=120,      # wall-clock seconds per task
    max_retry_limit=2,           # retries after a task-level error
    tool_failure_policy="raise", # a tool that reports failure aborts the task
)

summary = Task(
    description="Summarise {product} for the catalogue.",
    expected_output="A plain-text summary under 300 words.",
    agent=writer,
    guardrail=under_300_words,
    guardrail_max_retries=2,
)

crew = Crew(agents=[writer], tasks=[summary], max_rpm=30)   # requests per minute, whole crew
print(crew.kickoff(inputs={"product": "a USB-C dock"}).raw)
```

- **Guardrails** run after the task; a failure sends the reason back to the agent for another attempt, and `guardrail_max_retries=2` allows three attempts in all before the task raises. On success, return the output (`output.raw`) as the second value; that value becomes the task output. `Agent(guardrail=...)` is not a substitute: it only applies to `Agent.kickoff()`, not to tasks in a crew. See [Task API → Guardrails](/crewai-guide/crewai_task_api_python/#guardrails).
- **`tool_failure_policy`** (`"ignore"`, `"warn"`, `"raise"`) decides what happens when a tool call fails: the tool returned a `ToolFailure`, raised, exceeded its usage cap and so on. Set it on the tool, task, agent or crew (most specific wins); the default is `"warn"`. A tool that merely returns an error string is not treated as failed. See [Tools → Reporting tool failures](/crewai-guide/crewai_tools_python/#reporting-tool-failures).
- **Tool caps**: `@tool("search", max_usage_count=5)` stops a tool after five calls. The count lives on the tool object, so it accumulates across tasks until you call `reset_usage_count()`.
- `LLM(timeout=...)` bounds each model call; `max_execution_time` bounds the whole task.

## Event listeners

Every crew, task, agent, LLM call, tool call, flow method, memory and knowledge operation emits a typed event on `crewai_event_bus`. A listener subclass registers its handlers once and then sees every run in the process:

```python
import json
import logging

from crewai.events import (
    BaseEventListener,
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    LLMCallFailedEvent,
    TaskFailedEvent,
    ToolFailureDetectedEvent,
)

log = logging.getLogger("crew.ops")


def emit(kind: str, **fields) -> None:
    log.info(json.dumps({"event": kind, **fields}, default=str))


class OpsListener(BaseEventListener):
    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_done(source, event):
            emit("crew_completed", crew=event.crew_name, total_tokens=event.total_tokens)

        @crewai_event_bus.on(CrewKickoffFailedEvent)
        def on_crew_failed(source, event):
            emit("crew_failed", crew=event.crew_name, error=event.error)

        @crewai_event_bus.on(TaskFailedEvent)
        def on_task_failed(source, event):
            agent = event.task.agent.role if event.task and event.task.agent else None
            emit("task_failed", task=event.task_name, agent=agent, error=event.error)

        @crewai_event_bus.on(ToolFailureDetectedEvent)
        def on_tool_failure(source, event):
            emit("tool_failure", tool=event.tool_name, agent=event.agent_role,
                 code=event.failure.code, error=event.failure.message, policy=event.policy.value)

        @crewai_event_bus.on(LLMCallFailedEvent)
        def on_llm_failed(source, event):
            emit("llm_failed", model=event.model, error=event.error)


ops_listener = OpsListener()   # instantiate once, at import time, before any kickoff
```

- Task events do not fill in `agent_role` (it is `None` on `TaskStartedEvent`, `TaskCompletedEvent` and `TaskFailedEvent` in 1.15.28), which is why the task handler above reads the agent from `event.task`. Tool events do carry it.
- A tool that raises, or returns a `ToolFailure`, produces a `ToolFailureDetectedEvent` (unless the policy is `"ignore"`); in our runs it was this event, not `ToolUsageErrorEvent`, that fired for a tool exception on the function-calling path.
- Handlers take `(source, event)`. Sync handlers run on a thread pool, so they don't block the agents; call `crewai_event_bus.flush()` when you need them finished (in tests, or before a short-lived process exits).
- Event families include `Crew*`, `Task*`, `Agent*`, `LLMCall*`, `LLMStreamChunkEvent`, `ToolUsage*`, `ToolFailureDetectedEvent`, `MCP*`, `Memory*`, `Knowledge*`, `Flow*`, `MethodExecution*`, `HumanFeedback*`, `Conversation*` and `Checkpoint*`; all are importable from `crewai.events`.
- `crewai_event_bus.scoped_handlers()` is a context manager that removes handlers registered inside it, which keeps tests independent.
- Use `Crew(step_callback=..., task_callback=...)` only for quick per-crew hooks; listeners are the general mechanism.

## Tracing

### Your own OpenTelemetry backend

CrewAI does not install a global OpenTelemetry `TracerProvider`, so spans for your backend come from a listener. This one records a span per task:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from crewai.events import (
    BaseEventListener,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))  # swap for an OTLP exporter
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("my-crew")


class TaskSpans(BaseEventListener):
    def __init__(self):
        self._spans = {}
        super().__init__()

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(TaskStartedEvent)
        def started(source, event):
            self._spans[event.task_id] = tracer.start_span(f"task {event.task_name or event.task_id}")

        @crewai_event_bus.on(TaskCompletedEvent)
        def completed(source, event):
            if span := self._spans.pop(event.task_id, None):
                span.set_attribute("crewai.output_chars", len(event.output.raw))
                span.end()

        @crewai_event_bus.on(TaskFailedEvent)
        def failed(source, event):
            if span := self._spans.pop(event.task_id, None):
                span.set_status(trace.Status(trace.StatusCode.ERROR, event.error))
                span.end()


task_spans = TaskSpans()
```

The same pattern extends to `LLMCallStartedEvent` / `LLMCallCompletedEvent` (keyed by `event.call_id`) and the tool events. Do not set `OTEL_SDK_DISABLED=true` to silence CrewAI's telemetry when you use this; it turns these spans into no-ops too.

### CrewAI AMP tracing

CrewAI can also send execution traces to CrewAI AMP, where `crewai eval` grades them. Enable it with any one of:

- `Crew(tracing=True)` or `Flow(tracing=True)` in code;
- `CREWAI_TRACING_ENABLED=true` in the environment;
- `crewai traces enable` (records your consent in CrewAI's per-user data; `crewai traces status` and `crewai traces disable` also exist).

`tracing=False` or `CREWAI_TRACING_ENABLED=false` turns it off for a run. Traces contain prompts and outputs, so treat this as sending data to a third party.

## Hooks: block or rewrite calls

Hooks run synchronously inside the agent loop, so unlike listeners they can change what happens. They are registered globally when the decorated function is defined:

```python
import re

from crewai.hooks import (
    LLMCallHookContext,
    ToolCallHookContext,
    after_llm_call,
    before_tool_call,
)

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


@before_tool_call(tools=["delete_customer"])
def block_deletes(context: ToolCallHookContext) -> bool | None:
    """Refuse destructive calls unless the agent passed confirmed=True."""
    if not context.tool_input.get("confirmed"):
        return False          # the tool does not run; the agent sees "Tool execution blocked by hook"
    return None


@after_llm_call
def redact_emails(context: LLMCallHookContext) -> str | None:
    """Mask email addresses in every model response before CrewAI uses it."""
    if context.response and EMAIL.search(context.response):
        return EMAIL.sub("[email]", context.response)
    return None               # keep the response unchanged
```

- `before_llm_call` and `before_tool_call` hooks return `False` to block the call; they may edit `context.messages` / `context.tool_input` in place (never replace the list or dict).
- `after_llm_call` and `after_tool_call` hooks return a replacement string, or `None` to keep the original.
- `tools=[...]` and `agents=[...]` filter which calls a hook sees. `crewai.hooks.clear_all_global_hooks()` removes them all (useful between tests).

### Run-level hooks: `@on(InterceptionPoint...)`

The four decorators above are adapters over a more general mechanism in `crewai.hooks`. `@on(point)` registers a function for any interception point, and a hook can stop the operation by raising `HookAborted(reason)`:

| Point | Fires | Context highlights |
|---|---|---|
| `INPUT` / `EXECUTION_START` | Before a crew (or flow) runs | `payload` is the inputs dict |
| `OUTPUT` | After the final result is built | `payload` is the `CrewOutput` |
| `EXECUTION_END` | After success **or** failure | `status` is `"completed"` or `"failed"`; `error` on failure |
| `PRE_STEP` / `POST_STEP` | Around each task (or flow method) | `kind`, `step_name`, `payload` |
| `PRE_MODEL_CALL` / `POST_MODEL_CALL`, `PRE_TOOL_CALL` / `POST_TOOL_CALL` | Around model and tool calls | The same contexts as the four decorators |

```python
import re

from crewai.hooks import HookAborted, InterceptionPoint, on


@on(InterceptionPoint.INPUT)
def reject_banned_topics(ctx):
    if "forbidden" in str(ctx.payload.get("topic", "")):
        raise HookAborted("topic not allowed")        # kickoff() raises HookAborted


@on(InterceptionPoint.OUTPUT)
def redact_ssn(ctx):
    ctx.payload.raw = re.sub(r"\d{3}-\d{2}-\d{4}", "[ssn]", ctx.payload.raw)   # mutate in place


@on(InterceptionPoint.EXECUTION_END)
def audit(ctx):
    print("run finished:", ctx.status)
```

We ran these around a one-task crew: the redaction was visible in both `result.raw` and `result.tasks_output[0].raw`, a banned topic made `kickoff()` raise `HookAborted: topic not allowed`, and `EXECUTION_END` reported `"failed"` for that run. A hook may mutate its context in place, or return a replacement for `ctx.payload`. Any exception other than `HookAborted` raised inside a hook is swallowed, so a buggy hook cannot crash a run; it also cannot be relied on to block one. Use `clear_all_hooks()` between tests.

## Token usage

`kickoff()` returns a `CrewOutput` whose `token_usage` is a `UsageMetrics` with `total_tokens`, `prompt_tokens`, `cached_prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `cache_creation_tokens` and `successful_requests`:

```python
result = crew.kickoff(inputs={"product": "a USB-C dock"})
usage = result.token_usage
print(usage.prompt_tokens, usage.completion_tokens, usage.cached_prompt_tokens)
```

Multiply by your provider's current prices to report cost; keep the prices in configuration rather than code, because they change. For per-call figures (per model, per agent), read `event.usage` and `event.model` in an `LLMCallCompletedEvent` listener.

## Recovery with checkpoints

A checkpoint is a JSON snapshot of the run written when chosen events fire. If the process dies, resume from the last one instead of starting again:

```python
from crewai import Agent, Crew, Task, LLM, CheckpointConfig

llm = LLM(model="openai/%%model=openai-small%%")
writer = Agent(role="Writer", goal="Write about {topic}", backstory="Technical writer.", llm=llm)
outline = Task(description="Outline an article on {topic}.", expected_output="Five headings.", agent=writer)
draft = Task(description="Write the article from the outline.", expected_output="About 800 words.", agent=writer)

crew = Crew(
    agents=[writer],
    tasks=[outline, draft],
    checkpoint=CheckpointConfig(location="./.checkpoints", on_events=["task_completed"], max_checkpoints=20),
)
crew.kickoff(inputs={"topic": "vector search"})

# After a crash: rebuild the same crew and resume from a saved file
# crew.kickoff(from_checkpoint=CheckpointConfig(restore_from="./.checkpoints/main/<file>.json"))
```

- `crewai checkpoint list ./.checkpoints` shows each file with the number of tasks completed; `crewai checkpoint resume` picks the latest. See [CLI → Checkpoints](/crewai-guide/crewai_cli_python/#checkpoints).
- In containers, put `location` on a persistent volume, or use the SQLite provider (`crewai.state.provider.sqlite_provider`).
- Flows can use the same `checkpoint=` / `from_checkpoint=`, or `@persist` for state-level resume and paused human review ([Flows → Persistence](/crewai-guide/crewai_flows_guide/#persistence-and-resume--persist)).
- Checkpoints contain task outputs and inputs. Treat the directory as sensitive data.

## Testing without a model

Unit tests should not call a provider. Subclass `BaseLLM` with canned replies and pass it as the agent's `llm`; everything else (interpolation, task wiring, guardrails, events, flow routing) runs for real. These two files pass with `pytest` and no API key:

```python
# tests/test_crew.py: runs offline, no API key, no network
from typing import Any

from crewai import Agent, Crew, Task
from crewai.crews.crew_output import CrewOutput
from crewai.events import TaskCompletedEvent, crewai_event_bus
from crewai.llms.base_llm import BaseLLM


class ScriptedLLM(BaseLLM):
    """Returns canned replies in order and records every prompt it was sent."""

    replies: list[str] = []
    prompts: list[Any] = []

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None, **kwargs):
        self.prompts.append(messages)
        reply = self.replies.pop(0) if self.replies else "done"
        return f"Thought: I now know the final answer\nFinal Answer: {reply}"

    def supports_function_calling(self) -> bool:
        return False


def build_crew(llm: BaseLLM) -> Crew:
    analyst = Agent(
        role="Analyst",
        goal="Summarise {topic}",
        backstory="Writes short, factual summaries.",
        llm=llm,
    )
    summary = Task(
        description="Summarise {topic} in one sentence.",
        expected_output="One sentence.",
        agent=analyst,
    )
    return Crew(agents=[analyst], tasks=[summary])


def test_crew_returns_summary_and_interpolates_inputs():
    llm = ScriptedLLM(model="scripted", replies=["Tides are caused by the Moon."])
    result = build_crew(llm).kickoff(inputs={"topic": "tides"})

    assert isinstance(result, CrewOutput)
    assert result.raw == "Tides are caused by the Moon."
    assert "tides" in str(llm.prompts[0])          # {topic} reached the prompt


def test_task_completed_event_fires():
    seen = []
    with crewai_event_bus.scoped_handlers():       # handlers removed on exit
        @crewai_event_bus.on(TaskCompletedEvent)
        def record(source, event):
            seen.append(event.output.raw)

        build_crew(ScriptedLLM(model="scripted", replies=["ok"])).kickoff(inputs={"topic": "x"})
        crewai_event_bus.flush()                   # handlers run on a thread pool

    assert seen == ["ok"]
```

```python
# tests/test_flow.py: flow routing is plain Python, so test it without any LLM
from typing import Literal

from pydantic import BaseModel
from crewai.flow import Flow, listen, router, start


class TicketState(BaseModel):
    text: str = ""


class TriageFlow(Flow[TicketState]):
    @start()
    def receive(self):
        return self.state.text

    @router(receive)
    def route(self) -> Literal["urgent", "routine"]:
        return "urgent" if "outage" in self.state.text.lower() else "routine"

    @listen("urgent")
    def page(self):
        return "paged"

    @listen("routine")
    def queue(self):
        return "queued"


def test_outage_pages_on_call():
    assert TriageFlow().kickoff(inputs={"text": "EU outage"}) == "paged"


def test_everything_else_is_queued():
    assert TriageFlow().kickoff(inputs={"text": "typo"}) == "queued"


def test_router_declares_its_labels():
    route = TriageFlow.flow_definition().to_dict()["methods"]["route"]
    assert route["emit"] == ["urgent", "routine"]
```

- The stub answers in CrewAI's text format (`Final Answer: ...`) and reports no native function calling, which keeps the agent on the text path. To exercise a tool, return `Action: <tool name>` / `Action Input: {...}` first and the final answer on the next call.
- For structured output, return JSON matching the model when `response_model` is passed to `call`.
- Set `CREWAI_DISABLE_TELEMETRY=true` and `CREWAI_TRACING_ENABLED=false` in the test environment so nothing leaves the machine.

### Evaluation with a real model

- `crewai test -n 3 -m %%model=openai-small%%` runs the crew three times and scores each task with an OpenAI model. It costs tokens, and in 1.15.28 it exits 0 even when the run errors, so it is a report rather than a CI gate.
- `crewai eval` grades the last traced run through CrewAI AMP and exits non-zero on a failed gate; `crewai eval --models a,b` compares models on a deployment. It needs tracing and `crewai login`.

## Deploying

### CrewAI AMP

CrewAI AMP is CrewAI's hosted platform. The CLI is the interface; there is no `crewai.amp` Python API.

```bash
crewai login                       # authenticate the CLI
crewai deploy validate             # local pre-flight checks, no network
crewai deploy create               # first deployment (from the git origin remote, or a ZIP upload)
crewai deploy push                 # redeploy after changes
crewai deploy status -u <uuid>
crewai deploy logs -u <uuid>
```

`crewai deploy validate` runs the checks `create` and `push` run first, and exits non-zero on blocking errors: `pyproject.toml` with a project name, a lockfile (`uv.lock` or `poetry.lock`) that is present and not stale, the `src/<package>/` layout and standard crew files (or a `Flow` subclass in `main.py`), a resolvable wheel target, a clean import of the crew or flow module, environment variables referenced in code but missing from `.env`, and the installed `crewai` against the lockfile pin. It is cheap to run in CI.

Remember that `create` uploads the variables in your local `.env` (see [Configuration](#configuration-and-secrets)).

### Your own container

A crew is an ordinary Python program, so any container platform works. The CrewAI-specific points:

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY src ./src
RUN uv sync --frozen
ENV CREWAI_DISABLE_TELEMETRY=true
# No .env in the image: pass secrets at run time, e.g. docker run --env-file prod.env ...
CMD ["uv", "run", "run_crew"]
```

- `run_crew` is the console script the classic scaffold defines in `pyproject.toml`; flow projects define their own.
- Memory (LanceDB, under `memory/`), knowledge (ChromaDB), flow persistence (`flow_states.db`) and the latest-kickoff task outputs used by `crewai replay` (`latest_kickoff_task_outputs.db`) are written to CrewAI's per-user data directory, whose name comes from the project folder. Setting `CREWAI_STORAGE_DIR` moves them (we confirmed this for memory, flow persistence and the task-output store). Checkpoints are written to `./.checkpoints` unless `CheckpointConfig(location=...)` says otherwise, and `SQLiteFlowPersistence(path)` takes an explicit file. Point all of these at a mounted volume or they vanish with the container. Writes to these stores take a lock: a file lock in the system temp directory by default, or a Redis lock when `REDIS_URL` is set and `redis` is installed (`crewai_core.lock_store.set_lock_backend` swaps in your own). A temp-directory lock only coordinates processes on one machine, so containers that share a network volume should use `REDIS_URL` or a store built for concurrent writers.
- Behind a web API, run kickoffs as background jobs (a run takes seconds to minutes) and build a fresh crew per request instead of sharing one instance between concurrent requests. In async servers use `await crew.akickoff(...)` or `await flow.kickoff_async(...)`; `kickoff()` blocks the calling thread.
- Stream partial output with `Crew(stream=True)` / `Flow(stream=True)`, or listen for `LLMStreamChunkEvent`.

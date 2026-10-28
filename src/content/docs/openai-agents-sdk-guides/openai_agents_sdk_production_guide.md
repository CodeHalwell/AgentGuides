---
title: "OpenAI Agents SDK: Production Guide"
description: "Running Python Agents SDK apps in production: run limits, timeouts and retries, guardrails, approvals, sessions, tracing and data controls, and deterministic tests."
framework: "openai-agents-sdk"
language: "python"
sidebar:
  label: "Production"
  order: 5
---

> Verified against `openai-agents` %%version=openai-agents%% (checked %%versions-checked%%). Every Python example on this page was run with `agents.testing.ScriptedModel` standing in for the live model.

This page covers the controls the SDK itself gives you for running agents in production. Hosting is ordinary Python: the SDK is a library, so an agent runs wherever your web framework, worker or job runner does. For an HTTP streaming endpoint with a Dockerfile and Kubernetes manifest, see [Streaming server (FastAPI)](/openai-agents-sdk-guides/openai_agents_streaming_server_fastapi/).

## Pin the release

The SDK is versioned `0.Y.Z`, and a minor (`Y`) bump may change public interfaces. Pin a minor range and read the [release notes](/openai-agents-sdk-guides/#whats-new) before moving it:

```bash
pip install "openai-agents==%%version=openai-agents%%"
```

Also set `model=` explicitly on every agent. The SDK's default model has changed several times (0.16, 0.20), and an unpinned default changes behaviour and cost when you upgrade.

## Bound every run

An agent run is a loop. Bound each part of it.

| Limit | Where | Default | On breach |
|-------|-------|---------|-----------|
| Turns (model calls) per run | `Runner.run(..., max_turns=N)` | 10 | `MaxTurnsExceeded`, or your `error_handlers["max_turns"]` |
| One model-call attempt | `ModelSettings(timeout=seconds)` | none | the attempt is cancelled (and retried if a policy allows it) |
| Model-call retries | `ModelSettings(retry=ModelRetrySettings(...))` | off | the last error is raised |
| One function-tool call | `@function_tool(timeout=seconds)` | none | an error string is returned to the model (`timeout_behavior="error_as_result"`), or `"raise_exception"` |
| Output tokens per call | `ModelSettings(max_tokens=N)` | model limit | the response is truncated |

Runner-managed retries are opt-in. A policy decides which failures are retried; `retry_policies` provides the common ones and lets you combine them:

```python
import asyncio

from agents import (
    Agent,
    ModelRetrySettings,
    ModelSettings,
    Runner,
    function_tool,
    retry_policies,
)

resilient = ModelSettings(
    timeout=60,
    max_tokens=2_000,
    retry=ModelRetrySettings(
        max_retries=3,
        backoff={"initial_delay": 1.0, "max_delay": 20.0, "multiplier": 2.0, "jitter": True},
        policy=retry_policies.any(
            retry_policies.provider_suggested(),   # honour the API's retry advice
            retry_policies.network_error(),        # connection resets and timeouts
            retry_policies.http_status([429, 500, 502, 503]),
        ),
    ),
)


@function_tool(timeout=10)
async def lookup_account(account_id: str) -> str:
    """Fetch an account summary."""
    return f"Account {account_id}: active"


agent = Agent(
    name="Support",
    instructions="Answer account questions.",
    model="%%model=openai%%",
    model_settings=resilient,
    tools=[lookup_account],
)


async def main() -> None:
    result = await Runner.run(agent, "Is account 77 active?", max_turns=6)
    print(result.final_output)


asyncio.run(main())
```

The SDK only retries when it is safe to replay the request, so a stream that has already started sending tokens is not silently re-sent. Retries happen inside one `Runner.run` call, so do not also wrap `Runner.run` in your own retry loop: a retried run would re-execute tools that already had side effects.

To turn a hit turn limit into a normal answer instead of an exception, register an error handler:

```python
import asyncio

from agents import Agent, Runner
from agents.run_error_handlers import RunErrorHandlerInput, RunErrorHandlerResult


def give_up_politely(data: RunErrorHandlerInput) -> RunErrorHandlerResult:
    return RunErrorHandlerResult(
        final_output="Sorry, I couldn't finish that. A colleague will follow up.",
        include_in_history=False,
    )


agent = Agent(name="Assistant", instructions="Help the user.")


async def main() -> None:
    result = await Runner.run(
        agent, "Hello", max_turns=3, error_handlers={"max_turns": give_up_politely}
    )
    print(result.final_output)


asyncio.run(main())
```

`error_handlers` also accepts `"model_refusal"` (a `ModelRefusalError`, since 0.15) and `"invalid_final_output"`.

## Guardrails

Agent-level guardrails apply to the first agent's input and the last agent's output; tool guardrails wrap individual function tools. Two production points:

- Input guardrails run **in parallel** with the first model call unless you set `run_in_parallel=False`. Use blocking mode for checks that should prevent any spend.
- Keyword or regex guardrails are brittle in both directions. A small, fast model with a structured verdict is the pattern the SDK documentation uses:

```python
import asyncio

from pydantic import BaseModel

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    Runner,
    input_guardrail,
)


class Screening(BaseModel):
    off_topic: bool
    reason: str


screener = Agent(
    name="Screener",
    instructions="Decide whether the message is about our banking products. Be strict.",
    model="%%model=openai-small%%",
    output_type=Screening,
)


@input_guardrail(run_in_parallel=False)
async def on_topic(ctx, agent, input_data) -> GuardrailFunctionOutput:
    verdict = await Runner.run(screener, input_data, context=ctx.context)
    screening = verdict.final_output_as(Screening)
    return GuardrailFunctionOutput(output_info=screening, tripwire_triggered=screening.off_topic)


assistant = Agent(
    name="Banking assistant",
    instructions="Help with accounts and cards.",
    model="%%model=openai%%",
    input_guardrails=[on_topic],
)


async def main() -> None:
    try:
        result = await Runner.run(assistant, "How do I freeze my card?")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as exc:
        print("Blocked:", exc.guardrail_result.output.output_info)


asyncio.run(main())
```

For tool-level checks (argument limits, redaction of tool output) see [Tool guardrails](/openai-agents-sdk-guides/openai_agents_sdk_middleware_python/#tool-guardrails).

## Approvals for side effects

Tools that move money, send email or change records should carry `needs_approval`. The run pauses with `result.interruptions`; you store `result.to_state().to_string()` in your database, show the pending call to a person, then reload it with `RunState.from_string(agent, saved)`, approve or reject, and call `Runner.run(agent, state)`. The full flow is on the [hooks and approvals page](/openai-agents-sdk-guides/openai_agents_sdk_middleware_python/#human-in-the-loop-approvals).

Treat stored state as sensitive and trusted-only: it contains the conversation and the pending tool arguments, and loading it resumes execution.

## Sessions

Pass a `session=` to `Runner.run` and the SDK loads history before the run and appends new items after it. Key the session by conversation, for example `f"{tenant_id}:{conversation_id}"`; a key that changes on every request (a timestamp, a random UUID) silently gives the user a new conversation each time.

| Backend | Import | Install | Use when |
|---------|--------|---------|----------|
| `SQLiteSession` | `agents` | – | Single process, local file or `:memory:` for tests |
| `AsyncSQLiteSession`, `AdvancedSQLiteSession` | `agents.extensions.memory` | – | SQLite with async I/O; branching and usage tracking |
| `RedisSession` | `agents.extensions.memory` | `[redis]` | Several workers sharing short-lived history |
| `SQLAlchemySession` | `agents.extensions.memory` | `[sqlalchemy]` + async driver | Postgres/MySQL you already run |
| `MongoDBSession` | `agents.extensions.memory` | `[mongodb]` | MongoDB deployments |
| `DaprSession` | `agents.extensions.memory` | `[dapr]` | Dapr state stores |
| `OpenAIConversationsSession` | `agents` | – | Let OpenAI store the history (Conversations API) |
| `EncryptedSession` | `agents.extensions.memory` | `[encrypt]` | Wrap any session to encrypt items at rest, with a TTL |
| `OpenAIResponsesCompactionSession` | `agents` | – | Wrap any session and compact long histories with the Responses API |

```python
import asyncio
import os

from agents import Agent, Runner
from agents.extensions.memory import EncryptedSession, SQLAlchemySession


async def answer(tenant_id: str, conversation_id: str, text: str) -> str:
    session_id = f"{tenant_id}:{conversation_id}"
    store = SQLAlchemySession.from_url(
        session_id,
        url=os.environ.get("SESSION_DB_URL", "sqlite+aiosqlite:///sessions.db"),
        create_tables=True,  # run migrations yourself in production
    )
    session = EncryptedSession(
        session_id=session_id,
        underlying_session=store,
        encryption_key=os.environ["SESSION_KEY"],  # e.g. Fernet.generate_key().decode()
        ttl=7 * 24 * 3600,
    )
    agent = Agent(name="Assistant", instructions="Be helpful.", model="%%model=openai%%")
    result = await Runner.run(agent, text, session=session)
    return result.final_output


async def main() -> None:
    os.environ.setdefault("SESSION_KEY", "replace-with-a-real-key-from-your-secret-store")
    print(await answer("acme", "c-1", "Hello"))


asyncio.run(main())
```

Keep API keys, database handles and other secrets out of the run `context` if they are not needed by tools: the context object is passed to every hook, tool and guardrail.

## Tracing and data controls

Tracing is on by default and sends traces to the OpenAI dashboard using the `OPENAI_API_KEY`. The settings that matter in production:

| Need | Setting |
|------|---------|
| Keep prompts, tool arguments and outputs out of traces | `RunConfig(trace_include_sensitive_data=False)` or `OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA=false` |
| No tracing at all (e.g. Zero Data Retention organisations) | `set_tracing_disabled(True)`, `RunConfig(tracing_disabled=True)` or `OPENAI_AGENTS_DISABLE_TRACING=1` |
| Keep model and tool payloads out of SDK debug logs | `OPENAI_AGENTS_DONT_LOG_MODEL_DATA=1`, `OPENAI_AGENTS_DONT_LOG_TOOL_DATA=1` |
| Group runs from one conversation | `RunConfig(workflow_name=..., group_id=conversation_id, trace_metadata={...})` |
| Export elsewhere as well | `add_trace_processor(MyProcessor())` |
| Export elsewhere instead | `set_trace_processors([MyProcessor()])` |
| Short-lived workers (serverless, jobs) | call `flush_traces()` before the process exits |
| Tracing key differs from the model key | `set_tracing_export_api_key("sk-...")` |

A processor receives every trace and span. This one forwards finished spans to your logging pipeline as JSON:

```python
import asyncio
import json
import logging

from agents import Agent, RunConfig, Runner, TracingProcessor, add_trace_processor, flush_traces

log = logging.getLogger("agent-traces")


class JsonLogProcessor(TracingProcessor):
    def on_trace_start(self, trace) -> None:
        pass

    def on_trace_end(self, trace) -> None:
        log.info(json.dumps({"trace": trace.export()}, default=str))

    def on_span_start(self, span) -> None:
        pass

    def on_span_end(self, span) -> None:
        log.info(json.dumps({"span": span.export()}, default=str))

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


add_trace_processor(JsonLogProcessor())

agent = Agent(name="Assistant", instructions="Be brief.", model="%%model=openai%%")


async def main() -> None:
    result = await Runner.run(
        agent,
        "Hello",
        run_config=RunConfig(
            workflow_name="support-chat",
            group_id="conversation-123",
            trace_include_sensitive_data=False,
        ),
    )
    print(result.final_output)
    flush_traces()


asyncio.run(main())
```

Most observability vendors (Langfuse, Logfire, Arize/OpenInference, AgentOps, Braintrust and others) ship an instrumentor or a `TracingProcessor` for this SDK; install theirs rather than writing one. The upstream [tracing docs](https://openai.github.io/openai-agents-python/tracing/) keep the list current.

## Usage and cost

Token counts for the whole run are on `result.context_wrapper.usage` (`requests`, `input_tokens`, `output_tokens`, `total_tokens`, plus cached and reasoning token details). For per-call metrics, read `response.usage` in a `RunHooks.on_llm_end` hook and send it to your metrics system:

```python
import asyncio

from agents import Agent, RunHooks, Runner


class UsageMetrics(RunHooks):
    async def on_llm_end(self, context, agent, response) -> None:
        model = agent.model if isinstance(agent.model, str) else "default"
        # replace print with your metrics client, e.g. a Prometheus Counter
        print("tokens", agent.name, model, response.usage.input_tokens, response.usage.output_tokens)


agent = Agent(name="Assistant", instructions="Be brief.", model="%%model=openai%%")


async def main() -> None:
    result = await Runner.run(agent, "Hello", hooks=UsageMetrics())
    print("run total:", result.context_wrapper.usage.total_tokens)


asyncio.run(main())
```

Two SDK-level levers reduce cost: route simple work to a smaller model (`model="%%model=openai-small%%"` on that agent), and keep stable instructions and tool definitions at the start of the prompt so prompt caching applies (`ModelSettings(prompt_cache_retention=...)` extends how long cached prefixes live).

Long runs also pay for old tool output on every turn. `ToolOutputTrimmer` trims oversized results from earlier turns before each model call; see [Trimming old tool output](/openai-agents-sdk-guides/openai_agents_sdk_comprehensive_guide/#trimming-old-tool-output). For agents with many tools, `ToolSearchTool()` with deferred tools keeps unused definitions out of the prompt; see [Tool search and namespaces](/openai-agents-sdk-guides/openai_agents_sdk_comprehensive_guide/#tool-search-and-namespaces).

## Testing without the API

`agents.testing` (since 0.21) provides `ScriptedModel`, a deterministic `Model` that returns the steps you give it and records every call. Tests run offline, in milliseconds, and can assert on exactly which tools the agent called and what it sent to the model.

```python
import asyncio

import pytest

from agents import Agent, ModelRetrySettings, ModelSettings, Runner, function_tool, retry_policies
from agents.testing import ModelStep, ScriptedModel, assistant_message, function_call

refunds: list[tuple[str, float]] = []


@function_tool
def issue_refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    refunds.append((order_id, amount))
    return "refunded"


def build_agent(model: ScriptedModel) -> Agent:
    return Agent(name="Refunds", instructions="Issue refunds.", tools=[issue_refund], model=model)


def test_agent_calls_refund_tool() -> None:
    refunds.clear()
    model = ScriptedModel(
        [
            [function_call("issue_refund", {"order_id": "A-1", "amount": 20.0}, call_id="call_1")],
            [assistant_message("Your refund of 20.00 is on its way.")],
        ]
    )
    result = Runner.run_sync(build_agent(model), "Refund 20 on A-1")

    assert refunds == [("A-1", 20.0)]
    assert result.final_output == "Your refund of 20.00 is on its way."
    assert model.first_call is not None
    assert "Issue refunds." in (model.first_call.system_instructions or "")
    model.assert_complete()  # every scripted step was used


def test_retries_a_timeout() -> None:
    model = ScriptedModel(
        [ModelStep.raise_error(TimeoutError("upstream timed out")), [assistant_message("ok")]]
    )
    agent = Agent(
        name="Assistant",
        model=model,
        model_settings=ModelSettings(
            retry=ModelRetrySettings(
                max_retries=1,
                backoff={"initial_delay": 0.0},
                policy=retry_policies.network_error(),
            )
        ),
    )
    result = Runner.run_sync(agent, "hi")
    assert result.final_output == "ok"
    assert len(model.calls) == 2


@pytest.mark.asyncio
async def test_streaming_produces_final_output() -> None:
    model = ScriptedModel([[assistant_message("streamed")]])
    result = Runner.run_streamed(Agent(name="A", model=model), "hi")
    async for _ in result.stream_events():
        pass
    assert result.final_output == "streamed"


if __name__ == "__main__":
    test_agent_calls_refund_tool()
    test_retries_a_timeout()
    asyncio.run(test_streaming_produces_final_output())
    print("all passed")
```

Useful extras:

- `ModelStep.respond(fn)` computes the response from the recorded `ModelCall` (its `input`, `tools`, `output_schema`), which suits agents with structured output.
- `ScriptedModel(..., default_usage=Usage(...))` lets you test cost accounting.
- `agents.testing.scripted_sandbox_session` does the same for sandbox agents' shell and filesystem calls.
- To swap the model for every agent without touching agent definitions, pass `RunConfig(model=ScriptedModel(...))`.

Keep a small set of live "smoke" tests against the real API for prompts whose wording matters, and run them separately from the offline suite.

## Durable and long-running work

`Runner.run` lives in your process: if the process dies, the run is lost, though the session keeps the history up to the last completed turn. For runs that must survive restarts or wait days for an approval:

- **Approvals and pauses:** persist `RunState.to_string()` (above) and resume later in any process.
- **Durable execution:** the `[temporal]` extra installs Temporal's integration (`temporalio.contrib.openai_agents`, registered on the worker with `OpenAIAgentsPlugin`). Model calls and tools run as Temporal activities, so a crashed worker resumes the run where it stopped. Follow Temporal's documentation for the worker set-up.
- **Background mode:** for very long single model calls, OpenAI Responses supports background responses; poll rather than holding a request open.

## Checklist

- Pin `openai-agents` to a minor version and set `model=` on every agent.
- Set `max_turns`, `ModelSettings(timeout=..., retry=...)` and tool `timeout=` for every production agent.
- Put `needs_approval` on tools with side effects; store paused `RunState` server-side.
- Key sessions by conversation, not by request.
- Decide what may appear in traces (`trace_include_sensitive_data`) and where traces go.
- Test with `ScriptedModel` in CI; no API key needed.

---
title: "PydanticAI: Production Guide"
description: "Running PydanticAI agents in production with the framework's own features: Logfire and OpenTelemetry instrumentation, usage and cost limits, timeouts, retries and fallback models, concurrency limits, approvals and safe fetching, durable execution, and testing and evals in CI."
framework: "pydanticai"
language: "python"
sidebar:
  label: "Production"
  order: 21
---

Checked against **pydantic-ai %%version=pydantic-ai%%** (%%versions-checked%%). Every example on this page was run against that release, with `TestModel` standing in for the real model.

This page covers what PydanticAI itself gives you for production: observability, limits, resilience, concurrency, safety, durability and testing. Containers, load balancers, Kubernetes and queues are ordinary Python-service concerns and are not repeated here; an agent is a library object inside your service, so deploy it the way you deploy the rest of the service.

## Build agents once

An `Agent` is stateless between runs: conversation state lives in the message history you pass in, and per-request data lives in `deps`. Create agents at import time and reuse them across requests and tasks. Rebuilding an agent per request repeats tool-schema generation and throws away anything the agent caches.

```python
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext


@dataclass
class SupportDeps:
    customer_id: int
    tier: str


class Reply(BaseModel):
    answer: str
    escalate: bool


support_agent = Agent(
    'anthropic:%%model=anthropic%%',
    deps_type=SupportDeps,
    output_type=Reply,
    instructions='You are a support agent. Escalate anything involving refunds over 100 GBP.',
    name='support',
)


@support_agent.instructions
def customer_context(ctx: RunContext[SupportDeps]) -> str:
    return f'Customer {ctx.deps.customer_id} is on the {ctx.deps.tier} plan.'


async def handle(question: str, customer_id: int, tier: str) -> Reply:
    result = await support_agent.run(question, deps=SupportDeps(customer_id, tier))
    return result.output
```

`name=` matters in production: it labels the agent's spans and is required by the durable-execution integrations.

## Observability

PydanticAI emits OpenTelemetry traces and metrics that follow the GenAI semantic conventions: a span per run, per model request and per tool call, with token usage, cost (where pricing is known) and, optionally, message content. [Logfire](https://pydantic.dev/logfire) is Pydantic's own OTel backend and needs one call; any other OTel backend works too.

### Logfire

```python
import logfire
from pydantic_ai import Agent

logfire.configure(service_name='support-api', send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()   # instrument every agent in the process

agent = Agent('openai:%%model=openai%%', name='triage')
result = agent.run_sync('Classify: "my invoice is wrong"')
```

`logfire.configure()` reads `LOGFIRE_TOKEN` from the environment. `logfire.instrument_httpx()` adds the raw HTTP requests to the provider underneath each model span, which helps when debugging provider errors. Libraries such as SQLAlchemy or asyncpg have their own `logfire.instrument_*()` calls; each needs the matching extra (`pip install "logfire[sqlalchemy,asyncpg]"`).

Uninstrumented agents print a one-off banner suggesting observability; set `PYDANTIC_AI_NO_BANNER=1` to silence it.

### Per-agent settings and plain OpenTelemetry

Without Logfire, `Agent.instrument_all()` turns on instrumentation for every agent, or the `Instrumentation` capability turns it on for one. Both take `InstrumentationSettings`, which accepts your own `TracerProvider` and `MeterProvider`:

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from pydantic_ai import Agent, InstrumentationSettings
from pydantic_ai.capabilities import Instrumentation

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))   # use an OTLP exporter in practice

settings = InstrumentationSettings(
    tracer_provider=provider,
    include_content=False,         # keep prompts, outputs and tool arguments off the spans
    include_binary_content=False,
)

Agent.instrument_all(settings)                                       # every agent, or:
billing_agent = Agent('openai:%%model=openai%%', capabilities=[Instrumentation(settings)])
```

`include_content=False` is the setting to reach for when prompts can contain personal data: spans keep timings, token counts, model names and tool names, but not the text. `version=` (2–6, default 5) selects the attribute format; the [model providers page](/pydanticai-guide/pydantic_ai_model_providers/#instrumentedmodel-and-instrumentationsettings) lists what each version emits and the metrics recorded. `Agent(instrument=...)` was removed in 2.0; use the capability.

## Limits: tokens, cost, time

### Usage and cost limits

`UsageLimits` is checked during the run, not after it. The defaults are permissive: only `request_limit=50` is set.

```python
from decimal import Decimal

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

agent = Agent('anthropic:%%model=anthropic%%')

limits = UsageLimits(
    request_limit=10,                       # model requests per run
    tool_calls_limit=20,                    # successful tool executions per run
    total_tokens_limit=50_000,
    per_request_input_tokens_limit=20_000,  # one oversized request, independent of the running total
    cost_limit=Decimal('0.25'),             # USD, from genai-prices
)

try:
    result = agent.run_sync('Summarise the attached contract.', usage_limits=limits)
    print(result.usage.total_tokens, result.usage.cost)   # cost is a Decimal, or None if unpriced
except UsageLimitExceeded as exc:
    print(f'Stopped: {exc}')
```

`RunUsage.cost` is filled in from the [genai-prices](https://github.com/pydantic/genai-prices) data when the model is priced; it is `None` otherwise. If `cost_limit` is set and no cost can be calculated, PydanticAI emits `UsageLimitUnavailableWarning` rather than silently passing: turn that warning into an error in tests if a cost cap is a hard requirement. `count_tokens_before_request=True` checks token limits before sending, on providers that can count tokens.

To cap a whole conversation or a multi-agent workflow, pass the same `RunUsage` object into each run with `usage=`; limits then apply to the running total.

### Timeouts and cancellation

```python
from pydantic_ai import Agent
from pydantic_ai.exceptions import RunTimedOut

agent = Agent('openai:%%model=openai%%', tool_timeout=15)   # per tool call; Tool(timeout=...) overrides


@agent.tool_plain(timeout=60)
def generate_report(quarter: str) -> str:
    """Build the quarterly report (slow)."""
    return f'report for {quarter}'


try:
    result = agent.run_sync('Build the Q3 report.', timeout=120)   # the whole run
except RunTimedOut as exc:
    partial = exc.all_messages()   # what the run completed, for logging or resuming
```

A tool that exceeds its timeout counts as a failed attempt and the model gets a retry prompt. A run that exceeds `timeout=` raises `RunTimedOut`; a run started from inside another run inherits the outer deadline. To stop a run from elsewhere (a client disconnect, an admin action), pass a `CancellationToken` and call `.cancel()`; the run raises `RunCancelled`. The [error handling page](/pydanticai-guide/pydantic_ai_advanced_error_testing/#cancellation-and-timeouts--runcancelled--runtimedout) has the details.

## Resilience: retries and fallback

There are three separate retry layers. Keep them distinct, or a single outage multiplies into many paid requests.

| Layer | What it retries | Configure with |
|---|---|---|
| Agent | The model's own mistakes: invalid tool arguments, failed output validation, `ModelRetry` | `Agent(retries=...)`, `Tool(max_retries=...)`, `ToolOutput(max_retries=...)` |
| HTTP | Transient provider failures: 429, 5xx, network errors | the provider SDK's built-in retries, or a tenacity transport |
| Model | A provider that stays down | `FallbackModel` |

### Agent retries

```python
from pydantic_ai import Agent

agent = Agent('openai:%%model=openai%%', retries={'tools': 3, 'output': 2})   # an int sets both
```

The default is 1 for both. Raising `ModelRetry` from a tool or output validator spends one retry; `ToolFailed` reports a definitive failure to the model without spending one.

### HTTP retries

The provider SDKs (OpenAI, Anthropic, Groq, Cohere) already retry twice. `model_settings={'max_retries': N}` sets that count per agent or per run. For full control, including honouring `Retry-After`, install the `retries` extra (`pip install "pydantic-ai-slim[retries]"`) and give the provider an HTTP client with a retrying transport:

```python
import httpx2
from tenacity import retry_if_exception_type, stop_after_attempt

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.retries import AsyncHTTPX2TenacityTransport, RetryConfig, wait_retry_after

transport = AsyncHTTPX2TenacityTransport(
    RetryConfig(
        retry=retry_if_exception_type(httpx2.HTTPStatusError),
        wait=wait_retry_after(max_wait=60),   # Retry-After if present, else exponential back-off
        stop=stop_after_attempt(4),
        reraise=True,
    ),
    validate_response=lambda response: response.raise_for_status(),
)
model = OpenAIResponsesModel(
    '%%model=openai%%',
    provider=OpenAIProvider(http_client=httpx2.AsyncClient(transport=transport)),
)
agent = Agent(model, model_settings={'max_retries': 0})   # the transport owns retrying now
```

### Fallback models

`FallbackModel` tries each model in turn when a request raises `ModelAPIError` (the base class of `ModelHTTPError` and the connection, timeout and rate-limit errors). Mixing providers protects against a single provider's outage:

```python
from pydantic_ai import Agent
from pydantic_ai.exceptions import FallbackExceptionGroup
from pydantic_ai.models.fallback import FallbackModel

model = FallbackModel(
    'anthropic:%%model=anthropic%%',
    'openai:%%model=openai%%',
    'google:%%model=gemini%%',
)
agent = Agent(model, model_settings={'max_retries': 1})   # fail over quickly instead of retrying each provider

try:
    result = agent.run_sync('Draft a status update.')
except FallbackExceptionGroup as group:   # every model failed
    for exc in group.exceptions:
        print(type(exc).__name__, exc)
```

`fallback_on=` also accepts a function that inspects the `ModelResponse` and rejects weak or truncated replies. The [model providers page](/pydanticai-guide/pydantic_ai_model_providers/#fallbackmodel--wrap-primaries-with-backups) covers handlers in detail. Model settings are passed to whichever model handles the request, so keep them to keys every provider accepts.

## Concurrency limits

Provider rate limits are per API key, not per agent. Limit the **model** with a shared `ConcurrencyLimiter` so every agent using that key queues behind the same limit, and use `max_concurrency=` on an agent to cap how many runs of it are in flight:

```python
import asyncio

from pydantic_ai import Agent, ConcurrencyLimiter, limit_model_concurrency
from pydantic_ai.exceptions import ConcurrencyLimitExceeded

openai_pool = ConcurrencyLimiter(max_running=8, max_queued=200, name='openai')

summariser = Agent(limit_model_concurrency('openai:%%model=openai%%', openai_pool))
classifier = Agent(limit_model_concurrency('openai:%%model=openai-small%%', openai_pool), max_concurrency=20)


async def classify(text: str) -> str | None:
    try:
        return (await classifier.run(text)).output
    except ConcurrencyLimitExceeded:   # queue full: shed load instead of waiting
        return None


async def main() -> None:
    results = await asyncio.gather(*(classify(f'ticket {i}') for i in range(50)))
    print(sum(r is not None for r in results))


asyncio.run(main())
```

Waiting for a slot is recorded as an OTel span, so queueing shows up in traces. For limits shared across processes, implement `AbstractConcurrencyLimiter` (the [concurrency page](/pydanticai-guide/pydantic_ai_concurrency/#custom-limiter--redis-backed-distributed-limiting) has a Redis-backed example).

## Safety

Treat everything that reaches the model from outside (user input, fetched pages, tool results and MCP server output) as data that may contain instructions. The framework gives you places to enforce that.

**Approval for side effects.** Gate tools that change things behind human approval, so a prompt-injected request cannot run them unattended:

```python
from pydantic_ai import Agent, ApprovalRequiredToolset, DeferredToolRequests, FunctionToolset

ops = FunctionToolset()


@ops.tool_plain
def restart_service(name: str) -> str:
    """Restart a production service."""
    return f'restarted {name}'


@ops.tool_plain
def service_status(name: str) -> str:
    """Read a service's status."""
    return f'{name}: healthy'


gated = ApprovalRequiredToolset(ops, approval_required_func=lambda ctx, tool_def, args: tool_def.name == 'restart_service')
agent = Agent('openai:%%model=openai%%', toolsets=[gated], output_type=[str, DeferredToolRequests])

result = agent.run_sync('The API is slow; restart it if needed.')
if isinstance(result.output, DeferredToolRequests):
    pending = [(call.tool_name, call.args_as_dict()) for call in result.output.approvals]
    # Show `pending` to an operator, then resume with
    # agent.run_sync(message_history=result.all_messages(), deferred_tool_results=...)
```

`Tool(requires_approval=True)` does the same for a single tool. The approval round trip, including denying a call or overriding its arguments, is on the [toolsets page](/pydanticai-guide/pydantic_ai_toolsets/#approvalrequiredtoolset--human-in-the-loop).

**Fetching URLs.** `web_fetch_tool()` (the `web-fetch` extra) and the local fallback of the `WebFetch` capability refuse private, loopback and cloud-metadata addresses by default, re-check every redirect hop, and can be restricted to an allow-list. Use them rather than a bare `httpx.get` when a model chooses the URL:

```python
from pydantic_ai import Agent
from pydantic_ai.common_tools.web_fetch import web_fetch_tool

agent = Agent(
    'openai:%%model=openai%%',
    tools=[web_fetch_tool(allowed_domains=['docs.python.org'], max_content_length=20_000)],
)
```

**Front ends.** The AG-UI and Vercel AI adapters default to `manage_system_prompt='server'`, which drops system prompts sent by the browser and re-applies your own, and they only accept `http`/`https` file URLs. Keep those defaults unless you control the client.

**MCP servers.** An MCP server decides which tools it offers. Wrap an `MCPToolset` in `FilteredToolset` to expose only the tools you have reviewed, and in `ApprovalRequiredToolset` for any that write.

## Durable execution

For runs that must survive a crash or a deploy, or that wait hours for a human, add a durability capability: `TemporalDurability`, `DBOSDurability` or `PrefectDurability`. Model requests and tool calls become journalled steps, so a restarted worker resumes from the last completed step instead of re-paying for it. The agent code stays an ordinary `Agent`:

```python
from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import TemporalDurability

agent = Agent('openai:%%model=openai%%', name='research', capabilities=[TemporalDurability()])
```

Outside a workflow the capability does nothing, so the same agent still runs in tests. Workers, workflows and the engine-specific settings are on the [durable execution page](/pydanticai-guide/pydantic_ai_durable_execution/). When an engine owns retries, set `model_settings={'max_retries': 0}` so retries are not stacked.

## Serving

- **Streaming APIs.** `run_stream()` and `run_stream_events()` behind a FastAPI or Starlette endpoint: see [Streaming server (FastAPI)](/pydanticai-guide/pydantic_ai_streaming_server_fastapi/).
- **Chat front ends.** `AGUIAdapter.dispatch_request(request, agent=agent)` and `VercelAIAdapter.dispatch_request(...)` speak the AG-UI and Vercel AI SDK protocols directly ([integrations](/pydanticai-guide/pydantic_ai_integrations/#ui-protocols-and-front-ends)).
- **Internal tools.** `agent.to_web()` returns a Starlette chat app, useful for an internal demo or a support console.
- **One key for several providers.** The Pydantic AI Gateway proxies OpenAI, Anthropic, Google and others behind one API key with central spend limits; use a `gateway/` model prefix such as `'gateway/anthropic:%%model=anthropic%%'` ([model providers](/pydanticai-guide/pydantic_ai_model_providers/#gateway-routing)).

## Testing in CI

Unit tests should never reach a real model. Set `ALLOW_MODEL_REQUESTS = False` once, so an agent that was not overridden fails loudly instead of spending money, and swap the model per test with `agent.override()`:

```python
# conftest.py
import pytest
from pydantic_ai import models

models.ALLOW_MODEL_REQUESTS = False


@pytest.fixture
def anyio_backend():
    return 'asyncio'
```

```python
# test_support.py
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext, capture_run_messages
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel


@dataclass
class Deps:
    orders: dict[str, str]


agent = Agent('openai:%%model=openai%%', deps_type=Deps)


@agent.tool
def order_status(ctx: RunContext[Deps], order_id: str) -> str:
    """Look up an order."""
    return ctx.deps.orders.get(order_id, 'unknown')


def test_calls_the_tool():
    with agent.override(model=TestModel(call_tools=['order_status'])):
        result = agent.run_sync('Where is order A1?', deps=Deps({'A1': 'shipped'}))
    assert 'order_status' in result.output   # TestModel echoes the tool results it saw


def scripted(messages, info: AgentInfo) -> ModelResponse:
    """Call the tool once, then answer from its result."""
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('order_status', {'order_id': 'A1'})])
    tool_result = messages[-1].parts[0].content
    return ModelResponse(parts=[TextPart(f'Your order is {tool_result}.')])


def test_answer_uses_tool_result():
    with capture_run_messages() as messages:
        with agent.override(model=FunctionModel(scripted)):
            result = agent.run_sync('Where is order A1?', deps=Deps({'A1': 'shipped'}))
    assert result.output == 'Your order is shipped.'
    assert len(messages) == 4   # request, tool call, tool return, final answer


test_calls_the_tool()
test_answer_uses_tool_result()
```

`TestModel` exercises the plumbing (schemas, tools, validators); `FunctionModel` scripts exact model behaviour for edge cases. The [testing page](/pydanticai-guide/pydantic_ai_testing/) has more patterns, and the [error handling page](/pydanticai-guide/pydantic_ai_advanced_error_testing/#testing-error-paths) covers failure paths.

### Evals as a regression gate

Unit tests check the code; evals check the behaviour of the real model. Keep a small `pydantic_evals` dataset of representative cases, run it on a schedule or before releasing a prompt or model change, and fail the job when the scores fall:

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, MaxDuration

agent = Agent('openai:%%model=openai%%', instructions='Answer with the city name only.')


async def answer(question: str) -> str:
    return (await agent.run(question)).output


dataset = Dataset(
    name='capitals-smoke',
    cases=[
        Case(name='france', inputs='Capital of France?', expected_output='Paris'),
        Case(name='japan', inputs='Capital of Japan?', expected_output='Tokyo'),
    ],
    evaluators=[Contains(value='Paris', case_sensitive=False), MaxDuration(seconds=20)],
)

with agent.override(model=TestModel(custom_output_text='Paris')):   # drop this to evaluate the real model
    report = dataset.evaluate_sync(answer, max_concurrency=4)

report.print()
pass_rate = report.averages().assertions
assert pass_rate is not None and pass_rate >= 0.5, f'eval pass rate fell to {pass_rate:.0%}'
```

`report.averages()` returns the mean scores and the assertion pass rate; compare against a stored baseline report to see per-case changes. With Logfire configured every evaluation run is traced, and `OnlineEvaluation` samples live traffic with the same evaluators. See the [evals page](/pydanticai-guide/pydantic_ai_evals/).

## Checklist

- Agents created once at import, with `name=` set.
- Instrumentation on (`logfire.instrument_pydantic_ai()` or `Agent.instrument_all()`), with `include_content=False` where prompts hold personal data.
- `UsageLimits` on every run that a user or a loop can drive; `cost_limit` where prices are known.
- `tool_timeout=` on the agent and `timeout=` on runs started from a request.
- One owner per retry layer; `FallbackModel` across providers for availability.
- A shared `ConcurrencyLimiter` per API key.
- Approval on tools with side effects; `web_fetch_tool` for model-chosen URLs; MCP tools filtered.
- A durability capability for long or human-in-the-loop runs.
- `ALLOW_MODEL_REQUESTS = False` in tests; an eval dataset run before prompt or model changes.

## Revision history

| Date | Change |
|---|---|
| 2026-10-21 | Rewritten around PydanticAI's own production features (instrumentation, limits, retries and fallback, concurrency, approvals, durability, testing and evals). Generic FastAPI, Docker, Kubernetes, Prometheus, Redis, nginx and Celery material removed. |
| 2025-11 | First edition (1.x). |

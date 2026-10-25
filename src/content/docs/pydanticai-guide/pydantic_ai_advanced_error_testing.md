---
title: "PydanticAI: Error Handling & Testing Error Paths"
description: "Comprehensive guide to ModelRetry, UnexpectedModelBehavior, UsageLimitExceeded, ConcurrencyLimitExceeded, AgentRunError, UserError, and testing error paths with FunctionModel and capture_run_messages."
framework: "pydanticai"
language: "python"
sidebar:
  label: "Error handling"
  order: 11
---

Verified against **pydantic-ai %%version=pydantic-ai%%** (checked %%versions-checked%%) — source modules: `pydantic_ai.exceptions`, `pydantic_ai.usage`, `pydantic_ai.concurrency`, `pydantic_ai.models.function`, `pydantic_ai.agent`.

PydanticAI exposes a small, predictable exception hierarchy. Understanding each exception type — when it is raised, how to catch it, and how to test that your agent handles it correctly — is essential for production-grade agents.

---

## Exception hierarchy

```
RuntimeError
├── AgentRunError                 # base for all errors that occur during agent.run*()
│   ├── UsageLimitExceeded        # token / request / tool-call / cost budget exhausted
│   ├── ConcurrencyLimitExceeded  # queue depth exceeded when using ConcurrencyLimiter
│   ├── RunCancelled              # run cancelled (RunTimedOut is a subclass, and also a TimeoutError)
│   ├── HookTimeoutError          # a capability hook ran past its timeout (also a TimeoutError)
│   ├── UnexpectedModelBehavior   # model produced structurally invalid output
│   │   ├── ContentFilterError
│   │   └── IncompleteToolCall
│   └── ModelAPIError             # provider/API failure
│       ├── ModelHTTPError        # non-2xx HTTP from provider (has .status_code)
│       ├── ModelRateLimitError, ModelQuotaExceededError, ModelServerError,
│       │   ModelUnavailableError (→ ModelOverloadedError),
│       │   ModelContextWindowExceededError, ModelConnectionError (→ ModelTimeoutError)
│       └── ...                   # HTTP errors are also raised as the matching category
└── UserError                     # programming error (bad arguments, unsupported combination)

Exception
├── ModelRetry                    # not an error — a signal to retry (raised inside tools/validators)
├── ToolFailed                    # report a definitive tool failure to the model, no retry
├── ApprovalRequired, CallDeferred  # control flow for deferred / HITL tool calls
└── FallbackExceptionGroup        # (an ExceptionGroup) all FallbackModel candidates failed
```

The categorised `ModelAPIError` subclasses let you catch "rate limited" or "context window exceeded" without inspecting status codes per provider: a 429 from any provider is both a `ModelHTTPError` and a `ModelRateLimitError`.

---

## `ModelRetry` — ask the model to try again

`ModelRetry(message)` is raised inside a **tool** or **output validator** to tell PydanticAI "the model's output/args are wrong; send this feedback and retry". It is not a real exception — it is caught by the framework before it reaches your caller.

```python
from pydantic_ai import Agent, ModelRetry, RunContext

agent = Agent('openai:%%model=openai%%')

@agent.tool
async def lookup_user(ctx: RunContext[object], user_id: int) -> dict:
    if user_id <= 0:
        raise ModelRetry(f'user_id must be positive, got {user_id!r}')
    return {'id': user_id, 'name': 'Alice'}

result = agent.run_sync('Look up user -1, then user 42.')
print(result.output)
```

After `max_retries` retries, if the model keeps producing bad args, PydanticAI raises `UnexpectedModelBehavior`.

### `ModelRetry` in output validators

```python
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext

class Summary(BaseModel):
    title: str
    bullets: list[str]
    word_count: int

agent = Agent('openai:%%model=openai%%', output_type=Summary)

@agent.output_validator
async def validate_bullets(ctx: RunContext[object], out: Summary) -> Summary:
    if len(out.bullets) < 3:
        raise ModelRetry(
            f'Need at least 3 bullet points, got {len(out.bullets)}. '
            'Please add more detail.'
        )
    if out.word_count <= 0:
        raise ModelRetry('word_count must be positive.')
    return out

result = agent.run_sync('Summarise the PydanticAI framework.')
print(result.output)
```

---

## `ToolFailed` — a definitive failure, not a retry

`ModelRetry` tells the model "fix your call and try again" and spends the tool's retry budget. `ToolFailed(message)` reports that the call has *definitively* failed (not found, permanently gone, not permitted): the model sees your message verbatim as the tool result, no retry instructions are added, and no retry budget is used. Bound repeated failures with `UsageLimits(tool_calls_limit=...)`.

| | `ModelRetry` | `ToolFailed` |
|---|---|---|
| Model sees | retry prompt + your message | your message only |
| Uses the retry budget | yes | no |
| Use for | fixable input (bad ID format, missing field) | definitive outcomes |

## Cancellation and timeouts — `RunCancelled` / `RunTimedOut`

A run can be stopped from outside without losing its history:

- `agent.run(..., timeout=seconds)` raises `RunTimedOut` (a `RunCancelled` subclass).
- `agent.run(..., cancellation_token=token)`, then `token.cancel()` from anywhere (a request handler, a UI "stop" button).
- `run.cancel()` on an `AgentRun` from `agent.iter()`, or `ctx.cancel()` inside a tool.

`RunCancelled` carries `all_messages()`, `new_messages()`, `usage`, `run_id` and `conversation_id`; tool calls that never finished are closed with synthetic `outcome='interrupted'` returns, so `all_messages()` can be passed straight back as `message_history` to resume. Cancellation is terminal: capability hooks can observe it and clean up, but cannot turn it into a successful result.

```python
import asyncio

from pydantic_ai import Agent, CancellationToken
from pydantic_ai.exceptions import RunCancelled, RunTimedOut

agent = Agent('openai:%%model=openai%%')


@agent.tool_plain
async def slow_report(name: str) -> str:
    """Build a report (slow)."""
    await asyncio.sleep(5)
    return f'report {name}'


async def main():
    # 1. A deadline for the whole run
    try:
        await agent.run('Build the Q3 report.', timeout=0.5)
    except RunTimedOut as exc:
        print('timed out after', exc.usage.requests, 'request(s);', len(exc.all_messages()), 'messages kept')
        history = exc.all_messages()   # safe to pass back as message_history

    resumed = await agent.run('Tell me what you managed so far.', message_history=history)
    print(resumed.output)

    # 2. Cancel from elsewhere, e.g. a "stop" button handler holding the token
    token = CancellationToken()
    task = asyncio.create_task(agent.run('Build the Q4 report.', cancellation_token=token))
    await asyncio.sleep(0.2)
    token.cancel()
    try:
        await task
    except RunCancelled as exc:
        print('cancelled run', exc.run_id)


asyncio.run(main())
```

External cancellation is different on purpose: if the task running the agent is cancelled by `asyncio.timeout()`, a task group or a durable-execution engine, the `CancelledError` keeps propagating so that asyncio can tear down correctly. Catch it, call `RunCancelled.from_cancellation(exc)` to recover the same run state, and re-raise.

## `UnexpectedModelBehavior` — model misbehaved

Raised when:

- The model returns output that cannot be parsed or validated after exhausting all retries.
- The model emits an unrecognised response structure.
- A tool-call loop exceeds the retry budget.

```python
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart

# Simulate a model that always returns invalid JSON for a structured output
def broken_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content='THIS IS NOT VALID JSON')])

from pydantic import BaseModel

class Answer(BaseModel):
    value: int

agent = Agent(FunctionModel(broken_model), output_type=Answer)

with capture_run_messages() as msgs:
    try:
        agent.run_sync('What is 2 + 2?')
    except UnexpectedModelBehavior as e:
        print('Model misbehaved:', e)
        print('Messages up to failure:', len(msgs))
```

### Catching `UnexpectedModelBehavior` at the call site

```python
import logging
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior

logger = logging.getLogger(__name__)
agent = Agent('openai:%%model=openai%%')

async def safe_run(prompt: str) -> str | None:
    try:
        result = await agent.run(prompt)
        return result.output
    except UnexpectedModelBehavior as e:
        logger.error('agent_misbehaved', exc_info=e, extra={'prompt': prompt[:200]})
        return None   # or raise a domain-specific error
```

---

## `UsageLimitExceeded` — budget controls

`UsageLimits` is a dataclass you pass to `agent.run*(usage_limits=...)`. When any limit fires, `UsageLimitExceeded` (a subclass of `AgentRunError`) is raised.

```python
from pydantic_ai import Agent
from pydantic_ai import UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

agent = Agent('openai:%%model=openai%%')

try:
    result = agent.run_sync(
        'Write a 10 000-word essay on the history of computing.',
        usage_limits=UsageLimits(
            output_tokens_limit=500,    # hard cap on output tokens
            request_limit=3,            # max LLM round-trips
        ),
    )
except UsageLimitExceeded as e:
    print('Limit hit:', e)
```

### All `UsageLimits` fields

```python
from decimal import Decimal
from pydantic_ai import UsageLimits

limits = UsageLimits(
    request_limit=50,           # max number of model requests (default 50)
    cost_limit=Decimal('0.50'), # max estimated cost per run (uses genai-prices)
    tool_calls_limit=20,        # max successful tool executions
    input_tokens_limit=8_000,   # max prompt tokens per run
    output_tokens_limit=2_000,  # max generated tokens per run
    total_tokens_limit=10_000,  # combined input + output cap
    per_request_input_tokens_limit=6_000,  # cap on any single request's prompt
    count_tokens_before_request=True,  # pre-check tokens (Anthropic, Google, Bedrock Converse, OpenAI Responses)
)
```

### Tracking usage after a successful run

```python
from pydantic_ai import Agent, RunUsage

agent = Agent('openai:%%model=openai%%')
result = agent.run_sync('Hello')

usage: RunUsage = result.usage
print(f'requests={usage.requests}  input={usage.input_tokens}  output={usage.output_tokens}  total={usage.total_tokens}')
```

### Accumulating usage across multiple runs

```python
from pydantic_ai import Agent, RunUsage

agent = Agent('openai:%%model=openai%%')
shared_usage = RunUsage()

for prompt in ['One', 'Two', 'Three']:
    result = agent.run_sync(prompt, usage=shared_usage)

print('Grand total tokens:', shared_usage.total_tokens)
```

---

## `UserError` — programming mistakes

`UserError` means you (the developer) passed invalid arguments or an unsupported combination: an unknown model name, a missing API key, a native tool the model does not support, an output type override that conflicts with a validator. Most are raised when the agent is built or before the first request; a few (such as a deferred tool call without `DeferredToolRequests` in the output type) can only be detected during the run.

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError

try:
    agent = Agent('openai:%%model=openai%%', output_type=int)

    @agent.output_validator
    def validate_int(ctx: RunContext[object], out: int) -> int:
        return out

    # Overriding output_type when an output_validator is already registered
    # raises UserError because the validator's type would no longer match.
    agent.run_sync('hi', output_type=str)
except UserError as e:
    print('Developer error:', e)
```

In tests, `UserError` is an expected result of misconfigured agents — assert on it rather than catching it silently.

---

## `ConcurrencyLimitExceeded` — queue depth exceeded

When you use `ConcurrencyLimiter(max_running=N, max_queued=M)` and more than `N + M` tasks arrive simultaneously, `ConcurrencyLimitExceeded` is raised.

```python
import asyncio
from pydantic_ai import Agent, ConcurrencyLimiter
from pydantic_ai.exceptions import ConcurrencyLimitExceeded

agent = Agent(
    'openai:%%model=openai%%',
    max_concurrency=ConcurrencyLimiter(max_running=2, max_queued=3, name='demo'),
)

async def main():
    tasks = [agent.run(f'Task {i}') for i in range(10)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, r in enumerate(results):
        if isinstance(r, ConcurrencyLimitExceeded):
            print(f'Task {i}: queue full, rejected')
        else:
            print(f'Task {i}: ok')

asyncio.run(main())
```

---

## `ModelHTTPError` — provider HTTP failures

`ModelHTTPError` (source: `pydantic_ai/exceptions.py`) exposes:

| Attribute            | Type              | Notes                                                                       |
| -------------------- | ----------------- | --------------------------------------------------------------------------- |
| `status_code`        | `int`             | 4xx or 5xx.                                                                 |
| `body`               | `object | None`   | Provider payload if available.                                              |
| `headers`            | `dict[str, str] | None` | Lowercased header names — use `.get('retry-after')`.                    |
| `suggested_model_id` | `str | None`      | Set when the provider returned "did you mean <id>?".                        |
| `retry_after`        | `float | None`    | Seconds to wait, parsed from `Retry-After` (integer seconds *or* RFC-9110 date) when the provider sent one. |
| `provider_error_code` / `provider_error_type` | `str | None` | Provider-specific error identifiers, when available. |

```python
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, AgentRunError

agent = Agent('openai:%%model=openai%%')

try:
    result = agent.run_sync('Hello')
except ModelHTTPError as e:
    print(f'Provider returned HTTP {e.status_code}: {e}')
    if e.retry_after is not None:
        # float | None — already parsed from the Retry-After header.
        print(f'  retry after {e.retry_after:.1f}s')
    if e.suggested_model_id:
        print(f'  provider suggested {e.suggested_model_id!r}')
    if e.status_code == 429:
        # Rate limited — implement backoff
        pass
    elif e.status_code >= 500:
        # Server error — retry or fail gracefully
        pass
except AgentRunError as e:
    print(f'Run failed: {e}')
```

---

## `FallbackExceptionGroup` — multi-model fallback failures

```python
from pydantic_ai import Agent
from pydantic_ai.exceptions import FallbackExceptionGroup
from pydantic_ai.models.fallback import FallbackModel

agent = Agent(
    FallbackModel('openai:%%model=openai%%', 'anthropic:%%model=anthropic-flagship%%', 'google:%%model=gemini%%')
)

try:
    result = agent.run_sync('Hello')
except FallbackExceptionGroup as eg:
    print(f'All {len(eg.exceptions)} models failed:')
    for exc in eg.exceptions:
        print(f'  {type(exc).__name__}: {exc}')
```

---

## Error hooks — intercept errors without try/except

A `model_request_error` hook runs when the model call itself fails. It has three ways out: **re-raise** the error (or another) to fail the run, **return** a `ModelResponse` to carry on as if the call had succeeded, or **raise `ModelRetry`** to repeat the request. The example backs off on HTTP 429 and retries; the `FunctionModel` stands in for a provider that rate-limits the first call:

```python
import asyncio

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

calls = 0


def flaky_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    global calls
    calls += 1
    if calls == 1:
        raise ModelHTTPError(status_code=429, model_name='demo', headers={'Retry-After': '0.1'})
    return ModelResponse(parts=[TextPart('ok')])


hooks = Hooks()


@hooks.on.model_request_error
async def back_off_on_rate_limit(ctx, *, request_context, error: Exception) -> ModelResponse:
    if isinstance(error, ModelHTTPError) and error.status_code == 429:
        # retry_after is parsed from Retry-After (seconds or an HTTP date); asyncio.sleep keeps the loop free
        await asyncio.sleep(min(error.retry_after or 5.0, 30.0))
        raise ModelRetry('Rate limited; retrying the same request.')
    raise error


agent = Agent(FunctionModel(flaky_model), capabilities=[hooks])
result = agent.run_sync('Hello')
print(result.output, calls)
#> ok 2
```

A `ModelRetry` from this hook adds a `RetryFeedbackPart` to the history before the repeat request (a `ModelRetry` raised in a tool is recorded differently: as that call's `ToolReturnPart` with `outcome='retried'`), so the model sees that a retry happened. For plain transport problems (429, 5xx, timeouts) the HTTP-level retries [below](#transport-level-http-retries) are cleaner: they repeat the request without touching the conversation.

---

## Testing error paths

### 1. Test that `ModelRetry` triggers properly

Use `FunctionModel` to simulate a model that first produces bad output, then good output.

```python
import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.test import TestModel

class Price(BaseModel):
    amount: float
    currency: str

agent = Agent('openai:%%model=openai%%', output_type=Price)

@agent.output_validator
async def no_negative(ctx: RunContext[object], p: Price) -> Price:
    if p.amount < 0:
        raise ModelRetry('Price cannot be negative, please fix.')
    return p

def test_validator_retries():
    call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Simulate bad output on first attempt
            return ModelResponse(parts=[TextPart('{"amount": -5, "currency": "USD"}')])
        # Good output on retry
        return ModelResponse(parts=[TextPart('{"amount": 9.99, "currency": "USD"}')])

    with agent.override(model=FunctionModel(model_fn)):
        result = agent.run_sync('Price of widget?')

    assert result.output.amount == 9.99
    assert call_count == 2  # one retry happened
```

### 2. Assert `UsageLimitExceeded` is raised

```python
import pytest
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models.test import TestModel

agent = Agent('openai:%%model=openai%%')

def test_request_limit():
    with agent.override(model=TestModel()):
        with pytest.raises(UsageLimitExceeded, match='request'):
            agent.run_sync('hi', usage_limits=UsageLimits(request_limit=0))
```

### 3. Verify `UnexpectedModelBehavior` is raised when retries are exhausted

```python
import pytest
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

agent = Agent('openai:%%model=openai%%')

@agent.tool(retries=2)
def always_fails(ctx: RunContext[object]) -> str:
    raise ModelRetry('always broken')

def test_exhausted_retries():
    with agent.override(model=TestModel()):
        with pytest.raises(UnexpectedModelBehavior):
            agent.run_sync('go')
```

### 4. Inspect messages after a failure with `capture_run_messages`

```python
import pytest
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel
from pydantic import BaseModel

class Strict(BaseModel):
    required_field: str

agent = Agent('openai:%%model=openai%%', output_type=Strict, retries=1)

def test_messages_on_failure():
    with capture_run_messages() as messages:
        with pytest.raises(UnexpectedModelBehavior):
            # output-tool arguments that fail validation (the required field is missing)
            with agent.override(model=TestModel(custom_output_args={'wrong_field': 'x'})):
                agent.run_sync('go')

    # Messages contain the full conversation up to the point of failure
    assert len(messages) > 0
    print('Conversation had', len(messages), 'messages before failure')
```

### 5. Inject specific exceptions via `FunctionModel`

```python
import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponse

def model_that_raises(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    # Simulate provider returning an unexpected response
    return ModelResponse(parts=[])   # empty parts triggers UnexpectedModelBehavior

agent = Agent(FunctionModel(model_that_raises))

def test_empty_response():
    with pytest.raises(UnexpectedModelBehavior):
        agent.run_sync('anything')
```

---

## `TestModel` — zero-LLM test double

`TestModel` never calls a real LLM. Configure it with canned outputs and inspect every call it received:

```python
import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

class Answer(BaseModel):
    value: int
    reasoning: str

agent = Agent('openai:%%model=openai%%', output_type=Answer)

def test_structured_output():
    with agent.override(model=TestModel()):
        result = agent.run_sync('What is 2 + 2?')
        assert isinstance(result.output, Answer)
        assert isinstance(result.output.value, int)
```

### Configuring `TestModel` responses

```python
from pydantic_ai.models.test import TestModel

# Return a specific text response
model = TestModel(custom_output_text='The answer is 42.')

# Return structured output as a dict
model = TestModel(custom_output_args={'value': 42, 'reasoning': 'Basic arithmetic'})

# Simulate a tool call before responding
model = TestModel(
    call_tools=['web_search'],    # tool names to call
    custom_output_text='Found it.',
)
```

### Inspecting calls made to `TestModel`

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai.models.test import TestModel

model = TestModel()
agent = Agent('openai:%%model=openai%%', output_type=str)

@agent.tool_plain
def lookup(sku: str) -> str:
    return f'{sku}: in stock'

with agent.override(model=model):
    result = agent.run_sync('Is ABC-1 in stock?')

# What the agent offered the model on the last request
params = model.last_model_request_parameters
print([t.name for t in params.function_tools])   # ['lookup']

# What was actually exchanged
requests = [m for m in result.all_messages() if isinstance(m, ModelRequest)]
responses = [m for m in result.all_messages() if isinstance(m, ModelResponse)]
print(len(requests), len(responses))             # 2 2
```

---

## `FunctionModel` — Python function as model

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel, AgentInfo

def my_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    last_user = next(
        (p.content for m in reversed(messages)
         for p in m.parts if hasattr(p, 'content') and isinstance(p.content, str)),
        'no message'
    )
    return ModelResponse(parts=[TextPart(content=f'Echo: {last_user}')])

agent = Agent('openai:%%model=openai%%')

def test_echo_response():
    with agent.override(model=FunctionModel(my_model)):
        result = agent.run_sync('Hello there')
        assert result.output == 'Echo: Hello there'
```

### Simulating tool calls in `FunctionModel`

```python
import json

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

def tool_calling_model(messages, info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        # First turn — call a tool
        return ModelResponse(parts=[
            ToolCallPart(
                tool_name='get_weather',
                args=json.dumps({'city': 'Paris'}),
                tool_call_id='call_1',
            )
        ])
    # Second turn — respond with text
    return ModelResponse(parts=[TextPart(content='The weather in Paris is sunny.')])


agent = Agent(FunctionModel(tool_calling_model))


@agent.tool_plain
def get_weather(city: str) -> str:
    return f'Sunny in {city}'


result = agent.run_sync('Weather in Paris?')
assert result.output == 'The weather in Paris is sunny.'
```

---

## `capture_run_messages` — inspect all messages after a run

```python
from pydantic_ai import Agent, capture_run_messages

agent = Agent('openai:%%model=openai%%')

with capture_run_messages() as messages:
    result = agent.run_sync('What is 2 + 2?')

for msg in messages:
    print(type(msg).__name__, '—', [type(p).__name__ for p in msg.parts])
```

---

## Retry patterns

### Tool with exponential backoff

```python
import asyncio
import httpx
from pydantic_ai import Agent, ModelRetry, RunContext

agent = Agent('openai:%%model=openai%%')

@agent.tool(retries=4)
async def fetch_api(ctx: RunContext[object], url: str) -> str:
    delay = 2 ** ctx.retry  # 1s, 2s, 4s, 8s
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, timeout=10.0)
            r.raise_for_status()
            return r.text[:2000]
    except (httpx.HTTPStatusError, httpx.TimeoutException) as e:
        await asyncio.sleep(delay)
        raise ModelRetry(f'Request failed ({e}); retrying (attempt {ctx.retry + 1})')
```

### Conditional retry based on error type

```python
from pydantic_ai import Agent, ModelRetry, RunContext
import json

agent = Agent('openai:%%model=openai%%')

@agent.tool(retries=3)
def parse_structured(ctx: RunContext[object], raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ModelRetry(
            f'Invalid JSON at position {e.pos}: {e.msg}. '
            'Return only valid JSON, no markdown fences.'
        )
    if 'id' not in data:
        raise ModelRetry('Response JSON must include an "id" field.')
    return data
```

---

### Transport-level HTTP retries

`ModelRetry` is about the *model's* mistakes. For flaky networks and 429/5xx responses, retry at the HTTP layer with the `retries` extra (`pip install "pydantic-ai-slim[retries]"`, which adds `tenacity`; it is not in the default bundle): wrap the provider's HTTP client in a tenacity transport.

```python
import httpx2
from tenacity import retry_if_exception_type, stop_after_attempt

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.retries import AsyncHTTPX2TenacityTransport, RetryConfig, wait_retry_after

transport = AsyncHTTPX2TenacityTransport(
    config=RetryConfig(
        retry=retry_if_exception_type(httpx2.HTTPStatusError),
        wait=wait_retry_after(max_wait=60),      # honours Retry-After, else exponential back-off
        stop=stop_after_attempt(4),
        reraise=True,
    ),
    validate_response=lambda r: r.raise_for_status(),   # turn 429/5xx into exceptions to retry
)
provider = OpenAIProvider(http_client=httpx2.AsyncClient(transport=transport))
agent = Agent(OpenAIChatModel('%%model=openai%%', provider=provider))
```

`RetryConfig` is a `TypedDict` of `tenacity.retry` arguments. `HTTPX2TenacityTransport` is the sync twin, and the older `AsyncTenacityTransport` / `TenacityTransport` (for plain `httpx`) are deprecated. Provider SDKs retry too: the OpenAI, Anthropic, Groq and Cohere clients PydanticAI builds retry twice by default, Google's not at all; set `model_settings={'max_retries': 0}` if you want the transport to own retries entirely. Not every model honours `max_retries`; those that do not emit an `UnsupportedSettingWarning`.

## Structured error reporting

Use a Pydantic model for the error surface so callers get typed information:

```python
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import AgentRunError, UsageLimitExceeded, UnexpectedModelBehavior

class RunSuccess(BaseModel):
    output: str
    tokens_used: int

@dataclass
class RunFailure:
    reason: str
    kind: str

async def run_with_report(agent: Agent, prompt: str) -> RunSuccess | RunFailure:
    try:
        result = await agent.run(prompt)
        return RunSuccess(
            output=result.output,
            tokens_used=result.usage.total_tokens,
        )
    except UsageLimitExceeded as e:
        return RunFailure(reason=str(e), kind='budget_exceeded')
    except UnexpectedModelBehavior as e:
        return RunFailure(reason=str(e), kind='model_error')
    except AgentRunError as e:
        return RunFailure(reason=str(e), kind='run_error')
```

---

## pytest fixtures — agents pre-wired for error testing

```python
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart

@pytest.fixture
def deterministic_agent():
    agent = Agent('openai:%%model=openai%%', system_prompt='Be helpful.')
    with agent.override(model=TestModel(seed=42)):
        yield agent

@pytest.fixture
def failing_agent():
    agent = Agent('openai:%%model=openai%%')
    def always_empty(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('')])
    with agent.override(model=FunctionModel(always_empty)):
        yield agent

@pytest.fixture
def test_model():
    return TestModel(custom_output_text='mocked response')

def test_agent_returns_string(test_model):
    agent = Agent('openai:%%model=openai%%', output_type=str)
    with agent.override(model=test_model):
        result = agent.run_sync('Any prompt')
        assert isinstance(result.output, str)
```

### Parametrised snapshot testing

```python
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

@pytest.mark.parametrize('prompt,expected', [
    ('Say hello to the world', 'Hello, world!'),
    ('Say goodbye to the world', 'Goodbye, world!'),
])
def test_snapshot(prompt, expected):
    agent = Agent('openai:%%model=openai%%')
    with agent.override(model=TestModel(custom_output_text=expected)):
        result = agent.run_sync(prompt)
        assert result.output == expected
```

---

## Reference

| Symbol | Module | Notes |
|---|---|---|
| `ModelRetry` | `pydantic_ai.exceptions` | Triggers a model retry from tool/validator |
| `UnexpectedModelBehavior` | `pydantic_ai.exceptions` | Terminal failure after retries exhausted |
| `AgentRunError` | `pydantic_ai.exceptions` | Base for all run-time failures |
| `ModelHTTPError` | `pydantic_ai.exceptions` | Non-2xx HTTP — has `.status_code` |
| `UsageLimitExceeded` | `pydantic_ai.exceptions` | Usage budget exceeded |
| `UserError` | `pydantic_ai.exceptions` | Developer configuration error |
| `ConcurrencyLimitExceeded` | `pydantic_ai.exceptions` | Queue depth exceeded |
| `FallbackExceptionGroup` | `pydantic_ai.exceptions` | All fallback candidates failed |
| `ApprovalRequired` | `pydantic_ai.exceptions` | HITL approval needed |
| `UsageLimits` | `pydantic_ai` | Budget configuration dataclass |
| `RunUsage` | `pydantic_ai` | Token/request counters |
| `TestModel` | `pydantic_ai.models.test` | Zero-LLM test double |
| `FunctionModel` | `pydantic_ai.models.function` | Python function as model |
| `capture_run_messages` | `pydantic_ai` | Collect messages from a run |
| `agent.override()` | `pydantic_ai.agent.Agent` | Context-manager model swap |

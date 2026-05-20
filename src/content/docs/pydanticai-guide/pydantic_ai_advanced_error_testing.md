---
title: "PydanticAI: Advanced Error Handling & Testing"
description: "Comprehensive guide to ModelRetry, UnexpectedModelBehavior, UsageLimitExceeded, ConcurrencyLimitExceeded, AgentRunError, UserError, and testing error paths with FunctionModel and capture_run_messages."
framework: pydanticai
language: python
---

# Advanced Error Handling & Testing

Verified against **pydantic-ai==1.99.0** — source modules: `pydantic_ai.exceptions`, `pydantic_ai.usage`, `pydantic_ai.concurrency`, `pydantic_ai.models.function`, `pydantic_ai.agent`.

PydanticAI exposes a small, predictable exception hierarchy. Understanding each exception type — when it is raised, how to catch it, and how to test that your agent handles it correctly — is essential for production-grade agents.

---

## Exception hierarchy

```
Exception
└── AgentRunError            # base for all errors that occur during agent.run*()
    ├── UsageLimitExceeded   # token / request / tool-call budget exhausted
    └── UnexpectedModelBehavior   # model produced structurally invalid output
ModelRetry                   # not an error — a signal to retry (raised inside tools/validators)
UserError                    # programming error (bad arguments, unsupported combination)
ConcurrencyLimitExceeded     # queue depth exceeded when using ConcurrencyLimiter
```

---

## `ModelRetry` — ask the model to try again

`ModelRetry(message)` is raised inside a **tool** or **output validator** to tell PydanticAI "the model's output/args are wrong; send this feedback and retry". It is not a real exception — it is caught by the framework before it reaches your caller.

```python
from pydantic_ai import Agent, ModelRetry, RunContext

agent = Agent('openai:gpt-4o')

@agent.tool
async def lookup_user(ctx: RunContext[None], user_id: int) -> dict:
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

agent = Agent('openai:gpt-4o', output_type=Summary)

@agent.output_validator
async def validate_bullets(ctx: RunContext[None], out: Summary) -> Summary:
    if len(out.bullets) < 3:
        raise ModelRetry(
            f'Need at least 3 bullet points, got {len(out.bullets)}. '
            'Please add more detail.'
        )
    if out.word_count <= 0:
        raise ModelRetry('word_count must be positive.')
    return out

result = agent.run_sync('Summarise the Pydantic AI framework.')
print(result.output)
```

---

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
agent = Agent('openai:gpt-4o')

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

agent = Agent('openai:gpt-4o')

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
from pydantic_ai import UsageLimits

limits = UsageLimits(
    request_limit=50,           # max number of model requests (default 50)
    tool_calls_limit=20,        # max successful tool executions
    input_tokens_limit=8_000,   # max prompt tokens per run
    output_tokens_limit=2_000,  # max generated tokens per run
    total_tokens_limit=10_000,  # combined input + output cap
    count_tokens_before_request=True,  # pre-check tokens (Anthropic, Google, Bedrock)
)
```

### Tracking usage after a successful run

```python
from pydantic_ai import Agent, RunUsage

agent = Agent('openai:gpt-4o')
result = agent.run_sync('Hello')

usage: RunUsage = result.usage()
print(f'requests={usage.requests}  input={usage.input_tokens}  output={usage.output_tokens}  total={usage.total_tokens}')
```

### Accumulating usage across multiple runs

```python
from pydantic_ai import Agent, RunUsage

agent = Agent('openai:gpt-4o')
shared_usage = RunUsage()

for prompt in ['One', 'Two', 'Three']:
    result = agent.run_sync(prompt, usage=shared_usage)

print('Grand total tokens:', shared_usage.total_tokens)
```

---

## `UserError` — programming mistakes

`UserError` means you (the developer) passed invalid arguments. It is raised at agent construction or at the start of a run, never mid-stream.

```python
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError

try:
    # Mixing tool output types incorrectly
    agent = Agent('openai:gpt-4o', output_type=int)
    # UserError can also surface from bad run() arguments:
    agent.run_sync('hi', output_type=int)   # overriding output_type with an output_validator set raises UserError
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
    'openai:gpt-4o',
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

## Testing error paths

### 1. Test that `ModelRetry` triggers properly

Use `FunctionModel` to simulate a model that first produces bad output, then good output.

```python
import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.test import TestModel

class Price(BaseModel):
    amount: float
    currency: str

agent = Agent('openai:gpt-4o', output_type=Price)

@agent.output_validator
async def no_negative(ctx: RunContext[None], p: Price) -> Price:
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
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models.test import TestModel

agent = Agent('openai:gpt-4o')

def test_request_limit():
    with agent.override(model=TestModel()):
        with pytest.raises(UsageLimitExceeded, match='request'):
            agent.run_sync('hi', usage_limits=UsageLimits(request_limit=0))
```

### 3. Verify `UnexpectedModelBehavior` is raised when retries are exhausted

```python
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

agent = Agent('openai:gpt-4o')

@agent.tool(retries=2)
def always_fails(ctx: RunContext[None]) -> str:
    raise ModelRetry('always broken')

def test_exhausted_retries():
    with agent.override(model=TestModel()):
        with pytest.raises(UnexpectedModelBehavior):
            agent.run_sync('go')
```

### 4. Inspect messages after a failure with `capture_run_messages`

```python
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel
from pydantic import BaseModel

class Strict(BaseModel):
    required_field: str

agent = Agent('openai:gpt-4o', output_type=Strict, retries=1)

def test_messages_on_failure():
    with capture_run_messages() as messages:
        with pytest.raises((UnexpectedModelBehavior, Exception)):
            with agent.override(model=TestModel(custom_output_text='not json')):
                agent.run_sync('go')

    # Messages contain the full conversation up to the point of failure
    assert len(messages) > 0
    print('Conversation had', len(messages), 'messages before failure')
```

### 5. Inject specific exceptions via `FunctionModel`

```python
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior

def model_that_raises(messages, info):
    # Simulate provider returning an unexpected response
    return ModelResponse(parts=[])   # empty parts triggers UnexpectedModelBehavior

agent = Agent(FunctionModel(model_that_raises))

def test_empty_response():
    with pytest.raises(UnexpectedModelBehavior):
        agent.run_sync('anything')
```

---

## Retry patterns

### Tool with exponential backoff

```python
import asyncio
import httpx
from pydantic_ai import Agent, ModelRetry, RunContext

agent = Agent('openai:gpt-4o')

@agent.tool(retries=4)
async def fetch_api(ctx: RunContext[None], url: str) -> str:
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

agent = Agent('openai:gpt-4o')

@agent.tool(retries=3)
def parse_structured(ctx: RunContext[None], raw: str) -> dict:
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
            tokens_used=result.usage().total_tokens,
        )
    except UsageLimitExceeded as e:
        return RunFailure(reason=str(e), kind='budget_exceeded')
    except UnexpectedModelBehavior as e:
        return RunFailure(reason=str(e), kind='model_error')
    except AgentRunError as e:
        return RunFailure(reason=str(e), kind='run_error')
```

---

## pytest fixture: agents pre-wired for error testing

```python
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart

@pytest.fixture
def deterministic_agent(my_agent: Agent):
    with my_agent.override(model=TestModel(seed=42)):
        yield my_agent

@pytest.fixture
def failing_agent(my_agent: Agent):
    def always_empty(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('')])
    with my_agent.override(model=FunctionModel(always_empty)):
        yield my_agent
```

---

## Reference

- `ModelRetry` — `pydantic_ai/exceptions.py`
- `UnexpectedModelBehavior`, `AgentRunError`, `UserError` — `pydantic_ai/exceptions.py`
- `UsageLimitExceeded` — `pydantic_ai/usage.py`
- `ConcurrencyLimitExceeded` — `pydantic_ai/concurrency.py`
- `UsageLimits` — `pydantic_ai/usage.py`
- `RunUsage` — `pydantic_ai/usage.py`
- `capture_run_messages()` — `pydantic_ai/_agent_graph.py`
- `FunctionModel`, `AgentInfo` — `pydantic_ai/models/function.py`
- `TestModel` — `pydantic_ai/models/test.py`
- `Agent.override(...)` — `pydantic_ai/agent/__init__.py`

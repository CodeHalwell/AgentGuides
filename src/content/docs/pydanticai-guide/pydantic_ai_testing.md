---
title: "PydanticAI: Testing with TestModel, FunctionModel & Overrides"
description: "Deterministic testing of PydanticAI agents using TestModel, FunctionModel, capture_run_messages, agent.override(), and pytest fixtures."
framework: pydanticai
language: python
---

# Testing Agents

Verified against **pydantic-ai==2.8.0** — source modules: `pydantic_ai.models.test`, `pydantic_ai.models.function`, `pydantic_ai.agent`.

PydanticAI ships two model implementations built for tests: `TestModel` (auto-generates tool calls + a response from JSON schema) and `FunctionModel` (you write the response-generating function). Combined with `agent.override(...)` and `capture_run_messages`, you can unit-test agents hermetically — no network, no API keys, deterministic.

## Minimal runnable example

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

agent = Agent('openai:gpt-5.2', system_prompt='Be terse.')

def test_greet():
    with agent.override(model=TestModel()):
        out = agent.run_sync('Hi')
        assert isinstance(out.output, str)
```

`agent.override(...)` returns a context manager — always use `with` (or hold a single instance and call `__enter__` / `__exit__` on _that_ same object). The override reverts on exit.

## `TestModel` — structural test double

Lives at `pydantic_ai.models.test.TestModel` (`models/test.py:60`). Given the agent's tool schemas, it:

1. Calls every tool once (unless you restrict with `call_tools=[...]`).
2. Synthesises tool arguments that match each tool's JSON schema.
3. Produces a final response — a string, or args that match the output tool's schema.

Constructor (`models/test.py:94`):

| Argument              | Type                                | Default | Purpose                                                       |
| --------------------- | ----------------------------------- | ------- | ------------------------------------------------------------- |
| `call_tools`          | `list[str] \| Literal['all']`       | `'all'` | Which tools to call. Empty list = skip tools, go straight to output. |
| `custom_output_text`  | `str \| None`                       | `None`  | Force this string as the final text output.                   |
| `custom_output_args`  | `Any \| None`                       | `None`  | Force these args for the output tool (overrides schema-gen).  |
| `seed`                | `int`                               | `0`     | Seed for the schema-driven arg generator.                     |
| `model_name` / `profile` / `settings` | — | — | Forwarded to the base `Model`. |

After the run, `TestModel.last_model_request_parameters` holds the final `ModelRequestParameters` — useful for asserting tools were offered in a given shape.

### Asserting tool calls happened

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel

agent = Agent('openai:gpt-5.2')

@agent.tool
def lookup(ctx: RunContext[None], sku: str) -> dict:
    return {'sku': sku, 'price': 9.99}

def test_model_calls_lookup():
    tm = TestModel()
    with agent.override(model=tm):
        result = agent.run_sync('Price of SKU ABC?')
    # TestModel invokes every tool; verify via the messages
    tool_names = [
        p.tool_name
        for m in result.all_messages()
        for p in m.parts
        if getattr(p, 'part_kind', None) == 'tool-call'
    ]
    assert 'lookup' in tool_names
```

### Forcing specific output

```python
tm = TestModel(custom_output_text='mocked reply')
with agent.override(model=tm):
    result = agent.run_sync('ignored')
assert result.output == 'mocked reply'
```

For agents with `output_type=MyModel`, set `custom_output_args` to a dict matching the schema.

## `FunctionModel` — write your own response

`pydantic_ai.models.function.FunctionModel` (`models/function.py:45`) lets you implement the model as a function. You get the full message history and metadata, and return a `ModelResponse`.

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

def echo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    last = messages[-1].parts[-1]
    return ModelResponse(parts=[TextPart(content=f'echo: {last.content}')])

agent = Agent(FunctionModel(echo))

result = agent.run_sync('hello')
assert result.output == 'echo: hello'
```

`AgentInfo` (`models/function.py:219`) exposes what the agent decided to send this step:

- `function_tools: list[ToolDefinition]`
- `output_tools: list[ToolDefinition]`
- `allow_text_output: bool`
- `model_settings: ModelSettings | None`
- `model_request_parameters: ModelRequestParameters`
- `instructions: str | None`

Use `FunctionModel` when you need branching behaviour (e.g. "first call returns a tool call, second returns the final answer"):

```python
from pydantic_ai.messages import ToolCallPart

def first_tool_then_answer(messages, info):
    calls = [p for m in messages for p in m.parts if isinstance(p, ToolCallPart)]
    if not calls:
        return ModelResponse(parts=[ToolCallPart(tool_name='lookup', args={'sku': 'ABC'})])
    return ModelResponse(parts=[TextPart('done')])
```

### Streaming `FunctionModel`

Pass `stream_function=` (either alone or alongside `function=`). The stream function is an **async generator** that yields `str` chunks (text deltas), `DeltaToolCalls`, or `DeltaThinkingCalls`. Yielding a `str` produces a text delta; `DeltaToolCalls` simulates a streaming tool call:

```python
import asyncio
from collections.abc import AsyncIterator
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

# A simple stream function that yields a sentence word-by-word
async def stream_word_by_word(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[str]:
    sentence = 'The answer is forty-two.'
    for word in sentence.split():
        yield word + ' '

agent = Agent(FunctionModel(stream_function=stream_word_by_word))

async def test_streaming():
    chunks = []
    async with agent.run_stream('What is the answer?') as stream:
        async for chunk in stream.stream_text(delta=True):
            chunks.append(chunk)
        final = await stream.get_output()

    assert final == 'The answer is forty-two. '
    assert len(chunks) > 1   # Multiple streamed chunks received

asyncio.run(test_streaming())
```

### Streaming tool calls with `DeltaToolCalls`

To simulate a streaming model that emits a tool call then returns text, use `DeltaToolCalls` — a `dict[int, DeltaToolCall]` keyed by a per-call index (verified in `models/function.py` source):

```python
import asyncio
from collections.abc import AsyncIterator
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

calls_log: list[str] = []

def spy_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Non-streaming: inspect AgentInfo before deciding what to return."""
    calls_log.append(f"turn {len(messages)} — tools available: {[t.name for t in info.function_tools]}")

    prior_tool_calls = [p for m in messages for p in m.parts if isinstance(p, ToolCallPart)]
    if not prior_tool_calls:
        # First turn: emit a tool call
        return ModelResponse(parts=[ToolCallPart(tool_name='lookup', args={'id': '42'})])
    # Second turn: return the answer using context from the tool result
    return ModelResponse(parts=[TextPart(content='The answer is 42.')])

async def streaming_tool_call(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[str | DeltaToolCalls]:
    """Streaming: first turn emits a DeltaToolCalls dict, second turn yields text tokens."""
    prior_tool_calls = [p for m in messages for p in m.parts if isinstance(p, ToolCallPart)]
    if not prior_tool_calls:
        # First turn: stream the tool call incrementally via DeltaToolCalls
        # DeltaToolCalls is dict[int, DeltaToolCall] — key is the per-call index
        yield {0: DeltaToolCall(name='lookup', json_args='{"id": ')}
        yield {0: DeltaToolCall(json_args='"42"}')}
    else:
        # Second turn (tool already ran): stream the answer word by word
        for word in 'The answer is forty-two .'.split():
            yield word + ' '

agent = Agent(
    FunctionModel(spy_function, stream_function=streaming_tool_call),
    deps_type=None,
)

@agent.tool_plain
def lookup(id: str) -> str:
    return f'record:{id}'

# Non-streaming: two-turn dialogue (tool call → answer)
result = agent.run_sync('What is 42?')
assert result.output == 'The answer is 42.'
print(calls_log)  # shows two entries — one per LLM turn

# Streaming: two-turn flow — first turn streams a tool call, second turn streams text
async def test_stream():
    tokens: list[str] = []
    async with agent.run_stream('What is 42?') as stream:
        # debounce_by=None gives one token per yield so we can verify multiple chunks
        async for chunk in stream.stream_text(delta=True, debounce_by=None):
            tokens.append(chunk)
        final = await stream.get_output()
    assert 'forty-two' in final
    assert len(tokens) > 1  # multiple word chunks from the second turn

asyncio.run(test_stream())
```

`AgentInfo` fields available inside any function/stream function (verified `models/function.py`):

| Field | Type | Description |
|---|---|---|
| `function_tools` | `list[ToolDefinition]` | Tools registered via `@agent.tool` / `@agent.tool_plain` |
| `output_tools` | `list[ToolDefinition]` | Output tool schema(s) for structured output |
| `allow_text_output` | `bool` | `True` if plain-text output is accepted |
| `model_settings` | `ModelSettings \| None` | Settings passed to the run |
| `model_request_parameters` | `ModelRequestParameters` | Full wire parameters |
| `instructions` | `str \| None` | Resolved instruction string sent to the model |

## `Agent.override(...)` — swap parts per run/test

`agent/__init__.py:1639`. Temporarily replaces any of:

- `model` (`Model`, a `KnownModelName`, or any `str`)
- `deps`
- `toolsets`, `tools`
- `instructions`
- `model_settings`, `metadata`, `name`

Returns a context manager — everything reverts on exit. Overrides are captured in `contextvars`, so they are safe under asyncio concurrency (each task sees its own overrides).

```python
with agent.override(model=TestModel(), deps=FakeDB()):
    result = agent.run_sync('query')
```

### pytest fixture

```python
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

@pytest.fixture
def test_agent(my_agent: Agent):
    with my_agent.override(model=TestModel()):
        yield my_agent
```

## `capture_run_messages` — test error paths

```python
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior

def test_bad_output_surfaces_messages():
    agent = Agent(..., output_type=StrictSchema)
    with capture_run_messages() as msgs:
        with pytest.raises(UnexpectedModelBehavior):
            agent.run_sync('...')
    # Inspect what the model actually produced
    assert any('tool-call' == getattr(p, 'part_kind', None) for m in msgs for p in m.parts)
```

Only the **first** `run*` call inside the `with` is captured — don't loop runs inside one context.

## Snapshotting tool traffic

```python
from pydantic_ai.messages import ModelMessagesTypeAdapter

def test_tool_flow(snapshot):
    with agent.override(model=TestModel()):
        result = agent.run_sync('go')
    snapshot.assert_match(
        ModelMessagesTypeAdapter.dump_json(result.all_messages(), indent=2),
        'tool_flow.json',
    )
```

Pair with [syrupy](https://github.com/syrupy-project/syrupy) or pytest-snapshot.

## Feature comparison

| Feature                 | `TestModel`                          | `FunctionModel`                                        |
| ----------------------- | ------------------------------------ | ------------------------------------------------------ |
| Auto-generates args     | yes (from JSON schema + `seed`)      | no — you build the `ModelResponse`                     |
| Calls every tool        | yes (unless `call_tools=[...]`)      | only if your function emits `ToolCallPart`             |
| Deterministic           | yes                                  | yes                                                    |
| Good for                | smoke tests, end-to-end schema check | protocol tests, multi-step scenarios, error injection  |
| Streaming               | yes (via `TestStreamedResponse`)     | yes (pass `stream_function=`)                          |

## Mocking real providers vs `TestModel`

| Approach                           | When to use                                                   |
| ---------------------------------- | ------------------------------------------------------------- |
| `TestModel`                        | You want schema-correct, deterministic behaviour. Fast.      |
| `FunctionModel`                    | You need to control the exact `ModelResponse`.                |
| Real model + `pytest-vcr`          | Regression-test against the real provider. Slow, flaky, needs API keys on record. |
| `respx` / `httpx_mock` + real `OpenAIModel` | HTTP-layer testing. Flakier than `FunctionModel`.      |

For most unit tests, prefer `TestModel` or `FunctionModel`. For contract tests, use real models in a CI nightly job.

## Patterns

### 1. Test an agent's _contract_ without calling an LLM

```python
def test_pricing_tool_is_registered():
    tm = TestModel()
    with agent.override(model=tm):
        agent.run_sync('anything')
    names = [t.name for t in tm.last_model_request_parameters.function_tools]
    assert 'pricing' in names
```

### 2. Inject failing tools to test recovery

```python
@agent.tool
def fragile(ctx):
    raise ValueError('boom')

def test_retries_then_succeeds():
    with agent.override(model=TestModel()):
        result = agent.run_sync('do it')
    assert result.output  # agent recovered past the tool error
```

### 3. Drive a multi-turn protocol with `FunctionModel`

```python
def script(messages, info):
    step = sum(1 for m in messages if m.kind == 'response')
    if step == 0:
        return ModelResponse(parts=[ToolCallPart('search', {'q': 'x'})])
    if step == 1:
        return ModelResponse(parts=[ToolCallPart('refine', {'doc_id': 1})])
    return ModelResponse(parts=[TextPart('final')])
```

### 4. Assert `ModelRetry` is triggered by an output validator

```python
from pydantic_ai import ModelRetry

@agent.output_validator
async def must_be_uppercase(ctx, out: str) -> str:
    if out != out.upper():
        raise ModelRetry('uppercase please')
    return out

def test_validator_retries():
    with agent.override(model=TestModel(custom_output_text='hello')):
        with pytest.raises(UnexpectedModelBehavior):
            agent.run_sync('go')
```

After `output_retries` attempts, `ModelRetry` bubbles up as `UnexpectedModelBehavior`.

### 5. Swap `deps` per test without rebuilding the agent

```python
fake_db = FakeDB([{'sku': 'ABC', 'price': 9.99}])
with agent.override(deps=fake_db, model=TestModel()):
    result = agent.run_sync('price ABC')
```

## Gotchas

- **`TestModel` randomises tool args**. Pin behaviour with `seed=42` if you assert on them.
- **`override` leaks if you manually `__enter__` without `__exit__`.** Always use `with`.
- **Async tests**: prefer `await agent.run(...)` inside `async def test_*` — don't mix `run_sync` and a running event loop.
- **`include_return_schema`**: `TestModel` does _not_ honour tool return schemas unless the agent is set to include them; see `IncludeReturnSchemasToolset`.

## Testing `AgentInfo` — assert what the agent offers the model

`AgentInfo` is passed to your `FunctionModel` function. Inspect it to write whitebox assertions about tool registration, output modes, and model settings:

```python
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

class WeatherReport(BaseModel):
    city: str
    temperature_c: float
    conditions: str

@dataclass
class Deps:
    api_key: str

agent = Agent('test', output_type=WeatherReport, deps_type=Deps)

@agent.tool
def get_weather(ctx: RunContext[Deps], city: str) -> str:
    return f'Current weather in {city}: 22°C, sunny'

captured_info: AgentInfo | None = None

def capture_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    global captured_info
    captured_info = info
    # Return a valid WeatherReport-like tool output
    return ModelResponse(parts=[TextPart(content='{"city":"London","temperature_c":22.0,"conditions":"sunny"}')])

def test_agent_offers_correct_tools():
    with agent.override(model=FunctionModel(capture_model)):
        agent.run_sync('What is the weather in London?', deps=Deps(api_key='key'))

    assert captured_info is not None
    tool_names = {t.name for t in captured_info.function_tools}
    assert 'get_weather' in tool_names

    # Verify output tools are configured
    assert len(captured_info.output_tools) > 0   # WeatherReport output tool

    # Verify allow_text_output behaviour
    assert captured_info.allow_text_output is False  # Structured output only

test_agent_offers_correct_tools()
print('All assertions passed.')
```

## Testing structured output with `TestModel`

Use `custom_output_args` to inject specific structured output and verify your output validators:

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

class Invoice(BaseModel):
    amount: float
    currency: str
    description: str

agent = Agent('test', output_type=Invoice)

@agent.output_validator
async def validate_currency(ctx, invoice: Invoice) -> Invoice:
    if invoice.currency not in ('USD', 'EUR', 'GBP'):
        from pydantic_ai import ModelRetry
        raise ModelRetry(f'Unknown currency: {invoice.currency!r}. Use USD, EUR, or GBP.')
    return invoice

def test_valid_invoice():
    tm = TestModel(custom_output_args={
        'amount': 99.99,
        'currency': 'USD',
        'description': 'Monthly subscription',
    })
    with agent.override(model=tm):
        result = agent.run_sync('Generate an invoice.')
    assert result.output.amount == 99.99
    assert result.output.currency == 'USD'

test_valid_invoice()
```

## Reference

- `TestModel` — `models/test.py:60`
- `FunctionModel`, `AgentInfo`, `DeltaToolCalls` — `models/function.py`
- `Agent.override(...)` — `agent/__init__.py:1639`
- `capture_run_messages()` — `_agent_graph.py:1791`
- `ModelMessagesTypeAdapter` — `messages.py:2034`

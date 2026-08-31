---
title: "PydanticAI: 10 More Source-Verified Class Deep Dives (2.36.0)"
description: "Runnable, source-verified code examples for AgentRun, AgentRunResult, StreamedRunResult, ModelSettings, Tool, ToolDefinition, RunUsage/RequestUsage, ConcurrencyLimiter/ConcurrencyLimit, MCPToolset, and ApprovalRequiredToolset/DynamicToolset — verified against pydantic-ai 2.36.0."
framework: pydanticai
language: python
---

# 10 More Source-Verified Class Deep Dives

Verified against **pydantic-ai 2.36.0** (installed package, sources read via `inspect.getsource`).
Modules consulted: `pydantic_ai/run.py`, `pydantic_ai/result.py`, `pydantic_ai/settings.py`,
`pydantic_ai/tools.py`, `pydantic_ai/usage.py`, `pydantic_ai/concurrency.py`,
`pydantic_ai/mcp.py`, `pydantic_ai/toolsets/{approval_required,_dynamic}.py`.

This page complements the first deep-dive page (`pydantic_ai_class_examples_2026_08.md`), which
covered `Agent`, `RunContext`, `UsageLimits`, `ToolReturn`, `DeferredToolRequests`/`Results`,
`CachePoint`, `PrefixedToolset`/`FilteredToolset`/`RenamedToolset`, `WebSearchTool`, and the
error taxonomy. All classes here are distinct from that set.

```bash
pip install "pydantic-ai==2.36.0"
python -c "import pydantic_ai; print(pydantic_ai.__version__)"
#> 2.36.0
```

---

## 1. `AgentRun` — node-by-node iteration

`pydantic_ai.run.AgentRun` is the object you get inside `async with agent.iter(...) as agent_run:`.
It exposes every node the graph executes so you can inspect, mutate, skip, or cancel mid-flight.

### Core public API

| Method / property | What it does |
|---|---|
| `async for node in agent_run` | Yield each graph node as it completes (auto-drive loop). |
| `await agent_run.next(node)` | Manually run the given node; returns the next node or `End`. |
| `agent_run.cancel()` | Cancel the whole run; in-flight tools are drained; raises `RunCancelled` on context exit. |
| `agent_run.enqueue(*content)` | Inject content into the conversation mid-run (e.g. from a webhook). |
| `agent_run.all_messages()` | All messages so far (includes history passed via `message_history`). |
| `agent_run.new_messages()` | Only messages produced by this run. |
| `agent_run.result` | `AgentRunResult` once `End` is reached; `AttributeError` if run is not finished. |

### Runnable example — recording every node

```python
import asyncio
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o-mini', system_prompt='Be concise.')


async def main() -> None:
    nodes = []
    async with agent.iter('What is 6 × 7?') as run:
        async for node in run:
            nodes.append(type(node).__name__)

    print('Nodes visited:', nodes)
    # e.g. ['UserPromptNode', 'ModelRequestNode', 'CallToolsNode', ...]
    print('Answer:', run.result.output)
    print('Requests:', run.result.usage().requests)


asyncio.run(main())
```

### Manual drive — inspect before advancing

`agent_run.next(node)` lets you swap out a node's content or selectively skip steps:

```python
import asyncio
from pydantic_ai import Agent
from pydantic_graph import End

agent = Agent('openai:gpt-4o-mini')


async def main() -> None:
    async with agent.iter('What is the capital of France?') as run:
        # next_node is a property exposing the first pending node
        node = run.next_node

        while not isinstance(node, End):
            print(f'  Running: {type(node).__name__}')
            node = await run.next(node)

    print('Result:', run.result.output)


asyncio.run(main())
```

### Cancellation mid-flight

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.exceptions import RunCancelled

agent = Agent('openai:gpt-4o-mini', system_prompt='Think step by step.')


async def main() -> None:
    try:
        async with agent.iter('Enumerate primes to 1000') as run:
            async for node in run:
                # Cancel if a ModelRequestNode is seen (demo only)
                if type(node).__name__ == 'ModelRequestNode':
                    run.cancel()
                    break
    except RunCancelled as exc:
        # Partial history is preserved
        print('Cancelled. Messages so far:', len(exc.all_messages()))


asyncio.run(main())
```

---

## 2. `AgentRunResult` — the final result object

`pydantic_ai.run.AgentRunResult[OutputDataT]` is returned by `agent.run()` / `agent.run_sync()`
and is available on `AgentRun.result` after iteration completes.

### Fields and methods

| Member | Type | Purpose |
|---|---|---|
| `output` | `OutputDataT` | The validated output value. |
| `all_messages()` | `list[ModelMessage]` | Every message (includes prior history). |
| `new_messages()` | `list[ModelMessage]` | Only this run's messages. |
| `all_messages_json()` | `bytes` | JSON-serialised full message list. |
| `new_messages_json()` | `bytes` | JSON-serialised new messages only. |
| `usage()` | `RunUsage` | Token and cost counters for this run. |

### Runnable example — accessing the full result surface

```python
import json
from pydantic import BaseModel
from pydantic_ai import Agent


class Summary(BaseModel):
    title: str
    key_points: list[str]


agent = Agent(
    'openai:gpt-4o-mini',
    output_type=Summary,
    system_prompt='Summarise the text.',
)

result = agent.run_sync(
    'PydanticAI is a type-safe agent framework. '
    'It uses Pydantic for validation and supports many LLM providers.'
)

# Structured output
print(result.output.title)
print(result.output.key_points)

# Usage
u = result.usage()
print(f'Requests: {u.requests}, in={u.input_tokens}, out={u.output_tokens}')

# Serialise for storage
msgs_bytes = result.new_messages_json()
msgs = json.loads(msgs_bytes)
print(f'{len(msgs)} messages serialised ({len(msgs_bytes)} bytes)')

# Feed history into the next run
followup = agent.run_sync(
    'Give me a one-line version of those key points.',
    message_history=result.all_messages(),
)
print(followup.output.title)
```

---

## 3. `StreamedRunResult` — streaming structured data

`pydantic_ai.result.StreamedRunResult[AgentDepsT, OutputDataT]` is the object inside
`async with agent.run_stream(...) as streamed:`. It wraps the live `AgentStream` and provides
ergonomic iteration methods.

### Public methods

| Method | Signature | Notes |
|---|---|---|
| `stream_text()` | `AsyncIterator[str]` | Yields raw text chunks regardless of `output_type`; pass `delta=True` for incremental chunks. |
| `stream_output()` | `AsyncIterator[OutputDataT]` | Partially-validated structured output on each chunk; use this when `TextOutput` functions must apply. |
| `stream_response()` | `AsyncIterator[ModelResponse]` | Cumulative `ModelResponse` snapshots; `response.state` is `'incomplete'` while streaming, `'complete'` on the final yield. |
| `await get_output()` | `-> OutputDataT` | Drain the stream and return the final validated value. |
| `await cancel()` | `-> None` | Abort the stream; interrupted response is recorded in message history. |
| `all_messages()` | `-> list[ModelMessage]` | Full message history after stream completes. |
| `new_messages()` | `-> list[ModelMessage]` | Only this run's messages. |
| `is_complete` | `bool` | `True` once one of the stream/get methods fully completes. |

### Runnable example — streaming text

```python
import asyncio
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o-mini')


async def main() -> None:
    async with agent.run_stream('Count slowly from 1 to 5') as streamed:
        async for chunk in streamed.stream_text():
            print(chunk, end='', flush=True)
        print()  # newline
        print('Complete:', streamed.is_complete)
        print('Tokens:', streamed.usage().output_tokens)


asyncio.run(main())
```

### Runnable example — streaming structured output with partial validation

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent


class Plan(BaseModel):
    steps: list[str]
    estimated_hours: float


agent = Agent('openai:gpt-4o-mini', output_type=Plan)


async def main() -> None:
    async with agent.run_stream('Plan a 3-step sprint for a new REST API') as streamed:
        # stream_output yields partial Plan objects as tokens arrive
        async for partial in streamed.stream_output(debounce_by=0.05):
            print(f'  steps so far: {len(partial.steps)}')

        # Use the public get_output() to retrieve the final validated value
        final = await streamed.get_output()
        print('Final plan:', final)


asyncio.run(main())
```

### `get_output()` — drain without manual iteration

```python
import asyncio
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o-mini')


async def main() -> None:
    async with agent.run_stream('What colour is the sky?') as streamed:
        # Drains the stream internally, returns the complete output string
        answer = await streamed.get_output()
    print(answer)


asyncio.run(main())
```

---

## 4. `ModelSettings` — cross-provider LLM knobs

`pydantic_ai.settings.ModelSettings` is a `TypedDict` (all fields optional) containing every
setting that pydantic-ai sends on the wire. Pass it to `Agent(model_settings=...)` or to
individual `agent.run(model_settings=...)` calls.

### All 16 fields (source-verified, 2.36.0)

| Field | Type | Provider support (abbreviated) |
|---|---|---|
| `max_tokens` | `int` | OpenAI, Anthropic, Google, Groq, Bedrock, Mistral, Cohere, xAI, HuggingFace, Cerebras, Ollama, OpenRouter, Snowflake, Z.AI, MCP Sampling |
| `temperature` | `float` | All of the above |
| `top_p` | `float` | All of the above (except MCP Sampling) |
| `top_k` | `int` | Anthropic, Google, Groq, xAI, Cerebras, Ollama, OpenRouter, Snowflake, Z.AI, HuggingFace |
| `timeout` | `float \| httpx.Timeout` | All (client-level default override) |
| `parallel_tool_calls` | `bool` | OpenAI (some models), Anthropic, Groq |
| `tool_choice` | `ToolChoiceT` | OpenAI, Anthropic, Google, Groq, Bedrock |
| `seed` | `int` | OpenAI, Google, Groq, xAI, Cerebras, HuggingFace, Ollama, OpenRouter |
| `presence_penalty` | `float` | OpenAI, Google, Groq, xAI, Cerebras, Ollama, OpenRouter |
| `frequency_penalty` | `float` | OpenAI, Google, Groq, xAI, Cerebras, Ollama, OpenRouter |
| `logit_bias` | `dict[str, int]` | OpenAI Chat Completions |
| `stop_sequences` | `list[str]` | OpenAI Chat Completions, Google, Groq, Bedrock, Cohere, Mistral |
| `extra_headers` | `dict[str, str]` | OpenAI, Anthropic, Google |
| `thinking` | `bool \| ThinkingSettings` | Anthropic (extended thinking), Google (thinking budget) |
| `service_tier` | `ServiceTier` | OpenAI |
| `extra_body` | `dict[str, Any]` | OpenAI, Anthropic, Groq, HuggingFace, Cerebras |

### Runnable example — deterministic coding agent

```python
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

coding_agent = Agent(
    'openai:gpt-4o',
    system_prompt='You are a Python expert. Return only code, no prose.',
    model_settings=ModelSettings(
        temperature=0.0,       # deterministic
        max_tokens=512,
        seed=42,
        stop_sequences=['```'],  # stop after the first code block
        parallel_tool_calls=False,
    ),
)

result = coding_agent.run_sync('Write a function that returns the nth Fibonacci number.')
print(result.output)
```

### Per-call override (additive merge)

```python
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

agent = Agent(
    'anthropic:claude-opus-4-5',
    model_settings=ModelSettings(temperature=0.3, max_tokens=1024),
)

# The per-run settings are merged on top — temperature stays 0.3, max_tokens bumped
result = agent.run_sync(
    'Draft a creative short story',
    model_settings=ModelSettings(max_tokens=2048),
)
print(len(result.output))
```

### Enabling extended thinking (Anthropic)

```python
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

agent = Agent(
    'anthropic:claude-opus-4-5',
    model_settings=ModelSettings(
        thinking=True,       # enable with provider default budget
        temperature=1.0,     # required when thinking is on (Anthropic)
        max_tokens=8000,
    ),
)
result = agent.run_sync('Solve: if a train travels 120 km in 1.5 hours, what is its speed?')
print(result.output)
```

---

## 5. `Tool` — explicit tool construction

`pydantic_ai.tools.Tool` is the underlying object every `@agent.tool` decorator produces. You
can construct one explicitly when you need full control over its metadata.

### Constructor signature (source-verified, 2.36.0)

```python
Tool(
    function,              # sync or async callable
    *,
    takes_ctx=None,        # auto-detect if first param is RunContext
    max_retries=None,      # overrides Agent's default retries for this tool
    name=None,             # override inferred name
    description=None,      # override docstring description
    prepare=None,          # async fn(ctx, tool_def) -> ToolDefinition | None
    args_validator=None,   # fn(ctx, args) -> args | raise
    docstring_format='auto',            # 'auto' | 'google' | 'numpy' | 'sphinx'
    require_parameter_descriptions=False,
    strict=None,           # None = provider default; True = strict JSON schema
    sequential=False,      # if True, tool calls are serialised (not parallel)
    requires_approval=False, # if True, tool call is paused for HITL approval
    metadata=None,         # arbitrary dict attached to ToolDefinition
    timeout=None,          # seconds; overrides Agent's tool_timeout
    defer_loading=False,   # don't inspect function until first use
    include_return_schema=None,
)
```

### Runnable example — explicit construction with all knobs

```python
from pydantic_ai import Agent, RunContext, Tool


async def fetch_price(ctx: RunContext[str], ticker: str) -> float:
    """Return the current stock price for *ticker*.

    Args:
        ticker: The stock symbol (e.g. AAPL).
    """
    # In production, call a real API here
    prices = {'AAPL': 189.5, 'GOOG': 175.2}
    return prices.get(ticker.upper(), 0.0)


price_tool = Tool(
    fetch_price,
    name='get_stock_price',
    description='Fetch the latest stock price for a given ticker symbol.',
    max_retries=2,
    timeout=10.0,
    sequential=False,        # allow parallel calls
    requires_approval=False,
    metadata={'category': 'finance', 'version': '1'},
)

agent = Agent('openai:gpt-4o-mini', deps_type=str, tools=[price_tool])

result = agent.run_sync('What is the price of Apple stock?', deps='user-123')
print(result.output)
```

### Dynamic tool hiding with `prepare`

The `prepare` callback receives `RunContext` and the proposed `ToolDefinition`. Return `None` to
hide the tool from this step, or return a (possibly modified) `ToolDefinition` to expose it:

```python
import asyncio
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.tools import ToolDefinition


async def admin_action(ctx: RunContext[dict], action: str) -> str:
    """Perform a privileged admin action."""
    return f'Executed: {action}'


async def prepare_admin(ctx: RunContext[dict], tool_def: ToolDefinition) -> ToolDefinition | None:
    if ctx.deps.get('role') != 'admin':
        return None  # hide the tool for non-admins
    return tool_def


admin_tool = Tool(admin_action, prepare=prepare_admin)

agent = Agent('openai:gpt-4o-mini', deps_type=dict, tools=[admin_tool])


async def main() -> None:
    # Non-admin: tool is invisible
    result = await agent.run('Do an admin action', deps={'role': 'user'})
    print('user:', result.output)

    # Admin: tool is visible
    result = await agent.run('Do an admin action', deps={'role': 'admin'})
    print('admin:', result.output)


asyncio.run(main())
```

---

## 6. `ToolDefinition` — the schema object sent to the model

`pydantic_ai.tools.ToolDefinition` is the dataclass that carries the JSON schema for a tool.
Models receive this; your code may inspect or modify it (e.g. inside a `prepare` function).

### Fields (source-verified, 2.36.0)

| Field | Type | Default | Purpose |
|---|---|---|---|
| `name` | `str` | — | Tool name the model calls. |
| `parameters_json_schema` | `ObjectJsonSchema` | `{'type': 'object', 'properties': {}}` | JSON schema for arguments. |
| `description` | `str \| None` | `None` | Sent verbatim to the model. |
| `outer_typed_dict_key` | `str \| None` | `None` | For output tools whose schema is not `object`. |
| `strict` | `bool \| None` | `None` | `True` = strict schema enforcement; `False` = disable; `None` = provider default. |

### Runnable example — inspecting the generated schema

```python
import asyncio
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.tools import ToolDefinition


def calculate_tax(income: float, rate: float = 0.2) -> float:
    """Calculate income tax.

    Args:
        income: Gross income in GBP.
        rate: Tax rate (0.0–1.0). Defaults to 20%.
    """
    return income * rate


async def inspect_schema(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition:
    print('Tool name :', tool_def.name)
    print('Schema    :', tool_def.parameters_json_schema)
    print('Strict    :', tool_def.strict)
    return tool_def


tax_tool = Tool(calculate_tax, prepare=inspect_schema)
agent = Agent('openai:gpt-4o-mini', tools=[tax_tool])


async def main() -> None:
    result = await agent.run('What is the tax on £50,000 at 22%?', deps=None)
    print(result.output)


asyncio.run(main())
```

### Modifying the schema in `prepare`

```python
import asyncio
from copy import deepcopy
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.tools import ToolDefinition


def search(query: str, max_results: int = 5) -> list[str]:
    """Search the knowledge base."""
    return [f'Result {i} for {query}' for i in range(max_results)]


async def add_enum_constraint(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition:
    schema = deepcopy(tool_def.parameters_json_schema)
    # Restrict max_results to a small set at the schema level
    schema['properties']['max_results'] = {'type': 'integer', 'enum': [3, 5, 10]}
    return ToolDefinition(
        name=tool_def.name,
        description=tool_def.description,
        parameters_json_schema=schema,
        strict=True,
    )


agent = Agent('openai:gpt-4o-mini', tools=[Tool(search, prepare=add_enum_constraint)])


asyncio.run(agent.run('Search for python async patterns', deps=None))
```

---

## 7. `RunUsage` / `RequestUsage` — token and cost accounting

Both live in `pydantic_ai.usage`. `RunUsage` aggregates across all requests in a run;
`RequestUsage` represents one request's usage (its `requests` property always returns `1`).

### `RunUsage` fields (source-verified, 2.36.0)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `requests` | `int` | `0` | Total LLM API calls made. |
| `tool_calls` | `int` | `0` | Successful tool executions. |
| `input_tokens` | `int` | `0` | Prompt tokens sent to the model. |
| `cache_write_tokens` | `int` | `0` | Tokens written to the prompt cache (Anthropic). |
| `cache_read_tokens` | `int` | `0` | Cache hit tokens (charged at a lower rate). |
| `input_audio_tokens` | `int` | `0` | Audio input tokens (multimodal models). |
| `cache_audio_read_tokens` | `int` | `0` | Cached audio tokens read. |
| `output_tokens` | `int` | `0` | Tokens generated by the model. |
| `output_audio_tokens` | `int` | `0` | Audio output tokens. |
| `details` | `dict[str, int]` | `{}` | Provider-specific extra counters. |
| `cost` | `Decimal \| None` | `None` | Total cost in USD (when genai-prices data is available). |

### Runnable example — reading usage after a run

```python
from decimal import Decimal
from pydantic_ai import Agent, UsageLimits

agent = Agent('openai:gpt-4o-mini')

result = agent.run_sync(
    'Explain the difference between TCP and UDP in one paragraph.',
    usage_limits=UsageLimits(output_tokens_limit=300),
)

u = result.usage()
print(f'requests          : {u.requests}')
print(f'input_tokens      : {u.input_tokens}')
print(f'output_tokens     : {u.output_tokens}')
print(f'cache_read_tokens : {u.cache_read_tokens}')
print(f'cost              : {u.cost}')  # Decimal or None
```

### Accumulating usage across multiple runs

`RunUsage` supports `+` (produces a new `RunUsage`) and `.incr()` (mutates in place):

```python
from pydantic_ai import Agent
from pydantic_ai.usage import RunUsage

agent = Agent('openai:gpt-4o-mini')

questions = [
    'What is a monad?',
    'What is a functor?',
    'What is currying?',
]

total = RunUsage()
for q in questions:
    r = agent.run_sync(q)
    total.incr(r.usage())

print(f'Total over {len(questions)} runs:')
print(f'  requests     : {total.requests}')
print(f'  input_tokens : {total.input_tokens}')
print(f'  output_tokens: {total.output_tokens}')
```

### Per-request granularity with `RequestUsage`

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse

agent = Agent('openai:gpt-4o-mini')


async def main() -> None:
    result = await agent.run('Name three functional programming languages.')
    for msg in result.all_messages():
        if isinstance(msg, ModelResponse) and hasattr(msg, 'usage'):
            ru = msg.usage  # RequestUsage
            print(f'  in={ru.input_tokens} out={ru.output_tokens} '
                  f'cache_read={ru.cache_read_tokens}')


asyncio.run(main())
```

---

## 8. `ConcurrencyLimiter` + `ConcurrencyLimit` — backpressure for model calls

`pydantic_ai.concurrency.ConcurrencyLimiter` wraps an `anyio.CapacityLimiter` and adds
optional backpressure (queue depth cap) and OpenTelemetry span creation for waiting operations.
`ConcurrencyLimit` is a lightweight config dataclass for passing limits to `Agent`.

### `ConcurrencyLimiter` constructor

```python
ConcurrencyLimiter(
    max_running: int,         # max simultaneous operations (>= 1)
    *,
    max_queued: int | None = None,   # None = unlimited queue; int = raise when exceeded
    name: str | None = None,  # shown in OTel spans
    tracer: Tracer | None = None,    # OTel tracer (auto-detected when None)
)
```

### `ConcurrencyLimit` fields

```python
@dataclass
class ConcurrencyLimit:
    max_running: int
    max_queued: int | None = None  # None = no queue cap
```

### Runnable example — rate-limiting parallel agent calls

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.concurrency import ConcurrencyLimiter

agent = Agent('openai:gpt-4o-mini')

# At most 3 concurrent requests to the LLM API; queue up to 10 more
limiter = ConcurrencyLimiter(max_running=3, max_queued=10, name='batch-agent')


async def ask(question: str) -> str:
    # ConcurrencyLimiter uses acquire/release — it is NOT an async context manager
    await limiter.acquire(source='agent:batch-agent')
    try:
        result = await agent.run(question)
        return result.output
    finally:
        limiter.release()


async def main() -> None:
    questions = [f'What is {i} × {i}?' for i in range(1, 11)]
    answers = await asyncio.gather(*(ask(q) for q in questions))
    for q, a in zip(questions, answers):
        print(f'{q} → {a[:60]}')


asyncio.run(main())
```

### Attach a limiter directly to an Agent

```python
from pydantic_ai import Agent
from pydantic_ai.concurrency import ConcurrencyLimit

# ConcurrencyLimit is passed to Agent.max_concurrency;
# internally PydanticAI creates a ConcurrencyLimiter from it
agent = Agent(
    'openai:gpt-4o-mini',
    max_concurrency=ConcurrencyLimit(max_running=5, max_queued=20),
)

result = agent.run_sync('Summarise async/await in Python.')
print(result.output)
```

### Handling `ConcurrencyLimitExceeded`

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.concurrency import ConcurrencyLimiter
from pydantic_ai.exceptions import UserError

agent = Agent('openai:gpt-4o-mini')
# Very tight: only 1 running, only 1 queued
tight = ConcurrencyLimiter(max_running=1, max_queued=1, name='tight')


async def safe_ask(q: str) -> str | None:
    try:
        await tight.acquire(source='agent:tight')
    except Exception as exc:
        if 'ConcurrencyLimitExceeded' in type(exc).__name__:
            print(f'Dropped (queue full): {q}')
            return None
        raise
    try:
        result = await agent.run(q)
        return result.output
    finally:
        tight.release()


async def main() -> None:
    tasks = [safe_ask(f'Question {i}') for i in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if r:
            print(r[:60])


asyncio.run(main())
```

---

## 9. `MCPToolset` — connecting to MCP servers

`pydantic_ai.mcp.MCPToolset` is pydantic-ai's first-class MCP integration. It wraps a
FastMCP `Client` and exposes the server's tools, resources, and prompts as a toolset.

### Constructor (source-verified, 2.36.0)

```python
MCPToolset(
    client,                  # URL str, script path, FastMCP Server instance, or fastmcp.Client
    *,
    # Pydantic AI layer
    id=None,                 # required for durable execution (Temporal/DBOS)
    max_retries=None,
    tool_error_behavior='retry',   # 'retry' | 'error' | 'failed'
    process_tool_call=None,        # async callback for audit / transform
    prefer_tasks=True,             # run tool calls as async tasks (durable-exec-friendly)
    cache_tools=True,
    cache_resources=True,
    cache_prompts=True,
    include_instructions=False,    # inject MCP server instructions as system prompt
    include_return_schema=None,
    # Sampling
    sampling_model=None,
    sampling_handler=None,
    # MCP protocol
    elicitation_handler=None,
    log_handler=None,
    log_level=None,
    progress_handler=None,
    message_handler=None,
    client_info=None,
    init_timeout=None,
    read_timeout=None,
    roots=None,
    # HTTP-specific
    auth=None,             # httpx.Auth | 'oauth' | str bearer token
    verify=None,
    headers=None,
    http_client=None,
)
```

### Runnable example — HTTP MCP server

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

# Connects to a locally running MCP server over Streamable HTTP
toolset = MCPToolset(
    'http://localhost:8000/mcp',
    include_instructions=True,   # server instructions go into the system prompt
    tool_error_behavior='error',
    cache_tools=True,
)

agent = Agent('openai:gpt-4o-mini', toolsets=[toolset])


async def main() -> None:
    # The MCPToolset is an async context manager — agent.run opens it automatically
    result = await agent.run('Search the knowledge base for "async patterns"')
    print(result.output)


# asyncio.run(main())  # uncomment with a real server running
```

### Stdio (subprocess) MCP server

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

# Spawns a local Python script as an MCP server over stdio
toolset = MCPToolset(
    'path/to/my_mcp_server.py',
    max_retries=3,
    tool_error_behavior='retry',
)

agent = Agent('openai:gpt-4o', toolsets=[toolset])
# result = agent.run_sync('Use the calculator tool to compute 1337 * 42')
```

### In-process FastMCP server (testing / integration)

```python
from fastmcp import FastMCP
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

mcp_server = FastMCP('test-server')


@mcp_server.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


toolset = MCPToolset(mcp_server)
agent = Agent('openai:gpt-4o-mini', toolsets=[toolset])

# result = agent.run_sync('What is 17 + 25?')
# print(result.output)
```

### Loading multiple servers from a Claude-Desktop-style config

`load_mcp_toolsets` reads the `mcpServers` JSON config and wraps each server in a
`PrefixedToolset` to avoid name collisions:

```python
# mcp_config.json:
# {
#   "mcpServers": {
#     "calculator": { "command": "python", "args": ["calc_server.py"] },
#     "search":     { "url": "http://localhost:8001/mcp" }
#   }
# }

from pydantic_ai import Agent
from pydantic_ai.mcp import load_mcp_toolsets

toolsets = load_mcp_toolsets('mcp_config.json')
# toolsets[0] is PrefixedToolset('calculator', MCPToolset(...))
# toolsets[1] is PrefixedToolset('search', MCPToolset(...))

agent = Agent('openai:gpt-4o', toolsets=toolsets)
# result = agent.run_sync('Add 5 and 3, then search for Python tutorials')
```

### OAuth-authenticated MCP server

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

toolset = MCPToolset(
    'https://api.example.com/mcp',
    auth='oauth',            # pydantic-ai handles the OAuth flow via FastMCP
    headers={'X-Client-Id': 'my-app'},
    init_timeout=10.0,
    read_timeout=30.0,
)

agent = Agent('openai:gpt-4o', toolsets=[toolset])
# result = await agent.run('Fetch my calendar events for today')
```

---

## 10. `ApprovalRequiredToolset` + `DynamicToolset` — gated and adaptive tools

These two toolsets cover the HITL approval pattern and run-time toolset composition.

### `ApprovalRequiredToolset` — pause for human approval

`pydantic_ai.toolsets.ApprovalRequiredToolset` wraps another toolset. Before each tool call it
checks an `approval_required_func`. If the function returns `True`, pydantic-ai raises
`ApprovalRequired` (which surfaces as `DeferredToolRequests`), pausing the run until you call
it again with the approved (or denied) results.

```python
@dataclass
class ApprovalRequiredToolset(WrapperToolset[AgentDepsT]):
    approval_required_func: Callable[
        [RunContext[AgentDepsT], ToolDefinition, dict[str, Any]], bool
    ] = lambda ctx, tool_def, tool_args: True
```

#### Runnable example — approve expensive operations

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai._deferred import DeferredToolRequests
from pydantic_ai.toolsets import FunctionToolset, ApprovalRequiredToolset
from pydantic_ai.tools import ToolDefinition


async def send_email(ctx: RunContext[None], to: str, body: str) -> str:
    """Send an email."""
    print(f'[EMAIL] To={to} Body={body[:40]}...')
    return f'Email sent to {to}'


async def check_balance(ctx: RunContext[None]) -> float:
    """Check account balance."""
    return 1234.56


def needs_approval(ctx: RunContext[None], tool_def: ToolDefinition, args: dict) -> bool:
    # Only gate send_email; check_balance is always allowed
    return tool_def.name == 'send_email'


base_toolset = FunctionToolset([send_email, check_balance])
gated = ApprovalRequiredToolset(base_toolset, approval_required_func=needs_approval)

# output_type must include DeferredToolRequests so the agent can surface the pause
agent = Agent(
    'openai:gpt-4o-mini',
    output_type=[str, DeferredToolRequests],
    toolsets=[gated],
)


async def main() -> None:
    # First pass — model decides to call send_email; agent returns DeferredToolRequests
    result = await agent.run(
        'Check my balance and then email support@example.com saying I have sufficient funds.'
    )

    if isinstance(result.output, DeferredToolRequests):
        deferred = result.output
        print('Approval needed for:',
              [c.tool_name for c in deferred.approvals])

        # Approve all pending approval-requiring calls at once
        approved_results = deferred.build_results(approve_all=True)

        # Resume: pass approved results + full message history so the model has context
        final = await agent.run(
            'Continue.',
            deferred_tool_results=approved_results,
            message_history=result.all_messages(),
        )
        print(final.output)
    else:
        # No approval needed — plain string output
        print(result.output)


asyncio.run(main())
```

### `DynamicToolset` — build the toolset from `RunContext`

`pydantic_ai.toolsets._dynamic.DynamicToolset` evaluates a factory function on every run (or
every model step, when `per_run_step=True`) to return a different toolset based on context:

```python
class DynamicToolset(AbstractToolset[AgentDepsT]):
    def __init__(
        self,
        toolset_func: Callable[[RunContext[AgentDepsT]], AbstractToolset | None | Awaitable],
        *,
        per_run_step: bool = True,  # re-evaluate each model step
        id: str | None = None,       # required for durable execution
    ): ...
```

#### Runnable example — tools that depend on the user's subscription tier

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset, DynamicToolset


async def basic_search(ctx: RunContext[dict], query: str) -> str:
    """Search the public index."""
    return f'Public results for: {query}'


async def premium_search(ctx: RunContext[dict], query: str) -> str:
    """Search the premium index with semantic re-ranking."""
    return f'Premium results (re-ranked) for: {query}'


async def export_csv(ctx: RunContext[dict], data: str) -> str:
    """Export data as CSV."""
    return f'CSV export: {data[:20]}...'


basic_toolset = FunctionToolset([basic_search])
premium_toolset = FunctionToolset([basic_search, premium_search, export_csv])


def select_toolset(ctx: RunContext[dict]):
    tier = ctx.deps.get('subscription', 'free')
    return premium_toolset if tier == 'premium' else basic_toolset


agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=dict,
    toolsets=[DynamicToolset(select_toolset, per_run_step=True)],
)


async def main() -> None:
    # Free user — only basic_search available
    r1 = await agent.run('Search for async patterns', deps={'subscription': 'free'})
    print('Free:', r1.output[:80])

    # Premium user — all three tools available
    r2 = await agent.run('Search for async patterns', deps={'subscription': 'premium'})
    print('Premium:', r2.output[:80])


asyncio.run(main())
```

#### Async factory + `per_run_step=False` (evaluate once per run)

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset, DynamicToolset


async def fetch_tools_from_registry(ctx: RunContext[dict]) -> FunctionToolset:
    """Fetch enabled tools from a remote registry at run start."""
    # In production, call an API here
    enabled = ctx.deps.get('enabled_tools', [])
    fns = [fn for fn in [basic_search] if fn.__name__ in enabled]
    return FunctionToolset(fns)


async def basic_search(ctx: RunContext[dict], q: str) -> str:  # noqa: F811
    return f'Results for {q}'


agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=dict,
    toolsets=[DynamicToolset(fetch_tools_from_registry, per_run_step=False)],
)

# asyncio.run(agent.run('Search something', deps={'enabled_tools': ['basic_search']}))
```

---

## Quick reference

| Class | Import | Key constructor arg(s) | Common use |
|---|---|---|---|
| `AgentRun` | `pydantic_ai.run` | (obtained via `agent.iter()`) | Node-by-node inspection, cancellation, enqueue |
| `AgentRunResult` | `pydantic_ai.run` | (returned by `agent.run()`) | Access output, messages, usage |
| `StreamedRunResult` | `pydantic_ai.result` | (inside `agent.run_stream()`) | Streaming text or structured output |
| `ModelSettings` | `pydantic_ai.settings` | TypedDict — all fields optional | Per-agent or per-call LLM knobs |
| `Tool` | `pydantic_ai.tools` | `function`, optional `prepare`, `requires_approval` | Explicit tool with full metadata control |
| `ToolDefinition` | `pydantic_ai.tools` | `name`, `parameters_json_schema` | Inspect/modify schema inside `prepare` |
| `RunUsage` | `pydantic_ai.usage` | (returned by `result.usage()`) | Token + cost accounting, `incr()`, `+` |
| `ConcurrencyLimiter` | `pydantic_ai.concurrency` | `max_running`, `max_queued` | Rate-limit parallel LLM calls |
| `ConcurrencyLimit` | `pydantic_ai.concurrency` | `max_running`, `max_queued` | Config dataclass for `Agent(max_concurrency=)` |
| `MCPToolset` | `pydantic_ai.mcp` | `client` (URL/path/server) | Connect to any MCP server |
| `ApprovalRequiredToolset` | `pydantic_ai.toolsets` | `toolset`, `approval_required_func` | HITL gate on tool calls |
| `DynamicToolset` | `pydantic_ai.toolsets` | `toolset_func`, `per_run_step` | Context-dependent toolset selection |

---
title: "PydanticAI: Class Deep Dives"
description: "Source-verified deep dives into AgentRun, AgentRunResult, ConcurrencyLimiter, the Direct API, capture_run_messages, Tool advanced params, ModelSettings, Hooks, Thinking, WebSearch, UsageLimits, and more."
framework: pydanticai
language: python
---

# PydanticAI: Class Deep Dives

Verified against **pydantic-ai==1.101.0** — each section links to the source module inspected.

This guide covers twelve classes/utilities from the installed source that get light treatment elsewhere. Every example is derived directly from the `__init__`, docstrings, and behaviour seen in the installed package.

---

## 1. `AgentRun` — node-level iteration

Source: `pydantic_ai/run.py` — `AgentRun[AgentDepsT, OutputDataT]`

`agent.iter()` returns an `AgentRun` async context manager. Iterating it yields graph nodes one by one: `UserPromptNode` → `ModelRequestNode` → `CallToolsNode` → `End`. Use it when you need to inspect, mutate, or branch on individual steps.

### Properties at a glance

| Property | Type | Description |
|----------|------|-------------|
| `next_node` | `AgentNode \| End` | The node that will run on the next `next()` call |
| `result` | `AgentRunResult \| None` | `None` until an `End` is reached |
| `run_id` | `str` | Unique ID for this run (stable across retries) |
| `conversation_id` | `str` | Shared ID for all runs in the same conversation |
| `metadata` | `dict \| None` | Arbitrary metadata attached at construction |
| `ctx` | `GraphRunContext` | Raw graph context (state + deps) |

Methods: `all_messages()`, `new_messages()`, `all_messages_json()`, `new_messages_json()`, `next(node)`.

### Minimal iteration

```python
import asyncio
from pydantic_ai import Agent
from pydantic_graph import End

agent = Agent('openai:gpt-4o')

async def main():
    async with agent.iter('What is the capital of France?') as agent_run:
        print('run_id:', agent_run.run_id)

        async for node in agent_run:
            print(type(node).__name__, '—', repr(node)[:80])

        # result is populated once End is reached
        print('output:', agent_run.result.output)

asyncio.run(main())
```

### Manual driving with `next()`

`next(node)` fires capability hooks (`before_node_run`, `wrap_node_run`, `after_node_run`) — important when capabilities like `Hooks` are registered. Bare `async for` skips those hooks.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_graph import End

agent = Agent('openai:gpt-4o')

async def main():
    async with agent.iter('Write one sentence about Python.') as run:
        node = run.next_node          # start with the first scheduled node
        while not isinstance(node, End):
            print('Executing:', type(node).__name__)
            node = await run.next(node)  # drive manually

        print('Final output:', run.result.output)
        print('run_id:', run.run_id)
        print('conversation_id:', run.conversation_id)

asyncio.run(main())
```

### Inspecting messages mid-run

```python
import asyncio
from pydantic_ai import Agent
from pydantic_graph import End

agent = Agent('openai:gpt-4o')

async def main():
    async with agent.iter('Summarise Python in one sentence.') as run:
        node = run.next_node
        while not isinstance(node, End):
            node = await run.next(node)
            # Peek at messages accumulated so far
            partial_messages = run.all_messages()
            print(f'After {type(node).__name__}: {len(partial_messages)} message(s)')

    # After the run, all_messages() is the full history
    print('Total messages:', len(run.all_messages()))
    print('New messages (this run):', len(run.new_messages()))

asyncio.run(main())
```

### Multi-turn with shared `conversation_id`

```python
import asyncio
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def chat_session():
    # Turn 1
    result1 = await agent.run('My name is Alice.')
    print('conversation_id:', result1.conversation_id)

    # Turn 2 — same conversation_id is shared
    async with agent.iter(
        'What is my name?',
        message_history=result1.all_messages(),
    ) as run2:
        node = run2.next_node
        from pydantic_graph import End
        while not isinstance(node, End):
            node = await run2.next(node)

        print('conversation_id matches:', run2.conversation_id == result1.conversation_id)
        print(run2.result.output)   # Your name is Alice.

asyncio.run(chat_session())
```

---

## 2. `AgentRunResult` — the final result object

Source: `pydantic_ai/run.py` — `AgentRunResult[OutputDataT]`

`agent.run()` and `agent.run_sync()` return `AgentRunResult`. Beyond `.output`, it exposes rich metadata and message manipulation tools.

### Key properties and methods

```python
import asyncio
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def main():
    result = await agent.run('Explain Python in one sentence.')

    # Core output
    print(result.output)

    # Run identity
    print('run_id:', result.run_id)
    print('conversation_id:', result.conversation_id)

    # result.usage is a read property in v1.98+ — calling result.usage() emits DeprecationWarning
    print('usage:', result.usage)                   # RunUsage(...)
    print('timestamp:', result.timestamp)           # datetime

    # The last ModelResponse from the model
    print('model used:', result.response.model_name)

    # Message history
    all_msgs = result.all_messages()                # includes system/user/tool messages
    new_msgs = result.new_messages()                # only messages from this run
    as_json  = result.all_messages_json()           # bytes — store in a DB column

    print(f'{len(all_msgs)} total, {len(new_msgs)} new')

asyncio.run(main())
```

### `output_tool_return_content` — continuing a conversation with modified context

When an agent's output type is a Pydantic model, PydanticAI internally calls a hidden *output tool* to extract the structured value. If you want to continue the conversation and modify what the model "sees" as the tool's return value, pass `output_tool_return_content`:

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent

class Summary(BaseModel):
    title: str
    body: str

agent = Agent('openai:gpt-4o', output_type=Summary)

async def main():
    result = await agent.run('Summarise Python.')

    # Inject a custom return so the model can refer to the summary in follow-up
    modified_history = result.all_messages(
        output_tool_return_content='Summary accepted. Please proceed.'
    )

    # Follow-up uses the modified history
    follow_up = await agent.run(
        'Now add three bullet points to the summary.',
        message_history=modified_history,
    )
    print(follow_up.output)

asyncio.run(main())
```

### `metadata` — attaching arbitrary data to a run

```python
import asyncio
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def main():
    result = await agent.run(
        'Tell me about Python.',
        metadata={'user_id': 42, 'source': 'web'},
    )
    # Metadata is available on the result
    print(result.metadata)   # {'user_id': 42, 'source': 'web'}

asyncio.run(main())
```

---

## 3. `ConcurrencyLimiter` and `limit_model_concurrency`

Source: `pydantic_ai/concurrency.py`

New in **v1.96.0**. Use `ConcurrencyLimiter` to cap parallel model requests — useful for rate-limit compliance, cost control, and fairness across tenants.

### Simple cap on a single model

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai import ConcurrencyLimiter, limit_model_concurrency

# Allow at most 3 parallel requests; queue up to 10 more
model = limit_model_concurrency('openai:gpt-4o', limiter=ConcurrencyLimiter(3, max_queued=10))
agent = Agent(model)

async def process_batch(prompts: list[str]) -> list[str]:
    """Process many prompts with a shared concurrency cap."""
    tasks = [agent.run(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    return [r.output for r in results]

async def main():
    prompts = [f'Summarise topic {i}' for i in range(20)]
    outputs = await process_batch(prompts)
    print(f'Processed {len(outputs)} prompts')

asyncio.run(main())
```

### `ConcurrencyLimit` dataclass for explicit backpressure

```python
from pydantic_ai import ConcurrencyLimiter
from pydantic_ai.concurrency import ConcurrencyLimit

# Using the dataclass for named config
limit_config = ConcurrencyLimit(max_running=5, max_queued=20)
limiter = ConcurrencyLimiter.from_limit(limit_config, name='gpt4o-pool')

print('max_running:', limiter.max_running)     # 5
print('waiting_count:', limiter.waiting_count) # 0
print('running_count:', limiter.running_count) # 0
print('available:', limiter.available_count)   # 5
```

### Sharing a limiter across multiple agents

```python
import asyncio
from pydantic_ai import Agent, ConcurrencyLimiter, limit_model_concurrency

# One shared limiter for all agents in this service
shared_limiter = ConcurrencyLimiter(max_running=10, max_queued=50, name='service-pool')

agent_a = Agent(limit_model_concurrency('openai:gpt-4o', limiter=shared_limiter))
agent_b = Agent(limit_model_concurrency('anthropic:claude-sonnet-4-6', limiter=shared_limiter))

async def main():
    # Both agents compete for slots from the same pool
    results = await asyncio.gather(
        agent_a.run('What is 1+1?'),
        agent_b.run('What is 2+2?'),
    )
    for r in results:
        print(r.output)

asyncio.run(main())
```

### Handling `ConcurrencyLimitExceeded`

```python
import asyncio
from pydantic_ai import Agent, ConcurrencyLimiter, limit_model_concurrency
from pydantic_ai.exceptions import ConcurrencyLimitExceeded

tight_limiter = ConcurrencyLimiter(max_running=1, max_queued=0)
agent = Agent(limit_model_concurrency('openai:gpt-4o', limiter=tight_limiter))

async def safe_run(prompt: str) -> str | None:
    try:
        result = await agent.run(prompt)
        return result.output
    except ConcurrencyLimitExceeded as e:
        print(f'Dropped: {e}')
        return None

async def main():
    tasks = [safe_run(f'Question {i}') for i in range(5)]
    results = await asyncio.gather(*tasks)
    print([r for r in results if r])

asyncio.run(main())
```

### Custom `AbstractConcurrencyLimiter` (e.g. Redis-backed)

```python
from pydantic_ai.concurrency import AbstractConcurrencyLimiter

class RedisConcurrencyLimiter(AbstractConcurrencyLimiter):
    """Distribute concurrency limits across processes via Redis."""

    def __init__(self, redis_client, key: str, max_running: int):
        self._redis = redis_client
        self._key = key
        self._max_running = max_running

    async def acquire(self, source: str) -> None:
        # Use a Redis atomic increment + expiry to track running count
        current = await self._redis.incr(self._key)
        if current > self._max_running:
            await self._redis.decr(self._key)
            raise RuntimeError(f'Concurrency limit {self._max_running} exceeded')
        # Set expiry as a safety valve
        await self._redis.expire(self._key, 60)

    def release(self) -> None:
        import asyncio
        asyncio.create_task(self._redis.decr(self._key))
```

---

## 4. Direct API — `model_request` and friends

Source: `pydantic_ai/direct.py`

The **direct API** bypasses `Agent` entirely. It gives you a thin, provider-agnostic wrapper around the raw model protocol — useful for custom pipelines, evaluation harnesses, and LLM middleware that doesn't need tool calling.

All functions are importable from `pydantic_ai.direct`.

### Non-streamed async: `model_request`

```python
import asyncio
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request

async def main():
    response = await model_request(
        'openai:gpt-4o',
        [ModelRequest.user_text_prompt('What is the capital of France?')],
    )
    # ModelResponse with parts list and usage
    for part in response.parts:
        print(part)
    print('tokens:', response.usage)
    print('model:', response.model_name)

asyncio.run(main())
```

### Non-streamed sync: `model_request_sync`

```python
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_sync

response = model_request_sync(
    'anthropic:claude-haiku-4-5',
    [ModelRequest.user_text_prompt('One sentence about Python.')],
)
print(response.parts[0].content)  # TextPart.content
```

### Streamed async: `model_request_stream`

```python
import asyncio
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_stream

async def main():
    messages = [ModelRequest.user_text_prompt('Tell me a joke.')]
    async with model_request_stream('openai:gpt-4o', messages) as stream:
        async for event in stream:
            # PartStartEvent, PartDeltaEvent, PartEndEvent, FinalResultEvent
            print(type(event).__name__, repr(event)[:80])
        # After streaming: full response
        print('model_name:', stream.model_name)
        print('timestamp:', stream.timestamp)

asyncio.run(main())
```

### Streamed sync: `model_request_stream_sync` / `StreamedResponseSync`

`StreamedResponseSync` wraps the async producer in a background thread, giving you a plain `for` loop in synchronous code.

```python
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_stream_sync

messages = [ModelRequest.user_text_prompt('Who was Einstein?')]

with model_request_stream_sync('openai:gpt-4o', messages) as stream:
    for event in stream:
        print(type(event).__name__, repr(event)[:60])
    # Access the assembled response after iteration
    print('model:', stream.model_name)
    print('response:', stream.response)
```

### Multi-turn with the direct API

```python
import asyncio
from pydantic_ai import ModelRequest, ModelResponse
from pydantic_ai.messages import UserPromptPart, TextPart, SystemPromptPart
from pydantic_ai.direct import model_request

async def multi_turn():
    history: list[ModelRequest | ModelResponse] = []

    # System message
    system_msg = ModelRequest(parts=[SystemPromptPart(content='You are a concise assistant.')])
    history.append(system_msg)

    for user_input in ['My name is Bob.', 'What is my name?']:
        history.append(ModelRequest(parts=[UserPromptPart(content=user_input)]))
        response = await model_request('openai:gpt-4o', history)
        history.append(response)
        print(f'User: {user_input}')
        for part in response.parts:
            if isinstance(part, TextPart):
                print(f'Assistant: {part.content}')
        print()

asyncio.run(multi_turn())
```

### Direct API with `model_settings`

```python
import asyncio
from pydantic_ai import ModelRequest, ModelSettings
from pydantic_ai.direct import model_request

async def main():
    response = await model_request(
        'openai:gpt-4o',
        [ModelRequest.user_text_prompt('List 3 Python benefits.')],
        model_settings=ModelSettings(temperature=0.1, max_tokens=200),
    )
    print(response.parts[0].content)

asyncio.run(main())
```

---

## 5. `capture_run_messages` — debug message history from failed runs

Source: `pydantic_ai/agent/__init__.py`

When an agent run raises an exception (e.g. `UsageLimitExceeded`, `UnexpectedModelBehavior`, or your own `ModelRetry` loop failing), the messages collected before the failure are normally lost. `capture_run_messages` is a context manager that captures them regardless.

```python
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai import UsageLimits

agent = Agent('openai:gpt-4o')

with capture_run_messages() as messages:
    try:
        result = agent.run_sync(
            'Count from 1 to 1000.',
            usage_limits=UsageLimits(request_limit=1),
        )
    except UsageLimitExceeded:
        # Inspect whatever messages were built before the limit hit
        print(f'Captured {len(messages)} messages before failure')
        for msg in messages:
            print(type(msg).__name__, str(msg)[:100])
```

### Async usage

```python
import asyncio
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai import UsageLimits

agent = Agent('openai:gpt-4o')

async def main():
    with capture_run_messages() as messages:
        try:
            result = await agent.run(
                'What is 1+1?',
                usage_limits=UsageLimits(total_tokens_limit=5),
            )
        except UsageLimitExceeded as e:
            print(f'Failed after {len(messages)} messages: {e}')
            for m in messages:
                print(' ', type(m).__name__)

asyncio.run(main())
```

### Only the first `run*` call is captured

If you call `run_sync` (or `run` / `run_stream`) more than once inside a single `capture_run_messages` block, only the _first_ call's messages are captured. Nest two context managers for two runs:

```python
from pydantic_ai import Agent, capture_run_messages

agent = Agent('openai:gpt-4o')

with capture_run_messages() as msgs_1:
    result1 = agent.run_sync('First question.')

with capture_run_messages() as msgs_2:
    result2 = agent.run_sync('Second question.')

print(len(msgs_1), 'vs', len(msgs_2))
```

### Logging production errors

```python
import logging
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import ModelMessagesTypeAdapter

log = logging.getLogger(__name__)
agent = Agent('openai:gpt-4o')

def run_safe(prompt: str) -> str | None:
    with capture_run_messages() as messages:
        try:
            return agent.run_sync(prompt).output
        except Exception as e:
            log.error(
                'Agent run failed',
                exc_info=True,
                extra={'messages': ModelMessagesTypeAdapter.dump_python(messages)},
            )
            return None
```

---

## 6. `format_as_xml` — structure data for LLMs

Source: `pydantic_ai/format_prompt.py`

LLMs often parse semi-structured data more reliably from XML than from JSON or plain text. `format_as_xml` serialises Python objects (Pydantic models, dataclasses, dicts, lists, primitives) into indented XML.

```python
from pydantic_ai import format_as_xml

# Plain dict
print(format_as_xml({'name': 'Alice', 'age': 30}, root_tag='user'))
# <user>
#   <name>Alice</name>
#   <age>30</age>
# </user>

# List of dicts
items = [{'id': 1, 'title': 'Python'}, {'id': 2, 'title': 'Rust'}]
print(format_as_xml(items, root_tag='languages', item_tag='language'))
```

### Pydantic models and dataclasses

```python
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import format_as_xml

@dataclass
class Address:
    street: str
    city: str

@dataclass
class Person:
    name: str
    age: int
    address: Address

alice = Person('Alice', 30, Address('123 Main St', 'Springfield'))
print(format_as_xml(alice, root_tag='person'))
# <person>
#   <name>Alice</name>
#   <age>30</age>
#   <address>
#     <street>123 Main St</street>
#     <city>Springfield</city>
#   </address>
# </person>

class Product(BaseModel):
    name: str = Field(description='Product name')
    price: float = Field(description='Price in USD')
    in_stock: bool

p = Product(name='Widget', price=9.99, in_stock=True)
print(format_as_xml(p, root_tag='product', include_field_info=True))
```

### Injecting XML into a system prompt

```python
from pydantic_ai import Agent, RunContext, format_as_xml
from dataclasses import dataclass

@dataclass
class CustomerRecord:
    name: str
    account_tier: str
    open_tickets: int

agent = Agent('openai:gpt-4o', deps_type=CustomerRecord)

@agent.system_prompt
async def inject_customer_xml(ctx: RunContext[CustomerRecord]) -> str:
    customer_xml = format_as_xml(ctx.deps, root_tag='customer')
    return f'You are a support agent. The current customer is:\n{customer_xml}'

result = agent.run_sync(
    'How many open tickets do I have?',
    deps=CustomerRecord(name='Bob', account_tier='premium', open_tickets=3),
)
print(result.output)
# Bob, you have 3 open tickets.
```

### `include_field_info='once'` — annotate schemas

```python
from pydantic import BaseModel, Field
from pydantic_ai import format_as_xml

class Task(BaseModel):
    title: str = Field(description='Short task title')
    priority: int = Field(description='1 (low) to 5 (high)')

tasks = [Task(title='Fix bug', priority=5), Task(title='Write docs', priority=2)]

# include_field_info='once' adds description attributes on the first occurrence only
print(format_as_xml(tasks, root_tag='tasks', include_field_info='once'))
```

### Controlling output format

```python
from pydantic_ai import format_as_xml

data = {'x': 1, 'y': None, 'tags': ['a', 'b']}

# No indentation (compact)
compact = format_as_xml(data, indent=None)

# Custom none string
with_none = format_as_xml(data, none_str='N/A')

# No outer wrapper tag
no_root = format_as_xml(data)   # root_tag=None by default

print(compact)
print(with_none)
print(no_root)
```

---

## 7. `common_tools` — ready-made search tools

Source: `pydantic_ai/common_tools/` — four tool factories, each requiring an optional extra.

`pydantic_ai.common_tools` ships tool *factories* — functions that return `Tool` or `FunctionToolset` objects you can pass directly to an `Agent`.

| Factory | Extra | Package | Returns |
|---------|-------|---------|---------|
| `duckduckgo_search_tool()` | `duckduckgo` | `ddgs` | `Tool[Any]` |
| `tavily_search_tool()` | `tavily` | `tavily-python` | `Tool[Any]` |
| `exa_search_tool()` / `exa_find_similar_tool()` / `exa_get_contents_tool()` / `exa_answer_tool()` | `exa` | `exa-py` | `Tool[Any]` |
| `web_fetch_tool()` | `web-fetch` | `httpx` + `markdownify` | `Tool[Any]` |

### DuckDuckGo search

```bash
pip install "pydantic-ai-slim[duckduckgo]"
```

```python
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

agent = Agent(
    'openai:gpt-4o',
    tools=[duckduckgo_search_tool(max_results=5)],
    instructions='Search the web to answer questions accurately.',
)

result = agent.run_sync('What is the latest Python release?')
print(result.output)
```

The tool is registered under the name `duckduckgo_search` and accepts a single `query: str` argument. It returns `list[DuckDuckGoResult]` where each result is a `TypedDict` with `title`, `href`, and `body`.

### Tavily search

```bash
pip install "pydantic-ai-slim[tavily]"
```

```python
import os, asyncio
from pydantic_ai import Agent
from pydantic_ai.common_tools.tavily import tavily_search_tool
from tavily import AsyncTavilyClient

async def main():
    client = AsyncTavilyClient(api_key=os.environ['TAVILY_API_KEY'])

    agent = Agent(
        'openai:gpt-4o',
        tools=[tavily_search_tool(tavily_client=client, max_results=3)],
    )

    result = await agent.run('Find the latest news on PydanticAI.')
    print(result.output)

asyncio.run(main())
```

`TavilySearchResult` fields: `title`, `url`, `content`, `score`. The tool accepts `search_depth` (`'basic'` / `'advanced'`), `topic`, `time_range`, `include_domains`, `exclude_domains`.

### Exa toolset (search + content + AI answers)

```bash
pip install "pydantic-ai-slim[exa]"
```

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.common_tools.exa import ExaToolset, exa_search_tool
from exa_py import AsyncExa

async def main():
    client = AsyncExa(api_key='your-exa-key')

    # Option A: individual tool
    agent_single = Agent(
        'openai:gpt-4o',
        tools=[exa_search_tool(exa_client=client, num_results=5)],
    )

    # Option B: full ExaToolset (search + find_similar + get_contents + answer)
    agent_full = Agent(
        'openai:gpt-4o',
        toolsets=[ExaToolset(client=client)],
    )

    result = await agent_full.run('What are the top Python frameworks for 2026?')
    print(result.output)

asyncio.run(main())
```

`ExaToolset` provides four tools: `exa_search`, `exa_find_similar`, `exa_get_contents`, and `exa_answer`.

### Web fetch (local, SSRF-protected)

```bash
pip install "pydantic-ai-slim[web-fetch]"
```

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.common_tools.web_fetch import web_fetch_tool

agent = Agent(
    'openai:gpt-4o',
    tools=[
        web_fetch_tool(
            max_content_length=5000,    # cap page size
            allowed_domains=['docs.pydantic.dev', 'ai.pydantic.dev'],
            timeout=10,
        )
    ],
)

async def main():
    result = await agent.run('Fetch and summarise https://ai.pydantic.dev')
    print(result.output)

asyncio.run(main())
```

`WebFetchResult` fields: `url`, `title`, `content` (markdown). Binary responses (PDFs, images) are returned as `BinaryContent` for native model processing.

### Combining search tools

```python
import asyncio
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.common_tools.web_fetch import web_fetch_tool

async def main():
    agent = Agent(
        'openai:gpt-4o',
        tools=[
            duckduckgo_search_tool(max_results=3),
            web_fetch_tool(max_content_length=3000, timeout=15),
        ],
        instructions=(
            'Use duckduckgo_search to find relevant URLs, '
            'then use web_fetch to read the most promising one.'
        ),
    )

    result = await agent.run(
        'Find the official PEP for Python type hints and summarise it.'
    )
    print(result.output)

asyncio.run(main())
```

---

## 8. `ExternalToolset` — full round-trip

Source: `pydantic_ai/toolsets/external.py`

`ExternalToolset` declares tool *schemas* to the model but executes them outside the agent run. The agent returns `DeferredToolRequests` as its output, your code fulfils the calls, and you resume with `DeferredToolResults`.

### Complete round-trip example

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai import ExternalToolset
from pydantic_ai.output import DeferredToolRequests, DeferredToolResults
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.messages import ToolReturn

# 1. Declare the tool schema
external = ExternalToolset([
    ToolDefinition(
        name='send_email',
        description='Send an email to a user.',
        parameters_json_schema={
            'type': 'object',
            'properties': {
                'to': {'type': 'string', 'description': 'Recipient email address'},
                'subject': {'type': 'string'},
                'body': {'type': 'string'},
            },
            'required': ['to', 'subject', 'body'],
        },
    ),
    ToolDefinition(
        name='create_ticket',
        description='Create a support ticket in the system.',
        parameters_json_schema={
            'type': 'object',
            'properties': {
                'title': {'type': 'string'},
                'priority': {'type': 'string', 'enum': ['low', 'medium', 'high']},
            },
            'required': ['title', 'priority'],
        },
    ),
])

# 2. Agent uses the external toolset
agent = Agent(
    'openai:gpt-4o',
    output_type=[str, DeferredToolRequests],  # union: normal output OR deferred calls
    toolsets=[external],
)

async def execute_tool(tool_name: str, args: dict) -> str:
    """Simulate external tool execution."""
    if tool_name == 'send_email':
        return f"Email sent to {args['to']} with subject '{args['subject']}'"
    elif tool_name == 'create_ticket':
        return f"Ticket #{42} created: {args['title']} (priority: {args['priority']})"
    return 'Unknown tool'

async def main():
    # 3. First run — agent may decide to call external tools
    result1 = await agent.run(
        'Send a welcome email to new@example.com and create a high-priority onboarding ticket.'
    )

    if isinstance(result1.output, str):
        # Agent finished without needing tools
        print('Direct answer:', result1.output)
        return

    # 4. Deferred tool calls — execute them externally
    deferred: DeferredToolRequests = result1.output
    print(f'Agent requested {len(deferred.calls)} tool call(s)')

    results: dict[str, ToolReturn] = {}
    for call in deferred.calls:
        print(f'  Executing: {call.tool_name}({call.args})')
        output = await execute_tool(call.tool_name, call.args)
        results[call.tool_call_id] = ToolReturn(content=output)

    # 5. Resume — feed results back with the full message history (no new user prompt)
    result2 = await agent.run(
        message_history=result1.all_messages(),
        deferred_tool_results=DeferredToolResults(calls=results),
    )
    print('Final answer:', result2.output)

asyncio.run(main())
```

### Multi-step external tool loop

```python
import asyncio
from pydantic_ai import Agent, ExternalToolset
from pydantic_ai.output import DeferredToolRequests, DeferredToolResults
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.messages import ToolReturn

external = ExternalToolset([
    ToolDefinition(
        name='database_query',
        description='Run a read-only SQL query against the production database.',
        parameters_json_schema={
            'type': 'object',
            'properties': {'sql': {'type': 'string'}},
            'required': ['sql'],
        },
    ),
])

agent = Agent(
    'openai:gpt-4o',
    output_type=[str, DeferredToolRequests],
    toolsets=[external],
)

async def run_with_external_tools(prompt: str) -> str:
    """Drive the agent-external tool loop until a string result."""
    result = await agent.run(prompt)
    history = result.all_messages()

    while isinstance(result.output, DeferredToolRequests):
        tool_returns: dict[str, ToolReturn] = {}
        for call in result.output.calls:
            # Your real DB executor goes here
            db_result = f"SELECT returned 42 rows for: {call.args.get('sql', '')}"
            tool_returns[call.tool_call_id] = ToolReturn(content=db_result)

        result = await agent.run(
            message_history=history,
            deferred_tool_results=DeferredToolResults(calls=tool_returns),
        )
        history = result.all_messages()

    return result.output

async def main():
    answer = await run_with_external_tools('How many active users are there?')
    print(answer)

asyncio.run(main())
```

---

## 9. `FilteredToolset` — dynamic, per-step tool visibility

Source: `pydantic_ai/toolsets/filtered.py`

`FilteredToolset` wraps any toolset and applies a `(ctx, tool_def) -> bool` predicate before each model step. Returning `False` hides the tool from the model for that step. Both sync and async predicates are accepted.

### Role-based tool access

```python
from dataclasses import dataclass
from pydantic_ai import Agent, FunctionToolset, FilteredToolset, RunContext
from pydantic_ai.tools import ToolDefinition

@dataclass
class UserContext:
    user_id: int
    role: str   # 'user' | 'admin'

all_tools = FunctionToolset[UserContext]()

@all_tools.tool
def read_data(ctx: RunContext[UserContext], table: str) -> str:
    """Read rows from a database table."""
    return f'Read from {table}'

@all_tools.tool
def write_data(ctx: RunContext[UserContext], table: str, data: dict) -> str:
    """Write data to a database table."""
    return f'Wrote to {table}'

@all_tools.tool
def delete_table(ctx: RunContext[UserContext], table: str) -> bool:
    """Permanently delete a database table."""
    return True

def role_filter(ctx: RunContext[UserContext], tool_def: ToolDefinition) -> bool:
    """Admins see all tools; regular users only see read_data."""
    if ctx.deps.role == 'admin':
        return True
    return tool_def.name == 'read_data'

agent = Agent(
    'openai:gpt-4o',
    deps_type=UserContext,
    toolsets=[FilteredToolset(all_tools, filter_func=role_filter)],
)

# Regular user — only sees read_data
result = agent.run_sync(
    'List the tables and delete the old_logs table.',
    deps=UserContext(user_id=1, role='user'),
)
print(result.output)   # Will say it cannot delete (tool not available)

# Admin — sees all three tools
result_admin = agent.run_sync(
    'Delete the old_logs table.',
    deps=UserContext(user_id=99, role='admin'),
)
print(result_admin.output)
```

### Async predicate

```python
import asyncio
from pydantic_ai import Agent, FunctionToolset, FilteredToolset, RunContext
from pydantic_ai.tools import ToolDefinition

tools = FunctionToolset[None]()

@tools.tool_plain
def expensive_operation(n: int) -> int:
    return n * 1000

@tools.tool_plain
def cheap_operation(n: int) -> int:
    return n + 1

async def cost_gate(ctx: RunContext[None], tool_def: ToolDefinition) -> bool:
    """Async predicate — could call a rate-limit service."""
    if tool_def.name == 'expensive_operation':
        # Simulate async check (e.g. check quota from Redis)
        await asyncio.sleep(0)   # placeholder
        budget_remaining = 100   # from your quota store
        return budget_remaining > 0
    return True

agent = Agent('openai:gpt-4o', toolsets=[FilteredToolset(tools, filter_func=cost_gate)])

result = agent.run_sync('Run an expensive operation on 5.')
print(result.output)
```

### State-driven tool progression

```python
from dataclasses import dataclass
from pydantic_ai import Agent, FunctionToolset, FilteredToolset, RunContext
from pydantic_ai.tools import ToolDefinition

@dataclass
class WorkflowState:
    step: int = 1   # which step of the workflow we're on

tools = FunctionToolset[WorkflowState]()

@tools.tool
def step1_validate(ctx: RunContext[WorkflowState], data: str) -> str:
    ctx.deps.step = 2
    return f'Validated: {data}'

@tools.tool
def step2_transform(ctx: RunContext[WorkflowState], data: str) -> str:
    ctx.deps.step = 3
    return f'Transformed: {data}'

@tools.tool
def step3_publish(ctx: RunContext[WorkflowState], data: str) -> bool:
    ctx.deps.step = 4
    return True

def step_filter(ctx: RunContext[WorkflowState], tool_def: ToolDefinition) -> bool:
    """Only expose the tool for the current workflow step."""
    step_map = {1: 'step1_validate', 2: 'step2_transform', 3: 'step3_publish'}
    return tool_def.name == step_map.get(ctx.deps.step)

agent = Agent(
    'openai:gpt-4o',
    deps_type=WorkflowState,
    toolsets=[FilteredToolset(tools, filter_func=step_filter)],
)

state = WorkflowState(step=1)
result = agent.run_sync('Process this data: "raw input"', deps=state)
print(result.output)
```

---

## 10. `FunctionToolset` with `instructions`

Source: `pydantic_ai/toolsets/function.py`

`FunctionToolset` accepts an `instructions` parameter — a string, a `TemplateStr`, or an async callable — that is **automatically injected into the model's instructions whenever any tool in the set is active**. This lets tool documentation follow the tools, not the agent definition.

### Static instruction string

```python
from pydantic_ai import Agent, FunctionToolset, RunContext

db_tools = FunctionToolset[None](
    instructions=(
        'When using database tools: always use read-only queries first. '
        'Never execute DELETE without explicit user confirmation. '
        'Prefer LIMIT clauses on large tables.'
    )
)

@db_tools.tool_plain
def query_users(sql: str) -> list[dict]:
    """Execute a read-only SQL query on the users table."""
    return [{'id': 1, 'name': 'Alice'}]

@db_tools.tool_plain
def count_records(table: str) -> int:
    """Count the number of records in a table."""
    return 42

agent = Agent('openai:gpt-4o', toolsets=[db_tools])

# The DB instructions are injected automatically since db_tools is active
result = agent.run_sync('How many users are there?')
print(result.output)
```

### Dynamic instructions via async callable

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, FunctionToolset, RunContext

@dataclass
class TenantContext:
    tenant_id: str
    db_schema: str

async def tenant_instructions(ctx: RunContext[TenantContext]) -> str:
    """Build instructions tailored to the current tenant."""
    return (
        f'You are querying tenant {ctx.deps.tenant_id!r}. '
        f'All queries must include WHERE schema = {ctx.deps.db_schema!r}. '
        'Never cross-query tenant data.'
    )

tenant_tools = FunctionToolset[TenantContext](instructions=tenant_instructions)

@tenant_tools.tool
async def get_records(ctx: RunContext[TenantContext], table: str) -> list[dict]:
    """Fetch records for the current tenant."""
    return [{'tenant': ctx.deps.tenant_id, 'table': table}]

agent = Agent('openai:gpt-4o', deps_type=TenantContext, toolsets=[tenant_tools])

async def main():
    result = await agent.run(
        'Fetch the orders table.',
        deps=TenantContext(tenant_id='acme', db_schema='acme_prod'),
    )
    print(result.output)

asyncio.run(main())
```

### Combining multiple toolsets with distinct instructions

```python
from pydantic_ai import Agent, FunctionToolset, CombinedToolset, PrefixedToolset

search_tools = FunctionToolset[None](
    instructions='Search tools query external APIs. Respect rate limits — max 3 calls per run.'
)

@search_tools.tool_plain
def web_search(query: str) -> list[str]:
    """Search the web and return URLs."""
    return ['https://example.com/result1']

analytics_tools = FunctionToolset[None](
    instructions='Analytics tools may be slow. Cache results and avoid duplicate queries.'
)

@analytics_tools.tool_plain
def run_report(report_name: str) -> dict:
    """Run a named analytics report."""
    return {'report': report_name, 'rows': 1000}

agent = Agent(
    'openai:gpt-4o',
    toolsets=[
        PrefixedToolset(search_tools, prefix='search_'),
        PrefixedToolset(analytics_tools, prefix='analytics_'),
    ],
)

# Both sets of instructions are active when the agent runs
result = agent.run_sync('Search for Python and then run the python_adoption report.')
print(result.output)
```

### `FunctionToolset` with `timeout` per tool

The `timeout` constructor argument applies a deadline (in seconds) to every tool call in the set. If the tool exceeds it, the model receives a retry prompt automatically.

```python
from pydantic_ai import Agent, FunctionToolset, RunContext
import asyncio

slow_tools = FunctionToolset[None](timeout=5.0)   # 5-second deadline per call

@slow_tools.tool_plain
def slow_api_call(endpoint: str) -> str:
    """Call a slow external API (may time out)."""
    import time
    time.sleep(3)   # usually within budget; 6 s would exceed it
    return f'Response from {endpoint}'

agent = Agent('openai:gpt-4o', toolsets=[slow_tools])
result = agent.run_sync('Call the /metrics endpoint.')
print(result.output)
```

---

## 11. `UsageLimits` — budget controls for every run

Source: `pydantic_ai/usage.py` — `UsageLimits`

`UsageLimits` is a dataclass you pass as `usage_limits=` to any `run*` call. PydanticAI checks the budgets **before each request** (request count) and **after each response** (token counts). When a limit fires, `UsageLimitExceeded` (a subclass of `AgentRunError`) is raised immediately.

### Fields at a glance

| Field | Default | Description |
|-------|---------|-------------|
| `request_limit` | `50` | Max number of model round-trips per run |
| `tool_calls_limit` | `None` | Max successful tool executions |
| `input_tokens_limit` | `None` | Max prompt tokens per run |
| `output_tokens_limit` | `None` | Max generated tokens per run |
| `total_tokens_limit` | `None` | Combined input + output cap |
| `count_tokens_before_request` | `False` | Pre-check tokens before sending (requires provider support: Anthropic, Google, Bedrock, OpenAI Responses) |

### Minimal example

```python
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

agent = Agent('openai:gpt-4o')

try:
    result = agent.run_sync(
        'Write a 10 000-word essay.',
        usage_limits=UsageLimits(output_tokens_limit=200, request_limit=2),
    )
except UsageLimitExceeded as e:
    print('Budget hit:', e)
```

### Tracking usage after a run

```python
from pydantic_ai import Agent, RunUsage

agent = Agent('openai:gpt-4o')
result = agent.run_sync('Hello')

usage: RunUsage = result.usage()
print(
    f'requests={usage.requests} '
    f'input={usage.input_tokens} '
    f'output={usage.output_tokens} '
    f'total={usage.total_tokens}'
)
```

### Accumulating usage across many runs

Pass a single `RunUsage` instance into every `run*` call. PydanticAI increments it in place, so you get a running total across an entire session.

```python
from pydantic_ai import Agent, RunUsage

agent = Agent('openai:gpt-4o')
shared = RunUsage()

for prompt in ['One', 'Two', 'Three']:
    agent.run_sync(prompt, usage=shared)

print('Grand total:', shared.total_tokens)
```

### Pre-checking tokens before the request

When `count_tokens_before_request=True`, PydanticAI calls the model's token-counting API before sending the actual request. This enforces `input_tokens_limit` *before* the model even sees the prompt — useful when prompt construction is expensive.

```python
from pydantic_ai import Agent, UsageLimits

agent = Agent('anthropic:claude-sonnet-4-6')

result = agent.run_sync(
    'Summarise this document...',
    usage_limits=UsageLimits(
        input_tokens_limit=4_000,
        count_tokens_before_request=True,   # Anthropic / Google / Bedrock only
    ),
)
```

### Combining all limits for a strict production budget

```python
from pydantic_ai import Agent, UsageLimits

agent = Agent('openai:gpt-4o')

STRICT = UsageLimits(
    request_limit=5,
    tool_calls_limit=10,
    input_tokens_limit=8_000,
    output_tokens_limit=2_000,
    total_tokens_limit=10_000,
)

result = agent.run_sync('Research and summarise quantum computing.', usage_limits=STRICT)
```

---

## 12. `ConcurrencyLimiter` — control parallel agent runs

Source: `pydantic_ai/concurrency.py` — `ConcurrencyLimiter`

`ConcurrencyLimiter` wraps an `anyio.CapacityLimiter` and adds observability (span tracing) and a configurable queue depth. Pass it to `Agent(max_concurrency=...)` to cap how many runs execute simultaneously.

### Constructor

```python
from pydantic_ai import ConcurrencyLimiter

limiter = ConcurrencyLimiter(
    max_running=5,           # max simultaneous in-flight runs
    max_queued=20,           # max tasks allowed to queue (None = unlimited)
    name='my-agent-pool',   # appears in OpenTelemetry spans
)
```

### Properties

| Property | Description |
|----------|-------------|
| `running_count` | Number of runs currently executing |
| `waiting_count` | Number of runs blocked in the queue |
| `available_count` | Free slots (= `max_running - running_count`) |
| `name` | Name used for observability spans |

### Minimal example

```python
import asyncio
from pydantic_ai import Agent, ConcurrencyLimiter

limiter = ConcurrencyLimiter(max_running=3, name='batch')
agent = Agent('openai:gpt-4o', max_concurrency=limiter)

async def main():
    prompts = [f'Tell me about topic {i}' for i in range(10)]
    results = await asyncio.gather(*[agent.run(p) for p in prompts])
    for r in results:
        print(r.output[:80])

asyncio.run(main())
```

### Rejecting tasks when the queue is full

When `max_queued` is set and the queue depth is exceeded, `ConcurrencyLimitExceeded` is raised immediately — the task is never queued.

```python
import asyncio
from pydantic_ai import Agent, ConcurrencyLimiter
from pydantic_ai.exceptions import ConcurrencyLimitExceeded

limiter = ConcurrencyLimiter(max_running=2, max_queued=3)
agent = Agent('openai:gpt-4o', max_concurrency=limiter)

async def main():
    tasks = [agent.run(f'Task {i}') for i in range(10)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    accepted = [r for r in results if not isinstance(r, ConcurrencyLimitExceeded)]
    rejected = [r for r in results if isinstance(r, ConcurrencyLimitExceeded)]
    print(f'Accepted: {len(accepted)}, Rejected (queue full): {len(rejected)}')

asyncio.run(main())
```

### Creating from a `ConcurrencyLimit` config object

`ConcurrencyLimit` is a Pydantic-serialisable config; `ConcurrencyLimiter.from_limit()` builds the runtime limiter from it.

```python
from pydantic_ai import ConcurrencyLimiter
from pydantic_ai.concurrency import ConcurrencyLimit

config = ConcurrencyLimit(max_running=5, max_queued=50)
limiter = ConcurrencyLimiter.from_limit(config, name='production-pool')
```

### Sharing a limiter across multiple agents

One limiter can enforce a shared budget across several agents — useful when all agents compete for the same LLM API rate limit.

```python
from pydantic_ai import Agent, ConcurrencyLimiter

shared = ConcurrencyLimiter(max_running=10, name='shared-api-budget')

research_agent = Agent('openai:gpt-4o', max_concurrency=shared)
summary_agent  = Agent('openai:gpt-4o', max_concurrency=shared)
qa_agent       = Agent('openai:gpt-4o', max_concurrency=shared)
```

### Observing limiter metrics

```python
import asyncio
from pydantic_ai import Agent, ConcurrencyLimiter

limiter = ConcurrencyLimiter(max_running=4, name='monitored')
agent = Agent('openai:gpt-4o', max_concurrency=limiter)

async def main():
    tasks = [agent.run(f'Q {i}') for i in range(8)]

    async def monitor():
        while True:
            await asyncio.sleep(0.2)
            print(
                f'running={limiter.running_count} '
                f'waiting={limiter.waiting_count} '
                f'available={limiter.available_count}'
            )

    monitor_task = asyncio.create_task(monitor())
    try:
        await asyncio.gather(*tasks)
    finally:
        monitor_task.cancel()

asyncio.run(main())
```

---

## 13. `Tool` — advanced constructor parameters

Source: `pydantic_ai/tools.py` — `Tool[AgentDepsT]`

`@agent.tool` and `@agent.tool_plain` are convenient but reach for `Tool(fn, ...)` directly when you need the full parameter surface.

### Full constructor reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `function` | callable | required | The Python function to expose |
| `takes_ctx` | `bool \| None` | auto-detect | `True` means first arg is `RunContext` |
| `max_retries` | `int \| None` | None (inherit) | Retry budget for this tool; overrides agent-level default |
| `name` | `str \| None` | function name | Name sent to the model |
| `description` | `str \| None` | from docstring | Description sent to the model |
| `prepare` | `ToolPrepareFunc \| None` | None | Hook called every step to mutate or suppress the tool definition |
| `args_validator` | `ArgsValidatorFunc \| None` | None | Called after arg validation, before execution; raise `ModelRetry` to reject |
| `docstring_format` | `DocstringFormat` | `'auto'` | `'google'`, `'numpy'`, `'sphinx'`, `'auto'` |
| `require_parameter_descriptions` | `bool` | `False` | Raise at registration if any parameter lacks a description |
| `strict` | `bool \| None` | None | OpenAI/Anthropic strict JSON schema mode |
| `sequential` | `bool` | `False` | Prevent this tool from running in parallel with others |
| `requires_approval` | `bool` | `False` | Raise `ApprovalRequired` before execution — HITL gate |
| `metadata` | `dict \| None` | None | Arbitrary tags for filtering and toolset rules |
| `timeout` | `float \| None` | None | Seconds before a `ModelRetry` is sent back automatically |

### `args_validator` — validate arguments before execution

`ArgsValidatorFunc` receives the same typed parameters as the tool itself (with `RunContext` first when `takes_ctx=True`). Raise `ModelRetry` to reject.

```python
import asyncio
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.exceptions import ModelRetry

def validate_age(ctx: RunContext[None], age: int) -> None:
    if age < 0 or age > 150:
        raise ModelRetry(f'Age {age} is not plausible. Please provide a real age.')

def get_birth_year(ctx: RunContext[None], age: int) -> int:
    return 2026 - age

agent = Agent(
    'openai:gpt-4o',
    tools=[Tool(get_birth_year, args_validator=validate_age)],
)

result = agent.run_sync('What year was someone born if they are 30 years old?')
print(result.output)
```

### `prepare` — conditional and dynamic tool definitions

`prepare` is called before each model request. Return `None` to hide the tool from this step, or a modified `ToolDefinition` to change its schema on the fly.

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.tools import ToolDefinition

@dataclass
class Deps:
    is_admin: bool

async def admin_delete(ctx: RunContext[Deps], item_id: str) -> str:
    return f'Deleted {item_id}'

async def only_for_admins(
    ctx: RunContext[Deps], tool_def: ToolDefinition
) -> ToolDefinition | None:
    return tool_def if ctx.deps.is_admin else None

agent = Agent(
    'openai:gpt-4o',
    deps_type=Deps,
    tools=[Tool(admin_delete, prepare=only_for_admins)],
)

async def main():
    # Non-admin: tool hidden from the model
    result = await agent.run('Delete item 42', deps=Deps(is_admin=False))
    print('non-admin:', result.output)

    # Admin: tool visible
    result = await agent.run('Delete item 42', deps=Deps(is_admin=True))
    print('admin:', result.output)

asyncio.run(main())
```

### `requires_approval` — human-in-the-loop gate

When `requires_approval=True`, calling the tool raises `ApprovalRequired`. The run terminates with a `DeferredToolRequests` output so your code can inspect the pending call, show it to a human, then resume.

```python
import asyncio
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDenied

def send_email(ctx: RunContext[None], to: str, subject: str, body: str) -> str:
    return f'Email sent to {to}: {subject}'

agent = Agent(
    'openai:gpt-4o',
    output_type=[str, DeferredToolRequests],
    tools=[Tool(send_email, requires_approval=True)],
)

async def main():
    result = await agent.run('Send a welcome email to alice@example.com')

    if isinstance(result.output, DeferredToolRequests):
        print('Pending approvals:')
        for call in result.output.approvals:
            print(f'  {call.tool_name}({call.args_as_dict()})')

        # User approves all calls
        approvals = {call.tool_call_id: ToolApproved() for call in result.output.approvals}
        result2 = await agent.run(
            'continue',
            message_history=result.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals=approvals),
        )
        print('Final:', result2.output)
    else:
        print(result.output)

asyncio.run(main())
```

### `sequential` — enforce serial execution

When the model calls multiple tools in one step, PydanticAI normally runs them in parallel. Mark a tool `sequential=True` to force it to run alone (never concurrently with other tools that step).

```python
from pydantic_ai import Agent, RunContext, Tool

def write_file(ctx: RunContext[None], path: str, content: str) -> str:
    # Must not run concurrently with other file writes
    return f'Written {len(content)} bytes to {path}'

agent = Agent(
    'openai:gpt-4o',
    tools=[Tool(write_file, sequential=True)],
)
```

### `timeout` — kill slow tools automatically

```python
import asyncio
import time
from pydantic_ai import Agent, RunContext, Tool

def slow_operation(ctx: RunContext[None], n: int) -> str:
    time.sleep(n)   # simulate slow work
    return f'Done after {n}s'

agent = Agent(
    'openai:gpt-4o',
    # If slow_operation takes longer than 5 seconds, the model receives a retry prompt
    tools=[Tool(slow_operation, timeout=5.0)],
)
```

### `metadata` — tagging for filtering

```python
from pydantic_ai import Agent, RunContext, Tool, FilteredToolset, FunctionToolset

def search_docs(ctx: RunContext[None], query: str) -> str:
    return f'Results for {query}'

def delete_record(ctx: RunContext[None], record_id: str) -> str:
    return f'Deleted {record_id}'

# Tag tools with a 'scope' so FilteredToolset can gate them
tools = FunctionToolset([
    Tool(search_docs, metadata={'scope': 'read'}),
    Tool(delete_record, metadata={'scope': 'write'}),
])

def read_only(ctx, tool_def):
    return tool_def.metadata.get('scope') != 'write'

agent = Agent('openai:gpt-4o', toolsets=[FilteredToolset(tools, filter_func=read_only)])
```

---

## 14. `ModelSettings` — complete field reference

Source: `pydantic_ai/settings.py` — `ModelSettings` (TypedDict)

`ModelSettings` is a `TypedDict` (all keys optional) that controls how the model generates a response. Pass it to `Agent(model_settings=...)` for a global default or `agent.run(..., model_settings=...)` to override per call. Run-level settings are **merged on top** of agent-level settings.

### All fields

```python
from pydantic_ai import ModelSettings

settings: ModelSettings = {
    # --- Generation control ---
    'max_tokens': 1024,           # Hard stop on output token count
    'temperature': 0.7,           # 0.0 = deterministic, higher = creative
    'top_p': 0.9,                 # Nucleus sampling probability mass
    'top_k': 40,                  # Top-K vocabulary restriction (Gemini/Anthropic/Cohere)
    'seed': 42,                   # Reproducibility seed (OpenAI/Groq/Cohere/Mistral/Gemini)
    'stop_sequences': ['END'],    # Generation stops when any of these appear

    # --- Token biasing ---
    'presence_penalty': 0.2,      # Penalise tokens that have appeared at all
    'frequency_penalty': 0.3,     # Penalise tokens proportional to how often they appeared
    'logit_bias': {'1234': -100}, # Force/ban specific token IDs (OpenAI/Groq)

    # --- Thinking / reasoning ---
    'thinking': 'medium',         # 'minimal'|'low'|'medium'|'high'|'xhigh'|True|False

    # --- Tool use ---
    'parallel_tool_calls': True,  # Allow parallel tool execution (OpenAI/Groq/Anthropic)
    'tool_choice': 'auto',        # 'auto'|'none'|'required'|list[str]|ToolOrOutput

    # --- Latency / cost tier ---
    'service_tier': 'default',    # 'auto'|'default'|'flex'|'priority'

    # --- Network ---
    'timeout': 30.0,              # Request timeout in seconds (or httpx.Timeout)
    'extra_headers': {'X-Org': 'acme'},  # Extra HTTP headers

    # --- Provider-specific escape hatch ---
    'extra_body': {'best_of': 3}, # Forwarded verbatim in the JSON body
}
```

### Thinking levels across providers

```python
import asyncio
from pydantic_ai import Agent

# Anthropic: maps to extended-thinking budget tokens
claude_agent = Agent('anthropic:claude-opus-4-7', model_settings={'thinking': 'high'})

# OpenAI: maps to reasoning_effort
o1_agent = Agent('openai:o1', model_settings={'thinking': 'xhigh'})

# Google: enables thinking mode on Gemini 2.5+
gemini_agent = Agent('google:gemini-2.5-pro', model_settings={'thinking': 'medium'})

async def main():
    r = await claude_agent.run('Solve: what is the sum of angles in a pentagon?')
    print(r.output)

asyncio.run(main())
```

### `tool_choice` — `ToolOrOutput` for selective tool control

`'required'` or `list[str]` force every model step to call a tool, preventing final output. Use `ToolOrOutput` to specify which *function* tools are available while still allowing structured output:

```python
from pydantic_ai import Agent
from pydantic_ai.settings import ToolOrOutput

agent = Agent(
    'openai:gpt-4o',
    model_settings={
        # On the first step only offer 'search'; the model can still return a final answer
        'tool_choice': ToolOrOutput(function_tools=['search']),
    },
)
```

### Dynamic `model_settings` via capability

For per-step changes (e.g. force a tool on step 1 only), return a callable from a capability rather than a static dict:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.settings import ModelSettings, ToolOrOutput
from pydantic_ai.tools import RunContext
from dataclasses import dataclass

@dataclass
class ForcedFirstToolCall(AbstractCapability):
    tool_name: str

    def get_model_settings(self):
        def settings(ctx: RunContext) -> ModelSettings:
            if ctx.run_step == 0:
                return {'tool_choice': ToolOrOutput(function_tools=[self.tool_name])}
            return {}
        return settings

agent = Agent(
    'openai:gpt-4o',
    capabilities=[ForcedFirstToolCall('search')],
)
```

### `service_tier` — cost/latency trade-offs

```python
from pydantic_ai import Agent

# Lowest-cost batch processing (OpenAI flex / Bedrock flex)
batch_agent = Agent('openai:gpt-4o', model_settings={'service_tier': 'flex'})

# Lowest-latency production path (OpenAI priority)
realtime_agent = Agent('openai:gpt-4o', model_settings={'service_tier': 'priority'})
```

### Merging settings at run-time

```python
from pydantic_ai import Agent, ModelSettings

agent = Agent('openai:gpt-4o', model_settings={'temperature': 0.3, 'max_tokens': 500})

# Override just temperature for this specific call; max_tokens inherits from agent-level
result = agent.run_sync('Write a haiku.', model_settings={'temperature': 0.9})
```

---

## 15. `Hooks` capability — decorator-based lifecycle hooks

Source: `pydantic_ai/capabilities/hooks.py` — `Hooks[AgentDepsT]`

`Hooks` is the ergonomic alternative to subclassing `AbstractCapability`. Register functions through the `hooks.on` namespace; each attribute corresponds to a lifecycle event.

### Available hook names

| Hook | When it fires | Can modify |
|------|--------------|-----------|
| `before_run` | Before the agent run starts | — |
| `after_run` | After the agent run completes | `AgentRunResult` |
| `wrap_run` | Wraps the entire run | `AgentRunResult` |
| `on_run_error` | On any exception in the run | can return a result instead |
| `before_node_run` | Before each graph node | `AgentNode` |
| `after_node_run` | After each graph node | `NodeResult` |
| `wrap_node_run` | Wraps each node execution | `NodeResult` |
| `on_node_run_error` | On node execution error | can return a result |
| `before_model_request` | Before each model API call | `ModelRequestContext` |
| `after_model_request` | After each model API call | `ModelResponse` |
| `wrap_model_request` | Wraps each model API call | `ModelResponse` |
| `before_tool_validate` | Before tool arg validation | `RawToolArgs` |
| `after_tool_validate` | After tool arg validation | `ValidatedToolArgs` |
| `wrap_tool_validate` | Wraps tool arg validation | `ValidatedToolArgs` |
| `before_tool_execute` | Before tool function runs | `ValidatedToolArgs` |
| `after_tool_execute` | After tool function runs | tool return value |
| `wrap_tool_execute` | Wraps tool execution | tool return value |

### Decorator style

```python
import asyncio
import time
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks

hooks = Hooks()

@hooks.on.before_model_request
async def log_request(ctx, request_context):
    print(f'[step {ctx.run_step}] sending to {ctx.model.model_name}')
    return request_context

@hooks.on.after_model_request
async def log_response(ctx, *, request_context, response):
    print(f'[step {ctx.run_step}] got response: {len(response.parts)} parts')
    return response

@hooks.on.before_run
def record_start(ctx):
    ctx.metadata = ctx.metadata or {}
    ctx.metadata['start_time'] = time.monotonic()

@hooks.on.after_run
def record_duration(ctx, *, result):
    start = (ctx.metadata or {}).get('start_time', time.monotonic())
    print(f'Run took {time.monotonic() - start:.3f}s')
    return result

agent = Agent('openai:gpt-4o', capabilities=[hooks])

async def main():
    result = await agent.run('What is 1+1?')
    print(result.output)

asyncio.run(main())
```

### Constructor-kwarg style

Pass functions directly as keyword arguments matching the hook names:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks

def log_req(ctx, request_context):
    print(f'Request at step {ctx.run_step}')
    return request_context

agent = Agent(
    'openai:gpt-4o',
    capabilities=[Hooks(before_model_request=log_req)],
)
```

### Tool-specific hooks with `tools=` filter

`before_tool_execute`, `after_tool_execute`, and `wrap_tool_execute` accept an optional `tools=` argument to restrict the hook to specific tools. The filter can be `'all'`, a list of names, a metadata dict, or a callable.

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks

agent = Agent('openai:gpt-4o')

@agent.tool_plain
def search(query: str) -> str:
    return f'Results for: {query}'

@agent.tool_plain
def send_email(to: str, subject: str) -> str:
    return f'Sent to {to}'

hooks = Hooks()

# Hook fires only when 'send_email' is called
@hooks.on.before_tool_execute(tools=['send_email'])
def audit_email(ctx, *, validated_args):
    print(f'AUDIT: send_email called with {validated_args}')
    return validated_args

# Hook fires for all tools
@hooks.on.after_tool_execute
def log_tool_result(ctx, *, validated_args, result):
    print(f'Tool {ctx.tool_name} -> {result!r}')
    return result

agent_with_hooks = Agent('openai:gpt-4o', capabilities=[hooks])
```

### `wrap_run` — full run middleware

```python
import asyncio
import logging
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks

log = logging.getLogger(__name__)
hooks = Hooks()

@hooks.on.wrap_run
async def add_error_logging(ctx, *, handler):
    try:
        result = await handler()
        log.info('Run completed: run_id=%s usage=%s', result.run_id, result.usage)
        return result
    except Exception as e:
        log.error('Run failed: run_id=%s error=%s', ctx.run_id, e)
        raise

agent = Agent('openai:gpt-4o', capabilities=[hooks])
```

### Hook timeouts

Register a hook with a timeout to prevent slow callbacks from blocking the run:

```python
from pydantic_ai.capabilities import Hooks

hooks = Hooks()

@hooks.on.before_model_request(timeout=2.0)
async def slow_lookup(ctx, request_context):
    # If this takes > 2s, HookTimeoutError is raised
    await some_remote_cache_lookup(ctx)
    return request_context
```

---

## 16. `Thinking` capability — cross-provider reasoning control

Source: `pydantic_ai/capabilities/thinking.py` — `Thinking`

`Thinking` is a one-field dataclass that enables and configures model reasoning/thinking via the unified `ModelSettings.thinking` field. It's a thin wrapper that's easier to pass around than a raw `model_settings` dict.

### Constructor

```python
from pydantic_ai.capabilities import Thinking

# Default: enable with provider's default effort
thinking_on  = Thinking()             # effort=True
thinking_off = Thinking(effort=False) # disable
thinking_low  = Thinking(effort='low')
thinking_high = Thinking(effort='high')
thinking_max  = Thinking(effort='xhigh')  # 'xhigh' maps to 'high' on providers without it
```

### Attaching to an agent

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

# Enable medium-effort thinking for every run
agent = Agent(
    'anthropic:claude-opus-4-7',
    capabilities=[Thinking(effort='medium')],
)
result = agent.run_sync('Is P=NP? Show your reasoning.')
print(result.output)
```

### Per-run override

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

agent = Agent('openai:o3')

# Low effort for fast responses
result_fast = agent.run_sync('What is 2+2?', capabilities=[Thinking(effort='low')])

# Max effort for hard problems
result_deep = agent.run_sync(
    'Prove the Riemann hypothesis.',
    capabilities=[Thinking(effort='xhigh')],
)
```

### Provider mapping

| `effort` | Anthropic | OpenAI | Google Gemini |
|---------|-----------|--------|---------------|
| `True` | default budget | default effort | default |
| `False` | thinking off | thinking off | thinking off |
| `'minimal'` | 1 024 budget tokens | `'low'` (no minimal) | minimal |
| `'low'` | 2 048 budget tokens | `'low'` | low |
| `'medium'` | 8 192 budget tokens | `'medium'` | medium |
| `'high'` | 16 384 budget tokens | `'high'` | high |
| `'xhigh'` | max budget | `'high'` (no xhigh) | highest |

### Combining with other capabilities

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking, Hooks

hooks = Hooks()

@hooks.on.after_run
def log_thinking_tokens(ctx, *, result):
    print(f'Output tokens: {result.usage.output_tokens}')
    return result

agent = Agent(
    'anthropic:claude-opus-4-7',
    capabilities=[Thinking(effort='high'), hooks],
)
```

---

## 17. `WebSearch` capability — native + local fallback

Source: `pydantic_ai/capabilities/web_search.py` — `WebSearch[AgentDepsT]`

`WebSearch` uses the model's native web search when available, falling back to a local DuckDuckGo tool when it isn't. This makes it portable across providers without changing agent code.

### Constructor

```python
from pydantic_ai.capabilities import WebSearch

ws = WebSearch(
    native=True,                       # Use provider's native search (default)
    local=True,                        # Fallback to DuckDuckGo if native not supported
    search_context_size='medium',      # 'low'|'medium'|'high' — how much context to retrieve
    user_location=None,                # WebSearchUserLocation for localised results
    blocked_domains=['spam.com'],      # Exclude specific domains
    allowed_domains=['docs.python.org'],  # Restrict to specific domains
    max_uses=5,                        # Limit searches per run
)
```

### Basic usage — cross-provider web search

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[WebSearch()],
    system_prompt='You are a research assistant. Always cite your sources.',
)

async def main():
    result = await agent.run('What are the most significant AI releases in the past month?')
    print(result.output)

asyncio.run(main())
```

### Domain-scoped search

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

docs_agent = Agent(
    'openai:gpt-4o',
    capabilities=[WebSearch(allowed_domains=['docs.pydantic.dev', 'docs.python.org'])],
    system_prompt='Answer only using official Python and Pydantic documentation.',
)

result = docs_agent.run_sync('How do I use Pydantic field validators?')
print(result.output)
```

### Localised results

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.native_tools import WebSearchUserLocation

uk_agent = Agent(
    'openai:gpt-4o',
    capabilities=[WebSearch(
        user_location=WebSearchUserLocation(country='GB', city='London'),
        search_context_size='high',
    )],
)

result = uk_agent.run_sync('What are the current UK interest rates?')
print(result.output)
```

### Explicit local-only mode

Force DuckDuckGo (no native fallback) by passing `native=False`. Requires `pip install "pydantic-ai-slim[duckduckgo]"`:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

# Always use DuckDuckGo regardless of model capabilities
agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[WebSearch(native=False, local='duckduckgo')],
)

result = agent.run_sync('Search for recent Python releases.')
print(result.output)
```

### Disable fallback

If the model doesn't support native search and you don't want a local fallback, pass `local=False`. The agent will raise a `UserError` at run time if the model can't handle web search:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

# Native only — fail fast if model doesn't support it
agent = Agent(
    'openai:gpt-4o',
    capabilities=[WebSearch(native=True, local=False)],
)
```

---

## Revision history

| Date | Version | Notes |
|------|---------|-------|
| 2026-05-22 | 1.101.0 | Added `Tool` advanced params (§13), `ModelSettings` complete reference (§14), `Hooks` capability (§15), `Thinking` capability (§16), `WebSearch` capability (§17); bumped verified version to 1.101.0. |
| 2026-05-20 | 1.99.0 | Added `UsageLimits` (§11) and `ConcurrencyLimiter` (§12) deep-dives; bumped verified version to 1.99.0. |
| 2026-05-19 | 1.98.0 | Initial deep-dives guide — 10 classes sourced from installed pydantic-ai 1.98.0. All imports verified. |

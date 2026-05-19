---
title: "PydanticAI: Class Deep Dives"
description: "Source-verified deep dives into AgentRun, AgentRunResult, ConcurrencyLimiter, the Direct API, capture_run_messages, format_as_xml, common_tools, ExternalToolset, FilteredToolset, and FunctionToolset with instructions."
framework: pydanticai
language: python
---

# PydanticAI: Class Deep Dives

Verified against **pydantic-ai==1.98.0** — each section links to the source module inspected.

This guide covers ten classes/utilities from the installed source that get light treatment elsewhere. Every example is derived directly from the `__init__`, docstrings, and behaviour seen in the installed package.

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

    # Usage (accessed as a property, not a method call)
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
        run_kwargs={'metadata': {'user_id': 42, 'source': 'web'}},
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
            if hasattr(part, 'content'):
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

    # 5. Resume — feed results back with the full message history
    result2 = await agent.run(
        '',
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
            '',
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

## Revision history

| Date | Version | Notes |
|------|---------|-------|
| 2026-05-19 | 1.98.0 | Initial deep-dives guide — 10 classes sourced from installed pydantic-ai 1.98.0. All imports verified. |

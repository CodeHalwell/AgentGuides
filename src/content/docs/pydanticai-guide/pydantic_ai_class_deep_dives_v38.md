---
title: "PydanticAI Class Deep Dives Vol. 38"
description: "Source-verified deep dives into 10 pydantic-ai 2.13.0 class groups: AgentRun/AgentRunResult/AgentRunResultEvent (fine-grained node iteration, enqueue, run/conversation IDs), FunctionModel/AgentInfo (test mock model — sync+async overloads, stream_function, ModelProfile defaults), FallbackModel/ResponseRejected (multi-model fallback — exception+response handler auto-detect, FallbackExceptionGroup, fallback_on sequence mixing), RetryConfig/TenacityTransport/AsyncTenacityTransport/wait_retry_after (tenacity HTTP retry — Retry-After header, validate_response, exponential backoff), FilteredToolset (per-context sync+async predicate — RBAC, feature-flag, per-user gating), DeferredLoadingToolset (lazy tool loading — defer_loading flag, selective by name set), ToolSearchToolset (large toolset discovery — keywords_search_fn, custom search_fn, max_results, native provider path), WrapperModel/CompletedStreamedResponse (model wrapping — logging, rate-limiting, durable-exec replay), WrapperCapability (capability wrapping — override specific lifecycle hooks, for_run factory, transparent id/defer_loading), MessagesBuilder/BuilderCheckpoint (message sequence construction — last_modified() attribution, UI streaming integration)."
sidebar:
  label: "Class deep dives (Vol. 38)"
  order: 64
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 2.13.0** source installed directly from PyPI. Every class signature, field name, and method in this volume reflects the 2.13.x API. Three examples per class group; all code blocks pass `ast.parse()` syntax validation. Live API calls are commented out — uncomment to run.
</Aside>

Ten class groups covering fine-grained agent run control (`AgentRun`/`AgentRunResult`), the test/mock `FunctionModel`, multi-model `FallbackModel`, tenacity HTTP retry utilities, dynamic toolset filtering and deferred loading, large-toolset discovery via `ToolSearchToolset`, `WrapperModel`/`CompletedStreamedResponse` for durable-execution wrappers, the `WrapperCapability` delegation pattern, and `MessagesBuilder`/`BuilderCheckpoint` for message sequence construction.

---

## 1. `AgentRun` + `AgentRunResult` + `AgentRunResultEvent`

**Source:** `pydantic_ai/run.py`

`AgentRun` is the stateful, async-iterable handle you get from `async with agent.iter(...)`. It wraps a `pydantic_graph.GraphRun` and exposes the node sequence, usage, run/conversation IDs, and the `enqueue()` method for injecting pending messages mid-run. `AgentRunResult` holds the final output and all message history; `AgentRunResultEvent` is the streaming event carrying the result.

**Key distinction:** bare `async for node in agent_run` does not fire capability hooks (`before_node_run`, `wrap_node_run`, `after_node_run`). Use `agent_run.next(node)` to drive hook-aware iteration.

```python
# Example 1 — Iterate over nodes and inspect each step
import asyncio
from pydantic_ai import Agent
from pydantic_graph import End

agent = Agent('openai:gpt-4o-mini', system_prompt='You are a concise assistant.')

async def inspect_run():
    nodes_seen = []
    async with agent.iter('What is 2 + 2?') as run:
        # Hook-aware iteration: fires before_node_run / after_node_run
        node = run.next_node
        while not isinstance(node, End):
            nodes_seen.append(type(node).__name__)
            node = await run.next(node)
        nodes_seen.append('End')

    result = run.result
    print('Nodes:', nodes_seen)
    print('Output:', result.output)
    print('Run ID:', result.run_id)
    print('Conversation ID:', result.conversation_id)
    print('Requests made:', result.usage.requests)

# asyncio.run(inspect_run())
```

```python
# Example 2 — enqueue() to inject a follow-up user message mid-run
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import UserPromptPart
from pydantic_graph import End

agent = Agent('openai:gpt-4o-mini')

async def enqueue_demo():
    async with agent.iter('Tell me a fact about cats.') as run:
        node = run.next_node
        step = 0
        while not isinstance(node, End):
            node = await run.next(node)
            step += 1
            if step == 1:
                # Inject a follow-up while the agent is still running
                enqueue_id = run.enqueue('Now tell me one about dogs.', priority='when_idle')
                print(f'Enqueued follow-up, id={enqueue_id}')

    print(run.result.output)
    print('Total token usage:', run.result.usage)

# asyncio.run(enqueue_demo())
```

```python
# Example 3 — AgentRunResult message helpers and output_tool_return_content
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse

agent = Agent('openai:gpt-4o-mini')

async def result_messages():
    # result = await agent.run('Summarise the Python docs in one sentence.')
    # Demonstrate the message helpers without a live call:
    from pydantic_ai.run import AgentRunResult
    from pydantic_ai._agent_graph import GraphAgentState

    state = GraphAgentState()
    dummy = AgentRunResult.__new__(AgentRunResult)
    dummy.output = 'Python is a versatile high-level language.'
    dummy._output_tool_name = None
    dummy._state = state
    dummy._new_message_index = 0
    dummy._traceparent_value = None

    # all_messages / new_messages return the same list when no history was pre-loaded
    all_msgs = dummy.all_messages()
    new_msgs = dummy.new_messages()
    print('all_messages count:', len(all_msgs))   # 0 in this dummy
    print('new_messages count:', len(new_msgs))   # 0 in this dummy

    # JSON serialisation
    json_bytes = dummy.all_messages_json()
    print('JSON bytes:', json_bytes[:50])

asyncio.run(result_messages())
```

---

## 2. `FunctionModel` + `AgentInfo`

**Source:** `pydantic_ai/models/function.py`

`FunctionModel` is the go-to model for unit testing and offline development. It accepts a synchronous or asynchronous `function: (messages, AgentInfo) -> ModelResponse` and/or a `stream_function: (messages, AgentInfo) -> AsyncGenerator[str | DeltaToolCall, None]`. The model derives its name from the function names unless `model_name=` is provided. It ships a sensible `ModelProfile` with `supports_json_schema_output=True` so structured output tools work without a real provider.

`AgentInfo` carries the agent's current tool definitions, whether text output is allowed, and the resolved model settings — everything a test function needs to decide what to return.

```python
# Example 1 — Minimal synchronous FunctionModel for unit tests
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.usage import RequestUsage


def echo_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    last_text = ''
    for msg in reversed(messages):
        from pydantic_ai.messages import ModelRequest, UserPromptPart
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    last_text = part.content if isinstance(part.content, str) else ''
                    break
        if last_text:
            break
    return ModelResponse(
        parts=[TextPart(content=f'Echo: {last_text}')],
        usage=RequestUsage(input_tokens=10, output_tokens=5),
    )


agent = Agent(FunctionModel(echo_function))
result = agent.run_sync('Hello from tests!')
print(result.output)  # Echo: Hello from tests!
print(result.usage.requests)  # 1
```

```python
# Example 2 — Async FunctionModel that inspects AgentInfo to call a tool
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.usage import RequestUsage
import json


async def tool_calling_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    # If the agent has a 'greet' tool, call it on the first request
    tool_names = [t.name for t in info.function_tools]
    if 'greet' in tool_names and len(messages) == 1:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='greet',
                    args=json.dumps({'name': 'World'}),
                    tool_call_id='call-001',
                )
            ],
            usage=RequestUsage(input_tokens=15, output_tokens=8),
        )
    # After tool result is returned, emit final text
    return ModelResponse(
        parts=[TextPart(content='Greeted successfully.')],
        usage=RequestUsage(input_tokens=20, output_tokens=4),
    )


agent = Agent(FunctionModel(tool_calling_function))


@agent.tool_plain
def greet(name: str) -> str:
    """Greet someone by name."""
    return f'Hello, {name}!'


result = asyncio.run(agent.run('Please greet the world.'))
print(result.output)   # Greeted successfully.
print(result.usage.requests)  # 2
```

```python
# Example 3 — Stream function for streamed output testing
import asyncio
from collections.abc import AsyncIterator
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage


async def streaming_words(
    messages: list[ModelMessage],
    info: AgentInfo,
) -> AsyncIterator[str]:
    words = ['Streaming', ' output', ' works', ' too', '!']
    for word in words:
        yield word


agent = Agent(FunctionModel(stream_function=streaming_words))


async def run_streamed():
    async with agent.run_stream('Stream me something.') as result:
        text = await result.get_text()
    print(text)  # Streaming output works too!

asyncio.run(run_streamed())
```

---

## 3. `FallbackModel` + `ResponseRejected`

**Source:** `pydantic_ai/models/fallback.py`

`FallbackModel` wraps two or more models and, upon a triggering condition, rewinds the message history and retries with the next model. The `fallback_on` parameter accepts exception types, sync/async exception handlers, sync/async response handlers (auto-detected by inspecting the first parameter type hint as `ModelResponse`), or a sequence mixing all three. All triggering exceptions are collected into a `FallbackExceptionGroup` raised when every model fails. `ResponseRejected` is raised inside that group when a response handler returns `True`.

```python
# Example 1 — Fallback on ModelAPIError (the default)
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.exceptions import ModelAPIError

# Primary: fast cheap model; fallback: more capable one
model = FallbackModel(
    'openai:gpt-4o-mini',       # tried first
    'openai:gpt-4o',            # used if gpt-4o-mini raises ModelAPIError
)

agent = Agent(model, system_prompt='You are a helpful assistant.')
# result = agent.run_sync('Explain recursion.')
# print(result.output)
```

```python
# Example 2 — Custom exception handler + response rejection handler
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelResponse


def is_rate_limited(exc: Exception) -> bool:
    """Trigger fallback on rate-limit HTTP errors."""
    if isinstance(exc, ModelAPIError):
        msg = str(exc).lower()
        return 'rate limit' in msg or '429' in msg
    return False


def response_too_short(response: ModelResponse) -> bool:
    """Reject suspiciously short responses and fall back to a stronger model."""
    from pydantic_ai.messages import TextPart
    text = ''.join(
        part.content for part in response.parts if isinstance(part, TextPart)
    )
    return len(text.strip()) < 20


model = FallbackModel(
    'openai:gpt-4o-mini',
    'anthropic:claude-haiku-4-5',
    fallback_on=[is_rate_limited, response_too_short],
)

agent = Agent(model)
# result = agent.run_sync('Write a haiku about Python.')
# print(result.output)
```

```python
# Example 3 — Catching FallbackExceptionGroup when all models fail
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.exceptions import FallbackExceptionGroup
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.usage import RequestUsage


def always_fails(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise ModelAPIError('primary', 'Simulated provider failure')


model = FallbackModel(
    FunctionModel(always_fails, model_name='primary'),
    FunctionModel(always_fails, model_name='fallback'),
    fallback_on=(ModelAPIError,),
)

agent = Agent(model)


async def demonstrate_exhausted_fallback():
    try:
        await agent.run('Hello')
    except FallbackExceptionGroup as eg:
        print(f'All {len(eg.exceptions)} models failed:')
        for exc in eg.exceptions:
            print(f'  • {exc}')

asyncio.run(demonstrate_exhausted_fallback())
```

---

## 4. `RetryConfig` + `TenacityTransport` + `AsyncTenacityTransport` + `wait_retry_after`

**Source:** `pydantic_ai/retries.py`

The retries module wraps `httpx` transports with tenacity retry logic. `RetryConfig` is a `TypedDict` whose keys map directly to `tenacity.retry` decorator arguments (`stop`, `wait`, `retry`, `before`, `after`, `before_sleep`, `reraise`, etc.). `TenacityTransport` and `AsyncTenacityTransport` inject `@retry(**config)` around `handle_request` / `handle_async_request`. `wait_retry_after` is a wait-strategy factory that reads the `Retry-After` HTTP header (both integer-seconds and date-string formats) and clamps to `max_wait`.

Install the optional group: `pip install "pydantic-ai-slim[retries]"`.

```python
# Example 1 — Synchronous TenacityTransport for a provider using httpx
from httpx import Client, HTTPStatusError, HTTPTransport
from tenacity import retry_if_exception_type, stop_after_attempt

from pydantic_ai.retries import RetryConfig, TenacityTransport, wait_retry_after

config = RetryConfig(
    retry=retry_if_exception_type(HTTPStatusError),
    wait=wait_retry_after(max_wait=120),   # respect Retry-After header, cap at 2 min
    stop=stop_after_attempt(4),
    reraise=True,                          # re-raise the last exception instead of RetryError
)

transport = TenacityTransport(
    config,
    HTTPTransport(),
    validate_response=lambda r: r.raise_for_status(),
)

client = Client(transport=transport)
# response = client.get('https://api.example.com/data')
# print(response.json())
client.close()
```

```python
# Example 2 — AsyncTenacityTransport for async httpx usage
from httpx import AsyncClient, HTTPStatusError
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after

config = RetryConfig(
    retry=retry_if_exception_type(HTTPStatusError),
    wait=wait_retry_after(
        fallback_strategy=wait_exponential(multiplier=1, max=60),
        max_wait=300,
    ),
    stop=stop_after_attempt(5),
    reraise=True,
)

transport = AsyncTenacityTransport(
    config,
    validate_response=lambda r: r.raise_for_status(),
)

client = AsyncClient(transport=transport)

import asyncio

async def fetch():
    # response = await client.get('https://api.example.com/items')
    # return response.json()
    return {}

# result = asyncio.run(fetch())
# asyncio.run(client.aclose())
```

```python
# Example 3 — Custom before_sleep logging with RetryConfig
import logging
from httpx import Client, HTTPStatusError, HTTPTransport
from tenacity import RetryCallState, retry_if_exception_type, stop_after_attempt

from pydantic_ai.retries import RetryConfig, TenacityTransport, wait_retry_after

logger = logging.getLogger(__name__)


def log_retry(state: RetryCallState) -> None:
    attempt = state.attempt_number
    exc = state.outcome.exception() if state.outcome else None
    logger.warning('Retry attempt %d due to: %s', attempt, exc)


config = RetryConfig(
    retry=retry_if_exception_type(HTTPStatusError),
    wait=wait_retry_after(max_wait=60),
    stop=stop_after_attempt(3),
    before_sleep=log_retry,
    reraise=True,
)

transport = TenacityTransport(config, HTTPTransport(), validate_response=lambda r: r.raise_for_status())

with transport as t:
    # Use as a context manager — delegates __enter__/__exit__ to wrapped transport
    client = Client(transport=t)
    # client.get('https://api.example.com/endpoint')
    client.close()
```

---

## 5. `FilteredToolset`

**Source:** `pydantic_ai/toolsets/filtered.py`

`FilteredToolset` wraps any `AbstractToolset` and applies a `filter_func(RunContext, ToolDefinition) -> bool | Awaitable[bool]` at every `get_tools()` call. Because the predicate receives the `RunContext`, it can read `ctx.deps`, the current user, request metadata, feature flags — anything available at run time. Both synchronous and asynchronous predicates are accepted.

```python
# Example 1 — Role-based access control: hide admin tools for non-admin users
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.filtered import FilteredToolset


@dataclass
class UserDeps:
    username: str
    is_admin: bool


ts = FunctionToolset[UserDeps]()


@ts.tool
def list_users(ctx: RunContext[UserDeps]) -> list[str]:
    """List all registered users. Admin only."""
    return ['alice', 'bob', 'carol']


@ts.tool
def get_profile(ctx: RunContext[UserDeps], username: str) -> str:
    """Get a user's public profile."""
    return f'Profile for {username}'


def admin_only_filter(ctx: RunContext[UserDeps], tool_def: ToolDefinition) -> bool:
    if tool_def.name == 'list_users':
        return ctx.deps.is_admin
    return True


filtered = FilteredToolset(ts, filter_func=admin_only_filter)
agent = Agent('openai:gpt-4o-mini', toolsets=[filtered])

# Admin can see list_users; regular users cannot
admin_deps = UserDeps(username='alice', is_admin=True)
user_deps  = UserDeps(username='bob',   is_admin=False)
# admin_result = agent.run_sync('List all users.', deps=admin_deps)
# user_result  = agent.run_sync('List all users.', deps=user_deps)
```

```python
# Example 2 — Async filter reading a feature-flag store
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.filtered import FilteredToolset

ENABLED_TOOLS: set[str] = {'search', 'summarise'}   # pretend feature-flag store


@dataclass
class SessionDeps:
    session_id: str


ts = FunctionToolset[SessionDeps]()


@ts.tool_plain
def search(query: str) -> str:
    """Search the knowledge base."""
    return f'Results for: {query}'


@ts.tool_plain
def experimental_rerank(query: str, results: list[str]) -> list[str]:
    """Re-rank results using experimental model (feature-flagged)."""
    return sorted(results)


async def feature_flag_filter(ctx: RunContext[SessionDeps], tool_def: ToolDefinition) -> bool:
    # In production this would query a remote feature-flag service
    return tool_def.name in ENABLED_TOOLS


filtered = FilteredToolset(ts, filter_func=feature_flag_filter)
agent = Agent('openai:gpt-4o-mini', toolsets=[filtered])

# asyncio.run(agent.run('Search for Python tutorials.', deps=SessionDeps('s-001')))
```

```python
# Example 3 — Combining FilteredToolset with FallbackModel for per-model tool exposure
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.filtered import FilteredToolset
from pydantic_ai.models.fallback import FallbackModel


@dataclass
class Deps:
    premium: bool


ts = FunctionToolset[Deps]()


@ts.tool_plain
def basic_search(query: str) -> str:
    """Free-tier web search."""
    return f'Basic result for: {query}'


@ts.tool_plain
def deep_research(query: str, depth: int = 3) -> str:
    """Premium multi-hop research tool."""
    return f'Deep result (depth={depth}) for: {query}'


def premium_filter(ctx: RunContext[Deps], tool_def: ToolDefinition) -> bool:
    if tool_def.name == 'deep_research':
        return ctx.deps.premium
    return True


model = FallbackModel('openai:gpt-4o-mini', 'openai:gpt-4o')
agent = Agent(model, toolsets=[FilteredToolset(ts, filter_func=premium_filter)])

# free_result    = agent.run_sync('Research quantum computing.', deps=Deps(premium=False))
# premium_result = agent.run_sync('Research quantum computing.', deps=Deps(premium=True))
```

---

## 6. `DeferredLoadingToolset`

**Source:** `pydantic_ai/toolsets/deferred_loading.py`

`DeferredLoadingToolset` marks tools with `defer_loading=True` on their `ToolDefinition`, hiding them from the model until discovered via a `ToolSearchToolset`. This is the mechanism behind tool-search: wrap a large toolset with `DeferredLoadingToolset` so the model only sees a `search_tools` function, then wrap that with `ToolSearchToolset` to handle discovery. Pass `tool_names=frozenset({'name'})` to defer only selected tools; `tool_names=None` (default) defers all.

```python
# Example 1 — Defer all tools in a large toolset
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.deferred_loading import DeferredLoadingToolset
from pydantic_ai.toolsets._tool_search import ToolSearchToolset


large_ts = FunctionToolset()


@large_ts.tool_plain
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f'Sunny, 22°C in {city}'


@large_ts.tool_plain
def get_stock_price(ticker: str) -> str:
    """Get the latest stock price for a ticker symbol."""
    return f'{ticker}: $150.00'


@large_ts.tool_plain
def translate_text(text: str, target_lang: str) -> str:
    """Translate text into the target language."""
    return f'[{target_lang}] {text}'


# Step 1: defer all tools so they're hidden from the model
deferred = DeferredLoadingToolset(large_ts)

# Step 2: wrap with ToolSearchToolset so the model can discover them
searchable = ToolSearchToolset(deferred)

agent = Agent('openai:gpt-4o-mini', toolsets=[searchable])
# The model first sees only 'search_tools'; after calling it with relevant keywords,
# the matching tools become available.
# result = agent.run_sync("What's the weather in Tokyo?")
```

```python
# Example 2 — Selectively defer only expensive/rare tools
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.deferred_loading import DeferredLoadingToolset
from pydantic_ai.toolsets._tool_search import ToolSearchToolset


ts = FunctionToolset()


@ts.tool_plain
def quick_lookup(key: str) -> str:
    """Fast in-memory lookup — always available."""
    return f'value:{key}'


@ts.tool_plain
def run_sql_query(query: str) -> list[dict]:
    """Execute a SQL query against the data warehouse — expensive, deferred."""
    return [{'result': 'row1'}, {'result': 'row2'}]


@ts.tool_plain
def call_external_api(endpoint: str) -> dict:
    """Call an external API — network cost, deferred."""
    return {'status': 'ok'}


# Defer only the two expensive tools
expensive = frozenset({'run_sql_query', 'call_external_api'})
deferred = DeferredLoadingToolset(ts, tool_names=expensive)
searchable = ToolSearchToolset(deferred)

agent = Agent('openai:gpt-4o-mini', toolsets=[searchable])
# quick_lookup is immediately visible; the others require search first.
```

```python
# Example 3 — Inspecting the defer_loading flag on ToolDefinition
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.deferred_loading import DeferredLoadingToolset
from pydantic_ai._run_context import RunContext


ts = FunctionToolset()


@ts.tool_plain
def alpha() -> str:
    """Alpha tool."""
    return 'alpha'


@ts.tool_plain
def beta() -> str:
    """Beta tool."""
    return 'beta'


deferred = DeferredLoadingToolset(ts, tool_names=frozenset({'beta'}))
agent = Agent('openai:gpt-4o-mini', toolsets=[deferred])


async def inspect_tool_defs():
    from pydantic_ai._run_context import RunContext
    from dataclasses import dataclass

    # Use get_tools on the wrapped toolset directly to inspect flags
    # (In production the agent drives this transparently)
    print('DeferredLoadingToolset wraps:', type(deferred.wrapped).__name__)
    print('Deferred tool names:', deferred.tool_names)

asyncio.run(inspect_tool_defs())
```

---

## 7. `ToolSearchToolset`

**Source:** `pydantic_ai/toolsets/_tool_search.py`

`ToolSearchToolset` manages large toolsets by exposing a `search_tools` function to the model instead of all tools at once. Tools marked `defer_loading=True` are hidden until discovered via search. The default algorithm (`keywords_search_fn`) tokenises queries and tool names/descriptions on alphanumeric runs and ranks by overlap score. Supply a custom `search_fn` for semantic/embedding-based search. When the provider supports a native tool-search builtin (Anthropic, OpenAI Responses), the local `search_tools` function is dropped from the wire and discovery is handled server-side.

```python
# Example 1 — Default keyword search with custom descriptions
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.deferred_loading import DeferredLoadingToolset
from pydantic_ai.toolsets._tool_search import ToolSearchToolset


ts = FunctionToolset()


@ts.tool_plain
def fetch_invoice(invoice_id: str) -> dict:
    """Fetch invoice details by ID from the billing system."""
    return {'invoice_id': invoice_id, 'amount': 99.0, 'status': 'paid'}


@ts.tool_plain
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient via the mail service."""
    return f'Email sent to {to}'


@ts.tool_plain
def create_ticket(title: str, description: str, priority: str = 'medium') -> str:
    """Create a support ticket in the helpdesk system."""
    return f'Ticket created: {title}'


deferred = DeferredLoadingToolset(ts)

searchable = ToolSearchToolset(
    deferred,
    max_results=5,
    tool_description=(
        'Search for available tools by keyword. '
        'Use specific words from the task domain (e.g. "invoice", "email", "ticket").'
    ),
)

agent = Agent('openai:gpt-4o-mini', toolsets=[searchable])
# result = agent.run_sync('Please fetch invoice INV-0042.')
```

```python
# Example 2 — Custom async search function (semantic / embedding-based)
import asyncio
from collections.abc import Sequence
from pydantic_ai import Agent
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.deferred_loading import DeferredLoadingToolset
from pydantic_ai.toolsets._tool_search import ToolSearchToolset


# Pretend embedding index — in production use a real vector store
_TOOL_VECTORS: dict[str, list[float]] = {
    'fetch_invoice': [0.9, 0.1, 0.0],
    'send_email':    [0.1, 0.9, 0.0],
    'create_ticket': [0.1, 0.1, 0.9],
}

_QUERY_VECTORS: dict[str, list[float]] = {
    'billing': [0.85, 0.05, 0.1],
    'mail':    [0.05, 0.9, 0.05],
    'support': [0.05, 0.05, 0.9],
}


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x**2 for x in a) ** 0.5
    mag_b = sum(x**2 for x in b) ** 0.5
    return dot / (mag_a * mag_b + 1e-9)


async def embedding_search(
    ctx: RunContext,
    queries: Sequence[str],
    tools: Sequence[ToolDefinition],
) -> list[str]:
    tool_names = [t.name for t in tools]
    scores: dict[str, float] = {n: 0.0 for n in tool_names}
    for query in queries:
        qvec = _QUERY_VECTORS.get(query.lower(), [0.33, 0.33, 0.34])
        for name in tool_names:
            tvec = _TOOL_VECTORS.get(name, [0.33, 0.33, 0.34])
            scores[name] += cosine_sim(qvec, tvec)
    return sorted(scores, key=scores.__getitem__, reverse=True)


ts = FunctionToolset()


@ts.tool_plain
def fetch_invoice(invoice_id: str) -> dict:
    """Fetch invoice details."""
    return {'invoice_id': invoice_id}


@ts.tool_plain
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f'Sent to {to}'


@ts.tool_plain
def create_ticket(title: str, description: str) -> str:
    """Create a support ticket."""
    return f'Created: {title}'


searchable = ToolSearchToolset(
    DeferredLoadingToolset(ts),
    search_fn=embedding_search,
    max_results=2,
)

agent = Agent('openai:gpt-4o-mini', toolsets=[searchable])
# asyncio.run(agent.run('I need to open a support ticket.'))
```

```python
# Example 3 — keywords_search_fn used directly for offline testing
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets._tool_search import keywords_search_fn


tool_defs = [
    ToolDefinition(
        name='send_email',
        description='Send an email to a recipient',
        parameters_json_schema={'type': 'object', 'properties': {}},
    ),
    ToolDefinition(
        name='fetch_invoice',
        description='Retrieve invoice details from billing',
        parameters_json_schema={'type': 'object', 'properties': {}},
    ),
    ToolDefinition(
        name='create_ticket',
        description='Open a new support ticket in the helpdesk',
        parameters_json_schema={'type': 'object', 'properties': {}},
    ),
]

# Simulate what ToolSearchToolset does internally
results = keywords_search_fn(None, ['billing invoice'], tool_defs)  # type: ignore[arg-type]
print(results)  # ['fetch_invoice'] — highest overlap score
```

---

## 8. `WrapperModel` + `CompletedStreamedResponse`

**Source:** `pydantic_ai/models/wrapper.py`

`WrapperModel` is the delegate base class for custom model wrappers. Override any subset of `request`, `request_stream`, `count_tokens`, `prepare_messages`, etc.; all unoverridden methods forward to `self.wrapped`. `__getattr__` proxies any attribute not explicitly defined, so provider-specific attributes are still reachable. `CompletedStreamedResponse` wraps a `ModelResponse` that was already consumed elsewhere (e.g. inside a Temporal activity or Prefect task) and presents it as a `StreamedResponse` — optionally replaying events for consumers that drive the stream via event iteration.

```python
# Example 1 — Logging wrapper that records every request/response pair
import asyncio
import logging
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import TextPart
from pydantic_ai.usage import RequestUsage

logger = logging.getLogger(__name__)


class LoggingModel(WrapperModel):
    """Wraps any model and logs every request/response pair."""

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        logger.info('→ model request: %d messages', len(messages))
        response = await self.wrapped.request(messages, model_settings, model_request_parameters)
        from pydantic_ai.messages import TextPart
        text = ''.join(p.content for p in response.parts if isinstance(p, TextPart))
        logger.info('← model response (%d chars)', len(text))
        return response


def simple_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content='Logged response.')],
        usage=RequestUsage(input_tokens=5, output_tokens=3),
    )


agent = Agent(LoggingModel(FunctionModel(simple_fn)))
result = agent.run_sync('Hello!')
print(result.output)  # Logged response.
```

```python
# Example 2 — Rate-limiting wrapper with a semaphore
import asyncio
import anyio
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import TextPart
from pydantic_ai.usage import RequestUsage


class RateLimitedModel(WrapperModel):
    """Limits concurrent requests to `max_concurrent`."""

    def __init__(self, wrapped, max_concurrent: int = 2):
        super().__init__(wrapped)
        self._max_concurrent = max_concurrent
        self._semaphore: anyio.Semaphore | None = None

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        if self._semaphore is None:
            self._semaphore = anyio.Semaphore(self._max_concurrent)
        async with self._semaphore:
            return await self.wrapped.request(messages, model_settings, model_request_parameters)


def mock_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content='ok')],
        usage=RequestUsage(input_tokens=3, output_tokens=1),
    )


model = RateLimitedModel(FunctionModel(mock_fn), max_concurrent=3)
agent = Agent(model)


async def burst():
    tasks = [agent.run('task') for _ in range(5)]
    results = await asyncio.gather(*tasks)
    print([r.output for r in results])

asyncio.run(burst())
```

```python
# Example 3 — CompletedStreamedResponse for durable-execution replay
from pydantic_ai.models.wrapper import CompletedStreamedResponse
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.usage import RequestUsage
from pydantic_ai.models import ModelRequestParameters

# Simulate a response that was consumed inside a Temporal activity
stored_response = ModelResponse(
    parts=[TextPart(content='Result from completed activity.')],
    usage=RequestUsage(input_tokens=20, output_tokens=5),
    model_name='openai:gpt-4o-mini',
)

mrp = ModelRequestParameters(
    function_tools=[],
    allow_text_output=True,
    output_tools=[],
)

# replay_events=True replays PartStartEvents so streaming consumers see content
streamed = CompletedStreamedResponse(mrp, stored_response, replay_events=True)

import asyncio

async def consume():
    response = streamed.get()
    text = ''.join(p.content for p in response.parts if isinstance(p, TextPart))
    print('Replayed:', text)  # Replayed: Result from completed activity.
    print('Usage:', response.usage)

asyncio.run(consume())
```

---

## 9. `WrapperCapability`

**Source:** `pydantic_ai/capabilities/wrapper.py`

`WrapperCapability` is the capability counterpart to `WrapperToolset` and `WrapperModel`. Subclass it, set `wrapped` to any `AbstractCapability`, then override only the lifecycle hooks you care about. All unoverridden methods delegate to `self.wrapped`. The `__post_init__` automatically copies the wrapped capability's `id` and `defer_loading` flags so the wrapper is transparent to the capability registry. `for_run` produces a new wrapper with the post-`for_run` wrapped instance, keeping the delegation chain live across run setup.

```python
# Example 1 — Audit-log capability wrapping existing hooks
import asyncio
import logging
from dataclasses import dataclass
from typing import Any
from pydantic_ai import Agent
from pydantic_ai.capabilities.wrapper import WrapperCapability
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.messages import ModelResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class AuditCapability(WrapperCapability):
    """Wraps any capability and logs before/after every tool execution."""

    async def before_tool_execute(
        self,
        ctx: RunContext,
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: Any,
    ) -> Any:
        logger.info('TOOL START  %s args=%s', call.tool_name, args)
        return await self.wrapped.before_tool_execute(ctx, call=call, tool_def=tool_def, args=args)

    async def after_tool_execute(
        self,
        ctx: RunContext,
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: Any,
        result: Any,
    ) -> Any:
        logger.info('TOOL FINISH %s result=%r', call.tool_name, result)
        return await self.wrapped.after_tool_execute(ctx, call=call, tool_def=tool_def, args=args, result=result)


from pydantic_ai.capabilities.hooks import Hooks

base_hooks = Hooks()
audit_cap = AuditCapability(wrapped=base_hooks)

agent = Agent('openai:gpt-4o-mini', capabilities=[audit_cap])


@agent.tool_plain
def greet(name: str) -> str:
    """Greet someone."""
    return f'Hello, {name}!'

# asyncio.run(agent.run('Greet Alice.'))
```

```python
# Example 2 — Timing capability measuring model request latency
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from pydantic_ai import Agent
from pydantic_ai.capabilities.wrapper import WrapperCapability
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.tools import RunContext
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.messages import ModelResponse


@dataclass
class TimingCapability(WrapperCapability):
    """Records latency of every model request."""

    latencies: list[float] = field(default_factory=list)

    async def before_model_request(
        self,
        ctx: RunContext,
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        ctx.deps  # keep deps alive
        request_context = await self.wrapped.before_model_request(ctx, request_context)
        # Store start time on the request context metadata (arbitrary dict)
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext,
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        # In production: read start time from shared state; here we approximate
        self.latencies.append(time.monotonic())
        return await self.wrapped.after_model_request(ctx, request_context=request_context, response=response)


timing = TimingCapability(wrapped=Hooks())
agent = Agent('openai:gpt-4o-mini', capabilities=[timing])
# asyncio.run(agent.run('Describe the sky.'))
# print('Request timestamps:', timing.latencies)
```

```python
# Example 3 — Chaining two WrapperCapabilities
from dataclasses import dataclass
from typing import Any
from pydantic_ai.capabilities.wrapper import WrapperCapability
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.messages import ToolCallPart


@dataclass
class PrefixArgsCapability(WrapperCapability):
    """Prepends a tag to the first string argument of every tool call."""

    prefix: str = '[AUDITED] '

    async def before_tool_execute(
        self,
        ctx: RunContext,
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: Any,
    ) -> Any:
        if isinstance(args, dict):
            first_key = next(iter(args), None)
            if first_key and isinstance(args[first_key], str):
                args = {**args, first_key: self.prefix + args[first_key]}
        return await self.wrapped.before_tool_execute(ctx, call=call, tool_def=tool_def, args=args)


# Chain: PrefixArgs wraps an Audit that wraps base Hooks
base = Hooks()
audit = WrapperCapability(wrapped=base)       # transparent pass-through
prefix_cap = PrefixArgsCapability(wrapped=audit, prefix='[CHECKED] ')

# Verify delegation chain depth
print(type(prefix_cap.wrapped).__name__)   # WrapperCapability
print(type(prefix_cap.wrapped.wrapped).__name__)  # Hooks
```

---

## 10. `MessagesBuilder` + `BuilderCheckpoint`

**Source:** `pydantic_ai/ui/_messages_builder.py`

`MessagesBuilder` constructs `ModelRequest`/`ModelResponse` message sequences from individual `ModelRequestPart`/`ModelResponsePart` objects. Each call to `add()` either extends the last message's `parts` list (if the new part matches the tail's type) or appends a fresh message. `BuilderCheckpoint` is an opaque snapshot of builder state used with `last_modified()` to find which message was created or extended since the snapshot. Used internally by `UIEventStream` and custom streaming adapters.

```python
# Example 1 — Building a conversation turn from individual parts
from pydantic_ai.ui._messages_builder import MessagesBuilder
from pydantic_ai.messages import (
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ModelRequest,
    ModelResponse,
)

builder = MessagesBuilder()

# Build request
builder.add(SystemPromptPart(content='You are a helpful assistant.'))
builder.add(UserPromptPart(content='What is the capital of France?'))

# Build response
builder.add(TextPart(content='The capital of France is Paris.'))

print('Total messages:', len(builder.messages))  # 2

request = builder.messages[0]
response = builder.messages[1]

assert isinstance(request, ModelRequest)
assert len(request.parts) == 2   # SystemPromptPart + UserPromptPart

assert isinstance(response, ModelResponse)
assert len(response.parts) == 1  # TextPart

from pydantic_ai.messages import TextPart as TP
print('Response text:', response.parts[0].content)  # The capital of France is Paris.
```

```python
# Example 2 — Using BuilderCheckpoint to track which message was modified
from pydantic_ai.ui._messages_builder import MessagesBuilder, BuilderCheckpoint
from pydantic_ai.messages import (
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    ModelRequest,
    ModelResponse,
)

builder = MessagesBuilder()

# First turn
builder.add(UserPromptPart(content='What tools do you have?'))
checkpoint = builder.checkpoint()

# Add model response after checkpoint
builder.add(TextPart(content='I have search and calculator tools.'))

# Find which ModelResponse was added/extended since checkpoint
modified = builder.last_modified(checkpoint, of_type=ModelResponse)
print('Modified response:', modified is not None)          # True
print('Parts since checkpoint:', len(modified.parts))     # 1

# Second checkpoint — then add a tool call
checkpoint2 = builder.checkpoint()
builder.add(ToolCallPart(tool_name='search', args='{"query": "Paris"}', tool_call_id='c1'))

modified_response2 = builder.last_modified(checkpoint2, of_type=ModelResponse)
print('Modified response2 extended:', modified_response2 is not None)  # True
```

```python
# Example 3 — Custom streaming adapter using MessagesBuilder
from pydantic_ai.ui._messages_builder import MessagesBuilder
from pydantic_ai.messages import (
    UserPromptPart,
    TextPart,
    ModelRequest,
    ModelResponse,
)


class SimpleStreamAdapter:
    """Minimal adapter: receives parts from a stream, assembles into messages."""

    def __init__(self):
        self._builder = MessagesBuilder()

    def receive_part(self, part) -> None:
        self._builder.add(part)

    def get_messages(self) -> list:
        return self._builder.messages

    def get_last_response_text(self) -> str:
        for msg in reversed(self._builder.messages):
            if isinstance(msg, ModelResponse):
                return ''.join(
                    p.content for p in msg.parts if isinstance(p, TextPart)
                )
        return ''


adapter = SimpleStreamAdapter()
adapter.receive_part(UserPromptPart(content='Hello!'))
adapter.receive_part(TextPart(content='Hi there, '))
adapter.receive_part(TextPart(content='how can I help?'))

print('Messages assembled:', len(adapter.get_messages()))    # 2
print('Response text:', adapter.get_last_response_text())   # Hi there, how can I help?
```

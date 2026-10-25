---
title: "PydanticAI: Comprehensive Guide"
description: "Single-page tutorial to PydanticAI V2 — installation, agents, RunContext, tools, structured output, dependency injection and upgrading from V1 — followed by a map of the public API and where each part is documented."
framework: "pydanticai"
language: "python"
sidebar:
  label: "Comprehensive guide"
  order: 1
---

Checked against **pydantic-ai %%version=pydantic-ai%%** (%%versions-checked%%). Requires Python 3.11+. The examples were run against that release with `TestModel` in place of the real model.

This page takes you from installation to a typed, tool-using agent, then maps the rest of the public API to the pages that cover it. Topic pages carry the depth: [output types](/pydanticai-guide/pydantic_ai_output_types/), [toolsets](/pydanticai-guide/pydantic_ai_toolsets/), [streaming](/pydanticai-guide/pydantic_ai_streaming/), [testing](/pydanticai-guide/pydantic_ai_testing/) and the others listed in the [API map](#api-map).

## Contents

1. [Core concepts](#core-concepts)
2. [Installation and setup](#installation-and-setup)
3. [Core fundamentals](#core-fundamentals)
4. [Simple agents](#simple-agents)
5. [Type safety and validation](#type-safety-and-validation)
6. [Structured output](#structured-output)
7. [Tools and function calling](#tools-and-function-calling)
8. [Dependency injection](#dependency-injection)
9. [Upgrading from V1](#upgrading-from-v1)
10. [Agent reference](#agent-reference)
11. [API map](#api-map)

## Core concepts

PydanticAI is the Pydantic team's agent framework. It aims for the feel of FastAPI: plain Python functions, type hints that drive validation and schemas, and dependency injection instead of globals.

- **Typed throughout.** `Agent[DepsT, OutputT]` is generic; tools, instructions and validators receive a typed `RunContext[DepsT]`, and outputs are validated by Pydantic.
- **Model-agnostic.** One interface over OpenAI, Anthropic, Google, Bedrock, Groq, Mistral, xAI, OpenRouter and many OpenAI-compatible providers, selected by a `provider:model` string.
- **Agents are reusable.** Create an agent once and run it many times; per-run data goes in `deps`, conversation state in `message_history`.
- **Composable.** Tools, toolsets and *capabilities* (hooks, instrumentation, native tools, durability) attach to an agent without subclassing it.
- **Testable and observable.** `TestModel` and `FunctionModel` replace the LLM in tests; OpenTelemetry instrumentation and Logfire integration are built in.

| Problem | What PydanticAI provides |
|---|---|
| Unpredictable LLM output | `output_type=` with Pydantic validation and automatic retries |
| Model lock-in | One `Agent` API across providers; `FallbackModel` across several |
| Tool plumbing | `@agent.tool` with schemas from type hints and docstrings |
| Passing state to tools | `deps_type=` and `RunContext` |
| Production visibility | OpenTelemetry spans and metrics, Logfire |
| Testing without API calls | `TestModel`, `FunctionModel`, `agent.override()` |

## Installation and setup

```bash
pip install pydantic-ai            # or: uv add pydantic-ai
pip install "pydantic-ai==%%version=pydantic-ai%%"   # pinned
```

Since 2.0 the standard package installs `pydantic-ai-slim[openai,anthropic,google,cli,mcp,evals,web,logfire]`: the OpenAI, Anthropic and Google providers plus the CLI, MCP, evals, web UI and Logfire integrations. Everything else is an extra, for example `pip install "pydantic-ai[bedrock,groq,mistral,xai,temporal,ag-ui,retries]"`. There is no `[all]` extra.

For a smaller install, use `pydantic-ai-slim` with only the extras you need:

```bash
pip install "pydantic-ai-slim[anthropic]"
pip install "pydantic-ai-slim[openai,logfire]"
pip install "pydantic-ai-slim[openai,temporal]"   # or dbos, prefect
```

### API keys

Providers read their key from the environment: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY` and so on. A `.env` file is convenient in development:

```ini
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
LOGFIRE_TOKEN=...          # optional, for observability
```

PydanticAI does not load `.env` files itself; use `python-dotenv` (`pip install python-dotenv`) or your process manager:

```python
import os

from dotenv import load_dotenv

load_dotenv()
assert os.getenv('OPENAI_API_KEY'), 'OPENAI_API_KEY not set'
```

### Hello world

```python
from pydantic_ai import Agent

agent = Agent('openai:%%model=openai%%')

result = agent.run_sync('What is 2 + 2?')
print(result.output)
#> 2 + 2 equals 4.

print(result.usage)   # a property since 2.0; it was a method in V1
#> RunUsage(input_tokens=14, output_tokens=5, requests=1)
```

`Agent('test')` runs the same code against the built-in test model without an API key.

## Core fundamentals

### Agent

The primary class. Create it once and reuse it.

```python
from dataclasses import dataclass

from pydantic_ai import Agent

agent = Agent('openai:%%model=openai%%')

agent_with_instructions = Agent(
    'openai:%%model=openai%%',
    instructions='Be concise and professional. Reply with one or two sentences.',
)


@dataclass
class UserContext:
    user_id: int
    username: str


agent_complete = Agent(
    'openai:%%model=openai%%',
    deps_type=UserContext,
    output_type=str | None,
    instructions='Provide clear, working code examples.',
    retries=2,          # retry budget for tools and output validation
    name='PythonHelper',
)
```

`instructions=` is the recommended way to give the model its brief. Each request carries the instructions of the agent making it: they are recorded on `ModelRequest.instructions` for inspection, but they are not replayed from `message_history`, so a later run with a different agent sends its own instructions rather than inheriting the old ones. `system_prompt=` text, by contrast, is stored as a `SystemPromptPart` in the history and travels with it.

### RunContext

Tools, instruction functions and validators can take a `RunContext[DepsT]` as their first argument. It carries the run's dependencies, the model, usage so far, the message history and retry state.

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext


@dataclass
class AppDependencies:
    database_url: str
    api_key: str


agent = Agent('openai:%%model=openai%%', deps_type=AppDependencies)


@agent.tool
async def fetch_user_data(ctx: RunContext[AppDependencies], user_id: int) -> str:
    """Fetch a user's record.

    Args:
        user_id: The user identifier.
    """
    db_url = ctx.deps.database_url          # typed dependencies
    model_name = ctx.model.model_name       # the model handling this run
    history = ctx.messages                  # messages so far in this run
    attempt = ctx.retry                     # 0 on the first attempt
    return f'User {user_id} from {db_url} ({len(history)} messages, attempt {attempt}, {model_name})'
```

The most used fields are `deps`, `model`, `usage`, `messages`, `retry`, `max_retries`, `tool_name`, `tool_call_approved`, `run_id` and `metadata`.

### ModelRetry

Raise `ModelRetry` from a tool or output validator to send an error message back to the model and let it try again.

```python
import re

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext


class EmailAddress(BaseModel):
    email: str = Field(description='Valid email address')
    name: str = Field(description='Person name')


agent = Agent('openai:%%model=openai%%', output_type=EmailAddress)


@agent.output_validator
async def validate_email(ctx: RunContext, output: EmailAddress) -> EmailAddress:
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', output.email):
        raise ModelRetry(f'Invalid email format: {output.email}. Provide a valid address.')
    if len(output.name) < 2:
        raise ModelRetry(f'Name too short: {output.name}. Provide a full name.')
    return output


result = agent.run_sync('Extract the contact from: Contact John Doe at john@example.com')
print(result.output)
#> email='john@example.com' name='John Doe'
```

When the retry budget is spent the run raises `UnexpectedModelBehavior`. For a failure that retrying cannot fix, raise `ToolFailed` instead: the model is told the call failed and no retry is spent.

### Tools

Functions decorated with `@agent.tool` (which receive `RunContext`) or `@agent.tool_plain` (which do not) become tools. The JSON schema comes from the type hints; parameter descriptions come from the docstring.

```python
import asyncio
from datetime import datetime
from typing import Literal

from pydantic_ai import Agent, RunContext

agent = Agent('openai:%%model=openai%%')


@agent.tool
async def search_database(ctx: RunContext, query: str, limit: int = 10) -> str:
    """Search the document database.

    Args:
        query: Search query.
        limit: Maximum results.
    """
    await asyncio.sleep(0.1)   # simulate I/O
    return f'Found {limit} results for {query!r}'


@agent.tool_plain
def get_current_time() -> str:
    """Get the current server time in ISO format."""
    return datetime.now().isoformat()


@agent.tool(strict=True)   # ask OpenAI for strict schema adherence
async def calculate(
    ctx: RunContext, a: int, b: int, operation: Literal['add', 'subtract', 'multiply', 'divide']
) -> int:
    """Apply an arithmetic operation.

    Args:
        a: First number.
        b: Second number.
        operation: The operation to apply.
    """
    operations = {'add': a + b, 'subtract': a - b, 'multiply': a * b, 'divide': a // b if b else 0}
    return operations[operation]


result = agent.run_sync('What time is it?')
print(result.output)
```

### Choosing a model

A model is named `provider:model`. Common prefixes:

| Prefix | Provider | Example |
|---|---|---|
| `openai:` | OpenAI Responses API | `'openai:%%model=openai%%'` |
| `openai-chat:` | OpenAI Chat Completions | `'openai-chat:%%model=openai%%'` |
| `anthropic:` | Anthropic | `'anthropic:%%model=anthropic%%'` |
| `google:` / `google-cloud:` | Gemini API / Vertex AI | `'google:%%model=gemini%%'` |
| `bedrock:` | Amazon Bedrock (extra) | `'bedrock:us.anthropic.%%model=anthropic%%'` |
| `groq:`, `mistral:`, `xai:`, `deepseek:`, `openrouter:`, … | other providers (most need an extra) | `'mistral:%%model=mistral%%'` |
| `gateway/<provider>:` | Pydantic AI Gateway | `'gateway/anthropic:%%model=anthropic%%'` |

Since 2.0 a model name without a prefix raises `UserError`. OpenAI-compatible services (Ollama, vLLM, Perplexity, Together, …) use an OpenAI model class with a provider that sets the base URL. The [model providers page](/pydanticai-guide/pydantic_ai_model_providers/) lists every prefix, explicit provider construction, local models and `FallbackModel`.

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

local = OpenAIChatModel('qwen3', provider=OpenAIProvider(base_url='http://localhost:11434/v1'))
agent = Agent(local)
```

### Model settings and usage limits

```python
from pydantic import BaseModel
from pydantic_ai import Agent, ModelSettings, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

agent = Agent('openai:%%model=openai%%', instructions='Be concise.', retries=3, name='MyAgent')

settings = ModelSettings(temperature=0.7, max_tokens=500, top_p=0.95)
result = agent.run_sync('What is Python?', model_settings=settings)   # also accepted by Agent(...)


class Article(BaseModel):
    title: str
    content: str
    keywords: list[str]


article_agent = Agent('openai:%%model=openai%%', output_type=Article)

limits = UsageLimits(
    request_limit=5,           # model requests per run (default 50)
    tool_calls_limit=10,       # successful tool executions per run
    total_tokens_limit=10_000,
)
result = article_agent.run_sync('Write about type safety in Python', usage_limits=limits)

usage = result.usage
print(usage.requests, usage.tool_calls, usage.input_tokens, usage.output_tokens, usage.total_tokens)

try:
    agent.run_sync('Count every prime below 10000.', usage_limits=UsageLimits(request_limit=1, output_tokens_limit=50))
except UsageLimitExceeded as exc:
    print(f'Run aborted: {exc}')
```

Most `ModelSettings` keys a model does not support (`temperature`, `seed`, …) are ignored silently, so one settings dict works across providers. The settings whose absence changes what a request guarantees — `timeout`, `max_retries`, `stop_sequences`, `parallel_tool_calls=False`, `extra_headers` and `extra_body` — emit an `UnsupportedSettingWarning` instead; turn it into an error with `warnings.filterwarnings('error', category=UnsupportedSettingWarning)`. Cost limits, timeouts and the other production guards are on the [production page](/pydanticai-guide/pydantic_ai_production_guide/#limits-tokens-cost-time).

## Simple agents

### Running an agent

```python
import asyncio

from pydantic_ai import Agent

agent = Agent('openai:%%model=openai%%', instructions='Respond with exactly one sentence.')

result = agent.run_sync('What is the capital of France?')   # scripts and notebooks
print(result.output)
#> The capital of France is Paris.


async def main() -> str:
    result = await agent.run('Explain type safety in Python.')   # inside async code
    return result.output


print(asyncio.run(main()))
```

`run_sync()` cannot be called from inside a running event loop; use `await agent.run(...)` there. `agent.run_stream()` streams the output and `agent.iter()` walks the run node by node (see [streaming](/pydanticai-guide/pydantic_ai_streaming/)).

### Instructions: static and dynamic

```python
from datetime import datetime, timezone

from pydantic_ai import Agent, RunContext

agent = Agent('openai:%%model=openai%%', instructions='You are a helpful assistant.')


@agent.instructions
def current_time(ctx: RunContext) -> str:
    """Re-evaluated for every run."""
    return f'The current UTC time is {datetime.now(timezone.utc):%Y-%m-%d %H:%M}.'


result = agent.run_sync('What day is it?')

# Extra instructions for a single run are appended to the agent's own
result = agent.run_sync('Explain closures in Python.', instructions='Explain at a beginner level.')
print(len(result.all_messages()), result.usage.input_tokens)
```

`@agent.system_prompt` and `system_prompt=` still exist. Use them when the prompt must stay in the message history (for example when another agent will continue the conversation); otherwise prefer instructions.

### Streaming

```python
import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent

agent = Agent('openai:%%model=openai%%')


class Article(BaseModel):
    title: str
    content: str


structured_agent = Agent('openai:%%model=openai%%', output_type=Article)


async def stream_text() -> None:
    async with agent.run_stream('Write a haiku about Python') as response:
        async for text in response.stream_text(delta=True):
            print(text, end='', flush=True)
    print()


async def stream_structured() -> None:
    async with structured_agent.run_stream('Write an article about type safety') as response:
        async for partial in response.stream_output():   # partially validated Article objects
            print(partial.title or '…')
        article = await response.get_output()
    print(f'Title: {article.title}; {len(article.content)} characters')


asyncio.run(stream_text())
asyncio.run(stream_structured())
```

`stream_text()` only works when the output type is text; for structured output use `stream_output()`. The [streaming page](/pydanticai-guide/pydantic_ai_streaming/) covers `run_stream_events()`, debouncing and cancellation.

### Validation and retries on structured output

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior


class CodeReview(BaseModel):
    issues: list[str] = Field(min_length=1, description='Code issues found')
    severity: str = Field(pattern='^(low|medium|high)$')
    suggestions: list[str]


agent = Agent('openai:%%model=openai%%', output_type=CodeReview, retries=2)


@agent.output_validator
async def one_suggestion_per_issue(ctx: RunContext, output: CodeReview) -> CodeReview:
    if len(output.issues) > 10:
        raise ModelRetry('Limit the review to at most 10 issues.')
    if len(output.suggestions) != len(output.issues):
        raise ModelRetry(
            f'Give one suggestion per issue: {len(output.issues)} issues, {len(output.suggestions)} suggestions.'
        )
    return output


try:
    result = agent.run_sync('Review this Python code: x = 1; y = 2; z = x+y')
    print(result.output.severity, len(result.output.issues))
except UnexpectedModelBehavior as exc:   # retries exhausted
    print(f'Validation kept failing: {exc}')
```

Pydantic field constraints (`min_length`, `pattern`, …) are enforced first; a failure there is sent back to the model as a retry in the same way as `ModelRetry`.

## Type safety and validation

### Pydantic models as output

Anything Pydantic can validate can be an output type: models, dataclasses, `TypedDict`s, scalars, lists and unions.

```python
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_ai import Agent


class UserProfile(BaseModel):
    model_config = ConfigDict(json_schema_extra={'example': {'id': 1, 'name': 'John', 'email': 'john@example.com'}})

    id: int = Field(gt=0, description='Unique user identifier')
    name: str = Field(min_length=1, max_length=100)
    email: str
    age: int | None = Field(default=None, ge=0, le=150)
    premium: bool = False


class ValidatedArticle(BaseModel):
    title: str = Field(min_length=5, max_length=200)
    content: str = Field(min_length=100)
    tags: list[str] = Field(default_factory=list, max_length=10)
    published_date: datetime | None = None

    @field_validator('tags')
    @classmethod
    def normalise_tags(cls, v: list[str]) -> list[str]:
        return sorted({tag.lower() for tag in v})


article_agent = Agent('openai:%%model=openai%%', output_type=ValidatedArticle)
result = article_agent.run_sync('Write an article about Python type hints')
print(result.output.title, result.output.tags)
```

Field descriptions and constraints go into the JSON schema the model sees, so they steer the model as well as validating its answer.

### Typed dependencies

```python
from dataclasses import dataclass

import httpx
from pydantic_ai import Agent, RunContext


@dataclass
class ServiceDependencies:
    http_client: httpx.AsyncClient
    api_base: str
    api_key: str
    user_id: int


agent = Agent('openai:%%model=openai%%', deps_type=ServiceDependencies)


@agent.tool
async def fetch_profile(ctx: RunContext[ServiceDependencies], include_preferences: bool = False) -> dict:
    """Fetch the current user's profile.

    Args:
        include_preferences: Whether to include preference data.
    """
    headers = {'X-API-Key': ctx.deps.api_key}
    base = f'{ctx.deps.api_base}/users/{ctx.deps.user_id}'
    data = (await ctx.deps.http_client.get(base, headers=headers)).json()
    if include_preferences:
        data['preferences'] = (await ctx.deps.http_client.get(f'{base}/preferences', headers=headers)).json()
    return data


@agent.instructions
def personalise(ctx: RunContext[ServiceDependencies]) -> str:
    return f'You are helping user {ctx.deps.user_id}.'


async def main() -> None:
    async with httpx.AsyncClient() as client:
        deps = ServiceDependencies(client, 'https://api.example.com', 'secret-key', 123)
        result = await agent.run('Tell me about my profile', deps=deps)
        print(result.output)
```

A type checker now knows the shape of `ctx.deps` in every tool, and `agent.run(deps=...)` is checked against `deps_type`.

## Structured output

### Nested models

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent


class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str


class Contact(BaseModel):
    email: str = Field(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    phone: str | None = None


class Company(BaseModel):
    name: str
    founded: int = Field(ge=1800, le=2100)
    employees: int = Field(gt=0)
    address: Address
    contacts: list[Contact]
    website: str | None = None


agent = Agent('openai:%%model=openai%%', output_type=Company)
result = agent.run_sync('Extract company information for Pydantic: founded 2017, about 50 employees, based in London.')

company = result.output
print(company.name, company.address.city, company.contacts[0].email if company.contacts else None)
```

### Unions and discriminated unions

Pass several output types as a list (each becomes its own output tool), or use a discriminated union inside a model:

```python
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent


class TextResult(BaseModel):
    kind: Literal['text']
    content: str


class JsonResult(BaseModel):
    kind: Literal['json']
    data: dict


class ProcessingResult(BaseModel):
    status: str
    output: Annotated[TextResult | JsonResult, Field(discriminator='kind')]


agent = Agent('openai:%%model=openai%%', output_type=ProcessingResult)
result = agent.run_sync('Output JSON data about Python')

match result.output.output:
    case JsonResult(data=data):
        print('JSON keys:', list(data))
    case TextResult(content=content):
        print('Text:', content)


class Refusal(BaseModel):
    reason: str


either_agent = Agent('openai:%%model=openai%%', output_type=[ProcessingResult, Refusal])
```

`output_type=[A, B]` gives the model a separate output tool per type, which models handle more reliably than one large union schema.

### Choosing the output mode

By default structured output uses a tool call. `NativeOutput` uses the provider's JSON-schema response format, `PromptedOutput` puts the schema in the prompt, `TextOutput` passes plain text to your parser and `StructuredDict` validates against a JSON schema without a Pydantic model. The [output types page](/pydanticai-guide/pydantic_ai_output_types/) compares them.

## Tools and function calling

### Parameters and return types

```python
import random
import re
from typing import Any, Literal

from pydantic import Field
from pydantic_ai import Agent, RunContext

agent = Agent('openai:%%model=openai%%')


@agent.tool
async def calculate_factorial(ctx: RunContext, n: int) -> int:
    """Calculate n!."""
    if n < 0:
        raise ValueError('Factorial not defined for negative numbers')
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


@agent.tool
async def search_and_rank(
    ctx: RunContext,
    query: str,
    filters: dict[str, Any],
    sort_by: Literal['relevance', 'date', 'popularity'] = 'relevance',
    limit: int = Field(10, ge=1, le=100),
) -> dict[str, Any]:
    """Search documents and rank the results.

    Args:
        query: Search query string.
        filters: Filter conditions.
        sort_by: Field to sort by.
        limit: Maximum results to return.
    """
    results = [{'id': i, 'score': 1 - i * 0.1, 'title': f'Result {i}'} for i in range(min(limit, 5))]
    return {'results': results, 'total': 1000, 'query': query, 'sort_by': sort_by}


@agent.tool_plain
def get_random_number(min_value: int = 0, max_value: int = 100) -> int:
    """Generate a random integer between min_value and max_value."""
    return random.randint(min_value, max_value)


@agent.tool_plain
def validate_email(email: str) -> dict[str, bool | str]:
    """Check whether an email address is well formed."""
    return {'valid': bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email)), 'email': email}


result = agent.run_sync('Pick a random number and check the email bob@example.com.')
```

`Literal`, enums, `Field(...)` constraints and Pydantic models all appear in the schema. Arguments that fail validation are sent back to the model as a retry. The return value can be anything Pydantic can serialise.

### Async tools

Async tools run on the event loop; sync tools run in a worker thread so they do not block it. When the model requests several tools in one response, they run concurrently.

```python
import asyncio

import httpx
from pydantic_ai import Agent, RunContext

agent = Agent('openai:%%model=openai%%')


@agent.tool
async def fetch_webpage(ctx: RunContext, url: str) -> str:
    """Fetch the first 1000 characters of a page."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=10)
        return response.text[:1000]


@agent.tool
async def parallel_searches(ctx: RunContext, queries: list[str]) -> list[str]:
    """Run several searches concurrently."""

    async def search_one(q: str) -> str:
        await asyncio.sleep(0.1)
        return f'Results for {q!r}'

    return await asyncio.gather(*(search_one(q) for q in queries))
```

For a model-chosen URL, prefer `web_fetch_tool()` from `pydantic_ai.common_tools`, which blocks private and cloud-metadata addresses ([built-in tools](/pydanticai-guide/pydantic_ai_builtin_tools/#web_fetch_tool--ssrf-protected-url-fetching)).

### Conditional tools with `prepare`

A `prepare` function receives the tool's `ToolDefinition` before each model request and returns it (possibly modified) or `None` to hide the tool for that step.

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import ToolDefinition


@dataclass
class Deps:
    user_role: str


agent = Agent('openai:%%model=openai%%', deps_type=Deps)


async def only_if_admin(ctx: RunContext[Deps], tool_def: ToolDefinition) -> ToolDefinition | None:
    return tool_def if ctx.deps.user_role == 'admin' else None


@agent.tool(prepare=only_if_admin)
async def delete_record(ctx: RunContext[Deps], record_id: int) -> bool:
    """Delete a record (admins only)."""
    return True


result = agent.run_sync('Delete record 7', deps=Deps(user_role='viewer'))   # the tool is not offered
```

To filter or rewrite many tools at once, use the toolset wrappers (`FilteredToolset`, `PreparedToolset`) or the `PrepareTools` capability; see [toolsets](/pydanticai-guide/pydantic_ai_toolsets/).

### Errors in tools

```python
import httpx
from pydantic_ai import Agent, ModelRetry, RunContext, ToolFailed

agent = Agent('openai:%%model=openai%%')


@agent.tool
async def fetch_resource(ctx: RunContext, resource_id: int) -> dict:
    """Fetch a resource by ID."""
    if resource_id <= 0:
        raise ModelRetry(f'Invalid resource_id {resource_id}: it must be positive.')
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f'https://api.example.com/resources/{resource_id}', timeout=5)
    except httpx.TimeoutException:
        raise ModelRetry('The request timed out; try again.')
    if response.status_code == 404:
        raise ToolFailed(f'Resource {resource_id} does not exist.')   # not worth retrying
    response.raise_for_status()
    return response.json()
```

- `ModelRetry` sends the message back and spends one retry.
- `ToolFailed` tells the model the call failed, without asking it to retry.
- Any other exception propagates and ends the run.

The [error handling page](/pydanticai-guide/pydantic_ai_advanced_error_testing/) lists every exception and how to test each path.

### Native tools

Provider-native tools (web search, code execution, image generation, web fetch, file search, …) run on the provider's side. Since 2.0 they are attached as capabilities: `capabilities=[WebSearch()]` (with an optional local fallback) or `capabilities=[NativeTool(WebSearchTool())]` (the raw tool). Support varies by provider; the [built-in tools page](/pydanticai-guide/pydantic_ai_builtin_tools/#the-catalogue) has the support table.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

agent = Agent('anthropic:%%model=anthropic%%', capabilities=[WebSearch(max_uses=3)])
result = agent.run_sync('What is the latest Python release? Cite the source.')
```

## Dependency injection

`deps` is the only state an agent run sees that you did not put in the prompt. Use it for database pools, HTTP clients, configuration and per-request identity. Dependencies are passed per run, so one agent can serve many tenants.

```python
import asyncio
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, RunContext


@dataclass
class ApplicationState:
    user_id: int
    session_id: str
    cache: dict[str, Any] = field(default_factory=dict)


agent = Agent('openai:%%model=openai%%', deps_type=ApplicationState)


@agent.tool
async def store_value(ctx: RunContext[ApplicationState], key: str, value: str) -> str:
    """Remember a value for this session."""
    ctx.deps.cache[key] = value
    return f'Stored {key!r}'


@agent.tool
async def retrieve_value(ctx: RunContext[ApplicationState], key: str) -> str | None:
    """Recall a value stored earlier in this session."""
    return ctx.deps.cache.get(key)


@agent.instructions
def session_context(ctx: RunContext[ApplicationState]) -> str:
    return f'You are assisting user {ctx.deps.user_id} in session {ctx.deps.session_id}.'


async def main() -> None:
    state = ApplicationState(user_id=123, session_id='sess_abc123')
    first = await agent.run('Store my favourite language as Python', deps=state)
    # The same deps object is passed again, so the cache persists; pass the history too
    second = await agent.run('What is my favourite language?', deps=state, message_history=first.all_messages())
    print(second.output)


asyncio.run(main())
```

In tests, pass fake dependencies the same way, or replace them for a block of code with `agent.override(deps=...)`.

## Upgrading from V1

PydanticAI 2.0.0 (June 2026) collected the breaking changes V1's stability promise held back. Most were announced by `PydanticAIDeprecationWarning`s from 1.95.0 onwards, so the smooth path is: upgrade to the last 1.x, clear every `PydanticAIDeprecationWarning`, then move to 2.x. The changes most likely to affect code written from older guides:

| V1 | V2 |
|---|---|
| `openai:` prefix = Chat Completions | `openai:` = Responses API; use `openai-chat:` for Chat Completions |
| `google-gla:`, `google-vertex:`, `grok:` prefixes; `GeminiModel`, `OpenAIModel` | `google:`, `google-cloud:`, `xai:`; `GoogleModel`, `OpenAIChatModel` |
| `Agent('gpt-5')` without a prefix | raises `UserError` |
| `Agent(builtin_tools=[...])`, `pydantic_ai.builtin_tools`, `UrlContextTool` | `capabilities=[NativeTool(...)]`, `pydantic_ai.native_tools`, `WebFetchTool` |
| `Agent(instrument=..., event_stream_handler=..., history_processors=..., prepare_tools=...)` | `capabilities=[Instrumentation(...)]`, `ProcessEventStream(...)`, `ProcessHistory(...)`, `PrepareTools(...)` |
| `Agent(mcp_servers=[...])`, `MCPServerStdio` / `MCPServerSSE` / `MCPServerStreamableHTTP` | `toolsets=[MCPToolset(...)]` |
| `result.usage()`, `result.timestamp()`, `stream.get()` | `result.usage`, `result.timestamp`, `stream.response` (properties) |
| `async for e in agent.run_stream_events(...)` | `async with agent.run_stream_events(...) as events:` |
| `stream_responses()`, `FunctionToolResultEvent.result` | `stream_response()`, `FunctionToolResultEvent.part` |
| `DeferredToolCalls`, `DeferredToolset` | `DeferredToolRequests`, `ExternalToolset` |
| `usage.request_tokens` / `response_tokens` | `usage.input_tokens` / `output_tokens` |
| `Agent.to_a2a()`, `Agent.to_ag_ui()` | `fasta2a.pydantic_ai.agent_to_a2a`, `pydantic_ai.ui.ag_ui.AGUIAdapter` |
| `pydantic_graph.beta`, `pydantic_graph.persistence` | top-level `pydantic_graph`; persistence removed |
| `Dataset(...)` without a name | `Dataset(name=...)` required |
| Default `end_strategy='early'` | Default `'graceful'`: function tools requested before an output tool still run |
| `ModelProfile` dataclasses | `TypedDict`s; use `merge_profile()` |
| Bare `pip install pydantic-ai` included Bedrock, Groq, Mistral, … | Only OpenAI, Anthropic and Google providers by default; add extras |
| `TemporalAgent`, `DBOSAgent`, `PrefectAgent` wrappers | deprecated since 2.14 in favour of the `TemporalDurability` / `DBOSDurability` / `PrefectDurability` capabilities ([durable execution](/pydanticai-guide/pydantic_ai_durable_execution/#migrating-from-the-wrapper-classes)) |

The full list, with the PR for each change, is in the [official upgrade guide](https://pydantic.dev/docs/ai/changelog/). Message history serialised with V1 still loads in V2.

## Agent reference

### Constructor

```text
Agent(
    model: Model | KnownModelName | str | None = None,
    *,
    output_type: OutputSpec = str,
    instructions: AgentInstructions = None,
    system_prompt: str | Sequence[str] = (),
    deps_type: type = object,
    name: str | None = None,
    description: TemplateStr | str | None = None,
    model_settings: AgentModelSettings | None = None,
    retries: int | AgentRetries | None = None,      # default 1 for tools and output
    validation_context: Any | Callable[[RunContext], Any] = None,
    tools: Sequence[Tool | ToolFunc] = (),
    toolsets: Sequence[AgentToolset] | None = None,
    defer_model_check: bool = False,
    end_strategy: EndStrategy = 'graceful',
    metadata: AgentMetadata | None = None,
    tool_timeout: float | None = None,
    max_concurrency: int | ConcurrencyLimit | AbstractConcurrencyLimiter | None = None,
    capabilities: Sequence[AgentCapability] | None = None,
)
```

- **`instructions`** accepts a string, a `TemplateStr`, a function of `RunContext` (sync or async), or a list mixing these; `@agent.instructions` registers more.
- **`model_settings`**, **`toolsets`** and **`capabilities`** each also accept a function of `RunContext`, resolved per run, for per-tenant settings or feature flags.
- **`retries`** is an `int` (both budgets) or an `AgentRetries` dict: `retries={'tools': 3, 'output': 1}`.
- **`defer_model_check=True`** postpones resolving the model string (and the API-key check) until the first run, useful when tests override the model.
- **`metadata`** is attached to the run and its spans but never sent to the model.
- **`tool_timeout`** and **`max_concurrency`** are covered on the [production page](/pydanticai-guide/pydantic_ai_production_guide/).

### End strategy

`end_strategy` decides what happens to function tools the model requests in the same response as its final output. It is set on the agent only; `run()` does not take it.

| Value | Behaviour |
|---|---|
| `'graceful'` (default since 2.0) | Tools run in the order the model emitted them, so function tools before the output tool complete. The first successful output tool wins; later output tools are skipped. A `ModelRetry` from a function tool suppresses the output and surfaces the retry. |
| `'early'` | The run ends at the first successful output tool; function tools in that response are not executed (the V1 default). |
| `'exhaustive'` | Every requested tool runs (in parallel by default); the first valid output by emission order becomes the result. |

```python
from pydantic_ai import Agent

agent = Agent('openai:%%model=openai%%', end_strategy='early', retries={'tools': 3, 'output': 1})
```

### Run methods

| Method | Use |
|---|---|
| `await agent.run(prompt, ...)` | Run to completion; returns `AgentRunResult` |
| `agent.run_sync(prompt, ...)` | The same, from synchronous code |
| `async with agent.run_stream(prompt) as stream:` | Stream text or partial output ([streaming](/pydanticai-guide/pydantic_ai_streaming/)) |
| `agent.run_stream_sync(prompt)` | Synchronous streaming |
| `async with agent.run_stream_events(prompt) as events:` | Every event, ending with `AgentRunResultEvent` |
| `async with agent.iter(prompt) as run:` | Step through the graph node by node |
| `agent.override(model=..., deps=..., toolsets=...)` | Replace parts of the agent in a `with` block (tests) |
| `agent.to_web()`, `agent.to_cli()` | Chat web app, terminal chat ([integrations](/pydanticai-guide/pydantic_ai_integrations/#built-in-chat-ui-and-cli)) |
| `Agent.from_spec(...)`, `Agent.from_file(path)` | Build an agent from a YAML/JSON `AgentSpec` |

The run methods share these keyword arguments: `message_history` or `conversation` (a whole `Conversation`, see below), `deps`, `instructions` (appended for this run), `model`, `model_settings`, `output_type`, `usage_limits`, `usage`, `timeout`, `cancellation_token`, `metadata`, `retries`, `toolsets`, `capabilities`, `deferred_tool_results`, `conversation_id` and `run_id`; `run`, `run_sync`, `run_stream` and `run_stream_sync` also take a per-run `event_stream_handler`.

`AgentRunResult` exposes `output`, `usage` (a property), `response` (the final `ModelResponse`), `run_id`, `conversation_id`, `metadata`, `conversation` (messages, usage, conversation ID and pending deferred calls in one object, ready to pass back as `conversation=`), and `all_messages()` / `new_messages()` with `_json()` variants for storage ([message history](/pydanticai-guide/pydantic_ai_message_history/#conversation--carry-the-whole-conversation-not-just-the-messages)).

### Iterating a run

```python
import asyncio

from pydantic_ai import Agent, CallToolsNode, ModelRequestNode

agent = Agent('openai:%%model=openai%%')


@agent.tool_plain
def lookup(city: str) -> str:
    """Look up a city's population."""
    return f'{city}: 9 million'


async def main() -> None:
    async with agent.iter('How many people live in London?') as run:
        async for node in run:
            if isinstance(node, ModelRequestNode):
                print('request with', [type(p).__name__ for p in node.request.parts])
            elif isinstance(node, CallToolsNode):
                print('response:', node.model_response.finish_reason)
    print(run.result.output)


asyncio.run(main())
```

`async for` yields each node before it runs and advances through `run.next()`, so capability hooks fire exactly as they do for `agent.run()`. Call `node = await run.next(node)` yourself when you want to inspect or replace a node before it executes, or stop early by breaking out of the loop.

### Agents from a spec

An `AgentSpec` describes an agent's model, instructions, settings, retries and capabilities as data, so it can live in YAML or JSON. Capabilities use short forms: a bare name (`'Thinking'`) or `{Name: {kwargs}}`.

```python
from pydantic_ai import Agent, AgentSpec

spec = AgentSpec.from_dict({
    'model': 'anthropic:%%model=anthropic%%',
    'name': 'support-agent',
    'instructions': 'You are a helpful support agent.',
    'model_settings': {'temperature': 0.3},
    'retries': 3,
    'capabilities': ['Thinking'],
})
agent = Agent.from_spec(spec)   # or Agent.from_file('agent.yaml')
```

`AgentSpec.from_file()`, `.from_text()` and `.to_file()` read and write YAML or JSON. There is no `spec.to_agent()`; build through `Agent.from_spec()` or `Agent.from_file()`.

## API map

The public API grouped by area, with the page that covers each part. Everything listed is importable from the module shown; most names are also exported from `pydantic_ai` itself. The [upstream documentation](https://pydantic.dev/docs/ai/core-concepts/agent/) and the docstrings in the installed package document every class and argument.

### Agents and runs

| Names | Module | Covered in |
|---|---|---|
| `Agent`, `AgentRunResult`, `AgentRun`, `RunContext` | `pydantic_ai` | this page |
| `UserPromptNode`, `ModelRequestNode`, `CallToolsNode`, `AgentRunResultEvent` | `pydantic_ai` | [Streaming](/pydanticai-guide/pydantic_ai_streaming/#agentiter--drive-the-graph-yourself) |
| `AgentSpec`, `TemplateStr` | `pydantic_ai` | [Agents from a spec](#agents-from-a-spec) |
| `WrapperAgent`, `AbstractAgent` | `pydantic_ai.agent` | [Advanced patterns](/pydanticai-guide/pydantic_ai_advanced_patterns/) |
| `UsageLimits`, `RunUsage`, `RequestUsage` | `pydantic_ai` / `pydantic_ai.usage` | [Production](/pydanticai-guide/pydantic_ai_production_guide/#limits-tokens-cost-time) |
| `CancellationToken`, `RunCancelled`, `RunTimedOut` | `pydantic_ai`, `pydantic_ai.exceptions` | [Error handling](/pydanticai-guide/pydantic_ai_advanced_error_testing/#cancellation-and-timeouts--runcancelled--runtimedout) |
| `capture_run_messages` | `pydantic_ai` | [Message history](/pydanticai-guide/pydantic_ai_message_history/#capture_run_messages--inspect-mid-failure) |
| `model_request`, `model_request_sync`, `model_request_stream`, `model_request_stream_sync` | `pydantic_ai.direct` | [Direct API](/pydanticai-guide/pydantic_ai_direct_api/) |

### Models and providers

| Names | Module | Covered in |
|---|---|---|
| `OpenAIResponsesModel`, `OpenAIChatModel`, `AnthropicModel`, `GoogleModel`, `BedrockConverseModel`, `GroqModel`, `MistralModel`, `XaiModel`, `OpenRouterModel`, … | `pydantic_ai.models.<provider>` | [Model providers](/pydanticai-guide/pydantic_ai_model_providers/) |
| `OpenAIProvider`, `AnthropicProvider`, `GoogleProvider`, `GoogleCloudProvider`, `BedrockProvider`, … | `pydantic_ai.providers.<provider>` | [Model providers](/pydanticai-guide/pydantic_ai_model_providers/#explicit-provider-construction) |
| `ModelSettings` and the per-provider `*ModelSettings` | `pydantic_ai.settings`, `pydantic_ai.models.<provider>` | [Model providers](/pydanticai-guide/pydantic_ai_model_providers/#modelsettings--provider-agnostic-knobs) |
| `FallbackModel`, `FallbackExceptionGroup` | `pydantic_ai.models.fallback`, `pydantic_ai.exceptions` | [Model providers](/pydanticai-guide/pydantic_ai_model_providers/#fallbackmodel--wrap-primaries-with-backups), [Production](/pydanticai-guide/pydantic_ai_production_guide/#fallback-models) |
| `WrapperModel`, `InstrumentedModel`, `InstrumentationSettings` | `pydantic_ai.models.wrapper`, `pydantic_ai.models.instrumented`, `pydantic_ai` | [Model providers](/pydanticai-guide/pydantic_ai_model_providers/#instrumentedmodel-and-instrumentationsettings) |
| `ModelProfile`, `merge_profile` | `pydantic_ai.profiles` | [Model providers](/pydanticai-guide/pydantic_ai_model_providers/) |
| `TestModel`, `FunctionModel`, `AgentInfo` | `pydantic_ai.models.test`, `pydantic_ai.models.function` | [Testing](/pydanticai-guide/pydantic_ai_testing/) |
| `ConcurrencyLimiter`, `ConcurrencyLimit`, `limit_model_concurrency` | `pydantic_ai` | [Concurrency](/pydanticai-guide/pydantic_ai_concurrency/) |
| `AsyncHTTPX2TenacityTransport`, `RetryConfig`, `wait_retry_after` | `pydantic_ai.retries` (`retries` extra) | [Production](/pydanticai-guide/pydantic_ai_production_guide/#http-retries) |
| `Embedder`, `EmbeddingModel` | `pydantic_ai`, `pydantic_ai.embeddings` | [Embeddings](/pydanticai-guide/pydantic_ai_embeddings/) |

### Tools and toolsets

| Names | Module | Covered in |
|---|---|---|
| `Tool`, `ToolDefinition`, `ToolReturn` | `pydantic_ai`, `pydantic_ai.tools` | [Tools](#tools-and-function-calling), [ToolReturn](#returning-extra-content-from-a-tool) |
| `ModelRetry`, `ToolFailed` | `pydantic_ai` | [Error handling](/pydanticai-guide/pydantic_ai_advanced_error_testing/) |
| `FunctionToolset`, `CombinedToolset`, `PrefixedToolset`, `RenamedToolset`, `FilteredToolset`, `PreparedToolset`, `SetMetadataToolset`, `IncludeReturnSchemasToolset`, `DynamicToolset`, `WrapperToolset`, `AbstractToolset` | `pydantic_ai.toolsets` | [Toolsets](/pydanticai-guide/pydantic_ai_toolsets/) |
| `ApprovalRequiredToolset`, `ExternalToolset`, `DeferredLoadingToolset` | `pydantic_ai.toolsets` | [Toolsets](/pydanticai-guide/pydantic_ai_toolsets/#deferred-tools-approval-and-external-execution) |
| `ApprovalRequired`, `CallDeferred`, `DeferredToolRequests`, `DeferredToolResults`, `ToolApproved`, `ToolDenied` | `pydantic_ai` | [Toolsets](/pydanticai-guide/pydantic_ai_toolsets/#deferring-from-inside-a-tool--approvalrequired-and-calldeferred) |
| `MCPToolset`, `load_mcp_toolsets` | `pydantic_ai.mcp` | [Integrations](/pydanticai-guide/pydantic_ai_integrations/#model-context-protocol-mcp) |
| `WebSearchTool`, `WebFetchTool`, `CodeExecutionTool`, `ImageGenerationTool`, `FileSearchTool`, `MemoryTool`, `MCPServerTool`, `XSearchTool`, `AdvisorTool` | `pydantic_ai` / `pydantic_ai.native_tools` | [Built-in tools](/pydanticai-guide/pydantic_ai_builtin_tools/) |
| `web_fetch_tool`, `duckduckgo_search_tool`, `tavily_search_tool`, `ExaToolset` | `pydantic_ai.common_tools.*` | [Built-in tools](/pydanticai-guide/pydantic_ai_builtin_tools/#common-tools--local-python-tools-pydantic_aicommon_tools) |
| `LangChainToolset`, `tool_from_langchain` | `pydantic_ai.ext.langchain` | [upstream tools docs](https://pydantic.dev/docs/ai/tools-toolsets/tools/) |

### Capabilities

Capabilities bundle behaviour (hooks, instructions, tools, settings) and attach with `Agent(capabilities=[...])` or per run.

| Names | Module | Covered in |
|---|---|---|
| `Hooks`, `AbstractCapability`, `WrapperCapability`, `CombinedCapability`, `Capability` | `pydantic_ai.capabilities` | [Hooks](/pydanticai-guide/pydantic_ai_hooks/), [Bundling tools](#bundling-tools-and-instructions) |
| `Instrumentation` | `pydantic_ai.capabilities` | [Production](/pydanticai-guide/pydantic_ai_production_guide/#observability) |
| `ProcessHistory`, `ReinjectSystemPrompt` | `pydantic_ai.capabilities` | [Message history](/pydanticai-guide/pydantic_ai_message_history/#processhistory--trim-summarise-or-redact-in-flight) |
| `ProcessEventStream` | `pydantic_ai.capabilities` | [Streaming](/pydanticai-guide/pydantic_ai_streaming/) |
| `PrepareTools`, `PrepareOutputTools`, `SetToolMetadata`, `PrefixTools`, `Toolset`, `ToolSearch`, `IncludeToolReturnSchemas` | `pydantic_ai.capabilities` | [Toolsets](/pydanticai-guide/pydantic_ai_toolsets/) |
| `HandleDeferredToolCalls` | `pydantic_ai.capabilities` | [Recipes](/pydanticai-guide/pydantic_ai_recipes/) |
| `WebSearch`, `WebFetch`, `ImageGeneration`, `XSearch`, `MCP`, `NativeTool`, `NativeOrLocalTool`, `Thinking` | `pydantic_ai.capabilities` | [Built-in tools](/pydanticai-guide/pydantic_ai_builtin_tools/) |
| `SelectModel`, `ResolveModelId` | `pydantic_ai.capabilities` | [Choosing the model per step](#choosing-the-model-per-step) |
| `RaiseContentFilterError`, `UseThreadExecutor`, `Caching`, `LocalWorkspace` | `pydantic_ai.capabilities` | [upstream capabilities docs](https://pydantic.dev/docs/ai/capabilities/overview/) |
| `TemporalDurability`, `DBOSDurability`, `PrefectDurability` | `pydantic_ai.durable_exec.<engine>` | [Durable execution](/pydanticai-guide/pydantic_ai_durable_execution/) |
| `OnlineEvaluation` | `pydantic_evals.online_capability` | [Evals](/pydanticai-guide/pydantic_ai_evals/#8-online-evaluation--sampling-production-traffic) |

### Output, messages and streaming

| Names | Module | Covered in |
|---|---|---|
| `ToolOutput`, `NativeOutput`, `PromptedOutput`, `TextOutput`, `StructuredDict` | `pydantic_ai` / `pydantic_ai.output` | [Output types](/pydanticai-guide/pydantic_ai_output_types/) |
| `ModelMessage`, `ModelRequest`, `ModelResponse` and their parts, `ModelMessagesTypeAdapter` | `pydantic_ai.messages` | [Message history](/pydanticai-guide/pydantic_ai_message_history/) |
| `BinaryContent`, `ImageUrl`, `AudioUrl`, `VideoUrl`, `DocumentUrl`, `UploadedFile`, `CachePoint` | `pydantic_ai` | [Message history](/pydanticai-guide/pydantic_ai_message_history/#cachepoint--provider-native-prompt-caching), [Recipes](/pydanticai-guide/pydantic_ai_recipes/) |
| `PartStartEvent`, `PartDeltaEvent`, `PartEndEvent`, `FinalResultEvent`, `FunctionToolCallEvent`, `FunctionToolResultEvent` | `pydantic_ai.messages` | [Streaming](/pydanticai-guide/pydantic_ai_streaming/#run_stream_events--raw-protocol-events) |
| `format_as_xml` | `pydantic_ai` | [Recipes](/pydanticai-guide/pydantic_ai_recipes/) |

### Graphs, evals, UI and protocols

| Names | Module | Covered in |
|---|---|---|
| `GraphBuilder`, `BaseNode`, `End`, `GraphRunContext` | `pydantic_graph` | [Graph support](/pydanticai-guide/pydantic_ai_graph_support/) |
| `Dataset`, `Case`, `Evaluator`, `LLMJudge` and the built-in evaluators | `pydantic_evals`, `pydantic_evals.evaluators` | [Evals](/pydanticai-guide/pydantic_ai_evals/) |
| `AGUIAdapter`, `VercelAIAdapter` | `pydantic_ai.ui.ag_ui`, `pydantic_ai.ui.vercel_ai` | [Integrations](/pydanticai-guide/pydantic_ai_integrations/#ui-protocols-and-front-ends), [Streaming server](/pydanticai-guide/pydantic_ai_streaming_server_fastapi/) |
| `agent_to_a2a` | `fasta2a.pydantic_ai` (separate package) | [Integrations](/pydanticai-guide/pydantic_ai_integrations/#agent-to-agent-a2a) |

### Exceptions

```text
AgentRunError (RuntimeError)
├── UsageLimitExceeded
├── ConcurrencyLimitExceeded
├── RunCancelled
│   └── RunTimedOut (also a TimeoutError)
├── HookTimeoutError (also a TimeoutError)
├── UnexpectedModelBehavior
│   ├── ContentFilterError
│   └── IncompleteToolCall
└── ModelAPIError
    ├── ModelHTTPError (status_code, model_name, body, headers, retry_after)
    ├── ModelConnectionError
    │   └── ModelTimeoutError
    ├── ModelRequestTimeout
    └── ModelRateLimitError, ModelQuotaExceededError, ModelServerError,
        ModelContextWindowExceededError, ModelUnavailableError (└── ModelOverloadedError)
        — when one comes from an HTTP response it is also a ModelHTTPError
UserError (RuntimeError)          — a programming mistake, raised before or during a run
FallbackExceptionGroup            — every model in a FallbackModel failed
ModelRetry, ToolFailed            — raised by your code to steer the model
ApprovalRequired, CallDeferred    — raised by your code to defer a tool call
```

The [error handling page](/pydanticai-guide/pydantic_ai_advanced_error_testing/#exception-hierarchy) shows when each is raised and how to test for it. Deprecations use `PydanticAIDeprecationWarning`, a `UserWarning` subclass, so they show by default; turn it into an error in CI with `-W error::pydantic_ai.PydanticAIDeprecationWarning` or a `filterwarnings` entry.

### Returning extra content from a tool

`ToolReturn` separates what the model gets as the tool result from extra content (images, documents, long text) sent alongside it, and from metadata your application keeps:

```python
from pydantic_ai import Agent, BinaryContent, ToolReturn

agent = Agent('anthropic:%%model=anthropic%%')


@agent.tool_plain
def screenshot(url: str) -> ToolReturn:
    """Take a screenshot of a web page."""
    png = b'\x89PNG...'   # real code: capture the page
    return ToolReturn(
        return_value=f'Captured {url}',
        content=['Here is the screenshot:', BinaryContent(png, media_type='image/png')],
        metadata={'bytes': len(png)},   # stays on the ToolReturnPart, never sent to the model
    )
```

### Choosing the model per step

`SelectModel` picks the model before each model request, from the step number, the messages so far or the usage. It returns a `Model` or a model string:

```python
from pydantic_ai import Agent, ModelSelectionContext
from pydantic_ai.capabilities import SelectModel


def escalate(ctx: ModelSelectionContext[None]) -> str:
    # a cheap model for the first request, a stronger one once tools are in play
    return 'anthropic:%%model=anthropic-small%%' if ctx.run_step == 1 else 'anthropic:%%model=anthropic-flagship%%'


agent = Agent(capabilities=[SelectModel(escalate)])
```

### Templated instructions

`TemplateStr` renders a Handlebars template against `deps` for each run, which keeps per-user text out of code and works inside an `AgentSpec`:

```python
from dataclasses import dataclass

from pydantic_ai import Agent, TemplateStr


@dataclass
class UserProfile:
    name: str
    language: str


agent = Agent(
    'openai:%%model=openai-small%%',
    deps_type=UserProfile,
    instructions=TemplateStr('The user is {{name}}. Always respond in {{language}}.'),
)
result = agent.run_sync('What is a decorator?', deps=UserProfile(name='Alice', language='French'))
```

### Bundling tools and instructions

`Capability` groups instructions and tools under one id without subclassing. With `defer_loading=True` the model sees only its description until it asks to load it, which keeps large tool collections out of the context window:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability

finance = Capability(
    id='finance-tools',
    description='Stock prices and portfolio returns.',
    instructions='Quote prices in USD with two decimal places.',
    defer_loading=True,
)


@finance.tool
def get_stock_price(ctx: RunContext[object], ticker: str) -> float:
    """Return the latest closing price."""
    return 150.0


agent = Agent('openai:%%model=openai%%', capabilities=[finance])
```

## Revision history

The framework's own history is in the [upstream changelog](https://pydantic.dev/docs/ai/changelog/); this guide's history is in git.

| Date | Change |
|---|---|
| 2026-10-25 | Re-audited against pydantic-ai %%version=pydantic-ai%%: corrected the instructions-versus-history note, `UnsupportedSettingWarning` scope, the start of V2 deprecation warnings (1.95.0), `agent.iter()` hook behaviour and the `OnlineEvaluation` module; added `Conversation` to the run reference and fuller exception tree. |
| 2026-10-21 | Rewritten. The 4,700-line class reference (much of it private internals and repeats of topic pages) replaced by an [Agent reference](#agent-reference) and an [API map](#api-map) that links to the topic pages. Corrected the `end_strategy` default (`'graceful'`), the `AgentRetries` keys, the removed `MCPServer*` classes and the native-tool support table; fixed the tutorial examples (`instructions=` per run, `stream_output()` for structured streaming, `min_length=`, imports). |
| 2026-08-21 | Class & API Reference section added, consolidating the earlier per-class pages. |
| 2025-11 | First edition (1.x). |

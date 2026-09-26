---
title: "PydanticAI: 10 Source-Verified Class Deep Dives (2.46.0)"
description: "Runnable, source-verified code examples for FallbackModel, FunctionToolset, TextOutput/ToolOutput/NativeOutput/PromptedOutput, Embedder/EmbeddingResult, MCPToolset (FastMCP 4), load_mcp_toolsets, CombinedToolset, StructuredDict, ModelSettings lesser-known fields, and AdvisorTool — verified against pydantic-ai 2.46.0."
framework: pydanticai
language: python
sidebar:
  order: 128
---

# 10 Source-Verified Class Deep Dives — v2.46.0

Verified against **pydantic-ai 2.46.0** (installed package, sources read directly).
Modules consulted: `pydantic_ai/models/fallback.py`, `pydantic_ai/toolsets/function.py`,
`pydantic_ai/output.py`, `pydantic_ai/embeddings/base.py`, `pydantic_ai/embeddings/__init__.py`,
`pydantic_ai/mcp.py`, `pydantic_ai/toolsets/combined.py`, `pydantic_ai/settings.py`,
`pydantic_ai/native_tools/__init__.py`.

This page covers classes that are **new since 2.43.0** or were only **thinly documented** in
earlier pages. Cross-references to the earlier series appear at the end.

```bash
pip install "pydantic-ai==2.46.0"
python -c "import pydantic_ai; print(pydantic_ai.__version__)"
#> 2.46.0
```

---

## 1. `FallbackModel` — multi-model resilience with response-level routing

**Module:** `pydantic_ai.models.fallback`

`FallbackModel` wraps several models and tries them in sequence when the primary fails. What
changed in 2.2.x–2.5.x (now stable in 2.46.0) is the `fallback_on` parameter: it now accepts
**response handlers** in addition to exception types, letting you fall through to the next model
based on the *content* of a response — not just whether the call crashed.

### Constructor signature (from source)

```python
def __init__(
    self,
    default_model: Model | KnownModelName | str,
    *fallback_models: Model | KnownModelName | str,
    fallback_on: FallbackOn = (ModelAPIError,),
): ...
```

`FallbackOn` is a flexible union:
- A **tuple of exception types** — `(ModelAPIError,)` is the default.
- A **single callable** — auto-detected as an exception handler (first param ≠ `ModelResponse`)
  or a response handler (first param typed as `ModelResponse`).
- A **sequence mixing** exception types and callables.

> **Gotcha — tuples mean exception types only**: any `tuple` passed to `fallback_on` is treated as
> a tuple of exception types. Pass handlers as a bare callable or inside a **list** —
> `fallback_on=(my_handler,)` either raises `TypeError` or silently ignores the handler.

### Example 1 — basic provider failover

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.exceptions import ModelAPIError

model = FallbackModel(
    'openai:gpt-5',
    'anthropic:claude-sonnet-5',   # first fallback
    'google:gemini-2.5-pro',       # second fallback
)

agent = Agent(model)

async def main():
    result = await agent.run('Hello from the most available model')
    print(result.output)

asyncio.run(main())
```

### Example 2 — exception handler as callable

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models.fallback import FallbackModel


def should_fallback(exc: Exception) -> bool:
    """Fall through on any API error or rate limit."""
    if isinstance(exc, ModelAPIError):
        status = getattr(exc, 'status_code', None)
        return status is None or status in {429, 500, 502, 503, 504}
    return False


model = FallbackModel(
    'openai:gpt-5',
    'anthropic:claude-sonnet-5',
    fallback_on=[should_fallback],  # list, not tuple
)

agent = Agent(model, output_type=str)

async def main():
    result = await agent.run('Which provider answered?')
    print(result.output)

asyncio.run(main())
```

### Example 3 — response handler: skip empty answers

A **response handler** receives a `ModelResponse` (not an exception) and returns `True` to
trigger fallback. The handler is identified automatically: if its first parameter is typed as
`ModelResponse`, it is treated as a response handler.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.fallback import FallbackModel


def response_too_short(response: ModelResponse) -> bool:
    """Fall through if the model returned fewer than 10 meaningful characters."""
    from pydantic_ai.messages import TextPart
    text = ''.join(
        p.content for p in response.parts if isinstance(p, TextPart)
    )
    return len(text.strip()) < 10


model = FallbackModel(
    'openai:gpt-5',
    'anthropic:claude-sonnet-5',
    fallback_on=response_too_short,  # bare callable (or a list)
)

agent = Agent(model, output_type=str)

async def main():
    result = await agent.run('Describe the Eiffel Tower in two sentences.')
    print(result.output)

asyncio.run(main())
```

### Example 4 — async handlers

Both exception handlers and response handlers can be `async`:

```python
import asyncio
import httpx
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models.fallback import FallbackModel


async def is_provider_down(exc: Exception) -> bool:
    """Probe a health endpoint before deciding to fall through."""
    if not isinstance(exc, ModelAPIError):
        return False
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get('https://status.openai.com/api/v2/status.json', timeout=2)
            data = r.json()
            return data.get('status', {}).get('indicator') != 'none'
    except Exception:
        return True


model = FallbackModel(
    'openai:gpt-5',
    'anthropic:claude-sonnet-5',
    fallback_on=[is_provider_down],
)
```

### Example 5 — mixing exception types and callables

```python
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.fallback import FallbackModel


def low_confidence(response: ModelResponse) -> bool:
    from pydantic_ai.messages import TextPart
    text = ''.join(p.content for p in response.parts if isinstance(p, TextPart))
    return 'I am not sure' in text or 'I cannot' in text


model = FallbackModel(
    'openai:gpt-4o-mini',
    'openai:gpt-5',
    'anthropic:claude-opus-5',
    fallback_on=[ModelAPIError, low_confidence],  # list mixing type and handler
)
```

> **Gotcha — `FallbackExceptionGroup`**: when *all* models fail, `FallbackModel` raises a
> `FallbackExceptionGroup` that contains the individual exceptions from each model attempt.
> Catch it with `except* ModelAPIError` (Python 3.11+) or iterate `.exceptions` on the group.

---

## 2. `FunctionToolset` — building reusable, configurable tool groups

**Module:** `pydantic_ai.toolsets.function`

`FunctionToolset` is the backbone of every `@agent.tool` decorator. Using it explicitly lets you:
- share a toolset across multiple agents,
- set defaults for `max_retries`, `timeout`, `sequential`, and `requires_approval` per group,
- attach `instructions` — extra system-prompt text injected whenever the toolset is active,
- hide tools from the model until discovered via tool search or `load_capability` with `defer_loading=True`.

### Constructor (from source, condensed)

```python
FunctionToolset(
    tools: Sequence[Tool | ToolFuncEither] = [],
    *,
    max_retries: int | None = None,
    timeout: float | None = None,
    docstring_format: DocstringFormat = 'auto',
    require_parameter_descriptions: bool = False,
    schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
    strict: bool | None = None,
    sequential: bool = False,
    requires_approval: bool = False,
    metadata: dict[str, Any] | None = None,
    defer_loading: bool = False,
    include_return_schema: bool | None = None,
    id: str | None = None,
    instructions: AgentInstructions = None,
)
```

### Example 1 — shared toolset across two agents

```python
import asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai import FunctionToolset


weather_tools = FunctionToolset()


@weather_tools.tool
def get_temperature(ctx: RunContext[None], city: str) -> str:
    """Return the current temperature for a city."""
    return f'22°C in {city}'


@weather_tools.tool
def get_humidity(ctx: RunContext[None], city: str) -> str:
    """Return the current humidity for a city."""
    return f'65% humidity in {city}'


# Reuse across agents
agent_en = Agent('openai:gpt-5', toolsets=[weather_tools], system_prompt='Answer in English.')
agent_fr = Agent('openai:gpt-5', toolsets=[weather_tools], system_prompt='Répondez en français.')


async def main():
    r = await agent_en.run('What is the weather in London?')
    print(r.output)

asyncio.run(main())
```

### Example 2 — `instructions` parameter (extra system-prompt text)

```python
from pydantic_ai import Agent, FunctionToolset, RunContext

db_tools = FunctionToolset(
    instructions='You have access to a read-only database. Never modify data. Always cite the table you queried.',
)


@db_tools.tool
def query_users(ctx: RunContext[None], name: str) -> list[dict]:
    """Query the users table by name."""
    return [{'id': 1, 'name': name, 'email': f'{name.lower()}@example.com'}]


agent = Agent('openai:gpt-5', toolsets=[db_tools])
result = agent.run_sync('Find users named Alice')
print(result.output)
```

### Example 3 — `sequential=True` for serialized execution

`sequential=True` makes every tool in the toolset a **barrier**: if the model requests multiple
tools in the same response, each tool in this toolset runs alone — it won't overlap with other
tool calls in that step. This prevents intra-step races when tools share mutable state such as a
database transaction or file handle. It does **not** protect against races from concurrent
`agent.run()` calls; use an external lock for that.

```python
from pydantic_ai import Agent, FunctionToolset, RunContext

# Each tool runs alone (no overlap) if the model calls multiple tools in one step.
counter_state = {'value': 0}

serial_tools = FunctionToolset(sequential=True)


@serial_tools.tool
def increment(ctx: RunContext[None], amount: int) -> int:
    """Add amount to the shared counter and return the new value."""
    counter_state['value'] += amount
    return counter_state['value']


@serial_tools.tool
def reset(ctx: RunContext[None]) -> int:
    """Reset the shared counter to zero."""
    counter_state['value'] = 0
    return counter_state['value']


agent = Agent('openai:gpt-5', toolsets=[serial_tools])
result = agent.run_sync('Increment the counter by 3, then by 7')
print(result.output)
```

### Example 4 — `requires_approval=True` for HITL at the toolset level

When `requires_approval=True` the entire toolset is wrapped with approval gating — every call
returns a `DeferredToolRequests` unless explicitly approved.

```python
from pydantic_ai import Agent, DeferredToolRequests, FunctionToolset, RunContext

sensitive_tools = FunctionToolset(requires_approval=True)


@sensitive_tools.tool
def send_email(ctx: RunContext[None], to: str, body: str) -> str:
    """Send an email to a recipient."""
    return f'Email sent to {to}: {body}'


agent = Agent('openai:gpt-5', toolsets=[sensitive_tools], output_type=[str, DeferredToolRequests])

result = agent.run_sync('Send a welcome email to alice@example.com')

if isinstance(result.output, DeferredToolRequests):
    print('Pending approvals:', [c.tool_name for c in result.output.approvals])
    # build DeferredToolResults and pass to next run to approve
```

### Example 5 — `defer_loading=True` to hide tools until discovered

`defer_loading=True` **hides** the toolset's tools from the model until discovered. The
`ToolSearch` capability is auto-injected into every agent, so discovery works without extra
setup; pass `ToolSearch(...)` explicitly only to configure it (strategy, `max_results`). On providers that support native search (Anthropic
BM25/regex, OpenAI Responses), the provider exposes hidden tools once discovered; elsewhere a
local `search_tools` function is added to the model's tool list.

```python
from pydantic_ai import Agent, FunctionToolset, RunContext
from pydantic_ai.capabilities import ToolSearch

# Tools are HIDDEN from the model until ToolSearch reveals them.
hidden_tools = FunctionToolset(defer_loading=True, id='hidden-ops')


@hidden_tools.tool
def secret_lookup(ctx: RunContext[None], query: str) -> str:
    """Look up internal records (only available after tool discovery)."""
    return f'Internal result for {query}'


# ToolSearch() is auto-injected; passing it explicitly lets you configure the strategy.
agent = Agent(
    'openai:gpt-5',
    toolsets=[hidden_tools],
    capabilities=[ToolSearch()],  # default: native search where supported, else local keywords
)
```

---

## 3. Output strategy quartet — `TextOutput`, `ToolOutput`, `NativeOutput`, `PromptedOutput`

**Module:** `pydantic_ai.output`

These four marker classes control *how* the agent asks the model for structured output. They wrap
the same `output_type` value but tell the runtime which extraction strategy to use.

| Class | Mechanism | When to use |
|---|---|---|
| `ToolOutput` | Tool call | Explicit tool-call strategy. Best compatibility across providers. |
| `NativeOutput` | Provider JSON mode | JSON schema enforced by provider. Faster, fewer tokens. |
| `PromptedOutput` | Prompt injection | Works on any model, even those without native JSON mode. |
| `TextOutput` | Plain text + function | Transform free text into a Python value. |

### `ToolOutput` — named tool with custom description

```python
from pydantic import BaseModel
from pydantic_ai import Agent, ToolOutput


class Report(BaseModel):
    """A structured business report."""
    title: str
    summary: str
    risk_level: int  # 1-10


agent = Agent(
    'openai:gpt-5',
    output_type=ToolOutput(
        Report,
        name='submit_report',
        description='Submit the final structured report. Always call this at the end.',
    ),
)

result = agent.run_sync('Analyse the Q3 revenue drop and produce a report.')
print(result.output.title)
```

### `NativeOutput` — provider JSON mode, union of types

```python
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput


class Fruit(BaseModel):
    name: str
    color: str


class Vehicle(BaseModel):
    name: str
    wheels: int


agent = Agent(
    'openai:gpt-5',
    output_type=NativeOutput(
        [Fruit, Vehicle],
        name='fruit_or_vehicle',       # machine-safe name (letters/digits/underscores only)
        description='Classify the item as a fruit or vehicle.',
        strict=False,  # strict=True rejects undiscriminated root unions on OpenAI
    ),
)

result = agent.run_sync('What is a banana?')
print(repr(result.output))  # Fruit(name='banana', color='yellow')
```

### `PromptedOutput` — works on every model

```python
from pydantic import BaseModel
from pydantic_ai import Agent, PromptedOutput


class Summary(BaseModel):
    one_line: str
    key_points: list[str]


agent = Agent(
    'ollama:llama3.2',  # PromptedOutput works on any model, including self-hosted Ollama
    output_type=PromptedOutput(
        Summary,
        template='Return valid JSON matching this schema:\n{schema}\n\nNow respond:',
    ),
)

result = agent.run_sync('Summarise the French Revolution.')
print(result.output.one_line)
```

### `TextOutput` — transform the model's text response

```python
from pydantic_ai import Agent, TextOutput


def parse_csv(text: str) -> list[list[str]]:
    return [row.split(',') for row in text.strip().splitlines()]


agent = Agent(
    'openai:gpt-5',
    output_type=TextOutput(parse_csv),
    system_prompt='Output exactly one CSV row per country, columns: name,capital,population.',
)

result = agent.run_sync('Give me 3 European countries.')
for row in result.output:
    print(row)
# ['France', 'Paris', '67000000']
# ...
```

> **Streaming note**: `TextOutput` functions are **not applied** during `stream_text()`. Use
> `stream_output()` to receive the transformed value incrementally.

### Choosing between the four

```python
# Rule of thumb
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput, PromptedOutput, TextOutput, ToolOutput


class Answer(BaseModel):
    value: str
    confidence: float


# Default — agent selects output mode based on model profile (may be tool, native, or prompted)
agent_default = Agent('openai:gpt-5', output_type=Answer)

# Explicit NativeOutput — faster on OpenAI/Google, requires provider support
agent_native = Agent('openai:gpt-5', output_type=NativeOutput(Answer, strict=True))

# PromptedOutput — use on models without JSON mode
agent_prompted = Agent('ollama:llama3.2', output_type=PromptedOutput(Answer))

# TextOutput — when you just need a string transformation
def shout(text: str) -> str:
    return text.upper()


agent_text = Agent('openai:gpt-5', output_type=TextOutput(shout))
```

---

## 4. `Embedder` + `EmbeddingResult` — high-level embedding generation

**Module:** `pydantic_ai.embeddings`

The `Embedder` class (added in 2.2.x) is the high-level interface for generating vector
embeddings. It supports OpenAI, Cohere, Google, Bedrock, and sentence-transformers, with
optional OpenTelemetry instrumentation.

### Constructor signature

```python
Embedder(
    model: EmbeddingModel | KnownEmbeddingModelName | str,
    *,
    settings: EmbeddingSettings | None = None,
    instrument: InstrumentationSettings | bool | None = None,
)
```

### Example 1 — simple single-input embedding

```python
import asyncio
from pydantic_ai import Embedder


async def main():
    embedder = Embedder('openai:text-embedding-3-small')
    result = await embedder.embed_query('What is machine learning?')
    print(f'Vector dimension: {len(result.embeddings[0])}')
    print(f'First 5 values: {result.embeddings[0][:5]}')
    print(f'Model: {result.model_name}')
    print(f'Tokens used: {result.usage.total_tokens}')


asyncio.run(main())
```

### Example 2 — batch embedding for a document corpus

```python
import asyncio
from pydantic_ai import Embedder


async def main():
    embedder = Embedder('openai:text-embedding-3-large')

    documents = [
        'Pydantic AI is a framework for building type-safe AI agents.',
        'FastAPI is a web framework for building APIs with Python.',
        'LangChain is an orchestration framework for LLM applications.',
    ]

    result = await embedder.embed_documents(documents)
    print(f'Embedded {len(result.embeddings)} documents')
    print(f'Dimension: {len(result.embeddings[0])}')

    # Compute cosine similarity between first two
    import math
    a, b = result.embeddings[0], result.embeddings[1]
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x**2 for x in a))
    mag_b = math.sqrt(sum(x**2 for x in b))
    similarity = dot / (mag_a * mag_b)
    print(f'Similarity between doc[0] and doc[1]: {similarity:.4f}')


asyncio.run(main())
```

### Example 3 — Cohere with document vs. query input types

Different embedding providers distinguish between query and document embedding. `embed_query`
and `embed_documents` pass the right `input_type` automatically:

```python
import asyncio
from pydantic_ai import Embedder


async def main():
    # Cohere recommends different input_type for queries vs. indexed docs
    embedder = Embedder('cohere:embed-v4.0')

    query_vec = await embedder.embed_query('What causes inflation?')
    doc_vec = await embedder.embed_documents(['Central banks control money supply.'])

    print('Query embedding shape:', len(query_vec.embeddings[0]))
    print('Doc embedding shape:', len(doc_vec.embeddings[0]))


asyncio.run(main())
```

### Example 4 — `EmbeddingResult` attributes

```python
import asyncio
from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingResult


async def main() -> None:
    embedder = Embedder('openai:text-embedding-3-small')
    result: EmbeddingResult = await embedder.embed_query('Hello')

    # Core attributes
    print(result.embeddings)      # list[list[float]]
    print(result.model_name)       # 'text-embedding-3-small'
    print(result.usage.total_tokens)

    # Normalise vectors manually for cosine similarity (no built-in helper)
    def normalise(v: list[float]) -> list[float]:
        mag = sum(x**2 for x in v) ** 0.5
        return [x / mag for x in v]

    norms = [normalise(list(e)) for e in result.embeddings]
    print(f'Norm of first vector: {sum(x**2 for x in norms[0])**0.5:.4f}')  # ~1.0


asyncio.run(main())
```

### Example 5 — `Embedder.instrument_all` for global tracing

```python
from pydantic_ai import Embedder
import logfire

logfire.configure()
Embedder.instrument_all()  # traces all Embedder calls in this process

embedder = Embedder('openai:text-embedding-3-small')
# All calls are now instrumented with Logfire spans automatically
```

---

## 5. `MCPToolset` v2.46.0 — FastMCP 4 integration

**Module:** `pydantic_ai.mcp`

`MCPToolset` was rewritten around the **FastMCP Client** in 2.5.x and stabilised through 2.46.0.
The previous per-transport classes (`MCPServerStdio`, `MCPServerSSE`, `MCPServerStreamableHTTP`)
are gone — pass anything FastMCP can build a transport from: a URL, a script path, a `FastMCP`
server instance, or a pre-built `fastmcp.Client`.

### Constructor signature (condensed from source)

```python
MCPToolset(
    client_or_server,            # URL, path, FastMCP server, or Client
    *,
    tool_error_behavior: Literal['retry', 'error', 'failed'] = 'retry',
    max_retries: int | None = None,
    prefer_tasks: bool = True,
    cache_tools: bool = True,    # default True — fetched once, reused
    ...
)
```

### Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `tool_error_behavior` | `'retry'` | `'retry'` → `ModelRetry`; `'error'` → propagate `ToolError`; `'failed'` → `ToolFailed` |
| `prefer_tasks` | `True` | Prefer task-augmented (SEP-1686) execution when supported |
| `cache_tools` | `True` | Cache tool list across `get_tools()` calls; set `False` if server changes tools mid-session |

### Example 1 — HTTP (Streamable HTTP / SSE) server

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset


async def main():
    toolset = MCPToolset('http://localhost:8000/mcp')
    agent = Agent('openai:gpt-5', toolsets=[toolset])

    async with agent:
        result = await agent.run('List available files')
    print(result.output)


asyncio.run(main())
```

### Example 2 — stdio MCP server (script path)

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset


async def main():
    # Pydantic AI launches and manages the subprocess
    toolset = MCPToolset('path/to/my_mcp_server.py')
    agent = Agent('openai:gpt-5', toolsets=[toolset])

    async with agent:
        result = await agent.run('Run the data pipeline')
    print(result.output)


asyncio.run(main())
```

### Example 3 — in-process FastMCP server for testing

```python
import asyncio
from fastmcp import FastMCP
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset


mcp_server = FastMCP('test-server')


@mcp_server.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def main():
    toolset = MCPToolset(mcp_server)
    agent = Agent('openai:gpt-5', toolsets=[toolset])

    async with agent:
        result = await agent.run('What is 7 + 13?')
    print(result.output)  # 20


asyncio.run(main())
```

### Example 4 — `tool_error_behavior` and allowlisting via `FilteredToolset`

`MCPToolset` itself has no `allowed_tools` parameter. To restrict which tools the model can call,
wrap the toolset with `FilteredToolset`:

```python
import asyncio
from pydantic_ai import Agent, FilteredToolset
from pydantic_ai.mcp import MCPToolset


async def main():
    raw = MCPToolset(
        'http://localhost:8000/mcp',
        tool_error_behavior='failed',   # model sees the error, no retry
        cache_tools=False,              # fetch fresh list each get_tools() call
    )
    # Allow only safe read-only tools; deny write/delete tools
    toolset = FilteredToolset(raw, lambda _ctx, tool: tool.name in {'read_file', 'list_dir'})
    agent = Agent('openai:gpt-5', toolsets=[toolset])

    async with agent:
        result = await agent.run('Read README.md')
    print(result.output)


asyncio.run(main())
```

### Example 5 — pre-built `fastmcp.Client` for full control (OAuth, custom headers)

```python
import asyncio
from fastmcp.client import Client
from fastmcp.client.transports import StreamableHttpTransport
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset


async def main():
    transport = StreamableHttpTransport(
        'https://api.example.com/mcp',
        headers={'Authorization': 'Bearer my-token'},
    )
    client = Client(transport)
    toolset = MCPToolset(client)

    agent = Agent('openai:gpt-5', toolsets=[toolset])
    async with agent:
        result = await agent.run('Analyse the latest report')
    print(result.output)


asyncio.run(main())
```

---

## 6. `load_mcp_toolsets` — JSON config-based MCP loading

**Module:** `pydantic_ai.mcp`

`load_mcp_toolsets` reads the same `mcpServers` JSON shape used by Claude Desktop, Claude Code,
and Cursor. Each server entry produces one `MCPToolset`, wrapped in a `PrefixedToolset` using the
server name as prefix to prevent tool name collisions.

### Signature (from source)

```python
def load_mcp_toolsets(config_path: str | Path) -> list[AbstractToolset[Any]]:
    ...
```

### Config file format

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "python",
      "args": ["-m", "mcp_filesystem_server", "/allowed/path"],
      "env": {
        "LOG_LEVEL": "${LOG_LEVEL:-info}",
        "API_KEY": "${FILESYSTEM_API_KEY}"
      }
    },
    "search": {
      "url": "http://localhost:9000/mcp"
    }
  }
}
```

Environment variables use `${VAR}` (required) or `${VAR:-default}` (with fallback) syntax.

### Example 1 — loading toolsets from config

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import load_mcp_toolsets


async def main():
    toolsets = load_mcp_toolsets('.claude/mcp.json')
    agent = Agent('openai:gpt-5', toolsets=toolsets)

    async with agent:
        result = await agent.run('List files in /allowed/path then search for Python projects')
    print(result.output)


asyncio.run(main())
```

### Example 2 — tool names after prefix wrapping

Each server's tools are prefixed with the server name. A tool named `read_file` from the
`filesystem` server becomes `filesystem_read_file`:

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import load_mcp_toolsets


async def describe_available_tools():
    toolsets = load_mcp_toolsets('mcp_config.json')
    # Pass to an agent and ask it to list what it can do
    agent = Agent('openai:gpt-5', toolsets=toolsets, output_type=str)
    async with agent:
        result = await agent.run('List all the tools you have access to, grouped by prefix.')
    print(result.output)


asyncio.run(describe_available_tools())
```

### Example 3 — combining with programmatic toolsets

```python
import asyncio
from pydantic_ai import Agent, FunctionToolset, RunContext
from pydantic_ai.mcp import load_mcp_toolsets


custom = FunctionToolset()


@custom.tool
def get_current_user(ctx: RunContext[None]) -> str:
    """Return the current authenticated user."""
    return 'alice@example.com'


async def main():
    mcp_toolsets = load_mcp_toolsets('mcp_config.json')
    agent = Agent('openai:gpt-5', toolsets=[custom, *mcp_toolsets])
    async with agent:
        result = await agent.run('Who am I and what files can I access?')
    print(result.output)


asyncio.run(main())
```

---

## 7. `CombinedToolset` — composing toolsets without name collisions

**Module:** `pydantic_ai.toolsets.combined`

`CombinedToolset` merges multiple toolsets into one. Unlike just passing `toolsets=[a, b]` to
`Agent`, using it explicitly lets you wrap the combined set in further toolset decorators
(e.g., `FilteredToolset`, `PrefixedToolset`).

### Signature (from source)

```python
@dataclass
class CombinedToolset(AbstractToolset[AgentDepsT]):
    toolsets: Sequence[AbstractToolset[AgentDepsT]]
```

### Name conflict resolution

When two toolsets expose the same tool name, `CombinedToolset` raises `UserError` when the tools
are listed. Wrapping each toolset in a `PrefixedToolset` renames its tools so the names no longer
clash. The right pattern is to prefix before combining:

```python
from pydantic_ai import Agent, FunctionToolset, PrefixedToolset, CombinedToolset, RunContext


db_tools = FunctionToolset()
api_tools = FunctionToolset()


@db_tools.tool
def search(ctx: RunContext[None], query: str) -> list[str]:
    """Search the database."""
    return [f'db:{query}:result1', f'db:{query}:result2']


@api_tools.tool
def search(ctx: RunContext[None], query: str) -> list[str]:
    """Search the external API."""
    return [f'api:{query}:result1']


# Prefix before combining to avoid the name clash
combined = CombinedToolset([
    PrefixedToolset(db_tools, prefix='db'),
    PrefixedToolset(api_tools, prefix='api'),
])

agent = Agent('openai:gpt-5', toolsets=[combined])
result = agent.run_sync('Search both database and API for "pydantic"')
print(result.output)
```

### Example 2 — wrapping combined with FilteredToolset

```python
from pydantic_ai import (
    Agent, CombinedToolset, FilteredToolset, FunctionToolset, RunContext
)
from pydantic_ai.tools import ToolDefinition


read_tools = FunctionToolset()
write_tools = FunctionToolset()


@read_tools.tool
def read_document(ctx: RunContext[None], doc_id: str) -> str:
    """Read a document by ID."""
    return f'Content of {doc_id}'


@write_tools.tool
def create_document(ctx: RunContext[None], title: str, body: str) -> str:
    """Create a new document."""
    return f'Created: {title}'


combined = CombinedToolset([read_tools, write_tools])

# Explicit allowlist — safer than a deny-list: only listed tools are exposed
TRUSTED_TOOLS = {'read_document'}

def is_read_tool(_ctx: RunContext[str], tool_def: ToolDefinition) -> bool:
    return tool_def.name in TRUSTED_TOOLS


safe_toolset = FilteredToolset(combined, filter_func=is_read_tool)

agent = Agent('openai:gpt-5', toolsets=[safe_toolset], deps_type=str)
result = agent.run_sync('Read doc abc-123', deps='untrusted_user')
print(result.output)
```

### Example 3 — `for_run` lifecycle

`CombinedToolset` delegates `for_run` and `for_run_step` to each child toolset, enabling
dynamic toolsets inside the combined set to refresh per-step:

```python
import asyncio
from pydantic_ai import Agent, CombinedToolset, FunctionToolset, RunContext


static_tools = FunctionToolset()
dynamic_tools = FunctionToolset()


@static_tools.tool
def static_helper(ctx: RunContext[None]) -> str:
    """Always available."""
    return 'static result'


@dynamic_tools.tool
def step_counter(ctx: RunContext[None]) -> str:
    """Track how many model steps have occurred."""
    return f'Currently on run step {ctx.run_step}'


combined = CombinedToolset([static_tools, dynamic_tools])
agent = Agent('openai:gpt-5', toolsets=[combined])

result = agent.run_sync('Use both tools and tell me the results.')
print(result.output)
```

---

## 8. `StructuredDict` — raw JSON schema as output type

**Module:** `pydantic_ai.output` (factory function, not a class)

`StructuredDict` is a factory that creates a `dict[str, Any]` subclass with an attached JSON
schema. Use it when you have an existing JSON schema (from an API, a form spec, an external
system) and don't want to define a Pydantic model.

### Signature (from source)

```python
def StructuredDict(
    json_schema: JsonSchemaValue,
    name: str | None = None,
    description: str | None = None,
) -> type[dict[str, Any]]:
    ...
```

### Example 1 — basic usage

```python
from pydantic_ai import Agent, StructuredDict

person_schema = {
    'type': 'object',
    'title': 'Person',
    'properties': {
        'name': {'type': 'string'},
        'age': {'type': 'integer', 'minimum': 0},
        'email': {'type': 'string', 'format': 'email'},
    },
    'required': ['name', 'age'],
}

PersonDict = StructuredDict(person_schema, name='PersonDict')  # explicit name; falls back to schema title otherwise
agent = Agent('openai:gpt-5', output_type=PersonDict)

result = agent.run_sync('Create a person called John who is 30 years old')
print(result.output)          # {'name': 'John', 'age': 30}
print(type(result.output))    # <class 'dict'>
```

### Example 2 — nested schemas with `$defs`

`StructuredDict` automatically inlines `$defs` references to work around a Pydantic limitation:

```python
from pydantic_ai import Agent, StructuredDict

order_schema = {
    'type': 'object',
    'title': 'Order',
    '$defs': {
        'Item': {
            'type': 'object',
            'properties': {
                'sku': {'type': 'string'},
                'qty': {'type': 'integer'},
            },
            'required': ['sku', 'qty'],
        }
    },
    'properties': {
        'order_id': {'type': 'string'},
        'items': {
            'type': 'array',
            'items': {'$ref': '#/$defs/Item'},
        },
    },
    'required': ['order_id', 'items'],
}

OrderDict = StructuredDict(order_schema)
agent = Agent('openai:gpt-5', output_type=OrderDict)

result = agent.run_sync('Create an order ORD-001 with 2 units of SKU-A and 5 of SKU-B')
print(result.output)
# {'order_id': 'ORD-001', 'items': [{'sku': 'SKU-A', 'qty': 2}, {'sku': 'SKU-B', 'qty': 5}]}
```

### Example 3 — using alongside Pydantic models

```python
from pydantic import BaseModel
from pydantic_ai import Agent, StructuredDict


class FallbackAnswer(BaseModel):
    text: str
    confidence: float


dynamic_schema = {
    'type': 'object',
    'properties': {
        'answer': {'type': 'string'},
        'sources': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['answer'],
}

DynamicAnswer = StructuredDict(dynamic_schema, name='DynamicAnswer')

# Union of a Pydantic model and a StructuredDict
agent = Agent('openai:gpt-5', output_type=[FallbackAnswer, DynamicAnswer])
result = agent.run_sync('What is the capital of France?')
print(result.output)
```

### Example 4 — loading schema from an external JSON file

```python
import json
from pathlib import Path
from pydantic_ai import Agent, StructuredDict


schema = json.loads(Path('schemas/invoice.json').read_text())
InvoiceDict = StructuredDict(schema, description='Extract invoice data from the text.')
agent = Agent('openai:gpt-5', output_type=InvoiceDict)

invoice_text = """
Invoice #INV-2026-001
Date: 2026-09-21
Total: $1,234.56
Vendor: Acme Corp
"""

result = agent.run_sync(f'Extract invoice data:\n{invoice_text}')
print(result.output)
```

---

## 9. `ModelSettings` lesser-known fields

**Module:** `pydantic_ai.settings`

`ModelSettings` is a `TypedDict` with `total=False` (all fields optional). The comprehensive
guide covers `max_tokens`, `temperature`, and `top_p`; this section documents the fields that
are less commonly used but highly useful in production.

### `parallel_tool_calls` — control multi-tool fan-out

When `False`, the model issues at most one tool call per response instead of fanning out in
parallel. This limits concurrent side-effects — it does **not** enforce call ordering between
different tools; the model still chooses which tool to call first. Providers that do not support
this setting silently ignore it, so verify your target provider's adapter before relying on it.

```python
from pydantic_ai import Agent, FunctionToolset, RunContext
from pydantic_ai.settings import ModelSettings


sequential_settings = ModelSettings(parallel_tool_calls=False)
tools = FunctionToolset()


@tools.tool
def step_a(ctx: RunContext[None]) -> str:
    """First step."""
    return 'A done'


@tools.tool
def step_b(ctx: RunContext[None]) -> str:
    """Second step (model decides call order; one-at-a-time fan-out)."""
    return 'B done'


agent = Agent('openai:gpt-5', toolsets=[tools], model_settings=sequential_settings)
result = agent.run_sync('Run step A then step B in order.')
print(result.output)
```

### `seed` — reproducible outputs for testing

```python
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

agent = Agent(
    'openai:gpt-5',
    model_settings=ModelSettings(seed=42, temperature=0.0),
)

r1 = agent.run_sync('Roll a dice')
r2 = agent.run_sync('Roll a dice')
# r1.output == r2.output with seed + temperature=0 (best effort, not guaranteed)
print(r1.output, r2.output)
```

### `stop_sequences` — early termination tokens

```python
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

agent = Agent(
    'openai:gpt-5',
    model_settings=ModelSettings(stop_sequences=['###END###', '<STOP>']),
)

result = agent.run_sync(
    'Write a short poem. End with ###END### when done.\n'
    'Roses are red,'
)
print(result.output)
```

### `extra_headers` — inject custom HTTP headers

```python
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

# Pass a request ID for distributed tracing
agent = Agent(
    'openai:gpt-5',
    model_settings=ModelSettings(
        extra_headers={
            'X-Request-ID': 'req-2026-09-21-abc',
            'X-User-Tier': 'enterprise',
        }
    ),
)
```

### `presence_penalty` + `frequency_penalty` — reduce repetition

```python
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

agent = Agent(
    'openai:gpt-5',
    model_settings=ModelSettings(
        presence_penalty=0.6,   # penalise reusing tokens that appeared before
        frequency_penalty=0.4,  # penalise reusing frequently appearing tokens
    ),
)

result = agent.run_sync(
    'Write a 200-word product description for an ergonomic chair. '
    'Avoid repeating phrases.'
)
print(result.output)
```

### `logit_bias` — steer token probabilities

```python
import tiktoken
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings


def token_ids(model: str, words: list[str]) -> dict[str, int]:
    enc = tiktoken.encoding_for_model(model)  # use the exact model's tokenizer
    bias: dict[str, int] = {}
    for word in words:
        for tok in enc.encode(word):
            bias[str(tok)] = 100  # strongly encourage
    return bias


# Force the model to favour certain words
agent = Agent(
    'openai:gpt-5',
    model_settings=ModelSettings(
        logit_bias=token_ids('gpt-5', ['Python', 'pydantic', 'type']),
    ),
)

result = agent.run_sync('Name three benefits of this AI framework.')
print(result.output)
```

### `extra_body` — pass provider-specific fields not in `ModelSettings`

```python
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

# Example: OpenAI "store: true" for model distillation
agent = Agent(
    'openai:gpt-5',
    model_settings=ModelSettings(
        extra_body={'store': True, 'metadata': {'project': 'finetune-v2'}},
    ),
)
```

---

## 10. `AdvisorTool` — native Anthropic executor/advisor pattern

**Module:** `pydantic_ai.native_tools`

`AdvisorTool` is a **native tool** (not a Python function tool) that lets a fast executor model
consult a more powerful advisor model mid-generation. Anthropic introduced this as a first-party
API primitive; OpenRouter exposes a compatible subset.

This is different from chaining agents in your Python code: the executor model decides *when* to
call the advisor without your code being involved — the round-trip happens at the provider layer.

### Signature (from source, condensed)

```python
@dataclass(kw_only=True)
class AdvisorTool(AbstractNativeTool):
    model: AdvisorModelName
    max_uses: int | None = None              # Anthropic only; OpenRouter ignores
    max_tokens: int | None = None            # min 1024; maps to max_completion_tokens on OpenRouter
    caching: Literal['5m', '1h'] | None = None  # Anthropic only; ephemeral context cache TTL
    kind: str = 'advisor'                    # fixed sentinel
```

### Example 1 — fast executor with powerful advisor

```python
import asyncio
from pydantic_ai import Agent, AdvisorTool
from pydantic_ai.capabilities import NativeTool


advisor = AdvisorTool(
    model='claude-opus-5',   # Advisor: powerful, slow
    max_uses=3,              # consult at most 3 times per request
    max_tokens=1024,
)

agent = Agent(
    'anthropic:claude-haiku-4-5-20251001',  # Executor: fast, cheap
    capabilities=[NativeTool(advisor)],     # wrap in NativeTool to register as a capability
)


async def main():
    result = await agent.run(
        'Explain the proof of Fermat\'s Last Theorem at a high level, '
        'then give one paragraph suitable for a general audience.'
    )
    print(result.output)


asyncio.run(main())
```

### Example 2 — ephemeral `caching` to reduce repeated advisor context cost

The `caching` field tells Anthropic to cache the advisor's context for `'5m'` or `'1h'`.
Warm requests skip re-tokenising the advisor's system context, cutting latency and cost on
repeated queries. OpenRouter ignores this field.

```python
import asyncio
from pydantic_ai import Agent, AdvisorTool
from pydantic_ai.capabilities import NativeTool


advisor = AdvisorTool(
    model='claude-opus-5',
    max_tokens=1024,
    caching='5m',  # cache advisor context for 5 minutes (Anthropic only)
)

agent = Agent('anthropic:claude-haiku-4-5-20251001', capabilities=[NativeTool(advisor)])


async def main():
    # First call cold; subsequent calls within 5 min benefit from cached advisor context.
    for question in ['Derive the quadratic formula.', 'What is the binomial theorem?']:
        result = await agent.run(question)
        print(result.output)


asyncio.run(main())
```

### Example 3 — OpenRouter gateway

```python
from pydantic_ai import Agent, AdvisorTool
from pydantic_ai.capabilities import NativeTool

advisor = AdvisorTool(
    model='anthropic/claude-opus-4.8',  # OpenRouter catalog slug (no prefix for advisor model)
    max_tokens=1024,
)

agent = Agent(
    'openrouter:anthropic/claude-haiku-4-5',  # executor via OpenRouter prefix
    capabilities=[NativeTool(advisor)],
)

result = agent.run_sync('What is the Riemann hypothesis and why does it matter?')
print(result.output)
```

### `AdvisorTool` vs. multi-agent orchestration

| | `AdvisorTool` | Multi-agent (Python) |
|---|---|---|
| Round-trip location | Provider API | Your Python code |
| Executor control | None (provider decides) | Full |
| Latency | Lower (no Python hop) | Higher |
| Observability | Provider traces only | Full traces in your code |
| When to use | Speed-critical with trusted provider | Full control needed |

---

## Quick-reference: what each deep-dive series covers

| Series | Classes |
|--------|---------|
| [Aug 2026 (2.33.0)](./pydantic_ai_class_examples_2026_08/) | Agent, RunContext, UsageLimits, ToolReturn, DeferredToolRequests/Results, CachePoint, PrefixedToolset/FilteredToolset/RenamedToolset, WebSearchTool, ModelRetry/UnexpectedModelBehavior |
| [v2.36.0](./pydantic_ai_class_deep_dives_v2_36/) | AgentRun, AgentRunResult, StreamedRunResult, ModelSettings, Tool, ToolDefinition, RunUsage/RequestUsage, ConcurrencyLimiter, MCPToolset, ApprovalRequiredToolset/DynamicToolset |
| [v2.40.0](./pydantic_ai_class_deep_dives_v2_40/) | RealtimeSession, RealtimeModelSettings/TurnDetection, AgentRealtime, Capability, Hooks, TemplateStr, ExternalToolset, RetryConfig/HTTPX2TenacityTransport, DuckDuckGoSearchTool, ImageGenerationSubagentTool |
| [v2.43.0](./pydantic_ai_class_deep_dives_v2_43/) | ToolFailed, RunCancelled, ToolSelector, ToolOrOutput, ServiceTier/ThinkingLevel, AgentStream/StreamedRunResult, SkipModelRequest/SkipToolValidation/SkipToolExecution, ToolDefinition.sequential, OutputContext, ApprovalRequired |
| **v2.46.0 (this page)** | **FallbackModel, FunctionToolset, TextOutput/ToolOutput/NativeOutput/PromptedOutput, Embedder/EmbeddingResult, MCPToolset v2.46.0/FastMCP 4, load_mcp_toolsets, CombinedToolset, StructuredDict, ModelSettings lesser-known fields, AdvisorTool** |

---
title: "PydanticAI — Class Deep Dives Vol. 8"
description: "Source-verified deep dives into 10 PydanticAI classes: ToolOutput/NativeOutput/PromptedOutput/TextOutput/StructuredDict (output mode toolkit), ApprovalRequiredToolset (HITL guard), DeferredLoadingToolset (lazy discovery), Embedder/EmbeddingModel/EmbeddingResult/EmbeddingSettings (vector embeddings), web_fetch_tool/WebFetchLocalTool (SSRF-safe web fetching), PrefectAgent/TaskConfig (durable Prefect integration), ImageGenerationSubagentTool (subagent image gen), ConcurrencyLimitedModel/AbstractConcurrencyLimiter (model-level rate limiting), InstructionPart/AgentInstructions (instructions architecture)."
sidebar:
  label: "Class deep dives (Vol. 8)"
  order: 28
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 1.105.0** source installed directly from PyPI. Class signatures, field names, and behaviour match the installed package at `1.105.0`.
</Aside>

Ten class groups from the `pydantic_ai` 1.105.0 source covering: the four output-mode marker classes plus `StructuredDict` that give you surgical control over how the model delivers structured output; the toolset-level `ApprovalRequiredToolset` for human-in-the-loop guards; `DeferredLoadingToolset` for lazy tool discovery in large toolsets; the complete vector embeddings API (`Embedder`, `EmbeddingModel`, `EmbeddingResult`, `EmbeddingSettings`); the SSRF-protected `web_fetch_tool` for URL-to-markdown conversion; `PrefectAgent` and `TaskConfig` for Prefect durable execution; `ImageGenerationSubagentTool` for model-agnostic image generation fallback; `ConcurrencyLimitedModel` with `AbstractConcurrencyLimiter` for per-model HTTP rate limiting; and the `InstructionPart` / `AgentInstructions` architecture that powers static-vs-dynamic instruction caching.

---

## 1. `ToolOutput` + `NativeOutput` + `PromptedOutput` + `TextOutput` + `StructuredDict` — Output Mode Toolkit

**Module:** `pydantic_ai.output`  
**Import:** `from pydantic_ai import ToolOutput, NativeOutput, PromptedOutput, TextOutput, StructuredDict`

These five classes give you explicit, per-output-type control over the structured output mechanism PydanticAI uses. Without them you rely on the model profile's `default_structured_output_mode`. With them you can mix modes in a single `output_type` union and precisely tune every call.

### Class signatures

```python
@dataclass(init=False)
class ToolOutput(Generic[OutputDataT]):
    def __init__(
        self,
        type_: OutputTypeOrFunction[OutputDataT],
        *,
        name: str | None = None,
        description: str | None = None,
        max_retries: int | None = None,
        strict: bool | None = None,
    ) -> None: ...

@dataclass(init=False)
class NativeOutput(Generic[OutputDataT]):
    def __init__(
        self,
        outputs: OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
        template: str | Literal[False] | None = None,
    ) -> None: ...

@dataclass(init=False)
class PromptedOutput(Generic[OutputDataT]):
    def __init__(
        self,
        outputs: OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]],
        *,
        name: str | None = None,
        description: str | None = None,
        template: str | Literal[False] | None = None,
    ) -> None: ...

@dataclass
class TextOutput(Generic[OutputDataT]):
    output: TextOutputFunc[OutputDataT]

def StructuredDict(
    json_schema: JsonSchemaValue,
    name: str | None = None,
    description: str | None = None,
) -> type[JsonSchemaValue]: ...
```

### Output modes at a glance

| Mode | Class | How structured output is delivered |
|---|---|---|
| `tool` | `ToolOutput` | Model calls a synthetic tool whose args carry the output payload |
| `native` | `NativeOutput` | Model's native structured output API (e.g. OpenAI `response_format`) |
| `prompted` | `PromptedOutput` | JSON schema embedded in a prompt template |
| `text` | `TextOutput` | Plain text passed through a post-processing function |
| _schema-free dict_ | `StructuredDict` | `dict[str, Any]` with a JSON schema attached; no Pydantic model required |

### `ToolOutput` — per-tool retry and strict mode

```python
from pydantic import BaseModel
from pydantic_ai import Agent, ToolOutput


class Fruit(BaseModel):
    name: str
    color: str


class Vehicle(BaseModel):
    name: str
    wheels: int


# Name each output tool explicitly; override max_retries per output type.
agent = Agent(
    'openai:gpt-4o',
    output_type=[
        ToolOutput(Fruit, name='return_fruit', description='Call this for any fruit.', max_retries=3),
        ToolOutput(Vehicle, name='return_vehicle', max_retries=1),
    ],
)

result = agent.run_sync('What is a banana?')
print(repr(result.output))  # Fruit(name='banana', color='yellow')
```

`max_retries` on `ToolOutput` overrides the agent-level output-side retry budget for that specific output type. `strict=True` enables JSON schema strict mode on the tool (OpenAI-specific).

### `NativeOutput` — single and union native structured outputs

```python
from pydantic_ai import Agent, NativeOutput

# Single native output — most efficient on models that support it
agent_single = Agent(
    'openai:gpt-4o',
    output_type=NativeOutput(Fruit, strict=True),
)

# Union of types with a shared name and description
agent_union = Agent(
    'openai:gpt-4o',
    output_type=NativeOutput(
        [Fruit, Vehicle],
        name='Fruit or vehicle',
        description='Return a fruit or a wheeled vehicle.',
    ),
)

# Suppress the schema-in-prompt behaviour for models that don't need it
agent_no_prompt = Agent(
    'google:gemini-2.0-flash',
    output_type=NativeOutput(Fruit, template=False),
)
```

The `template` parameter controls whether the JSON schema is also injected as a prompt. `None` (default) lets the model profile decide. `False` suppresses it entirely. A custom string with `{schema}` injects the schema verbatim.

### `PromptedOutput` — maximum model compatibility

```python
from pydantic_ai import Agent, PromptedOutput

# Works with any model that can follow JSON-in-text instructions
agent = Agent(
    'anthropic:claude-3-5-haiku-latest',
    output_type=PromptedOutput(
        [Fruit, Vehicle],
        template='Reply only with valid JSON matching this schema:\n{schema}',
    ),
)

result = agent.run_sync('What is a Ford Explorer?')
print(repr(result.output))  # Vehicle(name='Ford Explorer', wheels=4)
```

### `TextOutput` — processing plain text responses

```python
from pydantic_ai import Agent, TextOutput


def word_count(text: str) -> dict[str, int]:
    return {'words': len(text.split()), 'chars': len(text)}


agent = Agent(
    'openai:gpt-4o-mini',
    output_type=TextOutput(word_count),
)

result = agent.run_sync('Write a haiku about Python.')
print(result.output)  # {'words': 17, 'chars': 82}
```

`TextOutput` accepts both sync and async functions, and optionally a `RunContext` as the first argument.

### `StructuredDict` — schema-defined dict output without a Pydantic model

```python
from pydantic_ai import Agent, StructuredDict

person_schema = {
    'type': 'object',
    'properties': {
        'name':   {'type': 'string', 'description': 'Full name'},
        'age':    {'type': 'integer', 'minimum': 0},
        'skills': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['name', 'age'],
}

PersonDict = StructuredDict(person_schema, name='Person', description='A person record')

agent = Agent('openai:gpt-4o-mini', output_type=PersonDict)
result = agent.run_sync('Create a senior Python developer named Ada.')
print(result.output)
# {'name': 'Ada Lovelace', 'age': 38, 'skills': ['Python', 'Machine Learning', 'Type Systems']}
```

`StructuredDict` returns a `type` (a `dict[str, Any]` subclass with `__get_pydantic_json_schema__` attached). The schema is inlined before being attached — any `$defs` are resolved so the schema is self-contained. Recursive `$ref`s are not supported.

---

## 2. `ApprovalRequiredToolset` — Toolset-Level HITL Approval Guard

**Module:** `pydantic_ai.toolsets.approval_required`  
**Import:** `from pydantic_ai.toolsets import ApprovalRequiredToolset`

`ApprovalRequiredToolset` wraps any existing toolset with a gate function. When the model requests a tool call that the gate marks as requiring approval, the call is suspended by raising `ApprovalRequired`. The outer orchestrator can then review the pending call and resume it by re-running with `deferred_tool_results=DeferredToolResults(...)`.

### Class signature

```python
@dataclass
class ApprovalRequiredToolset(WrapperToolset[AgentDepsT]):
    approval_required_func: Callable[
        [RunContext[AgentDepsT], ToolDefinition, dict[str, Any]], bool
    ] = lambda ctx, tool_def, tool_args: True
```

The default gate (`lambda ... True`) requires approval for **every** tool call in the wrapped toolset. Pass a custom callable to select only dangerous operations.

### Basic usage — approve by tool name

```python
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.toolsets import ApprovalRequiredToolset


def read_file(path: str) -> str:
    """Read a file from disk."""
    with open(path) as f:
        return f.read()


def delete_file(path: str) -> str:
    """Delete a file from disk."""
    import os
    os.remove(path)
    return f'Deleted {path}'


base_toolset = FunctionToolset([read_file, delete_file])

# Only require approval for destructive operations
guarded_toolset = ApprovalRequiredToolset(
    base_toolset,
    approval_required_func=lambda ctx, tool_def, args: tool_def.name in {'delete_file'},
)

agent = Agent('openai:gpt-4o', toolsets=[guarded_toolset])
```

### Handling `ApprovalRequired` in the run loop

```python
import asyncio
from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults


async def run_with_approval(agent: Agent, prompt: str) -> str:
    async with agent.run_stream_events(prompt) as agent_run:
        async for event in agent_run:
            pass

        # Check if a tool call is pending approval
        if isinstance(agent_run.result, DeferredToolRequests):
            pending = agent_run.result
            for call in pending.tool_calls:
                # Present to human
                approved = input(
                    f'Approve {call.tool_name}({call.args})? [y/n] '
                ) == 'y'
                if not approved:
                    raise RuntimeError('Tool call rejected by operator')

            # Resume with approval
            result = await agent.run(
                prompt,
                message_history=pending.all_messages(),
                deferred_tool_results=DeferredToolResults(pending.tool_calls),
            )
            return result.output

    return agent_run.result.output
```

### Approval gate with argument inspection

```python
SENSITIVE_PATHS = {'/etc/passwd', '/etc/shadow', '/root/.ssh'}


def path_approval_gate(ctx, tool_def, args):
    """Require approval when accessing sensitive filesystem paths."""
    if tool_def.name in {'read_file', 'delete_file'}:
        path = args.get('path', '')
        return path in SENSITIVE_PATHS or path.startswith('/etc/')
    return False  # No approval needed for other tools


guarded = ApprovalRequiredToolset(base_toolset, approval_required_func=path_approval_gate)
```

---

## 3. `DeferredLoadingToolset` — Lazy Tool Discovery

**Module:** `pydantic_ai.toolsets.deferred_loading`  
**Import:** `from pydantic_ai.toolsets import DeferredLoadingToolset`

When an agent has hundreds of tools it would be wasteful to list all of them in every request. `DeferredLoadingToolset` marks tools for lazy discovery: they are hidden from the model until the `ToolSearch` capability discovers them by name or description.

### Class signature

```python
@dataclass(init=False)
class DeferredLoadingToolset(PreparedToolset[AgentDepsT]):
    tool_names: frozenset[str] | None = None
    # tool_names=None  →  defer ALL tools in the wrapped toolset
    # tool_names={...} →  defer only the named tools; others remain visible

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        *,
        tool_names: frozenset[str] | None = None,
    ) -> None: ...
```

### Hiding a large toolset until searched

```python
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.toolsets import DeferredLoadingToolset
from pydantic_ai.capabilities import ToolSearch


# 200 specialised tools — too many to list in every request
def analyse_sentiment(text: str) -> str: ...
def translate_text(text: str, target_language: str) -> str: ...
def summarise_document(url: str) -> str: ...
# ... 197 more


big_toolset = FunctionToolset([analyse_sentiment, translate_text, summarise_document])

# Defer ALL tools — none are listed unless the model searches for them
deferred = DeferredLoadingToolset(big_toolset)

agent = Agent(
    'openai:gpt-4o',
    toolsets=[deferred],
    capabilities=[ToolSearch()],  # Required: enables tool discovery
)

# The model's first message sees an empty tool list.
# If it calls tool_search('translate'), translate_text becomes visible.
result = agent.run_sync('Translate "Hello" into Japanese.')
print(result.output)
```

### Partially deferring a mixed toolset

```python
from pydantic_ai import FunctionToolset
from pydantic_ai.toolsets import DeferredLoadingToolset


def get_weather(city: str) -> str: ...    # Always visible
def get_tide_tables(port: str) -> str: ...  # Deferred until searched
def get_uv_index(city: str) -> str: ...    # Deferred until searched


toolset = FunctionToolset([get_weather, get_tide_tables, get_uv_index])

# Only defer the specialist tools; keep get_weather always visible
partially_deferred = DeferredLoadingToolset(
    toolset,
    tool_names=frozenset({'get_tide_tables', 'get_uv_index'}),
)
```

### Combining with `FilteredToolset`

```python
from pydantic_ai.toolsets import DeferredLoadingToolset, FilteredToolset


def role_filter(ctx, tool_def):
    """Only expose admin tools to admin users."""
    if tool_def.name.startswith('admin_'):
        return ctx.deps.get('role') == 'admin'
    return True


# First filter by role, then defer the remainder for tool search
combined = DeferredLoadingToolset(
    FilteredToolset(big_toolset, filter_func=role_filter)
)
```

---

## 4. `Embedder` + `EmbeddingModel` + `EmbeddingResult` + `EmbeddingSettings` — Vector Embeddings API

**Module:** `pydantic_ai.embeddings`  
**Import:** `from pydantic_ai import Embedder`  
**Import (advanced):** `from pydantic_ai.embeddings import EmbeddingModel, EmbeddingResult, EmbeddingSettings, KnownEmbeddingModelName`

PydanticAI ships a complete, provider-agnostic embeddings API. The high-level `Embedder` class mirrors `Agent` in design: it wraps any `EmbeddingModel`, provides sync/async APIs, and supports OpenTelemetry instrumentation and model overrides for testing.

### `Embedder` signature

```python
@dataclass(init=False)
class Embedder:
    def __init__(
        self,
        model: EmbeddingModel | KnownEmbeddingModelName | str,
        *,
        settings: EmbeddingSettings | None = None,
        defer_model_check: bool = True,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> None: ...

    async def embed_query(self, query: str | Sequence[str], *, settings: EmbeddingSettings | None = None) -> EmbeddingResult: ...
    async def embed_documents(self, documents: str | Sequence[str], *, settings: EmbeddingSettings | None = None) -> EmbeddingResult: ...
    async def embed(self, inputs: str | Sequence[str], *, input_type: Literal['query', 'document'], settings: EmbeddingSettings | None = None) -> EmbeddingResult: ...

    # Synchronous variants (run event loop internally)
    def embed_query_sync(self, query: str | Sequence[str], ...) -> EmbeddingResult: ...
    def embed_documents_sync(self, documents: str | Sequence[str], ...) -> EmbeddingResult: ...

    @contextmanager
    def override(self, *, model: ...) -> Iterator[None]: ...

    @staticmethod
    def instrument_all(instrument: InstrumentationSettings | bool = True) -> None: ...
```

### Supported providers via `KnownEmbeddingModelName`

| Provider prefix | Example model |
|---|---|
| `openai:` | `openai:text-embedding-3-small` |
| `google:` / `google-cloud:` | `google:gemini-embedding-001` |
| `cohere:` | `cohere:embed-v4.0` |
| `bedrock:` | `bedrock:amazon.titan-embed-text-v2:0` |
| `voyageai:` | `voyageai:voyage-3.5` |
| `sentence-transformers:` | Any HuggingFace model name |

### Embedding queries and documents

```python
import asyncio
from pydantic_ai import Embedder


async def main() -> None:
    embedder = Embedder('openai:text-embedding-3-small')

    # Embed a single query
    query_result = await embedder.embed_query('What is machine learning?')
    print(len(query_result.embeddings[0]))  # 1536

    # Embed a batch of documents
    docs = [
        'PydanticAI is a Python agent framework.',
        'Embeddings map text to vector space.',
        'RAG combines retrieval with generation.',
    ]
    doc_result = await embedder.embed_documents(docs)
    print(f'Embedded {len(doc_result.embeddings)} documents')

    # Lookup by original text
    vec = doc_result['Embeddings map text to vector space.']
    print(f'First 5 dims: {vec[:5]}')

    # Check usage and cost
    print(f'Tokens used: {doc_result.usage.input_tokens}')
    cost = doc_result.cost()
    print(f'Estimated cost: ${cost.total_price:.6f}')


asyncio.run(main())
```

### `EmbeddingResult` — the result object

```python
@dataclass
class EmbeddingResult:
    embeddings: Sequence[Sequence[float]]  # One vector per input
    inputs: Sequence[str]                  # Original input texts
    input_type: Literal['query', 'document']
    model_name: str
    provider_name: str
    timestamp: datetime
    usage: RequestUsage                    # Token counts
    provider_details: dict[str, Any] | None
    provider_response_id: str | None

    def __getitem__(self, item: int | str) -> Sequence[float]: ...  # Index by position or text
    def cost(self) -> PriceCalculation: ...  # Requires genai-prices
```

### `EmbeddingSettings` — cross-provider configuration

```python
from pydantic_ai.embeddings import EmbeddingSettings

# Common settings (TypedDict, all keys optional)
settings: EmbeddingSettings = {
    'dimensions': 256,      # Truncate output dimensions (OpenAI, Google, Cohere, etc.)
    'truncate': True,       # Truncate inputs exceeding context length (Cohere, VoyageAI)
    'extra_headers': {'X-Custom': 'value'},
}

embedder = Embedder('openai:text-embedding-3-large', settings=settings)

# Per-call override
result = await embedder.embed_query('short query', settings={'dimensions': 512})
```

### Full RAG pipeline example

```python
import asyncio
import numpy as np
from pydantic_ai import Agent, Embedder


class VectorStore:
    def __init__(self):
        self.texts: list[str] = []
        self.vectors: list[list[float]] = []

    def add(self, text: str, vector: list[float]) -> None:
        self.texts.append(text)
        self.vectors.append(vector)

    def search(self, query_vec: list[float], top_k: int = 3) -> list[str]:
        if not self.vectors:
            return []
        q = np.array(query_vec)
        scores = [
            float(np.dot(q, np.array(v)) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-10))
            for v in self.vectors
        ]
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [self.texts[i] for i in top_indices]


async def build_rag_agent():
    embedder = Embedder('openai:text-embedding-3-small')
    store = VectorStore()

    # Index documents
    docs = [
        'PydanticAI is a production-ready Python agent framework.',
        'Pydantic validates data using Python type annotations.',
        'FastAPI is an async web framework built on Pydantic.',
    ]
    result = await embedder.embed_documents(docs)
    for text, vec in zip(result.inputs, result.embeddings):
        store.add(text, list(vec))

    # Answer a question using retrieved context
    agent = Agent('openai:gpt-4o-mini')

    async def answer(question: str) -> str:
        q_result = await embedder.embed_query(question)
        context = store.search(list(q_result.embeddings[0]))
        prompt = f"Context:\n{chr(10).join(context)}\n\nQuestion: {question}"
        result = await agent.run(prompt)
        return result.output

    return answer


async def main():
    answer = await build_rag_agent()
    print(await answer('What is PydanticAI?'))


asyncio.run(main())
```

### Testing with `TestEmbeddingModel`

```python
from pydantic_ai.embeddings import TestEmbeddingModel

# Returns deterministic unit vectors for reproducible tests
test_model = TestEmbeddingModel(dimensions=4)
embedder = Embedder(test_model)

with embedder.override(model=test_model):
    result = embedder.embed_query_sync('test input')
    assert len(result.embeddings[0]) == 4
```

---

## 5. `web_fetch_tool` + `WebFetchLocalTool` + `WebFetchResult` — SSRF-Protected Web Fetching

**Module:** `pydantic_ai.common_tools.web_fetch`  
**Import:** `from pydantic_ai.common_tools.web_fetch import web_fetch_tool, WebFetchResult`  
**Extra:** `pip install "pydantic-ai-slim[web-fetch]"`

`web_fetch_tool` builds a `Tool` that wraps `WebFetchLocalTool`. It fetches any URL, prefers `text/markdown` responses (sent via `Accept` header), falls back to HTML→markdown conversion via `markdownify`, and returns `BinaryContent` for non-text responses (PDFs, images). All HTTP requests go through `safe_download` to block SSRF attacks by default.

### `web_fetch_tool` factory signature

```python
def web_fetch_tool(
    *,
    max_content_length: int | None = 50_000,
    allow_local_urls: bool = False,
    timeout: int = 30,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    headers: dict[str, str] | None = None,
) -> Tool[Any]: ...
```

### `WebFetchLocalTool` fields

```python
@dataclass
class WebFetchLocalTool:
    max_content_length: int | None     # Chars returned, None = unlimited
    allow_local_urls: bool             # Permit 192.168.x.x / 127.0.0.1 etc.
    timeout: int                       # Request timeout in seconds
    allowed_domains: list[str] | None  # Allowlist (exact hostname match)
    blocked_domains: list[str] | None  # Blocklist (exact hostname match)
    headers: dict[str, str] | None     # Extra request headers
```

### `WebFetchResult` — the return type for text content

```python
class WebFetchResult(TypedDict):
    url: str      # URL that was fetched
    title: str    # Page <title> (empty string if absent)
    content: str  # Page content as markdown
```

For binary content types (PDF, images, audio…) the tool returns a `BinaryContent` so the model can process it natively via vision or document APIs.

### Basic usage

```python
from pydantic_ai import Agent
from pydantic_ai.common_tools.web_fetch import web_fetch_tool

agent = Agent(
    'openai:gpt-4o',
    tools=[
        web_fetch_tool(
            max_content_length=20_000,  # ~5,000 tokens
            timeout=15,
        )
    ],
)

result = agent.run_sync('Summarise the content at https://docs.pydantic.dev/latest/')
print(result.output)
```

### Domain allow/block lists

```python
# Only allow fetching from your own documentation domain
internal_fetch = web_fetch_tool(
    allowed_domains=['docs.example.com', 'api.example.com'],
    max_content_length=10_000,
)

# Block known ad/tracking networks
safe_fetch = web_fetch_tool(
    blocked_domains=['ads.example.com', 'tracking.example.net'],
)
```

### Allowing local URLs in integration tests

```python
# Safe for tests; never enable allow_local_urls=True in production
test_fetch = web_fetch_tool(
    allow_local_urls=True,
    timeout=5,
)
```

### Customising the Accept header

```python
# Force plain-text responses; disable markdown preference
html_fetch = web_fetch_tool(
    headers={'Accept': 'text/html'},
)

# Add authentication for private APIs
auth_fetch = web_fetch_tool(
    headers={
        'Authorization': 'Bearer my-api-token',
        'Accept': 'application/json',
    },
    allowed_domains=['api.internal.example.com'],
    allow_local_urls=True,
)
```

### Handling binary content returned by the tool

When the server returns a non-text MIME type, the tool returns a `BinaryContent` object instead of `WebFetchResult`. This is transparent to the agent — models with vision or document capabilities process it automatically:

```python
# Agent with a PDF-capable model will process binary returns transparently
pdf_agent = Agent(
    'google:gemini-2.0-flash',
    tools=[web_fetch_tool(max_content_length=None)],
)

result = pdf_agent.run_sync(
    'Summarise the executive summary from https://example.com/annual-report.pdf'
)
print(result.output)
```

---

## 6. `PrefectAgent` + `TaskConfig` — Prefect Durable Execution

**Module:** `pydantic_ai.durable_exec.prefect`  
**Import:** `from pydantic_ai.durable_exec.prefect import PrefectAgent`  
**Extra:** `pip install "pydantic-ai-slim[prefect]"`

`PrefectAgent` wraps any `Agent` (or `WrapperAgent`) so that model requests, tool calls, and MCP communications become Prefect tasks. Each task is automatically retried, persisted, and observable via the Prefect UI — providing durable execution without changing your agent code. Temporal and DBOS integrations were covered in earlier volumes; Prefect is the third pillar.

### `PrefectAgent.__init__` signature

```python
class PrefectAgent(WrapperAgent[AgentDepsT, OutputDataT]):
    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        name: str | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        mcp_task_config: TaskConfig | None = None,
        model_task_config: TaskConfig | None = None,
        tool_task_config: TaskConfig | None = None,
        tool_task_config_by_name: dict[str, TaskConfig | None] | None = None,
        event_stream_handler_task_config: TaskConfig | None = None,
        prefectify_toolset_func: ... | None = None,
    ) -> None: ...
```

The `name` parameter is **required** — it becomes the Prefect flow name prefix and identifies the agent's tasks in the Prefect UI. Agents without a `name` will raise `UserError`.

### `TaskConfig` — Prefect task options

```python
class TaskConfig(TypedDict, total=False):
    retries: int                                  # Max task-level retries
    retry_delay_seconds: float | list[float]      # Fixed or custom backoff list
    timeout_seconds: float                        # Per-task time limit
    cache_policy: CachePolicy                     # Prefect cache policy
    persist_result: bool                          # Persist task result to storage
    result_storage: ResultStorage                 # Block slug or storage object
    log_prints: bool                              # Route print() to Prefect logs
```

### Wrapping an agent for Prefect

```python
from pydantic_ai import Agent
from pydantic_ai.durable_exec.prefect import PrefectAgent
from prefect import flow


base_agent = Agent(
    'openai:gpt-4o',
    name='research-agent',
    system_prompt='You are a research assistant.',
)

prefect_agent = PrefectAgent(
    base_agent,
    model_task_config={'retries': 3, 'retry_delay_seconds': [1, 5, 30]},
    tool_task_config={'retries': 2, 'timeout_seconds': 30.0},
)


@flow(name='research-flow')
async def run_research(question: str) -> str:
    result = await prefect_agent.run(question)
    return result.output
```

### Per-tool task configuration

```python
from pydantic_ai.durable_exec.prefect import PrefectAgent

# Expensive tools get more retries; fast tools have tighter timeouts
prefect_agent = PrefectAgent(
    base_agent,
    tool_task_config={'retries': 1, 'timeout_seconds': 10.0},
    tool_task_config_by_name={
        'web_search': {'retries': 3, 'timeout_seconds': 60.0},  # Extra retries for flaky API
        'send_email': None,    # Disable task wrapping for idempotent-unsafe tools
    },
)
```

`None` in `tool_task_config_by_name` disables Prefect wrapping for that specific tool — useful for tools where automatic retries could cause duplicate side effects.

### Running inside a Prefect flow with result persistence

```python
import asyncio
from pydantic_ai.durable_exec.prefect import PrefectAgent
from prefect import flow


@flow(name='document-analyser', persist_result=True)
async def analyse_documents(urls: list[str]) -> list[str]:
    summaries = []
    for url in urls:
        result = await prefect_agent.run(f'Summarise the content at {url}')
        summaries.append(result.output)
    return summaries


# Run with result storage for checkpoint recovery
asyncio.run(analyse_documents(['https://example.com/doc1.pdf', 'https://example.com/doc2.pdf']))
```

---

## 7. `ImageGenerationSubagentTool` + `image_generation_tool` — Subagent Image Generation

**Module:** `pydantic_ai.common_tools.image_generation`  
**Import:** `from pydantic_ai.common_tools.image_generation import image_generation_tool`

The `ImageGeneration` capability (covered in Vol. 3) uses the outer agent's model for image generation. `ImageGenerationSubagentTool` takes a different approach: it spins up a **separate subagent** with a dedicated model to generate images, allowing the outer agent to use a text-only model (e.g. `gpt-4o-mini`) while delegating image generation to a capable model (e.g. `openai-responses:gpt-5.4`).

### `ImageGenerationSubagentTool` signature

```python
@dataclass(kw_only=True)
class ImageGenerationSubagentTool:
    model: Model | KnownModelName | str | ImageGenerationFallbackModelFunc
    native_tool: ImageGenerationTool       # e.g. ImageGenerationTool()
    instructions: str = (
        'Generate an image based on the user prompt. '
        'Do not ask clarifying questions.'
    )

    async def __call__(self, ctx: RunContext[Any], prompt: str) -> BinaryImage: ...
```

### `image_generation_tool` factory

```python
def image_generation_tool(
    model: Model | KnownModelName | str | ImageGenerationFallbackModelFunc,
    native_tool: ImageGenerationTool,
    *,
    instructions: str = '...',
) -> Tool[Any]: ...
```

### Basic usage — text model delegates to image model

```python
from pydantic_ai import Agent
from pydantic_ai.common_tools.image_generation import image_generation_tool
from pydantic_ai.native_tools import ImageGenerationTool

# Outer agent uses a cheap text model
agent = Agent(
    'openai:gpt-4o-mini',
    tools=[
        image_generation_tool(
            model='openai-responses:gpt-5.4',      # Image-capable model for subagent
            native_tool=ImageGenerationTool(
                quality='high',
                size='1024x1024',
            ),
        )
    ],
)

result = agent.run_sync(
    'Design a logo for a company called Axiom that specialises in AI data pipelines.'
)
print(type(result.output))  # <class 'pydantic_ai.messages.BinaryImage'>
```

### Dynamic model selection per run

`model` can be a callable that accepts a `RunContext` and returns a `Model` or model name. Use this to route to different image models based on quality tier, user preferences, or cost:

```python
from pydantic_ai.tools import RunContext


def select_image_model(ctx: RunContext) -> str:
    """Choose image model based on the user's subscription tier."""
    tier = ctx.deps.get('tier', 'free')
    return {
        'pro': 'openai-responses:gpt-5.5',
        'standard': 'openai-responses:gpt-5.4',
        'free': 'openai-responses:gpt-5.4',
    }[tier]


agent = Agent(
    'openai:gpt-4o-mini',
    tools=[
        image_generation_tool(
            model=select_image_model,
            native_tool=ImageGenerationTool(),
        )
    ],
)
```

### Custom subagent instructions

```python
from pydantic_ai.common_tools.image_generation import image_generation_tool
from pydantic_ai.native_tools import ImageGenerationTool

# Override subagent instructions for strict brand-guideline compliance
branded_tool = image_generation_tool(
    model='openai-responses:gpt-5.4',
    native_tool=ImageGenerationTool(quality='high', size='1792x1024'),
    instructions=(
        'Generate an image exactly as described. '
        'Use a clean, modern aesthetic. '
        'Do not add text overlays unless explicitly requested.'
    ),
)
```

<Aside type="note">
Do **not** pass a dedicated image-only model name such as `gpt-image-2` directly as the `model` argument — these models cannot run the conversational subagent loop. Use a multimodal LLM that has image generation capabilities instead (e.g. `openai-responses:gpt-5.4`).
</Aside>

---

## 8. `ConcurrencyLimitedModel` + `AbstractConcurrencyLimiter` — Model-Level Rate Limiting

**Module:** `pydantic_ai.models.concurrency`  
**Import:** `from pydantic_ai.models.concurrency import ConcurrencyLimitedModel, limit_model_concurrency`  
**Import (ABC):** `from pydantic_ai.concurrency import AbstractConcurrencyLimiter, ConcurrencyLimiter, ConcurrencyLimit`

<Aside type="note">
**Vol. 1 covered the agent-level `ConcurrencyLimiter`** — which limits how many *agent runs* execute concurrently. This section covers `ConcurrencyLimitedModel`, which limits concurrent **HTTP requests to a model's API endpoint**. The two work at different layers and are often combined.
</Aside>

### `ConcurrencyLimitedModel` signature

```python
@dataclass(init=False)
class ConcurrencyLimitedModel(WrapperModel):
    def __init__(
        self,
        wrapped: Model | KnownModelName,
        limiter: int | ConcurrencyLimit | AbstractConcurrencyLimiter,
    ) -> None: ...
```

`wrapped` can be a model name string; it is resolved via `infer_model()` at construction time.

### `ConcurrencyLimit` — detailed configuration

```python
@dataclass
class ConcurrencyLimit:
    max_running: int           # Hard cap on concurrent model requests
    max_queued: int | None = None  # None = unlimited queue; N = raise ConcurrencyLimitExceeded
```

### `AbstractConcurrencyLimiter` — extension point

```python
class AbstractConcurrencyLimiter(ABC):
    @abstractmethod
    async def acquire(self, source: str) -> None: ...
    @abstractmethod
    def release(self) -> None: ...
```

Subclass this to build Redis-backed distributed rate limiters, token-bucket limiters, or any other strategy.

### Simple integer limit

```python
from pydantic_ai import Agent
from pydantic_ai.models.concurrency import ConcurrencyLimitedModel

# At most 5 concurrent requests to this model
model = ConcurrencyLimitedModel('openai:gpt-4o', limiter=5)
agent = Agent(model)

# Or use the convenience function
from pydantic_ai.models.concurrency import limit_model_concurrency

agent = Agent(limit_model_concurrency('openai:gpt-4o', limiter=5))
```

### Backpressure — reject when the queue is full

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.concurrency import ConcurrencyLimitedModel
from pydantic_ai.concurrency import ConcurrencyLimit
from pydantic_ai.exceptions import ConcurrencyLimitExceeded

model = ConcurrencyLimitedModel(
    'anthropic:claude-3-5-haiku-latest',
    limiter=ConcurrencyLimit(max_running=10, max_queued=50),
)
agent = Agent(model)


async def main():
    try:
        result = await agent.run('Hello')
        return result.output
    except ConcurrencyLimitExceeded:
        # Queue is full; shed this request
        return {'error': 'Service temporarily at capacity'}

asyncio.run(main())
```

### Sharing a limiter across multiple models

```python
from pydantic_ai.concurrency import ConcurrencyLimiter

# All OpenAI models share a single 20-request pool
openai_pool = ConcurrencyLimiter(max_running=20, name='openai-pool')

gpt4o = ConcurrencyLimitedModel('openai:gpt-4o', limiter=openai_pool)
gpt4o_mini = ConcurrencyLimitedModel('openai:gpt-4o-mini', limiter=openai_pool)

agent_heavy = Agent(gpt4o)
agent_light = Agent(gpt4o_mini)

# Together they cannot exceed 20 concurrent API requests
```

### Custom distributed limiter (Redis example)

```python
import asyncio
from pydantic_ai.concurrency import AbstractConcurrencyLimiter


class RedisConcurrencyLimiter(AbstractConcurrencyLimiter):
    """Distributed rate limiter backed by Redis SETNX."""

    def __init__(self, redis_client, key: str, max_running: int):
        self._redis = redis_client
        self._key = key
        self._max = max_running
        self._sem = asyncio.Semaphore(max_running)

    async def acquire(self, source: str) -> None:
        await self._sem.acquire()
        # Also increment Redis counter for cross-process observability
        await self._redis.incr(f'{self._key}:running')

    def release(self) -> None:
        self._sem.release()
        asyncio.get_event_loop().create_task(
            self._redis.decr(f'{self._key}:running')
        )


redis_limiter = RedisConcurrencyLimiter(redis_client, 'openai', max_running=100)
model = ConcurrencyLimitedModel('openai:gpt-4o', limiter=redis_limiter)
```

---

## 9. `InstructionPart` + `AgentInstructions` — Instructions Architecture

**Module:** `pydantic_ai.messages` + `pydantic_ai._instructions`  
**Import:** `from pydantic_ai.messages import InstructionPart`

PydanticAI's instruction system distinguishes between *static* instructions (literal strings known at agent definition time) and *dynamic* instructions (produced by functions or `TemplateStr` at run time). This distinction is surfaced via `InstructionPart` and drives intelligent prompt-caching decisions.

### `InstructionPart` signature

```python
@dataclass(repr=False)
class InstructionPart:
    content: str                        # The instruction text
    dynamic: bool = False               # True = came from a function/template/toolset
    part_kind: Literal['instruction'] = 'instruction'  # Discriminator

    @staticmethod
    def join(parts: Sequence[InstructionPart]) -> str | None: ...
    @staticmethod
    def sorted(parts: Sequence[InstructionPart]) -> list[InstructionPart]: ...
```

`sorted()` places static parts before dynamic ones, enabling Anthropic's prompt-caching to cache the stable prefix while leaving the dynamic suffix uncached.

### `AgentInstructions` type alias

```python
AgentInstructions = (
    TemplateStr[AgentDepsT]
    | str
    | SystemPromptFunc[AgentDepsT]
    | Sequence[TemplateStr[AgentDepsT] | str | SystemPromptFunc[AgentDepsT]]
    | None
)
```

This is the type of the `instructions` parameter on `Agent(instructions=...)`. All four forms normalise into `InstructionPart` objects during the run.

### Static vs dynamic instructions

```python
from datetime import datetime
from pydantic_ai import Agent
from pydantic_ai.messages import InstructionPart


# Static instruction — compiled into the agent at definition time
# → InstructionPart(content='...', dynamic=False)
agent_static = Agent(
    'anthropic:claude-3-5-sonnet-latest',
    instructions='You are a helpful assistant. Always be concise.',
)

# Dynamic instruction via a function
# → InstructionPart(content='...', dynamic=True)
def time_aware_instructions() -> str:
    hour = datetime.now().hour
    tone = 'cheerful' if 6 <= hour < 18 else 'calm and patient'
    return f'You are a {tone} assistant. Today is {datetime.now():%A %B %d}.'


agent_dynamic = Agent(
    'anthropic:claude-3-5-sonnet-latest',
    instructions=time_aware_instructions,
)

# Mixed — static prefix cached, dynamic suffix uncached
agent_mixed = Agent(
    'anthropic:claude-3-5-sonnet-latest',
    instructions=[
        'You are a financial analyst assistant.',          # static
        time_aware_instructions,                           # dynamic
    ],
)
```

### Why `InstructionPart.sorted()` matters for prompt caching

When Anthropic sees the instruction list, static parts are placed first so they form a stable prefix that the provider can cache across requests. Dynamic parts come last and are never cached:

```python
from pydantic_ai.messages import InstructionPart

parts = [
    InstructionPart('Always cite sources.', dynamic=False),
    InstructionPart('Today is Monday.', dynamic=True),
    InstructionPart('Speak formally.', dynamic=False),
]

sorted_parts = InstructionPart.sorted(parts)
# Result: static parts first, then dynamic
# ['Always cite sources.', 'Speak formally.', 'Today is Monday.']
joined = InstructionPart.join(sorted_parts)
print(joined)
```

### Toolset-contributed instructions

Toolsets can return instructions via `get_instructions()`. These are always treated as dynamic:

```python
from pydantic_ai import FunctionToolset
from pydantic_ai.messages import InstructionPart


class WeatherToolset(FunctionToolset):
    async def get_instructions(self, ctx) -> str:
        # Dynamically fetched — always InstructionPart(dynamic=True)
        units = ctx.deps.get('units', 'metric')
        return f'Report temperatures in {units} units.'


# The toolset instruction is merged with agent-level instructions
# and InstructionPart.sorted() ensures static parts come first
agent = Agent(
    'openai:gpt-4o',
    instructions='You are a weather assistant.',  # static
    toolsets=[WeatherToolset()],                  # dynamic additions
)
```

---

## 10. `StructuredDict` + `OutputObjectDefinition` — Schema-First Dict Output

**Module:** `pydantic_ai.output`  
**Import:** `from pydantic_ai import StructuredDict; from pydantic_ai.output import OutputObjectDefinition`

`StructuredDict` (introduced in section 1) deserves a deeper look alongside `OutputObjectDefinition`, which is the normalised representation the output pipeline uses internally whenever structured output is configured.

### `OutputObjectDefinition` — the internal output schema record

```python
@dataclass
class OutputObjectDefinition:
    json_schema: ObjectJsonSchema    # Validated JSON schema dict
    name: str | None = None         # Schema title / output tool name
    description: str | None = None  # Schema description
    strict: bool | None = None      # Enable strict mode (OpenAI)
```

`OutputObjectDefinition` is produced from `ToolOutput`, `NativeOutput`, `PromptedOutput`, and `StructuredDict` by the output pipeline. You rarely construct one directly, but it appears as `OutputContext.object_def` in output hooks.

### Using `StructuredDict` with union output types

```python
from pydantic import BaseModel
from pydantic_ai import Agent, StructuredDict


class WeatherForecast(BaseModel):
    location: str
    temperature_c: float
    conditions: str


# Mix a Pydantic model with a schema-defined dict in a union
raw_schema = {
    'type': 'object',
    'properties': {
        'error': {'type': 'string'},
        'code': {'type': 'integer'},
    },
    'required': ['error', 'code'],
}

ErrorDict = StructuredDict(raw_schema, name='Error', description='An error response')

agent = Agent(
    'openai:gpt-4o-mini',
    output_type=[WeatherForecast, ErrorDict],
)

result = agent.run_sync('What is the weather in a made-up city called Zorbania?')
print(result.output)
# ErrorDict: {'error': 'City not found', 'code': 404}
# or
# WeatherForecast(location='Zorbania', temperature_c=22.0, conditions='sunny')
```

### Reading `OutputObjectDefinition` from an output hook

```python
from pydantic_ai import Agent
from pydantic_ai.output import OutputContext


def log_output_schema(ctx: OutputContext) -> None:
    """Inspect the output schema at output-processing time."""
    if ctx.object_def:
        print(f'Output mode: {ctx.mode}')
        print(f'Schema name: {ctx.object_def.name}')
        print(f'Schema keys: {list(ctx.object_def.json_schema.get("properties", {}).keys())}')
        print(f'Has output function: {ctx.has_function}')


PersonDict = StructuredDict(
    {
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
            'role': {'type': 'string'},
        },
        'required': ['name', 'role'],
    },
    name='PersonRecord',
)

agent = Agent(
    'openai:gpt-4o-mini',
    output_type=PersonDict,
)

# Attach hook to inspect the schema on every run
agent.add_before_output_validate(log_output_schema)
```

### `StructuredDict` with deeply nested schemas

`$defs` are automatically inlined before attachment so the schema is always self-contained:

```python
from pydantic_ai import StructuredDict

# Schema with $defs — these are inlined automatically
address_schema = {
    'type': 'object',
    'properties': {
        'name': {'type': 'string'},
        'address': {'$ref': '#/$defs/Address'},
    },
    '$defs': {
        'Address': {
            'type': 'object',
            'properties': {
                'street': {'type': 'string'},
                'city': {'type': 'string'},
            },
        }
    },
}

# StructuredDict inlines the $defs automatically
CustomerDict = StructuredDict(address_schema, name='Customer')
# The resulting schema has no $defs — Address properties are inlined
```

<Aside type="caution">
`StructuredDict` does **not** support recursive `$ref`s (schemas that reference themselves via `$defs`). Use a Pydantic `BaseModel` with `model_rebuild()` for self-referential schemas.
</Aside>

---

## Cross-references

| Class | Covered in | Notes |
|---|---|---|
| `ToolOutput` + `NativeOutput` + `PromptedOutput` + `TextOutput` | **This volume** | `OutputContext` (the hook parameter) covered in Vol. 7 |
| `StructuredDict` + `OutputObjectDefinition` | **This volume** | `JsonSchemaTransformer` covered in Vol. 6 |
| `ApprovalRequiredToolset` | **This volume** | `ApprovalRequired` exception + `DeferredToolRequests` flow in Vol. 2 |
| `DeferredLoadingToolset` | **This volume** | `ToolSearch` capability in Vol. 2 |
| `Embedder` / `EmbeddingModel` / `EmbeddingResult` / `EmbeddingSettings` | **This volume** | Not covered elsewhere |
| `web_fetch_tool` / `WebFetchLocalTool` | **This volume** | `safe_download` (SSRF primitives) covered in Vol. 5 |
| `PrefectAgent` + `TaskConfig` | **This volume** | `TemporalAgent` in Vol. 6; `DBOSAgent` in Vol. 5 |
| `ImageGenerationSubagentTool` | **This volume** | `ImageGeneration` capability in Vol. 3 |
| `ConcurrencyLimitedModel` | **This volume** | Agent-level `ConcurrencyLimiter` + `limit_model_concurrency` in Vol. 1 |
| `InstructionPart` / `AgentInstructions` | **This volume** | `TemplateStr` covered in Vol. 4; `ProcessHistory` in Vol. 4 |

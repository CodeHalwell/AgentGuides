---
title: "PydanticAI — Class Deep Dives Vol. 15"
description: "Source-verified deep dives into 10 class groups from pydantic-ai 1.107.0: CombinedToolset (multi-toolset merging with conflict detection), HuggingFaceModel + HuggingFaceModelSettings + HuggingFaceStreamedResponse (open-source model inference via HuggingFace Hub), TestEmbeddingModel (deterministic mock embeddings for tests), SentenceTransformerEmbeddingModel + SentenceTransformersEmbeddingSettings (local privacy-preserving embeddings), TemporalRunContext + deserialize_run_context (serializable RunContext across Temporal activity boundaries), PydanticAIWorkflow (Temporal Workflow base class), TemporalDynamicToolset (dynamic toolsets across Temporal workflows), PrefectAgentInputs + DEFAULT_PYDANTIC_AI_CACHE_POLICY (Prefect cache policy for PydanticAI agents), ACIToolset + tool_from_aci (deprecated ACI.dev integration + migration guide), and provider model profiles family (GrokModelProfile, GroqModelProfile, deepseek/qwen/cohere/moonshotai profiles). All verified against pydantic-ai 1.107.0 source."
sidebar:
  label: "Class deep dives (Vol. 15)"
  order: 41
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 1.107.0** source installed directly from PyPI. Class signatures, field names, and behaviour match the installed package at this version.
</Aside>

Ten class groups spanning the toolset composition layer, open-source model integration, local embeddings, Temporal and Prefect durable execution internals, a deprecated ACI.dev extension with migration guidance, and the full family of provider-specific model profile functions: `CombinedToolset` (union of multiple toolsets with name-conflict detection); `HuggingFaceModel` + `HuggingFaceModelSettings` + `HuggingFaceStreamedResponse` (inference against any model on the HuggingFace Hub — DeepSeek-R1, Llama-4, Qwen3); `TestEmbeddingModel` (deterministic mock embeddings for unit tests); `SentenceTransformerEmbeddingModel` + `SentenceTransformersEmbeddingSettings` (on-device embeddings from HuggingFace models via `sentence-transformers`); `TemporalRunContext` + `deserialize_run_context` (the serializable RunContext subclass that crosses Temporal activity boundaries); `PydanticAIWorkflow` (the Temporal Workflow base class for direct `__pydantic_ai_agents__` registration); `TemporalDynamicToolset` (wraps `DynamicToolset` so its toolset factory runs inside a Temporal activity); `PrefectAgentInputs` + `DEFAULT_PYDANTIC_AI_CACHE_POLICY` (Prefect `CachePolicy` that strips non-deterministic `RunContext` fields before hashing); `ACIToolset` + `tool_from_aci` (deprecated ACI.dev bridge with migration path to `Tool.from_schema`); and the provider model-profile function family (`GrokModelProfile`, `GroqModelProfile`, `deepseek_model_profile`, `qwen_model_profile`, `cohere_model_profile`, `moonshotai_model_profile`).

---

## 1. `CombinedToolset` — Multi-Toolset Composition

**Module:** `pydantic_ai.toolsets.combined`  
**Imports:**
```python
from pydantic_ai.toolsets.combined import CombinedToolset
```

`CombinedToolset` is the standard way to present multiple toolsets to an agent as a single unit. It is equivalent to the agent's built-in `toolsets=[a, b, c]` parameter — in fact that parameter creates a `CombinedToolset` internally — but you can also create one explicitly for reuse and testing.

### Constructor

```python
@dataclass
class CombinedToolset(AbstractToolset[AgentDepsT]):
    toolsets: Sequence[AbstractToolset[AgentDepsT]]
```

| Argument | Purpose |
|---|---|
| `toolsets` | Ordered list of toolsets to merge. All are entered as async context managers together. |

### Name-conflict detection

When `get_tools` is called, each child toolset is queried in order. If two toolsets define a tool with the same name, a `UserError` is raised immediately (using the `tool_name_conflict_hint` from the conflicting toolset). This fails fast rather than silently shadowing the second definition.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.combined import CombinedToolset

search_ts = FunctionToolset()
storage_ts = FunctionToolset()

@search_ts.tool
async def search(query: str) -> str:
    return f"results for {query}"

@storage_ts.tool
async def store(key: str, value: str) -> bool:
    return True

combined = CombinedToolset(toolsets=[search_ts, storage_ts])
agent = Agent('openai:gpt-4o', toolsets=[combined])
```

### `for_run` and `for_run_step` lifecycle

`CombinedToolset` delegates lifecycle methods to each child. If a child's `for_run_step` returns a new instance (signalling changed state), the combined toolset replaces only that child's entry, rebuilding itself with `replace(self, toolsets=new_toolsets)`. If all children return themselves unchanged, the same `CombinedToolset` instance is returned — no allocation.

```python
# Combined toolsets work transparently with per-run context
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset

user_ts = FunctionToolset(id='user-tools')
admin_ts = FunctionToolset(id='admin-tools')

@user_ts.tool
async def greet(name: str) -> str:
    return f"Hello, {name}!"

@admin_ts.tool
async def delete_user(user_id: int) -> str:
    return f"Deleted user {user_id}"

combined = CombinedToolset(toolsets=[user_ts, admin_ts])
agent = Agent('openai:gpt-4o', toolsets=[combined])

async def main():
    result = await agent.run('Greet Alice')
    print(result.output)
```

### `apply` and `visit_and_replace`

Both traversal methods delegate to each child in order:

```python
def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
    for toolset in self.toolsets:
        toolset.apply(visitor)

def visit_and_replace(
    self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
) -> AbstractToolset[AgentDepsT]:
    return replace(self, toolsets=[toolset.visit_and_replace(visitor) for toolset in self.toolsets])
```

This lets you use `FilteredToolset.wrap(combined)` or `PrefixedToolset.wrap(combined)` directly — the wrapper sees every leaf toolset through `visit_and_replace`.

### Instructions aggregation

`get_instructions` gathers instructions from every child that returns any, and flattens the list:

```python
async def get_instructions(self, ctx: RunContext[AgentDepsT]) -> list[str | InstructionPart] | None:
    results = await gather(*(ts.get_instructions(ctx) for ts in self.toolsets))
    parts = []
    for r in results:
        if r is not None:
            parts.extend(r if not isinstance(r, (str, InstructionPart)) else [r])
    return parts or None
```

### Nesting combined toolsets

`CombinedToolset` nests freely — you can combine combined toolsets:

```python
from pydantic_ai.toolsets.combined import CombinedToolset
from pydantic_ai.toolsets import FunctionToolset

ts_a, ts_b, ts_c = FunctionToolset(), FunctionToolset(), FunctionToolset()
inner = CombinedToolset(toolsets=[ts_a, ts_b])
outer = CombinedToolset(toolsets=[inner, ts_c])
# outer sees tools from all three
```

<Aside type="tip">
`CombinedToolset` is used internally by `Agent` when you pass `toolsets=[a, b]`. Constructing one explicitly is useful when you want to share a reusable "bundle" across multiple agents, or wrap the bundle with a `FilteredToolset` or `PrefixedToolset`.
</Aside>

---

## 2. `HuggingFaceModel` + `HuggingFaceModelSettings` + `HuggingFaceStreamedResponse`

**Module:** `pydantic_ai.models.huggingface`  
**Imports:**
```python
from pydantic_ai.models.huggingface import (
    HuggingFaceModel,
    HuggingFaceModelSettings,
    HuggingFaceStreamedResponse,
    HuggingFaceModelName,
    LatestHuggingFaceModelNames,
)
```

**Install extra:** `pip install "pydantic-ai-slim[huggingface]"`

`HuggingFaceModel` connects to any model served via the [HuggingFace Inference API](https://huggingface.co/inference-api). Supported named shortcuts include DeepSeek-R1, Llama-4 variants, Qwen3, and QwQ.

### `HuggingFaceModel` constructor

```python
@dataclass(init=False)
class HuggingFaceModel(Model[AsyncInferenceClient]):
    def __init__(
        self,
        model_name: str,
        *,
        provider: Literal['huggingface'] | Provider[AsyncInferenceClient] = 'huggingface',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        ...
```

| Argument | Purpose |
|---|---|
| `model_name` | HuggingFace model ID (e.g. `'deepseek-ai/DeepSeek-R1'`, `'Qwen/Qwen3-32B'`) |
| `provider` | `'huggingface'` (default) or a `Provider[AsyncInferenceClient]` instance |
| `profile` | Override the auto-detected `ModelProfile`. Useful to set `supports_thinking=True` for models that reason. |
| `settings` | Default `HuggingFaceModelSettings` applied to every request |

### Latest model shortcuts

```python
LatestHuggingFaceModelNames = Literal[
    'deepseek-ai/DeepSeek-R1',
    'meta-llama/Llama-3.3-70B-Instruct',
    'meta-llama/Llama-4-Maverick-17B-128E-Instruct',
    'meta-llama/Llama-4-Scout-17B-16E-Instruct',
    'Qwen/QwQ-32B',
    'Qwen/Qwen2.5-72B-Instruct',
    'Qwen/Qwen3-235B-A22B',
    'Qwen/Qwen3-32B',
]
```

### `HuggingFaceModelSettings`

A `TypedDict` subclass of `ModelSettings`. All fields must be prefixed `huggingface_` to allow safe merging:

```python
class HuggingFaceModelSettings(ModelSettings, total=False):
    pass  # placeholder for future huggingface-specific settings
```

Standard `ModelSettings` fields all work — `max_tokens`, `temperature`, `top_p`, `stop_sequences`, `seed`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `logprobs`, `top_logprobs`.

### Basic usage

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.huggingface import HuggingFaceModel

model = HuggingFaceModel('meta-llama/Llama-4-Scout-17B-16E-Instruct')
agent = Agent(model, system_prompt='You are a helpful assistant.')

async def main():
    result = await agent.run('Explain what a transformer model is in 2 sentences.')
    print(result.output)

asyncio.run(main())
```

### Using a custom provider

```python
from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.providers import infer_provider

# Use the HuggingFace provider explicitly for custom base URL
provider = infer_provider('huggingface')
model = HuggingFaceModel('Qwen/Qwen3-32B', provider=provider)
```

### Thinking / reasoning models

DeepSeek-R1 and QwQ emit `<think>...</think>` blocks. Set a profile to capture them as `ThinkingPart`:

```python
from pydantic_ai import Agent
from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.profiles import ModelProfile

model = HuggingFaceModel(
    'deepseek-ai/DeepSeek-R1',
    profile=ModelProfile(
        supports_thinking=True,
        thinking_always_enabled=True,
        thinking_tags=('<think>', '</think>'),
        ignore_streamed_leading_whitespace=True,
    ),
)

agent = Agent(model)

async def main():
    async with agent.run_stream('What is 17 * 23?') as stream:
        async for event in stream:
            print(event)
```

### Tool call handling

The model negotiates `tool_choice` through `HuggingFaceModel._get_tool_choice`. When a single named tool is required, it maps to `ChatCompletionInputToolChoiceClass`. When the provider doesn't support limiting tools via API parameter, `tool_defs` is filtered client-side:

```python
# Force a specific tool
from pydantic_ai import Agent
from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.settings import ModelSettings

model = HuggingFaceModel('Qwen/Qwen2.5-72B-Instruct')

async def get_weather(city: str) -> str:
    return f"Sunny in {city}"

agent = Agent(model, tools=[get_weather])

async def main():
    result = await agent.run('What is the weather in Paris?')
    print(result.output)
```

### Error handling

API errors from `huggingface_hub` are caught by `_map_api_errors` and re-raised as `ModelHTTPError`:

```python
from pydantic_ai import ModelHTTPError
from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai import Agent

model = HuggingFaceModel('nonexistent/model-xyz')
agent = Agent(model)

async def main():
    try:
        result = await agent.run('Hello')
    except ModelHTTPError as e:
        print(f"HTTP {e.status_code}: {e.model_name}")
```

### `HuggingFaceStreamedResponse`

```python
@dataclass
class HuggingFaceStreamedResponse(StreamedResponse):
    _model_name: str
    _model_profile: ModelProfile
    _response: PeekableAsyncStream[ChatCompletionStreamOutput, ...]
    _provider_name: str
    _provider_url: str
    _provider_timestamp: datetime | None = None
    _timestamp: datetime = field(default_factory=_utils.now_utc)
```

The streamed response peeks at the first chunk to get the model name and timestamp before yielding events. Tool call deltas are tracked by `vendor_part_id` (the delta's `index` field) and assembled into `ToolCallPart` events.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.huggingface import HuggingFaceModel

model = HuggingFaceModel('meta-llama/Llama-3.3-70B-Instruct')
agent = Agent(model)

async def main():
    async with agent.run_stream('Write a haiku about Python.') as stream:
        async for text in stream.stream_text():
            print(text, end='', flush=True)
    print()
    print(f'Model: {stream.model_name}')

asyncio.run(main())
```

---

## 3. `TestEmbeddingModel` — Mock Embeddings for Tests

**Module:** `pydantic_ai.embeddings.test`  
**Import:**
```python
from pydantic_ai.embeddings import TestEmbeddingModel
# or
from pydantic_ai.embeddings.test import TestEmbeddingModel
```

`TestEmbeddingModel` is a deterministic `EmbeddingModel` that returns all-`1.0` vectors. It tracks the last `EmbeddingSettings` used via `last_settings`, making it easy to assert that your code passed the right settings.

### Constructor

```python
@dataclass(init=False)
class TestEmbeddingModel(EmbeddingModel):
    __test__ = False  # prevents pytest from collecting it

    def __init__(
        self,
        model_name: str = 'test',
        *,
        provider_name: str = 'test',
        dimensions: int = 8,
        settings: EmbeddingSettings | None = None,
    ):
```

| Argument | Default | Purpose |
|---|---|---|
| `model_name` | `'test'` | Name reported in `EmbeddingResult.model_name` |
| `provider_name` | `'test'` | Provider reported in `EmbeddingResult.provider_name` |
| `dimensions` | `8` | Vector length for generated embeddings |
| `settings` | `None` | Default settings applied before each call |

### `last_settings` attribute

After each `embed` call, `last_settings` holds the resolved `EmbeddingSettings`. Useful for asserting that downstream code forwarded the right overrides.

### `max_input_tokens` and `count_tokens`

`max_input_tokens()` returns `1024`. `count_tokens(text)` splits on `[\s",.:]+` — the same regex `FunctionModel` uses — giving a reproducible approximation that matches test model token estimation elsewhere in the framework.

### Usage in tests

```python
import asyncio
import pytest
from pydantic_ai import Embedder
from pydantic_ai.embeddings import TestEmbeddingModel

async def test_embedder_passes_dimensions():
    test_model = TestEmbeddingModel(dimensions=16)
    embedder = Embedder('openai:text-embedding-3-small')

    async with embedder.override(model=test_model):
        result = await embedder.embed_query('hello world')

    assert len(result.embeddings[0]) == 16
    assert all(v == 1.0 for v in result.embeddings[0])
    assert test_model.last_settings is not None
    print(result.model_name)   # 'test'
    print(result.provider_name)  # 'test'
```

### Testing custom dimensions override via settings

```python
import asyncio
from pydantic_ai.embeddings import TestEmbeddingModel
from pydantic_ai.embeddings.settings import EmbeddingSettings

async def test_dimensions_from_settings():
    model = TestEmbeddingModel(dimensions=8)
    result = await model.embed(
        ['a', 'b', 'c'],
        input_type='document',
        settings=EmbeddingSettings(dimensions=32),
    )
    assert len(result.embeddings[0]) == 32  # settings override wins
    assert model.last_settings is not None
    assert model.last_settings.get('dimensions') == 32
```

### Override in integration tests

```python
import asyncio
from pydantic_ai import Embedder
from pydantic_ai.embeddings import TestEmbeddingModel

async def main():
    real_embedder = Embedder('openai:text-embedding-3-small')
    test_model = TestEmbeddingModel()

    async with real_embedder.override(model=test_model):
        result = await real_embedder.embed_many(['cat', 'dog'])

    assert result.embeddings[0] == [1.0] * 8
    assert result.embeddings[1] == [1.0] * 8

asyncio.run(main())
```

---

## 4. `SentenceTransformerEmbeddingModel` + `SentenceTransformersEmbeddingSettings`

**Module:** `pydantic_ai.embeddings.sentence_transformers`  
**Install extra:** `pip install "pydantic-ai-slim[sentence-transformers]"`  
**Import:**
```python
from pydantic_ai.embeddings.sentence_transformers import (
    SentenceTransformerEmbeddingModel,
    SentenceTransformersEmbeddingSettings,
)
```

`SentenceTransformerEmbeddingModel` runs embedding inference entirely on your machine. No API key required. Suitable for privacy-sensitive workloads, air-gapped environments, and reducing API costs on high-volume embedding tasks.

### Constructor

```python
@dataclass(init=False)
class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model: SentenceTransformer | str,
        *,
        settings: EmbeddingSettings | None = None,
    ) -> None:
```

| Argument | Purpose |
|---|---|
| `model` | A HuggingFace model ID string (downloaded on first use), a local path, or an existing `SentenceTransformer` instance |
| `settings` | Default `SentenceTransformersEmbeddingSettings` applied to each call |

### `SentenceTransformersEmbeddingSettings`

```python
class SentenceTransformersEmbeddingSettings(EmbeddingSettings, total=False):
    sentence_transformers_device: str
    # 'cpu', 'cuda', 'cuda:0', 'mps'

    sentence_transformers_normalize_embeddings: bool
    # L2-normalize to unit length for cosine similarity

    sentence_transformers_batch_size: int
    # Batch size for encoding; defaults to SentenceTransformers default
```

### Loading by name vs. instance

```python
from pydantic_ai.embeddings.sentence_transformers import SentenceTransformerEmbeddingModel

# Load from HuggingFace on first use
model_by_name = SentenceTransformerEmbeddingModel('sentence-transformers/all-MiniLM-L6-v2')

# Reuse an existing instance (deep-copied to prevent shared state)
from sentence_transformers import SentenceTransformer
st = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
model_by_instance = SentenceTransformerEmbeddingModel(st)
```

### Query vs. document encoding

`embed` maps `input_type='query'` to `model.encode_query` and `'document'` to `model.encode_document`. This respects the model's asymmetric instruction prefix if it uses one (e.g. E5, Qwen3-Embedding, GTE-Qwen):

```python
import asyncio
from pydantic_ai.embeddings.sentence_transformers import SentenceTransformerEmbeddingModel

async def main():
    model = SentenceTransformerEmbeddingModel('sentence-transformers/all-MiniLM-L6-v2')

    query_result = await model.embed('What is machine learning?', input_type='query')
    doc_result = await model.embed(
        ['Machine learning is a subset of AI.', 'Deep learning uses neural networks.'],
        input_type='document',
    )
    print(query_result.embeddings[0][:5])
    print(f'Docs embedded: {len(doc_result.embeddings)}')

asyncio.run(main())
```

### GPU inference with MPS (Apple Silicon)

```python
import asyncio
from pydantic_ai.embeddings.sentence_transformers import (
    SentenceTransformerEmbeddingModel,
    SentenceTransformersEmbeddingSettings,
)

async def main():
    model = SentenceTransformerEmbeddingModel(
        'Qwen/Qwen3-Embedding-0.6B',
        settings=SentenceTransformersEmbeddingSettings(
            sentence_transformers_device='mps',
            sentence_transformers_normalize_embeddings=True,
        ),
    )

    result = await model.embed_many(['apple', 'orange', 'car'], input_type='document')
    print(f'Embeddings shape: {len(result.embeddings)} x {len(result.embeddings[0])}')

asyncio.run(main())
```

### Token counting

`count_tokens` runs the model tokenizer in a thread executor to avoid blocking the event loop. `max_input_tokens` returns the model's `max_seq_length`:

```python
import asyncio
from pydantic_ai.embeddings.sentence_transformers import SentenceTransformerEmbeddingModel

async def main():
    model = SentenceTransformerEmbeddingModel('sentence-transformers/all-MiniLM-L6-v2')
    print(await model.max_input_tokens())  # 256 for all-MiniLM-L6-v2
    print(await model.count_tokens('Hello, world!'))

asyncio.run(main())
```

### Integration with `Embedder`

```python
import asyncio
from pydantic_ai import Embedder
from pydantic_ai.embeddings.sentence_transformers import SentenceTransformerEmbeddingModel

async def main():
    local_model = SentenceTransformerEmbeddingModel('sentence-transformers/all-MiniLM-L6-v2')
    embedder = Embedder(local_model)

    q = await embedder.embed_query('search term')
    docs = await embedder.embed_many(['doc one', 'doc two'], input_type='document')
    print(q.embeddings[0][:3], docs.embeddings)

asyncio.run(main())
```

---

## 5. `TemporalRunContext` — Serializable RunContext for Temporal Activities

**Module:** `pydantic_ai.durable_exec.temporal._run_context`  
**Import:**
```python
from pydantic_ai.durable_exec.temporal import TemporalAgent
# TemporalRunContext is in the internal module
from pydantic_ai.durable_exec.temporal._run_context import TemporalRunContext, deserialize_run_context
```

When a Temporal agent runs a tool, the tool function executes inside a Temporal activity. Activities are isolated processes — they cannot use closures over live Python objects. `TemporalRunContext` is a `RunContext` subclass that carries only the fields that can be serialized to JSON and sent across the activity boundary.

### What is excluded

The `capabilities` registry is intentionally excluded. It holds live toolset objects, callables, and toolmanagers that are not serializable. As a consequence, `available_capability_ids` (which reads `capabilities`) is unavailable inside an activity; `available_tool_names` still works via the `discovered_tool_names` fallback.

### Serialized fields (all by default)

```python
@classmethod
def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
    return {
        'run_id': ctx.run_id,
        'metadata': ctx.metadata,
        'retries': ctx.retries,
        'tool_call_id': ctx.tool_call_id,
        'tool_name': ctx.tool_name,
        'tool_call_approved': ctx.tool_call_approved,
        'tool_call_metadata': ctx.tool_call_metadata,
        'retry': ctx.retry,
        'max_retries': ctx.max_retries,
        'run_step': ctx.run_step,
        'partial_output': ctx.partial_output,
        'usage': ctx.usage,
        'loaded_capability_ids': ctx.loaded_capability_ids,
        'discovered_tool_names': ctx.discovered_tool_names,
        'capability_loaded': ctx.capability_loaded,
    }
```

### Accessing unavailable attributes

If a tool function tries to access an excluded attribute (e.g. `ctx.messages`), `TemporalRunContext.__getattribute__` catches the `AttributeError` and raises a `UserError` with instructions on how to extend the class:

```python
from pydantic_ai.durable_exec.temporal._run_context import TemporalRunContext
from pydantic_ai.tools import RunContext
from typing import Any

class MyTemporalRunContext(TemporalRunContext):
    """Custom subclass that also serializes `messages`."""

    @classmethod
    def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
        base = super().serialize_run_context(ctx)
        base['messages'] = ctx.messages
        return base
```

Then pass `run_context_type=MyTemporalRunContext` to `TemporalAgent`.

### `deserialize_run_context` helper

```python
def deserialize_run_context(
    run_context_type: type[TemporalRunContext[Any]],
    serialized: dict[str, Any],
    *,
    deps: Any,
    agent: AbstractAgent[Any, Any] | None,
) -> RunContext[Any]:
```

Called inside a Temporal activity (by `TemporalDynamicToolset`, `TemporalMCPToolset`, etc.) to reconstruct the run context from its serialized dict and attach the live `agent` and `deps`.

### Full example

```python
# agent.py
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import TemporalAgent

@dataclass
class Deps:
    db_url: str

agent = Agent('openai:gpt-4o', deps_type=Deps)

@agent.tool
async def lookup_record(ctx, record_id: int) -> str:
    return f"Record {record_id} from {ctx.deps.db_url}"

temporal_agent = TemporalAgent(agent, name='lookup-agent')

# worker.py
async def run_worker():
    from temporalio.client import Client
    from temporalio.worker import Worker

    client = await Client.connect('localhost:7233')
    async with Worker(
        client,
        task_queue='my-queue',
        workflows=temporal_agent.workflows,
        activities=temporal_agent.activities,
    ):
        await asyncio.Future()
```

---

## 6. `PydanticAIWorkflow` — Temporal Workflow Base Class

**Module:** `pydantic_ai.durable_exec.temporal._workflow`  
**Import:**
```python
from pydantic_ai.durable_exec.temporal._workflow import PydanticAIWorkflow
```

`PydanticAIWorkflow` is a minimal base class that marks a Temporal Workflow class as owning one or more `TemporalAgent` instances. Agents registered via `__pydantic_ai_agents__` are discovered automatically by `TemporalAgent.from_workflow()` so their activities are extracted without manual enumeration.

### Class definition

```python
class PydanticAIWorkflow:
    """Temporal Workflow base class that provides `__pydantic_ai_agents__` for direct agent registration."""

    __pydantic_ai_agents__: Sequence[TemporalAgent[Any, Any]]
```

### When to use

Use `PydanticAIWorkflow` when you want to bundle multiple `TemporalAgent` instances with a workflow class, so `TemporalAgent.from_workflow` can discover and register them all at once:

```python
import asyncio
from collections.abc import Sequence
from temporalio import workflow
from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import TemporalAgent
from pydantic_ai.durable_exec.temporal._workflow import PydanticAIWorkflow

planner_agent = Agent('openai:gpt-4o')
executor_agent = Agent('openai:gpt-4o-mini')

temporal_planner = TemporalAgent(planner_agent, name='planner')
temporal_executor = TemporalAgent(executor_agent, name='executor')

@workflow.defn
class PipelineWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__: Sequence = [temporal_planner, temporal_executor]

    @workflow.run
    async def run(self, input_text: str) -> str:
        plan = await temporal_planner.run(input_text)
        return await temporal_executor.run(plan.output)
```

### Activity extraction

The `TemporalAgent.activities` property and `from_workflow` class method traverse `__pydantic_ai_agents__` to assemble the complete list of Temporal activity functions that need to be registered with the worker:

```python
# Enumerate all activities from the workflow class
from pydantic_ai.durable_exec.temporal import TemporalAgent

all_activities = []
for agent in PipelineWorkflow.__pydantic_ai_agents__:
    all_activities.extend(agent.activities)

# Or use the convenience path (if implemented in your TemporalAgent version)
# activities = TemporalAgent.activities_from_workflow(PipelineWorkflow)
```

---

## 7. `TemporalDynamicToolset` — Dynamic Toolsets Across Temporal Activities

**Module:** `pydantic_ai.durable_exec.temporal._dynamic_toolset`  
**Import:**
```python
from pydantic_ai.durable_exec.temporal._dynamic_toolset import TemporalDynamicToolset
```

`DynamicToolset` normally runs its factory function inline within the agent loop. In a Temporal workflow, the agent loop is a workflow function — no I/O allowed. `TemporalDynamicToolset` wraps a `DynamicToolset` so that both `get_tools` (discovering available tools) and `call_tool` (executing them) each become Temporal activities that run in worker processes where I/O is permitted.

### Constructor

```python
class TemporalDynamicToolset(TemporalWrapperToolset[AgentDepsT]):
    def __init__(
        self,
        toolset: DynamicToolset[AgentDepsT],
        *,
        activity_name_prefix: str,
        activity_config: ActivityConfig,
        tool_activity_config: dict[str, ActivityConfig | Literal[False]],
        deps_type: type[AgentDepsT],
        run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
        agent: AbstractAgent[AgentDepsT, Any] | None = None,
    ):
```

| Argument | Purpose |
|---|---|
| `toolset` | The `DynamicToolset` to wrap |
| `activity_name_prefix` | Prefix for auto-generated activity names (must be unique per worker) |
| `activity_config` | Default Temporal `ActivityConfig` (timeout, retry policy, etc.) for both get/call activities |
| `tool_activity_config` | Per-tool activity config overrides. `False` bypasses the activity and calls the tool inline |
| `deps_type` | Python type of `AgentDepsT` — needed to annotate the activity function signature |
| `run_context_type` | `TemporalRunContext` subclass to use for serialization (default: base class) |
| `agent` | Optional agent instance injected into the deserialized context inside activities |

### How it works

Two Temporal activities are created in `__init__`:

1. **`get_tools_activity`** — calls the wrapped `DynamicToolset.for_run` and `for_run_step` to discover what tools are available this step, returning `dict[str, _ToolInfo]` (serializable).
2. **`call_tool_activity`** — re-instantiates the dynamic toolset and calls the named tool, returning a `CallToolResult`.

Both activities receive the serialized run context (via `TemporalRunContext.serialize_run_context`) and the deps object as arguments.

### Activity name pattern

Activity names follow the pattern:
```
{activity_name_prefix}__dynamic_toolset__{toolset.id}__get_tools
{activity_name_prefix}__dynamic_toolset__{toolset.id}__call_tool
```

These must be unique within your Temporal namespace and task queue.

### Per-tool config override

Set `tool_activity_config={'my_tool': False}` to bypass the activity and call `my_tool` inline in the workflow (useful for deterministic, pure-Python tools that don't need retry isolation):

```python
from pydantic_ai.durable_exec.temporal._dynamic_toolset import TemporalDynamicToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset
from temporalio.workflow import ActivityConfig

dynamic_ts = DynamicToolset(my_factory_function)

temporal_dynamic_ts = TemporalDynamicToolset(
    dynamic_ts,
    activity_name_prefix='my-agent',
    activity_config=ActivityConfig(
        start_to_close_timeout=30,
        retry_policy={'maximum_attempts': 3},
    ),
    tool_activity_config={
        'fast_pure_tool': False,  # run inline, no activity overhead
    },
    deps_type=MyDeps,
)
```

### `temporal_activities` property

```python
@property
def temporal_activities(self) -> list[Callable[..., Any]]:
    return [self.get_tools_activity, self.call_tool_activity]
```

These must be registered with the Temporal worker alongside any other activities.

### Full integration example

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.durable_exec.temporal import TemporalAgent
from pydantic_ai.durable_exec.temporal._dynamic_toolset import TemporalDynamicToolset
from temporalio.workflow import ActivityConfig

@dataclass
class MyDeps:
    user_role: str

async def toolset_factory(ctx) -> FunctionToolset:
    ts = FunctionToolset()
    if ctx.deps.user_role == 'admin':
        @ts.tool
        async def delete_record(record_id: int) -> str:
            return f"Deleted {record_id}"
    @ts.tool
    async def read_record(record_id: int) -> str:
        return f"Read {record_id}"
    return ts

dynamic_ts = DynamicToolset(toolset_factory)

config = ActivityConfig(start_to_close_timeout=30)
temporal_dynamic_ts = TemporalDynamicToolset(
    dynamic_ts,
    activity_name_prefix='my-agent',
    activity_config=config,
    tool_activity_config={},
    deps_type=MyDeps,
)

base_agent = Agent('openai:gpt-4o', toolsets=[dynamic_ts], deps_type=MyDeps)
temporal_agent = TemporalAgent(
    base_agent,
    name='my-agent',
    extra_toolsets=[temporal_dynamic_ts],
)
```

---

## 8. `PrefectAgentInputs` + `DEFAULT_PYDANTIC_AI_CACHE_POLICY`

**Module:** `pydantic_ai.durable_exec.prefect._cache_policies`  
**Import:**
```python
from pydantic_ai.durable_exec.prefect._cache_policies import (
    PrefectAgentInputs,
    DEFAULT_PYDANTIC_AI_CACHE_POLICY,
)
```

When a Prefect `PrefectAgent` runs tool-call tasks, Prefect can cache the results so identical calls are skipped on re-run (e.g. after a failure). The standard Prefect `INPUTS` cache policy hashes task inputs naively — it breaks on `RunContext` objects (not hashable) and is non-deterministic across runs (timestamps, run IDs). `PrefectAgentInputs` fixes both problems.

### `PrefectAgentInputs` — what it does

```python
class PrefectAgentInputs(CachePolicy):
    def compute_key(
        self,
        task_ctx: TaskRunContext,
        inputs: dict[str, Any],
        flow_parameters: dict[str, Any],
        **kwargs: Any,
    ) -> str | None:
```

Three transformations are applied before the key is computed:

1. **`_replace_toolsets`** — replaces `ToolsetTool` objects (not hashable) with a dict of their hashable fields, excluding `toolset` itself.
2. **`_replace_run_context`** — replaces `RunContext` instances with a deterministic dict of hashable fields: `retries`, `tool_call_id`, `tool_name`, `tool_call_approved`, `tool_call_metadata`, `retry`, `max_retries`, `run_step`, `loaded_capability_ids` (sorted), `discovered_tool_names` (sorted).
3. **`_strip_cache_excluded_fields`** — recursively walks the resulting dict and removes `timestamp` and `run_id` from any nested dataclass — these vary per-run and must not affect the cache key.

### Why `loaded_capability_ids` and `discovered_tool_names` are included

Two runs that are identical except for which capabilities have been loaded see different tools. Including `loaded_capability_ids` (sorted) and `discovered_tool_names` (sorted) in the cache key ensures they never share a cache entry. `capability_loaded` is deliberately omitted — it is derived from `loaded_capability_ids` plus the static capability set.

### `DEFAULT_PYDANTIC_AI_CACHE_POLICY`

```python
DEFAULT_PYDANTIC_AI_CACHE_POLICY = PrefectAgentInputs() + TASK_SOURCE + RUN_ID
```

The default cache policy combines:
- `PrefectAgentInputs()` — deterministic input hash
- `TASK_SOURCE` — changes if the task function's source code changes (catches code deploys)
- `RUN_ID` — scopes the cache to the current flow run (prevents cross-run pollution)

### Usage with `PrefectAgent`

The `DEFAULT_PYDANTIC_AI_CACHE_POLICY` is applied automatically by `PrefectAgent` for tool-call tasks. To override:

```python
from pydantic_ai.durable_exec.prefect import PrefectAgent
from pydantic_ai.durable_exec.prefect._cache_policies import PrefectAgentInputs
from prefect.cache_policies import TASK_SOURCE
from pydantic_ai import Agent

base_agent = Agent('openai:gpt-4o')
my_cache_policy = PrefectAgentInputs() + TASK_SOURCE  # no RUN_ID — share across flow runs

prefect_agent = PrefectAgent(
    base_agent,
    name='my-agent',
    model_step_config={'cache_policy': my_cache_policy},
)
```

### Writing a custom cache policy

```python
from pydantic_ai.durable_exec.prefect._cache_policies import PrefectAgentInputs, _strip_cache_excluded_fields
from prefect.cache_policies import CachePolicy, TASK_SOURCE, RUN_ID
from prefect.context import TaskRunContext
from typing import Any

class MyCache(CachePolicy):
    def compute_key(
        self,
        task_ctx: TaskRunContext,
        inputs: dict[str, Any],
        flow_parameters: dict[str, Any],
        **kwargs: Any,
    ) -> str | None:
        # Use PrefectAgentInputs as a sub-step, then add custom logic
        base_policy = PrefectAgentInputs()
        base_key = base_policy.compute_key(task_ctx, inputs, flow_parameters, **kwargs)
        if base_key is None:
            return None
        # Salt with a version tag
        return f"v2:{base_key}"

my_policy = MyCache() + TASK_SOURCE + RUN_ID
```

---

## 9. `ACIToolset` + `tool_from_aci` — Deprecated ACI.dev Integration

**Module:** `pydantic_ai.ext.aci`  
**Import:**
```python
# These are deprecated — they emit PydanticAIDeprecationWarning on import/use
from pydantic_ai.ext.aci import tool_from_aci, ACIToolset
```

**Install extra:** `pip install aci-sdk`

<Aside type="caution">
Both `tool_from_aci` and `ACIToolset` are deprecated as of pydantic-ai 1.x and will be removed in 2.0. The migration path is to use `Tool.from_schema` directly. See the migration guide below.
</Aside>

### `tool_from_aci` (deprecated)

```python
@deprecated(...)
def tool_from_aci(aci_function: str, linked_account_owner_id: str) -> Tool:
```

Creates a `Tool` from an ACI.dev function definition. Internally it calls `aci.functions.get_definition()` and passes the JSON schema to `Tool.from_schema`. Non-standard keys (`visible`) are stripped from the schema before conversion.

### `ACIToolset` (deprecated)

```python
@deprecated(...)
class ACIToolset(FunctionToolset):
    def __init__(
        self,
        aci_functions: Sequence[str],
        linked_account_owner_id: str,
        *,
        id: str | None = None,
    ):
```

Wraps multiple ACI.dev functions into a single `FunctionToolset`. Internally calls `tool_from_aci` per function.

### Migration to `Tool.from_schema`

Replace `tool_from_aci` with direct `Tool.from_schema` usage:

```python
# BEFORE (deprecated):
import warnings
from pydantic_ai.ext.aci import tool_from_aci
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    tool = tool_from_aci('GITHUB__CREATE_ISSUE', user_id)

# AFTER (recommended):
from aci import ACI
from pydantic_ai.tools import Tool

def _clean_schema(schema):
    if isinstance(schema, dict):
        return {k: _clean_schema(v) for k, v in schema.items() if k != 'visible'}
    elif isinstance(schema, list):
        return [_clean_schema(item) for item in schema]
    return schema

def make_aci_tool(function_name: str, user_id: str) -> Tool:
    aci = ACI()
    defn = aci.functions.get_definition(function_name)['function']
    params = defn['parameters']
    schema = _clean_schema({
        'type': params.get('type', 'object'),
        'properties': params.get('properties', {}),
        'required': params.get('required', []),
        'additionalProperties': params.get('additionalProperties', False),
    })

    def impl(**kwargs):
        return aci.handle_function_call(function_name, kwargs, linked_account_owner_id=user_id, allowed_apps_only=True)

    return Tool.from_schema(
        function=impl,
        name=defn['name'],
        description=defn['description'],
        json_schema=schema,
    )

github_tool = make_aci_tool('GITHUB__CREATE_ISSUE', 'user-123')
```

### Replace `ACIToolset` with `FunctionToolset`

```python
# BEFORE (deprecated):
from pydantic_ai.ext.aci import ACIToolset
ts = ACIToolset(['GITHUB__CREATE_ISSUE', 'GITHUB__LIST_REPOS'], user_id='user-123')

# AFTER (recommended):
from pydantic_ai.toolsets import FunctionToolset

functions = ['GITHUB__CREATE_ISSUE', 'GITHUB__LIST_REPOS']
ts = FunctionToolset(
    [make_aci_tool(fn, 'user-123') for fn in functions],
    id='aci-tools',
)
```

### Still using ACI.dev in 2025?

The ACI SDK now provides its own PydanticAI integration. Check the `aci-sdk` documentation for the current recommended approach to wrapping ACI functions, which may have changed since pydantic-ai deprecated this module.

---

## 10. Provider Model Profile Functions — Grok, Groq, DeepSeek, Qwen, Cohere, MoonshotAI

**Module:** `pydantic_ai.profiles.*`  
**Imports:**
```python
from pydantic_ai.profiles.grok import GrokModelProfile, grok_model_profile, GrokReasoningEffort
from pydantic_ai.profiles.groq import GroqModelProfile, groq_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
```

These functions are called by the corresponding `Provider` implementation during model construction to set capability flags (thinking support, JSON schema output, tool-choice modes, etc.) based on the model name string.

### `GrokModelProfile` + `GrokReasoningEffort`

```python
GrokReasoningEffort: TypeAlias = Literal['none', 'low', 'medium', 'high']

@dataclass(kw_only=True)
class GrokModelProfile(ModelProfile):
    grok_supports_builtin_tools: bool = False
    grok_supports_tool_choice_required: bool = True
    grok_reasoning_efforts: frozenset[GrokReasoningEffort] = frozenset()
```

| Field | Purpose |
|---|---|
| `grok_supports_builtin_tools` | Model supports `web_search`, `x_search`, `code_execution`, `mcp` native tools |
| `grok_supports_tool_choice_required` | Provider accepts `tool_choice='required'` |
| `grok_reasoning_efforts` | Set of valid `reasoning_effort` values for this model |

`grok_model_profile` applies the following logic:

- Grok 4.x, `code`/`build` models, and models in `_GROK_43_REASONING_MODELS` → `grok_supports_builtin_tools=True`
- Models in `_GROK_43_REASONING_MODELS` → four-level reasoning (`none`, `low`, `medium`, `high`) and `thinking_always_enabled=False`
- `grok-3-mini*` → two-level reasoning (`low`, `high`) and `thinking_always_enabled=True` (can't disable)

```python
from pydantic_ai.profiles.grok import grok_model_profile, GrokModelProfile

profile = grok_model_profile('grok-4.3')
assert isinstance(profile, GrokModelProfile)
assert profile.grok_supports_builtin_tools is True
assert profile.supports_thinking is True
assert 'none' in profile.grok_reasoning_efforts  # can disable thinking

profile_mini = grok_model_profile('grok-3-mini')
assert profile_mini.thinking_always_enabled is True  # always reasons
```

### `GroqModelProfile` + `groq_model_profile`

```python
@dataclass(kw_only=True)
class GroqModelProfile(ModelProfile):
    groq_always_has_web_search_builtin_tool: bool = False
```

Groq-specific compound models (e.g. `compound-beta`) always have web search available via a native tool, even without explicit tool configuration. Reasoning models on Groq are `qwen/qwen3-*`, `qwen-qwq-*`, `deepseek-r1-*`, `llama-4-maverick-*`:

```python
from pydantic_ai.profiles.groq import groq_model_profile

profile_compound = groq_model_profile('compound-beta')
assert profile_compound.groq_always_has_web_search_builtin_tool is True

profile_qwen3 = groq_model_profile('qwen/qwen3-32b')
assert profile_qwen3.supports_thinking is True
assert profile_qwen3.thinking_always_enabled is False  # can use reasoning_effort='none'

profile_r1 = groq_model_profile('deepseek-r1-distill-llama-70b')
assert profile_r1.thinking_always_enabled is True  # legacy: can't disable
```

### `deepseek_model_profile`

```python
def deepseek_model_profile(model_name: str) -> ModelProfile | None:
    is_r1 = model_name.startswith('deepseek-r1') or model_name == 'deepseek-reasoner'
    is_v4 = model_name.startswith('deepseek-v4-')
    return ModelProfile(
        ignore_streamed_leading_whitespace=is_r1,
        supports_thinking=is_r1 or is_v4,
        thinking_always_enabled=is_r1,
    )
```

- R1 and `deepseek-reasoner` → thinking always on, leading whitespace stripped from streamed output
- V4 models (`deepseek-v4-flash`, `deepseek-v4-pro`) → thinking optional via `reasoning_effort`

```python
from pydantic_ai.profiles.deepseek import deepseek_model_profile

r1_profile = deepseek_model_profile('deepseek-r1')
assert r1_profile.supports_thinking is True
assert r1_profile.thinking_always_enabled is True
assert r1_profile.ignore_streamed_leading_whitespace is True

v4_profile = deepseek_model_profile('deepseek-v4-flash')
assert v4_profile.supports_thinking is True
assert v4_profile.thinking_always_enabled is False
```

### `qwen_model_profile`

```python
def qwen_model_profile(model_name: str) -> ModelProfile | None:
    if model_name.startswith('qwen-3-coder'):
        return OpenAIModelProfile(
            json_schema_transformer=InlineDefsJsonSchemaTransformer,
            openai_supports_tool_choice_required=False,
            openai_supports_strict_tool_definition=False,
            ignore_streamed_leading_whitespace=True,
        )
    if _QWEN_3_5_RE.search(model_name):  # matches qwen-3.5, qwen3-5, etc.
        return ModelProfile(
            json_schema_transformer=InlineDefsJsonSchemaTransformer,
            ignore_streamed_leading_whitespace=True,
            supports_json_schema_output=True,
            supports_json_object_output=True,
        )
    return ModelProfile(
        json_schema_transformer=InlineDefsJsonSchemaTransformer,
        ignore_streamed_leading_whitespace=True,
    )
```

Key behaviours:
- All Qwen models → `InlineDefsJsonSchemaTransformer` (flattens `$defs` references inline — Qwen models don't support external `$defs`)
- All Qwen models → `ignore_streamed_leading_whitespace=True` (strips the leading `\n` some Qwen versions emit)
- `qwen-3-coder` → `openai_supports_tool_choice_required=False` (coder models reject `required`)
- Qwen 3.5 family → `supports_json_schema_output=True` and `supports_json_object_output=True`

```python
from pydantic_ai.profiles.qwen import qwen_model_profile

base = qwen_model_profile('qwen-2.5-72b')
assert base.ignore_streamed_leading_whitespace is True

coder = qwen_model_profile('qwen-3-coder')
from pydantic_ai.profiles.openai import OpenAIModelProfile
assert isinstance(coder, OpenAIModelProfile)
assert coder.openai_supports_tool_choice_required is False

q35 = qwen_model_profile('qwen-3.5-72b')
assert q35.supports_json_schema_output is True
```

### `cohere_model_profile`

```python
def cohere_model_profile(model_name: str) -> ModelProfile | None:
    is_reasoning = 'reasoning' in model_name
    if is_reasoning:
        return ModelProfile(supports_thinking=True, thinking_always_enabled=True)
    return None
```

Cohere models with `'reasoning'` in their name always have thinking enabled. Non-reasoning models return `None`, meaning the provider-level defaults apply.

```python
from pydantic_ai.profiles.cohere import cohere_model_profile

reasoning = cohere_model_profile('command-r-reasoning')
assert reasoning.thinking_always_enabled is True

non_reasoning = cohere_model_profile('command-r-plus')
assert non_reasoning is None  # no profile override needed
```

### `moonshotai_model_profile`

```python
def moonshotai_model_profile(model_name: str) -> ModelProfile | None:
    return ModelProfile(ignore_streamed_leading_whitespace=True)
```

All MoonshotAI (Kimi) models emit leading whitespace in streamed responses — this single-field profile strips it universally.

### Using profiles to override model behaviour

You can apply any profile function's output manually to override what the provider would choose:

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.profiles.deepseek import deepseek_model_profile

# Use DeepSeek R1 via OpenAI-compatible endpoint with correct profile
model = OpenAIModel(
    'deepseek-r1',
    base_url='https://api.deepseek.com/v1',
    api_key='YOUR_KEY',
    profile=deepseek_model_profile('deepseek-r1'),
)

agent = Agent(model)

async def main():
    result = await agent.run('Prove that sqrt(2) is irrational.')
    for part in result.all_messages()[-1].parts:
        print(type(part).__name__, ':', repr(part)[:80])

asyncio.run(main())
```

### Profile merging with `update`

All profile objects support `update(other)` which returns a new profile with fields from `other` overwriting non-default fields in `self`:

```python
from pydantic_ai.profiles.grok import GrokModelProfile, grok_model_profile
from pydantic_ai.profiles import ModelProfile

# Start with a base profile and add grok-specific fields
base = ModelProfile(supports_json_schema_output=True)
grok = grok_model_profile('grok-4.3')
merged = base.update(grok)
assert merged.supports_json_schema_output is True
assert isinstance(merged, GrokModelProfile)
```

---

## Cross-reference with previous volumes

| Class / group | Volume |
|---|---|
| `CombinedCapability` | Vol. 5 |
| `AbstractToolset` + `ToolsetTool` | Vol. 10 |
| `DynamicCapability` | Vol. 3 |
| `PendingMessage` + `RunContext.enqueue` | Vol. 5 |
| `TemporalAgent` + `ToolsetTool` | Vol. 6 |
| `PrefectAgent` + `TaskConfig` | Vol. 8 |
| `DBOSAgent` | Vol. 5 |
| `LangChainTool` + `LangChainToolset` | Vol. 4 |
| `ModelProfile` + `ModelProfileSpec` | Vol. 2 |
| `AnthropicModelProfile` + `OpenAIModelProfile` | Vol. 14 |
| `WrapperEmbeddingModel` + `InstrumentedEmbeddingModel` | Vol. 14 |
| `GoogleEmbeddingModel`, `BedrockEmbeddingModel`, etc. | Vol. 14 |
| `MCPSamplingModel` | Vol. 12 |

## Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-06-14 | Initial publication, verified against pydantic-ai 1.107.0 |

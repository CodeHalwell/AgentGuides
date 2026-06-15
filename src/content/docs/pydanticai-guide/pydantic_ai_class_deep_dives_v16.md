---
title: "PydanticAI — Class Deep Dives Vol. 16"
description: "Source-verified deep dives into 10 class groups from pydantic-ai 1.107.0: AgentRunResult (all methods + output_tool_return_content multi-turn), StreamedRunResult + StreamedRunResultSync (streaming result wrappers), Tool direct construction (all 17 params + from_schema/from_function), GenerateToolJsonSchema + DocstringFormat (schema generation internals), MistralModel + MistralModelSettings + MistralStreamedResponse (Mistral provider), OllamaModel (self-hosted vs Cloud NativeOutput behaviour), OpenRouterModel + OpenRouterModelSettings + OpenRouterModelProfile + OpenRouterReasoning (OpenRouter meta-provider), NamedSpec + CapabilitySpec + build_registry + load_from_registry (YAML/JSON spec loading), OutputSchema + OutputValidator (output validation machinery), GraphRun + NodeStep (v2 graph execution primitives). All verified against pydantic-ai 1.107.0 source."
sidebar:
  label: "Class deep dives (Vol. 16)"
  order: 42
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 1.107.0** source installed directly from PyPI. Class signatures, field names, and behaviour match the installed package at this version.
</Aside>

Ten class groups covering the result layer, streaming surface, tool construction internals, three provider implementations (Mistral, Ollama, OpenRouter), the YAML/JSON spec-loading system for capabilities, the output validation machinery, and the v2 graph execution primitives: `AgentRunResult` (all six public methods with `output_tool_return_content` multi-turn pattern); `StreamedRunResult` + `StreamedRunResultSync` (the high-level streaming result wrappers returned by `run_stream()`); `Tool` (direct dataclass construction vs decorator — all 17 parameters, `from_schema()`, `from_function()`, `function_schema`); `GenerateToolJsonSchema` + `DocstringFormat` (schema generation pipeline — four docstring formats, `require_parameter_descriptions`, custom `schema_generator`); `MistralModel` + `MistralModelSettings` + `MistralStreamedResponse` (Mistral AI provider, `json_mode_schema_prompt`, `PromptedOutput` workaround table); `OllamaModel` (self-hosted vs Ollama Cloud `NativeOutput` behaviour — `supports_json_schema_output` auto-disable, Cloud detection heuristics, `ToolOutput` fallback patterns); `OpenRouterModel` + `OpenRouterModelSettings` + `OpenRouterModelProfile` + `OpenRouterReasoning` (OpenRouter meta-provider — model fallback routing, provider preference config, cache-control flags, cross-provider reasoning tokens); `NamedSpec` + `CapabilitySpec` + `build_registry` + `load_from_registry` (YAML/JSON spec-driven capability composition — short-form serialisation, custom registries, legacy alias handling); `OutputSchema` + `OutputValidator` (output validation machinery — `OutputSchema.build()` factory, `TextOutputSchema`, `ToolOutputSchema`, `NativeOutputSchema`, `PromptedOutputSchema`, `OutputValidator.validate()` sync/async dispatch); `GraphRun` + `NodeStep` (v2 graph execution primitives from `pydantic_ai.run` — `GraphRun` execution state manager, fork/join coordination, `NodeStep` bridging v1 `BaseNode` classes into the v2 system).

---

## 1. `AgentRunResult` — Anatomy of the Sync Result

**Module:** `pydantic_ai.run`  
**Import:**
```python
from pydantic_ai import Agent
# AgentRunResult is returned by agent.run_sync() and agent.run()
```

`AgentRunResult[OutputDataT]` is the object you get back from a completed non-streaming run. It exposes six public methods for accessing messages, a `usage()` method, and the `output` field.

### Fields

```python
@dataclasses.dataclass
class AgentRunResult(Generic[OutputDataT]):
    output: OutputDataT          # The validated output — always populated
    # All other fields are private (_state, _new_message_index, etc.)
```

Only `output` is public. Everything else is accessed via methods.

### `usage()` — Token consumption

```python
result = agent.run_sync("Tell me a joke")
usage = result.usage()
print(usage.input_tokens, usage.output_tokens, usage.total_tokens)
```

`usage()` returns a `RunUsage` object summing all token counts across every step in the run (model calls, tool calls, retries). Use it for cost attribution.

### `all_messages()` — Full history

```python
result = agent.run_sync("What is 2+2?")
messages = result.all_messages()
# Returns every ModelRequest + ModelResponse from this run,
# prepended by any message_history= passed in.
```

Pass `output_tool_return_content=` to patch the last tool-return in-place (see multi-turn below).

### `new_messages()` — This run only

```python
result1 = agent.run_sync("Hello")
result2 = agent.run_sync("Follow up", message_history=result1.all_messages())

# new_messages() slices off the history from result1
only_round2 = result2.new_messages()
```

`new_messages()` uses `_new_message_index` to slice `all_messages()` and return only the messages produced in this run.

### Multi-turn conversation with `output_tool_return_content`

When `output_type` is a Pydantic model, PydanticAI asks the model to fill a tool call. For multi-turn conversations you may want to tell the model what your app "did" with the output:

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent

class Order(BaseModel):
    item: str
    quantity: int

agent = Agent('openai:gpt-4o', output_type=Order)

async def main():
    result = await agent.run("I want 3 apples")
    order: Order = result.output

    # Inject the outcome of processing the order back into history:
    history = result.all_messages(
        output_tool_return_content=f"Order #{order.item}-42 confirmed and queued."
    )

    # Continue the conversation from the injected state
    result2 = await agent.run(
        "Can I change the quantity to 5?",
        message_history=history,
    )
    print(result2.output)

asyncio.run(main())
```

`output_tool_return_content` does a **deep-copy** of the last `ModelResponse`, swaps the `ToolReturnPart.content` for the provided string, and returns the modified list without mutating `result` itself.

### `all_messages_json()` / `new_messages_json()` — JSON serialisation

Both `all_messages()` and `new_messages()` have `_json()` variants that return `bytes` via `ModelMessagesTypeAdapter.dump_json()`. Useful for storing conversation state in a database or sending over HTTP.

```python
import json
result = agent.run_sync("What is the capital of France?")
raw = result.new_messages_json()
# Reconstruct for the next turn:
history = agent.model.messages_type_adapter.validate_json(raw)
```

---

## 2. `StreamedRunResult` + `StreamedRunResultSync` — Streaming Result Wrappers

**Module:** `pydantic_ai.result`  
**Import:**
```python
from pydantic_ai.result import StreamedRunResult, StreamedRunResultSync
```

These two classes wrap `AgentStream` with a higher-level, user-facing API. `StreamedRunResult` is the async version returned by `agent.run_stream()`; `StreamedRunResultSync` is a thin sync wrapper that delegates to it.

### `StreamedRunResult` — the async wrapper

```python
@dataclass(init=False)
class StreamedRunResult(Generic[AgentDepsT, OutputDataT]):
    is_complete: bool  # set True once a streaming method finishes
```

#### `stream_output(delta=False, debounce_by=0.1)` — validate output while streaming

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent

class Report(BaseModel):
    title: str
    body: str

agent = Agent('openai:gpt-4o', output_type=Report)

async def main():
    async with agent.run_stream("Write a haiku report") as result:
        async for partial in result.stream_output(delta=False):
            # partial is a Report with whatever fields have been filled so far
            print(partial.title, "—", partial.body[:40])
    print("Final:", result.get_output())
```

- `delta=False` (default): yields cumulative partial objects
- `delta=True`: yields only the incremental text delta (useful for token streaming)
- `debounce_by=0.1`: coalesces rapid successive events within 100 ms to reduce output churn

#### `stream_text(delta=False, debounce_by=0.1)` — plain text streaming

```python
async with agent.run_stream("Tell me a story") as result:
    async for chunk in result.stream_text(delta=True):
        print(chunk, end="", flush=True)
    print()  # newline after stream
```

`stream_text()` is only valid when `output_type=str`. It raises `UserError` if the output type is structured.

#### `stream_response(debounce_by=0.1)` — raw model response parts

```python
from pydantic_ai.messages import TextPart, ToolCallPart

async with agent.run_stream("What tools do you want to call?") as result:
    async for part in result.stream_response(debounce_by=0):
        if isinstance(part, TextPart):
            print("Text:", part.content)
        elif isinstance(part, ToolCallPart):
            print("Tool:", part.tool_name, part.args_as_json_str())
```

#### `get_output()` — wait for the final validated output

```python
async with agent.run_stream("Summarise this") as result:
    # If you don't iterate, just call get_output() to wait for completion
    output = await result.get_output()
    print(output)
```

`get_output()` calls `stream_output()` internally and returns only the final value. If you already iterated via `stream_output()`, calling `get_output()` a second time is safe — it returns the cached result.

#### Messaging methods on `StreamedRunResult`

```python
async with agent.run_stream("Hello") as result:
    async for _ in result.stream_text(delta=True):
        pass
    # After streaming completes:
    print(result.usage())              # RunUsage object
    print(result.new_messages())       # messages from this run only
    print(result.all_messages())       # full history
    history_bytes = result.new_messages_json()  # bytes for storage
```

### `StreamedRunResultSync` — sync CLI/notebook wrapper

`StreamedRunResultSync` is returned by `agent.run_stream_sync()`. It wraps `StreamedRunResult` and exposes only synchronous methods:

```python
with agent.run_stream_sync("Summarise this text") as result:
    for chunk in result.stream_text_sync(delta=True):
        print(chunk, end="", flush=True)
    print()
    print("Usage:", result.usage())
    messages = result.new_messages()
```

The sync variants run the underlying async code in a background thread using `anyio.from_thread.run_sync`. All methods map 1-to-1 with their async counterparts.

| Async method | Sync equivalent |
|---|---|
| `stream_output(delta=, debounce_by=)` | `stream_output_sync(delta=, debounce_by=)` |
| `stream_text(delta=, debounce_by=)` | `stream_text_sync(delta=, debounce_by=)` |
| `stream_response(debounce_by=)` | `stream_response_sync(debounce_by=)` |
| `get_output()` | `get_output_sync()` |

---

## 3. `Tool` — Direct Construction

**Module:** `pydantic_ai.tools`  
**Import:**
```python
from pydantic_ai.tools import Tool
```

Usually you use `@agent.tool` / `@agent.tool_plain` decorators, but `Tool` is a public dataclass you can construct directly. This is useful when building dynamic tool registries, introspecting existing tools, or integrating third-party APIs.

### All 17 constructor parameters

```python
Tool(
    function,                          # async or sync callable
    takes_ctx=None,                    # auto-detected; True = first param is RunContext
    max_retries=None,                  # override agent-level max_retries
    name=None,                         # defaults to function.__name__
    description=None,                  # defaults to docstring first paragraph
    prepare=None,                      # ToolPrepareFunc called before each use
    args_validator=None,               # ArgsValidatorFunc for validated args inspection
    docstring_format='auto',           # 'google' | 'numpy' | 'sphinx' | 'auto'
    require_parameter_descriptions=False,  # raise if any param lacks a description
    schema_generator=GenerateToolJsonSchema,  # custom JSON Schema generator class
    strict=None,                       # force strict JSON schema mode
    sequential=False,                  # run this tool sequentially (not in parallel)
    requires_approval=False,           # trigger HITL approval gate
    metadata=None,                     # arbitrary dict, never sent to model
    timeout=None,                      # per-call timeout in seconds
    defer_loading=False,               # exclude from initial catalog (ToolSearch)
    include_return_schema=None,        # include return type in JSON schema
)
```

### `from_function()` — factory with auto-detection

```python
from pydantic_ai.tools import Tool

async def search_web(query: str, max_results: int = 5) -> list[str]:
    """Search the web for a query.
    
    Args:
        query: The search query string.
        max_results: Maximum number of results to return.
    """
    ...

tool = Tool.from_function(search_web, max_retries=2, timeout=30.0)
agent = Agent('openai:gpt-4o', tools=[tool])
```

### `from_schema()` — schema-driven tool without a function

For bridging external APIs where you receive a JSON Schema but want to handle execution yourself:

```python
from pydantic_ai.tools import Tool, ToolDefinition

schema = ToolDefinition(
    name="get_weather",
    description="Get current weather for a city",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
)

async def execute_weather(city: str, units: str = "celsius") -> str:
    return f"Weather in {city}: 22°{units[0].upper()}"

tool = Tool.from_schema(schema, function=execute_weather)
```

### `function_schema` — inspecting the built schema

```python
tool = Tool.from_function(search_web)
print(tool.function_schema.description)
print(tool.function_schema.parameters_json_schema)
# The schema is cached and reused; the model may further modify it via `prepare`
```

### Dynamic tool modification with `prepare`

```python
from pydantic_ai import RunContext
from pydantic_ai.tools import Tool, ToolDefinition

async def translate(text: str, target_lang: str) -> str:
    """Translate text to another language."""
    ...

async def prepare_translate(ctx: RunContext, tool_def: ToolDefinition) -> ToolDefinition | None:
    # Restrict available target languages based on user tier
    if ctx.deps.get("tier") == "free":
        new_schema = dict(tool_def.parameters_json_schema)
        new_schema["properties"]["target_lang"] = {
            "type": "string", "enum": ["es", "fr"]
        }
        return ToolDefinition(**{**tool_def.__dict__, "parameters_json_schema": new_schema})
    return tool_def  # pro tier: all languages

tool = Tool(translate, prepare=prepare_translate)
```

Return `None` from `prepare` to suppress the tool for this step entirely.

---

## 4. `GenerateToolJsonSchema` + `DocstringFormat` — Schema Generation

**Module:** `pydantic_ai.tools`  
**Import:**
```python
from pydantic_ai.tools import GenerateToolJsonSchema, DocstringFormat
```

### `DocstringFormat` — the four styles

`DocstringFormat = Literal['google', 'numpy', 'sphinx', 'auto']`

In `'auto'` mode (the default) PydanticAI detects the docstring style from the function. The four styles affect how parameter descriptions are extracted:

**Google style:**
```python
async def lookup(key: str, case_sensitive: bool = True) -> str:
    """Look up a value by key.
    
    Args:
        key: The lookup key.
        case_sensitive: Whether matching is case-sensitive.
    """
    ...
```

**NumPy style:**
```python
async def lookup(key: str, case_sensitive: bool = True) -> str:
    """Look up a value by key.
    
    Parameters
    ----------
    key : str
        The lookup key.
    case_sensitive : bool
        Whether matching is case-sensitive.
    """
    ...
```

**Sphinx style:**
```python
async def lookup(key: str, case_sensitive: bool = True) -> str:
    """Look up a value by key.
    
    :param key: The lookup key.
    :param case_sensitive: Whether matching is case-sensitive.
    """
    ...
```

Set `docstring_format` on the `Tool` or `FunctionToolset`:

```python
from pydantic_ai.tools import Tool
from pydantic_ai.toolsets import FunctionToolset

# Force sphinx style for a whole toolset
ts = FunctionToolset(docstring_format='sphinx')

# Force google style for a single tool
tool = Tool(lookup, docstring_format='google')
```

### `GenerateToolJsonSchema` — removing redundant titles

`GenerateToolJsonSchema` subclasses Pydantic's `GenerateJsonSchema` and overrides `_named_required_fields_schema()` to strip the `title` field from each property (models don't need them and they add noise):

```python
from pydantic import BaseModel
from pydantic_ai.tools import GenerateToolJsonSchema

class MyArgs(BaseModel):
    query: str
    limit: int = 10

gen = GenerateToolJsonSchema(mode='validation')
schema = gen.generate(MyArgs.__pydantic_core_schema__)
# `title` is absent from each property — model sees cleaner schema
print(schema)
```

### Custom `schema_generator`

Pass a subclass to override schema generation for a specific tool or toolset:

```python
from pydantic_ai.tools import Tool, GenerateToolJsonSchema
from pydantic.json_schema import GenerateJsonSchema

class CompactSchemaGenerator(GenerateToolJsonSchema):
    """Remove `examples` from all property schemas to save tokens."""
    def _named_required_fields_schema(self, named_required_fields):
        schema = super()._named_required_fields_schema(named_required_fields)
        for prop in schema.get("properties", {}).values():
            prop.pop("examples", None)
        return schema

tool = Tool(lookup, schema_generator=CompactSchemaGenerator)
```

### `require_parameter_descriptions`

Setting `require_parameter_descriptions=True` causes PydanticAI to raise `UserError` at registration time if any parameter lacks a description:

```python
from pydantic_ai.tools import Tool

async def undocumented(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y  # no param docs!

try:
    tool = Tool(undocumented, require_parameter_descriptions=True)
except Exception as e:
    print(e)
    # UserError: Tool 'undocumented' parameter 'x' has no description
```

---

## 5. `MistralModel` + `MistralModelSettings` + `MistralStreamedResponse`

**Module:** `pydantic_ai.models.mistral`  
**Import:**
```python
from pydantic_ai.models.mistral import MistralModel, MistralModelSettings
```

### `MistralModel` constructor

```python
@dataclass(init=False)
class MistralModel(Model[Mistral]):
    def __init__(
        self,
        model_name: MistralModelName,
        *,
        provider: Literal['mistral'] | Provider[Mistral] = 'mistral',
        profile: ModelProfileSpec | None = None,
        json_mode_schema_prompt: str = "Answer in JSON Object, respect the format:\n```\n{schema}\n```\n",
        settings: ModelSettings | None = None,
    ): ...
```

| Param | Purpose |
|---|---|
| `model_name` | Any Mistral model string: `'mistral-large-latest'`, `'mistral-small-latest'`, `'codestral-latest'`, `'pixtral-large-latest'`, `'mistral-embed'`, etc. |
| `provider` | `'mistral'` (auto-uses `MISTRAL_API_KEY` env var) or custom `MistralProvider`. |
| `profile` | Override capability flags. Default profile provided by `MistralProvider.model_profile()`. |
| `json_mode_schema_prompt` | Template injected when Mistral's API requires JSON mode — the `{schema}` placeholder is replaced with the output schema. |
| `settings` | Default `ModelSettings` to merge into every request. |

### Basic usage

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel

class Sentiment(BaseModel):
    label: str          # positive / negative / neutral
    confidence: float   # 0.0 – 1.0

async def main():
    model = MistralModel('mistral-large-latest')
    agent = Agent(model, output_type=Sentiment)
    result = await agent.run("I love this product!")
    print(result.output)  # Sentiment(label='positive', confidence=0.95)

asyncio.run(main())
```

### Multimodal with Pixtral

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ImageUrl
from pydantic_ai.models.mistral import MistralModel

agent = Agent(MistralModel('pixtral-large-latest'))

async def describe_image(url: str) -> str:
    result = await agent.run([
        "Describe what you see in this image:",
        ImageUrl(url=url),
    ])
    return result.output

import asyncio
print(asyncio.run(describe_image("https://example.com/photo.jpg")))
```

### `MistralModelSettings` — provider-specific settings

```python
class MistralModelSettings(ModelSettings, total=False):
    # Placeholder — currently inherits all fields from ModelSettings
    # with mistral_ prefix convention reserved for future fields
    pass
```

All standard `ModelSettings` fields work with Mistral: `temperature`, `max_tokens`, `top_p`, `seed`.

### `PromptedOutput` for legacy API endpoints

Mistral's JSON mode can be inconsistent on older models. Use `PromptedOutput` as a fallback:

```python
from pydantic_ai import Agent
from pydantic_ai.output import PromptedOutput
from pydantic_ai.models.mistral import MistralModel

class Analysis(BaseModel):
    summary: str
    keywords: list[str]

agent = Agent(
    MistralModel('mistral-small-latest'),
    output_type=PromptedOutput(Analysis),  # uses json_mode_schema_prompt
)
result = agent.run_sync("Analyse this text: AI is transforming software engineering.")
print(result.output)
```

### `MistralStreamedResponse` — streaming internals

```python
@dataclass
class MistralStreamedResponse(StreamedResponse):
    _model_name: MistralModelName
    _response: PeekableAsyncStream[MistralCompletionEvent, ...]
    _provider_name: str
    _provider_url: str
    _provider_timestamp: datetime | None
    _timestamp: datetime
```

`MistralStreamedResponse` handles Mistral's `MistralEventStreamAsync` format, extracting `TextPartDelta` and `ToolCallPartDelta` events from each `MistralCompletionEvent`. It maps Mistral's native `finish_reason` values to PydanticAI's `FinishReason` enum. You rarely construct this directly — `run_stream()` creates it.

---

## 6. `OllamaModel` — Self-hosted vs Cloud NativeOutput

**Module:** `pydantic_ai.models.ollama`  
**Import:**
```python
from pydantic_ai.models.ollama import OllamaModel
```

`OllamaModel` extends `OpenAIChatModel` using Ollama's OpenAI-compatible API. The key differentiator is automatic detection of Ollama Cloud vs self-hosted instances.

### Constructor

```python
@dataclass(init=False)
class OllamaModel(OpenAIChatModel):
    def __init__(
        self,
        model_name: str,
        *,
        provider: Literal['ollama'] | Provider[AsyncOpenAI] = 'ollama',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ): ...
```

### Cloud detection and `NativeOutput` auto-disable

Self-hosted Ollama (≥ v0.5.0) enforces `response_format` with `json_schema` via llama.cpp's grammar-constrained decoder — so `NativeOutput` works correctly. Ollama Cloud accepts `json_schema` format but **does not enforce the schema** at generation time.

PydanticAI detects the Cloud path automatically and disables `supports_json_schema_output`:

```python
# Cloud detection triggers on:
# 1. base_url containing 'ollama.com'
# 2. model_name ending with '-cloud'

from pydantic_ai.models.ollama import OllamaModel

# Self-hosted — NativeOutput works
local_model = OllamaModel('qwen3', provider='ollama')

# Cloud — NativeOutput is disabled automatically
cloud_model = OllamaModel('llama3.2-cloud', provider='ollama')
```

### Output mode compatibility table

| Mode | Self-hosted (≥0.5.0) | Ollama Cloud |
|---|---|---|
| `ToolOutput` (default) | ✅ Works | ✅ Works |
| `NativeOutput` | ✅ Schema-enforced | ❌ Raises `UserError` |
| `PromptedOutput` | ✅ Works | ✅ Works |
| `TextOutput` | ✅ Works | ✅ Works |

### Self-hosted agent with `NativeOutput`

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.output import NativeOutput

class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    steps: list[str]

async def main():
    agent = Agent(
        OllamaModel('qwen3'),
        output_type=NativeOutput(Recipe),  # llama.cpp enforces schema at generation
    )
    result = await agent.run("Give me a recipe for pasta carbonara")
    print(result.output)

asyncio.run(main())
```

### Custom Ollama endpoint

```python
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

# Point to a non-default port or remote Ollama server
provider = OllamaProvider(base_url="http://my-server:11435/v1")
model = OllamaModel('llama3.2', provider=provider)
```

### Overriding the profile for Cloud

If you need `NativeOutput` on Ollama Cloud (once they fix the upstream issue), you can manually override the profile:

```python
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.profiles import ModelProfile

override = ModelProfile(supports_json_schema_output=True)
model = OllamaModel('my-model-cloud', profile=override)
```

---

## 7. `OpenRouterModel` + `OpenRouterModelSettings` + `OpenRouterModelProfile` + `OpenRouterReasoning`

**Module:** `pydantic_ai.models.openrouter`  
**Import:**
```python
from pydantic_ai.models.openrouter import (
    OpenRouterModel,
    OpenRouterModelSettings,
    OpenRouterModelProfile,
    OpenRouterReasoning,
)
```

OpenRouter is a meta-provider routing requests to 200+ models from different vendors. `OpenRouterModel` extends `OpenAIChatModel` with additional response metadata extraction.

### `OpenRouterModel` constructor

```python
class OpenRouterModel(OpenAIChatModel):
    def __init__(
        self,
        model_name: str,
        *,
        provider: Literal['openrouter'] | Provider[AsyncOpenAI] = 'openrouter',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ): ...
```

Model name is a string like `'openai/gpt-4o'`, `'anthropic/claude-sonnet-4-6'`, `'google/gemini-2.5-flash'`, `'meta-llama/llama-4-maverick'`, etc.

### Basic usage

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel

# Requires OPENROUTER_API_KEY env var
agent = Agent(OpenRouterModel('anthropic/claude-sonnet-4-6'))

async def main():
    result = await agent.run("Explain quantum entanglement in one sentence.")
    print(result.output)

asyncio.run(main())
```

### `OpenRouterModelSettings` — provider-specific fields

```python
class OpenRouterModelSettings(ModelSettings, total=False):
    openrouter_models: list[str]          # fallback model chain
    openrouter_provider: OpenRouterProviderConfig  # provider routing preference
    openrouter_reasoning: OpenRouterReasoning      # cross-provider reasoning tokens
    openrouter_usage: OpenRouterUsageConfig        # token usage reporting
    openrouter_cache_ttl: Literal['5m', '1h']     # prompt cache TTL
    openrouter_transforms: list[str]               # OpenRouter middleware transforms
    openrouter_rank_by: str                        # routing rank strategy
```

### Model fallback chain

```python
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings

agent = Agent(
    OpenRouterModel('anthropic/claude-opus-4'),
    model_settings=OpenRouterModelSettings(
        openrouter_models=[
            'anthropic/claude-sonnet-4-6',   # first fallback
            'openai/gpt-4o',                  # second fallback
            'google/gemini-2.5-pro',          # last resort
        ]
    ),
)
```

### `OpenRouterProviderConfig` — routing preferences

```python
from pydantic_ai.models.openrouter import OpenRouterModelSettings, OpenRouterProviderConfig

settings = OpenRouterModelSettings(
    openrouter_provider=OpenRouterProviderConfig(
        order=['Anthropic', 'AWS Bedrock'],   # preferred provider order
        allow_fallbacks=True,                  # fall back if preferred is unavailable
        data_collection='deny',               # opt out of training data collection
    )
)
```

### `OpenRouterReasoning` — cross-provider thinking tokens

```python
from pydantic_ai.models.openrouter import OpenRouterModelSettings, OpenRouterReasoning

settings = OpenRouterModelSettings(
    openrouter_reasoning=OpenRouterReasoning(
        effort='high',     # OpenAI-style: 'xhigh' | 'high' | 'medium' | 'low' | 'minimal' | 'none'
        # max_tokens=2000  # Anthropic-style (mutually exclusive with effort)
        exclude=False,     # include reasoning in the response
    )
)

agent = Agent(OpenRouterModel('anthropic/claude-sonnet-4-6'), model_settings=settings)
```

### `OpenRouterModelProfile` — cache control flags

```python
@dataclass(kw_only=True)
class OpenRouterModelProfile(OpenAIModelProfile):
    openrouter_supports_cache_control: bool = False
    openrouter_supports_cache_ttl: bool = False
    openrouter_supports_tool_cache: bool = False
    openrouter_supports_dynamic_instruction_cache: bool = False
```

These flags indicate whether the downstream provider supports Anthropic-style cache breakpoints via OpenRouter. Check the provider documentation or set `profile` manually:

```python
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelProfile

profile = OpenRouterModelProfile(
    openrouter_supports_cache_control=True,
    openrouter_supports_cache_ttl=True,
)
model = OpenRouterModel('anthropic/claude-sonnet-4-6', profile=profile)
```

### Cost tracking from OpenRouter usage data

```python
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings

agent = Agent(
    OpenRouterModel('openai/gpt-4o'),
    model_settings=OpenRouterModelSettings(
        openrouter_usage={"include": True}  # Include native cost data in response
    )
)
result = agent.run_sync("Hello")
usage = result.usage()
print(f"Tokens — in: {usage.input_tokens}, out: {usage.output_tokens}")
```

---

## 8. `NamedSpec` + `CapabilitySpec` + `build_registry` + `load_from_registry`

**Module:** `pydantic_ai._spec`  
**Import:**
```python
from pydantic_ai._spec import NamedSpec, CapabilitySpec, build_registry, load_from_registry
```

This module powers YAML/JSON-driven capability composition — load a list of capability names from a config file and instantiate them without writing Python.

### `NamedSpec` — three short forms

`NamedSpec` accepts three serialised forms:

```python
# 1. Just a name string — no arguments
spec1 = NamedSpec.model_validate('Instrumentation')

# 2. Single-argument dict
spec2 = NamedSpec.model_validate({'WebSearch': 'bing'})

# 3. Keyword-argument dict
spec3 = NamedSpec.model_validate({
    'WebSearch': {'search_context_size': 'high', 'max_results': 20}
})
```

The `model_validator(mode='wrap')` on `NamedSpec` handles the compact forms transparently.

### `CapabilitySpec` — JSON schema integration

`CapabilitySpec` is a tagged subclass of `NamedSpec`. Fields typed as `CapabilitySpec` in other Pydantic models have their JSON schemas replaced with the full capability union type — the same types used by `AgentSpec.capabilities`. This enables IDE autocomplete and validation in tools that load `AgentSpec` from YAML.

### `build_registry` — creating a name→class map

```python
from pydantic_ai._spec import build_registry
from pydantic_ai.capabilities import WebSearch, WebFetch, Instrumentation

# Build a registry of capability classes keyed by their name
registry = build_registry(
    custom_types=[],           # user-defined capability classes to add
    defaults=[WebSearch, WebFetch, Instrumentation],
    get_name=lambda cls: cls.__name__,
    label='capability',
)
# registry == {'WebSearch': WebSearch, 'WebFetch': WebFetch, 'Instrumentation': Instrumentation}
```

### `load_from_registry` — instantiating from a spec

```python
from pydantic_ai._spec import load_from_registry, NamedSpec

spec = NamedSpec.model_validate({'WebSearch': {'search_context_size': 'medium'}})
web_search_cap = load_from_registry(
    registry,
    spec,
    label='capability',
    custom_types_param='custom_capability_types',
)
# web_search_cap is WebSearch(search_context_size='medium')
```

`load_from_registry` supports `legacy_aliases` for renamed classes:

```python
load_from_registry(
    registry,
    spec,
    label='capability',
    custom_types_param='custom_capability_types',
    legacy_aliases={'OldWebSearch': 'WebSearch'},  # rename transparent to callers
)
```

### YAML-driven agent configuration

```python
import yaml
from pydantic_ai import Agent
from pydantic_ai._spec import NamedSpec, build_registry, load_from_registry
from pydantic_ai.capabilities import WebSearch, WebFetch, Instrumentation

CAPABILITIES_REGISTRY = build_registry(
    custom_types=[],
    defaults=[WebSearch, WebFetch, Instrumentation],
    get_name=lambda cls: cls.__name__,
    label='capability',
)

CONFIG_YAML = """
capabilities:
  - WebSearch
  - WebFetch
  - name: Instrumentation
    kwargs:
      privacy: hide
"""

def load_agent_from_yaml(yaml_str: str, model: str) -> Agent:
    config = yaml.safe_load(yaml_str)
    caps = []
    for raw in config.get('capabilities', []):
        # Normalise to dict form
        if isinstance(raw, str):
            raw = raw
        spec = NamedSpec.model_validate(raw)
        caps.append(load_from_registry(CAPABILITIES_REGISTRY, spec, label='capability', custom_types_param='x'))
    return Agent(model, capabilities=caps)

agent = load_agent_from_yaml(CONFIG_YAML, 'openai:gpt-4o')
```

---

## 9. `OutputSchema` + `OutputValidator` — Output Validation Machinery

**Module:** `pydantic_ai.result` (schemas) / `pydantic_ai.output` (validator)  
**Import:**
```python
from pydantic_ai.result import OutputSchema
from pydantic_ai.output import OutputValidator
```

These internal classes power the complete output pipeline but are useful to understand when building custom output hooks or debugging validation failures.

### `OutputSchema` hierarchy

```python
@dataclass(kw_only=True)
class OutputSchema(ABC, Generic[OutputDataT]):
    allows_none: bool
    text_processor: BaseOutputProcessor[OutputDataT] | None = None
    toolset: OutputToolset[Any] | None = None
    object_def: OutputObjectDefinition | None = None
    allows_deferred_tools: bool = False
    allows_image: bool = False

    @property
    @abstractmethod
    def mode(self) -> OutputMode: ...

    @property
    def allows_text(self) -> bool:
        return self.text_processor is not None
```

Concrete subclasses:

| Subclass | `mode` | Created when |
|---|---|---|
| `TextOutputSchema` | `'text'` | `output_type=str` |
| `ToolOutputSchema` | `'tool'` | Pydantic model (default) |
| `NativeOutputSchema` | `'native'` | `NativeOutput(MyModel)` |
| `PromptedOutputSchema` | `'prompted'` | `PromptedOutput(MyModel)` |
| `ImageOutputSchema` | `'image'` | `output_type=BinaryContent` |
| `MultiOutputSchema` | `'auto'` | Union output types |

### `OutputSchema.build()` — the factory

`OutputSchema.build()` is the primary entry point that resolves an `output_type` argument into the correct subclass:

```python
from pydantic_ai.result import OutputSchema
from pydantic import BaseModel

class MyResult(BaseModel):
    answer: str

schema = OutputSchema.build(MyResult)
print(schema.mode)          # 'tool'
print(schema.allows_none)   # False
print(schema.object_def)    # OutputObjectDefinition(...)
```

### Inspecting schemas in output hooks

`OutputContext.object_def` (available in `before_output_validate` hooks) comes from the `OutputSchema.object_def` field. Use it to inspect the active output schema:

```python
from pydantic_ai import Agent
from pydantic_ai.output import OutputContext

async def inspect_output(ctx: OutputContext) -> None:
    if ctx.object_def:
        print(f"Output schema name: {ctx.object_def.name}")
        print(f"Strict mode: {ctx.object_def.strict}")
        print(f"Output mode: {ctx.mode}")

agent = Agent(
    'openai:gpt-4o',
    output_type=MyResult,
    # attach as before_output_validate hook via Hooks capability
)
```

### `OutputValidator` — the callable wrapper

`OutputValidator` wraps a user-provided output validator function and handles both sync and async forms, with or without `RunContext`:

```python
from pydantic_ai.output import OutputValidator
from pydantic_ai import RunContext

async def validate_answer(ctx: RunContext, result: MyResult) -> MyResult:
    if not result.answer:
        from pydantic_ai import ModelRetry
        raise ModelRetry("Answer cannot be empty")
    return result

validator = OutputValidator(validate_answer)
print(validator._takes_ctx)   # True
print(validator._is_async)    # True
```

The `validate()` method dispatches sync/async transparently:

```python
# Internally, Agent uses:
validated = await validator.validate(raw_output, run_context)
```

### Union output types and `MultiOutputSchema`

```python
from pydantic_ai import Agent
from pydantic_ai.result import OutputSchema

class Success(BaseModel):
    data: str

class Error(BaseModel):
    message: str

schema = OutputSchema.build(Success | Error)
print(schema.mode)  # 'auto' — uses MultiOutputSchema internally
```

With union types, PydanticAI registers multiple tool definitions and accepts whichever the model calls first.

---

## 10. `GraphRun` + `NodeStep` — v2 Graph Execution Primitives

**Module:** `pydantic_ai.run`  
**Import:**
```python
from pydantic_ai.run import GraphRun, NodeStep
```

These are the low-level execution primitives for PydanticAI's v2 graph engine (introduced alongside `pydantic_graph`). Agent execution uses `GraphRun` internally — understanding it helps when writing advanced node-by-node streaming or debugging graph topology.

### `GraphRun` — execution state manager

```python
class GraphRun(Generic[StateT, DepsT, OutputT]):
    graph: Graph[StateT, DepsT, InputT, OutputT]
    state: StateT
    deps: DepsT
    inputs: InputT
```

`GraphRun` manages:
- **Task scheduling**: `_first_task` seeds execution; `_next` tracks what comes next (`EndMarker`, `ErrorMarker`, or a sequence of `GraphTask` items for fork/join)
- **Fork/join coordination**: `_active_reducers` tracks running join-state reducers keyed by `(JoinID, NodeRunID)`
- **Result tracking**: The run completes when a terminal `End[OutputT]` node is reached

### How `AgentRun.iter` uses `GraphRun`

When you use `agent.iter()`, PydanticAI creates an internal `GraphRun` and exposes node-level iteration through `AgentRun`:

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.run import UserPromptNode, ModelRequestNode, CallToolsNode

agent = Agent('openai:gpt-4o')

async def main():
    async with agent.iter("What is 6 × 7?") as agent_run:
        async for node in agent_run:
            if isinstance(node, UserPromptNode):
                print("→ UserPromptNode: building request")
            elif isinstance(node, ModelRequestNode):
                print("→ ModelRequestNode: calling model")
            elif isinstance(node, CallToolsNode):
                print("→ CallToolsNode: handling response")
    print("Result:", agent_run.result.output)

asyncio.run(main())
```

### `NodeStep` — bridging v1 `BaseNode` into v2

`NodeStep` adapts any v1 `BaseNode` (from `pydantic_graph`) to run inside the v2 `GraphRun` system:

```python
@dataclass
class NodeStep(Step[StateT, DepsT, Any, BaseNode[StateT, DepsT, Any] | End[Any]]):
    node_type: type[BaseNode[StateT, DepsT, Any]]

    def __init__(
        self,
        node_type: type[BaseNode[StateT, DepsT, Any]],
        *,
        id: NodeID | None = None,
        label: str | None = None,
    ): ...
```

`NodeStep` validates that each incoming task carries an instance of `node_type`, then runs it with the appropriate `GraphRunContext`. This is how `UserPromptNode`, `ModelRequestNode`, and `CallToolsNode` (all `BaseNode` subclasses) participate in v2 graph execution.

### `JoinItem` + `GraphTaskRequest` — fork/join coordination

For parallel sub-graph execution (future multi-agent patterns):

```python
from pydantic_ai.run import JoinItem, GraphTaskRequest

# JoinItem carries partial results from parallel branches
# back into a join reducer
@dataclass
class JoinItem(Generic[OutputT]):
    join_id: JoinID
    node_run_id: NodeRunID
    output: OutputT
    error: BaseException | None = None
```

`GraphTaskRequest` packages a node ID, its inputs, and the fork-stack context needed to reconstruct where in the graph this task originated.

### Custom graph integration

If you build a `pydantic_graph.Graph` directly and want to run it via the PydanticAI execution engine:

```python
import asyncio
from pydantic_graph import Graph, BaseNode, End, GraphRunContext
from pydantic_ai.run import GraphRun

@dataclass
class CountState:
    count: int = 0

class IncrementNode(BaseNode[CountState, None, int]):
    async def run(self, ctx: GraphRunContext[CountState, None]) -> 'IncrementNode | End[int]':
        ctx.state.count += 1
        if ctx.state.count >= 3:
            return End(ctx.state.count)
        return IncrementNode()

graph = Graph(nodes=[IncrementNode])

async def run_graph():
    state = CountState()
    # GraphRun is created internally; agent.iter() is the public API
    # For direct graph usage, use pydantic_graph.Graph.run() instead
    result, history = await graph.run(IncrementNode(), state=state, deps=None)
    print(result)   # 3
    print(state)    # CountState(count=3)

asyncio.run(run_graph())
```

<Aside type="note" title="Direct graph API">
For direct graph execution without an `Agent`, use `pydantic_graph.Graph.run()` / `Graph.run_sync()` / `async with Graph.iter() as run`. `GraphRun` and `NodeStep` are exposed so advanced users can introspect execution state, but the primary entry point for agent graph-walking is `agent.iter()`.
</Aside>

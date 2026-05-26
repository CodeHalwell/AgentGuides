---
title: "PydanticAI — Class Deep Dives Vol. 2"
description: "Source-verified deep dives into 10 more PydanticAI classes: FallbackModel, ModelProfile, WrapperToolset family, InstrumentationSettings, UploadedFile, TemplateStr, ProcessHistory/ReinjectSystemPrompt, AbstractCapability, EndStrategy/AgentRetries, ModelRequestContext."
sidebar:
  label: "Class deep dives (Vol. 2)"
  order: 22
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 1.102.0** source installed directly from PyPI. Class signatures, field names, and behaviour match the installed package at `1.102.0`.
</Aside>

Ten new classes picked from the `pydantic_ai` 1.102.0 source, covering production resilience,
model-capability profiles, toolset composition, OpenTelemetry instrumentation, file handling, Handlebars
templating, history-transformation capabilities, custom capability authoring, agent end strategies, and
per-request context hooks.

---

## 1. `FallbackModel` + `FallbackExceptionGroup`

**Module:** `pydantic_ai.models.fallback`  
**Export:** `from pydantic_ai import FallbackModel, FallbackExceptionGroup`

`FallbackModel` wraps two or more `Model` instances and tries them in order.
When the current model raises an exception *or* returns a response that a custom handler
rejects, execution falls through to the next candidate.

### Constructor

```python
FallbackModel(
    default_model: Model | KnownModelName | str,
    *fallback_models: Model | KnownModelName | str,
    fallback_on: FallbackOn = (ModelAPIError,),   # default: any API error
)
```

`fallback_on` accepts **any mix** of:

| Type | When triggered |
|------|----------------|
| `tuple[type[Exception], ...]` | Default — falls back when any listed exception is raised |
| `Callable[[Exception], bool]` | Custom exception handler — return `True` to fall back |
| `Callable[[ModelResponse], bool]` | **Response handler** — return `True` to reject the response and fall back |
| `Sequence` of the above | All types mixed freely |

Handler type is **auto-detected** by inspecting type hints on the first parameter:
if it is hinted `ModelResponse`, it's a response handler; otherwise it's an exception handler.

### Basic usage

```python
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel

model = FallbackModel(
    "openai:gpt-4o",              # try first
    "anthropic:claude-3-5-haiku", # fallback #1
    "google-gla:gemini-2.0-flash",# fallback #2
)

agent = Agent(model, instructions="Answer concisely.")
result = agent.run_sync("What is 2 + 2?")
print(result.data)
```

### Custom exception handler

```python
from pydantic_ai.exceptions import ModelAPIError

def _is_overloaded(exc: Exception) -> bool:
    """Fallback on 429 / 529 only, not on every API error."""
    if isinstance(exc, ModelAPIError):
        status = getattr(exc, "status_code", None)
        return status in {429, 529}
    return False

model = FallbackModel(
    "openai:gpt-4o",
    "anthropic:claude-3-5-sonnet-20241022",
    fallback_on=(_is_overloaded,),
)
```

### Response-quality fallback

```python
from pydantic_ai.messages import ModelResponse

def _response_too_short(response: ModelResponse) -> bool:
    """Reject one-word answers and try the next model."""
    for part in response.parts:
        if hasattr(part, "content") and isinstance(part.content, str):
            return len(part.content.split()) < 3
    return False

model = FallbackModel(
    "openai:gpt-4o-mini",
    "openai:gpt-4o",
    fallback_on=(_response_too_short,),
)
```

### Mixed `fallback_on`

```python
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelResponse

def _reject_empty(response: ModelResponse) -> bool:
    return not any(hasattr(p, "content") and p.content for p in response.parts)

model = FallbackModel(
    "openai:gpt-4o",
    "anthropic:claude-opus-4-7",
    fallback_on=[ModelAPIError, _reject_empty],  # exception type + response handler
)
```

### Catching `FallbackExceptionGroup`

When every model in the chain fails, a `FallbackExceptionGroup` (subclass of `ExceptionGroup`) is raised containing all individual exceptions.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel, FallbackExceptionGroup

model = FallbackModel("openai:gpt-4o", "anthropic:claude-3-5-haiku")
agent = Agent(model)

async def run():
    try:
        result = await agent.run("Hello")
        return result.data
    except* FallbackExceptionGroup as eg:
        for exc in eg.exceptions:
            print(f"  failed with: {type(exc).__name__}: {exc}")
        raise

asyncio.run(run())
```

### `model_name` / `model_id` properties

```python
model = FallbackModel("openai:gpt-4o", "anthropic:claude-3-5-haiku")
print(model.model_name)
# fallback:gpt-4o,claude-3-5-haiku-20241022
print(model.model_id)
# fallback:openai:gpt-4o,anthropic:claude-3-5-haiku-20241022
```

### Production pattern — tiered latency with budget guard

```python
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.models.fallback import FallbackModel

model = FallbackModel(
    "openai:gpt-4o-mini",   # cheap + fast
    "openai:gpt-4o",        # more capable
    "anthropic:claude-opus-4-7",  # last resort
)

agent = Agent(model, instructions="Answer briefly.")

result = agent.run_sync(
    "Explain quantum entanglement in one paragraph.",
    usage_limits=UsageLimits(response_tokens_limit=300),
)
print(result.data)
print("Used model:", result.usage().model_name if hasattr(result.usage(), "model_name") else "see span")
```

---

## 2. `ModelProfile` + `ModelProfileSpec` + `DEFAULT_PROFILE`

**Module:** `pydantic_ai.profiles`  
**Export:** `from pydantic_ai import ModelProfile, ModelProfileSpec, DEFAULT_PROFILE`

`ModelProfile` is a `@dataclass` that describes **how** a particular model family needs to be
handled: which output modes it supports, whether it can think, which native tools it supports,
and how its JSON schemas should be transformed.

### All fields (1.102.0)

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `supports_tools` | `bool` | `True` | Whether the model accepts tool definitions |
| `supports_tool_return_schema` | `bool` | `False` | Native structured tool returns |
| `supports_json_schema_output` | `bool` | `False` | Native JSON-schema output (`NativeOutput`) |
| `supports_json_object_output` | `bool` | `False` | JSON-mode output (`PromptedOutput`) |
| `supports_image_output` | `bool` | `False` | Image generation output |
| `supports_inline_system_prompts` | `bool` | `False` | `SystemPromptPart` allowed mid-conversation |
| `default_structured_output_mode` | `StructuredOutputMode` | `'tool'` | `'tool'` / `'native'` / `'prompted'` |
| `prompted_output_template` | `str` | `"Always respond with JSON..."` | Template for prompted-mode instructions |
| `native_output_requires_schema_in_instructions` | `bool` | `False` | Whether to add schema to instructions in native mode |
| `json_schema_transformer` | `type[JsonSchemaTransformer] \| None` | `None` | Per-model schema normaliser |
| `supports_thinking` | `bool` | `False` | Thinking/reasoning config accepted |
| `thinking_always_enabled` | `bool` | `False` | Model always thinks (o-series, R1) |
| `thinking_tags` | `tuple[str, str]` | `('<think>', '</think>')` | Tag pair for thinking extraction |
| `ignore_streamed_leading_whitespace` | `bool` | `False` | Strip `<think>\n</think>` artefacts |
| `supported_native_tools` | `frozenset[type[AbstractNativeTool]]` | all built-ins | Which built-in tools work with this model |

### Reading the active profile for a model

```python
from pydantic_ai.models.openai import OpenAIChatModel

model = OpenAIChatModel("gpt-4o")
profile = model.profile
print(profile.supports_json_schema_output)   # True for gpt-4o
print(profile.supports_thinking)             # False
print(profile.default_structured_output_mode) # 'native' or 'tool'
```

### Writing a custom profile

```python
from pydantic_ai.profiles import ModelProfile

# Profile for a hypothetical on-prem model that supports tools but not structured output
my_profile = ModelProfile(
    supports_tools=True,
    supports_json_schema_output=False,
    supports_json_object_output=True,
    default_structured_output_mode="prompted",
    supports_thinking=False,
)
```

### `ModelProfileSpec` — profile or factory

`ModelProfileSpec` is `ModelProfile | Callable[[str], ModelProfile | None]`.
Pass it to `OpenAIChatModel(profile=...)` (or any provider model) to override the built-in profile:

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import ModelProfile

# Override for a custom endpoint that looks like OpenAI but doesn't support native JSON schema
strict_profile = ModelProfile(
    supports_json_schema_output=False,
    default_structured_output_mode="tool",
)

model = OpenAIChatModel(
    "my-company/llama-3-70b",
    base_url="https://my-llm-gateway.internal/v1",
    profile=strict_profile,
)
```

### Factory form of `ModelProfileSpec`

```python
from pydantic_ai.profiles import ModelProfile, DEFAULT_PROFILE
from pydantic_ai.models.openai import OpenAIChatModel

def _profile_factory(model_name: str) -> ModelProfile | None:
    """Return a custom profile for 'o-' series, None for everything else (keep built-in)."""
    if model_name.startswith("o"):
        return ModelProfile(
            supports_thinking=True,
            thinking_always_enabled=True,
            default_structured_output_mode="native",
        )
    return None

model = OpenAIChatModel("o3-mini", profile=_profile_factory)
```

### `DEFAULT_PROFILE`

`DEFAULT_PROFILE` is the baseline `ModelProfile` instance used when no provider-specific profile exists.
Inspect it to understand the conservative defaults:

```python
from pydantic_ai import DEFAULT_PROFILE

print(DEFAULT_PROFILE.supports_tools)           # True
print(DEFAULT_PROFILE.supports_thinking)        # False
print(DEFAULT_PROFILE.supported_native_tools)   # frozenset of all built-in native tool types
```

### Capability-gating pattern

```python
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.profiles import ModelProfile
from pydantic import BaseModel

class Summary(BaseModel):
    headline: str
    bullets: list[str]

def build_agent(model_name: str) -> Agent:
    from pydantic_ai.models.openai import OpenAIChatModel
    m = OpenAIChatModel(model_name)

    if m.profile.supports_json_schema_output:
        output = NativeOutput(Summary)
    else:
        output = Summary  # falls back to ToolOutput

    return Agent(m, output_type=output)
```

---

## 3. `WrapperToolset` + `RenamedToolset` + `PrefixedToolset` + `SetMetadataToolset`

**Module:** `pydantic_ai.toolsets`  
**Exports:** all from `pydantic_ai`

These four classes form the **wrapper toolset family** — a composition layer over `AbstractToolset`.
`WrapperToolset` is the base; the other three are ready-made specialisations.

### `WrapperToolset` — base class for decorating toolsets

```python
from dataclasses import dataclass, replace
from typing import Any
from pydantic_ai import RunContext
from pydantic_ai.toolsets.wrapper import WrapperToolset
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool

@dataclass
class LoggingToolset(WrapperToolset[Any]):
    """A wrapper that logs every tool call."""

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> Any:
        print(f"[tool] {name}({tool_args})")
        result = await super().call_tool(name, tool_args, ctx, tool)
        print(f"[tool] {name} → {result!r}")
        return result
```

```python
from pydantic_ai import Agent, FunctionToolset

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

base = FunctionToolset([add])
logged = LoggingToolset(wrapped=base)

agent = Agent("openai:gpt-4o-mini", toolsets=[logged])
result = agent.run_sync("What is 7 + 8?")
print(result.data)
# [tool] add({'a': 7, 'b': 8})
# [tool] add → 15
```

### `RenamedToolset` — rename specific tools

Use `RenamedToolset` to present different names to the model while keeping the original
implementation unchanged. The `name_map` maps **new** names to **original** names.

```python
from pydantic_ai import Agent, FunctionToolset, RenamedToolset

def search_products(query: str) -> list[str]:
    """Search the product catalogue."""
    return [f"Product matching '{query}'"]

def get_price(product_id: str) -> float:
    """Retrieve the price for a product."""
    return 9.99

base = FunctionToolset([search_products, get_price])

renamed = RenamedToolset(
    wrapped=base,
    name_map={
        "find_products": "search_products",   # new → original
        "check_price":   "get_price",
    },
)

agent = Agent("openai:gpt-4o-mini", toolsets=[renamed])
# The model sees tools named "find_products" and "check_price"
```

### `PrefixedToolset` — namespace isolation

Prepend a prefix to all tool names in a toolset, preventing name collisions when merging
multiple toolsets from different domains.

```python
from pydantic_ai import Agent, FunctionToolset, PrefixedToolset, CombinedToolset

def search(query: str) -> list[str]:
    """Search the web."""
    return [f"Web result: {query}"]

def search_internal(query: str) -> list[str]:
    """Search the internal knowledge base."""
    return [f"KB result: {query}"]

web_tools = PrefixedToolset(wrapped=FunctionToolset([search]), prefix="web")
kb_tools  = PrefixedToolset(wrapped=FunctionToolset([search_internal]), prefix="kb")

# Model sees: web_search, kb_search_internal
agent = Agent("openai:gpt-4o-mini", toolsets=[CombinedToolset([web_tools, kb_tools])])
```

### `SetMetadataToolset` — tag all tools for downstream processing

Attaches arbitrary key-value metadata to every `ToolDefinition` in the wrapped toolset.
Useful for RBAC gating, audit tagging, or passing hints to `PreparedToolset`.

```python
from pydantic_ai import Agent, FunctionToolset, SetMetadataToolset

def delete_record(record_id: str) -> str:
    """Delete a database record."""
    return f"Deleted {record_id}"

destructive_tools = SetMetadataToolset(
    wrapped=FunctionToolset([delete_record]),
    metadata={"tier": "destructive", "requires_approval": True},
)

agent = Agent("openai:gpt-4o-mini", toolsets=[destructive_tools])
```

### Combining all four in one pipeline

```python
from pydantic_ai import (
    Agent, FunctionToolset, RenamedToolset, PrefixedToolset,
    SetMetadataToolset, FilteredToolset, CombinedToolset,
)

def query_db(sql: str) -> list[dict]:
    """Run a read-only SQL query."""
    return [{"result": "data"}]

def write_db(sql: str) -> str:
    """Execute a write SQL statement."""
    return "ok"

# Rename for clarity
reads = RenamedToolset(
    wrapped=FunctionToolset([query_db]),
    name_map={"db_read": "query_db"},
)

# Prefix writes to make their nature explicit
writes = PrefixedToolset(
    wrapped=FunctionToolset([write_db]),
    prefix="unsafe",
)

# Tag everything with the data-tier
all_db = SetMetadataToolset(
    wrapped=CombinedToolset([reads, writes]),
    metadata={"layer": "database"},
)

# Only expose writes when explicitly permitted via deps
def _allow_writes(ctx, td):
    return not td.name.startswith("unsafe") or getattr(ctx.deps, "can_write", False)

safe_db = FilteredToolset(wrapped=all_db, filter=_allow_writes)

agent = Agent("openai:gpt-4o-mini", toolsets=[safe_db])
```

---

## 4. `InstrumentationSettings` + `InstrumentedModel`

**Module:** `pydantic_ai._instrumentation`, `pydantic_ai.models.instrumented`  
**Export:** `from pydantic_ai import InstrumentationSettings`

`InstrumentationSettings` configures how agent runs emit OpenTelemetry traces, spans, and metrics.
Pass it to `Agent(instrument=...)` or `Agent.instrument_all()`.

### Constructor

```python
InstrumentationSettings(
    *,
    tracer_provider: TracerProvider | None = None,   # defaults to global
    meter_provider: MeterProvider | None = None,     # defaults to global
    include_binary_content: bool = True,
    include_content: bool = True,
    version: Literal[1, 2, 3, 4, 5] = DEFAULT_INSTRUMENTATION_VERSION,
    event_mode: Literal['attributes', 'logs'] = 'attributes',  # v1 only
    logger_provider: LoggerProvider | None = None,             # v1 only
    use_aggregated_usage_attribute_names: bool = False,
)
```

### Version comparison table

| Version | Key behaviour |
|---------|--------------|
| `1` | Legacy event-based OTel GenAI spec. **Deprecated**, will be removed. |
| `2` | New OTel GenAI spec. Messages in `gen_ai.input.messages` / `gen_ai.output.messages` attributes. |
| `3` | Like v2 + thinking token support in traces. |
| `4` | Like v3 + multimodal content (images/audio/video) using `type='uri'` / `type='blob'`. |
| `5` | Like v4, but `CallDeferred` / `ApprovalRequired` no longer set span status to `ERROR`. |

### Basic usage with Logfire

```python
import logfire
from pydantic_ai import Agent
from pydantic_ai import InstrumentationSettings

logfire.configure()  # sets the global tracer provider

agent = Agent(
    "openai:gpt-4o-mini",
    instrument=InstrumentationSettings(
        version=5,                    # latest spec
        include_content=True,         # record prompt/completion text
        include_binary_content=False, # skip base64 image blobs
    ),
)

result = agent.run_sync("Summarise the Pydantic docs in 3 bullets.")
print(result.data)
```

### Instrument all agents in a process

```python
from pydantic_ai import Agent, InstrumentationSettings

# Apply to every agent created after this call
Agent.instrument_all(
    InstrumentationSettings(version=5, include_content=False)
)
```

### Privacy mode — scrub content

```python
from pydantic_ai import Agent, InstrumentationSettings

scrubbed = InstrumentationSettings(
    include_content=False,       # no prompt/completion text in spans
    include_binary_content=False,
)

agent = Agent("openai:gpt-4o", instrument=scrubbed)
```

### Custom tracer provider (e.g. OTLP exporter)

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from pydantic_ai import Agent, InstrumentationSettings

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

agent = Agent(
    "openai:gpt-4o-mini",
    instrument=InstrumentationSettings(
        tracer_provider=provider,
        version=5,
    ),
)
```

### `InstrumentedModel` — wrap any model

`InstrumentedModel` wraps a `Model` to add OTel instrumentation without touching `Agent`:

```python
from pydantic_ai import Agent, InstrumentationSettings
from pydantic_ai.models.instrumented import InstrumentedModel
from pydantic_ai.models.openai import OpenAIChatModel

base_model = OpenAIChatModel("gpt-4o-mini")
instrumented = InstrumentedModel(
    wrapped=base_model,
    options=InstrumentationSettings(version=5, include_content=True),
)

agent = Agent(instrumented)
```

### Aggregated usage — avoid double-counting

When a backend sums metrics across parent + child spans, set `use_aggregated_usage_attribute_names=True`
to emit `gen_ai.aggregated_usage.*` instead of the standard `gen_ai.usage.*`:

```python
settings = InstrumentationSettings(
    use_aggregated_usage_attribute_names=True,  # prevents double-count in Datadog/Grafana
    version=5,
)
```

---

## 5. `UploadedFile`

**Module:** `pydantic_ai.messages`  
**Export:** `from pydantic_ai import UploadedFile`

`UploadedFile` is a lightweight reference to a file already uploaded to a provider's storage.
Instead of sending binary content on every request, the model receives a durable file ID.

### Provider support matrix

| Provider | ID format | Notes |
|----------|-----------|-------|
| `OpenAIChatModel` | `file-abc123` | OpenAI Files API |
| `OpenAIResponsesModel` | `file-abc123` | Same Files API |
| `AnthropicModel` | Provider-specific ID | Anthropic Files API |
| `BedrockConverseModel` | `s3://bucket/key` | S3 URI |
| `GoogleModel` (Gemini API) | `https://generativelanguage.googleapis.com/...` | Files API URI |
| `GoogleModel` (Vertex AI) | `gs://bucket/path` | GCS URI |
| `XaiModel` | Provider-specific ID | xAI Files API |

### Constructor

```python
UploadedFile(
    file_id: str,
    provider_name: UploadedFileProviderName,   # e.g. "openai", "anthropic"
    *,
    media_type: str | None = None,   # inferred from file_id extension if omitted
    vendor_metadata: dict | None = None,
    identifier: str | None = None,   # logical name exposed to the model
)
```

### Upload + reference with OpenAI

```python
import asyncio
from openai import AsyncOpenAI
from pydantic_ai import Agent, UploadedFile
from pydantic_ai.messages import UserPromptPart

async def main():
    client = AsyncOpenAI()

    # Upload once
    with open("annual_report.pdf", "rb") as f:
        uploaded = await client.files.create(file=f, purpose="assistants")

    file_ref = UploadedFile(
        file_id=uploaded.id,
        provider_name="openai",
        media_type="application/pdf",
    )

    agent = Agent("openai:gpt-4o")
    result = await agent.run(
        [
            "Summarise the key financials from this report:",
            file_ref,   # pass directly in the message list
        ]
    )
    print(result.data)

asyncio.run(main())
```

### Google Gemini Files API

```python
from pydantic_ai import Agent, UploadedFile

agent = Agent("google-gla:gemini-2.0-flash")

# Assume file URI returned from google.generativeai.upload_file(...)
gemini_file_uri = "https://generativelanguage.googleapis.com/v1beta/files/abc123"

result = agent.run_sync(
    [
        "Describe what you see in this video:",
        UploadedFile(
            file_id=gemini_file_uri,
            provider_name="google",
            media_type="video/mp4",
            vendor_metadata={"video_metadata": {"fps": 1}},
        ),
    ]
)
```

### Amazon Bedrock (S3 URI)

```python
from pydantic_ai import Agent, UploadedFile

agent = Agent("bedrock:us.amazon.nova-pro-v1:0")

result = agent.run_sync(
    [
        "Analyse the data in this CSV:",
        UploadedFile(
            file_id="s3://my-bucket/data/sales_q1.csv",
            provider_name="bedrock",
            media_type="text/csv",
        ),
    ]
)
```

### `media_type` inference

If you omit `media_type`, it is inferred from the file extension in `file_id` using Python's
`mimetypes` module. For opaque IDs like `file-abc123` the default is `application/octet-stream`.

```python
from pydantic_ai import UploadedFile

f = UploadedFile("report.pdf", "openai")
print(f.media_type)  # application/pdf

f2 = UploadedFile("image.png", "anthropic")
print(f2.media_type)  # image/png

f3 = UploadedFile("file-abc123", "openai")
print(f3.media_type)  # application/octet-stream (opaque ID)
```

### `identifier` property — letting the model refer back

When `UploadedFile` is returned by a tool, its `identifier` is automatically passed to the
model so the model can reference the file in follow-up tool arguments.

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext, UploadedFile, FunctionToolset

@dataclass
class FileDeps:
    file_store: dict[str, bytes]  # file_id → content

def upload_user_file(ctx: RunContext[FileDeps], filename: str) -> UploadedFile:
    """Upload a file and return a reference."""
    data = ctx.deps.file_store[filename]
    # ... actual upload call ...
    file_id = f"file-{filename}-001"
    return UploadedFile(
        file_id=file_id,
        provider_name="openai",
        identifier=filename,  # the model will refer to this file by this name
    )
```

---

## 6. `TemplateStr`

**Module:** `pydantic_ai._template`  
**Export:** `from pydantic_ai import TemplateStr`

`TemplateStr` is a **Handlebars template string** that renders against `RunContext.deps`.
Use it to make system prompts data-driven without writing a `@agent.system_prompt` function.

Requires `pip install "pydantic-handlebars"` (optional dependency).

### Basic usage

```python
from dataclasses import dataclass
from pydantic_ai import Agent, TemplateStr

@dataclass
class CustomerDeps:
    name: str
    tier: str
    language: str

agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=CustomerDeps,
    instructions=TemplateStr(
        "You are a support agent. The customer is {{name}} ({{tier}} tier). "
        "Always respond in {{language}}."
    ),
)

result = agent.run_sync(
    "My order hasn't arrived.",
    deps=CustomerDeps(name="Alice", tier="gold", language="French"),
)
print(result.data)
```

### Conditionals and loops

Handlebars `{{#if}}` and `{{#each}}` are fully supported:

```python
from dataclasses import dataclass, field
from pydantic_ai import Agent, TemplateStr

@dataclass
class AgentDeps:
    user_name: str
    allowed_tools: list[str] = field(default_factory=list)
    debug_mode: bool = False

TEMPLATE = """
You are a helpful assistant for {{user_name}}.
{{#if debug_mode}}
DEBUG MODE ENABLED — explain every step you take.
{{/if}}
{{#if allowed_tools}}
You have access to the following tools:
{{#each allowed_tools}}
- {{this}}
{{/each}}
{{/if}}
""".strip()

agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=AgentDeps,
    instructions=TemplateStr(TEMPLATE),
)

result = agent.run_sync(
    "How do I reset my password?",
    deps=AgentDeps(
        user_name="Bob",
        allowed_tools=["search_kb", "create_ticket"],
        debug_mode=True,
    ),
)
```

### Standalone rendering

`TemplateStr` can be rendered outside of an agent:

```python
from dataclasses import dataclass
from pydantic_ai import TemplateStr

@dataclass
class Config:
    company: str
    region: str

tmpl = TemplateStr(
    "Welcome to {{company}} ({{region}} region).",
    deps_type=Config,
)

print(tmpl.render(Config(company="Acme", region="EU")))
# Welcome to Acme (EU region).
```

### Automatic validation in `Agent`

When you pass a string containing `{{` to `Agent(instructions=...)`, PydanticAI's Pydantic
validation context automatically wraps it in a `TemplateStr` — you don't need to do it
explicitly in most cases:

```python
# This works — the string is auto-promoted to TemplateStr during Agent init
agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=CustomerDeps,
    instructions="Hello {{name}}, I am your assistant.",
)
```

### Using `TemplateStr` in `AgentSpec` YAML

```yaml
# agent.yaml
instructions: |
  You are a {{role}} assistant at {{company}}.
  {{#if verbose}}Always explain your reasoning step by step.{{/if}}
```

---

## 7. `ProcessHistory` + `ReinjectSystemPrompt`

**Module:** `pydantic_ai.capabilities.process_history`  
**Export:** `from pydantic_ai.capabilities import ProcessHistory, ReinjectSystemPrompt`

Both are `AbstractCapability` subclasses that fire **before every model request** to transform
the message history.

### `ProcessHistory` — arbitrary history transformation

```python
@dataclass
class ProcessHistory(AbstractCapability[AgentDepsT]):
    processor: HistoryProcessorFunc[AgentDepsT]

    async def before_model_request(self, ctx, request_context) -> ModelRequestContext: ...
```

`HistoryProcessorFunc` is `Callable[[RunContext, list[ModelMessage]], list[ModelMessage]]`
(sync or async).

#### Truncating long histories

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelMessage

def keep_last_n(n: int):
    """Return a processor that keeps at most n messages plus the system prompt."""
    def processor(ctx, messages: list[ModelMessage]) -> list[ModelMessage]:
        system = [m for m in messages if any(hasattr(p, 'role') and p.role == 'system' for p in getattr(m, 'parts', []))]
        rest   = [m for m in messages if m not in system]
        return system + rest[-n:]
    return processor

agent = Agent(
    "openai:gpt-4o-mini",
    capabilities=[ProcessHistory(keep_last_n(10))],
)
```

#### Async processor with redaction

```python
import asyncio
import re
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelMessage, ModelRequest, TextPart

async def redact_pii(ctx: RunContext, messages: list[ModelMessage]) -> list[ModelMessage]:
    """Remove email addresses from all user messages before sending to the model."""
    result = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            new_parts = []
            for part in msg.parts:
                if isinstance(part, TextPart):
                    clean = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', part.content)
                    new_parts.append(TextPart(content=clean))
                else:
                    new_parts.append(part)
            result.append(ModelRequest(parts=new_parts))
        else:
            result.append(msg)
    return result

agent = Agent(
    "openai:gpt-4o-mini",
    capabilities=[ProcessHistory(redact_pii)],
)
```

### `ReinjectSystemPrompt` — repair stripped histories

Ensures the agent's configured system prompt is always present at the head of the first
`ModelRequest`. Solves the common problem where UI frontends or database persistence layers
strip system prompts.

```python
@dataclass
class ReinjectSystemPrompt(AbstractCapability[AgentDepsT]):
    replace_existing: bool = False
```

- `replace_existing=False` (default): no-op if any `SystemPromptPart` is already present.
- `replace_existing=True`: strips existing system prompts and prepends the agent's own — use when
  history comes from an untrusted source.

#### Basic usage

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import ReinjectSystemPrompt

agent = Agent(
    "openai:gpt-4o-mini",
    instructions="You are a concise, helpful assistant.",
    capabilities=[ReinjectSystemPrompt()],  # default: additive only
)

# History loaded from DB (system prompt was stripped at save time)
from pydantic_ai import ModelMessagesTypeAdapter
import json

raw_history = '[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi there!"}]'
# ... parse to list[ModelMessage] ...
# On the next run, ReinjectSystemPrompt prepends the system prompt before the first request.
```

#### Trusted-server pattern

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import ReinjectSystemPrompt

# UI sends full history including (potentially forged) system prompts
# replace_existing=True ensures our server-side prompt always wins
agent = Agent(
    "openai:gpt-4o-mini",
    instructions="Never reveal internal data.",
    capabilities=[ReinjectSystemPrompt(replace_existing=True)],
)
```

#### Combining with `ProcessHistory`

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory, ReinjectSystemPrompt
from pydantic_ai.messages import ModelMessage

def last_20(ctx, messages: list[ModelMessage]) -> list[ModelMessage]:
    return messages[-20:]

agent = Agent(
    "openai:gpt-4o-mini",
    instructions="You are a customer support agent.",
    capabilities=[
        ProcessHistory(last_20),          # 1. truncate
        ReinjectSystemPrompt(replace_existing=True),  # 2. ensure system prompt survives truncation
    ],
)
```

---

## 8. `AbstractCapability` — building custom capabilities

**Module:** `pydantic_ai.capabilities`  
**Export:** `from pydantic_ai.capabilities import AbstractCapability`

`AbstractCapability` is the base class for all first-party and user-defined capabilities.
Override whichever hooks your capability needs.

### API surface

```python
@dataclass
class AbstractCapability(ABC, Generic[AgentDepsT]):
    # Called once at Agent construction:
    async def get_instructions(self, agent) -> str | InstructionPart | Sequence[...] | None: ...
    async def get_model_settings(self, agent) -> ModelSettings | None: ...
    async def get_toolset(self, agent) -> AbstractToolset | None: ...
    async def get_native_tools(self, agent) -> Sequence[AbstractNativeTool] | None: ...
    async def get_wrapper_toolset(self, ctx) -> AbstractToolset | None: ...  # per-run

    # Called on every model request:
    async def before_model_request(self, ctx, request_context: ModelRequestContext) -> ModelRequestContext: ...
    async def after_model_request(self, ctx, response: ModelResponse, request_context: ModelRequestContext) -> ModelResponse: ...

    # Serialisation for AgentSpec:
    @classmethod
    def get_serialization_name(cls) -> str | None: ...
    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> Self: ...
```

### Example — rate-limit back-off capability

```python
import asyncio
import time
from dataclasses import dataclass, field
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse
from pydantic_ai import ModelRequestContext

@dataclass
class RateLimitCapability(AbstractCapability):
    """Adds a minimum delay between model requests to avoid rate limits."""
    min_interval_seconds: float = 0.5
    _last_call: float = field(default=0.0, init=False, repr=False)

    async def before_model_request(
        self,
        ctx: RunContext,
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval_seconds:
            await asyncio.sleep(self.min_interval_seconds - elapsed)
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext,
        response: ModelResponse,
        request_context: ModelRequestContext,
    ) -> ModelResponse:
        self._last_call = time.monotonic()
        return response
```

```python
from pydantic_ai import Agent

agent = Agent(
    "openai:gpt-4o-mini",
    capabilities=[RateLimitCapability(min_interval_seconds=0.25)],
)
```

### Example — capability that injects dynamic instructions

```python
from dataclasses import dataclass
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability

@dataclass
class CurrentDateCapability(AbstractCapability):
    """Injects the current date into the system prompt."""

    async def get_instructions(self, agent) -> str:
        from datetime import date
        return f"Today's date is {date.today().isoformat()}."
```

### Example — capability that adds a toolset

```python
from dataclasses import dataclass
from pydantic_ai import FunctionToolset
from pydantic_ai.capabilities import AbstractCapability

def get_current_time() -> str:
    """Return the current UTC time."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

@dataclass
class TimeCapability(AbstractCapability):
    """Provides a get_current_time tool."""

    async def get_toolset(self, agent):
        return FunctionToolset([get_current_time])
```

### Composing capabilities on `Agent`

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import ReinjectSystemPrompt

agent = Agent(
    "openai:gpt-4o-mini",
    instructions="You are a helpful assistant.",
    capabilities=[
        CurrentDateCapability(),
        TimeCapability(),
        RateLimitCapability(min_interval_seconds=0.1),
        ReinjectSystemPrompt(),
    ],
)
```

---

## 9. `EndStrategy` + `AgentRetries`

**Module:** `pydantic_ai.settings`, `pydantic_ai`  
**Exports:** `from pydantic_ai import EndStrategy, AgentRetries`

### `EndStrategy` — when to stop iterating

`EndStrategy` is `Literal['early', 'graceful', 'exhaustive']`. Pass it to `Agent.run(end_strategy=...)`.

| Value | Behaviour |
|-------|-----------|
| `'early'` | Stop the moment the model produces a valid final result, even if tool calls are pending. **Default** |
| `'graceful'` | Complete in-flight tool calls that started before the final result, then stop. |
| `'exhaustive'` | Run every tool call the model requested, even after a final result is already available. |

```python
from pydantic_ai import Agent

agent = Agent(
    "openai:gpt-4o-mini",
    instructions="Use the tools and then give a summary.",
)
```

#### `'early'` — fastest, minimal side effects

```python
result = agent.run_sync(
    "Do a search and give me 3 results.",
    end_strategy="early",   # stop as soon as a final result is returned
)
```

#### `'graceful'` — complete started work

```python
result = agent.run_sync(
    "Fetch these 3 URLs and summarise.",
    end_strategy="graceful",  # finish tool calls already dispatched before stopping
)
```

#### `'exhaustive'` — run every tool the model requested

```python
result = agent.run_sync(
    "Run all the following checks and report.",
    end_strategy="exhaustive",  # process every tool call even after result is known
)
```

#### Choosing the right strategy

```python
from pydantic_ai import Agent

# For read-only queries: 'early' is best — minimal latency, no extra side effects
read_agent = Agent("openai:gpt-4o-mini")
r = read_agent.run_sync("What is Python?", end_strategy="early")

# For pipelines that write data: 'graceful' ensures atomicity of started work
write_agent = Agent("openai:gpt-4o-mini")
r = write_agent.run_sync(
    "Update the order status and notify the customer.",
    end_strategy="graceful",
)

# For audit/compliance scenarios: 'exhaustive' ensures all intended actions execute
audit_agent = Agent("openai:gpt-4o-mini")
r = audit_agent.run_sync(
    "Run all compliance checks.",
    end_strategy="exhaustive",
)
```

### `AgentRetries` — per-agent retry counts

`AgentRetries` is `int | None`. Set it via `Agent(retries=...)` to control how many times
the agent re-attempts a run when `ModelRetry` is raised by a tool.

```python
from pydantic_ai import Agent, RunContext, ModelRetry

def validate_answer(ctx: RunContext[None], answer: str) -> str:
    """Validate and return the answer, retrying if invalid."""
    if len(answer) < 10:
        raise ModelRetry("Answer too short. Please elaborate.")
    return answer

agent = Agent(
    "openai:gpt-4o-mini",
    retries=3,   # allow up to 3 ModelRetry cycles before raising
)
```

#### Per-run override

```python
result = agent.run_sync(
    "Give me a detailed answer.",
    max_result_retries=5,  # override at run time (takes precedence over Agent.retries)
)
```

#### Zero retries — fail fast

```python
strict_agent = Agent("openai:gpt-4o-mini", retries=0)
# Any ModelRetry raised by a tool immediately propagates as an error
```

#### Combining `retries` with `end_strategy`

```python
from pydantic_ai import Agent, UsageLimits

agent = Agent(
    "openai:gpt-4o-mini",
    retries=2,
)

result = agent.run_sync(
    "Validate and return a 50-word summary.",
    end_strategy="graceful",
    usage_limits=UsageLimits(request_limit=10),
)
```

---

## 10. `ModelRequestContext`

**Module:** `pydantic_ai.run` (re-exported)  
**Export:** `from pydantic_ai import ModelRequestContext`

`ModelRequestContext` is the mutable dataclass passed to both
`AbstractCapability.before_model_request` and `AbstractCapability.after_model_request`.
Mutating its fields lets capabilities modify what the model sees on each request.

### Fields

```python
@dataclass(kw_only=True)
class ModelRequestContext:
    model: Model                                    # the model about to be called
    messages: list[ModelMessage]                    # mutate to change history
    model_settings: ModelSettings | None            # overrideable per-request
    model_request_parameters: ModelRequestParameters  # tools, output schema, etc.
```

### Reading context in `before_model_request`

```python
from dataclasses import dataclass
from pydantic_ai import RunContext, ModelRequestContext
from pydantic_ai.capabilities import AbstractCapability

@dataclass
class DebugCapability(AbstractCapability):
    """Log the number of messages and tools on each request."""

    async def before_model_request(
        self,
        ctx: RunContext,
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        n_messages = len(request_context.messages)
        n_tools    = len(request_context.model_request_parameters.tools or [])
        print(f"[request] model={request_context.model.model_name}, "
              f"messages={n_messages}, tools={n_tools}")
        return request_context
```

### Mutating `model_settings` per-request

```python
from dataclasses import dataclass, replace
from pydantic_ai import RunContext, ModelRequestContext, ModelSettings
from pydantic_ai.capabilities import AbstractCapability

@dataclass
class DynamicTemperatureCapability(AbstractCapability):
    """Lower temperature for requests that already have many messages (more context = more determinism)."""

    async def before_model_request(
        self,
        ctx: RunContext,
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        n = len(request_context.messages)
        temp = max(0.0, 0.7 - 0.05 * n)  # cool down as conversation grows
        settings = request_context.model_settings or ModelSettings()
        request_context.model_settings = replace(settings, temperature=temp)
        return request_context
```

### Filtering tools per-request via `model_request_parameters`

```python
from dataclasses import dataclass, replace
from pydantic_ai import RunContext, ModelRequestContext
from pydantic_ai.capabilities import AbstractCapability

@dataclass
class PhaseGateCapability(AbstractCapability):
    """Expose only planning tools on the first 2 requests; all tools after."""

    async def before_model_request(
        self,
        ctx: RunContext,
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        params = request_context.model_request_parameters
        if ctx.run_step < 2 and params.tools:
            planning_only = [t for t in params.tools if "plan" in t.name]
            request_context.model_request_parameters = replace(
                params, tools=planning_only
            )
        return request_context
```

### Post-response inspection in `after_model_request`

```python
from dataclasses import dataclass
from pydantic_ai import RunContext, ModelRequestContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse

@dataclass
class TokenBudgetLogger(AbstractCapability):
    """Log token usage after each model request."""

    async def after_model_request(
        self,
        ctx: RunContext,
        response: ModelResponse,
        request_context: ModelRequestContext,
    ) -> ModelResponse:
        if response.usage:
            print(
                f"[tokens] input={response.usage.input_tokens}, "
                f"output={response.usage.output_tokens}"
            )
        return response
```

### Production example — full capability pipeline

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import ReinjectSystemPrompt, ProcessHistory
from pydantic_ai.messages import ModelMessage

def keep_last_15(ctx, msgs: list[ModelMessage]) -> list[ModelMessage]:
    return msgs[-15:]

agent = Agent(
    "openai:gpt-4o-mini",
    instructions="You are a precise data analyst.",
    capabilities=[
        ProcessHistory(keep_last_15),
        ReinjectSystemPrompt(replace_existing=True),
        DynamicTemperatureCapability(),
        TokenBudgetLogger(),
        DebugCapability(),
    ],
)
```

---

## Quick-reference table

| Class | Module | Key use |
|-------|--------|---------|
| `FallbackModel` | `pydantic_ai.models.fallback` | Chain models; fall back on exception or bad response |
| `FallbackExceptionGroup` | `pydantic_ai.models.fallback` | All models failed; contains every individual error |
| `ModelProfile` | `pydantic_ai.profiles` | Capability flags per model family |
| `ModelProfileSpec` | `pydantic_ai.profiles` | Profile or factory callable |
| `DEFAULT_PROFILE` | `pydantic_ai.profiles` | Conservative baseline profile |
| `WrapperToolset` | `pydantic_ai.toolsets.wrapper` | Base class for toolset decorators |
| `RenamedToolset` | `pydantic_ai.toolsets` | Rename specific tools without touching implementation |
| `PrefixedToolset` | `pydantic_ai.toolsets` | Namespace isolation via prefix |
| `SetMetadataToolset` | `pydantic_ai.toolsets.set_metadata` | Tag all tools with metadata |
| `InstrumentationSettings` | `pydantic_ai._instrumentation` | OTel/Logfire trace config, 5 versions |
| `InstrumentedModel` | `pydantic_ai.models.instrumented` | Wrap any model with OTel |
| `UploadedFile` | `pydantic_ai.messages` | Cross-provider file reference by ID |
| `TemplateStr` | `pydantic_ai._template` | Handlebars system-prompt templating |
| `ProcessHistory` | `pydantic_ai.capabilities` | Transform message history before each request |
| `ReinjectSystemPrompt` | `pydantic_ai.capabilities` | Repair stripped system prompts |
| `AbstractCapability` | `pydantic_ai.capabilities` | Base class for custom capabilities |
| `EndStrategy` | `pydantic_ai` | `'early'` / `'graceful'` / `'exhaustive'` loop termination |
| `AgentRetries` | `pydantic_ai` | Max `ModelRetry` cycles (`int \| None`) |
| `ModelRequestContext` | `pydantic_ai` | Mutable per-request context for capability hooks |

---

## Revision history

| Date | Version | Summary |
|------|---------|---------|
| 2026-05-26 | pydantic-ai 1.102.0 | New guide. Ten classes source-verified against installed 1.102.0: `FallbackModel` + `FallbackExceptionGroup`, `ModelProfile` + `ModelProfileSpec` + `DEFAULT_PROFILE`, `WrapperToolset` + `RenamedToolset` + `PrefixedToolset` + `SetMetadataToolset`, `InstrumentationSettings` + `InstrumentedModel`, `UploadedFile`, `TemplateStr`, `ProcessHistory` + `ReinjectSystemPrompt`, `AbstractCapability`, `EndStrategy` + `AgentRetries`, `ModelRequestContext`. |

---
title: "PydanticAI Class Deep Dives Vol. 42"
description: "Source-verified deep dives into 10 pydantic-ai 2.31.0 class groups: AgentSpec (declarative YAML/JSON agent configuration — from_file, from_spec, to_file, deps_schema, TemplateStr integration), TemplateStr (Handlebars templates rendered against RunContext.deps — auto-compilation, schema validation, standalone rendering), NativeOutput + PromptedOutput (output-mode markers — native structured outputs vs. prompt-driven extraction, custom name/description/template), StructuredDict (schema-attached dict factory — object-schema validation, no-Pydantic-model shortcut, union output), SkipModelRequest + SkipToolExecution + SkipToolValidation (hook control-flow exceptions — injecting synthetic responses, short-circuiting tool execution, pre-validated args), SelectModel + ModelSelectionContext (per-step model switching — deps-driven routing, step-count escalation, usage-budget fallback), DeferredLoadingToolset + IncludeReturnSchemasToolset (toolset modifiers — hiding tools until discovered, return-schema injection), AdvisorTool (executor/advisor model pairing — max_uses, max_tokens, OpenRouter gateway), CachePoint + ToolAvailabilityDeltaPart (prompt-caching markers — TTL control, incremental tool disclosure in streaming), native-tool suite: XSearchTool + ImageGenerationTool + WebFetchTool + MemoryTool + CodeExecutionTool + FileSearchTool."
sidebar:
  label: "Class deep dives (Vol. 42)"
  order: 68
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 2.31.0** source installed directly from PyPI. Every class signature, field name, and method in this volume reflects the 2.31.x API. Three examples per class group; all code blocks pass `ast.parse()` syntax validation. Live API calls are commented out — uncomment to run.
</Aside>

Ten class groups covering declarative agent specs (`AgentSpec`), Handlebars template strings (`TemplateStr`), structured-output mode markers (`NativeOutput` + `PromptedOutput`), schema-attached dict outputs (`StructuredDict`), hook control-flow exceptions (`SkipModelRequest` + `SkipToolExecution` + `SkipToolValidation`), per-step model switching (`SelectModel` + `ModelSelectionContext`), toolset modifiers (`DeferredLoadingToolset` + `IncludeReturnSchemasToolset`), the executor/advisor native tool (`AdvisorTool`), prompt-caching and tool-availability streaming (`CachePoint` + `ToolAvailabilityDeltaPart`), and the full provider-specific native-tool suite (`XSearchTool`, `ImageGenerationTool`, `WebFetchTool`, `MemoryTool`, `CodeExecutionTool`, `FileSearchTool`).

---

## 1. `AgentSpec`

**Source:** `pydantic_ai/agent/spec.py`

`AgentSpec` is a Pydantic `BaseModel` that captures every agent-constructor argument in a form that serialises to YAML or JSON. It exists so that non-Python stakeholders (prompt engineers, ops teams) can configure agents without touching code, and so agent definitions can be stored in version-controlled files. `Agent.from_file(path)` loads a spec and returns a ready-to-run agent; `Agent.from_spec(spec_dict)` does the same from a plain dict; `AgentSpec.to_file(path)` persists the spec and optionally writes a companion JSON Schema file for editor auto-completion.

Key fields: `model` (provider-prefixed model ID string), `instructions` (static string, list, or `TemplateStr` value), `capabilities` (list of capability dicts serialised by `get_serialization_name()`), `deps_schema` (JSON Schema describing the deps object — used for `TemplateStr` variable validation), and `output_schema` (JSON Schema for the expected output shape).

```python
# Example 1 — Minimal: define an agent entirely in YAML and load it
from pydantic_ai import Agent

# agent.yaml would contain:
# model: anthropic:claude-haiku-4-5
# instructions: You are a concise summarisation assistant.

# At runtime:
# agent = Agent.from_file('agent.yaml')
# result = agent.run_sync('Summarise this article in two sentences.')
# print(result.output)

# Python-side equivalent using AgentSpec directly:
from pydantic_ai import AgentSpec

spec = AgentSpec(
    model='anthropic:claude-haiku-4-5',
    instructions='You are a concise summarisation assistant.',
)
# Save spec + companion schema for editor validation
# spec.to_file('agent.yaml')

agent = Agent.from_spec(spec.model_dump(by_alias=True, exclude_none=True))
# result = agent.run_sync('Summarise this article in two sentences.')
```

```python
# Example 2 — Capabilities and deps_schema: declarative WebSearch + TemplateStr instructions
from pydantic_ai import AgentSpec, Agent

spec = AgentSpec(
    model='openai:gpt-5.6-luna',
    # deps_schema declares the shape so TemplateStr variables are validated at load time
    deps_schema={
        'type': 'object',
        'properties': {
            'user_name': {'type': 'string'},
            'language': {'type': 'string'},
        },
        'required': ['user_name', 'language'],
    },
    # The {{ }} syntax is compiled to a TemplateStr automatically when deps_schema is present
    instructions='You are a helpful assistant for {{user_name}}. Always reply in {{language}}.',
    capabilities=[
        # Capabilities are identified by their get_serialization_name()
        {'WebSearch': {'local': 'duckduckgo'}},
    ],
)

# spec.to_file('personalised_agent.yaml')
# agent = Agent.from_file('personalised_agent.yaml')

# Or directly:
agent = Agent.from_spec(spec.model_dump(by_alias=True, exclude_none=True))
# result = agent.run_sync('What is the weather today?', deps={'user_name': 'Alice', 'language': 'French'})
```

```python
# Example 3 — Round-trip: build from Python, persist as YAML, reload and verify
from pydantic_ai import AgentSpec, Agent

original_spec = AgentSpec(
    model='google:gemini-3-flash',
    name='research-agent',
    instructions=[
        'You are a research assistant.',
        'Always cite your sources.',
    ],
    output_schema={
        'type': 'object',
        'properties': {
            'summary': {'type': 'string'},
            'sources': {'type': 'array', 'items': {'type': 'string'}},
        },
        'required': ['summary', 'sources'],
    },
)

# Persist (writes research_agent.yaml + research_agent_schema.json)
# original_spec.to_file('research_agent.yaml')

# Reload — verifies round-trip fidelity
# reloaded = AgentSpec.from_file('research_agent.yaml')
# assert reloaded.model == original_spec.model

# Serialise to dict to inspect what gets written
serialised = original_spec.model_dump(by_alias=True, exclude_none=True)
assert serialised['model'] == 'google:gemini-3-flash'
assert 'summary' in serialised['output_schema']['properties']
print('Round-trip OK:', serialised['name'])
```

---

## 2. `TemplateStr`

**Source:** `pydantic_ai/template.py`

`TemplateStr[AgentDepsT]` is a Handlebars template string that renders against `RunContext.deps` at the moment the agent assembles its system prompt or instruction block. Any string value in a field typed `TemplateStr[Deps]` that contains `{{` is automatically compiled into a template during Pydantic validation, so YAML specs and Python `Agent(instructions=...)` calls both benefit without extra ceremony.

Template variables are drawn directly from the deps object: for a `@dataclass` deps, `{{field}}` expands to `str(deps.field)`; for a `dict` deps, `{{key}}` expands to `str(deps[key])`; for a Pydantic model, attribute access applies. When a `deps_schema` is attached (in an `AgentSpec`) or a `deps_type` is known (in a live `Agent`), variable names are validated at compilation time so mis-spelled variable names fail early rather than silently emitting empty strings at runtime.

```python
# Example 1 — Simple dataclass deps: inject user name and role into system prompt
from dataclasses import dataclass
from pydantic_ai import Agent, TemplateStr


@dataclass
class Deps:
    user_name: str
    role: str


agent = Agent(
    'openai:gpt-5.6-luna',
    deps_type=Deps,
    # Strings with {{ are auto-compiled to TemplateStr during Agent construction
    instructions='You are a helpful assistant for {{user_name}}, who is a {{role}}.',
)

# At run time the template is rendered before the first model request:
# result = agent.run_sync('What should I focus on this week?', deps=Deps('Alice', 'product manager'))
# The model sees: "You are a helpful assistant for Alice, who is a product manager."
```

```python
# Example 2 — Standalone rendering: compile and render outside an Agent
from dataclasses import dataclass
from pydantic_ai import TemplateStr


@dataclass
class ReportDeps:
    company: str
    quarter: str
    currency: str


# Construct a TemplateStr explicitly (deps_type needed outside Agent context)
template = TemplateStr[ReportDeps](
    'Analyse {{company}} Q{{quarter}} financial results. Report all figures in {{currency}}.',
    deps_type=ReportDeps,
)

rendered = template.render(ReportDeps(company='Acme Corp', quarter='3', currency='USD'))
print(rendered)
# Output: Analyse Acme Corp Q3 financial results. Report all figures in USD.
```

```python
# Example 3 — List of instructions mixing static strings and TemplateStr
from dataclasses import dataclass
from pydantic_ai import Agent, TemplateStr


@dataclass
class Ctx:
    language: str
    max_words: int


agent = Agent(
    'anthropic:claude-sonnet-4-6',
    deps_type=Ctx,
    instructions=[
        # Static instruction — always included verbatim
        'You are a translation assistant.',
        # Templated instruction — rendered against deps at run time
        'Always translate into {{language}} and keep your reply under {{max_words}} words.',
        # Another static guard
        'Never include the original text in your response.',
    ],
)

# result = agent.run_sync('Hello, world!', deps=Ctx(language='Spanish', max_words=50))
# The assembled system prompt concatenates all three instructions, with the second rendered.
```

---

## 3. `NativeOutput` + `PromptedOutput`

**Source:** `pydantic_ai/output.py`

`NativeOutput` and `PromptedOutput` are marker dataclasses that change how an agent elicits structured output from the model.

`NativeOutput(outputs, *, name=None, description=None)` instructs the agent to use the provider's first-party structured-output mechanism (OpenAI's JSON schema mode, Anthropic's tool-based extraction, etc.). The model itself serialises the response into the declared shape, so there is no post-processing prompt. This is faster and more reliable when the provider's native mode is stable.

`PromptedOutput(outputs, *, name=None, description=None, template=None)` tells the agent to append a textual prompt asking the model to produce JSON conforming to the schema, then validates the reply. Use it with models or providers that lack reliable native structured-output support, or when you want to customise the extraction phrasing via `template='{schema}'`-style format strings.

Both accept a single type, a list of types, or output functions, matching the same union semantics as `output_type=` on `Agent`.

```python
# Example 1 — NativeOutput with multiple output types (union discrimination)
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput


class WeatherReport(BaseModel):
    location: str
    temperature_c: float
    condition: str


class ErrorReport(BaseModel):
    reason: str


agent = Agent(
    'openai:gpt-5.6-sol',
    output_type=NativeOutput(
        [WeatherReport, ErrorReport],
        name='Weather or error',
        description='Return a weather report or an error if the location is unknown.',
    ),
)

# result = agent.run_sync('What is the weather in Tokyo?')
# isinstance(result.output, WeatherReport)  # True
```

```python
# Example 2 — PromptedOutput for a model without native structured-output support
from pydantic import BaseModel
from pydantic_ai import Agent, PromptedOutput


class RecipeSummary(BaseModel):
    title: str
    ingredients: list[str]
    steps: list[str]
    prep_time_minutes: int


# PromptedOutput appends a JSON-extraction prompt instead of using native mode
agent = Agent(
    'xai:grok-3',
    output_type=PromptedOutput(
        RecipeSummary,
        name='Recipe',
        description='Structured recipe with all fields filled in.',
    ),
)

# result = agent.run_sync('Give me a recipe for banana bread.')
# print(result.output.title)
```

```python
# Example 3 — PromptedOutput with a custom extraction template
from pydantic import BaseModel
from pydantic_ai import Agent, PromptedOutput


class SentimentResult(BaseModel):
    label: str          # 'positive', 'neutral', 'negative'
    confidence: float   # 0.0–1.0
    explanation: str


# The {schema} placeholder is replaced with the JSON schema of SentimentResult
agent = Agent(
    'anthropic:claude-haiku-4-5',
    output_type=PromptedOutput(
        SentimentResult,
        template=(
            'Analyse the sentiment of the text above. '
            'Respond ONLY with valid JSON matching this schema:\n{schema}'
        ),
    ),
)

# result = agent.run_sync('The product exceeded all my expectations!')
# print(result.output.label)   # 'positive'
# print(result.output.confidence)  # e.g. 0.95
```

---

## 4. `StructuredDict`

**Source:** `pydantic_ai/output.py`

`StructuredDict(json_schema, *, name=None, description=None)` is a factory function (not a true class) that returns a `dict[str, Any]` subclass with a JSON Schema attached. Use it when you want validated, schema-driven structured output but cannot or do not want to define a Pydantic `BaseModel`. The returned type can be passed directly to `output_type=` and works alongside Pydantic models in union output types.

The `json_schema` argument must be an object-type JSON Schema. `name` and `description` fall back to the schema's `title` and `description` fields respectively. Validation at run time coerces the model's reply to the declared schema the same way a `BaseModel` would, but the result is an ordinary Python `dict`.

```python
# Example 1 — Minimal: extract a structured dict without defining a Pydantic model
from pydantic_ai import Agent, StructuredDict

PersonSchema = StructuredDict(
    {
        'type': 'object',
        'title': 'Person',
        'properties': {
            'name': {'type': 'string'},
            'age': {'type': 'integer', 'minimum': 0},
            'email': {'type': 'string', 'format': 'email'},
        },
        'required': ['name', 'age'],
    }
)

agent = Agent('openai:gpt-5.6-luna', output_type=PersonSchema)

# result = agent.run_sync('Extract person details: Alice, 30, alice@example.com')
# print(result.output)          # {'name': 'Alice', 'age': 30, 'email': 'alice@example.com'}
# print(type(result.output))    # <class 'dict'>
```

```python
# Example 2 — Nested object schema with descriptions for richer model guidance
from pydantic_ai import Agent, StructuredDict

InvoiceSchema = StructuredDict(
    {
        'type': 'object',
        'title': 'Invoice',
        'description': 'A parsed invoice extracted from the supplied text.',
        'properties': {
            'invoice_number': {'type': 'string', 'description': 'The invoice ID, e.g. INV-0042'},
            'vendor': {'type': 'string'},
            'total_amount': {'type': 'number', 'description': 'Total amount in the invoice currency'},
            'currency': {'type': 'string', 'enum': ['USD', 'EUR', 'GBP']},
            'line_items': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'description': {'type': 'string'},
                        'quantity': {'type': 'integer'},
                        'unit_price': {'type': 'number'},
                    },
                    'required': ['description', 'quantity', 'unit_price'],
                },
            },
        },
        'required': ['invoice_number', 'vendor', 'total_amount', 'currency'],
    },
    name='ParsedInvoice',
)

agent = Agent('anthropic:claude-sonnet-4-6', output_type=InvoiceSchema)
# result = agent.run_sync('Invoice INV-0099 from Acme Corp, total $450 USD. One widget @ $450.')
# print(result.output['vendor'])          # 'Acme Corp'
# print(result.output['total_amount'])    # 450.0
```

```python
# Example 3 — Union of StructuredDict + Pydantic model for mixed output routing
from pydantic import BaseModel
from pydantic_ai import Agent, StructuredDict


class ErrorResult(BaseModel):
    error: str
    retry_suggested: bool


# StructuredDict for the happy-path shape, BaseModel for the error shape
DataResult = StructuredDict(
    {
        'type': 'object',
        'title': 'DataResult',
        'properties': {
            'rows': {'type': 'integer'},
            'columns': {'type': 'array', 'items': {'type': 'string'}},
            'preview': {'type': 'array', 'items': {'type': 'object'}},
        },
        'required': ['rows', 'columns'],
    }
)

agent = Agent('openai:gpt-5.6-sol', output_type=[DataResult, ErrorResult])

# result = agent.run_sync('Parse the CSV header: id,name,score')
# if isinstance(result.output, dict):
#     print('Rows parsed:', result.output['rows'])
# else:
#     print('Error:', result.output.error)
```

---

## 5. `SkipModelRequest` + `SkipToolExecution` + `SkipToolValidation`

**Source:** `pydantic_ai/exceptions.py`

Three exceptions provide fine-grained control-flow in hook callbacks:

- **`SkipModelRequest(response)`** — raise inside a `before_model_request` or `wrap_model_request` hook to abort the model call and substitute a synthetic `ModelResponse`. Useful for short-circuit caching, test injection, or conditional early termination.
- **`SkipToolExecution(result)`** — raise inside a `before_tool` or `wrap_tool` hook to skip the tool body entirely; the provided `result` is returned to the model as the tool's output. Useful for caching, sandboxing, or permission checks.
- **`SkipToolValidation(validated_args)`** — raise inside a `before_tool_validation` hook to bypass Pydantic validation and inject pre-validated args. Useful when args have already been validated upstream or when you need to coerce types the schema cannot express.

All three are plain `Exception` subclasses with a single constructor argument.

```python
# Example 1 — SkipModelRequest: inject a cached response to avoid a live API call
from pydantic_ai import Agent
from pydantic_ai.exceptions import SkipModelRequest
from pydantic_ai.messages import ModelResponse, TextPart

RESPONSE_CACHE: dict[str, str] = {}


async def cache_interceptor(ctx, request_context):
    # Derive a cache key from the last user message
    messages = request_context.messages
    last_user = next(
        (p.content for m in reversed(messages) for p in m.parts if hasattr(p, 'content') and isinstance(p.content, str)),
        None,
    )
    if last_user and last_user in RESPONSE_CACHE:
        cached_text = RESPONSE_CACHE[last_user]
        raise SkipModelRequest(
            ModelResponse(parts=[TextPart(content=cached_text)])
        )
    return request_context  # cache miss — return context so the real model call proceeds


# Register via Hooks capability so the interceptor is actually invoked
from pydantic_ai.capabilities import Hooks

agent = Agent(
    'openai:gpt-5.6-luna',
    capabilities=[Hooks(before_model_request=cache_interceptor)],
)
# Populate the cache:
RESPONSE_CACHE['What is 2+2?'] = 'The answer is 4.'
# result = agent.run_sync('What is 2+2?')  # returns from cache, no API call
```

```python
# Example 2 — SkipToolExecution: sandbox tool calls in test environments
import os
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import SkipToolExecution


async def email_interceptor(ctx, *, call, tool_def, args):
    """Skip send_email and return a stub result when TESTING=1."""
    if os.getenv('TESTING') == '1' and tool_def.name == 'send_email':
        raise SkipToolExecution({'status': 'stubbed', 'to': args['to']})
    return args  # pass through unchanged on all other calls


agent = Agent(
    'openai:gpt-5.6-luna',
    capabilities=[Hooks(before_tool_execute=email_interceptor)],
)


@agent.tool_plain
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the given recipient."""
    return f'Email sent to {to}'


# With TESTING=1 set, the interceptor fires and send_email body never runs:
# os.environ['TESTING'] = '1'
# result = agent.run_sync('Send a welcome email to alice@example.com')
```

```python
# Example 3 — SkipToolValidation: inject pre-coerced args bypassing schema validation
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import SkipToolValidation


async def coerce_ids(ctx, *, call, tool_def, args):
    """Coerce string record_id to int before Pydantic validation runs."""
    if tool_def.name == 'lookup_record' and 'record_id' in args:
        # raw_args may arrive as strings from certain providers
        coerced = {**args, 'record_id': int(args['record_id'])}
        raise SkipToolValidation(coerced)
    return args  # pass through for all other tools


agent = Agent(
    'anthropic:claude-haiku-4-5',
    capabilities=[Hooks(before_tool_validate=coerce_ids)],
)


@agent.tool_plain
def lookup_record(record_id: int) -> dict:
    """Look up a record by its integer ID."""
    return {'id': record_id, 'name': f'Record {record_id}'}


# result = agent.run_sync('Look up record 42')
# coerce_ids fires before validation, converting '42' → 42 so the int check passes.
```

---

## 6. `SelectModel` + `ModelSelectionContext`

**Source:** `pydantic_ai/capabilities/__init__.py` and `pydantic_ai/models/__init__.py`

`SelectModel(selector)` is a capability that invokes `selector` before every logical model-request step to choose which model to use for that step. The selector can be a synchronous or asynchronous callable; it receives a `ModelSelectionContext` and returns either a model instance or a provider-prefixed model-name string.

`ModelSelectionContext[DepsT]` is a frozen dataclass exposing:
- `deps` — the run's dependency object (type `DepsT`)
- `model` — the lower-precedence model on step 1, or the model used in the previous step
- `run_step` — the 1-based request step counter (1 = first model call, 2 = after first tool-call round, etc.)
- `messages` — the full message history before this step
- `usage` — accumulated `RunUsage` before this step

Because the selector fires before step 1, the agent does not need a constructor-level model; `SelectModel` alone is sufficient.

```python
# Example 1 — Dependency-driven routing: pick model from a user-supplied tier
from dataclasses import dataclass
from typing import Literal

from pydantic_ai import Agent, ModelSelectionContext
from pydantic_ai.capabilities import SelectModel


@dataclass
class UserDeps:
    tier: Literal['free', 'pro', 'enterprise']


def pick_model(ctx: ModelSelectionContext[UserDeps]) -> str:
    match ctx.deps.tier:
        case 'enterprise':
            return 'openai:gpt-5.6-sol'
        case 'pro':
            return 'openai:gpt-5.6-luna'
        case _:
            return 'openai:gpt-5.6-mini'


agent = Agent(deps_type=UserDeps, capabilities=[SelectModel(pick_model)])

# result = agent.run_sync('Explain quantum entanglement.', deps=UserDeps(tier='pro'))
# The model used is gpt-5.6-luna for this run.
```

```python
# Example 2 — Step-escalation: start with a fast model, escalate to stronger on retry
from pydantic_ai import Agent, ModelSelectionContext
from pydantic_ai.capabilities import SelectModel


def escalating_selector(ctx: ModelSelectionContext[None]) -> str:
    # step 1 — fast model; step 2+ (tool-call round or retry) — stronger model
    if ctx.run_step == 1:
        return 'anthropic:claude-haiku-4-5'
    return 'anthropic:claude-opus-4-6'


agent = Agent(capabilities=[SelectModel(escalating_selector)])

# On a multi-step run:
# Step 1: haiku handles the initial response
# Step 2+: opus handles tool-call resolution or retries
# result = agent.run_sync('Research and summarise the latest ML papers.')
```

```python
# Example 3 — Usage-budget fallback: switch to a cheap model once token spend exceeds a threshold
from pydantic_ai import Agent, ModelSelectionContext
from pydantic_ai.capabilities import SelectModel

TOKEN_BUDGET = 50_000


def budget_aware_selector(ctx: ModelSelectionContext[None]) -> str:
    tokens_so_far = (ctx.usage.input_tokens or 0) + (ctx.usage.output_tokens or 0)
    if tokens_so_far > TOKEN_BUDGET:
        # Downgrade to reduce cost for the remaining steps
        return 'openai:gpt-5.6-mini'
    return 'openai:gpt-5.6-sol'


agent = Agent(capabilities=[SelectModel(budget_aware_selector)])

# Long research tasks automatically downgrade once the token budget is exhausted.
# result = agent.run_sync('Write a comprehensive 10,000-word report on renewable energy.')
```

---

## 7. `DeferredLoadingToolset` + `IncludeReturnSchemasToolset`

**Source:** `pydantic_ai/toolsets/deferred_loading.py` and `pydantic_ai/toolsets/include_return_schemas.py`

Both are `PreparedToolset` subclasses — wrappers that modify an inner toolset's `ToolDefinition` list before it reaches the model.

**`DeferredLoadingToolset(wrapped, *, tool_names=None)`** marks tools for deferred loading by setting `defer_loading=True` on their `ToolDefinition`s. A deferred tool is hidden from the model's initial context; it only becomes visible once the model calls a discovery/search mechanism (such as a tool-search native tool). This keeps the model's context lean when an MCP server or function toolset has dozens of endpoints. `tool_names=None` defers all tools; pass a `frozenset` of names to defer only specific ones.

**`IncludeReturnSchemasToolset(wrapped)`** sets `include_return_schema=True` on every `ToolDefinition` it wraps. Some providers use the return schema to choose the correct output format for a tool call; enabling it explicitly is useful when working with models that support schema-aware tool responses.

```python
# Example 1 — DeferredLoadingToolset: hide all MCP tools until discovered
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets._tool_search import ToolSearchToolset  # exposes discovery to model

# An MCP server exposing 50+ endpoints would bloat the context if listed upfront.
mcp = MCPToolset('http://localhost:8000/mcp')

# ToolSearchToolset wraps the deferred toolset and injects a search_tools function so
# the model can discover and enable individual MCP tools on demand via keyword queries.
agent = Agent(
    'openai:gpt-5.6-sol',
    toolsets=[ToolSearchToolset(mcp.defer_loading())],
)

# The model's first request sees only the search_tools function, not all MCP endpoints.
# After calling search_tools the matched tools become available for subsequent steps.
# result = await agent.run('Find and use the inventory lookup tool.')
```

```python
# Example 2 — DeferredLoadingToolset with tool_names: defer only large/expensive tools
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.toolsets import DeferredLoadingToolset

toolset = FunctionToolset()


@toolset.tool_plain
def quick_lookup(key: str) -> str:
    """Fast key-value lookup."""
    return f'value_for_{key}'


@toolset.tool_plain
def expensive_analysis(dataset_id: str, depth: int) -> dict:
    """Run a deep analysis pipeline — slow and expensive."""
    return {'result': 'analysis', 'dataset': dataset_id}


# Only expensive_analysis is deferred; quick_lookup is visible from the first request.
agent = Agent(
    'anthropic:claude-sonnet-4-6',
    toolsets=[
        DeferredLoadingToolset(
            toolset,
            tool_names=frozenset({'expensive_analysis'}),
        )
    ],
)
# result = agent.run_sync('Do a quick lookup for key "config".')
```

```python
# Example 3 — IncludeReturnSchemasToolset: attach return schemas for schema-aware providers
from pydantic import BaseModel
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.toolsets import IncludeReturnSchemasToolset


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


toolset = FunctionToolset()


@toolset.tool_plain
def web_search(query: str) -> list[SearchResult]:
    """Search the web and return structured results."""
    # Stub — real implementation calls a search API
    return [SearchResult(title='Example', url='https://example.com', snippet='An example page')]


# Wrap with IncludeReturnSchemasToolset so the model knows SearchResult's shape
agent = Agent(
    'openai:gpt-5.6-sol',
    toolsets=[IncludeReturnSchemasToolset(toolset)],
)
# result = agent.run_sync('Search for "pydantic ai" and summarise the top result.')
```

---

## 8. `AdvisorTool`

**Source:** `pydantic_ai/native_tools/__init__.py`

`AdvisorTool` is an `AbstractNativeTool` that exposes the provider's built-in *executor/advisor* mechanism. A fast *executor* model (the one running the agent) can pause mid-generation and ask a stronger *advisor* model for guidance on a specific sub-problem, then incorporate the advisor's reply before continuing. This is more efficient than spawning a full sub-agent because the advisor call stays within the same generation context.

Provider support: **Anthropic** natively; **OpenRouter** via gateway (honours `model` and `max_tokens`, ignores provider-specific fields).

Key fields:
- `model: AdvisorModelName` — the advisor model ID (Anthropic namespace on Anthropic, OpenRouter catalog slug on OpenRouter)
- `max_uses: int | None` — per-request cap on advisor consultations
- `max_tokens: int | None` — token budget for each advisor response
- `system_prompt: str | None` — a system prompt for the advisor (Anthropic only)

```python
# Example 1 — Minimal: haiku executor consults opus advisor for complex reasoning
from pydantic_ai import Agent
from pydantic_ai.native_tools import AdvisorTool

agent = Agent(
    'anthropic:claude-haiku-4-5',   # fast, cheap executor
    capabilities=[
        # Advisor tool registered as a native capability
    ],
    # In 2.31.0 native tools are attached via capabilities=[NativeTool(AdvisorTool(...))]
)

# Direct approach: provide AdvisorTool to NativeTool capability
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'anthropic:claude-haiku-4-5',
    capabilities=[
        NativeTool(
            AdvisorTool(
                model='claude-opus-4-8',  # strong advisor
                max_tokens=1024,
            )
        )
    ],
)

# result = agent.run_sync(
#     'Design a distributed consensus algorithm that handles Byzantine faults '
#     'with fewer than n/3 faulty nodes. Prove the correctness bound.'
# )
# haiku handles fast text; when it encounters the proof step it consults opus.
```

```python
# Example 2 — Capped advisor with a custom system prompt
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import AdvisorTool

agent = Agent(
    'anthropic:claude-haiku-4-5',
    system_prompt='You are a code-review assistant.',
    capabilities=[
        NativeTool(
            AdvisorTool(
                model='claude-opus-4-8',
                max_uses=2,           # consult the advisor at most twice per request
                max_tokens=512,       # keep advisor responses concise
                system_prompt=(       # Anthropic-specific advisor system prompt
                    'You are a senior security engineer. '
                    'Identify only critical vulnerabilities.'
                ),
            )
        )
    ],
)

# result = agent.run_sync('Review this authentication middleware: ...')
# haiku drafts a review; when it reaches security implications it consults opus (up to 2×).
```

```python
# Example 3 — OpenRouter gateway: advisor via OpenRouter model catalog
# Set OPENROUTER_API_KEY in the environment; the 'openrouter:' prefix resolves it automatically.
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import AdvisorTool
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

# OpenRouter exposes advisor as a gateway server tool.
# It honours `model` and `max_tokens`; other fields are silently ignored.
# Pass the provider explicitly so a runtime-supplied key is actually used.
provider = OpenRouterProvider(api_key='or-...')
model = OpenAIChatModel('meta-llama/llama-3.3-70b-instruct', provider=provider)

agent = Agent(
    model,
    capabilities=[
        NativeTool(
            AdvisorTool(
                model='anthropic/claude-opus-4.8',  # OpenRouter catalog slug
                max_tokens=2048,
            )
        )
    ],
)

# result = agent.run_sync('Explain the trade-offs between CAP and PACELC theorems.')
```

---

## 9. `CachePoint` + `ToolAvailabilityDeltaPart`

**Source:** `pydantic_ai/messages.py`

**`CachePoint`** is a `@dataclass` that can be inserted into a `UserPromptPart.content` list to mark prompt-caching boundaries for providers that support it (Anthropic, Bedrock Converse, OpenAI GPT-5.6 models, OpenRouter with compatible backends). Fields:
- `kind = 'cache-point'` — discriminator
- `ttl: Literal['5m', '1h']` — cache time-to-live; Anthropic and Bedrock support `'1h'`; OpenAI always uses `'5m'`

Providers that do not support caching silently filter out `CachePoint` markers, so the same message-construction code works cross-provider.

**`ToolAvailabilityDeltaPart`** is a streaming message part that records that new tool definitions became available to the model mid-stream (used internally by `DeferredLoadingToolset` when deferred tools are discovered and injected into a live request). It carries a `tools` list of `ToolDefinition` objects representing the tools that were just made available. Consuming code can inspect this part when parsing `ModelResponse` streams to understand when the tool surface changed.

```python
# Example 1 — CachePoint: cache a large system document with a 1-hour TTL
from pydantic_ai.messages import UserPromptPart, TextContent, CachePoint

large_policy_document = 'POLICY: ' + ('word ' * 5000)  # ~5000-token document

# Insert a cache point after the expensive content to mark the cache boundary.
# Content before the last CachePoint is eligible for caching.
prompt_part = UserPromptPart(
    content=[
        TextContent(content=large_policy_document),
        CachePoint(ttl='1h'),   # cache for 1 hour (Anthropic / Bedrock)
        TextContent(content='Given the policy above, answer the following question: '),
    ]
)

# On the first request the provider caches everything up to the CachePoint.
# Subsequent requests with the same prefix are served from cache, saving tokens.
```

```python
# Example 2 — CachePoint: cross-provider-safe construction (cache silently dropped for OpenAI non-5.6)
from pydantic_ai import Agent
from pydantic_ai.messages import UserPromptPart, TextContent, CachePoint

# Build a message with a CachePoint embedded — safe to send to any provider.
# Non-supporting providers filter the marker out transparently.
shared_context = 'CONTEXT: ' + ('data ' * 2000)

def make_user_part(question: str) -> UserPromptPart:
    return UserPromptPart(
        content=[
            TextContent(content=shared_context),
            CachePoint(ttl='5m'),
            TextContent(content=question),
        ]
    )

# The same factory function works with any agent, regardless of provider.
# agent_a = Agent('anthropic:claude-haiku-4-5')
# agent_b = Agent('openai:gpt-5.6-luna')   # CachePoint supported (GPT-5.6)
# agent_c = Agent('google:gemini-3-flash')  # CachePoint filtered out
```

```python
# Example 3 — ToolAvailabilityDeltaPart: inspect deferred-tool discovery events in a stream
from pydantic_ai import Agent
from pydantic_ai.messages import ToolAvailabilityDeltaPart
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets._tool_search import ToolSearchToolset  # required for discovery

mcp = MCPToolset('http://localhost:8000/mcp')

# ToolSearchToolset provides the search_tools function the model calls to discover
# deferred tools; without it the model sees no MCP endpoints and no delta events fire.
agent = Agent(
    'openai:gpt-5.6-sol',
    toolsets=[ToolSearchToolset(mcp.defer_loading())],
)

# run_stream_events() is the async context manager that yields raw AgentStreamEvents,
# which include PartStartEvent / PartEndEvent wrappers carrying model-response parts.
# async with agent.run_stream_events('Find and use the best tool for inventory lookup.') as events:
#     async for event in events:
#         # ToolAvailabilityDeltaPart fires when deferred tools are injected mid-stream
#         part = getattr(event, 'part', None)
#         if isinstance(part, ToolAvailabilityDeltaPart):
#             newly_available = [t.name for t in part.tools]
#             print(f'Tools now available: {newly_available}')

# This lets you log or display tool-discovery events to the user in real time.
```

---

## 10. Native Tool Suite: `XSearchTool`, `ImageGenerationTool`, `WebFetchTool`, `MemoryTool`, `CodeExecutionTool`, `FileSearchTool`

**Source:** `pydantic_ai/native_tools/__init__.py`

Pydantic AI 2.31.0 bundles six provider-specific native tools that map 1:1 to the provider's own server-side tool definitions. All extend `AbstractNativeTool` and are passed to the agent via `capabilities=[NativeTool(...)]`.

| Tool | Providers | Purpose |
|---|---|---|
| `XSearchTool` | xAI | Search X/Twitter posts and content |
| `ImageGenerationTool` | OpenAI Responses, Google | Generate or edit images from prompts |
| `WebFetchTool` | Anthropic, Google | Fetch and read web page content |
| `MemoryTool` | Anthropic | Provider-managed persistent memory |
| `CodeExecutionTool` | Anthropic, OpenAI Responses, Google, Bedrock (Nova 2.0), xAI | Execute code in a sandboxed environment |
| `FileSearchTool` | OpenAI Responses, Google, xAI | RAG over uploaded files via vector search |

```python
# Example 1 — XSearchTool + WebFetchTool: research agent with web access
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import XSearchTool, WebFetchTool
from datetime import datetime, timezone

# XSearchTool: restricted to a date range and specific handles
x_search = XSearchTool(
    allowed_x_handles=['OpenAI', 'AnthropicAI', 'GoogleDeepMind'],
    from_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
)

# WebFetchTool: only allowed domains to prevent misuse
web_fetch = WebFetchTool(
    allowed_domains=['arxiv.org', 'openai.com', 'anthropic.com'],
    max_uses=5,
)

# XSearchTool needs an xAI model; WebFetchTool needs Anthropic or Google.
# Use two agents in a pipeline, or pick one that covers both tools.
xai_agent = Agent(
    'xai:grok-3',
    capabilities=[NativeTool(x_search)],
)

anthropic_agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[NativeTool(web_fetch)],
)

# xai_result = xai_agent.run_sync('What are the top AI announcements from the last month?')
# fetch_result = anthropic_agent.run_sync('Summarise the abstract of arxiv.org/abs/2406.00001')
```

```python
# Example 2 — CodeExecutionTool + ImageGenerationTool: data analysis with visual output
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import CodeExecutionTool, ImageGenerationTool

# CodeExecutionTool: sandboxed Python (Anthropic, OpenAI Responses, Google, Bedrock, xAI)
code_agent = Agent(
    'anthropic:claude-sonnet-4-6',
    system_prompt='You are a data scientist. Always execute code to verify your results.',
    capabilities=[NativeTool(CodeExecutionTool())],
)

# result = code_agent.run_sync(
#     'Calculate the mean and standard deviation of [2, 4, 6, 8, 10, 12] and show the working.'
# )

# ImageGenerationTool: generate images (OpenAI Responses + Google)
image_agent = Agent(
    'openai-responses:gpt-5.6-sol',
    capabilities=[
        NativeTool(
            ImageGenerationTool(action='generate')  # 'generate', 'edit', or 'auto'
        )
    ],
)

# img_result = image_agent.run_sync('Generate a diagram of a microservices architecture.')
```

```python
# Example 3 — MemoryTool + FileSearchTool: personalised assistant with persistent memory and RAG
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import MemoryTool, FileSearchTool

# MemoryTool: Anthropic's provider-managed persistent memory
memory_agent = Agent(
    'anthropic:claude-sonnet-4-6',
    system_prompt='You are a personal assistant. Remember user preferences across sessions.',
    capabilities=[NativeTool(MemoryTool())],
)

# The model can read and write to its memory store between turns.
# result = memory_agent.run_sync('Remember that I prefer metric units and a vegetarian diet.')
# Later run:
# result = memory_agent.run_sync('Suggest a recipe for dinner.')
# The model recalls the vegetarian preference from memory.

# FileSearchTool: managed RAG over a vector store (OpenAI Responses / Google / xAI)
rag_agent = Agent(
    'openai-responses:gpt-5.6-sol',
    system_prompt='You are a document assistant. Search the knowledge base to answer questions.',
    capabilities=[
        NativeTool(
            FileSearchTool(
                file_store_ids=['vs_abc123'],  # OpenAI vector store ID
                max_num_results=5,
            )
        )
    ],
)

# result = rag_agent.run_sync('What does the Q3 report say about gross margin?')
# The model performs semantic search over the vector store and synthesises an answer.
```

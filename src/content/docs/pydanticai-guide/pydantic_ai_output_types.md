---
title: "PydanticAI: Output Types & Validators"
description: "Typed outputs with ToolOutput, NativeOutput, PromptedOutput, TextOutput, StructuredDict, multi-type unions, and agent.output_validator."
framework: pydanticai
language: python
---

# Output Types & Validators

Verified against **pydantic-ai==2.8.0** — source modules: `pydantic_ai.output`, `pydantic_ai.agent`.

The `output_type` argument on `Agent` (or on a `run*` call) drives how the model returns structured data. PydanticAI ships five "marker" wrappers — `ToolOutput`, `NativeOutput`, `PromptedOutput`, `TextOutput`, `StructuredDict` — plus a plain type / union shortcut. The right one depends on what the model natively supports.

## Minimal runnable example

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class Answer(BaseModel):
    value: int
    reasoning: str

agent = Agent('openai:gpt-5.2', output_type=Answer)
result = agent.run_sync('What is 15 + 27?')
print(result.output)
#> value=42 reasoning='...'
print(type(result.output))
#> <class '__main__.Answer'>
```

Passing a bare type is a shortcut. PydanticAI will pick the right `OutputMode` based on the model's profile (`ModelProfile.default_structured_output_mode`). To override, wrap the type in one of the marker classes below.

## Output mode comparison

| Marker class   | How it works                                          | When to use                                                     |
| -------------- | ----------------------------------------------------- | --------------------------------------------------------------- |
| bare type      | Auto-picks `tool` / `native` / `prompted` from the model profile. | Default — you don't care which mechanism is used.             |
| `ToolOutput`   | Model emits a structured "output tool" call that PydanticAI validates. | You want to name the tool, set `strict`, or use multi-type unions on models that lack native JSON schema. |
| `NativeOutput` | Uses the provider's native structured-outputs API (e.g. OpenAI `response_format=json_schema`). | Model supports native JSON schema and you want maximum fidelity. |
| `PromptedOutput` | Injects a JSON schema into the system prompt and parses text. | Provider has no native structured outputs (local / older models). |
| `TextOutput`   | Passes the model's plain text through a function.     | You want a custom parser (splitter, regex, domain extractor).   |
| `StructuredDict` | Returns `dict[str, Any]` with a runtime-attached JSON schema. | Schema is built at runtime (user-defined form, DB-driven). |

`OutputMode` literals (`output.py`): `'text' | 'tool' | 'native' | 'prompted' | 'image' | 'auto'` plus a deprecated `'tool_or_text'`. You rarely set the mode directly; picking a marker class above is the supported path.

## `ToolOutput` — multi-type unions

```python
from pydantic import BaseModel
from pydantic_ai import Agent, ToolOutput

class Fruit(BaseModel):
    name: str
    color: str

class Vehicle(BaseModel):
    name: str
    wheels: int

agent = Agent(
    'openai:gpt-5.2',
    output_type=[
        ToolOutput(Fruit, name='return_fruit'),
        ToolOutput(Vehicle, name='return_vehicle'),
    ],
)
result = agent.run_sync('What is a banana?')
print(repr(result.output))
#> Fruit(name='banana', color='yellow')
```

Arguments (`output.py` — verified against pydantic-ai 2.8.0):

| Arg | Type | Default | Notes |
|-----|------|---------|-------|
| `output` | `type \| callable` | required | Pydantic model, dataclass, or async callable |
| `name` | `str \| None` | `None` | Tool name sent to the model; auto-derived if unset |
| `description` | `str \| None` | `None` | Overrides the type's docstring as the tool description |
| `max_retries` | `int \| None` | `None` | Output-tool-specific retry budget; overrides the agent-level `retries` / `output_retries` |
| `strict` | `bool \| None` | `None` | Forwarded to providers that support strict JSON schema (OpenAI) |

### `ToolOutput.max_retries` — per-output retry budgets

`max_retries` lets you set a _different_ retry budget for each output branch. Expensive routes get fewer retries; cheap ones can afford more:

```python
from pydantic import BaseModel, field_validator
from pydantic_ai import Agent, ToolOutput, ModelRetry, RunContext

class QuickAnswer(BaseModel):
    """A short factual answer."""
    text: str

    @field_validator('text')
    @classmethod
    def non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('text must not be empty')
        return v

class DetailedReport(BaseModel):
    """A structured report with sections and citations."""
    title: str
    sections: list[str]
    citations: list[str]

    @field_validator('citations')
    @classmethod
    def requires_citations(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError('At least one citation is required')
        return v

agent = Agent(
    'openai:gpt-4o',
    output_type=[
        ToolOutput(QuickAnswer,     name='quick',  max_retries=3),   # more forgiving
        ToolOutput(DetailedReport,  name='report', max_retries=1),   # expensive, fail fast
    ],
)

# The agent chooses which output tool to call based on the question
result = agent.run_sync('What year was Python created?')
print(repr(result.output))  # QuickAnswer(text='1991')

result2 = agent.run_sync('Give me a detailed report on Python history with sources.')
print(repr(result2.output))  # DetailedReport(title=..., sections=[...], citations=[...])
```

## `NativeOutput` — provider-native JSON schema

Uses the provider's own structured-output mechanism (e.g. OpenAI's `response_format=json_schema`). This guarantees the model cannot deviate from the schema and avoids the extra tool-call round-trip of `ToolOutput`.

```python
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput

class Fruit(BaseModel):
    name: str
    color: str

class Vehicle(BaseModel):
    name: str
    wheels: int

# Single type — returns Fruit directly
agent_single = Agent('openai:gpt-4o', output_type=NativeOutput(Fruit))
result = agent_single.run_sync('What is a banana?')
print(repr(result.output))
# Fruit(name='banana', color='yellow')

# Multiple types — provider builds a tagged union
agent_union = Agent(
    'openai:gpt-4o',
    output_type=NativeOutput(
        [Fruit, Vehicle],
        name='fruit_or_vehicle',
        description='Return a fruit or a vehicle.',
    ),
)
result = agent_union.run_sync('What is a Ford Explorer?')
print(repr(result.output))
# Vehicle(name='Ford Explorer', wheels=4)
```

### `NativeOutput` with `strict=True`

OpenAI's strict mode rejects schemas that include `anyOf` without a discriminator. Use a `Literal` discriminator field when strict is required:

```python
from typing import Literal
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput

class FruitResult(BaseModel):
    kind: Literal['fruit'] = 'fruit'
    name: str
    color: str

class VehicleResult(BaseModel):
    kind: Literal['vehicle'] = 'vehicle'
    name: str
    wheels: int

agent = Agent(
    'openai:gpt-4o',
    output_type=NativeOutput([FruitResult, VehicleResult], strict=True),
)
```

### `NativeOutput` without schema injection (`template=False`)

When your model profile already injects a JSON schema into the system prompt, pass `template=False` to avoid duplication:

```python
from pydantic_ai import Agent, NativeOutput

agent = Agent(
    'ollama:llama3.2',
    output_type=NativeOutput(Vehicle, template=False),
)
```

Arguments (`output.py:141`):

- `outputs` — a type or sequence of types. A list produces a tagged union.
- `name`, `description` — override the schema name and description sent to the model.
- `strict` — forwarded to providers supporting strict JSON schema (OpenAI).
- `template` — overrides the schema-injection template; pass `False` to skip.

Availability by provider (verified in `models/<provider>.py`): OpenAI, Google, Anthropic (via tool adapter), Mistral, Groq. Older/local providers usually fall through to `PromptedOutput`.

## `PromptedOutput` — schema-in-prompt fallback

Works with any model that produces text. The schema is embedded in the prompt and the returned text is parsed.

```python
from pydantic_ai import Agent, PromptedOutput

agent = Agent(
    'ollama:llama3.1',
    output_type=PromptedOutput(
        [Vehicle, Fruit],
        template='Respond with JSON matching: {schema}',
    ),
)
```

Set `template=False` if your model profile already injects a schema.

## `TextOutput` — post-process raw text

```python
from pydantic_ai import Agent, TextOutput

def split_words(text: str) -> list[str]:
    return text.split()

agent = Agent('openai:gpt-5.2', output_type=TextOutput(split_words))
result = agent.run_sync('Who was Albert Einstein?')
print(result.output)
#> ['Albert', 'Einstein', 'was', 'a', 'German-born', 'theoretical', 'physicist.']
```

The function can optionally take `RunContext[Deps]` as its first argument.

## `StructuredDict` — runtime JSON schemas

```python
from pydantic_ai import Agent, StructuredDict

schema = {
    'type': 'object',
    'properties': {
        'title': {'type': 'string'},
        'tags': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['title'],
}

DynamicForm = StructuredDict(schema, name='form_response', description='Fill this form.')

agent = Agent('openai:gpt-5.2', output_type=DynamicForm)
result = agent.run_sync('Make up a blog post.')
print(result.output)
#> {'title': '...', 'tags': [...]}
```

`StructuredDict` returns a `dict[str, Any]` subclass with the schema baked in — use it when the schema is data, not a declared Python type.

## Output validators — `@agent.output_validator`

Run arbitrary code after the model produces an output, potentially asking the model to retry.

```python
from pydantic_ai import Agent, ModelRetry, RunContext

agent = Agent('openai:gpt-5.2', output_type=str)

@agent.output_validator
async def no_profanity(ctx: RunContext[None], output: str) -> str:
    if 'damn' in output.lower():
        raise ModelRetry('Please respond without profanity.')
    return output
```

The validator may:

- Return the same value (possibly transformed / sanitised).
- Raise `ModelRetry(msg)` to feed a retry prompt back to the model.
- Raise `UnexpectedModelBehavior` to terminate the run.

Up to `output_retries` (defaults to the agent `retries`) validator-triggered retries are allowed before the run fails.

## Output streaming

See the [streaming guide](./pydantic_ai_streaming/). `StreamedRunResult.get_output()` validates the final assembled output using the same `output_type` pipeline.

## Gotchas

- **Union of bare types**: `output_type=[Fruit, Vehicle]` works but you lose per-type tool naming. Use `ToolOutput(...)` per branch when the model gets confused about which to emit.
- **`strict=True`** (OpenAI): the model rejects any schema with `anyOf`/`oneOf` at the root without a discriminator. If you see "strict mode schema rejected" errors, drop `strict` or use `PromptedOutput`.
- **`NativeOutput` on older OpenAI models** (pre-`gpt-4o-2024-08-06`): silently falls back to prompted mode; check `result.response.model_name` and the raw messages if you need to confirm.
- **`TextOutput` and tools**: when you combine `TextOutput` with function tools, set `end_strategy='graceful'` on the agent so tool calls still run before the text is finalised.
- **`output_type` on `run()`**: the per-run override is only allowed when the agent has no `output_validator` — the validator's type wouldn't match.

## Patterns

### 1. Discriminated routing between shape types

```python
from typing import Literal
from pydantic import BaseModel

class Search(BaseModel):
    kind: Literal['search']
    query: str

class Action(BaseModel):
    kind: Literal['action']
    name: str

agent = Agent('openai:gpt-5.2', output_type=Search | Action)
```

PydanticAI generates a tagged union schema; the `kind` literal gives the model an unambiguous label to produce.

### 2. Retry with a more specific constraint

```python
@agent.output_validator
async def must_contain_sources(ctx: RunContext[None], out: Answer) -> Answer:
    if not out.reasoning:
        raise ModelRetry('Include a `reasoning` field citing at least one source.')
    return out
```

### 3. Convert model output into a domain type via `TextOutput`

```python
from datetime import date

def parse_iso_date(text: str) -> date:
    return date.fromisoformat(text.strip())

agent = Agent('openai:gpt-5.2', output_type=TextOutput(parse_iso_date))
```

### 4. Hybrid: structured output _plus_ a free-text summary

Use a Pydantic model whose schema contains both the structured and the prose fields; don't try to mix `TextOutput` and `ToolOutput` on the same run.

```python
class Report(BaseModel):
    summary: str
    findings: list[str]
    confidence: float
```

### 5. Runtime-built form with `StructuredDict`

```python
def build_form_schema(fields: list[dict]) -> type:
    schema = {'type': 'object', 'properties': {f['name']: f['schema'] for f in fields}}
    return StructuredDict(schema, name='form_response')

agent = Agent('openai:gpt-5.2', output_type=build_form_schema(user_fields))
```

## Advanced `NativeOutput` patterns

### Custom schema-injection template

Override the `template` to control exactly how the schema is presented to the model. `{schema}` is the only required placeholder:

```python
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput

class Analysis(BaseModel):
    topic: str
    sentiment: str
    key_points: list[str]

TEMPLATE = (
    'Analyse the input and return a JSON object that EXACTLY matches:\n'
    '```json\n{schema}\n```\n'
    'Nothing outside the JSON — no prose, no markdown, just the object.'
)

agent = Agent(
    'ollama:llama3.2',
    output_type=NativeOutput(Analysis, template=TEMPLATE),
)
result = agent.run_sync('Python 3.13 is fast and developer-friendly.')
print(result.output)
```

### Suppress schema injection (`template=False`)

When the provider sends the schema via API (e.g. OpenAI's `response_format=json_schema`), avoid duplicating it in the prompt:

```python
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput

class Report(BaseModel):
    title: str
    summary: str
    word_count: int

# OpenAI handles the schema natively — no text injection
agent = Agent('openai:gpt-4o', output_type=NativeOutput(Report, template=False))
```

### Per-call output type override with `NativeOutput`

Override `output_type` at `run()` time for dynamic schemas:

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput

class ShortAnswer(BaseModel):
    answer: str

class DetailedAnswer(BaseModel):
    answer: str
    reasoning: str
    confidence: float

base_agent = Agent('openai:gpt-4o')

async def main():
    # Simple question
    r1 = await base_agent.run(
        'What is 2+2?',
        output_type=NativeOutput(ShortAnswer),
    )
    print(r1.output.answer)

    # Complex question
    r2 = await base_agent.run(
        'Explain quantum entanglement.',
        output_type=NativeOutput(DetailedAnswer),
    )
    print(r2.output.reasoning)

asyncio.run(main())
```

## Advanced `PromptedOutput` patterns

### Multi-model portability

Use `PromptedOutput` when you need the same agent to run across providers with different structured-output support:

```python
import os
from pydantic import BaseModel
from pydantic_ai import Agent, PromptedOutput

class SummaryResult(BaseModel):
    title: str
    bullet_points: list[str]
    word_count: int

# Works with ANY text-capable model
model = os.getenv('MODEL', 'openai:gpt-4o')
agent = Agent(model, output_type=PromptedOutput(SummaryResult))
result = agent.run_sync('Summarise: Python is a high-level language loved for its readability.')
print(result.output)
```

### `PromptedOutput` with a branded system prompt

```python
from pydantic import BaseModel
from pydantic_ai import Agent, PromptedOutput

class TechReview(BaseModel):
    library: str
    rating: int  # 1-10
    pros: list[str]
    cons: list[str]
    verdict: str

SCHEMA_TEMPLATE = (
    'You are a senior software architect. Evaluate libraries objectively.\n\n'
    'Respond ONLY with a JSON object matching:\n{schema}'
)

agent = Agent(
    'groq:llama-3.3-70b-versatile',
    output_type=PromptedOutput(TechReview, template=SCHEMA_TEMPLATE),
)
result = agent.run_sync('Review the FastAPI web framework.')
print(f'{result.output.library}: {result.output.rating}/10')
```

## Advanced `TextOutput` patterns

### `TextOutput` with `RunContext` — deps-aware parsing

The parser function can take `RunContext` as its first argument to access dependencies, the run ID, or message history:

```python
import asyncio
import re
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext, TextOutput

@dataclass
class ParseDeps:
    decimal_separator: str = '.'  # '.' for US, ',' for EU

def parse_price(ctx: RunContext[ParseDeps], text: str) -> float:
    """Extract a price from the model output, respecting locale."""
    sep = ctx.deps.decimal_separator
    pattern = rf'\b\d{{1,3}}(?:[,\s]\d{{3}})*(?:{re.escape(sep)}\d{{1,2}})?\b'
    match = re.search(pattern, text)
    if match:
        raw = match.group(0).replace(' ', '')
        # Only strip commas as thousands separators when they are NOT the decimal separator
        if sep != ',':
            raw = raw.replace(',', '')
        raw = raw.replace(sep, '.')
        return float(raw)
    return 0.0

agent = Agent(
    'openai:gpt-4o',
    deps_type=ParseDeps,
    output_type=TextOutput(parse_price),
    instructions='State prices numerically, e.g. "The cost is 1,499.99".',
)

async def main():
    result = await agent.run('How much does a MacBook Pro cost?', deps=ParseDeps(decimal_separator='.'))
    print(f'Extracted price: {result.output:.2f}')

asyncio.run(main())
```

### Async `TextOutput` parser

```python
import asyncio
import httpx
from pydantic_ai import Agent, TextOutput

async def translate_to_german(text: str) -> str:
    """Translate the model's English output to German via an external API."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            'https://translate.example.com/v1/translate',
            json={'text': text, 'target': 'de'},
            timeout=10,
        )
        return resp.json().get('translated', text)

agent = Agent(
    'openai:gpt-4o',
    output_type=TextOutput(translate_to_german),
    instructions='Respond in English.',
)

async def main():
    result = await agent.run('Describe Python in one sentence.')
    print(result.output)   # German translation

asyncio.run(main())
```

## Reference

- `Agent.__init__(..., output_type=...)` — `agent/__init__.py:220`
- `ToolOutput`, `NativeOutput`, `PromptedOutput`, `TextOutput`, `StructuredDict` — `output.py`
- `output_validator` decorator — `agent/__init__.py:1911`
- `OutputSpec` type alias — `output.py`
- `TextOutputFunc` — `output.py` — `Callable[[str], T] | Callable[[RunContext, str], T]`
- Advanced patterns with `AgentSpec` and output — `pydantic_ai_advanced_classes_part2.md`

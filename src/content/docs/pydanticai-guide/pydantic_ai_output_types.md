---
title: "PydanticAI: Output Types & Validators"
description: "Typed outputs with ToolOutput, NativeOutput, PromptedOutput, TextOutput, StructuredDict, output functions, Choices, BoolCriteria, multi-type unions and agent.output_validator."
framework: "pydanticai"
language: "python"
sidebar:
  label: "Output types"
  order: 2
---

Verified against **pydantic-ai %%version=pydantic-ai%%** (checked %%versions-checked%%) — source modules: `pydantic_ai.output`, `pydantic_ai.agent`.

The `output_type` argument on `Agent` (or on a `run*` call) drives how the model returns structured data. It accepts a plain type or union, a list of types, a function (an *output function*), or one of the mode markers — `ToolOutput`, `NativeOutput`, `PromptedOutput`, `TextOutput` — and the schema helpers `StructuredDict` and `Choices`. Which marker is right depends on what the model natively supports.

## Minimal runnable example

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class Answer(BaseModel):
    value: int
    reasoning: str

agent = Agent('openai:%%model=openai%%', output_type=Answer)
result = agent.run_sync('What is 15 + 27?')
print(result.output)
#> value=42 reasoning='...'
print(type(result.output))
#> <class '__main__.Answer'>
```

Passing a bare type is a shortcut. The model's profile decides the mode (`ModelProfile['default_structured_output_mode']`), which is `'tool'` for almost every provider: the model is given an output tool whose arguments are your type's JSON schema. One exception is built in: when forcing a tool call would stop a thinking model from thinking, and the model supports JSON-schema output, `'native'` is used instead. To choose the mode yourself, wrap the type in one of the marker classes below.

## Output mode comparison

| Marker class   | How it works                                          | When to use                                                     |
| -------------- | ----------------------------------------------------- | --------------------------------------------------------------- |
| bare type      | Auto-picks `tool` / `native` / `prompted` from the model profile. | Default — you don't care which mechanism is used.             |
| `ToolOutput`   | Model emits a structured "output tool" call that PydanticAI validates. | You want to name the tool, set `strict`, or use multi-type unions on models that lack native JSON schema. |
| `NativeOutput` | Uses the provider's native structured-outputs API (e.g. OpenAI `response_format=json_schema`). | Model supports native JSON schema and you want maximum fidelity. |
| `PromptedOutput` | Puts the JSON schema in the instructions and parses the reply text. | Provider has no native structured outputs (local / older models). |
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
    'openai:%%model=openai%%',
    output_type=[
        ToolOutput(Fruit, name='return_fruit'),
        ToolOutput(Vehicle, name='return_vehicle'),
    ],
)
result = agent.run_sync('What is a banana?')
print(repr(result.output))
#> Fruit(name='banana', color='yellow')
```

Arguments (`output.py`):

| Arg | Type | Default | Notes |
|-----|------|---------|-------|
| `output` | `type \| callable` | required | Pydantic model, dataclass, or async callable |
| `name` | `str \| None` | `None` | Tool name sent to the model; auto-derived if unset |
| `description` | `str \| None` | `None` | Overrides the type's docstring as the tool description |
| `max_retries` | `int \| None` | `None` | Output-tool-specific retry budget; overrides the agent-level output budget (`Agent(retries={'output': N})`) |
| `strict` | `bool \| None` | `None` | Forwarded to providers that support strict tool schemas (OpenAI, Anthropic, Google, Bedrock among them); ignored elsewhere |
| `sequential` | `bool` | `False` | Act as a barrier — run alone, not overlapping with other tool calls (only meaningful under `end_strategy='exhaustive'`) |

### `ToolOutput.max_retries` — per-output retry budgets

`max_retries` lets you set a _different_ retry budget for each output branch. Expensive routes get fewer retries; cheap ones can afford more:

```python
from pydantic import BaseModel, field_validator
from pydantic_ai import Agent, ToolOutput

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
    'openai:%%model=openai%%',
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

### `ToolOutput.sequential` — barrier output under `exhaustive` end strategy

When `end_strategy='exhaustive'` the agent runs **all** tool calls the model emitted in a turn before stopping,
including multiple output tools. Setting `sequential=True` on a `ToolOutput` makes it act as a
barrier: function tools emitted before it finish first, then this output tool runs alone, then
tools emitted after it start. Under `'graceful'` (the default since 2.0) or `'early'`, output tools already run
sequentially, so `sequential=True` has no effect there.

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent, ToolOutput


class AuditRecord(BaseModel):
    """Audit record — must not overlap with live DB writes."""
    action: str
    timestamp: str
    user_id: int


class AnalysisResult(BaseModel):
    """Analysis result — safe to emit in parallel."""
    summary: str
    score: float


# AuditRecord must run alone (barrier); AnalysisResult can overlap with others.
agent = Agent(
    'openai:%%model=openai%%',
    output_type=[
        ToolOutput(AuditRecord, name='audit',    sequential=True),   # barrier
        ToolOutput(AnalysisResult, name='analysis', sequential=False),
    ],
    end_strategy='exhaustive',
)


async def main() -> None:
    result = await agent.run('Analyse the last login and log an audit record.')
    print(repr(result.output))


asyncio.run(main())
```

## `NativeOutput` — provider-native JSON schema

Uses the provider's own structured-output mechanism (e.g. OpenAI's `response_format=json_schema`). The answer arrives as the response text rather than as a tool call, and providers that enforce the schema (OpenAI with `strict`, for example) constrain decoding to it. Pydantic still validates the result, and a validation failure is sent back as a retry like any other.

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
agent_single = Agent('openai:%%model=openai%%', output_type=NativeOutput(Fruit))
result = agent_single.run_sync('What is a banana?')
print(repr(result.output))
# Fruit(name='banana', color='yellow')

# Multiple types — provider builds a tagged union
agent_union = Agent(
    'openai:%%model=openai%%',
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

`strict=True` asks the provider to enforce the schema exactly. Strict mode accepts a narrower JSON Schema subset (on OpenAI, every property must be required and `additionalProperties` false, which PydanticAI's schema transformer arranges where it can). For a list of types, PydanticAI already wraps each member as `{"kind": "<TypeName>", "data": {...}}` under a `result` key, so the union is tagged for you; a `Literal` field of your own is only needed if your code wants the tag on the returned object:

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
    'openai:%%model=openai%%',
    output_type=NativeOutput([FruitResult, VehicleResult], strict=True),
)
```

### `NativeOutput` without schema injection (`template=False`)

Some models need the schema repeated in the prompt even in native mode, and their profile supplies a default template for it. `template=False` turns that prompt off; a string replaces it (`{schema}` is substituted):

```python
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput


class Vehicle(BaseModel):
    name: str
    wheels: int


# The ollama: prefix needs OLLAMA_BASE_URL, e.g. http://localhost:11434/v1
agent = Agent(
    'ollama:llama3.2',
    output_type=NativeOutput(Vehicle, template=False),
)
```

Arguments (`pydantic_ai.output.NativeOutput`):

- `outputs` — a type or sequence of types. A list produces a tagged union.
- `name`, `description` — override the schema name and description sent to the model.
- `strict` — forwarded to providers supporting strict JSON schema (OpenAI).
- `template` — overrides the schema-injection template; pass `False` to skip.

Native output needs a model whose profile sets `supports_json_schema_output`: OpenAI, Gemini, recent Anthropic models, Groq, Bedrock, Ollama and several OpenAI-compatible providers do. On any other model the request fails with `UserError('Native structured output is not supported by this model.')` — there is no silent fallback, so use `PromptedOutput` (or plain `ToolOutput`) there.

## `PromptedOutput` — schema-in-prompt fallback

Works with any model that produces text. The schema is embedded in the prompt and the returned text is parsed.

```python
from pydantic import BaseModel
from pydantic_ai import Agent, PromptedOutput


class Fruit(BaseModel):
    name: str
    color: str


class Vehicle(BaseModel):
    name: str
    wheels: int


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

agent = Agent('openai:%%model=openai%%', output_type=TextOutput(split_words))
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

agent = Agent('openai:%%model=openai%%', output_type=DynamicForm)
result = agent.run_sync('Make up a blog post.')
print(result.output)
#> {'title': '...', 'tags': [...]}
```

`StructuredDict` returns a `dict[str, Any]` subclass with the schema baked in — use it when the schema is data, not a declared Python type.

## Output functions — let the model call your function to finish

`output_type` can be a function. The model "returns" by calling it as an output tool, with arguments built from the function's signature and docstring; whatever the function returns becomes `result.output`. This is the place for the last step of a run that needs code: looking something up, writing a record, or rejecting the model's answer with `ModelRetry` so it tries again with feedback.

```python
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext

INVENTORY = {'A-1': 12, 'B-2': 0}


class Reservation(BaseModel):
    sku: str
    quantity: int


def reserve_stock(ctx: RunContext[None], sku: str, quantity: int) -> Reservation:
    """Reserve stock for the customer's order.

    Args:
        sku: Product code, for example 'A-1'.
        quantity: Number of units to reserve.
    """
    available = INVENTORY.get(sku)
    if available is None:
        raise ModelRetry(f'Unknown SKU {sku!r}; valid codes are {sorted(INVENTORY)}.')
    if quantity > available:
        raise ModelRetry(f'Only {available} units of {sku} are available.')
    INVENTORY[sku] = available - quantity
    return Reservation(sku=sku, quantity=quantity)


agent = Agent('openai:%%model=openai%%', output_type=reserve_stock)
result = agent.run_sync('Reserve two A-1 please.')
print(repr(result.output))
#> Reservation(sku='A-1', quantity=2)
```

The `RunContext` parameter is optional; drop it when the function does not need deps. Output functions can be `async`, can be mixed with types in a list (`output_type=[reserve_stock, Refusal]`), and can be wrapped in `ToolOutput`, `NativeOutput` or `PromptedOutput` to control the mode. Because the function runs when the model calls it, keep side effects idempotent: a later validation failure can make the model call it again.

## `Choices` — pick one of a set known at run time

For a fixed set of options, a `Literal` or an `Enum` is the right output type. When the set only exists at run time (the open tickets for this customer, the actions available on screen), `Choices(...)` builds the type for you. Give a list of keys, or a mapping from key to description so the model can tell the options apart:

```python
from pydantic_ai import Agent, Choice, Choices

open_tickets = ['T-101', 'T-102', 'T-107']   # loaded for this customer
TicketId = Choices(open_tickets, name='ticket', description='The ticket this message is about.')

agent = Agent('openai:%%model=openai%%', output_type=TicketId)
print(agent.run_sync('Any update on the printer ticket?').output)
#> T-102

Intent = Choices(
    {
        'refund': 'The customer wants their money back.',
        'replace': 'The customer wants a working unit instead.',
        'escalate': Choice('Nobody on this tier can resolve it.', value=3),   # resolves to 3
    },
    name='customer_intent',
)
intent_agent = Agent('openai:%%model=openai%%', output_type=Intent)
```

The result is the chosen key, or the `value=` of a `Choice` when one is given. A callable `value` is called when the model picks it, which turns the set into a menu of actions; such a set can only be an agent's `output_type`. Like `StructuredDict`, `Choices` returns a type, so it also works as a field of a Pydantic model or a tool parameter.

## Describing enum members and booleans

Field descriptions reach the model through the JSON schema. Two helpers extend that to the values themselves:

- `UseEnumMemberDocstrings`, mixed into an `Enum`, sends each member's docstring as its description.
- `BoolCriteria(true=..., false=...)` inside `Annotated[bool, ...]` says what each answer means; the field stays a plain `bool`.

```python
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BoolCriteria, UseEnumMemberDocstrings


class Urgency(UseEnumMemberDocstrings, str, Enum):
    LOW = 'low'
    """Can wait until next week."""
    HIGH = 'high'
    """The customer is blocked right now."""


class Triage(BaseModel):
    urgency: Urgency
    refunded: Annotated[
        bool, BoolCriteria(true='Money was returned to the customer.', false='No refund was issued.')
    ] = Field(description='Was a refund issued?')


agent = Agent('openai:%%model=openai%%', output_type=Triage)
result = agent.run_sync('We have sent the 40 pounds back to your card; the printer can wait.')
print(result.output)
#> urgency=<Urgency.LOW: 'low'> refunded=True
```

Both produce an `anyOf` of `const` values, each with its description, in the schema the model sees.

## Allowing "no output" with `None`

Add `None` to the output list (or use `Contact | None`) when "nothing found" is a legitimate answer. In tool mode the model gets a separate `final_result_None` tool, so declining is an explicit choice rather than an empty object:

```python
from pydantic import BaseModel
from pydantic_ai import Agent


class Contact(BaseModel):
    name: str
    email: str


agent = Agent('openai:%%model=openai%%', output_type=[Contact, None])
result = agent.run_sync('Extract the contact: "Thanks for the update, see you Monday."')
print(result.output)
#> None
```

## Output validators — `@agent.output_validator`

Run arbitrary code after the model produces an output, potentially asking the model to retry.

```python
from pydantic_ai import Agent, ModelRetry, RunContext

agent = Agent('openai:%%model=openai%%', output_type=str)

@agent.output_validator
async def no_profanity(ctx: RunContext[object], output: str) -> str:
    if 'damn' in output.lower():
        raise ModelRetry('Please respond without profanity.')
    return output
```

The validator may:

- Return the same value (possibly transformed / sanitised).
- Raise `ModelRetry(msg)` to feed a retry prompt back to the model.
- Raise `UnexpectedModelBehavior` to terminate the run.

Up to the agent's output retry budget (`retries={'output': N}`; a bare `retries=N` sets both the tool and output budgets, default 1) validator-triggered retries are allowed before the run fails. The separate `output_retries=` argument from V1 no longer exists.

## Output streaming

See the [streaming guide](/pydanticai-guide/pydantic_ai_streaming/). `StreamedRunResult.get_output()` validates the final assembled output using the same `output_type` pipeline.

## Gotchas

- **Union of bare types**: `output_type=[Fruit, Vehicle]` works but you lose per-type tool naming. Use `ToolOutput(...)` per branch when the model gets confused about which to emit.
- **`strict=True`**: strict mode supports only part of JSON Schema. If the provider rejects your schema, simplify the type (no open-ended `dict`, no recursive models) or drop `strict`.
- **`NativeOutput` on a model without JSON-schema output** raises `UserError` before any request is sent. It does not fall back to prompted mode; choose `PromptedOutput` explicitly for such models.
- **`TextOutput` and tools**: function tools emitted alongside the final text still run under the default `end_strategy='graceful'` (since 2.0); only `'early'` would skip them.
- **`output_type` on `run()`**: the per-run override raises `UserError('Cannot set a custom run `output_type` when the agent has output validators')` if the agent has an `output_validator`, because the validator's type would not match.

## Patterns

### 1. Discriminated routing between shape types

```python
from typing import Literal
from pydantic import BaseModel
from pydantic_ai import Agent

class Search(BaseModel):
    kind: Literal['search']
    query: str

class Action(BaseModel):
    kind: Literal['action']
    name: str

agent = Agent('openai:%%model=openai%%', output_type=Search | Action)
```

PydanticAI generates a tagged union schema; the `kind` literal gives the model an unambiguous label to produce.

### 2. Retry with a more specific constraint

```python
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext


class Answer(BaseModel):
    value: int
    reasoning: str


agent = Agent('openai:%%model=openai%%', output_type=Answer)


@agent.output_validator
async def must_contain_sources(ctx: RunContext[object], out: Answer) -> Answer:
    if 'source' not in out.reasoning.lower():
        raise ModelRetry('Make `reasoning` cite at least one source, e.g. "Source: ...".')
    return out
```

### 3. Convert model output into a domain type via `TextOutput`

```python
from datetime import date

from pydantic_ai import Agent, TextOutput

def parse_iso_date(text: str) -> date:
    return date.fromisoformat(text.strip())

agent = Agent('openai:%%model=openai%%', output_type=TextOutput(parse_iso_date))
```

### 4. Hybrid: structured output _plus_ a free-text summary

Use a Pydantic model whose schema contains both the structured and the prose fields; don't try to mix `TextOutput` and `ToolOutput` on the same run.

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent


class Report(BaseModel):
    summary: str = Field(description='Two or three sentences of prose for the reader.')
    findings: list[str]
    confidence: float = Field(ge=0, le=1)


agent = Agent('openai:%%model=openai%%', output_type=Report)
```

### 5. Runtime-built form with `StructuredDict`

```python
from typing import Any

from pydantic_ai import Agent, StructuredDict


def build_form_schema(fields: list[dict[str, Any]]) -> type[dict[str, Any]]:
    schema = {
        'type': 'object',
        'properties': {f['name']: f['schema'] for f in fields},
        'required': [f['name'] for f in fields if f.get('required')],
    }
    return StructuredDict(schema, name='form_response')


user_fields = [   # e.g. loaded from the form builder's database
    {'name': 'full_name', 'schema': {'type': 'string'}, 'required': True},
    {'name': 'age', 'schema': {'type': 'integer', 'minimum': 0}},
]
agent = Agent('openai:%%model=openai%%', output_type=build_form_schema(user_fields))
print(agent.run_sync('I am Ada Lovelace, 36 years old.').output)
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
agent = Agent('openai:%%model=openai%%', output_type=NativeOutput(Report, template=False))
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

base_agent = Agent('openai:%%model=openai%%')

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
model = os.getenv('MODEL', 'openai:%%model=openai%%')
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
    'openai:%%model=openai%%',
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
    'openai:%%model=openai%%',
    output_type=TextOutput(translate_to_german),
    instructions='Respond in English.',
)

async def main():
    result = await agent.run('Describe Python in one sentence.')
    print(result.output)   # German translation

asyncio.run(main())
```

## Reference

- `Agent(..., output_type=...)` — `pydantic_ai.Agent`
- `ToolOutput`, `NativeOutput`, `PromptedOutput`, `TextOutput`, `StructuredDict` — `pydantic_ai.output` (all re-exported from `pydantic_ai`)
- `@agent.output_validator` decorator — `pydantic_ai.Agent.output_validator`
- `OutputSpec` type alias — `pydantic_ai.output`
- `TextOutputFunc` — `pydantic_ai.output` — `Callable[[str], T] | Callable[[RunContext, str], T]`
- `Choices`, `Choice`, `BoolCriteria` — `pydantic_ai.output` (re-exported from `pydantic_ai`); `UseEnumMemberDocstrings` — `pydantic_ai`
- Defining an agent, including its output, as data — see [Agents from a spec](/pydanticai-guide/pydantic_ai_comprehensive_guide/#agents-from-a-spec) in the comprehensive guide

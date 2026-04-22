---
title: "PydanticAI: Output Types & Validators"
description: "Typed outputs with ToolOutput, NativeOutput, PromptedOutput, TextOutput, StructuredDict, multi-type unions, and agent.output_validator."
framework: pydanticai
language: python
---

# Output Types & Validators

Verified against **pydantic-ai==1.85.1** — source: `/tmp/pydantic-ai-install/pydantic_ai/output.py`, `/tmp/pydantic-ai-install/pydantic_ai/agent/__init__.py`.

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

Arguments (`output.py:76`):

- `type_` — the Pydantic model, dataclass, or callable.
- `name` — tool name sent to the model. Default is `final_result` for single outputs; for multi-type outputs, the type's name is appended.
- `description` — overrides the docstring as the tool's description.
- `max_retries` — output-tool-specific retry budget; overrides the agent-level `retries` / `output_retries`.
- `strict` — forwarded to providers that support strict JSON schema (OpenAI).

## `NativeOutput` — provider-native JSON schema

```python
from pydantic_ai import Agent, NativeOutput

agent = Agent(
    'openai:gpt-5.2',
    output_type=NativeOutput(
        [Fruit, Vehicle],
        name='Fruit or vehicle',
        description='Return a fruit or vehicle.',
    ),
)
result = agent.run_sync('What is a Ford Explorer?')
print(repr(result.output))
#> Vehicle(name='Ford Explorer', wheels=4)
```

Arguments (`output.py:141`):

- `outputs` — a type or sequence of types. A list produces a tagged union.
- `name`, `description`, `strict`, `template`.
- `template` — overrides the schema-injection template used if the profile still requires it. Pass `False` to skip the schema prompt entirely.

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

## Reference

- `Agent.__init__(..., output_type=...)` — `agent/__init__.py:220`
- `ToolOutput`, `NativeOutput`, `PromptedOutput`, `TextOutput`, `StructuredDict` — `output.py`
- `output_validator` decorator — `agent/__init__.py:1911`
- `OutputSpec` type alias — `output.py`

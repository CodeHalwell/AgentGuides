---
title: "PydanticAI — Class Deep Dives Vol. 12"
description: "Source-verified deep dives into 10 class groups from pydantic-ai 1.106.0 and pydantic-evals 1.106.0: Dataset + Case (evaluation dataset management), Evaluator + EvaluatorContext + EvaluationResult + EvaluationReason (evaluator base framework), built-in evaluators Equals/EqualsExpected/Contains/IsInstance/MaxDuration/HasMatchingSpan, LLMJudge + GradingOutput + judge functions (LLM-as-judge), generate_dataset (AI-assisted test case generation), online evaluation evaluate decorator + OnlineEvalConfig + OnlineEvaluator (production traffic evaluation), SpanTree + SpanNode + SpanQuery (OTel span inspection in evaluators), MCPSamplingModel + MCPSamplingModelSettings (MCP sampling callback model), RetryConfig + TenacityTransport + AsyncTenacityTransport + wait_retry_after (tenacity-based HTTP retry infra), ExternalToolset (externally-resolved tool execution). All verified against pydantic-ai 1.106.0 / pydantic-evals 1.106.0 source."
sidebar:
  label: "Class deep dives (Vol. 12)"
  order: 38
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 1.106.0** / **pydantic-evals 1.106.0** source installed directly from PyPI. Class signatures, field names, and behaviour match the installed packages at these versions.
</Aside>

Ten class groups spanning evaluation infrastructure (`pydantic_evals`), production monitoring, and infrastructure primitives: `Dataset` + `Case` (the YAML-serialisable evaluation dataset); `Evaluator` + `EvaluatorContext` + `EvaluationResult` + `EvaluationReason` (the core evaluator API); the six built-in evaluators (`Equals`, `EqualsExpected`, `Contains`, `IsInstance`, `MaxDuration`, `HasMatchingSpan`); `LLMJudge` + `GradingOutput` + the four judge utility functions (LLM-as-judge); `generate_dataset` (AI-assisted test data generation); online evaluation via the `evaluate` decorator + `OnlineEvalConfig` + `OnlineEvaluator` (continuous production traffic evaluation); `SpanTree` + `SpanNode` + `SpanQuery` (structural OTel span inspection inside evaluators); `MCPSamplingModel` + `MCPSamplingModelSettings` (model that routes LLM calls back through an MCP client); `RetryConfig` + `TenacityTransport` + `AsyncTenacityTransport` + `wait_retry_after` (tenacity-based HTTP transport retry); and `ExternalToolset` (register tool definitions that are executed outside the agent run).

---

## 1. `Dataset` + `Case` — Evaluation Dataset Management

**Module:** `pydantic_evals.dataset`  
**Import:** `from pydantic_evals import Dataset, Case`

`Dataset` is a typed, serialisable collection of `Case` objects. It drives the evaluation loop (`evaluate` / `evaluate_sync`), loads and saves to YAML/JSON, and supports dataset-level and per-case evaluators.

### `Case` dataclass

```python
@dataclass(init=False)
class Case(Generic[InputsT, OutputT, MetadataT]):
    name: str | None
    inputs: InputsT
    metadata: MetadataT | None = None
    expected_output: OutputT | None = None
    evaluators: list[Evaluator[InputsT, OutputT, MetadataT]]

    def __init__(
        self,
        *,
        name: str | None = None,
        inputs: InputsT,
        metadata: MetadataT | None = None,
        expected_output: OutputT | None = None,
        evaluators: tuple[Evaluator[...], ...] = (),
    ): ...
```

`evaluators` must be a `tuple` on construction (pyright inference limitation) but is stored as a `list`.

### `Dataset` class

```python
class Dataset(BaseModel, Generic[InputsT, OutputT, MetadataT], extra='forbid'):
    name: str | None = None
    cases: list[Case[InputsT, OutputT, MetadataT]]
    evaluators: list[Evaluator[InputsT, OutputT, MetadataT]] = []
    report_evaluators: list[ReportEvaluator[...]] = []

    async def evaluate(
        self,
        task: Callable[[InputsT], Awaitable[OutputT]] | Callable[[InputsT], OutputT],
        *,
        name: str | None = None,
        max_concurrency: int | None = None,
        progress: bool = True,
        retry_task: RetryConfig | None = None,
        retry_evaluators: RetryConfig | None = None,
        task_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        repeat: int = 1,
        lifecycle: type[CaseLifecycle[...]] | None = None,
    ) -> EvaluationReport[InputsT, OutputT, MetadataT]: ...

    def evaluate_sync(self, task, ...) -> EvaluationReport: ...

    def add_case(self, *, name=None, inputs, metadata=None, expected_output=None, evaluators=()) -> None: ...
    def add_evaluator(self, evaluator, specific_case: str | None = None) -> None: ...

    @classmethod
    def from_file(cls, path, fmt=None, custom_evaluator_types=()) -> Self: ...
    def to_file(self, path, fmt=None, custom_evaluator_types=()) -> None: ...
    @classmethod
    def from_text(cls, text, fmt, ...) -> Self: ...
```

### Minimal runnable example

```python
import asyncio
from dataclasses import dataclass
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


@dataclass
class ExactMatch(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return ctx.output == ctx.expected_output


dataset: Dataset[str, str, None] = Dataset(
    name='shout_tests',
    cases=[
        Case(name='hello', inputs='hello', expected_output='HELLO'),
        Case(name='world', inputs='world', expected_output='WORLD'),
    ],
    evaluators=[ExactMatch()],
)


async def shout(text: str) -> str:
    return text.upper()


async def main():
    report = await dataset.evaluate(shout)
    report.print()


asyncio.run(main())
```

### YAML round-trip

```python
# Save to YAML then reload — uses pydantic_evals' YAML schema
dataset.to_file('shout_tests.yaml')

# Reload — requires explicit generic so the types are known
reloaded = Dataset[str, str, None].from_file('shout_tests.yaml')
```

A saved YAML file looks like:

```yaml
name: shout_tests
cases:
  - name: hello
    inputs: hello
    expected_output: HELLO
  - name: world
    inputs: world
    expected_output: WORLD
evaluators:
  - ExactMatch
```

### `add_case` / `add_evaluator`

```python
from dataclasses import dataclass
from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.evaluators.common import MaxDuration


@dataclass
class LengthCheck(Evaluator):
    min_length: int = 1

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return len(ctx.output) >= self.min_length


ds: Dataset[str, str, None] = Dataset(name='dynamic', cases=[])

ds.add_case(name='short', inputs='hi', expected_output='HI')
ds.add_case(name='long',  inputs='hello world', expected_output='HELLO WORLD')

# Dataset-level evaluator — applies to all cases
ds.add_evaluator(MaxDuration(seconds=1.0))

# Case-level evaluator — only for 'long'
ds.add_evaluator(LengthCheck(min_length=5), specific_case='long')
```

### Concurrency + retry

```python
from tenacity import retry_if_exception_type, stop_after_attempt
from httpx import HTTPStatusError
from pydantic_ai.retries import RetryConfig

report = await dataset.evaluate(
    shout,
    max_concurrency=4,
    retry_task=RetryConfig(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        reraise=True,
    ),
)
```

### `repeat` — stability testing

Run each case multiple times to measure variance:

```python
report = await dataset.evaluate(
    shout,
    repeat=5,       # each case runs 5 times; results are grouped by case name
    max_concurrency=10,
)
report.print()
```

---

## 2. `Evaluator` + `EvaluatorContext` + `EvaluationResult` + `EvaluationReason`

**Module:** `pydantic_evals.evaluators`  
**Import:** `from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationResult, EvaluationReason, EvaluatorOutput`

### `EvaluatorContext`

The single argument passed to every evaluator:

```python
@dataclass(kw_only=True)
class EvaluatorContext(Generic[InputsT, OutputT, MetadataT]):
    name: str | None             # case name
    inputs: InputsT              # task inputs
    metadata: MetadataT | None   # case metadata
    expected_output: OutputT | None
    output: OutputT              # actual task output
    duration: float              # seconds
    attributes: dict[str, Any]   # set via set_eval_attribute()
    metrics: dict[str, int|float]# set via increment_eval_metric()

    @property
    def span_tree(self) -> SpanTree: ...  # raises SpanTreeRecordingError if OTel unavailable
```

### `EvaluationReason` — scalar + explanation

```python
@dataclass
class EvaluationReason:
    value: EvaluationScalar      # bool | int | float | str
    reason: str | None = None
```

### `EvaluatorOutput` — union return type

```python
EvaluatorOutput = (
    EvaluationScalar              # bool | int | float | str
    | EvaluationReason
    | Mapping[str, EvaluationScalar | EvaluationReason]
)
```

### `Evaluator` abstract base

```python
@dataclass(repr=False)
class Evaluator(BaseEvaluator, Generic[InputsT, OutputT, MetadataT]):
    @abstractmethod
    def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput | Awaitable[EvaluatorOutput]: ...

    def get_default_evaluation_name(self) -> str: ...  # override to rename output column
    def get_evaluator_version(self) -> str | None: ...  # override to tag versioned runs
```

### Custom evaluators — patterns

```python
import asyncio
from dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationReason


# 1. Boolean assertion
@dataclass
class StartsWithCapital(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return ctx.output[:1].isupper()


# 2. Float score
@dataclass
class OverlapScore(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> float:
        expected = set(str(ctx.expected_output or '').split())
        actual = set(str(ctx.output).split())
        if not expected:
            return 1.0
        return len(expected & actual) / len(expected)


# 3. EvaluationReason with explanation
@dataclass
class MaxWords(Evaluator):
    limit: int = 100

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        count = len(str(ctx.output).split())
        ok = count <= self.limit
        return EvaluationReason(
            value=ok,
            reason=None if ok else f'Got {count} words, limit is {self.limit}',
        )


# 4. Multiple metrics from one evaluator (mapping output)
@dataclass
class QualityBundle(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> dict:
        output = str(ctx.output)
        return {
            'non_empty': bool(output.strip()),
            'word_count': len(output.split()),
        }


# 5. Async evaluator (e.g. calls an external API)
@dataclass
class RemoteCheck(Evaluator):
    async def evaluate(self, ctx: EvaluatorContext) -> bool:
        await asyncio.sleep(0)          # stand-in for an actual async call
        return bool(ctx.output)


# 6. Versioned evaluator — lets dashboards filter retired runs
@dataclass
class SemanticSimilarity(Evaluator):
    threshold: float = 0.8

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return True  # placeholder — real impl would compute cosine similarity

    def get_evaluator_version(self) -> str | None:
        return 'v2'  # bump whenever scoring logic changes
```

### `EvaluationResult` — inspecting report data

```python
from pydantic_evals.evaluators import EvaluationResult

# After report = await dataset.evaluate(task):
for report_case in report.cases:
    for result in report_case.evaluations:
        if isinstance(result, EvaluationResult):
            # downcast to more specific type for type-safe access
            bool_result = result.downcast(bool)
            if bool_result is not None:
                print(f'{result.name}: {"✔" if bool_result.value else "✘"} — {result.reason}')
```

---

## 3. Built-in Evaluators — `Equals`, `EqualsExpected`, `Contains`, `IsInstance`, `MaxDuration`, `HasMatchingSpan`

**Module:** `pydantic_evals.evaluators.common`  
**Import:** `from pydantic_evals.evaluators.common import Equals, EqualsExpected, Contains, IsInstance, MaxDuration, HasMatchingSpan`

### Quick-reference

| Class | Checks | Key params |
|---|---|---|
| `Equals(value)` | `ctx.output == value` | `evaluation_name` |
| `EqualsExpected()` | `ctx.output == ctx.expected_output` | `evaluation_name` |
| `Contains(value)` | substring / list / dict containment | `case_sensitive`, `as_strings`, `evaluation_name` |
| `IsInstance(type_name)` | `type.__name__ == type_name` | `evaluation_name` |
| `MaxDuration(seconds)` | `ctx.duration <= seconds` | accepts `timedelta` too |
| `HasMatchingSpan(query)` | `ctx.span_tree.any(query)` | `SpanQuery` dict, `evaluation_name` |

### `Equals` and `EqualsExpected`

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators.common import Equals, EqualsExpected

# Equals: hard-coded expected value
dataset1: Dataset[str, str, None] = Dataset(
    name='echo',
    cases=[Case(name='shout', inputs='hello', expected_output='HELLO')],
    evaluators=[Equals('HELLO')],
)

# EqualsExpected: compare output to case.expected_output
dataset2: Dataset[str, str, None] = Dataset(
    name='echo2',
    cases=[Case(name='shout', inputs='hello', expected_output='HELLO')],
    evaluators=[EqualsExpected()],
)
```

### `Contains` — flexible containment

```python
from pydantic_evals.evaluators.common import Contains

# String containment (case-insensitive)
contains_hello = Contains('hello', case_sensitive=False)

# Dict key-value check — all pairs in value must be in output
contains_fields = Contains({'status': 'ok', 'code': 200})

# Coerce both sides to str before checking
contains_str = Contains('200', as_strings=True)
```

### `IsInstance`

```python
from pydantic_evals.evaluators.common import IsInstance

# checks type(ctx.output).__name__ == 'dict'
is_dict = IsInstance('dict')
is_base_model = IsInstance('BaseModel')  # matches any class named 'BaseModel'
```

### `MaxDuration`

```python
from datetime import timedelta
from pydantic_evals.evaluators.common import MaxDuration

fast_check  = MaxDuration(seconds=0.5)
http_budget = MaxDuration(seconds=timedelta(seconds=2))
```

### `HasMatchingSpan` — OTel span assertions

```python
from pydantic_evals.evaluators.common import HasMatchingSpan

# Assert an LLM call occurred inside the task
has_llm_span = HasMatchingSpan(
    query={'name_contains': 'chat'},
)

# Assert a tool was called AND it completed within 500 ms
has_fast_tool = HasMatchingSpan(
    query={
        'name_contains': 'tool',
        'max_duration': 0.5,
    },
    evaluation_name='fast_tool_call',
)

# Assert a specific attribute is present on any descendant span
has_model_span = HasMatchingSpan(
    query={
        'has_attribute_keys': ['gen_ai.request.model'],
    },
)
```

### Combining evaluators on one dataset

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators.common import (
    EqualsExpected, Contains, MaxDuration, HasMatchingSpan,
)

dataset: Dataset[str, str, None] = Dataset(
    name='translation_checks',
    cases=[
        Case(name='en_fr', inputs='Hello', expected_output='Bonjour'),
        Case(
            name='en_de',
            inputs='Thank you',
            expected_output='Danke',
            evaluators=(Contains('Dank', case_sensitive=False),),  # per-case
        ),
    ],
    evaluators=[
        EqualsExpected(),
        MaxDuration(seconds=2.0),
        HasMatchingSpan({'name_contains': 'chat'}),
    ],
)
```

---

## 4. `LLMJudge` + `GradingOutput` + Judge Utility Functions

**Modules:** `pydantic_evals.evaluators.common`, `pydantic_evals.evaluators.llm_as_a_judge`  
**Imports:**
```python
from pydantic_evals.evaluators.common import LLMJudge
from pydantic_evals.evaluators.llm_as_a_judge import (
    GradingOutput,
    judge_output,
    judge_input_output,
    judge_output_expected,
    judge_input_output_expected,
    set_default_judge_model,
)
```

### `GradingOutput`

```python
class GradingOutput(BaseModel, populate_by_name=True):
    reason: str
    pass_: bool = Field(validation_alias='pass', serialization_alias='pass')
    score: float     # 0.0 – 1.0
```

### Judge utility functions

```python
# Judge output only
result: GradingOutput = await judge_output(
    output='The capital of France is Paris.',
    rubric='Correctly identifies the capital of France',
)

# Judge input + output
result = await judge_input_output(
    inputs='What is the capital of France?',
    output='Paris',
    rubric='Answers the question correctly',
)

# Judge output vs expected output
result = await judge_output_expected(
    output='Paris',
    expected_output='The capital is Paris',
    rubric='Contains the same factual answer',
)

# All four: input + output + expected
result = await judge_input_output_expected(
    inputs='What is 2 + 2?',
    output='Four',
    expected_output='4',
    rubric='Is numerically equivalent to the expected output',
)

print(result.pass_, result.score, result.reason)
```

### Setting the default judge model

```python
from pydantic_evals.evaluators.llm_as_a_judge import set_default_judge_model

# Use a cheaper model for bulk judgement
set_default_judge_model('openai:gpt-4o-mini')

# Or use Anthropic
set_default_judge_model('anthropic:claude-3-5-haiku-latest')
```

### `LLMJudge` evaluator

```python
from pydantic_evals.evaluators.common import LLMJudge

# Assertion only (default) — bool pass/fail with reason
assert_factual = LLMJudge(
    rubric='The answer is factually correct',
    model='anthropic:claude-3-5-haiku-latest',
)

# Score only — float 0.0–1.0
score_fluency = LLMJudge(
    rubric='The text is fluent and natural',
    score={'evaluation_name': 'fluency', 'include_reason': True},
    assertion=False,
)

# Both assertion + score — names become <name>_pass and <name>_score
quality_judge = LLMJudge(
    rubric='Response addresses the user question completely',
    score={'evaluation_name': 'completeness'},
    assertion={'evaluation_name': 'completeness', 'include_reason': True},
    include_input=True,            # include task inputs in the judging prompt
    include_expected_output=True,  # include expected output too
)
```

### Full pipeline example

```python
import asyncio
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators.common import LLMJudge, MaxDuration


class QAInput(BaseModel):
    question: str


# Use TestModel so this runs without real API keys
agent = Agent(TestModel(custom_result_text='42 is the answer.'))


async def qa_task(inputs: QAInput) -> str:
    result = await agent.run(inputs.question)
    return result.output


dataset: Dataset[QAInput, str, None] = Dataset(
    name='qa_eval',
    cases=[
        Case(
            name='meaning_of_life',
            inputs=QAInput(question='What is the meaning of life?'),
            expected_output='42',
        ),
    ],
    evaluators=[
        MaxDuration(seconds=5.0),
        LLMJudge(
            rubric='The answer mentions 42',
            model='openai:gpt-4o-mini',
        ),
    ],
)


async def main():
    report = await dataset.evaluate(qa_task)
    report.print()


asyncio.run(main())
```

---

## 5. `generate_dataset` — AI-Assisted Test Case Generation

**Module:** `pydantic_evals.generation`  
**Import:** `from pydantic_evals.generation import generate_dataset`

```python
async def generate_dataset(
    *,
    dataset_type: type[Dataset[InputsT, OutputT, MetadataT]],
    path: Path | str | None = None,
    custom_evaluator_types: Sequence[type[Evaluator[...]]] = (),
    model: Model | KnownModelName = 'openai:gpt-5.2',
    n_examples: int = 3,
    extra_instructions: str | None = None,
) -> Dataset[InputsT, OutputT, MetadataT]: ...
```

The function:
1. Calls `dataset_type.model_json_schema_with_evaluators(custom_evaluator_types)` to get the full JSON schema.
2. Sends it to the LLM asking for `n_examples` valid cases.
3. Parses the response with `dataset_type.from_text(output, fmt='json')`.
4. Optionally saves to `path`.

### Example

```python
import asyncio
from pydantic import BaseModel
from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators.common import EqualsExpected, MaxDuration
from pydantic_evals.generation import generate_dataset


class MathInput(BaseModel):
    a: int
    b: int
    operation: str  # 'add' | 'multiply'


class MathDataset(Dataset[MathInput, float, None]):
    pass


async def main():
    dataset = await generate_dataset(
        dataset_type=MathDataset,
        n_examples=5,
        model='openai:gpt-4o',
        extra_instructions='Generate arithmetic problems covering edge cases like zero and negative numbers.',
        path='math_cases.yaml',  # saved for future re-use
    )
    print(f'Generated {len(dataset.cases)} cases')
    for case in dataset.cases:
        print(f'  {case.name}: {case.inputs} → {case.expected_output}')


asyncio.run(main())
```

### Custom evaluator types in generation

```python
from dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class WithinTolerance(Evaluator):
    tolerance: float = 0.01

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        if ctx.expected_output is None:
            return True
        return abs(ctx.output - ctx.expected_output) <= self.tolerance


async def main():
    dataset = await generate_dataset(
        dataset_type=MathDataset,
        custom_evaluator_types=[WithinTolerance],  # LLM sees this schema and can add it to cases
        n_examples=4,
        model='openai:gpt-4o',
    )
```

### Regeneration guard

```python
from pathlib import Path

CACHE = Path('math_cases.yaml')

async def get_dataset() -> MathDataset:
    if CACHE.exists():
        return MathDataset.from_file(CACHE)
    return await generate_dataset(
        dataset_type=MathDataset,
        path=CACHE,
        n_examples=10,
    )
```

---

## 6. Online Evaluation — `evaluate` Decorator + `OnlineEvalConfig` + `OnlineEvaluator`

**Module:** `pydantic_evals.online`  
**Import:** `from pydantic_evals.online import evaluate, OnlineEvalConfig, OnlineEvaluator, configure, wait_for_evaluations, disable_evaluation`

Online evaluation runs the same `Evaluator` classes used for offline datasets, but wired to production functions via a decorator. Results are emitted as `gen_ai.evaluation.result` OTel log events.

### `OnlineEvalConfig`

```python
@dataclass
class OnlineEvalConfig:
    default_sink: EvaluationSink | Sequence[...] | SinkCallback | None = None
    default_sample_rate: float | Callable[[SamplingContext], float | bool] = 1.0
    emit_otel_events: bool = True
    include_baggage: bool = True
    sampling_mode: SamplingMode = 'independent'  # or 'correlated'
    enabled: bool = True
    metadata: dict[str, Any] | None = None
    on_max_concurrency: OnMaxConcurrencyCallback | None = None
    on_sampling_error: OnSamplingErrorCallback | None = None
    on_error: OnErrorCallback | None = None
```

### `OnlineEvaluator` — per-evaluator overrides

```python
@dataclass
class OnlineEvaluator:
    evaluator: Evaluator
    sample_rate: float | Callable[[SamplingContext], float | bool] | None = None
    max_concurrency: int | None = None
    sink: EvaluationSink | Sequence[...] | None = None
    on_max_concurrency: OnMaxConcurrencyCallback | None = None
    on_sampling_error: OnSamplingErrorCallback | None = None
    on_error: OnErrorCallback | None = None
```

### Module-level `evaluate` decorator

```python
from dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online import evaluate, OnlineEvaluator


@dataclass
class IsNonEmpty(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return bool(ctx.output)


@dataclass
class MaxResponseWords(Evaluator):
    limit: int = 200

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return len(str(ctx.output).split()) <= self.limit


# Evaluate every call with IsNonEmpty; MaxResponseWords at 50% sample rate
@evaluate(
    IsNonEmpty(),
    OnlineEvaluator(MaxResponseWords(limit=150), sample_rate=0.5),
)
async def summarise(document: str) -> str:
    return document[:100]   # simplified placeholder
```

### Custom `OnlineEvalConfig`

```python
from pydantic_evals.online import OnlineEvalConfig, configure, evaluate, OnlineEvaluator

# Build a config that captures results in-memory for testing
captured: list = []

def capture_sink(payload):
    captured.append(payload)

my_config = OnlineEvalConfig(
    default_sink=capture_sink,
    default_sample_rate=1.0,
    emit_otel_events=False,     # suppress OTel in test environments
    sampling_mode='correlated',  # lower-rate evaluators are a subset of higher-rate ones
    on_error=lambda exc, ctx, ev, loc: print(f'Eval error ({loc}): {exc}'),
)

@my_config.evaluate(IsNonEmpty())
async def process(text: str) -> str:
    return text.strip()
```

### Global config and `wait_for_evaluations`

```python
from pydantic_evals.online import configure, wait_for_evaluations, disable_evaluation

# Module-level defaults — affects all @evaluate-decorated functions
configure(
    default_sample_rate=0.1,        # sample 10% of calls
    sampling_mode='correlated',
    on_error=lambda exc, ctx, ev, loc: print(f'Error: {exc}'),
)


async def run_test():
    # Temporarily disable all online evaluation (e.g. in unit tests)
    with disable_evaluation():
        result = await summarise('some document')

    # After enabling, flush all in-flight evaluators
    await wait_for_evaluations(timeout=30.0)
```

### Custom sink — fan-out to multiple destinations

```python
from pydantic_evals.online import EvaluationSink, SinkPayload


class LogfireSink(EvaluationSink):
    async def submit(self, payload: SinkPayload) -> None:
        import logfire
        for result in payload.results:
            logfire.info(
                'eval.result',
                name=result.name,
                value=result.value,
                reason=result.reason,
            )


@evaluate(
    OnlineEvaluator(
        IsNonEmpty(),
        sink=LogfireSink(),
        sample_rate=0.2,
        max_concurrency=8,
    ),
)
async def generate_summary(text: str) -> str:
    return text[:200]
```

---

## 7. `SpanTree` + `SpanNode` + `SpanQuery` — OTel Span Inspection

**Module:** `pydantic_evals.otel.span_tree`  
**Import:** `from pydantic_evals.otel.span_tree import SpanTree, SpanNode, SpanQuery`

`SpanTree` captures the full OpenTelemetry span hierarchy recorded during a task run and exposes query-based traversal. Inside an evaluator, access it via `ctx.span_tree`.

### `SpanNode` — individual span

```python
@dataclass(repr=False, kw_only=True)
class SpanNode:
    name: str
    trace_id: int
    span_id: int
    parent_span_id: int | None
    start_timestamp: datetime
    end_timestamp: datetime
    attributes: dict[str, AttributeValue]

    @property
    def duration(self) -> timedelta: ...
    @property
    def children(self) -> list[SpanNode]: ...
    @property
    def parent(self) -> SpanNode | None: ...

    def matches_query(self, query: SpanQuery) -> bool: ...
```

### `SpanQuery` — TypedDict with AND logic

```python
class SpanQuery(TypedDict, total=False):
    # Name conditions
    name_equals: str
    name_contains: str
    name_matches_regex: str

    # Attribute conditions
    has_attributes: dict[str, Any]
    has_attribute_keys: list[str]

    # Timing conditions
    min_duration: timedelta | float
    max_duration: timedelta | float

    # Logical combinations
    not_: SpanQuery
    and_: list[SpanQuery]
    or_: list[SpanQuery]

    # Child conditions
    min_child_count: int
    max_child_count: int
    some_child_has: SpanQuery
    all_children_have: SpanQuery
    no_child_has: SpanQuery

    # Descendant / ancestor conditions
    min_descendant_count: int
    some_descendant_has: SpanQuery
    all_descendants_have: SpanQuery
    no_descendant_has: SpanQuery
    min_depth: int
    max_depth: int
    some_ancestor_has: SpanQuery
    stop_recursing_when: SpanQuery
```

### `SpanTree` — query methods

```python
class SpanTree:
    def any(self, query: SpanQuery) -> bool: ...
    def all(self, query: SpanQuery) -> bool: ...
    def find(self, query: SpanQuery) -> SpanNode | None: ...
    def find_all(self, query: SpanQuery) -> list[SpanNode]: ...
    def roots(self) -> list[SpanNode]: ...
```

### Evaluators using `span_tree`

```python
from dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationReason
from pydantic_evals.otel.span_tree import SpanQuery


@dataclass
class AtMostOneLLMCall(Evaluator):
    """Fail if the task made more than one LLM request."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        llm_spans = ctx.span_tree.find_all({'name_contains': 'chat'})
        ok = len(llm_spans) <= 1
        return EvaluationReason(
            value=ok,
            reason=None if ok else f'Made {len(llm_spans)} LLM calls',
        )


@dataclass
class ToolCalledWithin(Evaluator):
    """Assert a named tool was called within a duration budget."""

    tool_name: str
    max_seconds: float

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return ctx.span_tree.any({
            'name_contains': self.tool_name,
            'max_duration': self.max_seconds,
        })


@dataclass
class NoErrorSpans(Evaluator):
    """Assert no span carried an error attribute."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        error_spans = ctx.span_tree.find_all({'has_attribute_keys': ['error.type']})
        ok = len(error_spans) == 0
        return EvaluationReason(
            value=ok,
            reason=None if ok else f'{len(error_spans)} spans had errors: {[s.name for s in error_spans]}',
        )
```

### Composite `SpanQuery` using logical operators

```python
# Span that is a child of the root AND ran longer than 100ms
deep_slow: SpanQuery = {
    'and_': [
        {'min_depth': 1},
        {'min_duration': 0.1},
    ],
}

# Either a chat span or a tool span
any_model_activity: SpanQuery = {
    'or_': [
        {'name_contains': 'chat'},
        {'name_contains': 'tool'},
    ],
}

# Root span must have at least two descendants that are tool calls
root_with_tools: SpanQuery = {
    'max_depth': 0,  # root only
    'min_descendant_count': 2,
    'some_descendant_has': {'name_contains': 'tool'},
}
```

### `increment_eval_metric` + `set_eval_attribute` — injecting context

These helpers, called from inside the *task* being evaluated, populate `ctx.metrics` and `ctx.attributes` in the evaluator:

```python
from pydantic_evals import increment_eval_metric, set_eval_attribute
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')


async def rag_task(query: str) -> str:
    # Mark how many documents were retrieved
    docs = ['doc1', 'doc2', 'doc3']
    increment_eval_metric('docs_retrieved', len(docs))
    set_eval_attribute('retrieval_strategy', 'bm25')

    result = await agent.run(query)
    return result.output


# In an evaluator:
@dataclass
class SufficientRetrieval(Evaluator):
    min_docs: int = 2

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return ctx.metrics.get('docs_retrieved', 0) >= self.min_docs
```

---

## 8. `MCPSamplingModel` + `MCPSamplingModelSettings`

**Module:** `pydantic_ai.models.mcp_sampling`  
**Import:** `from pydantic_ai.models.mcp_sampling import MCPSamplingModel, MCPSamplingModelSettings`

`MCPSamplingModel` lets a PydanticAI agent route its LLM calls *back through the MCP client* that is currently connected to the server — implementing [MCP Sampling](https://modelcontextprotocol.io/docs/concepts/sampling). This is useful in MCP server implementations where the server needs to call an LLM using the client's credentials.

### Class signatures

```python
class MCPSamplingModelSettings(ModelSettings, total=False):
    mcp_model_preferences: ModelPreferences  # from mcp.types


@dataclass
class MCPSamplingModel(Model):
    session: ServerSession       # from mcp.ServerSession
    default_max_tokens: int = 16_384  # required by MCP; used when max_tokens not in ModelSettings

    @property
    def model_name(self) -> str: return 'mcp-sampling'
    @property
    def system(self) -> str: return 'MCP'

    # Streaming not supported — raises NotImplementedError
    async def request_stream(self, ...) -> AsyncIterator[StreamedResponse]: ...
```

### Using `MCPSamplingModel` in an MCP server tool

```python
from mcp.server import Server
from mcp.server.session import ServerSession
from mcp.types import ModelPreferences
from pydantic_ai import Agent
from pydantic_ai.models.mcp_sampling import MCPSamplingModel, MCPSamplingModelSettings

app = Server('my-mcp-server')


@app.call_tool()
async def summarise(session: ServerSession, document: str) -> str:
    """Summarise a document using the MCP client's LLM."""
    model = MCPSamplingModel(session=session)
    agent = Agent(model=model, system_prompt='Summarise the following document concisely.')
    result = await agent.run(document)
    return result.output
```

### Model preferences — steer the client's model selection

MCP clients expose a `ModelPreferences` hint that lets the server express a preference for cost, speed, or intelligence:

```python
from mcp.types import ModelPreferences

async def smart_analysis(session: ServerSession, query: str) -> str:
    model = MCPSamplingModel(session=session, default_max_tokens=8192)
    agent = Agent(model=model)
    result = await agent.run(
        query,
        model_settings=MCPSamplingModelSettings(
            mcp_model_preferences=ModelPreferences(
                intelligencePriority=0.9,
                speedPriority=0.1,
                costPriority=0.0,
            ),
            max_tokens=4096,
            temperature=0.2,
        ),
    )
    return result.output
```

### `default_max_tokens` — why it exists

MCP's `create_message` call requires `max_tokens` but `ModelSettings.max_tokens` is optional. `MCPSamplingModel.default_max_tokens` (default `16_384`) acts as the fallback:

```python
# With a tight token budget:
model = MCPSamplingModel(session=session, default_max_tokens=512)

# With per-call override via settings:
result = await agent.run(
    'Write a haiku',
    model_settings={'max_tokens': 50},   # overrides default_max_tokens
)
```

### Limitation: no streaming

`MCPSamplingModel.request_stream` raises `NotImplementedError`. Always use `agent.run()` (not `agent.run_stream()`):

```python
async def mcp_tool_handler(session: ServerSession, prompt: str) -> str:
    model = MCPSamplingModel(session=session)
    agent = Agent(model=model)
    result = await agent.run(prompt)   # ✔ non-streaming only
    return result.output
```

---

## 9. `RetryConfig` + `TenacityTransport` + `AsyncTenacityTransport` + `wait_retry_after`

**Module:** `pydantic_ai.retries`  
**Import:** `from pydantic_ai.retries import RetryConfig, TenacityTransport, AsyncTenacityTransport, wait_retry_after`  
**Requires:** `pip install "pydantic-ai-slim[retries]"` (installs `tenacity`)

This module provides tenacity-based HTTP transport wrappers so you can wrap any `httpx` client with retry logic — including respecting `Retry-After` response headers.

### `RetryConfig` — TypedDict wrapping tenacity `@retry` kwargs

```python
class RetryConfig(TypedDict, total=False):
    sleep: Callable[[int | float], None | Awaitable[None]]
    stop: StopBaseT       # tenacity stop strategy
    wait: WaitBaseT       # tenacity wait strategy
    retry: SyncRetryBaseT | RetryBaseT  # which exceptions trigger retry
    before: Callable[[RetryCallState], None | Awaitable[None]]
    after: Callable[[RetryCallState], None | Awaitable[None]]
    before_sleep: Callable[[RetryCallState], None | Awaitable[None]] | None
    reraise: bool         # re-raise last exc vs raise RetryError
    retry_error_cls: type[RetryError]
    retry_error_callback: Callable[[RetryCallState], Any | Awaitable[Any]] | None
```

### `TenacityTransport` — synchronous

```python
from httpx import Client, HTTPStatusError, HTTPTransport
from tenacity import retry_if_exception_type, stop_after_attempt
from pydantic_ai.retries import RetryConfig, TenacityTransport, wait_retry_after

transport = TenacityTransport(
    config=RetryConfig(
        retry=retry_if_exception_type(HTTPStatusError),
        wait=wait_retry_after(max_wait=120),
        stop=stop_after_attempt(5),
        reraise=True,
    ),
    wrapped=HTTPTransport(),
    validate_response=lambda r: r.raise_for_status(),  # turns 4xx/5xx into exceptions
)

client = Client(transport=transport)
response = client.get('https://api.example.com/data')
```

### `AsyncTenacityTransport` — asynchronous

```python
import httpx
from httpx import AsyncHTTPTransport, HTTPStatusError
from tenacity import retry_if_exception_type, stop_after_attempt
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after

transport = AsyncTenacityTransport(
    config=RetryConfig(
        retry=retry_if_exception_type(HTTPStatusError),
        wait=wait_retry_after(max_wait=300),
        stop=stop_after_attempt(5),
        reraise=True,
    ),
    wrapped=AsyncHTTPTransport(),
    validate_response=lambda r: r.raise_for_status(),
)

async_client = httpx.AsyncClient(transport=transport)
```

### `wait_retry_after` — honour `Retry-After` headers

```python
from pydantic_ai.retries import wait_retry_after
from tenacity import wait_exponential

# Pure Retry-After strategy (falls back to exponential if header absent)
wait = wait_retry_after(max_wait=300)

# Custom fallback strategy
wait_with_fallback = wait_retry_after(
    fallback_strategy=wait_exponential(multiplier=2, max=60),
    max_wait=600,
)
```

`wait_retry_after` parses both `Retry-After: 30` (seconds) and `Retry-After: Wed, 21 Oct 2025 07:28:00 GMT` (HTTP date) formats.

### Attaching to a PydanticAI model's HTTP client

```python
import httpx
from httpx import AsyncHTTPTransport, HTTPStatusError
from tenacity import retry_if_exception_type, stop_after_attempt
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after

transport = AsyncTenacityTransport(
    config=RetryConfig(
        retry=retry_if_exception_type(HTTPStatusError),
        wait=wait_retry_after(max_wait=120),
        stop=stop_after_attempt(4),
        reraise=True,
    ),
    validate_response=lambda r: r.raise_for_status(),
)

http_client = httpx.AsyncClient(transport=transport)
model = OpenAIModel(
    model_name='gpt-4o',
    provider=OpenAIProvider(http_client=http_client),
)
agent = Agent(model)
```

### Inspecting retry state with `before_sleep`

```python
import logging
from tenacity import RetryCallState, retry_if_exception_type, stop_after_attempt

logger = logging.getLogger(__name__)


def log_retry(state: RetryCallState) -> None:
    logger.warning(
        'Retrying attempt %d after: %s',
        state.attempt_number,
        state.outcome.exception() if state.outcome else 'unknown',
    )


transport = AsyncTenacityTransport(
    config=RetryConfig(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        reraise=True,
        before_sleep=log_retry,
    ),
)
```

---

## 10. `ExternalToolset` — Externally-Resolved Tool Execution

**Module:** `pydantic_ai.toolsets.external`  
**Import:** `from pydantic_ai.toolsets.external import ExternalToolset`

`ExternalToolset` registers tool *definitions* with an agent run but signals that the results will be produced **outside** the current agent call — by an external system, human operator, or separate process. The agent receives `ToolCallPart` messages for these tools but the toolset itself never calls them.

`DeferredToolset` is a deprecated alias for `ExternalToolset` — migrate to `ExternalToolset`.

### Class definition

```python
class ExternalToolset(AbstractToolset[AgentDepsT]):
    tool_defs: list[ToolDefinition]

    def __init__(self, tool_defs: list[ToolDefinition], *, id: str | None = None): ...

    async def get_tools(self, ctx: RunContext) -> dict[str, ToolsetTool]: ...

    # Raises NotImplementedError — external tools cannot be called inside the agent
    async def call_tool(self, name, tool_args, ctx, tool) -> Any: ...
```

All tools registered via `ExternalToolset` have `kind='external'` in their `ToolDefinition`.

### Use case: human-in-the-loop approval queue

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.toolsets.external import ExternalToolset
from pydantic_ai import ToolDefinition, DeferredToolResults

# 1. Declare what tools exist — the agent can call them but won't execute them
approval_toolset = ExternalToolset(
    tool_defs=[
        ToolDefinition(
            name='send_email',
            description='Send an email to a recipient.',
            parameters_json_schema={
                'type': 'object',
                'properties': {
                    'to': {'type': 'string'},
                    'subject': {'type': 'string'},
                    'body': {'type': 'string'},
                },
                'required': ['to', 'subject', 'body'],
            },
        ),
        ToolDefinition(
            name='delete_record',
            description='Permanently delete a database record.',
            parameters_json_schema={
                'type': 'object',
                'properties': {'record_id': {'type': 'string'}},
                'required': ['record_id'],
            },
        ),
    ],
    id='human-approval',  # used for durable execution resume
)

agent = Agent('openai:gpt-4o')


async def run_with_approval_queue(user_request: str):
    # First pass — agent decides which tools to call
    async with agent.iter(user_request, toolsets=[approval_toolset]) as agent_run:
        async for node in agent_run:
            pass  # run until agent requests external tools

    messages = agent_run.result.all_messages()
    pending_calls: list[ToolCallPart] = []
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, ToolCallPart):
                pending_calls.append(part)

    if not pending_calls:
        print('Agent answered without external tools.')
        return

    # Human reviews and approves/rejects
    tool_results: dict[str, str] = {}
    for call in pending_calls:
        args = call.args_as_dict()
        approved = input(f'Approve {call.tool_name}({args})? [y/n]: ').strip() == 'y'
        tool_results[call.tool_call_id] = (
            'approved' if approved else 'rejected by operator'
        )

    # Second pass — resume with approved results
    deferred = DeferredToolResults.from_tool_call_parts(
        pending_calls, tool_results
    )
    result = await agent.run(
        None,
        message_history=messages,
        deferred_tool_results=deferred,
    )
    print(result.output)


asyncio.run(run_with_approval_queue('Please send a welcome email to alice@example.com and delete record #42'))
```

### `ExternalToolset` vs `ApprovalRequiredToolset`

| | `ExternalToolset` | `ApprovalRequiredToolset` |
|---|---|---|
| Tool execution | Outside the agent run entirely | Inside the run, after approval callback |
| Approval mechanism | Your code drives resume via `DeferredToolResults` | `approval_required_func` callback in-process |
| Best for | External systems, async human review queues, durable execution | Synchronous in-process HITL gates |
| Tool kind | `'external'` | inherits from wrapped toolset |

### Filtering which tools need external approval

```python
from pydantic_ai.toolsets.external import ExternalToolset
from pydantic_ai import FunctionToolset, ToolDefinition


# Combine: some tools execute locally, others are external
local_tools = FunctionToolset()


@local_tools.tool
def lookup_user(user_id: str) -> str:
    return f'User {user_id} found'


# Only destructive tools are external
external = ExternalToolset(
    tool_defs=[
        ToolDefinition(
            name='delete_user',
            description='Permanently delete a user account.',
            parameters_json_schema={
                'type': 'object',
                'properties': {'user_id': {'type': 'string'}},
                'required': ['user_id'],
            },
        ),
    ]
)

agent = Agent('openai:gpt-4o', toolsets=[local_tools])

async def run(query: str):
    # Inject the external toolset at run time
    result = await agent.run(query, toolsets=[external])
    return result
```

### Multimodal example — id for durable execution

```python
# Assign an id so durable execution frameworks can track this toolset
image_approval = ExternalToolset(
    tool_defs=[
        ToolDefinition(
            name='approve_image',
            description='Human-review and approve or reject a generated image.',
            parameters_json_schema={
                'type': 'object',
                'properties': {
                    'image_url': {'type': 'string'},
                    'context': {'type': 'string'},
                },
                'required': ['image_url'],
            },
        ),
    ],
    id='image-approval-queue',
)
```

---

## Cross-reference with previous volumes

| Topic | Volume |
|---|---|
| `DeferredToolResults` + `CallDeferred` + `ApprovalRequired` | Vol. 3 |
| `ApprovalRequiredToolset` | Vol. 2 |
| `PreparedToolset` | Vol. 2 |
| `MCPToolset` + MCP server integration | Vol. 3 |
| `FunctionToolset` (all params) | Vol. 10 |
| `AbstractToolset` (ABC) | Vol. 10 |
| `WrapperCapability` | Vol. 10 |
| `ConcurrencyLimitedModel` + rate limiting | Vol. 8 |
| `FallbackModel` | Vol. 2 |
| `AgentInstructions` + `AgentMetadata` | Vol. 11 |
| `common_tools` (DuckDuckGo, Tavily, Exa) | Vol. 9 |

---

## Revision history

| Date | Package version | Notes |
|---|---|---|
| 2026-06-07 | pydantic-ai 1.106.0, pydantic-evals 1.106.0 | Initial Vol. 12. Ten class groups deep-dived: `Dataset`/`Case` (YAML/JSON round-trip, `add_case`/`add_evaluator`, `repeat` for stability testing, `retry_task`/`retry_evaluators`); `Evaluator`/`EvaluatorContext`/`EvaluationResult`/`EvaluationReason` (all five `EvaluatorOutput` forms, async evaluators, versioned evaluators, `EvaluationResult.downcast`); built-in evaluators `Equals`/`EqualsExpected`/`Contains`/`IsInstance`/`MaxDuration`/`HasMatchingSpan` (all params, `case_sensitive`/`as_strings`, `timedelta` for `MaxDuration`, `SpanQuery` in `HasMatchingSpan`); `LLMJudge`/`GradingOutput` and the four `judge_*` functions (`judge_output`/`judge_input_output`/`judge_output_expected`/`judge_input_output_expected`; `set_default_judge_model`; `score`/`assertion` `OutputConfig`; `include_input`/`include_expected_output`); `generate_dataset` (full schema inference from generic params; `custom_evaluator_types`; save/cache guard); online evaluation `evaluate` decorator/`OnlineEvalConfig`/`OnlineEvaluator` (`sample_rate` float or callable; `sampling_mode` independent/correlated; `disable_evaluation()`; `wait_for_evaluations()`; custom `EvaluationSink`); `SpanTree`/`SpanNode`/`SpanQuery` (all query fields; `any`/`all`/`find`/`find_all`; composite queries with `and_`/`or_`/`not_`; `increment_eval_metric`/`set_eval_attribute`); `MCPSamplingModel`/`MCPSamplingModelSettings` (`default_max_tokens`; `ModelPreferences`; streaming limitation; MCP server tool example); `RetryConfig`/`TenacityTransport`/`AsyncTenacityTransport`/`wait_retry_after` (TypedDict fields; `validate_response`; `Retry-After` header parsing; attaching to OpenAI model http client; `before_sleep` logging); `ExternalToolset` (kind `'external'`; approval queue pattern; HITL comparison table; `id` for durable execution). |

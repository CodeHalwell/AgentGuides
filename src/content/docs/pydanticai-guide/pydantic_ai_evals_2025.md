---
title: "Pydantic AI: Evals Guide"
description: "Systematic evaluation of Pydantic AI agents with pydantic_evals: datasets and cases, built-in, custom, LLM-judge and agentic evaluators, YAML datasets, reports, Logfire, and online evaluation."
framework: "pydanticai"
language: "python"
---

Verified against **pydantic-ai %%version=pydantic-ai%%**, which pins the matching `pydantic-evals` release (checked %%versions-checked%%). The examples were run against that release.

> **This page was rewritten for V2.** Earlier editions built evals from hand-rolled classes and imported things such as `pydantic_ai.eval`, `PostgreSQLDurableBackend` and `A2AAgentRegistry` that do not exist. Evaluation lives in the **`pydantic_evals`** package, installed with `pydantic-ai` (the `evals` extra is part of the default bundle).

---

## Testing vs evaluating

- **Unit tests** check deterministic behaviour with `TestModel` / `FunctionModel` and `agent.override()` — see the [testing guide](/pydanticai-guide/pydantic_ai_testing/). They should pass every time.
- **Evals** measure a real model's behaviour over a set of cases and report scores and pass rates. They are expected to be imperfect; you track them over time, across prompt changes and across models.

`pydantic_evals` gives you three pieces:

| Piece | Class | Role |
|---|---|---|
| Case | `Case(name=, inputs=, expected_output=, metadata=, evaluators=)` | One scenario. |
| Dataset | `Dataset(name=, cases=, evaluators=)` | A named collection of cases plus evaluators applied to every case. |
| Evaluator | subclass of `Evaluator` | Turns `(inputs, output, expected_output, metadata, duration, spans)` into a bool (assertion), number (score) or string (label). |

`dataset.evaluate(task)` / `evaluate_sync(task)` runs `task(inputs)` for every case, runs the evaluators and returns an `EvaluationReport`.

## V2 changes to watch for

- `Dataset(name=...)` is **required**.
- `evaluate()` / `evaluate_sync()` options (`name`, `max_concurrency`, `progress`, `retry_task`, `retry_evaluators`, …) are **keyword-only**, as is constructing `EvaluationResult` / `EvaluatorFailure`.
- An evaluator's name comes from `Evaluator.get_serialization_name()` (was the `name` classmethod); the `evaluation_name` / `evaluator_version` class attributes became `get_default_evaluation_name()` / `get_evaluator_version()`.
- The `Python` evaluator was removed back in 1.0.1 for security reasons.

---

## 1. A first dataset

The task here is an agent with structured output. `TestModel` keeps the example offline — it always answers "Paris", so the Japan case scores 0. Swap in a real model to evaluate it properly.

```python
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance, MaxDuration


class Answer(BaseModel):
    city: str
    country: str


agent = Agent(TestModel(custom_output_args={'city': 'Paris', 'country': 'France'}), output_type=Answer)


async def answer_question(question: str) -> Answer:
    result = await agent.run(question)
    return result.output


@dataclass
class CorrectCity(Evaluator[str, Answer]):
    """1.0 when the city matches the expected answer, else 0.0."""

    def evaluate(self, ctx: EvaluatorContext[str, Answer]) -> float:
        if ctx.expected_output is None:
            return 1.0
        return 1.0 if ctx.output.city.lower() == ctx.expected_output.city.lower() else 0.0


dataset = Dataset(
    name='capitals',                      # required since 2.0
    cases=[
        Case(name='france', inputs='What is the capital of France?',
             expected_output=Answer(city='Paris', country='France')),
        Case(name='japan', inputs='What is the capital of Japan?',
             expected_output=Answer(city='Tokyo', country='Japan'),
             metadata={'difficulty': 'easy'}),
    ],
    evaluators=[IsInstance(type_name='Answer'), CorrectCity(), MaxDuration(seconds=10)],
)

report = dataset.evaluate_sync(answer_question, max_concurrency=4)
report.print(include_input=True, include_output=True)
print(report.averages())
```

```
                      Evaluation Summary: answer_question
┏━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID  ┃ Inputs       ┃ Outputs      ┃ Scores      ┃ Assertions ┃ Duration ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ france   │ What is the  │ city='Paris' │ CorrectCity │ ✔✔         │   15.4ms │
│          │ capital of   │ country='Fra │ : 1.00      │            │          │
│          │ France?      │ nce'         │             │            │          │
├──────────┼──────────────┼──────────────┼─────────────┼────────────┼──────────┤
│ japan    │ What is the  │ city='Paris' │ CorrectCity │ ✔✔         │   13.1ms │
│          │ capital of   │ country='Fra │ : 0.000     │            │          │
│          │ Japan?       │ nce'         │             │            │          │
├──────────┼──────────────┼──────────────┼─────────────┼────────────┼──────────┤
│ Averages │              │              │ CorrectCity │ 100.0% ✔   │   14.2ms │
│          │              │              │ : 0.500     │            │          │
└──────────┴──────────────┴──────────────┴─────────────┴────────────┴──────────┘
```

The two ticks per row are the assertions (`IsInstance`, `MaxDuration`); `CorrectCity` returns a float, so it appears as a score. Return a `bool` for an assertion, a number for a score, a string for a label, or a dict to emit several at once. Wrap a value in `EvaluationReason(value, reason='...')` to attach an explanation.

## 2. Built-in evaluators

| Evaluator | Kind | Checks |
|---|---|---|
| `EqualsExpected()` | assertion | `output == expected_output` (skipped when there is no expected output) |
| `Equals(value)` | assertion | `output == value` |
| `Contains(value, case_sensitive=True, as_strings=False)` | assertion | substring / item / key containment |
| `IsInstance(type_name)` | assertion | the output's type name |
| `MaxDuration(seconds)` | assertion | task duration under a limit |
| `LLMJudge(rubric, model=None, include_input=False, include_expected_output=False, score=False, assertion=...)` | assertion and/or score | an LLM grades the output against a rubric |
| `GEval(criteria, evaluation_steps, score_range=(1, 5), ...)` | score | G-Eval-style chain-of-thought grading |
| `HasMatchingSpan(query)` | assertion | the run's OpenTelemetry span tree contains a matching span |

`LLMJudge` and `GEval` default to `openai:gpt-5.2` as the judge; change it per evaluator with `model=` or globally with `pydantic_evals.evaluators.llm_as_a_judge.set_default_judge_model(...)`. Use a different, preferably stronger, model than the one under test.

```python
from pydantic_evals.evaluators import LLMJudge

polite = LLMJudge(
    rubric='The reply is polite, answers the question, and does not invent order details.',
    model='anthropic:claude-sonnet-4-6',
    include_input=True,
)
```

## 3. Agentic evaluators — judging the trajectory, not just the answer

These read the spans of the run, so they need OpenTelemetry tracing on. Logfire is the easiest way; `send_to_logfire=False` keeps everything local.

```python
import logfire
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import MaxModelRequests, MaxToolCalls, ToolCorrectness

# Agentic evaluators read the run's OpenTelemetry spans, so tracing must be on.
logfire.configure(send_to_logfire=False)
logfire.instrument_pydantic_ai()

agent = Agent(TestModel(), instructions='Use the weather tool, then answer.')


@agent.tool_plain
def get_weather(city: str) -> str:
    """Current weather for a city."""
    return f'Sunny in {city}'


async def task(question: str) -> str:
    return (await agent.run(question)).output


dataset = Dataset(
    name='weather-agent',
    cases=[Case(name='london', inputs='What is the weather in London?')],
    evaluators=[
        ToolCorrectness(expected_tools=['get_weather']),
        MaxToolCalls(max_calls=2),
        MaxModelRequests(max_requests=3),
    ],
)
report = dataset.evaluate_sync(task)
report.print()
```

| Evaluator | Asserts |
|---|---|
| `ToolCorrectness(expected_tools, allow_extra=False)` | the agent called exactly this multiset of tools |
| `TrajectoryMatch(expected_trajectory, order='in_order')` | the tool-call sequence matches |
| `ArgumentCorrectness(...)` | tools were called with the expected arguments |
| `MaxToolCalls(max_calls)` / `MaxModelRequests(max_requests)` | the run stayed within a step budget |

The report also shows run metrics (requests, input and output tokens) collected from the spans.

## 4. Datasets as YAML or JSON

Keep cases in version control next to the code:

```python
dataset.to_file('weather_cases.yaml')        # also writes weather_cases_schema.json for editor completion
loaded = Dataset[str, str].from_file('weather_cases.yaml', custom_evaluator_types=[CorrectCity])
```

The file looks like this — evaluators are stored by name with their arguments:

```yaml
# yaml-language-server: $schema=weather_cases_schema.json
name: weather-agent
cases:
- name: london
  inputs: What is the weather in London?
  metadata: null
  expected_output: null
  evaluators: []
evaluators:
- ToolCorrectness:
  - get_weather
- MaxToolCalls: 2
- MaxModelRequests: 3
```

Custom evaluators must be passed in `custom_evaluator_types=` when loading. `pydantic_evals.generation.generate_dataset(dataset_type=..., n_examples=...)` can draft cases with an LLM for you to review.

## 5. Reports, regressions and CI

- `report.print(include_input=True, include_output=True)` renders the table; `report.render()` returns it as a string.
- `report.averages()` gives aggregate scores, assertion pass rate and durations — assert on these in CI to catch regressions (e.g. fail the build if the pass rate drops below an agreed threshold).
- Pass `baseline=` to `report.print()` with an earlier report to show the differences side by side.
- `repeat=N` on `evaluate()` runs every case N times to expose flaky behaviour; `retry_task=` / `retry_evaluators=` retry transient failures.
- `max_concurrency=` bounds parallel cases — useful against provider rate limits.

## 6. Logfire

With Logfire configured, each `evaluate()` call becomes a trace: one span per case with inputs, output, scores and the agent's own spans underneath. The Logfire UI then lets you compare experiments over time. Nothing in the evaluation code changes — configuring Logfire is enough.

## 7. Online evaluation — sampling production traffic

`OnlineEvaluation` is a capability that runs evaluators in the background after each real agent run, without delaying the response:

```python
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online_capability import OnlineEvaluation


@dataclass
class OutputNotEmpty(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return bool(ctx.output)


agent = Agent(
    'openai:gpt-5.2',
    name='assistant',
    capabilities=[OnlineEvaluation(evaluators=[OutputNotEmpty()])],
)
```

Results are emitted as OpenTelemetry events attached to the run's span, so they land next to the trace in Logfire. Sampling rates and other settings come from `OnlineEvalConfig` (pass `config=`), and evaluation is skipped automatically inside `Dataset.evaluate()`.

---

## Gotchas

- **Make the task a function of `inputs` only.** Anything else it needs (deps, model choice) should be closed over or read from the case's `metadata`.
- **Do not judge with the model under test.** Self-grading inflates scores.
- **Scores are not assertions.** A float never fails a case; return a `bool` (or set `assertion=` on `LLMJudge`) when you need pass/fail.
- **Agentic evaluators need spans.** Without tracing configured they have nothing to inspect.

## Reference

- `Dataset`, `Case` — `pydantic_evals` (`dataset.py`)
- `Evaluator`, `EvaluatorContext`, `EvaluationReason` and the built-ins — `pydantic_evals.evaluators`
- `EvaluationReport` — `pydantic_evals.reporting`
- `OnlineEvaluation` — `pydantic_evals.online_capability`; `OnlineEvalConfig` — `pydantic_evals.online`
- `generate_dataset` — `pydantic_evals.generation`

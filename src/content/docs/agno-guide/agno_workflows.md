---
title: "Agno: workflows"
description: "Deterministic pipelines in Agno: steps built from agents, teams and functions, StepInput and StepOutput, parallel branches, conditions, loops, routers, CEL expressions, session state, early stopping, human review, streaming and nesting."
framework: "agno"
language: "python"
sidebar:
  label: "Workflows"
  order: 30
---

Checked against **agno %%version=agno%%** (checked %%versions-checked%%). Every example ran offline: function steps ran as written, and agent steps used a scripted stand-in model.

## When to use a workflow

A `Workflow` runs a fixed sequence of steps. Each step is an agent, a team, a Python function or another workflow, and its output becomes the next step's input. Control flow (branches, loops, fan-out) is in your code, not decided by a model. Use one when the order of work is known; use a [team](/agno-guide/agno_teams/) when a model should decide who does what.

## A first workflow

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.workflow import Step, StepInput, StepOutput, Workflow

researcher = Agent(name="Researcher", model=OpenAIResponses(id="%%model=openai%%"),
                   instructions="List three facts about the topic.")
writer = Agent(name="Writer", model=OpenAIResponses(id="%%model=openai%%"),
               instructions="Write one paragraph from the notes you are given.")


def tidy(step_input: StepInput) -> StepOutput:
    notes = step_input.previous_step_content or ""
    return StepOutput(content=notes.strip())


article = Workflow(
    name="Article",
    steps=[
        Step(name="research", agent=researcher),
        Step(name="tidy", executor=tidy),
        Step(name="write", agent=writer),
    ],
)

run = article.run("solar power in the UK")
print(run.status, run.content)                                # content of the last step
print([(s.step_name, s.content) for s in run.step_results])
```

`Workflow` takes keyword arguments only (`Workflow("name", ...)` was removed in 3.0). The run returns a `WorkflowRunOutput` with `content` (the last step's output), `step_results` (one `StepOutput` per step), `step_executor_runs` (the agent and team runs inside), `status` and `metrics`.

The `steps` list also accepts bare agents, teams and functions; Agno wraps each in a `Step` named after the agent or function:

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.workflow import StepInput, StepOutput, Workflow

echo = Agent(name="Echo", model=OpenAIResponses(id="%%model=openai%%"))


def shout(step_input: StepInput) -> StepOutput:
    return StepOutput(content=str(step_input.previous_step_content).upper())


wf = Workflow(name="Short", steps=[echo, shout])
print([s.step_name for s in wf.run("hello").step_results])   # ['Echo', 'shout']
```

## Step input and output

A function step receives a `StepInput` and returns a `StepOutput` (a generator of them, or an async function, also works).

| `StepInput` | |
|-------------|--|
| `input` | The workflow's input (string, dict, list or Pydantic model) |
| `previous_step_content` | The previous step's `content` |
| `previous_step_outputs` | Dict of step name → `StepOutput` for every earlier step |
| `additional_data` | The `additional_data=` dict passed to `run()` (and human-review input, below) |
| `images`, `videos`, `audio`, `files` | Media passed to the run or produced by earlier steps |
| `get_step_content(name)`, `get_step_output(name)` | Look up an earlier step by name, including steps nested in `Parallel`, `Condition` and so on |

| `StepOutput` | |
|--------------|--|
| `content` | Any value: string, dict, list or model |
| `success`, `error` | Mark a failed step without raising |
| `stop=True` | End the workflow after this step |
| `images`, `videos`, `audio`, `files` | Media to pass on |

A function step may also declare a `run_context` parameter to read or change the workflow's session state (see [Session state](#session-state)).

`Step` options: `name`, `description`, `agent` / `team` / `executor` / `workflow` (exactly one), `max_retries` (default 3; a step that raises is retried), `skip_on_failure` (continue with a failed `StepOutput` instead of raising), `add_workflow_history` and `num_history_runs` (send earlier workflow runs to an agent step), `strict_input_validation`, `human_review`.

## Parallel

`Parallel(*steps, name=...)` runs its steps concurrently and records each result:

```python
from agno.workflow import Parallel, Step, StepInput, StepOutput, Workflow


def search_news(si: StepInput) -> StepOutput:
    return StepOutput(content=f"news about {si.input}")


def search_papers(si: StepInput) -> StepOutput:
    return StepOutput(content=f"papers about {si.input}")


def combine(si: StepInput) -> StepOutput:
    found = si.get_step_content("gather")   # {'news': '...', 'papers': '...'}
    return StepOutput(content=f"{found['news']} | {found['papers']}")


wf = Workflow(
    name="Research",
    steps=[
        Parallel(Step(name="news", executor=search_news), Step(name="papers", executor=search_papers), name="gather"),
        Step(name="combine", executor=combine),
    ],
)
print(wf.run("heat pumps").content)   # news about heat pumps | papers about heat pumps
```

Read parallel results by the `Parallel`'s name (a dict keyed by step name) or by each inner step's name. Do not rely on `previous_step_content` after a `Parallel`: in testing it held only one branch's output.

## Condition

`Condition` runs `steps` when `evaluator` is true and `else_steps` (optional) otherwise. The evaluator is a function of `StepInput`, a boolean, or a CEL expression string:

```python
from agno.workflow import Condition, Step, StepInput, StepOutput, Workflow


def is_urgent(si: StepInput) -> bool:
    return "urgent" in str(si.input).lower()


def page_on_call(si: StepInput) -> StepOutput:
    return StepOutput(content="Paged the on-call engineer")


def file_ticket(si: StepInput) -> StepOutput:
    return StepOutput(content="Filed a ticket")


triage = Workflow(
    name="Triage",
    steps=[
        Condition(
            name="urgency",
            evaluator=is_urgent,               # or: 'input.contains("urgent")'
            steps=[Step(name="page", executor=page_on_call)],
            else_steps=[Step(name="ticket", executor=file_ticket)],
        )
    ],
)
print(triage.run("URGENT: database down").content)   # Paged the on-call engineer
print(triage.run("Typo on the pricing page").content) # Filed a ticket
```

### CEL expressions

Conditions, loop end conditions and router selectors accept [CEL](https://cel.dev/) strings, evaluated by `cel-python` (`pip install "agno[cel]"`; `agno.workflow.CEL_AVAILABLE` tells you whether it is installed, and `validate_cel_expression()` checks a string). Variables available:

- Condition and Router: `input`, `previous_step_content`, `previous_step_outputs`, `additional_data`, `session_state` (Router adds `step_choices` and must return a step name).
- Loop `end_condition`: `current_iteration`, `max_iterations`, `all_success`, `last_step_content`, `step_outputs`.

Examples: `'input.contains("urgent")'`, `'session_state.retry_count < 3'`, `'last_step_content.contains("DONE")'`. CEL strings are useful when workflows are defined as data (for example through AgentOS); functions are easier to test.

## Loop

`Loop` repeats its steps until `end_condition` returns true or `max_iterations` (default 3) is reached. A function end condition receives the iteration's `StepOutput` list:

```python
from typing import List

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.workflow import Loop, Step, StepInput, StepOutput, Workflow

drafter = Agent(name="Drafter", model=OpenAIResponses(id="%%model=openai%%"),
                instructions="Improve the draft. End with APPROVED when nothing is left to fix.")


def approved(outputs: List[StepOutput]) -> bool:
    return any("APPROVED" in str(o.content) for o in outputs)


wf = Workflow(
    name="Refine",
    steps=[Loop(name="refine", steps=[Step(name="draft", agent=drafter)], end_condition=approved, max_iterations=4)],
)
run = wf.run("A short product description for a kettle")
```

`forward_iteration_output=True` feeds each iteration's output into the next iteration instead of the original input.

## Router

`Router` picks one or more of its `choices` at run time. The `selector` returns the step(s) to run:

```python
from agno.workflow import Router, Step, StepInput, StepOutput, Workflow


def solve_maths(si: StepInput) -> StepOutput:
    return StepOutput(content="maths route")


def write_prose(si: StepInput) -> StepOutput:
    return StepOutput(content="prose route")


maths = Step(name="maths", executor=solve_maths)
prose = Step(name="prose", executor=write_prose)


def choose(si: StepInput):
    return [maths] if any(ch.isdigit() for ch in str(si.input)) else [prose]


wf = Workflow(name="Routed", steps=[Router(name="route", selector=choose, choices=[maths, prose])])
print(wf.run("What is 12 * 7?").content)    # maths route
print(wf.run("Write a limerick").content)   # prose route
```

Returning several steps runs them in order. A Router with `human_review=HumanReview(requires_user_input=True)` and no selector lets a person choose (see below).

## Grouping and nesting

`Steps(name=..., steps=[...])` groups steps so they can be used as one unit inside a `Condition`, `Loop` or `Router`. A `Workflow` can be a step of another workflow (`Step(workflow=inner)` or the bare workflow in the list).

For logic that does not fit the primitives, `steps` can be a single function that receives the workflow and a `WorkflowExecutionInput` and returns the result. You call agents yourself inside it:

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.workflow import Workflow, WorkflowExecutionInput

summariser = Agent(model=OpenAIResponses(id="%%model=openai%%"))


def pipeline(workflow: Workflow, execution_input: WorkflowExecutionInput):
    first = summariser.run(f"Summarise: {execution_input.input}")
    return f"Summary: {first.content}"


wf = Workflow(name="Custom", steps=pipeline)
print(wf.run("a long text").content)
```

## Session state

With a `db`, a workflow keeps `session_state` per `session_id` across runs. A function step reaches it through `run_context`:

```python
from agno.db.sqlite import SqliteDb
from agno.run import RunContext
from agno.workflow import Step, StepInput, StepOutput, Workflow


def count_runs(step_input: StepInput, run_context: RunContext) -> StepOutput:
    state = run_context.session_state
    state["runs"] = state.get("runs", 0) + 1
    # Stop after the counter step on every run but the first
    return StepOutput(content=f"run number {state['runs']}", stop=state["runs"] > 1)


def expensive_setup(step_input: StepInput) -> StepOutput:
    return StepOutput(content="did the one-off setup")


wf = Workflow(
    name="Counter",
    db=SqliteDb(db_file="agno.db"),
    session_state={},
    steps=[Step(name="count", executor=count_runs), Step(name="setup", executor=expensive_setup)],
)
print(wf.run("go", session_id="w1").content)   # did the one-off setup
print(wf.run("go", session_id="w1").content)   # run number 2  (the setup step did not run)
```

`add_workflow_history_to_steps=True` (with `num_history_runs`) sends earlier runs of the session to agent steps.

`Workflow(agent=WorkflowAgent(model=...))` puts a small agent in front of the workflow: on each run it either answers from the session's earlier workflow runs or calls the workflow once. `WorkflowAgent` accepts only `model`, `instructions`, `add_workflow_history` and `num_history_runs`.

## Human review

Since 3.0 every pause point in a workflow is configured with one object, `HumanReview`, passed as `human_review=` to `Step`, `Steps`, `Loop`, `Condition` or `Router`. The workflow pauses (status `PAUSED`), `run.active_step_requirements` lists what is needed, and `workflow.continue_run(run)` resumes. Give the workflow a `db`.

```python
from agno.db.sqlite import SqliteDb
from agno.workflow import HumanReview, OnReject, Step, StepInput, StepOutput, Workflow


def publish(si: StepInput) -> StepOutput:
    return StepOutput(content="Published")


wf = Workflow(
    name="Publish",
    db=SqliteDb(db_file="agno.db"),
    steps=[
        Step(
            name="publish",
            executor=publish,
            human_review=HumanReview(
                requires_confirmation=True,
                confirmation_message="Publish this article?",
                on_reject=OnReject.cancel,
            ),
        )
    ],
)

run = wf.run("article-123")
for req in run.active_step_requirements:
    print(req.step_name, req.confirmation_message)   # publish Publish this article?
    req.confirm()                                    # or req.reject(feedback="Not yet")
run = wf.continue_run(run)
print(run.status, run.content)                       # RunStatus.completed Published
```

With `OnReject.cancel` a rejection ends the run with status `CANCELLED`. `OnReject.skip` (the default for most primitives) skips the step; `OnReject.else_branch` (the default for `Condition`) runs `else_steps`.

User input works the same way. The `@pause` decorator puts the review on a function:

```python
from agno.db.sqlite import SqliteDb
from agno.workflow import StepInput, StepOutput, Workflow, pause


@pause(
    requires_user_input=True,
    user_input_message="Which region?",
    user_input_schema=[{"name": "region", "field_type": "str"}],
)
def provision(si: StepInput) -> StepOutput:
    region = si.additional_data["user_input"]["region"]
    return StepOutput(content=f"Provisioned in {region}")


wf = Workflow(name="Provision", db=SqliteDb(db_file="agno.db"), steps=[provision])
run = wf.run("new database")
req = run.active_step_requirements[0]
req.set_user_input(region="eu-west-2")
print(wf.continue_run(run).content)   # Provisioned in eu-west-2
```

Other `HumanReview` fields: `requires_output_review` (pause after the step so a person can approve or `edit()` its output), `requires_iteration_review` (loops), `allow_multiple_selections` (routers; resolve with `req.select("a", "b")`), `on_error` (`OnError.fail`, `skip` or `pause`), `max_retries`, `timeout` with `on_timeout` (`OnTimeout.cancel`, `skip` or `approve`). A requirement reports what it needs through `needs_confirmation`, `needs_user_input`, `needs_output_review` and `needs_route_selection`.

Agent and team steps whose tools pause (confirmation, user input, external execution) pause the workflow too. The step requirement then has `needs_executor_resolution == True`, and its `executor_requirements` holds the agent's requirements as plain dicts. Turn them back into `RunRequirement` objects, resolve them, put them back and continue:

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.run.requirement import RunRequirement
from agno.tools import tool
from agno.workflow import Step, Workflow


@tool(requires_confirmation=True)
def ship_order(order_id: str) -> str:
    """Ship an order."""
    return f"Shipped {order_id}"


shipper = Agent(name="Shipper", model=OpenAIResponses(id="%%model=openai%%"), tools=[ship_order])
wf = Workflow(name="Fulfil", db=SqliteDb(db_file="agno.db"), steps=[Step(name="ship", agent=shipper)])

run = wf.run("Ship order 42")
for step_req in run.active_step_requirements or []:
    if step_req.needs_executor_resolution:
        reqs = [RunRequirement.from_dict(r) if isinstance(r, dict) else r for r in step_req.executor_requirements]
        for r in reqs:
            r.confirm()
        step_req.executor_requirements = reqs
run = wf.continue_run(run)
```

## Streaming and events

```python
from agno.workflow import Step, StepInput, StepOutput, Workflow


def step_a(si: StepInput) -> StepOutput:
    return StepOutput(content="A")


wf = Workflow(name="Events", steps=[Step(name="a", executor=step_a)])
print([event.event for event in wf.run("x", stream=True, stream_events=True)])
# ['WorkflowStarted', 'StepStarted', 'StepOutput', 'StepCompleted', 'WorkflowCompleted']
```

Agent and team steps add their own events inside the step (`stream_executor_events=True` is the default). The primitives have events of their own (`ParallelExecutionStarted`, `ConditionExecutionStarted`, `LoopIterationStarted`, `RouterExecutionStarted` and their `...Completed` pairs), and pauses emit `WorkflowPaused`, `StepPaused` or `StepExecutorPaused`. `store_events=True` keeps events on the stored run.

## Running workflows

`run()` and `arun()` take `input`, `additional_data`, `session_id`, `user_id`, `session_state`, `images`/`audio`/`videos`/`files`, `stream`, `stream_events`, `dependencies` and `metadata`. With an async database use `arun()` (`run()` raises). `input_schema=` on the workflow validates the input as on an agent. Under [AgentOS](/agno-guide/agno_agentos/) a workflow is served at `POST /workflows/{workflow_id}/runs`, and `background=true` queues it.

## Workflow reference

| Class | Main arguments |
|-------|---------------|
| `Workflow` | `name`, `id`, `description`, `steps`, `db`, `session_id`, `user_id`, `session_state`, `agent`, `input_schema`, `add_workflow_history_to_steps`, `num_history_runs`, `stream`, `stream_events`, `store_events`, `metadata`, `dependencies`, `debug_mode`, `telemetry` |
| `Step` | `name`, `agent` / `team` / `executor` / `workflow`, `description`, `max_retries`, `skip_on_failure`, `add_workflow_history`, `num_history_runs`, `strict_input_validation`, `human_review` |
| `Steps` | `name`, `steps`, `description`, `human_review` |
| `Parallel` | `*steps`, `name`, `description`, `human_review` |
| `Condition` | `evaluator`, `steps`, `else_steps`, `name`, `description`, `human_review` |
| `Loop` | `steps`, `end_condition`, `max_iterations`, `forward_iteration_output`, `name`, `description`, `human_review` |
| `Router` | `choices`, `selector`, `name`, `description`, `human_review` |
| `HumanReview` | `requires_confirmation`, `confirmation_message`, `requires_user_input`, `user_input_message`, `user_input_schema`, `allow_multiple_selections`, `requires_output_review`, `output_review_message`, `requires_iteration_review`, `on_reject`, `on_error`, `max_retries`, `timeout`, `on_timeout` |

All importable from `agno.workflow`.

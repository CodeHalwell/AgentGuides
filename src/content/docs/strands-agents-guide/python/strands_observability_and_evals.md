---
title: "Strands Agents (Python): Observability and Evals"
description: "Tracing, metrics and logging for Strands agents in Python with OpenTelemetry, and testing agent behaviour with the strands-agents-evals package: cases, experiments, deterministic and LLM-judge evaluators."
framework: "strands"
language: "python"
sidebar:
  label: "Observability and evals"
  order: 5
---

Checked against `strands-agents` %%version=strands-agents%% (%%versions-checked%%) and `strands-agents-evals` %%version=strands-agents-evals%%. Every example ran against an offline Bedrock stub; spans were captured with an in-memory exporter.

## What you get without configuration

Each call returns metrics on the result, and the `strands` logger reports what the loop is doing:

```python
import logging

from strands import Agent, tool

logging.getLogger("strands").setLevel(logging.DEBUG)
logging.basicConfig(format="%(levelname)s | %(name)s | %(message)s")


@tool
def lookup(term: str) -> str:
    """Look up a term in the glossary.

    Args:
        term: Term to look up.
    """
    return f"{term}: a defined term"


agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[lookup], callback_handler=None)
result = agent("What does 'idempotent' mean?")

call = result.metrics.latest_agent_invocation
print(call.usage)                                 # tokens for this call
print(len(call.cycles), "cycles")
summary = result.metrics.get_summary()            # totals for the agent so far
print(summary["accumulated_usage"], summary["tool_usage"].keys())
```

`result.metrics` belongs to the agent, so `accumulated_usage`, `cycle_count` and `tool_metrics` (calls, successes, errors and time per tool) are lifetime totals; `latest_agent_invocation` is the call you just made.

## OpenTelemetry tracing

Strands creates OpenTelemetry spans using the OpenTelemetry API. If no tracer provider is configured, they go nowhere. `StrandsTelemetry` configures one:

```python
# pip install "strands-agents[otel]==%%version=strands-agents%%"
from strands.telemetry import StrandsTelemetry

telemetry = StrandsTelemetry()            # creates and registers a global tracer provider
telemetry.setup_otlp_exporter()           # OTLP over HTTP; reads OTEL_EXPORTER_OTLP_ENDPOINT / _HEADERS
telemetry.setup_console_exporter()        # print spans, for local debugging
telemetry.setup_meter(enable_otlp_exporter=True)   # metrics as well as traces
```

Configure it once at start-up, before creating agents. If your application already sets up OpenTelemetry, pass your provider instead (`StrandsTelemetry(tracer_provider=provider)`) or skip `StrandsTelemetry`: Strands uses whatever global provider exists.

The exporter is configured with the standard environment variables:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"          # collector, Jaeger, Langfuse, ...
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic <base64 key>"
export OTEL_SERVICE_NAME="support-agent"
```

### The spans

One agent call produces this tree:

```text
invoke_agent <agent name>                 gen_ai.operation.name=invoke_agent, gen_ai.agent.name, gen_ai.agent.tools
├── execute_event_loop_cycle              event_loop.cycle_id
│   ├── chat                              gen_ai.request.model, gen_ai.usage.* tokens
│   └── execute_tool <tool name>          gen_ai.tool.name, gen_ai.tool.call.id
└── execute_event_loop_cycle              event_loop.parent_cycle_id
    └── chat
```

Attributes follow the OpenTelemetry GenAI semantic conventions (`gen_ai.system` is `strands-agents`). A `Graph` or `Swarm` run adds an `invoke_graph` or `invoke_swarm` span around its agents' `invoke_agent` spans, so a whole multi-agent request is one trace.

Add your own attributes to every span of an agent with `trace_attributes`, which is the usual way to make traces searchable by user or session:

```python
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from strands import Agent
from strands.telemetry import StrandsTelemetry

telemetry = StrandsTelemetry()
exporter = InMemorySpanExporter()
telemetry.tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

agent = Agent(
    model="us.anthropic.%%model=anthropic%%",
    name="support-agent",
    trace_attributes={"session.id": "s-123", "user.id": "u-17", "tags": ["support", "prod"]},
    callback_handler=None,
)
agent("Hello")

for span in exporter.get_finished_spans():
    print(span.name, span.attributes.get("session.id"))
```

`baggage_attributes=` sets OpenTelemetry baggage for each call instead, which propagates to downstream services; `telemetry.setup_baggage_processor()` copies baggage onto spans.

### Sending traces to AWS

On AgentCore Runtime, install `aws-opentelemetry-distro` and start the app with `opentelemetry-instrument`; traces then appear in CloudWatch (GenAI Observability) and X-Ray without code changes. Elsewhere on AWS, run the ADOT collector and point `OTEL_EXPORTER_OTLP_ENDPOINT` at it. Details are on the [production page](/strands-agents-guide/python/strands_production_guide/#tracing-and-logs).

Trace content can include prompts and tool inputs. Treat your tracing backend as holding user data, and redact in a hook (or an OpenTelemetry span processor) where needed.

## Evals with strands-agents-evals

`strands-agents-evals` is a separate package for testing agents: you write cases, a task function that runs your agent on a case, and evaluators that score the output or the tool trajectory.

```bash
pip install strands-agents-evals==%%version=strands-agents-evals%%
```

```python
from strands import Agent, tool
from strands_evals import Case, Experiment
from strands_evals.evaluators import Contains, OutputEvaluator, ToolCalled


@tool
def get_order(order_id: str) -> dict:
    """Fetch an order.

    Args:
        order_id: Order ID.
    """
    return {"order_id": order_id, "status": "shipped"}


def run_agent(case: Case) -> dict:
    agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[get_order], callback_handler=None)
    result = agent(case.input)
    tools_used = [
        block["toolUse"]["name"]
        for message in agent.messages
        for block in message["content"]
        if "toolUse" in block
    ]
    return {"output": str(result), "trajectory": tools_used}


cases = [
    Case(name="order-status", input="Where is order ORD-1?", expected_output="shipped"),
    Case(name="order-status-2", input="Has ORD-2 shipped yet?", expected_output="shipped"),
]

experiment = Experiment(
    cases=cases,
    evaluators=[
        Contains("shipped", case_sensitive=False),          # deterministic
        ToolCalled("get_order"),                             # checks the trajectory
        OutputEvaluator(                                     # LLM judge
            rubric="Score 1.0 if the answer states the order's shipping status, otherwise 0.0.",
            model="us.anthropic.%%model=anthropic-small%%",
        ),
    ],
)

report = experiment.run_evaluations(run_agent)
print(report.overall_score)
for row, passed, reason in zip(report.cases, report.test_passes, report.reasons):
    print(row["evaluator"], row["name"], passed, reason[:60])
```

The task returns either the output alone or `{"output": ..., "trajectory": ...}`. The trajectory is whatever your trajectory evaluators expect: a list of tool names is enough for `ToolCalled`; the package's extractors (`strands_evals.extractors`) build richer ones from agent messages, OpenTelemetry sessions, swarm handoffs or graph results.

| Evaluator | Kind | Checks |
|-----------|------|--------|
| `Equals`, `Contains`, `StartsWith` | Deterministic | The output |
| `ToolCalled`, `StateEquals`, `SkillInvoked` | Deterministic | The trajectory or final state |
| `StructuredOutputSimilarity` | Deterministic | Field-by-field match of structured output |
| `OutputEvaluator(rubric=...)` | LLM judge | The output against your rubric |
| `TrajectoryEvaluator(rubric=...)` | LLM judge | The sequence of tool calls |
| `HelpfulnessEvaluator`, `FaithfulnessEvaluator`, `CorrectnessEvaluator`, `HarmfulnessEvaluator`, `ToolSelectionAccuracyEvaluator`, `ToolParameterAccuracyEvaluator`, `GoalSuccessRateEvaluator`, ... | LLM judge | Built-in rubrics. These work on an OpenTelemetry session as the trajectory (captured with `StrandsEvalsTelemetry`), not on a list of tool names |

`run_evaluations_async(task, max_workers=10)` runs cases concurrently (the sync form runs them one at a time and rejects `async` tasks). The report has `overall_score`, per-row `scores`, `test_passes`, `reasons` and `cases`; `report.display()` prints a table and `report.to_file(path)` saves it. `Experiment.to_file()` / `from_file()` save the cases and evaluators so a suite can live in version control.

The package also has generators that draft cases from a description (`strands_evals.generators.ExperimentGenerator`), tool simulators, failure detectors, and red-teaming and chaos modules. Those are newer and change more between releases. Multi-turn tests with a simulated user are covered next.

### Multi-turn conversations with a simulated user

A single prompt rarely shows whether an agent can finish a task that takes several exchanges. `ActorSimulator` (also exported as `UserSimulator`) wraps an agent that plays the user. `ActorSimulator.from_case_for_user_simulator(case, max_turns=...)` asks a model to write a user profile and goal from `case.input` and `case.metadata["task_description"]`, then `act()` produces the user's reply to each agent message. `has_next()` turns false when the simulated user decides it is done or `max_turns` (10 by default) is reached:

```python
from strands import Agent
from strands_evals import ActorSimulator, Case, Experiment
from strands_evals.evaluators import Contains


def run_conversation(case: Case) -> dict:
    """Let a simulated customer talk to the agent for up to four turns."""
    user = ActorSimulator.from_case_for_user_simulator(case=case, max_turns=4)
    agent = Agent(
        model="us.anthropic.%%model=anthropic%%",
        system_prompt="You are a travel assistant. Say 'booking confirmed' once a booking is made.",
        callback_handler=None,
    )
    transcript = []
    message = case.input
    while user.has_next():
        reply = str(agent(message))
        transcript.append(reply)
        message = str(user.act(reply).structured_output.message)
    return {"output": "\n".join(transcript)}


cases = [
    Case(
        name="paris-flight",
        input="I need to book a flight to Paris next Friday.",
        metadata={"task_description": "The user gets a confirmed booking."},
    )
]

experiment = Experiment(cases=cases, evaluators=[Contains("booking confirmed", case_sensitive=False)])
report = experiment.run_evaluations(run_conversation)
print(report.overall_score)
```

Points to know. The case input must be a string. Building the profile calls a model, and that call uses Strands' default model (Bedrock) whatever you pass as `model=`, so it needs Bedrock access. The simulator's replies are as variable as any model output, so judge the whole transcript with an LLM evaluator, or with a deterministic check as above, rather than a fixed expected string. `act(..., structured_output_model=...)` replaces the default `ActorResponse` with your own Pydantic model, which must have `message` and `stop` fields.

### Running evals in CI

LLM judges cost money and vary between runs, so a common split is: deterministic evaluators on every pull request, with a scripted model (see [testing without the cloud](/strands-agents-guide/python/strands_production_guide/#testing-without-the-cloud)) where the model's choices do not matter; the full suite with real models and LLM judges on a schedule, tracking `overall_score` over time rather than failing on a single bad run.

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-24 | %%version=strands-agents%% | Audit against the installed packages. Added a multi-turn evaluation example with `ActorSimulator` and listed the generator, tool simulator, detector, red-team and chaos modules. |
| 2026-10-24 | 1.62.0 | New page. Result metrics (per call and lifetime), `StrandsTelemetry`, the span tree and attributes captured from a real run, trace attributes, and `strands-agents-evals` experiments with deterministic and LLM-judge evaluators. |

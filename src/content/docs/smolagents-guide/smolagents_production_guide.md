---
title: "smolagents: Production Deployment Guide"
description: "Running smolagents reliably in production — sandboxing, cost and token tracking, telemetry, error handling, deployment, scaling and testing."
framework: "smolagents"
language: "python"
---

> Verified against `smolagents` %%version=smolagents%% (checked %%versions-checked%%).

---

## Table of Contents

1. [Production Readiness Checklist](#production-readiness-checklist)
2. [Performance Optimisation](#performance-optimisation)
3. [Cost Management](#cost-management)
4. [Monitoring & Observability](#monitoring--observability)
5. [Security Best Practices](#security-best-practices)
6. [Error Handling & Resilience](#error-handling--resilience)
7. [Deployment Options](#deployment-options)
8. [Scaling Strategies](#scaling-strategies)
9. [Testing & Quality Assurance](#testing--quality-assurance)
10. [Observability Integration (Weights & Biases Weave)](#observability-integration-weights--biases-weave)

---

## Production Readiness Checklist

### Core Functionality

| Area | Check |
|------|-------|
| Versions | `smolagents` and its extras pinned (`smolagents[...]==%%version=smolagents%%`); model provider SDKs pinned |
| Execution | `CodeAgent` uses a remote executor (`docker`, `e2b`, `modal`, `blaxel`) for any untrusted input; `LocalPythonExecutor` is not a security boundary |
| Imports | `additional_authorized_imports` lists only what tasks need; never `"*"` |
| Limits | `max_steps` set per agent; `executor_kwargs={"timeout_seconds": ...}` for the local executor; HTTP timeouts on the model (`timeout=` on `InferenceClientModel`, `timeout` kwarg on LiteLLM/OpenAI models) |
| Output | `final_answer_checks` validate the answer's shape; `RunResult.state` checked for `"max_steps_error"` |
| Tools | Each tool unit-tested; descriptions reviewed; side-effecting tools idempotent |
| State | One agent instance per concurrent run (agents keep memory) |
| Telemetry | OpenTelemetry/OpenInference or Weave enabled; token usage recorded per run |

A thin production wrapper:

```python
import logging
from dataclasses import dataclass

from smolagents import CodeAgent, InferenceClientModel, RunResult

logger = logging.getLogger(__name__)


@dataclass
class AgentOutcome:
    ok: bool
    output: object
    state: str
    steps: int
    input_tokens: int
    output_tokens: int
    seconds: float


def build_agent() -> CodeAgent:
    return CodeAgent(
        tools=[],
        model=InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", timeout=60),
        max_steps=10,
        executor_type="local",                  # switch to "docker"/"e2b" for untrusted input
        executor_kwargs={"timeout_seconds": 30},
        verbosity_level=0,
    )


def run_task(task: str) -> AgentOutcome:
    if not task or len(task) > 10_000:
        raise ValueError("Task must be between 1 and 10,000 characters")
    agent = build_agent()                       # fresh agent: no memory shared between requests
    result: RunResult = agent.run(task, return_full_result=True)
    usage = result.token_usage
    outcome = AgentOutcome(
        ok=result.state == "success",
        output=result.output,
        state=result.state,
        steps=len(result.steps),
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
        seconds=result.timing.duration or 0.0,
    )
    logger.info("agent run", extra={"state": outcome.state, "steps": outcome.steps,
                                    "tokens_in": outcome.input_tokens, "tokens_out": outcome.output_tokens})
    return outcome
```

### Infrastructure Requirements

- Model provider quota sized for peak concurrency × average steps per run.
- A sandbox backend for `CodeAgent`: Docker daemon reachable from the service (for `executor_type="docker"`), or E2B/Modal/Blaxel credentials.
- Somewhere to keep conversation state if you support multi-turn sessions (agents' memory is in-process).
- A telemetry backend (any OTLP collector, Langfuse, Arize Phoenix, Weave).

A start-up health check that exercises the model without running an agent:

```python
from smolagents import ChatMessage, InferenceClientModel
from smolagents.models import MessageRole


def check_model(model) -> bool:
    reply = model.generate([ChatMessage(role=MessageRole.USER, content="Reply with OK.")])
    return "OK" in (reply.content or "")
```

### Operational Requirements

- Alerting on error rate, `max_steps_error` rate, p95 latency and tokens per run.
- A runbook for provider outages (fallback model, degraded mode).
- Log retention rules that respect the personal data in prompts and tool outputs.

---

## Performance Optimisation

### Model Selection for Performance

Latency is dominated by the number of model calls and the tokens per call. In practice:

- A strong code model often finishes in fewer steps than a cheaper one, so compare *cost per completed task*, not price per token.
- Use a cheaper model for managed sub-agents with narrow tasks and a stronger one for the manager.
- Keep `planning_interval` unset for short tasks; planning adds a model call.

### Caching & Memoisation

Cache at the tool level, where results are deterministic, rather than caching whole agent answers:

```python
from functools import lru_cache

from smolagents import tool


@lru_cache(maxsize=1024)
def _fetch_rate(base: str, quote: str) -> float:
    # Replace with a real API call
    return 0.92


@tool
def fx_rate(base: str, quote: str) -> float:
    """Return the exchange rate from one currency to another.

    Args:
        base: ISO code of the currency to convert from, e.g. "USD".
        quote: ISO code of the currency to convert to, e.g. "EUR".
    """
    return _fetch_rate(base.upper(), quote.upper())
```

For whole-answer caching (FAQ-style traffic), key on the normalised task plus model ID and prompt version, and cache only `state == "success"` results.

### Prompt Optimisation

- Put durable guidance in `instructions=` rather than in every task string.
- Write tool descriptions that state the return format; vague descriptions cause extra exploratory steps.
- Return compact tool outputs; trim pages and tables before returning them.
- `max_print_outputs_length` stops a stray `print(df)` flooding the context.

### Parallel Execution

- `ToolCallingAgent(max_tool_threads=N)` runs a step's tool calls in parallel.
- A `CodeAgent` can call several tools in one code block; nothing more to configure.
- Across tasks, run separate agent instances in threads or processes:

```python
from concurrent.futures import ThreadPoolExecutor

tasks = ["Summarise doc A", "Summarise doc B", "Summarise doc C"]
with ThreadPoolExecutor(max_workers=4) as pool:
    outcomes = list(pool.map(run_task, tasks))     # run_task builds a fresh agent each time
```

---

## Cost Management

### Token Usage Tracking

Every `ActionStep` and `PlanningStep` carries `token_usage`, and `RunResult.token_usage` sums them. Attach prices outside the library:

```python
from smolagents import ActionStep, PlanningStep

PRICE_PER_MTOK = {"input": 0.50, "output": 1.50}    # your provider's prices, per million tokens


def run_cost(result) -> float:
    u = result.token_usage
    return (u.input_tokens * PRICE_PER_MTOK["input"] + u.output_tokens * PRICE_PER_MTOK["output"]) / 1_000_000


class StepCostLogger:
    def __init__(self):
        self.total = 0

    def __call__(self, step, agent=None):
        if step.token_usage:
            self.total += step.token_usage.total_tokens
            print(f"{type(step).__name__} {getattr(step, 'step_number', '')}: running total {self.total} tokens")


cost_logger = StepCostLogger()
# CodeAgent(..., step_callbacks={ActionStep: cost_logger, PlanningStep: cost_logger})
```

### Cost Optimisation Strategies

- **Cap steps.** `max_steps` is the main cost ceiling; set it per agent from observed distributions.
- **Stop early on budget.** A step callback can call `agent.interrupt()` when a token budget is exceeded.
- **Trim memory.** For vision or browsing agents, clear `observations_images` on old steps in a callback.
- **Right-size models** per role, and re-evaluate when providers release cheaper tiers.
- **Avoid re-runs.** Validate inputs before starting an agent; a malformed request that fails at step 8 still costs eight steps.

```python
def budget_guard(limit_tokens: int):
    used = 0
    def callback(step, agent=None):
        nonlocal used
        if step.token_usage:
            used += step.token_usage.total_tokens
        if used > limit_tokens and agent is not None:
            agent.interrupt()
    return callback
```

---

## Monitoring & Observability

### Key Metrics to Track

| Metric | Source |
|--------|--------|
| Runs, success rate, `max_steps_error` rate | `RunResult.state` |
| Steps per run | `len(RunResult.steps)` |
| Tokens per run (in/out) | `RunResult.token_usage` |
| Latency per run and per step | `RunResult.timing`, `ActionStep.timing` |
| Tool error rate | `ActionStep.error` |
| Final-answer check failures | step errors after `final_answer` |

### Structured Logging

Set `verbosity_level=0` to silence the Rich console output, then log from a step callback:

```python
import json
import logging

from smolagents import ActionStep

log = logging.getLogger("agent")


def log_step(step: ActionStep, agent=None):
    log.info(json.dumps({
        "step": step.step_number,
        "duration_s": step.timing.duration if step.timing else None,
        "tokens": step.token_usage.total_tokens if step.token_usage else None,
        "tools": [tc.name for tc in (step.tool_calls or [])],
        "error": str(step.error) if step.error else None,
    }))
```

### OpenTelemetry tracing

The `telemetry` extra installs OpenInference's smolagents instrumentation, which emits OpenTelemetry spans for runs, steps, model calls and tool calls. Point it at any OTLP backend (Langfuse, Arize Phoenix, Jaeger, Grafana Tempo):

```python
# pip install "smolagents[telemetry]" opentelemetry-exporter-otlp
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")))
SmolagentsInstrumentor().instrument(tracer_provider=provider)
```

---

## Security Best Practices

### Input Validation

- Bound task length and reject binary or obviously malformed input before starting a run.
- Treat everything a tool returns (web pages, emails, files) as untrusted: it can contain prompt-injection text. Keep privileged tools away from agents that read untrusted content, or put a human approval step in front of them.
- Never interpolate secrets into tasks; tools should read credentials from the environment.

### Code Execution Sandboxing

`LocalPythonExecutor` restricts imports and dangerous builtins but runs in your process; it is a guard against accidents, not attackers. For untrusted input:

```python
from smolagents import CodeAgent, InferenceClientModel

agent = CodeAgent(
    tools=[],
    model=InferenceClientModel(),
    executor_type="e2b",                         # or "docker", "modal", "blaxel"
    additional_authorized_imports=["pandas"],
)
try:
    agent.run("Load the CSV at https://example.com/data.csv and report the column means.")
finally:
    agent.cleanup()
```

Notes:

- Remote executors do not support `managed_agents`; for multi-agent systems, run the whole service inside a locked-down container (no credentials beyond what tools need, egress restricted) instead.
- Tools are serialised and sent to the sandbox; give them only the credentials they need.
- With `executor_type="docker"`, the executor builds and starts a Jupyter-kernel container; supply your own image with `executor_kwargs={"image_name": ..., "build_new_image": False}` in production.

### Rate Limiting & Throttling

Limit at the edge (API gateway or middleware) per user, and inside the service per provider:

```python
import threading

PROVIDER_SLOTS = threading.BoundedSemaphore(8)


def run_task_limited(task: str):
    with PROVIDER_SLOTS:
        return run_task(task)
```

---

## Error Handling & Resilience

### Comprehensive Error Handling

Errors inside a step (tool exceptions, code errors, parse failures) are fed back to the model, which usually recovers. What reaches your code:

| Situation | How it surfaces |
|-----------|-----------------|
| Ran out of steps | `RunResult.state == "max_steps_error"`; output is a best-effort answer |
| Provider/API failure | `AgentGenerationError` raised from `run()` |
| Interrupted via `agent.interrupt()` | `AgentError` raised from `run()` |
| Your callback raised | The exception propagates |

```python
import time

from smolagents import AgentError, AgentGenerationError


def resilient_run(task: str, attempts: int = 3):
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            outcome = run_task(task)
            if outcome.ok:
                return outcome
            if attempt == attempts:
                return outcome                   # return the best-effort answer, flagged not ok
        except AgentGenerationError:
            if attempt == attempts:
                raise
            time.sleep(delay)
            delay *= 2
        except AgentError:
            raise                                # interrupted or unrecoverable: do not retry blindly
```

For provider failover, use `LiteLLMRouterModel` with several deployments, or catch `AgentGenerationError` and rebuild the agent with a fallback model.

---

## Deployment Options

### Containerised Deployment

```dockerfile
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
RUN useradd -m -u 1000 app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=app:app . .
USER app
EXPOSE 8000
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
```

`requirements.txt`:

```text
smolagents[toolkit,litellm,e2b,telemetry]==%%version=smolagents%%
fastapi
uvicorn[standard]
```

A minimal FastAPI service around `run_task` (agents are synchronous, so run them in a thread pool):

```python
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

app = FastAPI()


class TaskIn(BaseModel):
    task: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
async def run(body: TaskIn):
    try:
        outcome = await run_in_threadpool(run_task, body.task)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return outcome.__dict__
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: smolagents-api
spec:
  replicas: 3
  selector:
    matchLabels: { app: smolagents-api }
  template:
    metadata:
      labels: { app: smolagents-api }
    spec:
      securityContext: { runAsNonRoot: true, runAsUser: 1000 }
      containers:
        - name: api
          image: registry.example.com/smolagents-api:1.0.0
          ports: [{ containerPort: 8000 }]
          envFrom:
            - secretRef: { name: smolagents-secrets }    # HF_TOKEN, E2B_API_KEY, ...
          resources:
            requests: { cpu: 250m, memory: 512Mi }
            limits: { cpu: "1", memory: 1Gi }
          readinessProbe:
            httpGet: { path: /health, port: 8000 }
          livenessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 20
---
apiVersion: v1
kind: Service
metadata:
  name: smolagents-api
spec:
  selector: { app: smolagents-api }
  ports: [{ port: 80, targetPort: 8000 }]
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: smolagents-api
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: smolagents-api }
  minReplicas: 3
  maxReplicas: 15
  metrics:
    - type: Resource
      resource: { name: cpu, target: { type: Utilization, averageUtilization: 70 } }
```

Running `executor_type="docker"` from inside Kubernetes needs access to a Docker daemon, which is rarely appropriate; prefer E2B, Modal or Blaxel there.

---

## Scaling Strategies

### Horizontal Scaling

Agents are cheap to construct and hold per-run memory, so the simplest scalable design is **one agent per request**, with no shared mutable state in the process. Scale pods on concurrent requests or latency rather than CPU, since runs mostly wait on the model. For long tasks, move runs to a job queue (Celery, RQ, Arq, a Kubernetes `Job`) and return a job ID.

### Vertical Scaling

Only relevant when you host the model yourself (`TransformersModel`, `VLLMModel`): size GPU memory for the model plus KV cache at your target concurrency, and prefer vLLM's continuous batching to many single-request processes.

---

## Testing & Quality Assurance

### Unit Testing Tools

Tools are plain objects; test them directly:

```python
def test_fx_rate():
    assert fx_rate(base="usd", quote="eur") == 0.92
    assert fx_rate.name == "fx_rate"
    assert set(fx_rate.inputs) == {"base", "quote"}
```

Test agent wiring without network calls by subclassing `Model` with a scripted reply:

```python
from smolagents import ChatMessage, CodeAgent, Model
from smolagents.models import MessageRole


class ScriptedModel(Model):
    def __init__(self, replies):
        super().__init__(model_id="scripted")
        self.replies = list(replies)

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        return ChatMessage(role=MessageRole.ASSISTANT, content=self.replies.pop(0))


def test_agent_uses_tool():
    model = ScriptedModel([
        "Thought: convert.\n<code>\nrate = fx_rate(base='USD', quote='EUR')\nfinal_answer(round(100 * rate, 2))\n</code>",
    ])
    agent = CodeAgent(tools=[fx_rate], model=model, verbosity_level=0)
    assert agent.run("Convert 100 USD to EUR") == 92.0
```

For quality, keep a small evaluation set of real tasks with expected answers, run it on every prompt, tool or model change, and track success rate, steps and tokens over time.

---

## Observability Integration (Weights & Biases Weave)

### Weave Integration

Weave ships a smolagents integration that is enabled by `weave.init`; agent runs, steps, model and tool calls are then traced automatically.

```python
# pip install weave
import weave
from smolagents import CodeAgent, InferenceClientModel

weave.init("smolagents-production")

agent = CodeAgent(tools=[], model=InferenceClientModel())
agent.run("What is 2 ** 32?")      # appears as a trace in the Weave UI


@weave.op()
def answer_question(question: str) -> str:
    """Wrap your own entry point to group runs and attach inputs/outputs."""
    return str(agent.run(question))
```

Weave's `Evaluation` API can then score `answer_question` against a dataset; see the Weave documentation for current usage.

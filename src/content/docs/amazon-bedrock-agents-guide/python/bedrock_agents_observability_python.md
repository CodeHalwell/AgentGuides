---
title: "Observability and Monitoring for Amazon Bedrock Agents (Python)"
description: "Monitoring Amazon Bedrock agents: CloudWatch metrics and traces for Bedrock Agents (Classic), and OpenTelemetry-based AgentCore Observability for code-defined agents."
framework: "amazon-bedrock-agents"
language: "python"
sidebar:
  label: "Observability"
  order: 5
---

How to monitor and debug agents on Bedrock. There are two stacks, and they are observed differently:

| Stack | What you get | Where it lands |
|-------|--------------|----------------|
| **Bedrock Agents (Classic)** | Runtime metrics, per-invocation traces (`enableTrace=True`), model-invocation logging | CloudWatch metrics (`AWS/Bedrock/Agents`), your own logs |
| **AgentCore** (Runtime, Memory, Gateway, tools) | OpenTelemetry traces, spans, metrics and logs — **AgentCore Observability** | CloudWatch (GenAI Observability dashboards, Transaction Search), or any OTEL backend |

> Examples use `boto3` %%version=boto3%% and `strands-agents` %%version=strands-agents%% (checked %%versions-checked%%).

## 1. Amazon Bedrock AgentCore Observability

AgentCore Observability gives a unified view of agent sessions, traces and spans for agents running on (or instrumented for) AgentCore.

**Key Features:**

*   **OpenTelemetry (OTEL) Compatibility:** Telemetry is emitted in standard OTEL format, so frameworks that already emit OTEL spans (Strands, LangGraph via OpenInference, CrewAI and others) work without bespoke integration.
*   **CloudWatch Integration:** Traces, metrics and logs appear in CloudWatch, including the GenAI Observability dashboards; enable **CloudWatch Transaction Search** once per account to store spans.
*   **Third-Party Integrations:** Because the data is OTEL, you can also export to platforms such as Datadog, Dynatrace, Langfuse or Arize.

**Instrumenting an agent on AgentCore Runtime.** Add the AWS Distro for OpenTelemetry and start the process under the auto-instrumentation wrapper:

```dockerfile
RUN pip install aws-opentelemetry-distro
CMD ["opentelemetry-instrument", "python", "main.py"]
```

Strands emits spans for each agent invocation, event-loop cycle, model call and tool call, with token usage as span attributes. To add your own attributes to Strands spans:

```python
from strands import Agent

agent = Agent(
    model="us.anthropic.%%model=anthropic%%",
    trace_attributes={"tenant.id": "acme", "app.version": "2.3.1"},
)
```

## 2. Setting up Observability

### a. Enable traces and logging for Bedrock Agents (Classic)

- Pass `enableTrace=True` to `invoke_agent` and log the `trace` events: they show pre-processing, orchestration steps, action-group calls, knowledge-base lookups, collaborator calls and guardrail decisions.
- Turn on **model invocation logging** (Bedrock console → Settings) to capture prompts and completions to CloudWatch Logs or S3.

```python
import json

import boto3

runtime = boto3.client("bedrock-agent-runtime")

response = runtime.invoke_agent(
    agentId="AGENT_ID", agentAliasId="ALIAS_ID", sessionId="s-1",
    inputText="Book a meeting room for tomorrow at 10", enableTrace=True,
)
for event in response["completion"]:
    if "trace" in event:
        print(json.dumps(event["trace"]["trace"], default=str)[:500])
    elif "chunk" in event:
        print(event["chunk"]["bytes"].decode())
```

### b. Use CloudWatch Metrics

Bedrock Agents publishes runtime metrics in the `AWS/Bedrock/Agents` namespace, including:

*   `InvocationCount`
*   `TotalTime` and `TTFT` (time to first token, when streaming)
*   `InvocationThrottles`, `InvocationClientErrors`, `InvocationServerErrors`
*   `ModelLatency`, `ModelInvocationCount`
*   `InputTokenCount`, `OutputTokenCount`

Dimensions are `Operation` (`InvokeAgent`, `InvokeInlineAgent`), `ModelId` and `AgentAliasArn`. AgentCore services publish their own metrics (sessions, latency, errors, CPU and memory for Runtime) under the AgentCore namespaces shown in the CloudWatch console.

### c. Distributed tracing

For Classic agents, enable AWS X-Ray (or ADOT) on the action-group Lambda functions to trace downstream calls. For AgentCore, the OTEL traces above already span the agent and its tools; propagate the trace context to your own services with the standard `traceparent` header (`invoke_agent_runtime` also accepts `traceParent`/`traceId`).

## 3. Advanced Observability Patterns

### a. Custom Metrics

Publish business metrics from action-group Lambda functions or tools:

```python
import boto3

cloudwatch = boto3.client("cloudwatch")


def lambda_handler(event, context):
    # ... your action group logic ...

    cloudwatch.put_metric_data(
        Namespace="BedrockAgents/Bookings",
        MetricData=[
            {
                "MetricName": "SuccessfulBooking",
                "Dimensions": [{"Name": "AgentId", "Value": event["agent"]["id"]}],
                "Value": 1,
                "Unit": "Count",
            },
        ],
    )
```

### b. Structured Logging

Action-group Lambda events carry the agent, session and action details; log them as JSON so CloudWatch Logs Insights can query them:

```python
import json


def lambda_handler(event, context):
    print(json.dumps({
        "agent_id": event["agent"]["id"],
        "agent_alias": event["agent"]["alias"],
        "session_id": event["sessionId"],
        "action_group": event["actionGroup"],
        "api_path": event.get("apiPath") or event.get("function"),
    }))
```

### c. Custom spans with OpenTelemetry

Inside a Strands tool or any code running under ADOT, create spans with the standard OpenTelemetry API; they nest under the agent's tool-call span automatically:

```python
from opentelemetry import trace
from strands import tool

tracer = trace.get_tracer("bookings")


@tool
def book_room(room: str, start: str) -> str:
    """Book a meeting room.

    Args:
        room: Room name.
        start: ISO 8601 start time.
    """
    with tracer.start_as_current_span("bookings.reserve") as span:
        span.set_attribute("booking.room", room)
        return f"Booked {room} at {start}"
```

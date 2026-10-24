---
title: "Strands Agents (Python): Production Guide"
description: "Running Strands agents in production: AgentCore Runtime, Lambda, Fargate and containers; one agent per session; limits, timeouts and retries; tracing; security; and testing with a scripted model instead of the cloud."
framework: "strands"
language: "python"
sidebar:
  label: "Production"
  order: 7
---

Checked against `strands-agents` %%version=strands-agents%% and `bedrock-agentcore` %%version=bedrock-agentcore%% (%%versions-checked%%). The AgentCore, Lambda and FastAPI examples were run locally (AgentCore and FastAPI apps through Starlette's test client) against an offline Bedrock stub; nothing was deployed to AWS.

## Where to run it

A Strands agent is ordinary Python, so it runs anywhere Python does. The choice is about request length, state and how much infrastructure you want to own:

| Host | Fits | Watch out for |
|------|------|---------------|
| **Amazon Bedrock AgentCore Runtime** | Most agents on AWS: long sessions (8 hours maximum lifetime by default), streaming, per-session isolation, built-in identity, memory and observability | AWS-specific; deploy with the AgentCore CLI |
| **AWS Lambda** | Short, stateless requests triggered by API Gateway, SQS or EventBridge | 15-minute maximum, cold starts, 250 MB unzipped package limit, no response streaming with a plain handler |
| **ECS Fargate, EKS, App Runner or any container host** | Existing container platforms, WebSockets, custom networking | You own scaling, session storage and timeouts at the load balancer |
| **Your own server** | Development and internal tools | The same, plus patching |

Whichever you choose, the same rules apply: create an agent per conversation, keep conversation state in a session manager rather than process memory, bound every run, and send traces somewhere.

## One agent per conversation

An `Agent` holds a conversation and runs one call at a time; a second concurrent call raises `ConcurrencyException`. In a server, either build an agent per request and restore its history from a session manager, or keep a small cache of agents keyed by session ID:

```python
from strands import Agent
from strands.session import FileSessionManager

SYSTEM_PROMPT = "You are the support assistant for Example Ltd. Be brief and accurate."


def agent_for(session_id: str) -> Agent:
    """A fresh agent whose history and state are loaded from the session store."""
    return Agent(
        model="us.anthropic.%%model=anthropic%%",
        system_prompt=SYSTEM_PROMPT,
        session_manager=FileSessionManager(session_id=session_id, storage_dir="/tmp/sessions"),
        callback_handler=None,
    )


agent_for("user-42")("Hello, I'm Priya")
print(len(agent_for("user-42").messages))   # 2: restored from storage
```

Building an agent is cheap (no network calls unless it has MCP clients to start), so per-request construction is the simplest correct choice. Use `S3SessionManager`, `SnapshotSessionManager(storage=S3Storage(...))`, AgentCore Memory, or your own `SessionRepository` for anything with more than one instance.

## AgentCore Runtime

AgentCore Runtime hosts an HTTP app built with `BedrockAgentCoreApp` from `bedrock-agentcore`. Each runtime session gets its own isolated microVM, and the session ID arrives on every request:

```python
# main.py
# pip install "bedrock-agentcore==%%version=bedrock-agentcore%%" "strands-agents==%%version=strands-agents%%"
from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent, tool

app = BedrockAgentCoreApp()


@tool
def order_status(order_id: str) -> str:
    """Look up an order's status.

    Args:
        order_id: Order ID, for example ORD-1042.
    """
    return f"{order_id}: shipped"


agents: dict[str, Agent] = {}


def agent_for(session_id: str) -> Agent:
    if session_id not in agents:
        agents[session_id] = Agent(
            model="us.anthropic.%%model=anthropic%%",
            tools=[order_status],
            callback_handler=None,
        )
    return agents[session_id]


@app.entrypoint
async def invoke(payload, context):
    agent = agent_for(context.session_id or "local")
    async for event in agent.stream_async(payload["prompt"]):
        if "data" in event:
            yield event["data"]           # streamed to the caller as server-sent events


if __name__ == "__main__":
    app.run()                             # port 8080: POST /invocations, GET /ping
```

An `async def` generator entrypoint streams; a plain function that returns a dict returns JSON. Locally, `python main.py` serves the same endpoints:

```bash
curl -s localhost:8080/invocations -H 'Content-Type: application/json' -d '{"prompt": "Where is ORD-1042?"}'
```

Deploy with the AgentCore CLI (npm `@aws/agentcore`): `agentcore create --framework Strands ...`, put your entrypoint in the generated `main.py`, then `agentcore dev`, `agentcore deploy` and `agentcore invoke`. The [AgentCore guide](/amazon-bedrock-agents-guide/python/bedrock_agentcore_comprehensive_guide/#deploy-with-the-agentcore-cli) has the full walkthrough, and the [AgentCore production guide](/amazon-bedrock-agents-guide/python/bedrock_agents_production_guide/) covers environments, IAM, VPC and quotas.

Points specific to Strands on Runtime:

- **Memory.** The microVM's memory lasts only as long as the session. For conversations that should survive it, use `AgentCoreMemorySessionManager` ([sessions](/strands-agents-guide/python/strands_comprehensive_guide/#agentcore-memory)).
- **Credentials.** The runtime's execution role reaches Bedrock and AWS services. For third-party API keys (Anthropic, OpenAI, SaaS tools), use AgentCore Identity rather than environment variables.
- **Tracing.** Add `aws-opentelemetry-distro` to the dependencies and run under `opentelemetry-instrument`; Strands spans then appear in CloudWatch GenAI Observability.
- **Long work.** `@app.async_task` marks background work so the `/ping` health status reports `HealthyBusy` and the session is not reclaimed mid-task.

## AWS Lambda

For short requests, a handler is enough. Lambda sends one event at a time to each execution environment, but consecutive events can belong to different users, so build the agent inside the handler and load the conversation from a session manager:

```python
# handler.py
from strands import Agent
from strands.session import S3SessionManager

MODEL_ID = "us.anthropic.%%model=anthropic%%"


def handler(event, context):
    agent = Agent(
        model=MODEL_ID,
        session_manager=S3SessionManager(session_id=event["session_id"], bucket="agent-sessions"),
        callback_handler=None,
    )
    remaining_ms = context.get_remaining_time_in_millis()
    result = agent(event["prompt"], limits={"turns": 10})
    return {
        "answer": str(result).strip(),
        "stop_reason": result.stop_reason,
        "usage": result.metrics.latest_agent_invocation.usage,
        "budget_ms": remaining_ms,
    }
```

Set the function timeout above your longest expected run, and keep the model's read timeout and the turn limit inside it. Package `strands-agents` (and only the extras you use) in the deployment package or a layer; `strands-agents-tools` pulls in many dependencies, so import individual tools only if you need them. For streaming to browsers from Lambda, use a response-streaming function URL with a Node.js handler or move to a container host.

## Containers: FastAPI on Fargate or EKS

A FastAPI app that streams the agent's text, with sessions in S3 so any task can serve any request:

```python
# app.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from strands import Agent
from strands.session import S3SessionManager

app = FastAPI()


class ChatRequest(BaseModel):
    session_id: str
    prompt: str


def agent_for(session_id: str) -> Agent:
    return Agent(
        model="us.anthropic.%%model=anthropic%%",
        session_manager=S3SessionManager(session_id=session_id, bucket="agent-sessions", prefix="chat/"),
        callback_handler=None,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    agent = agent_for(request.session_id)

    async def body():
        async for event in agent.stream_async(request.prompt, limits={"turns": 12}):
            if "data" in event:
                yield event["data"]

    return StreamingResponse(body(), media_type="text/plain")
```

```dockerfile
FROM public.ecr.aws/docker/library/python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

Give the ECS task role `bedrock:InvokeModelWithResponseStream` (and `bedrock:InvokeModel`) on the models and inference profiles you use, plus S3 access to the session bucket. Raise the load balancer's idle timeout above your longest stream. Two requests for the same session at once would each load, then each save, the history; if clients can do that, serialise per session (a lock in a single process, or a conditional write in DynamoDB across instances).

## Limits and timeouts

Bound every run at several levels:

| Level | Setting |
|-------|---------|
| Agent loop | `limits={"turns": ..., "total_tokens": ..., "output_tokens": ...}` per call; `cancel_signal=` for wall-clock deadlines |
| Model output | `max_tokens` on the model (`MaxTokensReachedException` if hit) |
| Model request | `boto_client_config=Config(read_timeout=..., connect_timeout=...)` on `BedrockModel`. Without it, Strands uses a 120-second read timeout; a config you pass replaces that, so set `read_timeout` in it |
| Context size | The conversation manager, `context_window_limit`, and truncating large tool results |
| Multi-agent | `Swarm(max_handoffs=..., max_iterations=..., execution_timeout=..., node_timeout=...)`, `GraphBuilder.set_max_node_executions()`, `set_execution_timeout()`, `set_node_timeout()` |
| Tools | Your own timeouts on HTTP clients and subprocesses; tools can poll `tool_context.cancel_signal` |

```python
from botocore.config import Config
from strands import Agent
from strands.models import BedrockModel

model = BedrockModel(
    model_id="us.anthropic.%%model=anthropic%%",
    max_tokens=4096,
    boto_client_config=Config(
        read_timeout=120,
        connect_timeout=10,
        retries={"max_attempts": 1, "mode": "standard"},   # let Strands own the retries
    ),
)
agent = Agent(model=model, callback_handler=None)
result = agent("Summarise our refund policy", limits={"turns": 6, "total_tokens": 40_000})
```

## Retries

There are two retry layers, and they multiply:

1. **botocore** retries failed HTTP calls according to the client's retry mode (the AWS SDK default unless you pass `boto_client_config`).
2. **Strands** retries throttling (`ModelThrottledException`) with `ModelRetryStrategy`: 6 attempts by default, starting at 4 seconds and doubling to a 240-second cap.

The defaults suit batch work but can hold an interactive request for minutes. For a chat endpoint, shorten them:

```python
from strands import Agent, ModelRetryStrategy

agent = Agent(
    model="us.anthropic.%%model=anthropic%%",
    retry_strategy=ModelRetryStrategy(max_attempts=3, initial_delay=1, max_delay=8),
    callback_handler=None,
)
```

`retry_strategy=None` disables Strands' retries entirely. Fall back to another model or Region with a `ModelRouter` ([routing](/strands-agents-guide/python/strands_comprehensive_guide/#routing-and-fallback)). Tool retries are explicit, through `AfterToolCallEvent.retry` in a hook. Do not wrap `agent(...)` in a generic retry loop: a retried call appends the prompt again, and tools with side effects run twice. Use `idempotency_token=` for duplicate requests from clients.

## Tracing and logs

```python
import logging
import os

from strands.telemetry import StrandsTelemetry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logging.getLogger("strands").setLevel(logging.INFO)

if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
    StrandsTelemetry().setup_otlp_exporter()
```

Add `trace_attributes={"session.id": ..., "user.id": ...}` to each agent so traces can be found per conversation. On AgentCore Runtime, `opentelemetry-instrument` with the AWS distro does the exporter set-up for you. The spans and attributes are listed on the [observability page](/strands-agents-guide/python/strands_observability_and_evals/#the-spans).

Record per-call usage from `result.metrics.latest_agent_invocation.usage` for cost tracking; `accumulated_usage` is the agent's lifetime total.

## Security

- **Least-privilege roles.** The agent can do whatever its tools can do with the process's credentials. Prefer narrow tools (`get_invoice`) over general ones (`use_aws`, `shell`), and scope the IAM role to them.
- **Identity outside the model.** Pass user and tenant IDs in `invocation_state` and enforce them in tools; never let the model supply the ID that authorises an action.
- **Treat tool output as untrusted.** Web pages, emails and documents can contain instructions. Gate sensitive actions with an [intervention handler or interrupt](/strands-agents-guide/python/strands_hooks_and_interrupts/#interventions), not with the system prompt alone.
- **Sandboxes.** Run model-written code in a container sandbox or the AgentCore Code Interpreter, not on the host.
- **Guardrails.** Bedrock Guardrails on `BedrockModel` filter input and output for content policy.
- **Tool consent.** Some `strands-agents-tools` tools prompt on the terminal; in a server set `BYPASS_TOOL_CONSENT=true` deliberately, or choose tools that do not need it.

## Testing without the cloud

Two approaches cover most tests: a scripted model for unit tests, and a patched Bedrock client when you want the real `BedrockModel` code path.

### A scripted model

A custom `Model` that replays canned turns. It is also the shortest complete example of a model provider:

```python
import json
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel
from strands import Agent, tool
from strands.models import Model
from strands.types.content import Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec


class ScriptedModel(Model):
    """Replays canned turns: a str is a text reply, a dict is one tool call."""

    def __init__(self, turns: list[str | dict]):
        self.turns = list(turns)
        self.config: dict[str, Any] = {"model_id": "scripted"}
        self.requests: list[Messages] = []

    def update_config(self, **model_config: Any) -> None:
        self.config.update(model_config)

    def get_config(self) -> dict[str, Any]:
        return self.config

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        self.requests.append(list(messages))
        turn = self.turns.pop(0) if self.turns else "OK"
        yield {"messageStart": {"role": "assistant"}}
        if isinstance(turn, str):
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": turn}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        else:
            tool_use = {"toolUseId": f"t{len(self.requests)}", "name": turn["name"]}
            yield {"contentBlockStart": {"start": {"toolUse": tool_use}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(turn["input"])}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        yield {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {"latencyMs": 0}}}

    async def structured_output(self, output_model: type[BaseModel], prompt: Messages, system_prompt: str | None = None, **kwargs: Any):
        raise NotImplementedError("structured output goes through a tool call; script one instead")
        yield {}


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


def test_agent_uses_the_tool() -> None:
    model = ScriptedModel([{"name": "add", "input": {"a": 2, "b": 3}}, "2 + 3 = 5"])
    agent = Agent(model=model, tools=[add], callback_handler=None)

    result = agent("What is 2 + 3?")

    assert str(result).strip() == "2 + 3 = 5"
    tool_result = agent.messages[2]["content"][0]["toolResult"]
    assert tool_result["content"][0]["text"] == "5"
    assert len(model.requests) == 2


class Answer(BaseModel):
    value: int


def test_structured_output() -> None:
    model = ScriptedModel([{"name": "Answer", "input": {"value": 5}}])   # the output tool is named after the class
    result = Agent(model=model, callback_handler=None)("2 + 3?", structured_output_model=Answer)
    assert result.structured_output == Answer(value=5)


test_agent_uses_the_tool()
test_structured_output()
print("tests passed")
```

Tools, hooks, interventions, interrupts, sessions, swarms and graphs all run for real against it; only the model is fake. Script a `handoff_to_agent` call to test a swarm route, or an `interrupt` path by scripting the tool call that raises it.

### Patching the Bedrock client

To exercise `BedrockModel` itself (request formatting, caching, guardrail handling), replace `converse_stream` on the boto3 client. Strands creates the client in `BedrockModel.__init__`, so patch it on the model:

```python
import json

from strands import Agent
from strands.models import BedrockModel


def fake_converse_stream(**request):
    assert request["modelId"] == "us.anthropic.%%model=anthropic%%"
    text = "Hello from a stub"
    return {"stream": [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": text}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 4, "totalTokens": 7}, "metrics": {"latencyMs": 1}}},
    ]}


model = BedrockModel(model_id="us.anthropic.%%model=anthropic%%", region_name="us-west-2")
model.client.converse_stream = fake_converse_stream
agent = Agent(model=model, callback_handler=None)
print(agent("Hi"))
```

Every Python example in this guide was checked this way, with a patch applied to every `bedrock-runtime` client and `moto` standing in for S3.

### What to test

- Tools as plain functions (they are still callable after `@tool`).
- Routing: given a scripted model, the right tool runs with the right input; given a tool error, the agent recovers.
- Hooks and interventions: blocked calls are blocked, interrupts carry the right reason, resumes complete.
- Sessions: a second agent with the same session ID sees the history.
- Behaviour with real models, on a schedule: [evals](/strands-agents-guide/python/strands_observability_and_evals/#evals-with-strands-agents-evals).

## Checklist

- Model ID set explicitly; default model not relied on.
- One agent per conversation; history in a session manager, not process memory.
- `limits=` on every call; timeouts on the model client, orchestrators and tools.
- Retry layers sized for the use (interactive or batch); no extra wrapper retries.
- `callback_handler=None` in servers.
- Traces exported with session and user attributes; usage recorded per call.
- Tools narrow, credentials in roles or AgentCore Identity, sensitive actions behind interventions.
- Deprecated `strands-agents-tools` tools replaced before 0.9.
- Unit tests on a scripted model; evals with real models on a schedule.

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-24 | 1.62.0 | New page, expanding the deployment section of the former Bedrock Strands page: hosting options, per-session agents, AgentCore Runtime, Lambda and FastAPI examples, limits, the two retry layers, tracing, security and testing with a scripted model or a patched Bedrock client. |

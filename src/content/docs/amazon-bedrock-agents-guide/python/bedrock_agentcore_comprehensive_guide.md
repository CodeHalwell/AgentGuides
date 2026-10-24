---
title: "Amazon Bedrock AgentCore: Comprehensive Technical Guide"
description: "Technical reference for Amazon Bedrock AgentCore: Runtime, Memory, Gateway, Identity, Code Interpreter, Browser and Observability, with examples in the bedrock-agentcore Python SDK and boto3."
framework: "amazon-bedrock-agents"
language: "python"
sidebar:
  label: "AgentCore"
  order: 2
---

> TypeScript version of this: [Bedrock AgentCore TypeScript SDK](/amazon-bedrock-agents-guide/typescript/agentcore_typescript_guide/).

A technical reference for Amazon Bedrock AgentCore, AWS's platform for deploying and operating agents built with any framework and any model. AgentCore is a set of modular services — Runtime, Memory, Gateway, Identity, Code Interpreter, Browser, Observability, and the newer managed Harness — that you can use together or on their own.

> **Verified against:** `bedrock-agentcore` %%version=bedrock-agentcore%% and `boto3` %%version=boto3%% (checked %%versions-checked%%). Every class, function and boto3 operation shown here was checked against those installed packages.

**History:** AgentCore was announced in preview at AWS Summit New York in July 2025 and became generally available in October 2025. Preview-period pricing (free until 16 September 2025) no longer applies; see the [AgentCore pricing page](https://aws.amazon.com/bedrock/agentcore/pricing/).

**Relationship to Bedrock Agents:** Amazon Bedrock Agents (now "Agents Classic") stopped accepting new customers on 30 July 2026 and is in maintenance mode. AWS recommends AgentCore — either the managed harness or code-defined agents on Runtime — for new work and as the migration target. See [Bedrock Agents Classic maintenance mode](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-classic-maintenance-mode.html).

---

## Table of Contents

1. [Introduction to AgentCore](#introduction-to-agentcore)
2. [Core Features](#core-features)
3. [AgentCore Architecture](#agentcore-architecture)
4. [Getting Started](#getting-started)
5. [Runtime Application Features](#runtime-application-features)
6. [Managed Harness](#managed-harness)
7. [Secure Sandbox Runtime](#secure-sandbox-runtime)
8. [Supported Frameworks](#supported-frameworks)
9. [Code Interpreter, Browser and Memory](#code-interpreter-browser-and-memory)
10. [AgentCore APIs](#agentcore-apis)
11. [Integration Patterns](#integration-patterns)
12. [Migrating from Bedrock Agents Classic](#migrating-from-bedrock-agents-classic)
13. [Security and Compliance](#security-and-compliance)
14. [Best Practices](#best-practices)

---

## Introduction to AgentCore

AgentCore separates *what your agent does* (your code, in any framework) from *where and how it runs* (managed, session-isolated infrastructure with identity, memory, tools and telemetry). You keep your Strands, LangGraph, CrewAI, LlamaIndex, Google ADK or OpenAI Agents SDK code; AgentCore supplies the operational parts.

### Key Benefits

| Benefit | Description |
|---------|-------------|
| **Session isolation** | Each Runtime session runs in its own microVM; state does not leak between sessions or users |
| **Framework- and model-agnostic** | Any Python or Node.js framework; any model in or outside Bedrock |
| **Serverless** | No clusters to manage; pay for what sessions consume |
| **Protocol support** | HTTP, MCP, A2A and AG-UI server protocols on Runtime |
| **Composable services** | Use Memory, Gateway or Code Interpreter without Runtime, or vice versa |

### Use Cases

- **Hosting existing agents**: wrap an existing agent in a small entrypoint and deploy it to Runtime
- **Tool access**: expose REST APIs, Lambda functions or existing MCP servers to agents through Gateway
- **Long-running and asynchronous work**: sessions can run for hours, with background tasks reported to the health check
- **Data analysis**: run model-generated code in the Code Interpreter sandbox rather than in your own process

---

## Core Features

| Service | What it does | Python SDK entry point | boto3 client |
|---------|--------------|------------------------|--------------|
| **Runtime** | Hosts your agent or MCP/A2A server in session-isolated microVMs | `bedrock_agentcore.BedrockAgentCoreApp` | `bedrock-agentcore-control` (`create_agent_runtime`), `bedrock-agentcore` (`invoke_agent_runtime`) |
| **Harness** | Managed agent loop declared as configuration (model, prompt, tools) | — | `bedrock-agentcore-control` (`create_harness`), `bedrock-agentcore` (`invoke_harness`) |
| **Memory** | Short-term events and long-term extracted records | `bedrock_agentcore.memory.MemorySessionManager` (`MemoryClient` is the legacy interface) | `bedrock-agentcore-control` (`create_memory`), `bedrock-agentcore` (`create_event`, `retrieve_memory_records`) |
| **Gateway** | Turns APIs, Lambda and MCP servers into one MCP endpoint | `bedrock_agentcore.gateway.GatewayClient` (create and manage gateways and targets) | `bedrock-agentcore-control` (`create_gateway`, `create_gateway_target`) |
| **Identity** | Inbound auth (JWT/IAM) and outbound OAuth / API-key credentials | `bedrock_agentcore.identity.auth.requires_access_token` | `bedrock-agentcore-control` (credential providers) |
| **Code Interpreter** | Sandboxed Python/JavaScript/TypeScript execution | `bedrock_agentcore.tools.code_interpreter_client.code_session` | `bedrock-agentcore` (`invoke_code_interpreter`) |
| **Browser** | Managed headless browser for web automation | `bedrock_agentcore.tools.browser_client.browser_session` | `bedrock-agentcore` (`start_browser_session`) |
| **Observability** | OpenTelemetry traces, metrics and logs in CloudWatch | via OTEL / ADOT instrumentation | CloudWatch |

AgentCore also includes Policy, Evaluations and Payments services. The SDK has clients for them (`bedrock_agentcore.policy`, `.evaluation`, `.payments`), and evaluations are covered in the [production guide](/amazon-bedrock-agents-guide/python/bedrock_agents_production_guide/#evaluations); Policy and Payments are outside the scope of this guide.

---

## AgentCore Architecture

### System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│          Callers: your app, another agent, MCP/A2A client       │
└──────────────────────────┬─────────────────────────────────────┘
                           │  InvokeAgentRuntime (SigV4 or JWT)
┌──────────────────────────▼─────────────────────────────────────┐
│                   AgentCore Runtime (data plane)                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Session microVM  (one per runtimeSessionId)              │  │
│  │  Your container or code package:                          │  │
│  │    BedrockAgentCoreApp  →  /invocations, /ping (port 8080) │  │
│  │    Strands / LangGraph / CrewAI / custom agent            │  │
│  └───────────────┬───────────────┬───────────────┬──────────┘  │
└──────────────────┼───────────────┼───────────────┼─────────────┘
                   │               │               │
          ┌────────▼───┐   ┌───────▼─────┐  ┌──────▼────────┐
          │  Memory    │   │  Gateway    │  │ Code Interp.  │
          │ (events,   │   │ (MCP tools: │  │ / Browser     │
          │  records)  │   │ Lambda, API)│  │ sandboxes     │
          └────────────┘   └─────────────┘  └───────────────┘
                   │               │
          ┌────────▼───────────────▼──────────────┐
          │ Identity (inbound JWT/IAM, outbound   │
          │ OAuth / API keys) · Observability     │
          └───────────────────────────────────────┘
```

### Component Overview

**Control plane** (`bedrock-agentcore-control`): create, update and delete runtimes, runtime endpoints, memories, gateways and targets, code interpreters, browsers, credential providers and workload identities.

**Data plane** (`bedrock-agentcore`): invoke runtimes and harnesses, write and read memory, start and use code-interpreter and browser sessions, and fetch workload access tokens.

**Runtime contract:** a Runtime container listens on port 8080 and serves `POST /invocations` and `GET /ping`. `BedrockAgentCoreApp` implements that contract for you; MCP servers serve `/mcp`, and A2A servers follow the A2A protocol contract.

---

## Getting Started

### Prerequisites

- An AWS account with access to the Bedrock models you intend to use
- Python 3.10+ (required by `bedrock-agentcore`, `strands-agents` and `boto3`)
- AWS credentials configured locally
- Node.js 20.12+ for the AgentCore CLI, and [`uv`](https://docs.astral.sh/uv/) for the Python projects it generates

### Installation

```bash
pip install "bedrock-agentcore==%%version=bedrock-agentcore%%" "strands-agents==%%version=strands-agents%%" "boto3==%%version=boto3%%"

# AgentCore CLI (%%version=@aws/agentcore%%): scaffolds, runs locally, deploys with CDK
npm install -g @aws/agentcore
```

If you installed the older Python `bedrock-agentcore-starter-toolkit`, uninstall it first. Both packages provide an `agentcore` command.

### Quick Start Example

`main.py` is a Strands agent wrapped for Runtime:

```python
from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent

app = BedrockAgentCoreApp()

agent = Agent(
    model="us.anthropic.%%model=anthropic%%",  # Bedrock cross-Region inference profile
    system_prompt="You are a concise data analysis assistant.",
)


@app.entrypoint
def invoke(payload, context):
    """Called for every POST /invocations. `context` carries the session ID and headers."""
    prompt = payload.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    result = agent(prompt)
    return {"result": str(result), "session_id": context.session_id}


if __name__ == "__main__":
    app.run()  # serves on port 8080 locally, as Runtime expects
```

Test it locally:

```bash
python main.py &
curl -s -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Summarise the main drivers of churn in one paragraph."}'
```

### Deploy with the AgentCore CLI

The CLI keeps a project in two parts. `agentcore/agentcore.json` declares the resources (runtimes, harnesses, memories, gateways, credentials, evaluators) and `agentcore/cdk/` is the generated CDK app that deploys them. Deployment targets (an account and Region each) live in `agentcore/aws-targets.json`. On the first `agentcore deploy` the CLI creates a `default` target from your current AWS account and Region.

There are two ways to start. A **template** scaffolds a working agent:

```bash
agentcore create --name SalesAnalyst --template agent-python-strands
cd SalesAnalyst
```

`agentcore create --help` lists the templates: `agent-python-strands` (and a `-container` variant), `agent-python-langchain`, `agent-python-minimal`, `agent-typescript-strands`, `agent-typescript-vercel`, `a2a-python-strands`, `agui-python-strands`, `mcp-python-fastmcp` and `empty`. Without `--template` you get an empty project, which is where managed harnesses (below) start. The Strands templates also declare an AgentCore Memory resource, `agentMemory`, and the generated code reads its ID from `MEMORY_AGENTMEMORY_ID`, which the CLI injects on deploy.

To **bring your own code**, such as the entrypoint above, add a runtime to a project:

```bash
agentcore create --name SalesAnalyst
cd SalesAnalyst
agentcore add runtime --name analyst --language python
```

`add runtime` writes no code. It creates `app/analyst/` with a `README.md` and an `AGENTS.md` describing what the folder must contain, and registers the runtime in `agentcore.json` as a CodeZip build. You supply `main.py` (the entrypoint above) and a `pyproject.toml` that lists `bedrock-agentcore` and `aws-opentelemetry-distro` in `[project].dependencies` and sets `requires-python`. Dependencies must publish `aarch64` wheels, because Runtime runs on ARM64. Adding a `Dockerfile` to the folder switches the runtime to a container build, and `--protocol` selects `HTTP` (the default), `MCP`, `A2A` or `AGUI`. `agentcore dev`, `agentcore build` and `agentcore deploy` read your code and update `agentcore.json` to match, printing what changed.

Run, deploy and invoke:

```bash
agentcore dev                                          # run the runtime locally; --mode browser opens the Agent Inspector
agentcore invoke --local --runtime analyst --payload '{"prompt": "Hello"}'   # call the local server
agentcore deploy                                       # CDK deploy to the target in agentcore/aws-targets.json
agentcore invoke --runtime analyst \
  --session-id "$(uuidgen)-q4-analysis" \
  --payload '{"prompt": "Analyse monthly sales trends for Q4"}'
```

`--session-id` must be at least 33 characters. See [Environments and CI/CD](/amazon-bedrock-agents-guide/python/bedrock_agents_production_guide/#environments-and-cicd) for several targets and pipelines. `agentcore add memory|gateway|credentials|runtime-endpoint|evaluator|online-eval` adds more resources to an existing project, `agentcore status` shows what is deployed, and `agentcore migrate` converts a project created by a 0.x CLI to the current format while keeping its deployed resources.

Invoke the deployed runtime from any Python code with boto3:

```python
import json
import uuid

import boto3

client = boto3.client("bedrock-agentcore", region_name="us-west-2")

response = client.invoke_agent_runtime(
    agentRuntimeArn="arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/SalesAnalyst_analyst-abc123",
    runtimeSessionId=str(uuid.uuid4()) + "-session",  # must be at least 33 characters
    payload=json.dumps({"prompt": "Analyse monthly sales trends for Q4"}),
    qualifier="DEFAULT",
)

print(json.loads(response["response"].read()))
```

Reuse the same `runtimeSessionId` to keep talking to the same session microVM (and its in-memory state); use a new one for a fresh, isolated session. Passing `runtimeUserId` identifies the end user to the runtime. It requires the extra `bedrock-agentcore:InvokeAgentRuntimeForUser` permission alongside `InvokeAgentRuntime`.

---

## Runtime Application Features

`BedrockAgentCoreApp` is a Starlette application. It serves `POST /invocations` and `GET /ping` for you, and a few behaviours are worth knowing before you build on it:

- **Sync and async entrypoints.** A plain `def` entrypoint runs in a thread pool, so blocking calls do not stall `/ping`. An `async def` entrypoint runs on the event loop.
- **Streaming.** If the entrypoint is a generator or async generator, each yielded value is sent as a Server-Sent Events frame (`data: <json>`), and the caller receives `text/event-stream`.
- **Errors.** An exception in the entrypoint is logged with its stack trace as structured JSON and returned as HTTP 500 with `{"error": "<message>"}`. Validate input and raise explicit errors rather than letting arbitrary exceptions leak details.
- **Context.** The second argument is a `RequestContext` with `session_id` (from the `X-Amzn-Bedrock-AgentCore-Runtime-Session-Id` header) and `request_headers` (lower-cased names). Custom headers arrive only when named `X-Amzn-Bedrock-AgentCore-Runtime-Custom-*` and listed in the runtime's request-header allowlist. Declare the entrypoint with a single `payload` parameter if you do not need the context.
- **Health.** `/ping` reports `Healthy` or `HealthyBusy`. Runtime reclaims sessions it considers idle, so work that outlives a request must mark the session busy.

```python
import asyncio

from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.runtime import PingStatus

app = BedrockAgentCoreApp()


@app.async_task
async def rebuild_report(report_id: str) -> None:
    """While this coroutine runs, /ping reports HealthyBusy and the session is kept alive."""
    await asyncio.sleep(120)  # stand-in for real work


@app.entrypoint
async def invoke(payload, context):
    actor = (context.request_headers or {}).get("x-amzn-bedrock-agentcore-runtime-custom-actor-id", "anonymous")
    if payload.get("action") == "rebuild":
        asyncio.create_task(rebuild_report(payload["report_id"]))
        yield {"status": "started", "actor": actor}
        return
    for word in str(payload.get("prompt", "")).split():
        yield {"token": word}


if __name__ == "__main__":
    app.run()
```

For work that is not a single coroutine, call `app.add_async_task("name", {"detail": 1})` to get a task ID and `app.complete_async_task(task_id)` when it finishes. `app.get_async_task_info()` returns the active count and running jobs. To take full control of the status, decorate a function with `@app.ping` that returns `PingStatus.HEALTHY` or `PingStatus.HEALTHY_BUSY`.

Cross-cutting concerns such as authentication checks, payload limits and access logs belong in middleware. See [Middleware](/amazon-bedrock-agents-guide/python/bedrock_agents_middleware_python/#4-middleware-in-the-agentcore-runtime-app), which uses `BedrockAgentCoreApp(middleware=[...])`.

---
## Managed Harness

A harness is an AgentCore-managed agent loop declared as configuration: model, system prompt, tools, skills, memory, limits and lifecycle hooks. AgentCore runs it on Runtime, in the same session-isolated microVMs as code-defined agents, so there is no agent code to build or deploy. AWS recommends it as the starting point for new agents and as the first migration target for Agents Classic. Use a code-defined agent on Runtime instead when you need custom orchestration, multi-agent collaboration, stage-specific prompt overrides or a framework you already use.

Every harness session also has built-in `shell` and `file_operations` tools. They run inside the session's microVM, and `allowedTools` (default `["*"]`) controls which tools the model may choose. Tool definitions count towards input tokens even when the model never calls them, so restrict `allowedTools` to what the agent needs.

### With the CLI

```bash
agentcore create --name ResearchProject
cd ResearchProject
agentcore add harness --name ResearchAgent \
  --model '{"provider":"bedrock","modelId":"us.anthropic.%%model=anthropic%%"}' \
  --system-prompt "You are a careful research assistant. Cite your sources." \
  --tools '[{"type":"agentcore_code_interpreter","name":"code-interpreter"}]' \
  --max-iterations 20 --timeout-seconds 600
agentcore deploy
agentcore invoke --harness ResearchAgent --session-id "$(uuidgen)" \
  --prompt "Compare three CRM vendors for a 40-person team"
```

`add harness` writes `app/ResearchAgent/harness.yaml` and `system-prompt.md` and registers the harness in `agentcore.json`. Run it with no flags and only `--name` to get the defaults (Claude on Bedrock, managed memory, sliding-window truncation). The YAML mirrors the `CreateHarness` API, so every field below can also be edited by hand before the next `agentcore deploy`:

```yaml
name: ResearchAgent
model:
  bedrockModelConfig:
    modelId: us.anthropic.%%model=anthropic%%
tools:
  - type: agentcore_code_interpreter
    name: code-interpreter
  - type: remote_mcp
    name: docs
    config:
      remoteMcp:
        url: https://mcp.example.com/mcp
allowedTools: ["*"]
memory:
  managedMemoryConfiguration: {}      # or memory: { disabled: {} }
truncation:
  strategy: sliding_window
  config:
    slidingWindow:
      messagesCount: 40
maxIterations: 20
timeoutSeconds: 600
environmentVariables: {}
```

The system prompt lives in `system-prompt.md` next to the file (or inline under `systemPrompt`). The generated file also documents skills, VPC networking, EFS and S3 file mounts, container environments and JWT authorizers as commented examples. `agentcore dev` deploys the harness and prints how to invoke it, and `--skip-deploy` reuses a deployed one.

### With boto3

```python
import time
import uuid

import boto3

control = boto3.client("bedrock-agentcore-control", region_name="us-west-2")
data = boto3.client("bedrock-agentcore", region_name="us-west-2")

harness = control.create_harness(
    harnessName="ResearchAgent",
    executionRoleArn="arn:aws:iam::111122223333:role/ResearchHarnessRole",
    model={"bedrockModelConfig": {"modelId": "us.anthropic.%%model=anthropic%%", "maxTokens": 4096}},
    systemPrompt=[{"text": "You are a careful research assistant. Cite your sources."}],
    tools=[
        {"type": "agentcore_code_interpreter", "name": "code-interpreter"},
        {
            "type": "agentcore_gateway",
            "name": "orders",
            "config": {"agentCoreGateway": {"gatewayArn": "arn:aws:bedrock-agentcore:us-west-2:111122223333:gateway/orders-gw"}},
        },
    ],
    truncation={"strategy": "sliding_window", "config": {"slidingWindow": {"messagesCount": 40}}},
    maxIterations=20,
    timeoutSeconds=600,
)["harness"]

# Wait until the harness (and the Runtime behind it) is ready
while True:
    current = control.get_harness(harnessId=harness["harnessId"])["harness"]
    if current["status"] == "READY":
        break
    if current["status"].endswith("FAILED"):
        raise RuntimeError(current.get("failureReason"))
    time.sleep(10)

response = data.invoke_harness(
    harnessArn=harness["arn"],
    runtimeSessionId=str(uuid.uuid4()),  # 33 to 100 characters for a harness; reuse it to continue the session
    messages=[{"role": "user", "content": [{"text": "Compare three CRM vendors for a 40-person team"}]}],
)

for event in response["stream"]:
    if "contentBlockDelta" in event:
        text = event["contentBlockDelta"]["delta"].get("text")
        if text:
            print(text, end="", flush=True)
    elif "messageStop" in event:
        stop = event["messageStop"]["stopReason"]  # "end_turn", "tool_use", or the reason a limit stopped the run
    elif "runtimeClientError" in event:
        raise RuntimeError(event["runtimeClientError"].get("message"))
```

The stream carries `messageStart`, `contentBlockStart`, `contentBlockDelta`, `contentBlockStop`, `messageStop` and `metadata` (token usage and latency) events, plus `internalServerException`, `validationException`, `runtimeClientError` and `hookEvent`. A `contentBlockDelta` holds `text`, `toolUse`, `toolResult`, `reasoningContent` or `toolResultMetadata`. `invoke_harness` also accepts per-call overrides of `model`, `systemPrompt`, `tools`, `skills`, `allowedTools`, `maxIterations`, `maxTokens` and `timeoutSeconds`, and an `actorId` that scopes memory.

The execution role trusts `bedrock-agentcore.amazonaws.com` and needs model invocation, logs, X-Ray and access to any tools it uses. The [IAM page](/amazon-bedrock-agents-guide/python/bedrock_security_iam_examples/#agentcore-harness) has a scoped version. Callers need `bedrock-agentcore:InvokeHarness` **and** `bedrock-agentcore:InvokeAgentRuntime` on the harness ARN.

**Limits.** `maxIterations` (reasoning and action cycles per invocation, default 75), `timeoutSeconds` (wall-clock time per invocation, default 3600) and `maxTokens` (an output budget, unset by default) stop a runaway loop. `truncation.strategy` is `sliding_window`, `summarization` or `none`. `idleRuntimeSessionTimeout` and `maxLifetime` (defaults 900 and 28,800 seconds) bound the microVM and are set under `environment.agentCoreRuntimeEnvironment.lifecycleConfiguration` in the YAML, or with `--lifecycle-config` on `add harness`. Harness invocations are also subject to Runtime service quotas.

**Models.** The harness works with Bedrock models and, through other `model` configurations, with OpenAI, Gemini and LiteLLM-compatible providers. The CLI's `--model` takes `{"provider": "bedrock" | "open_ai" | "gemini" | "lite_llm", "modelId": ...}` and `--api-key` supplies a non-Bedrock key from `-` (stdin) or `file://path`.

**Tool types:** `agentcore_code_interpreter`, `agentcore_browser`, `agentcore_gateway` (every tool on the gateway becomes available to the model; `outboundAuth` selects IAM, OAuth or none), `remote_mcp` (any streamable-HTTP MCP server, with optional `headers`) and `inline_function`.

### Inline functions: running a tool in your own code

An inline function is the harness's return of control. You declare the schema; when the model calls it, the harness pauses, the stream ends with `stopReason` `tool_use`, and your code runs the function and sends back a `toolResult` on the same `runtimeSessionId`. The harness keeps the model's tool call in the session, so the follow-up request carries only the result. Use it for human approval, calls into internal systems that the harness cannot reach, or logic you want to control client-side.

```python
import json
import uuid

import boto3

data = boto3.client("bedrock-agentcore", region_name="us-west-2")

HARNESS_ARN = "arn:aws:bedrock-agentcore:us-west-2:111122223333:harness/ResearchAgent-abc123"

TOOLS = [
    {
        "type": "inline_function",
        "name": "get_weather",
        "config": {
            "inlineFunction": {
                "description": "Get the current weather for a city.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        },
    }
]


def get_weather(city: str) -> str:
    return f"18°C and cloudy in {city}"


LOCAL_FUNCTIONS = {"get_weather": get_weather}


def read_stream(response):
    """Collect the text, the tool calls and the stop reason from one harness stream."""
    text, tool_calls, current, stop_reason = "", [], None, None
    for event in response["stream"]:
        if "contentBlockStart" in event:
            start = event["contentBlockStart"]["start"]
            if "toolUse" in start:
                current = {
                    "index": event["contentBlockStart"]["contentBlockIndex"],
                    "toolUseId": start["toolUse"]["toolUseId"],
                    "name": start["toolUse"]["name"],
                    "input": "",
                }
        elif "contentBlockDelta" in event:
            block = event["contentBlockDelta"]
            delta = block["delta"]
            if "text" in delta:
                text += delta["text"]
            elif "toolUse" in delta and current and block["contentBlockIndex"] == current["index"]:
                current["input"] += delta["toolUse"].get("input", "")  # arrives as JSON fragments
        elif "contentBlockStop" in event and current:
            tool_calls.append(current)
            current = None
        elif "messageStop" in event:
            stop_reason = event["messageStop"]["stopReason"]
        elif "runtimeClientError" in event:
            raise RuntimeError(event["runtimeClientError"].get("message"))
    return text, tool_calls, stop_reason


def ask(prompt: str, session_id: str) -> str:
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    answer = ""
    while True:
        response = data.invoke_harness(
            harnessArn=HARNESS_ARN,
            runtimeSessionId=session_id,  # the same ID for every call in the conversation
            tools=TOOLS,                  # needed here because the tool is a per-call override
            messages=messages,
        )
        text, tool_calls, stop_reason = read_stream(response)
        answer += text
        if stop_reason != "tool_use":
            return answer
        results = []
        for call in tool_calls:
            output = LOCAL_FUNCTIONS[call["name"]](**json.loads(call["input"] or "{}"))
            results.append({"toolResult": {"toolUseId": call["toolUseId"], "content": [{"text": output}], "status": "success"}})
        messages = [{"role": "user", "content": results}]  # only the results; the harness keeps the call


print(ask("What is the weather in Leeds?", str(uuid.uuid4())))
```

If the inline function is part of the deployed harness (in `harness.yaml` or `create_harness`), leave `tools` out of the call. Return results for adjacent inline calls from one model response together, in one request. `status` can be `"error"` to tell the model the call failed. A missing, duplicated or stale `toolUseId` cannot resume the session.

**Security notes.** Anyone who can call `invoke_harness` can also override the model, system prompt, tools, skills and limits for that call. If you forward end-user input, strip or allowlist those fields in your own application layer. With SigV4 callers, the harness does not pass per-user identity to tools. Configure a JWT authorizer (`authorizerConfiguration.customJWTAuthorizer`) when tools need user-scoped OAuth tokens. `bedrock-agentcore:InvokeAgentRuntimeCommand` runs a shell command in a runtime directly, without the model and without `allowedTools`, so do not grant it to callers who should only chat.

### Exporting a harness to code

When the harness stops being enough, `agentcore export harness` generates an equivalent Strands agent as a new runtime in the same project. It carries over the system prompt, the tools (including re-implemented `shell` and `file_operations`), gateway connections and inline functions. Managed memory is not exported: the generated `EXPORT_NOTES.md` lists what needs manual follow-up, and you re-run the export after `agentcore add memory`.

```bash
agentcore export harness --name ResearchAgent    # writes app/ResearchAgentAgent/ and EXPORT_NOTES.md
agentcore deploy
```

Use `--arn` to export a harness created outside the project (with `--project-name` to create a project for it) and `--target-agent-name` to choose the name of the new runtime.

---

## Secure Sandbox Runtime

### Runtime Isolation

Each `runtimeSessionId` maps to a dedicated microVM with its own CPU, memory and filesystem. When the session ends — idle timeout or maximum lifetime — the microVM is torn down and its memory sanitised. Isolation is configured on the runtime resource rather than in your code:

```python
import boto3

control = boto3.client("bedrock-agentcore-control", region_name="us-west-2")

response = control.create_agent_runtime(
    agentRuntimeName="secure_analysis_agent",
    agentRuntimeArtifact={
        "containerConfiguration": {
            "containerUri": "111122223333.dkr.ecr.us-west-2.amazonaws.com/analysis-agent:latest"
        }
    },
    roleArn="arn:aws:iam::111122223333:role/AgentRuntimeRole",
    networkConfiguration={
        "networkMode": "VPC",  # or "PUBLIC"
        "networkModeConfig": {
            "subnets": ["subnet-0abc", "subnet-0def"],
            "securityGroups": ["sg-0123"],
        },
    },
    lifecycleConfiguration={
        "idleRuntimeSessionTimeout": 900,  # seconds
        "maxLifetime": 28800,              # seconds
    },
    environmentVariables={"LOG_LEVEL": "INFO"},
)

print(response["agentRuntimeArn"], response["status"])
```

### Language Runtime Support

Two artefact types are supported:

| Artefact | How | Notes |
|----------|-----|-------|
| **Container** | `containerConfiguration.containerUri` (ECR image) | Build for `linux/arm64`; any language that can serve the HTTP contract |
| **Direct code deployment** | `codeConfiguration` with `code.s3` (`bucket`, `prefix`), `runtime` and `entryPoint` | `runtime` is one of `PYTHON_3_10` … `PYTHON_3_14` or `NODE_22` |

```python
control.create_agent_runtime(
    agentRuntimeName="zip_deployed_agent",
    agentRuntimeArtifact={
        "codeConfiguration": {
            "code": {"s3": {"bucket": "my-agent-artifacts", "prefix": "agents/sales/deployment.zip"}},
            "runtime": "PYTHON_3_13",
            "entryPoint": ["main.py"],
        }
    },
    roleArn="arn:aws:iam::111122223333:role/AgentRuntimeRole",
    networkConfiguration={"networkMode": "PUBLIC"},
)
```

### Resource Management

There is no in-SDK resource monitor. Use:

- **Lifecycle settings** (`idleRuntimeSessionTimeout`, `maxLifetime`) to bound session cost.
- **The ping status** to tell Runtime whether a session is busy with background work, so it is not reclaimed while idle-looking (see [Runtime Application Features](#runtime-application-features)):

```python
import asyncio

from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()


@app.async_task
async def rebuild_report(report_id: str):
    """While this runs, /ping reports HealthyBusy, so the session is kept alive."""
    ...


@app.entrypoint
async def invoke(payload, context):
    # Start the work in the background and return immediately
    asyncio.create_task(rebuild_report(payload["report_id"]))
    return {"status": "started"}
```

- **CloudWatch metrics** emitted by AgentCore Observability for session counts, latency, errors and resource usage.

---

## Supported Frameworks

Runtime does not need framework-specific adapters: any code that can be called from the `@app.entrypoint` function works. The patterns below differ only in what happens inside the entrypoint.

### 1. Strands Agents SDK

**Strands on AgentCore.** The SDK itself (tools, model providers, sessions, hooks, multi-agent) is covered in the standalone [Strands Agents guide](/strands-agents-guide/python/); its [production page](/strands-agents-guide/python/strands_production_guide/#agentcore-runtime) covers one agent per session, limits, retries and tracing on Runtime. Here is the shape of a Strands entrypoint. Streaming is supported by making the entrypoint an async generator:

```python
from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent, tool

app = BedrockAgentCoreApp()


@tool
def total_revenue(region: str) -> float:
    """Return total revenue for a sales region.

    Args:
        region: Sales region name, for example "EMEA".
    """
    return {"EMEA": 1_250_000.0, "AMER": 2_100_000.0}.get(region, 0.0)


agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[total_revenue])


@app.entrypoint
async def invoke(payload, context):
    async for event in agent.stream_async(payload["prompt"]):
        if "data" in event:          # text deltas
            yield event["data"]


if __name__ == "__main__":
    app.run()
```

### 2. LangChain / LangGraph

```python
from bedrock_agentcore import BedrockAgentCoreApp
from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse

app = BedrockAgentCoreApp()

llm = ChatBedrockConverse(model="us.anthropic.%%model=anthropic%%")
graph = create_agent(llm, tools=[])  # a compiled LangGraph graph; add your tools here


@app.entrypoint
def invoke(payload, context):
    result = graph.invoke({"messages": [{"role": "user", "content": payload["prompt"]}]})
    return {"result": result["messages"][-1].content}


if __name__ == "__main__":
    app.run()
```

### 3. CrewAI

```python
from bedrock_agentcore import BedrockAgentCoreApp
from crewai import Agent, Crew, Task

app = BedrockAgentCoreApp()

researcher = Agent(
    role="Researcher",
    goal="Find relevant market data",
    backstory="Expert market analyst",
    llm="bedrock/us.anthropic.%%model=anthropic%%",
)
research_task = Task(
    description="Research trends for: {topic}",
    expected_output="Five bullet points",
    agent=researcher,
)
crew = Crew(agents=[researcher], tasks=[research_task])


@app.entrypoint
def invoke(payload, context):
    return {"result": str(crew.kickoff(inputs={"topic": payload["prompt"]}))}


if __name__ == "__main__":
    app.run()
```

### 4. MCP and A2A servers

Set `protocolConfiguration={"serverProtocol": "MCP"}` (or `"A2A"`, `"AGUI"`) on `create_agent_runtime` to host a protocol server instead of an HTTP agent. For A2A, the SDK provides `serve_a2a`:

```python
# pip install "bedrock-agentcore[a2a]"
from bedrock_agentcore.runtime import serve_a2a
from strands import Agent
from strands.multiagent.a2a import StrandsA2AExecutor

agent = Agent(model="us.anthropic.%%model=anthropic%%", system_prompt="You are a helpful assistant.")
serve_a2a(StrandsA2AExecutor(agent))
```

---

## Code Interpreter, Browser and Memory

### Code Interpretation

The Code Interpreter runs code in a managed sandbox, not in your agent's process. `code_session` starts a session on the AWS-managed `aws.codeinterpreter.v1` interpreter and stops it on exit. Python state (variables, imports) persists between `execute_code` calls in the same session, so an agent can work on a dataset step by step. The methods return the service's response, in which the output arrives as a stream of events, so a small helper turns it into text:

```python
from bedrock_agentcore.tools.code_interpreter_client import code_session


def output_of(result) -> str:
    """Join stdout and stderr from a code interpreter response stream."""
    parts = []
    for event in result["stream"]:
        outcome = event.get("result", {})
        details = outcome.get("structuredContent", {})
        parts += [details.get("stdout", ""), details.get("stderr", "")]
        if outcome.get("isError"):
            raise RuntimeError("".join(parts) or "code execution failed")
    return "".join(parts)


with code_session("us-west-2") as ci:
    ci.upload_file("sales.csv", "month,revenue\nJan,100\nFeb,120\nMar,150\n")
    print(output_of(ci.execute_code(
        """
import pandas as pd
df = pd.read_csv("sales.csv")
print(df.describe())
print("growth:", df.revenue.pct_change().mean())
df.plot(x="month", y="revenue").figure.savefig("revenue.png")
""",
        language="python",
    )))
    chart = ci.download_file("revenue.png")  # bytes for binary files, str for text
    print(len(chart), "bytes")
```

pandas, numpy, matplotlib and many other data libraries are preinstalled. `install_packages([...])` runs `pip install` for anything else, which needs the interpreter to have network access. `execute_command` runs a shell command, `upload_files` writes several files atomically, and `execute_code` also accepts `language="javascript"` or `"typescript"`. `clear_context=True` resets Python state. `CodeInterpreter(region).start(...)` takes `identifier` (use your own interpreter, for example one with VPC networking, created with `create_code_interpreter`) and `session_timeout_seconds` (default 900).

To give a Strands agent the same capability, wrap it in a tool (or use the `code_interpreter` tool from `strands-agents-tools`) and always stop the session:

```python
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
from strands import Agent, tool

ci = CodeInterpreter("us-west-2")
ci.start()


@tool
def run_python(code: str) -> str:
    """Execute Python code in a secure sandbox and return its output.

    Args:
        code: Python source to execute.
    """
    try:
        return output_of(ci.execute_code(code))
    except RuntimeError as error:
        return f"Execution failed: {error}"  # let the model read the error and try again


analyst = Agent(model="us.anthropic.%%model=anthropic%%", tools=[run_python])
try:
    analyst("Load sales.csv and report month-on-month growth.")
finally:
    ci.stop()
```

(The second example reuses `output_of` from the first and assumes `sales.csv` was uploaded with `ci.upload_file`.)

### Browser

The AgentCore Browser is a managed Chromium that your agent drives over the Chrome DevTools Protocol. `browser_session` starts a session and stops it on exit. `generate_ws_headers()` returns a SigV4-signed WebSocket URL and headers, which any CDP client, such as Playwright, can connect to. `pip install playwright` installs only the client library; no local browser is launched.

```python
import asyncio

from playwright.async_api import async_playwright

from bedrock_agentcore.tools.browser_client import browser_session


async def page_title(url: str) -> str:
    # browser_session starts a managed browser session and stops it on exit
    with browser_session("us-west-2") as client:
        ws_url, headers = client.generate_ws_headers()  # SigV4-signed WebSocket URL for CDP
        async with async_playwright() as playwright:
            browser = await playwright.chromium.connect_over_cdp(ws_url, headers=headers)
            try:
                context = browser.contexts[0] if browser.contexts else await browser.new_context()
                page = context.pages[0] if context.pages else await context.new_page()
                await page.goto(url)
                return await page.title()
            finally:
                await browser.close()


if __name__ == "__main__":
    print(asyncio.run(page_title("https://example.com")))
```

`client.generate_live_view_url()` returns a presigned URL (valid for 300 seconds by default) that shows the session in a browser, which helps when debugging an agent. `browser_session` also accepts `viewport`, `proxy_configuration`, `extensions`, `profile_configuration` (persisted profiles), `enterprise_policies` and `certificates`, and `identifier` for a custom browser. Treat page content as untrusted input: a page can contain instructions aimed at your model.

### Workflows and Memory

AgentCore has no workflow DSL of its own. Multi-step workflows are expressed either inside your agent (for example a Strands `GraphBuilder` graph; see the [Strands guide](/strands-agents-guide/python/strands_multi_agent/#graph)) or outside it with Step Functions or EventBridge calling `InvokeAgentRuntime` (see [Integration Patterns](#integration-patterns)).

AgentCore Memory has two layers. **Short-term memory** stores raw conversation *events* per actor (the end user) and session. **Long-term memory** is a set of *records* that memory strategies extract from those events asynchronously (semantic facts, user preferences, summaries, episodes) and file under namespaces such as `/facts/{actorId}/`. The memory resource, its strategies and its event expiry are control-plane configuration. Reading and writing events and records is the data plane.

`MemorySessionManager` and `MemorySession` are the SDK's recommended interface; the older `MemoryClient` still works but its README advises migrating. `MemoryControlPlaneClient` creates the resource:

```python
from bedrock_agentcore.memory import MemoryControlPlaneClient, MemorySessionManager
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole

# Control plane: create the memory resource once (or declare it with `agentcore add memory`)
control = MemoryControlPlaneClient(region_name="us-west-2")
memory = control.create_memory(
    name="CustomerOnboardingMemory",
    event_expiry_days=30,
    strategies=[
        {"semanticMemoryStrategy": {"name": "facts", "namespaceTemplates": ["/facts/{actorId}/"]}},
        {"userPreferenceMemoryStrategy": {"name": "prefs", "namespaceTemplates": ["/prefs/{actorId}/"]}},
    ],
    wait_for_active=True,
)

# Data plane: one session object per (actor, session) pair
manager = MemorySessionManager(memory_id=memory["id"], region_name="us-west-2")
session = manager.create_memory_session(actor_id="customer-42", session_id="onboarding-001")

# Short-term: record the turns as one event
session.add_turns([
    ConversationalMessage("We are a 40-person fintech in London.", MessageRole.USER),
    ConversationalMessage("Thanks, I'll set up the EU billing account.", MessageRole.ASSISTANT),
])

# Short-term: read the conversation back
for turn in session.get_last_k_turns(k=3):
    for message in turn:
        print(message)

# Long-term: records extracted asynchronously by the strategies (may take a minute or more)
for record in session.search_long_term_memories(
    query="company size and location",
    namespace_path="/facts/customer-42/",
    top_k=3,
):
    print(record)
```

Points to get right:

- **Extraction is asynchronous.** A fact written in this turn may not be searchable in the next. Do not rely on long-term recall for information the user just gave; keep the recent turns in the prompt (or read them with `get_last_k_turns`).
- **Namespaces carry the isolation.** Use `{actorId}` in the templates and search under the caller's own namespace, derived from the verified identity. Searching a broader prefix returns other users' records.
- **Roles.** `MessageRole` has `USER`, `ASSISTANT`, `TOOL` and `OTHER`; `BlobMessage` stores binary or structured payloads.
- **Branches.** `session.fork_conversation(...)` and `list_branches()` support alternative conversation paths, for example to compare two answers.
- **Other strategies.** Summarisation and episodic strategies are available. Episodic strategies also take reflection namespaces. `agentcore add memory --strategies SEMANTIC,SUMMARIZATION` declares the common ones with default namespaces.

With Strands, pass `AgentCoreMemorySessionManager` as the agent's `session_manager` so turns are written and retrieved automatically (see the [Strands guide](/strands-agents-guide/python/strands_comprehensive_guide/#agentcore-memory)). For a runtime deployed by the CLI, read the memory ID from the environment variable the CLI injects (`MEMORY_<NAME>_ID`) rather than hard-coding it.

---

## AgentCore APIs

The examples below use boto3 directly. `bedrock-agentcore-control` is the control plane; `bedrock-agentcore` is the data plane.

### Create Agent Runtime

```python
import boto3

control = boto3.client("bedrock-agentcore-control", region_name="us-west-2")

response = control.create_agent_runtime(
    agentRuntimeName="DataAnalysisAgent",
    description="Sales analysis agent",
    agentRuntimeArtifact={
        "containerConfiguration": {
            "containerUri": "111122223333.dkr.ecr.us-west-2.amazonaws.com/data-analysis-agent:latest"
        }
    },
    roleArn="arn:aws:iam::111122223333:role/AgentRuntimeRole",
    networkConfiguration={"networkMode": "PUBLIC"},
    protocolConfiguration={"serverProtocol": "HTTP"},
)

runtime_id = response["agentRuntimeId"]
print(response["agentRuntimeArn"], response["agentRuntimeVersion"])
```

### Invoke Agent Runtime

```python
import json

data = boto3.client("bedrock-agentcore", region_name="us-west-2")

response = data.invoke_agent_runtime(
    agentRuntimeArn=response["agentRuntimeArn"],
    runtimeSessionId="churn-analysis-session-0000000000001",  # >= 33 characters
    runtimeUserId="analyst-7",                                 # optional
    payload=json.dumps({"prompt": "Analyse customer churn for enterprise vs SMB in Q4"}),
)

if "text/event-stream" in response.get("contentType", ""):
    for line in response["response"].iter_lines():
        if line:
            print(line.decode())
else:
    print(json.loads(response["response"].read()))
```

### Update Agent Runtime

Updates create a new immutable runtime version; the `DEFAULT` endpoint follows the latest version unless you pin endpoints yourself.

```python
control.update_agent_runtime(
    agentRuntimeId=runtime_id,
    agentRuntimeArtifact={
        "containerConfiguration": {
            "containerUri": "111122223333.dkr.ecr.us-west-2.amazonaws.com/data-analysis-agent:v2"
        }
    },
    roleArn="arn:aws:iam::111122223333:role/AgentRuntimeRole",
    networkConfiguration={"networkMode": "PUBLIC"},
)
```

### Delete Agent Runtime

```python
control.delete_agent_runtime(agentRuntimeId=runtime_id)
```

---

## Integration Patterns

### Pattern 1: Gateway — expose existing APIs as MCP tools

Gateway turns Lambda functions, OpenAPI/Smithy APIs, API Gateway stages and existing MCP servers into a single MCP endpoint that any MCP-capable agent can use.

```python
import boto3

control = boto3.client("bedrock-agentcore-control", region_name="us-west-2")

gateway = control.create_gateway(
    name="analytics-tools",
    roleArn="arn:aws:iam::111122223333:role/GatewayRole",
    protocolType="MCP",
    authorizerType="AWS_IAM",  # or CUSTOM_JWT, NONE, AUTHENTICATE_ONLY
)

control.create_gateway_target(
    gatewayIdentifier=gateway["gatewayId"],
    name="orders-lambda",
    targetConfiguration={
        "mcp": {
            "lambda": {
                "lambdaArn": "arn:aws:lambda:us-west-2:111122223333:function:orders-api",
                "toolSchema": {
                    "inlinePayload": [
                        {
                            "name": "get_order",
                            "description": "Fetch an order by ID",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"order_id": {"type": "string"}},
                                "required": ["order_id"],
                            },
                        }
                    ]
                },
            }
        }
    },
    credentialProviderConfigurations=[{"credentialProviderType": "GATEWAY_IAM_ROLE"}],
)
```

A Strands agent consumes the gateway like any other streamable-HTTP MCP server. With a `CUSTOM_JWT` authorizer, pass the caller's bearer token:

```python
from strands import Agent
from strands.tools.mcp import MCPClient


def gateway_agent(gateway_url: str, access_token: str) -> Agent:
    """gateway_url is create_gateway()["gatewayUrl"], ending in /mcp."""
    gateway = MCPClient(url=gateway_url, headers={"Authorization": f"Bearer {access_token}"})
    return Agent(model="us.anthropic.%%model=anthropic%%", tools=[gateway])  # the agent manages the MCP session
```

Gateway target tools are named `<target>___<tool>` (for example `orders-lambda___get_order`). Strands connects to the gateway when the `Agent` is constructed, so a wrong URL or missing token fails at that point.

With the `AWS_IAM` authorizer (the CLI default), requests must be SigV4-signed for the `bedrock-agentcore` service, and a bearer header is no use. The AWS Labs package `mcp-proxy-for-aws` provides a signing transport that uses the default credential chain, which on Runtime is the execution role:

```python
# pip install mcp-proxy-for-aws
from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
from strands import Agent
from strands.tools.mcp import MCPClient

GATEWAY_URL = "https://orders-gw-abc123.gateway.bedrock-agentcore.us-west-2.amazonaws.com/mcp"

gateway = MCPClient(
    lambda: aws_iam_streamablehttp_client(
        endpoint=GATEWAY_URL,
        aws_service="bedrock-agentcore",
        aws_region="us-west-2",
    )
)

agent = Agent(model="us.anthropic.%%model=anthropic%%", tools=[gateway])
```

The execution role needs `bedrock-agentcore:InvokeGateway` on the gateway ARN.

In a CLI project, declare the gateway and its targets alongside your runtime instead of calling `create_gateway` yourself. `add gateway-target` takes an external MCP endpoint (`--endpoint`), a runtime in the project (`--runtime`) or a full target object (`--target-configuration`) with `targetType` one of `mcpServer`, `httpRuntime`, `apiGateway`, `openApiSchema`, `smithyModel`, `lambdaFunctionArn`, `connector` or `passthrough`. `add gateway-connector` adds the curated connectors, which include Bedrock knowledge bases and web search. The knowledge-base connector is how harnesses and migrated Classic agents reach a knowledge base:

```bash
agentcore add gateway --name orders-gw --authorizer-type AWS_IAM
agentcore add gateway-target --gateway orders-gw --name docs --endpoint https://mcp.example.com/mcp --outbound-auth none
agentcore add gateway-connector --gateway orders-gw --name kb --connector bedrock-knowledge-bases --knowledge-base ABCDEFGHIJ
agentcore deploy
```

### Pattern 2: AgentCore + Step Functions

Step Functions can call any AWS API through its SDK integrations, so a state machine can call `InvokeAgentRuntime` directly:

```python
import json

import boto3

sfn = boto3.client("stepfunctions")

definition = {
    "Comment": "Three-stage document pipeline on AgentCore Runtime",
    "StartAt": "Extract",
    "States": {
        "Extract": {
            "Type": "Task",
            "Resource": "arn:aws:states:::aws-sdk:bedrockagentcore:invokeAgentRuntime",
            "Parameters": {
                "AgentRuntimeArn": "arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/extractor-abc123",
                "Payload.$": "States.JsonToString($.input)",
            },
            "Next": "Report",
        },
        "Report": {
            "Type": "Task",
            "Resource": "arn:aws:states:::aws-sdk:bedrockagentcore:invokeAgentRuntime",
            "Parameters": {
                "AgentRuntimeArn": "arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/reporter-def456",
                "Payload.$": "States.JsonToString($)",
            },
            "End": True,
        },
    },
}

sfn.create_state_machine(
    name="DocumentPipeline",
    definition=json.dumps(definition),
    roleArn="arn:aws:iam::111122223333:role/StepFunctionsAgentCoreRole",
)
```

Check the response shape in the Step Functions console before chaining states: the SDK integration returns the raw `InvokeAgentRuntime` response, and payload handling differs from a plain Lambda task.

### Pattern 3: AgentCore + EventBridge

EventBridge cannot target an AgentCore runtime directly. Route events to a small Lambda function that calls `invoke_agent_runtime`:

```python
import json
import os

import boto3

agentcore = boto3.client("bedrock-agentcore")


def handler(event, context):
    """EventBridge → Lambda → AgentCore Runtime for each new S3 object."""
    key = event["detail"]["object"]["key"]
    response = agentcore.invoke_agent_runtime(
        agentRuntimeArn=os.environ["AGENT_RUNTIME_ARN"],
        runtimeSessionId=f"doc-{event['id']}",  # EventBridge event IDs are 36 characters
        payload=json.dumps({"prompt": f"Classify and summarise s3://document-bucket/{key}"}),
    )
    return json.loads(response["response"].read())
```

---

## Migrating from Bedrock Agents Classic

Agents Classic closed to new customers on 30 July 2026. Existing agents keep working with no announced end date, but no new features or models will be added. AWS's [migration guidance](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-classic-maintenance-mode.html) maps Classic concepts to AgentCore like this:

| Agents Classic | AgentCore |
|----------------|-----------|
| Managed orchestration loop | [Managed harness](#managed-harness) |
| Action groups (OpenAPI or function schema, Lambda executor) | Gateway targets exposed as MCP tools, or `@tool` functions in code |
| Associated knowledge bases | Gateway knowledge-base connector, or a retrieval tool in code |
| Return of control, `AMAZON.UserInput` | Harness `inline_function` tools (the harness pauses and returns `tool_use`) |
| `AMAZON.CodeInterpreter` | AgentCore Code Interpreter |
| Session TTL and `memoryConfiguration` | AgentCore Memory (short- and long-term strategies) |
| Guardrails on the agent | Bedrock Guardrails on the model call, Gateway policies |
| Prompt overrides per orchestration stage | No direct equivalent in the harness; use a code-defined agent |
| Multi-agent collaboration | Agents as tools in a harness, or Strands multi-agent code on Runtime |

**Start with the CLI import.** `agentcore add runtime --agent-id` generates a code-defined runtime in Strands or LangGraph from an existing agent alias, inside an existing project. It reads the agent, the alias, action groups, associated knowledge bases and collaborators through the Bedrock Agents APIs, so it needs credentials with `bedrock:GetAgent*` and `bedrock:ListAgent*` permissions in the agent's account. The alias must point at a prepared version, not `DRAFT`:

```bash
agentcore create --name OrdersProject
cd OrdersProject
agentcore add runtime --name OrdersAgent --agent-id ABCDEFGHIJ --agent-alias-id TSTALIASID --framework strands --region us-east-1
agentcore dev        # try it locally against the same Lambda functions and knowledge bases
agentcore deploy
```

Review the generated `app/OrdersAgent/` code before deploying. Whatever the Classic agent reached through action groups and knowledge bases (Lambda functions, `bedrock:Retrieve`) needs matching permissions on the new runtime's execution role.

**Or rebuild as a harness** when the Classic agent was purely declarative. Add a harness, set the model, copy the instruction into `system-prompt.md`, put the action groups behind a gateway, and add it as a tool:

```bash
agentcore create --name OrdersProject
cd OrdersProject
agentcore add gateway --name orders-gw --authorizer-type AWS_IAM
# add the action groups' Lambda functions or APIs with: agentcore add gateway-target --gateway orders-gw ...
agentcore add harness --name OrdersAgent \
  --model '{"provider":"bedrock","modelId":"us.anthropic.%%model=anthropic%%"}' \
  --system-prompt "$(cat classic-instruction.txt)"
# then list the gateway under tools: in app/OrdersAgent/harness.yaml (agentcore_gateway), and deploy
agentcore deploy
```

In `harness.yaml`, a gateway tool looks like this (the gateway ARN appears in `agentcore status` after the first deploy):

```yaml
tools:
  - type: agentcore_gateway
    name: orders
    config:
      agentCoreGateway:
        gatewayArn: arn:aws:bedrock-agentcore:us-east-1:111122223333:gateway/orders-gw-abc123
```

AWS also publishes a migration skill for coding agents in the [agent toolkit for AWS](https://github.com/aws/agent-toolkit-for-aws). It inspects a Classic agent, checks whether it can move to a harness and drives the same CLI. Run the old and new agents side by side on recorded conversations (with `agentcore eval`, see the [production guide](/amazon-bedrock-agents-guide/python/bedrock_agents_production_guide/#evaluations)) before switching callers over.

---

## Security and Compliance

### IAM Policies for AgentCore

Caller policy (who may deploy and invoke):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ManageRuntimes",
      "Effect": "Allow",
      "Action": [
        "bedrock-agentcore:CreateAgentRuntime",
        "bedrock-agentcore:UpdateAgentRuntime",
        "bedrock-agentcore:DeleteAgentRuntime",
        "bedrock-agentcore:GetAgentRuntime",
        "bedrock-agentcore:ListAgentRuntimes"
      ],
      "Resource": "arn:aws:bedrock-agentcore:*:111122223333:runtime/*"
    },
    {
      "Sid": "InvokeRuntimes",
      "Effect": "Allow",
      "Action": "bedrock-agentcore:InvokeAgentRuntime",
      "Resource": "arn:aws:bedrock-agentcore:*:111122223333:runtime/*"
    },
    {
      "Sid": "PassExecutionRole",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::111122223333:role/AgentRuntimeRole"
    }
  ]
}
```

Callers that pass `runtimeUserId` also need `bedrock-agentcore:InvokeAgentRuntimeForUser`. The runtime's execution role trusts `bedrock-agentcore.amazonaws.com` and needs ECR pull, CloudWatch Logs/X-Ray, `bedrock:InvokeModel`/`bedrock:InvokeModelWithResponseStream` on the models it calls, and access to any Memory, Gateway or Code Interpreter resources it uses. See [IAM Permissions for AgentCore Runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-permissions.html) for the full template.

### Inbound authentication

By default callers authenticate with SigV4 (IAM). To accept end-user tokens from an OAuth/OIDC identity provider instead, configure a JWT authorizer on the runtime:

```python
import boto3

control = boto3.client("bedrock-agentcore-control", region_name="us-west-2")

control.update_agent_runtime(
    agentRuntimeId="DataAnalysisAgent-abc123",
    agentRuntimeArtifact={
        "containerConfiguration": {"containerUri": "111122223333.dkr.ecr.us-west-2.amazonaws.com/data-analysis-agent:v2"}
    },
    roleArn="arn:aws:iam::111122223333:role/AgentRuntimeRole",
    networkConfiguration={"networkMode": "PUBLIC"},
    authorizerConfiguration={
        "customJWTAuthorizer": {
            "discoveryUrl": "https://cognito-idp.us-west-2.amazonaws.com/us-west-2_EXAMPLE/.well-known/openid-configuration",
            "allowedClients": ["my-app-client-id"],  # allowedAudience and allowedScopes are also available
        }
    },
)
```

Callers using a JWT invoke the runtime over HTTPS with a bearer token rather than through the SigV4-signed SDK call.

### Outbound credentials with Identity

AgentCore Identity keeps third-party credentials out of your code and images. Credential providers (OAuth 2.0 or API key) are created in the control plane (`create_oauth2_credential_provider`, `create_api_key_credential_provider`) or declared in a CLI project with `agentcore add credentials api-key|oauth`. Inside the runtime, decorators fetch (and cache) a credential using the request's workload access token and inject it into your function.

`requires_access_token` handles OAuth 2.0. `auth_flow` is `M2M` (client credentials, no user), `USER_FEDERATION` (three-legged OAuth: the user consents once and the token is stored against their identity) or `ON_BEHALF_OF_TOKEN_EXCHANGE`:

```python
from bedrock_agentcore.identity.auth import requires_access_token, requires_api_key


@requires_access_token(
    provider_name="google-workspace",   # credential provider created in AgentCore Identity
    scopes=["https://www.googleapis.com/auth/calendar.readonly"],
    auth_flow="USER_FEDERATION",
    on_auth_url=lambda url: print("Authorise at:", url),  # show this URL to the user
)
async def list_calendar_events(*, access_token: str):
    ...  # call the Google Calendar API with access_token


@requires_api_key(provider_name="weather-api")
def current_weather(city: str, *, api_key: str):
    ...  # call the weather API with api_key
```

The injected parameter is `access_token` or `api_key` by default; `into="token"` renames it. `requires_iam_access_token(audience=[...])` is separate: it asks AWS STS for a signed JWT (`GetWebIdentityToken`), for services that validate AWS-issued OIDC tokens, with no client secret. `force_authentication=True` bypasses the cache. The decorators work on both sync and async functions; call a decorated helper from inside a Strands `@tool` function to give an agent a credentialed API. In a CLI project the provider name comes from `agentcore add credentials` and the runtime's environment (`CREDENTIAL_<NAME>_NAME`).

### Encryption and Audit Logging

- Memory supports customer-managed KMS keys (`encryptionKeyArn` on `create_memory`); Gateway accepts `kmsKeyArn`.
- AgentCore control- and data-plane calls are recorded by CloudTrail.
- Runtime stdout/stderr go to CloudWatch Logs under `/aws/bedrock-agentcore/runtimes/…`; traces go to CloudWatch via OpenTelemetry once Transaction Search is enabled.

---

## Best Practices

### 1. Code organisation

```
my-agentcore-project/
├── main.py              # BedrockAgentCoreApp entrypoint only
├── agent/
│   ├── __init__.py
│   ├── build.py         # constructs the Strands/LangGraph agent
│   └── tools.py
├── tests/
│   └── test_tools.py
├── pyproject.toml       # pins bedrock-agentcore, strands-agents, boto3
└── Dockerfile           # linux/arm64 if you deploy a container
```

Keep the entrypoint thin: validate input, call the agent, shape the output. That keeps the agent testable without Runtime.

### 2. Error Handling

boto3 raises `botocore` `ClientError`. The service errors you will meet from `InvokeAgentRuntime` are `ThrottlingException`, `ServiceQuotaExceededException`, `InternalServerException` and `RetryableConflictException` (all worth retrying with backoff), and `ValidationException`, `AccessDeniedException`, `ResourceNotFoundException` and `RuntimeClientError` (fix the request, role or container instead; read the message of a `RuntimeClientError`, which AWS describes as a client-side problem such as a network or configuration fault). botocore's own retry handler already retries throttling and 5xx responses, so an application-level loop only adds the cases it does not cover:

```python
import json
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

agentcore = boto3.client(
    "bedrock-agentcore",
    region_name="us-west-2",
    # Streaming responses can be slow to finish; raise the read timeout above the default 60 s
    config=Config(retries={"max_attempts": 3, "mode": "standard"}, read_timeout=900),
)

# Temporary conditions. botocore's own retry handler already covers throttling and
# 5xx responses, so this loop only matters for the extra cases below.
RETRYABLE = {"ThrottlingException", "InternalServerException", "RetryableConflictException"}


def invoke_agent(runtime_arn: str, session_id: str, prompt: str, attempts: int = 4) -> dict:
    for attempt in range(attempts):
        try:
            response = agentcore.invoke_agent_runtime(
                agentRuntimeArn=runtime_arn,
                runtimeSessionId=session_id,
                payload=json.dumps({"prompt": prompt}),
            )
            if response["statusCode"] != 200:
                raise RuntimeError(f"runtime answered HTTP {response['statusCode']}")
            return json.loads(response["response"].read())
        except ClientError as err:
            code = err.response["Error"]["Code"]
            if code in RETRYABLE and attempt < attempts - 1:
                time.sleep(min(2 ** attempt, 20))
                continue
            # ValidationException, AccessDeniedException, ResourceNotFoundException and
            # RuntimeClientError will not succeed on retry: fix the request, role or container
            raise
    raise AssertionError("unreachable")
```

Retrying a call to the same `runtimeSessionId` repeats the work inside the same microVM, so make tool side effects idempotent. Errors raised by your own entrypoint reach the caller as an HTTP 500 from the container; see [Error handling](/amazon-bedrock-agents-guide/python/bedrock_agents_error_handling_python/#4-errors-on-agentcore) for what the caller sees and how to design for it.

### 3. Performance

- Build the agent object once at module import, not inside the entrypoint, so warm sessions reuse it.
- Reuse `runtimeSessionId` for multi-turn conversations; the session's in-memory state survives between turns.
- Stream responses (async-generator entrypoint) for anything user-facing.
- Use Bedrock prompt caching (`cache_config=CacheConfig(strategy="auto")` on Strands `BedrockModel`) for long, stable system prompts.

### 4. Monitoring

AgentCore Observability uses OpenTelemetry: Strands emits spans for agent loops, model calls and tool calls, and CLI-generated projects include the AWS Distro for OpenTelemetry. Setup, custom spans and metrics are on the [observability page](/amazon-bedrock-agents-guide/python/bedrock_agents_observability_python/).

---

## Production Deployment

Runtime versions and endpoints, CI/CD with the CLI, evaluations, quotas, IAM and cost are covered in the [production guide](/amazon-bedrock-agents-guide/python/bedrock_agents_production_guide/).

---

## Conclusion

AgentCore is the AWS-recommended place to run agents on Bedrock: bring an agent written in any framework, wrap it in `BedrockAgentCoreApp`, deploy it with the AgentCore CLI, and add Memory, Gateway, Identity and the sandboxed tools as you need them.

**Documentation:** https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/
**Python SDK:** https://github.com/aws/bedrock-agentcore-sdk-python
**CLI:** https://github.com/aws/agentcore-cli

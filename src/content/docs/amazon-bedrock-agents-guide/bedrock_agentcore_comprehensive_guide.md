---
title: "Amazon Bedrock AgentCore: Comprehensive Technical Guide"
description: "Technical reference for Amazon Bedrock AgentCore: Runtime, Memory, Gateway, Identity, Code Interpreter, Browser and Observability, with examples in the bedrock-agentcore Python SDK and boto3."
framework: amazon-bedrock-agents
---

# Amazon Bedrock AgentCore: Comprehensive Technical Guide

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
5. [Secure Sandbox Runtime](#secure-sandbox-runtime)
6. [Supported Frameworks](#supported-frameworks)
7. [Data Analysis and Workflow Automation](#data-analysis-and-workflow-automation)
8. [AgentCore APIs](#agentcore-apis)
9. [Integration Patterns](#integration-patterns)
10. [Security and Compliance](#security-and-compliance)
11. [Best Practices](#best-practices)
12. [Production Deployment](#production-deployment)

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
| **Memory** | Short-term events and long-term extracted records | `bedrock_agentcore.memory.MemoryClient` | `bedrock-agentcore-control` (`create_memory`), `bedrock-agentcore` (`create_event`, `retrieve_memory_records`) |
| **Gateway** | Turns APIs, Lambda and MCP servers into one MCP endpoint | `bedrock_agentcore.gateway.client.GatewayClient` | `bedrock-agentcore-control` (`create_gateway`, `create_gateway_target`) |
| **Identity** | Inbound auth (JWT/IAM) and outbound OAuth / API-key credentials | `bedrock_agentcore.identity.auth.requires_access_token` | `bedrock-agentcore-control` (credential providers) |
| **Code Interpreter** | Sandboxed Python/JavaScript/TypeScript execution | `bedrock_agentcore.tools.code_interpreter_client.code_session` | `bedrock-agentcore` (`invoke_code_interpreter`) |
| **Browser** | Managed headless browser for web automation | `bedrock_agentcore.tools.browser_client.browser_session` | `bedrock-agentcore` (`start_browser_session`) |
| **Observability** | OpenTelemetry traces, metrics and logs in CloudWatch | via OTEL / ADOT instrumentation | CloudWatch |

AgentCore also includes Policy, Evaluations and Payments services; they are outside the scope of this guide.

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
- Node.js, if you use the AgentCore CLI

### Installation

```bash
pip install "bedrock-agentcore==%%version=bedrock-agentcore%%" "strands-agents==%%version=strands-agents%%" "boto3==%%version=boto3%%"

# AgentCore CLI (recommended for packaging and deployment; generates CDK under the hood)
npm install -g @aws/agentcore
```

The older Python `bedrock-agentcore-starter-toolkit` (`agentcore configure` / `agentcore launch`) still exists, but the AgentCore SDK README now points to the npm CLI as the recommended route.

### Quick Start Example

`main.py` — a Strands agent wrapped for Runtime:

```python
from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent

app = BedrockAgentCoreApp()

agent = Agent(
    model="us.anthropic.claude-sonnet-5-5",  # Bedrock cross-Region inference profile
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

Test it locally, then deploy:

```bash
python main.py &
curl -s -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Summarise the main drivers of churn in one paragraph."}'

# Deploy with the AgentCore CLI
agentcore create --name SalesAnalyst --defaults
cd SalesAnalyst
agentcore deploy
agentcore invoke
```

Invoke the deployed runtime from any Python code with boto3:

```python
import json
import uuid

import boto3

client = boto3.client("bedrock-agentcore", region_name="us-west-2")

response = client.invoke_agent_runtime(
    agentRuntimeArn="arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/SalesAnalyst-abc123",
    runtimeSessionId=str(uuid.uuid4()) + "-session",  # must be at least 33 characters
    payload=json.dumps({"prompt": "Analyse monthly sales trends for Q4"}),
    qualifier="DEFAULT",
)

print(json.loads(response["response"].read()))
```

Reuse the same `runtimeSessionId` to keep talking to the same session microVM (and its in-memory state); use a new one for a fresh, isolated session.

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
- **The ping status** to tell Runtime whether a session is busy with background work, so it is not reclaimed while idle-looking:

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

Covered in detail in the [Strands SDK guide](/amazon-bedrock-agents-guide/bedrock_strands_sdk_guide/). Streaming is supported by making the entrypoint an async generator:

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


agent = Agent(model="us.anthropic.claude-sonnet-5-5", tools=[total_revenue])


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

llm = ChatBedrockConverse(model="us.anthropic.claude-sonnet-5-5")
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
    llm="bedrock/us.anthropic.claude-sonnet-5-5",
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

agent = Agent(model="us.anthropic.claude-sonnet-5-5", system_prompt="You are a helpful assistant.")
serve_a2a(StrandsA2AExecutor(agent))
```

---

## Data Analysis and Workflow Automation

### Code Interpretation

The Code Interpreter runs code in a managed sandbox, not in your agent's process. `code_session` starts a session on the AWS-managed `aws.codeinterpreter.v1` interpreter and stops it on exit:

```python
from bedrock_agentcore.tools.code_interpreter_client import code_session

with code_session("us-west-2") as ci:
    ci.upload_file("sales.csv", "month,revenue\nJan,100\nFeb,120\nMar,150\n")
    result = ci.execute_code(
        """
import pandas as pd
df = pd.read_csv("sales.csv")
print(df.describe())
print("growth:", df.revenue.pct_change().mean())
""",
        language="python",
    )
    print(result)
```

To give a Strands agent the same capability, wrap it in a tool (or use the `code_interpreter` tool from `strands-agents-tools`):

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
    return str(ci.execute_code(code))


analyst = Agent(model="us.anthropic.claude-sonnet-5-5", tools=[run_python])
analyst("Load sales.csv and report month-on-month growth.")
ci.stop()
```

### Workflow Automation Examples

AgentCore has no workflow DSL of its own. Multi-step workflows are expressed either inside your agent (for example a Strands `GraphBuilder` graph — see the [Strands guide](/amazon-bedrock-agents-guide/bedrock_strands_sdk_guide/#collaboration-pattern-3-agent-graphs)) or outside it with Step Functions or EventBridge calling `InvokeAgentRuntime` (see [Integration Patterns](#integration-patterns)).

**Remembering context across sessions with Memory:**

```python
from bedrock_agentcore.memory import MemoryClient

memory = MemoryClient(region_name="us-west-2")

mem = memory.create_memory_and_wait(
    name="CustomerOnboardingMemory",
    strategies=[
        {"semanticMemoryStrategy": {"name": "facts", "namespaceTemplates": ["/facts/{actorId}/"]}},
        {"userPreferenceMemoryStrategy": {"name": "prefs", "namespaceTemplates": ["/prefs/{actorId}/"]}},
    ],
    event_expiry_days=30,
)
memory_id = mem["id"]

# Short-term: record conversation turns as events
memory.create_event(
    memory_id=memory_id,
    actor_id="customer-42",
    session_id="onboarding-001",
    messages=[("We are a 40-person fintech in London.", "USER"),
              ("Thanks — I'll set up the EU billing account.", "ASSISTANT")],
)

# Long-term: query records extracted asynchronously by the strategies
records = memory.retrieve_memories(
    memory_id=memory_id,
    namespace="/facts/customer-42/",
    query="company size and location",
    top_k=3,
)
```

With Strands, pass `AgentCoreMemorySessionManager` as the agent's `session_manager` so turns are written and retrieved automatically (see the [Strands guide](/amazon-bedrock-agents-guide/bedrock_strands_sdk_guide/#memory-and-state-management)).

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

A Strands agent consumes the gateway like any other MCP server, via `strands.tools.mcp.MCPClient` (with SigV4 or a bearer token, depending on the authorizer).

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

### Pattern 4: Migrating from Bedrock Agents Classic

AWS's migration guidance maps Classic concepts onto AgentCore: action groups become MCP tools behind Gateway, knowledge bases are reached through Gateway, return-of-control becomes inline function tools, and session state moves to AgentCore Memory. Start from the managed harness when the Classic agent was purely declarative; use code-defined agents on Runtime when you relied on prompt overrides, custom orchestration or multi-agent collaboration.

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

The runtime's execution role trusts `bedrock-agentcore.amazonaws.com` and needs ECR pull, CloudWatch Logs/X-Ray, `bedrock:InvokeModel`/`bedrock:InvokeModelWithResponseStream` on the models it calls, and access to any Memory, Gateway or Code Interpreter resources it uses. See [IAM Permissions for AgentCore Runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-permissions.html) for the full template.

### Inbound authentication

By default callers authenticate with SigV4 (IAM). To accept end-user tokens from an OAuth/OIDC identity provider instead, configure a JWT authorizer on the runtime:

```python
control.update_agent_runtime(
    agentRuntimeId=runtime_id,
    agentRuntimeArtifact={"containerConfiguration": {"containerUri": IMAGE_URI}},
    roleArn=ROLE_ARN,
    networkConfiguration={"networkMode": "PUBLIC"},
    authorizerConfiguration={
        "customJWTAuthorizer": {
            "discoveryUrl": "https://cognito-idp.us-west-2.amazonaws.com/us-west-2_EXAMPLE/.well-known/openid-configuration",
            "allowedClients": ["my-app-client-id"],
        }
    },
)
```

Callers using a JWT invoke the runtime over HTTPS with a bearer token rather than through the SigV4-signed SDK call.

### Outbound credentials with Identity

`requires_access_token` fetches (and caches) an OAuth token from an AgentCore Identity credential provider and injects it into your function:

```python
from bedrock_agentcore.identity.auth import requires_access_token


@requires_access_token(
    provider_name="google-workspace",   # credential provider created in AgentCore Identity
    scopes=["https://www.googleapis.com/auth/calendar.readonly"],
    auth_flow="USER_FEDERATION",          # or "M2M"
    on_auth_url=lambda url: print("Authorise at:", url),
)
async def list_calendar_events(*, access_token: str):
    ...
```

`requires_api_key(provider_name=...)` does the same for API-key providers.

### Encryption and Audit Logging

- Memory supports customer-managed KMS keys (`encryptionKeyArn` on `create_memory`); Gateway accepts `kmsKeyArn`.
- AgentCore control- and data-plane calls are recorded by CloudTrail.
- Runtime stdout/stderr go to CloudWatch Logs under `/aws/bedrock-agentcore/runtimes/…`; traces go to CloudWatch via OpenTelemetry once Transaction Search is enabled.

---

## Best Practices

### 1. Code Organization

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

boto3 raises `botocore` `ClientError`; retry only throttling and transient errors:

```python
import json
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

agentcore = boto3.client(
    "bedrock-agentcore",
    config=Config(retries={"max_attempts": 5, "mode": "adaptive"}, read_timeout=900),
)

RETRYABLE = {"ThrottlingException", "ServiceQuotaExceededException", "InternalServerException"}


def invoke_with_retry(runtime_arn: str, session_id: str, prompt: str, attempts: int = 3) -> dict:
    for attempt in range(attempts):
        try:
            response = agentcore.invoke_agent_runtime(
                agentRuntimeArn=runtime_arn,
                runtimeSessionId=session_id,
                payload=json.dumps({"prompt": prompt}),
            )
            return json.loads(response["response"].read())
        except ClientError as err:
            code = err.response["Error"]["Code"]
            if code in RETRYABLE and attempt < attempts - 1:
                time.sleep(2 ** attempt)
                continue
            raise
```

### 3. Performance

- Build the agent object once at module import, not inside the entrypoint, so warm sessions reuse it.
- Reuse `runtimeSessionId` for multi-turn conversations; the session's in-memory state survives between turns.
- Stream responses (async-generator entrypoint) for anything user-facing.
- Use Bedrock prompt caching (`cache_config=CacheConfig(strategy="auto")` on Strands `BedrockModel`) for long, stable system prompts.

### 4. Monitoring

AgentCore Observability uses OpenTelemetry. For Python agents, add the AWS Distro for OpenTelemetry and run the agent under `opentelemetry-instrument`; Strands emits spans for model calls, tool calls and agent loops automatically.

```dockerfile
RUN pip install aws-opentelemetry-distro
CMD ["opentelemetry-instrument", "python", "main.py"]
```

Custom business metrics can still go straight to CloudWatch with `boto3.client("cloudwatch").put_metric_data(...)`.

---

## Production Deployment

### Deployment Checklist

- [ ] Agent tested locally against `POST /invocations`
- [ ] Execution role scoped to the specific models, memories and gateways used
- [ ] Inbound auth chosen (IAM or JWT) and tested
- [ ] Lifecycle timeouts set to match the workload
- [ ] VPC networking configured if the agent reaches private resources
- [ ] Observability (Transaction Search, OTEL) enabled
- [ ] Memory expiry and KMS settings reviewed
- [ ] Load test against concurrent-session quotas

### Versions and Endpoints (blue/green)

Every create or update produces an immutable runtime version. Endpoints are named pointers to versions, so blue/green is a matter of moving an endpoint:

```python
control = boto3.client("bedrock-agentcore-control")

# Stage the new version behind its own endpoint
control.create_agent_runtime_endpoint(
    agentRuntimeId=runtime_id,
    name="canary",
    agentRuntimeVersion="7",
)

# ... run tests against qualifier="canary" ...

# Promote: point the (previously created) production endpoint at the tested version
control.update_agent_runtime_endpoint(
    agentRuntimeId=runtime_id,
    endpointName="production",
    agentRuntimeVersion="7",
)

# Roll back by pointing it at the previous version again
```

Callers select an endpoint with the `qualifier` argument of `invoke_agent_runtime`.

### Cost Optimization

AgentCore has no cost-analysis API of its own. Use Cost Explorer (filter on the Amazon Bedrock AgentCore service) and these levers:

- Runtime bills for active CPU and memory per session; short idle timeouts stop idle sessions accruing charges.
- Pick the smallest model that holds quality for each step; Claude Haiku-class models suit routing and extraction.
- Memory long-term strategies run extraction on every event — enable only the strategies you query.
- Code Interpreter and Browser sessions bill while open; always `stop()` them (the `code_session` / `browser_session` context managers do this).

---

## Conclusion

AgentCore is the AWS-recommended place to run agents on Bedrock: bring an agent written in any framework, wrap it in `BedrockAgentCoreApp`, deploy it with the AgentCore CLI, and add Memory, Gateway, Identity and the sandboxed tools as you need them.

**Documentation:** https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/
**Python SDK:** https://github.com/aws/bedrock-agentcore-sdk-python
**CLI:** https://github.com/aws/agentcore-cli

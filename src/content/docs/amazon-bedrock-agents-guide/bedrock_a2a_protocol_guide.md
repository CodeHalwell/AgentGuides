---
title: "Agent-to-Agent (A2A) Protocol: Comprehensive Technical Guide"
description: "The open Agent2Agent (A2A) protocol on AWS: agent cards, JSON-RPC methods, tasks and messages, Strands A2A servers and clients, and hosting A2A agents on Amazon Bedrock AgentCore Runtime."
framework: amazon-bedrock-agents
---

A2A (Agent2Agent) is an open protocol that lets agents built with different frameworks, by different teams, discover one another and collaborate over HTTP. Google introduced it in April 2025; it is now developed as a Linux Foundation project. On AWS, A2A is supported by **Amazon Bedrock AgentCore Runtime** and by the **Strands Agents** SDK.

> **Verified against:** `bedrock-agentcore` %%version=bedrock-agentcore%% and `strands-agents` %%version=strands-agents%% (checked %%versions-checked%%), with the `a2a` extras installed. The wire examples follow A2A protocol version 0.3, which those SDKs use by default.

**Specification:** https://a2a-protocol.org/
**Python SDK:** `a2a-sdk` (installed by the `a2a` extras of Strands and AgentCore)

> **Bedrock Agents (Classic) does not implement A2A.** Its multi-agent collaboration uses a built-in supervisor/collaborator mechanism, and `invoke_agent` has no A2A parameters. Earlier versions of this page described an invented "A2A message envelope" for Bedrock Agents; that has been removed.

---

## Table of Contents

1. [Introduction to A2A Protocol](#introduction-to-a2a-protocol)
2. [Core Concepts](#core-concepts)
3. [Protocol Specification](#protocol-specification)
4. [Building an A2A Server with Strands](#building-an-a2a-server-with-strands)
5. [Hosting A2A Agents on AgentCore Runtime](#hosting-a2a-agents-on-agentcore-runtime)
6. [Calling A2A Agents](#calling-a2a-agents)
7. [Other Frameworks](#other-frameworks)
8. [Security](#security)
9. [Best Practices](#best-practices)

---

## Introduction to A2A Protocol

A2A complements MCP. MCP connects an agent to *tools and data*; A2A connects an agent to *other agents* that have their own reasoning, state and long-running work.

### Key Benefits

| Benefit | Description |
|---------|-------------|
| **Framework interoperability** | A Strands agent can delegate to a LangGraph or Google ADK agent without knowing how it is built |
| **Discovery** | Each agent publishes an *agent card* describing its skills, endpoint and auth requirements |
| **Long-running work** | Work is modelled as *tasks* with states, streaming updates and optional push notifications |
| **Opaque agents** | Agents share messages and artefacts, not internal memory, prompts or tools |
| **Standard web stack** | JSON-RPC 2.0 over HTTPS, Server-Sent Events for streaming, standard auth schemes |

### Supported Frameworks

Any framework with an `a2a-sdk` `AgentExecutor` can be served over A2A. Strands ships `StrandsA2AExecutor`; Google ADK and LangGraph have their own A2A integrations; AgentCore's `serve_a2a` hosts any of them.

### Use Cases

- A central assistant delegating to department-owned agents
- Cross-organisation workflows where each party runs its own agent
- Gradual migration: wrap an existing agent in A2A and call it from new code

---

## Core Concepts

### Agent Identity

An **agent card** is a JSON document served at `/.well-known/agent-card.json` (A2A 0.3; earlier drafts used `/.well-known/agent.json`). An example as served from AgentCore Runtime:

```json
{
  "name": "Research Agent",
  "description": "Researches a topic and returns sourced bullet points.",
  "version": "1.0.0",
  "url": "https://bedrock-agentcore.us-west-2.amazonaws.com/runtimes/<url-encoded-runtime-arn>/invocations/",
  "protocolVersion": "0.3.0",
  "preferredTransport": "JSONRPC",
  "capabilities": { "streaming": true },
  "defaultInputModes": ["text"],
  "defaultOutputModes": ["text"],
  "skills": [
    {
      "id": "research",
      "name": "Topic research",
      "description": "Finds and summarises sources on a topic",
      "tags": ["research"]
    }
  ]
}
```

Cards can also declare `securitySchemes` / `security` (for example OAuth 2.0 bearer tokens) so that clients know how to authenticate.

### Message Format

A **message** is a turn from a `user` (the calling agent or app) or an `agent`, made of **parts**:

```json
{
  "role": "user",
  "messageId": "6f1b2c1e-1d1a-4d3e-9c1e-4c8a4b2b9e10",
  "parts": [
    { "kind": "text", "text": "Summarise the attached report" },
    { "kind": "file", "file": { "uri": "https://example.com/report.pdf", "mimeType": "application/pdf" } },
    { "kind": "data", "data": { "audience": "board", "maxWords": 200 } }
  ]
}
```

### Message Types

| Object | Purpose |
|--------|---------|
| **Message** | A single turn (`role`, `parts`, `messageId`, optional `contextId` / `taskId`) |
| **Task** | A unit of work with an `id`, `contextId`, `status` (state + optional message), `history` and `artifacts` |
| **Artifact** | An output produced by the task (`artifactId`, `name`, `parts`) |
| **Status / artifact update events** | Streamed progress (`status-update`, `artifact-update`) during `message/stream` |

Task states: `submitted`, `working`, `input-required`, `auth-required`, `completed`, `canceled`, `failed`, `rejected`, `unknown`.

---

## Protocol Specification

### Message Structure

Requests are JSON-RPC 2.0 over HTTP POST to the agent's URL.

#### Required Fields

| Field | Meaning |
|-------|---------|
| `jsonrpc` | Always `"2.0"` |
| `id` | Request ID, echoed in the response |
| `method` | One of the methods below |
| `params` | Method parameters; for `message/send` a `message` object with `role`, `parts` and `messageId` |

#### Optional Fields

| Field | Meaning |
|-------|---------|
| `params.message.contextId` | Groups related tasks into one conversation |
| `params.message.taskId` | Continue an existing task (for example after `input-required`) |
| `params.configuration` | e.g. `acceptedOutputModes`, `blocking`, `historyLength`, `pushNotificationConfig` |
| `params.metadata` | Free-form metadata |

**Methods (A2A 0.3):**

| Method | Purpose |
|--------|---------|
| `message/send` | Send a message; returns a `Task` or a `Message` |
| `message/stream` | Same, but streams task status and artefact updates over SSE |
| `tasks/get` | Fetch a task's current state, history and artefacts |
| `tasks/cancel` | Cancel a running task |
| `tasks/resubscribe` | Re-attach to a task's event stream |
| `tasks/pushNotificationConfig/set` · `get` · `list` · `delete` | Manage webhook push notifications for long tasks |
| `agent/getAuthenticatedExtendedCard` | Fetch a richer card after authenticating |

### Payload Schemas

#### Request Payload

```json
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [{ "kind": "text", "text": "What is 101 * 11?" }],
      "messageId": "12345678-1234-1234-1234-123456789012"
    }
  }
}
```

#### Response Payload

```json
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "result": {
    "kind": "task",
    "id": "task-7c1",
    "contextId": "ctx-42",
    "status": { "state": "completed" },
    "artifacts": [
      {
        "artifactId": "a-1",
        "name": "agent_response",
        "parts": [{ "kind": "text", "text": "101 × 11 = 1111" }]
      }
    ]
  }
}
```

#### Notification Payload

For long-running tasks, a client can register a webhook with `tasks/pushNotificationConfig/set`; the server then POSTs task updates to it. Errors use standard JSON-RPC error objects (`code`, `message`, optional `data`); A2A reserves codes such as `-32001` (task not found) and `-32002` (task not cancelable).

---

## Building an A2A Server with Strands

```bash
pip install "strands-agents[a2a]==%%version=strands-agents%%"
```

```python
from strands import Agent, tool
from strands.multiagent.a2a import A2AServer


@tool
def add_numbers(a: int, b: int) -> int:
    """Return the sum of two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


agent = Agent(
    model="us.anthropic.claude-sonnet-5-5",
    name="Calculator Agent",
    description="Performs arithmetic.",
    tools=[add_numbers],
)

# Serves JSON-RPC at / and the agent card at /.well-known/agent-card.json
A2AServer(agent=agent, host="0.0.0.0", port=9000).serve()
```

Test it:

```bash
curl -s http://localhost:9000/.well-known/agent-card.json | jq .

curl -s -X POST http://localhost:9000/ -H "Content-Type: application/json" -d '{
  "jsonrpc": "2.0", "id": "1", "method": "message/send",
  "params": {"message": {"role": "user", "messageId": "m-1",
                         "parts": [{"kind": "text", "text": "what is 101 * 11?"}]}}
}' | jq .
```

`A2AServer` builds the agent card from the agent's `name`, `description` and tools; pass `skills=[...]` (a list of `a2a.types.AgentSkill`) to describe skills explicitly, and `http_url=` when the server sits behind a proxy so the card advertises the public URL.

---

## Hosting A2A Agents on AgentCore Runtime

AgentCore Runtime acts as a transparent proxy for A2A: JSON-RPC payloads sent with `InvokeAgentRuntime` are passed to your container unchanged, while AgentCore adds session isolation and SigV4 or OAuth authentication.

**Container contract for A2A:** listen on `0.0.0.0:9000`, serve JSON-RPC at `/` (POST), the agent card at `/.well-known/agent-card.json` (GET) and health at `/ping`; build for ARM64. `serve_a2a` from the AgentCore SDK does all of this:

```python
# pip install "bedrock-agentcore[a2a]" "strands-agents[a2a]"
from bedrock_agentcore.runtime import serve_a2a
from strands import Agent
from strands.multiagent.a2a import StrandsA2AExecutor

agent = Agent(
    model="us.anthropic.claude-sonnet-5-5",
    name="Research Agent",
    description="Researches a topic and returns sourced bullet points.",
)

if __name__ == "__main__":
    serve_a2a(StrandsA2AExecutor(agent))
```

Deploy with the AgentCore CLI:

```bash
agentcore create --project-name A2AProject --name ResearchAgent \
  --language Python --framework Strands --model-provider Bedrock --protocol A2A
cd A2AProject
agentcore dev      # local test with the agent inspector
agentcore deploy
```

Or, with boto3, set the protocol on the runtime:

```python
import boto3

control = boto3.client("bedrock-agentcore-control", region_name="us-west-2")
control.create_agent_runtime(
    agentRuntimeName="research_agent",
    agentRuntimeArtifact={"containerConfiguration": {"containerUri": "111122223333.dkr.ecr.us-west-2.amazonaws.com/research-agent:latest"}},
    roleArn="arn:aws:iam::111122223333:role/AgentRuntimeRole",
    networkConfiguration={"networkMode": "PUBLIC"},
    protocolConfiguration={"serverProtocol": "A2A"},
)
```

The data plane also exposes the card through the API (`bedrock-agentcore` `get_agent_card`), so callers with IAM access can discover an agent without hand-building URLs.

`bedrock-agentcore` also offers an `a2a-v1` extra for the newer A2A v1 protocol; it is mutually exclusive with the `a2a` extra, and Strands currently targets the 0.3 SDK.

---

## Calling A2A Agents

### From a Strands agent

`A2AAgent` wraps a remote A2A agent so it can be called directly, used as a tool, or placed in a Strands graph:

```python
from strands import Agent
from strands.agent.a2a_agent import A2AAgent

researcher = A2AAgent(
    endpoint="http://localhost:9000/",
    name="researcher",
    description="Researches a topic and returns sourced bullet points.",
)

writer = Agent(
    model="us.anthropic.claude-sonnet-5-5",
    system_prompt="Write clear briefings. Ask the researcher for facts first.",
    tools=[researcher],
)

writer("Write a one-paragraph briefing on grid-scale battery storage.")
```

For agents that should *discover* peers at run time, `strands-agents-tools` provides `A2AClientToolProvider(known_agent_urls=[...])`, which gives the model tools to list agent cards and send messages.

### With boto3 (AgentCore-hosted agent, SigV4)

```python
import json
import uuid

import boto3

agentcore = boto3.client("bedrock-agentcore", region_name="us-west-2")

request = {
    "jsonrpc": "2.0",
    "id": "req-001",
    "method": "message/send",
    "params": {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "Research heat-pump adoption in the UK"}],
            "messageId": str(uuid.uuid4()),
        }
    },
}

response = agentcore.invoke_agent_runtime(
    agentRuntimeArn="arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/research_agent-abc123",
    runtimeSessionId=str(uuid.uuid4()),
    payload=json.dumps(request),
)
result = json.loads(response["response"].read())
for artifact in result["result"].get("artifacts", []):
    for part in artifact["parts"]:
        if part["kind"] == "text":
            print(part["text"])
```

### With OAuth (any A2A client)

When the runtime has a JWT authorizer, generic A2A clients call the runtime's HTTPS URL directly with `Authorization: Bearer <token>` and an `X-Amzn-Bedrock-AgentCore-Runtime-Session-Id` header; this is how agents in other clouds or frameworks reach an AgentCore-hosted agent.

---

## Other Frameworks

| Framework | Serving over A2A | Calling A2A agents |
|-----------|------------------|--------------------|
| **Strands** | `A2AServer`, or `StrandsA2AExecutor` + `serve_a2a` on AgentCore | `A2AAgent`, `A2AClientToolProvider` |
| **Google ADK** | ADK's A2A integration (exposes an ADK agent as an A2A server) | ADK remote A2A agents |
| **LangGraph / LangChain** | Wrap the graph in an `a2a-sdk` `AgentExecutor`; AgentCore CLI scaffolds this | `a2a-sdk` client |
| **Anything else** | Implement `a2a.server.agent_execution.AgentExecutor` and serve with `a2a-sdk` (or `serve_a2a`) | `a2a-sdk` `ClientFactory` |

Check each framework's own documentation for current class names; A2A integrations outside Strands and AgentCore are not covered by this guide's verification.

---

## Security

- **Authenticate every call.** On AgentCore, A2A runtimes accept SigV4 (IAM) or OAuth 2.0 bearer tokens via a custom JWT authorizer. Advertise the scheme in the agent card's `securitySchemes`.
- **Treat remote agents as untrusted input.** Messages and artefacts from another agent can carry prompt injection; don't let them trigger privileged tools without checks, and keep remote agents out of your agent's system prompt.
- **Least privilege per agent.** Each AgentCore runtime has its own execution role; don't share one broad role across agents.
- **Pass identity explicitly.** Propagate end-user identity through AgentCore Identity (OAuth on-behalf-of / token exchange) rather than embedding user IDs in message text.
- **TLS only.** AgentCore endpoints are HTTPS; self-hosted A2A servers should be too.

---

## Best Practices

1. **Write good agent cards.** Clear `description` and `skills` are what a calling model uses to decide whether to delegate.
2. **Use tasks for long work.** Return a task quickly and stream updates (`message/stream`) or use push notifications instead of holding a request open.
3. **Keep `contextId` stable** for a conversation so the remote agent can keep context; AgentCore's session header gives each conversation its own microVM.
4. **Bound delegation.** Set timeouts on clients (`A2AAgent(timeout=...)`) and limits on graphs or swarms that include remote agents.
5. **Observe across agents.** Propagate W3C trace context (`traceparent`) so AgentCore Observability can stitch spans from both sides.
6. **Pin protocol versions.** A2A is still evolving (0.3 and v1 coexist); pin `a2a-sdk` with your framework and test before upgrading.

**Related guides:** [AgentCore](/amazon-bedrock-agents-guide/bedrock_agentcore_comprehensive_guide/) · [Strands SDK](/amazon-bedrock-agents-guide/bedrock_strands_sdk_guide/) · [Advanced multi-agent patterns](/amazon-bedrock-agents-guide/bedrock_agents_advanced_multi_agent_python/)

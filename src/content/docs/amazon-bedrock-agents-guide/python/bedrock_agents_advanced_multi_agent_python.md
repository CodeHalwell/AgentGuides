---
title: "Advanced Multi-Agent Patterns for Amazon Bedrock Agents (Python)"
description: "Hierarchical, dynamic and supervisor-based multi-agent patterns on Amazon Bedrock: Bedrock Agents multi-agent collaboration, Strands, A2A on AgentCore, and monitoring."
framework: "amazon-bedrock-agents"
language: "python"
sidebar:
  label: "Multi-agent patterns"
  order: 4
---

Patterns for building multi-agent systems on Amazon Bedrock, with two implementation routes:

- **Bedrock Agents multi-agent collaboration** (GA March 2025) — a managed supervisor with collaborator agents. Bedrock Agents (Agents Classic) closed to new customers on 30 July 2026 and is in maintenance mode, so this route is for existing accounts.
- **Code-defined agents** with Strands (or another framework) hosted on **AgentCore Runtime** — AWS's recommended route for new work.

> **Verified against:** `boto3` %%version=boto3%% and `strands-agents` %%version=strands-agents%% (checked %%versions-checked%%).

The examples reuse `AGENT_ROLE_ARN`, the model constants and the `prepare_and_alias` / `read_completion` helpers defined at the top of the [Recipes](/amazon-bedrock-agents-guide/python/bedrock_agents_recipes/) page.

## 1. Hierarchical Agent Systems

Agents form a tree: a top-level supervisor delegates to department supervisors, which delegate to specialists.

**Use Cases:**

*   Large-scale enterprise systems with multiple departments or business units.
*   Complex workflows that need several levels of delegation and control.

**Example: A Hierarchical Customer Support System**

```
                      ┌────────────────────┐
                      │  Global Supervisor │
                      └─────────┬──────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
      ┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
      │ Sales         │ │ Support       │ │ Billing       │
      │ Supervisor    │ │ Supervisor    │ │ Agent         │
      └───────┬───────┘ └───────┬───────┘ └───────────────┘
              │                 │
      ┌───────┴───────┐ ┌───────┴───────┐
      │               │ │               │
┌─────▼──┐      ┌─────▼──┐      ┌─────▼──┐      ┌────────┐
│ Product│      │ Lead   │      │ Tier 1 │      │ Tier 2 │
│ Agent  │      │ Gen    │      │ Agent  │      │ Agent  │
└────────┘      └────────┘      └────────┘      └────────┘
```

**Implementation with Bedrock Agents.** Build bottom-up: collaborators must be prepared and aliased before they can be associated with a supervisor (by alias ARN), and the supervisor must be saved before you associate collaborators with it.

```python
import boto3


class HierarchicalAgentSystem:
    def __init__(self, region="us-east-1"):
        self.agents = boto3.client("bedrock-agent", region_name=region)

    def _create_agent(self, name: str, instruction: str, model: str, collaboration: str = "DISABLED") -> str:
        return self.agents.create_agent(
            agentName=name,
            foundationModel=model,
            instruction=instruction,
            agentResourceRoleArn=AGENT_ROLE_ARN,
            agentCollaboration=collaboration,   # SUPERVISOR, SUPERVISOR_ROUTER or DISABLED
        )["agent"]["agentId"]

    def _alias_arn(self, agent_id: str) -> str:
        alias_id = prepare_and_alias(self.agents, agent_id)
        return self.agents.get_agent_alias(agentId=agent_id, agentAliasId=alias_id)["agentAlias"]["agentAliasArn"]

    def _associate(self, supervisor_id: str, collaborators: dict) -> str:
        """collaborators maps name -> (alias_arn, when-to-use instruction). Returns the supervisor's alias ARN."""
        for name, (alias_arn, instruction) in collaborators.items():
            self.agents.associate_agent_collaborator(
                agentId=supervisor_id,
                agentVersion="DRAFT",
                agentDescriptor={"aliasArn": alias_arn},
                collaboratorName=name,
                collaborationInstruction=instruction,
                relayConversationHistory="TO_COLLABORATOR",
            )
        return self._alias_arn(supervisor_id)

    def create_system(self) -> str:
        # Specialists
        product = self._alias_arn(self._create_agent("ProductAgent", "Provide product information.", HAIKU))
        lead_gen = self._alias_arn(self._create_agent("LeadGenAgent", "Qualify and record sales leads.", HAIKU))
        tier1 = self._alias_arn(self._create_agent("Tier1SupportAgent", "Handle basic support requests.", HAIKU))
        tier2 = self._alias_arn(self._create_agent("Tier2SupportAgent", "Handle complex support requests.", SONNET))
        billing = self._alias_arn(self._create_agent("BillingAgent", "Handle invoices and payments.", HAIKU))

        # Department supervisors (routers: pick one collaborator, lower latency)
        sales_sup = self._create_agent("SalesSupervisor", "Route sales requests.", SONNET, "SUPERVISOR_ROUTER")
        sales_sup_arn = self._associate(sales_sup, {
            "product": (product, "Questions about product features and specifications."),
            "lead_gen": (lead_gen, "Prospects who want a demo, a quote or to be contacted."),
        })
        support_sup = self._create_agent("SupportSupervisor", "Route support requests by complexity.", SONNET, "SUPERVISOR_ROUTER")
        support_sup_arn = self._associate(support_sup, {
            "tier1": (tier1, "Password resets, account status and simple how-to questions."),
            "tier2": (tier2, "Errors, outages, performance problems and integrations."),
        })

        # Global supervisor (coordinates and synthesises across departments)
        global_sup = self._create_agent(
            "GlobalSupervisor",
            "Top-level customer service coordinator. Delegate to the right department and combine answers.",
            SONNET,
            "SUPERVISOR",
        )
        self._associate(global_sup, {
            "sales": (sales_sup_arn, "Anything about buying, pricing or products."),
            "support": (support_sup_arn, "Anything technical or about using the product."),
            "billing": (billing, "Invoices, payments and refunds."),
        })
        return global_sup
```

Check the current [multi-agent collaboration documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-multi-agent-collaboration.html) and the service quotas for collaborator limits and nesting rules before designing deep hierarchies; flatter trees are also faster and cheaper. AWS also states that multi-agent collaboration has no direct equivalent in the managed harness: its migration guidance suggests exposing agents as MCP tools (the supervisor pattern) or writing multi-agent code on Runtime.

**The same hierarchy in Strands** is a few lines, because agents can be tools of other agents:

```python
from strands import Agent

product = Agent(model=HAIKU, name="product", description="Product features and specifications.")
lead_gen = Agent(model=HAIKU, name="lead_gen", description="Demos, quotes and contact requests.")
tier1 = Agent(model=HAIKU, name="tier1", description="Simple support questions.")
tier2 = Agent(model=SONNET, name="tier2", description="Errors, outages and integrations.")

sales = Agent(model=SONNET, name="sales", description="Sales department.", tools=[product, lead_gen])
support = Agent(model=SONNET, name="support", description="Support department.", tools=[tier1, tier2])

global_supervisor = Agent(
    model=SONNET,
    system_prompt="Delegate to the right department and combine their answers.",
    tools=[sales, support],
)
```

## 2. Dynamic Agent Composition

A coordinator selects specialists at run time from a registry, rather than having them fixed at build time.

**Use Cases:**

*   Handling ad-hoc or unpredictable requests.
*   Platforms where teams register new specialist agents independently.

**Implementation.** Keep a registry of deployed agents (here a dict; in production a DynamoDB table or the AgentCore registry), expose each as a tool, and let the model choose:

```python
import boto3
from strands import Agent, tool

runtime = boto3.client("bedrock-agent-runtime")

# name -> (agent_id, alias_id, capability description)
REGISTRY = {
    "weather": ("AGENT_ID_1", "ALIAS_ID_1", "Weather forecasts for a location."),
    "news": ("AGENT_ID_2", "ALIAS_ID_2", "Latest news headlines on a topic."),
    "translation": ("AGENT_ID_3", "ALIAS_ID_3", "Translates text between languages."),
}


@tool
def list_specialists() -> dict:
    """List the specialist agents that are available and what each one does."""
    return {name: desc for name, (_, _, desc) in REGISTRY.items()}


@tool
def ask_specialist(name: str, request: str, session_id: str) -> str:
    """Send a request to one specialist agent and return its answer.

    Args:
        name: Specialist name from list_specialists.
        request: What to ask the specialist.
        session_id: Conversation session ID to reuse across calls.
    """
    agent_id, alias_id, _ = REGISTRY[name]
    response = runtime.invoke_agent(agentId=agent_id, agentAliasId=alias_id,
                                    sessionId=session_id, inputText=request)
    return read_completion(response)


coordinator = Agent(
    model=SONNET,
    system_prompt="Check which specialists exist, call the ones you need, then write one combined answer.",
    tools=[list_specialists, ask_specialist],
)

coordinator("Translate today's top AI headline into French. Session: s-123")
```

## 3. Agent-to-Agent (A2A) Communication with the A2A Protocol

[A2A](https://a2a-protocol.org/) is an open protocol (introduced by Google in April 2025 and now a Linux Foundation project) that lets agents built in different frameworks discover each other through *agent cards* and exchange tasks and messages. Bedrock Agents (Classic) does not speak A2A; `invoke_agent` has no A2A parameters. On AWS, A2A is supported by **AgentCore Runtime** (`serverProtocol="A2A"`), and Strands provides both an A2A server executor and a client (`A2AAgent`).

**Example:** a Strands coordinator calling a remote agent — which could be built with LangGraph, Google ADK or the OpenAI Agents SDK — over A2A:

```python
from strands import Agent
from strands.agent.a2a_agent import A2AAgent

summariser = A2AAgent(
    endpoint="https://summariser.example.com/",   # any A2A-compliant agent
    name="summariser",
    description="Summarises news articles.",
)

coordinator = Agent(model=SONNET, tools=[summariser])
coordinator("Summarise the latest news about AI regulation in the EU.")
```

The server side, deployment to AgentCore and authentication are covered in the [A2A guide](/amazon-bedrock-agents-guide/python/bedrock_a2a_protocol_guide/).

---

## 4. Multi-Agent Collaboration (GA March 2025)

Bedrock Agents multi-agent collaboration has two supervisor modes, set with `agentCollaboration` on the supervisor:

| Mode | Behaviour | Use when |
|------|-----------|----------|
| `SUPERVISOR` | Plans, calls one or more collaborators (in parallel where possible) and synthesises their answers | Requests that span several specialists |
| `SUPERVISOR_ROUTER` | Routes each request to the single most suitable collaborator, which answers directly; falls back to full supervisor behaviour when the request is ambiguous | Lower latency for clearly separable requests |

There are no keyword- or intent-rule routing settings: routing is driven by the supervisor's instructions and each collaborator's `collaborationInstruction`.

### Supervisor Mode

```python
import boto3

agents = boto3.client("bedrock-agent")
runtime = boto3.client("bedrock-agent-runtime")

# 1. Collaborators (tools are attached as action groups, not passed to create_agent)
sales_id = agents.create_agent(
    agentName="SalesSpecialist",
    foundationModel=SONNET,
    agentResourceRoleArn=AGENT_ROLE_ARN,
    instruction="You are a sales specialist. Handle product inquiries, pricing and quotes.",
)["agent"]["agentId"]
support_id = agents.create_agent(
    agentName="SupportSpecialist",
    foundationModel=SONNET,
    agentResourceRoleArn=AGENT_ROLE_ARN,
    instruction="You are a technical support specialist. Handle troubleshooting and technical issues.",
)["agent"]["agentId"]
# ... create_agent_action_group(...) for each collaborator's tools ...

collaborators = {}
for name, agent_id in {"sales_specialist": sales_id, "support_specialist": support_id}.items():
    alias_id = prepare_and_alias(agents, agent_id)
    collaborators[name] = agents.get_agent_alias(agentId=agent_id, agentAliasId=alias_id)["agentAlias"]["agentAliasArn"]

# 2. Supervisor
supervisor_id = agents.create_agent(
    agentName="CustomerServiceSupervisor",
    foundationModel=SONNET,
    agentResourceRoleArn=AGENT_ROLE_ARN,
    instruction="You coordinate customer service. Delegate to the sales or support specialist as needed and combine their answers.",
    agentCollaboration="SUPERVISOR",
)["agent"]["agentId"]

agents.associate_agent_collaborator(
    agentId=supervisor_id, agentVersion="DRAFT",
    agentDescriptor={"aliasArn": collaborators["sales_specialist"]},
    collaboratorName="sales_specialist",
    collaborationInstruction="Sales, pricing and product information.",
    relayConversationHistory="TO_COLLABORATOR",
)
agents.associate_agent_collaborator(
    agentId=supervisor_id, agentVersion="DRAFT",
    agentDescriptor={"aliasArn": collaborators["support_specialist"]},
    collaboratorName="support_specialist",
    collaborationInstruction="Technical support and troubleshooting.",
    relayConversationHistory="TO_COLLABORATOR",
)
supervisor_alias = prepare_and_alias(agents, supervisor_id)

# 3. Invoke
response = runtime.invoke_agent(
    agentId=supervisor_id,
    agentAliasId=supervisor_alias,
    sessionId="session-123",
    inputText="I need pricing for the enterprise plan and help with a technical issue",
)
print(read_completion(response))
```

### Supervisor with Routing Mode

Identical, except for the supervisor's collaboration mode:

```python
router_id = agents.create_agent(
    agentName="SmartSupervisor",
    foundationModel=HAIKU,   # routing is a light task
    agentResourceRoleArn=AGENT_ROLE_ARN,
    instruction="Send each request to the one specialist best placed to answer it.",
    agentCollaboration="SUPERVISOR_ROUTER",
)["agent"]["agentId"]
# associate_agent_collaborator(...) as above, then prepare_and_alias(agents, router_id)
```

### Complete Multi-Agent System Example

Invoking a supervisor and inspecting which collaborators it used, from the trace:

```python
import uuid

import boto3


class ProductionMultiAgentSystem:
    def __init__(self, supervisor_id: str, supervisor_alias_id: str, region="us-east-1"):
        self.runtime = boto3.client("bedrock-agent-runtime", region_name=region)
        self.supervisor_id = supervisor_id
        self.supervisor_alias_id = supervisor_alias_id

    def invoke_system(self, user_request: str, session_id: str | None = None) -> dict:
        session_id = session_id or str(uuid.uuid4())
        response = self.runtime.invoke_agent(
            agentId=self.supervisor_id,
            agentAliasId=self.supervisor_alias_id,
            sessionId=session_id,
            inputText=user_request,
            enableTrace=True,   # traces include collaborator invocations
        )

        completion, collaborators_used = "", set()
        for event in response["completion"]:
            if "chunk" in event:
                completion += event["chunk"]["bytes"].decode()
            elif "trace" in event:
                trace = event["trace"]
                if trace.get("collaboratorName"):
                    collaborators_used.add(trace["collaboratorName"])

        return {"response": completion, "collaborators": sorted(collaborators_used), "session_id": session_id}


system = ProductionMultiAgentSystem("SUPERVISOR_AGENT_ID", "SUPERVISOR_ALIAS_ID")
result = system.invoke_system("I need help with a billing issue and also want to know about enterprise pricing")
print(result["response"])
print("Collaborators used:", result["collaborators"])
```

### Monitoring Multi-Agent Systems

Bedrock Agents publishes runtime metrics such as `InvocationCount`, `TotalTime`, `TTFT`, `InvocationThrottles`, `ModelLatency`, `InputTokenCount` and `OutputTokenCount`, with dimensions `Operation`, `ModelId` and `AgentAliasArn`. Discover the exact dimension sets in your account rather than hard-coding them:

```python
from datetime import datetime, timedelta, timezone

import boto3

cloudwatch = boto3.client("cloudwatch")


def monitor_agent_alias(agent_alias_arn: str, hours: int = 1) -> dict:
    end = datetime.now(timezone.utc)
    results = {}
    for metric in ("InvocationCount", "TotalTime", "InvocationThrottles"):
        for m in cloudwatch.list_metrics(Namespace="AWS/Bedrock/Agents", MetricName=metric)["Metrics"]:
            dims = {d["Name"]: d["Value"] for d in m["Dimensions"]}
            if dims.get("AgentAliasArn") != agent_alias_arn:
                continue
            stats = cloudwatch.get_metric_statistics(
                Namespace="AWS/Bedrock/Agents",
                MetricName=metric,
                Dimensions=m["Dimensions"],
                StartTime=end - timedelta(hours=hours),
                EndTime=end,
                Period=300,
                Statistics=["Sum", "Average"],
            )
            results.setdefault(metric, []).extend(stats["Datapoints"])
    return results
```

For per-collaborator detail, enable traces (as above) or model-invocation logging; for code-defined agents on AgentCore, use AgentCore Observability's OpenTelemetry traces.

See also: the [comprehensive guide](/amazon-bedrock-agents-guide/python/bedrock_agents_comprehensive_guide/), the [Strands multi-agent guide](/strands-agents-guide/python/strands_multi_agent/) and the [Recipes](/amazon-bedrock-agents-guide/python/bedrock_agents_recipes/#multi-agent-customer-service-system).

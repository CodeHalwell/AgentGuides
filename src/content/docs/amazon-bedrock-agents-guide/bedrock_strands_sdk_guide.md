---
title: "AWS Strands Agents SDK: Comprehensive Technical Guide"
description: "Technical reference for Strands Agents, the open-source, model-driven agent SDK from AWS: agents, tools, multi-agent patterns (agents as tools, swarms, graphs, workflows), sessions, memory and deployment to AgentCore."
framework: "amazon-bedrock-agents"
language: "python"
---

> TypeScript version of this: [Strands Agents TypeScript SDK](/amazon-bedrock-agents-guide/typescript/strands_typescript_guide/).

Strands Agents is an open-source (Apache 2.0) SDK from AWS for building agents with a model-driven loop: you give an agent a model, a system prompt and tools, and the model decides which tools to call. It runs on Amazon Bedrock by default but also supports Anthropic, OpenAI, Gemini, Ollama, LiteLLM and others. There are Python and TypeScript SDKs; this guide covers Python.

> **Verified against:** `strands-agents` %%version=strands-agents%% (checked %%versions-checked%%). Every import, class and parameter below was checked against that installed package. `strands-agents-tools` is versioned separately.

**Status:** Open source, generally available (1.0 released July 2025)
**Language:** Python 3.10+ (TypeScript SDK: `@strands-agents/sdk`)
**GitHub:** https://github.com/strands-agents
**Documentation:** https://strandsagents.com
**License:** Apache 2.0

---

## Table of Contents

1. [Introduction to Strands](#introduction-to-strands)
2. [Installation and Setup](#installation-and-setup)
3. [Core Concepts](#core-concepts)
4. [Collaboration Pattern 1: Agents as Tools](#collaboration-pattern-1-agents-as-tools)
5. [Collaboration Pattern 2: Swarms](#collaboration-pattern-2-swarms)
6. [Collaboration Pattern 3: Agent Graphs](#collaboration-pattern-3-agent-graphs)
7. [Collaboration Pattern 4: Workflows](#collaboration-pattern-4-workflows)
8. [Tool Definition and Integration](#tool-definition-and-integration)
9. [Memory and State Management](#memory-and-state-management)
10. [AWS Integration](#aws-integration)
11. [Best Practices](#best-practices)
12. [Production Deployment](#production-deployment)

---

## Introduction to Strands

Strands keeps the agent loop small and lets the model do the planning. The SDK supplies the loop, tool execution, conversation management, session persistence, hooks, multi-agent primitives and OpenTelemetry tracing.

### Key Features

| Feature | Description |
|---------|-------------|
| **Model-driven loop** | The model chooses tools and when to stop; no hand-written state machine required |
| **`@tool` decorator** | Plain Python functions become tools; type hints and docstrings become the schema |
| **Multi-agent primitives** | Agents as tools, `Swarm`, `GraphBuilder` graphs, and the `workflow` tool |
| **Model providers** | Bedrock (default), Anthropic, OpenAI, Gemini, Ollama, LiteLLM, SageMaker, Mistral, Writer and more |
| **MCP** | `MCPClient` loads tools from any MCP server |
| **Sessions and memory** | File, S3 and AgentCore Memory session managers; sliding-window and summarising conversation managers |
| **Observability** | OpenTelemetry spans for agent, model and tool calls |

### Use Cases

- **Customer service**: a supervisor routing to specialist agents
- **Data analysis**: agents with code execution and data tools
- **Content pipelines**: graphs of drafting, review and formatting agents
- **Research**: swarms of agents handing work to each other

---

## Installation and Setup

### Installation

```bash
pip install "strands-agents==%%version=strands-agents%%" strands-agents-tools

# Optional extras, for example
pip install "strands-agents[openai,a2a,otel]==%%version=strands-agents%%"
```

### Prerequisites

```bash
python --version   # 3.10 or later

# Bedrock is the default provider: configure AWS credentials and a region
aws configure
export AWS_REGION=us-west-2
```

Enable access to the Claude models you plan to use in the Bedrock console (Model access) before the first call.

### Quick Verification

```python
import importlib.metadata

from strands import Agent, tool

print("strands-agents", importlib.metadata.version("strands-agents"))


@tool
def hello(name: str) -> str:
    """Say hello to someone.

    Args:
        name: Person to greet.
    """
    return f"Hello, {name}!"


agent = Agent(
    model="us.anthropic.claude-sonnet-5-5",  # Bedrock cross-Region inference profile ID
    tools=[hello],
)

result = agent("Say hello to Alice")
print(result)  # AgentResult; str() gives the final text
```

If you omit `model`, Strands uses its default Bedrock model. Pin the model explicitly in anything you deploy so behaviour doesn't change when that default moves.

**Bedrock model IDs.** Current Claude models on Bedrock use IDs such as `anthropic.claude-sonnet-5-5`, `anthropic.claude-opus-5-5` and `anthropic.claude-haiku-5-5`; prefix them with `us.`, `eu.`, `jp.`, `au.` or `global.` to use a cross-Region inference profile. Check the [Bedrock model cards](https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html) for the profiles available in your Region.

---

## Core Concepts

### Agent Class

`Agent` takes keyword arguments; the common ones are:

```python
from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.models import BedrockModel, CacheConfig

model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-5-5",
    region_name="us-west-2",
    max_tokens=4096,
    cache_config=CacheConfig(strategy="auto"),  # Bedrock prompt caching
)

agent = Agent(
    model=model,                         # a Model instance or a model ID string
    name="Analyst",                      # used when the agent is a tool or swarm member
    description="Answers questions about sales data",
    system_prompt="You are a careful sales analyst.",
    tools=[],                            # @tool functions, modules, MCP clients, other agents
    conversation_manager=SlidingWindowConversationManager(window_size=40),
    state={"tenant": "acme"},            # JSON-serialisable agent state (not sent to the model)
)

result = agent("What drove Q4 revenue?")         # synchronous
print(result.stop_reason, result.metrics.accumulated_usage)
```

Async and streaming:

```python
import asyncio


async def main():
    async for event in agent.stream_async("Summarise Q4 in three bullets"):
        if "data" in event:
            print(event["data"], end="", flush=True)

asyncio.run(main())
```

### Tool Decorator

```python
from strands import tool


@tool
def calculate_sum(numbers: list[float]) -> float:
    """Calculate the sum of a list of numbers.

    Args:
        numbers: Numbers to add up.
    """
    return sum(numbers)


@tool
def search_products(query: str, limit: int = 10) -> list[dict]:
    """Search the product catalogue.

    Args:
        query: Free-text search query.
        limit: Maximum number of results.
    """
    return [{"sku": "A-1", "name": "Widget"}][:limit]
```

The function name becomes the tool name, the docstring becomes its description, and the `Args:` section documents each parameter.

### Context Management

There is no separate `Context` object. Pass per-request data with `invocation_state` and read it — together with the calling agent — from a tool by asking for a `ToolContext`:

```python
from strands import Agent, ToolContext, tool


@tool(context=True)
def account_status(tool_context: ToolContext) -> str:
    """Look up the current customer's account status."""
    customer_id = tool_context.invocation_state["customer_id"]
    tier = tool_context.agent.state.get("tier")
    return f"Customer {customer_id} ({tier}) is ACTIVE"


agent = Agent(model="us.anthropic.claude-sonnet-5-5", tools=[account_status], state={"tier": "premium"})
agent("Check my account status", invocation_state={"customer_id": "C-456"})
```

`invocation_state` is not shown to the model, so it is the right place for user IDs, credentials handles and request metadata.

---

## Collaboration Pattern 1: Agents as Tools

A supervisor agent calls specialist agents as if they were tools.

### Architecture

```
┌─────────────────────────────────────┐
│      Supervisor Agent               │
│   (Orchestrates and delegates)      │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────────┐
       │                   │
┌──────▼──────┐     ┌──────▼──────┐
│  Specialist │     │  Specialist │
│  Agent 1    │     │  Agent 2    │
│  (Tool)     │     │  (Tool)     │
└─────────────┘     └─────────────┘
```

### Complete Example

```python
from strands import Agent, tool

SONNET = "us.anthropic.claude-sonnet-5-5"
HAIKU = "us.anthropic.claude-haiku-5-5"


@tool
def get_invoice(invoice_id: str) -> dict:
    """Fetch an invoice.

    Args:
        invoice_id: Invoice identifier.
    """
    return {"invoice_id": invoice_id, "amount": 1200.0, "status": "OVERDUE"}


@tool
def get_plan_details(plan: str) -> dict:
    """Return pricing and features for a plan.

    Args:
        plan: Plan name, e.g. "enterprise".
    """
    return {"plan": plan, "price_per_seat": 45, "sso": True}


sales_agent = Agent(
    model=SONNET,
    name="sales_specialist",
    description="Answers questions about plans, pricing and purchases.",
    system_prompt="You are a sales specialist. Be accurate and concise.",
    tools=[get_plan_details],
)

billing_agent = Agent(
    model=HAIKU,
    name="billing_specialist",
    description="Answers questions about invoices and payments.",
    system_prompt="You are a billing specialist. Be precise about amounts and dates.",
    tools=[get_invoice],
)

supervisor = Agent(
    model=SONNET,
    system_prompt=(
        "You are a customer service supervisor. Delegate plan and pricing questions to "
        "sales_specialist and invoice questions to billing_specialist, then combine their answers."
    ),
    tools=[
        sales_agent.as_tool(),     # name/description default to the agent's own
        billing_agent.as_tool(),
    ],
)

print(supervisor("Invoice INV-77 looks wrong, and what does the enterprise plan cost?"))
```

`Agent.as_tool()` accepts `name`, `description` and `preserve_context` (keep the specialist's conversation between calls; off by default). You can also put an `Agent` directly in `tools=[...]`; it is wrapped the same way.

### Advanced Pattern: Multi-Level Hierarchy

```python
tier1 = Agent(model=HAIKU, name="tier1_support", description="Basic how-to questions.")
tier2 = Agent(model=SONNET, name="tier2_support", description="Configuration and integration issues.")
tier3 = Agent(model="us.anthropic.claude-opus-5-5", name="tier3_support", description="Complex incidents and root-cause analysis.")

support_lead = Agent(
    model=SONNET,
    name="support_lead",
    description="Handles all technical support requests.",
    system_prompt="Route each request to the lowest tier that can resolve it.",
    tools=[tier1, tier2, tier3],
)

front_desk = Agent(
    model=SONNET,
    system_prompt="Route requests to the right department.",
    tools=[support_lead, sales_agent],
)

front_desk("Our enterprise deployment is seeing high latency since yesterday's upgrade.")
```

---

## Collaboration Pattern 2: Swarms

A `Swarm` is a team of agents that hand work to one another. Each member automatically receives a `handoff_to_agent` tool and a shared view of the task and of what other agents have done; the swarm ends when an agent finishes without handing off, or when a limit is hit. It is collaborative and sequential — one agent active at a time — rather than "run everyone in parallel and vote".

### Architecture

```
            ┌──────────────┐  handoff   ┌──────────────┐
  task ───▶ │  researcher  │──────────▶ │   analyst    │
            └──────────────┘            └──────┬───────┘
                    ▲                          │ handoff
                    │       handoff     ┌──────▼───────┐
                    └────────────────── │   reviewer   │ ──▶ final answer
                                        └──────────────┘
```

### Complete Example

```python
from strands import Agent
from strands.multiagent import Swarm

SONNET = "us.anthropic.claude-sonnet-5-5"

financial = Agent(
    model=SONNET,
    name="financial_analyst",
    system_prompt="Assess costs, ROI and budget impact. Hand off to technical_analyst when done.",
)
technical = Agent(
    model=SONNET,
    name="technical_analyst",
    system_prompt="Assess architecture, scalability and delivery risk. Hand off to risk_analyst when done.",
)
risk = Agent(
    model=SONNET,
    name="risk_analyst",
    system_prompt="Assess security and compliance, then write the final balanced recommendation.",
)

swarm = Swarm(
    [financial, technical, risk],
    entry_point=financial,
    max_handoffs=10,
    max_iterations=15,
    execution_timeout=900.0,   # seconds, whole swarm
    node_timeout=300.0,        # seconds, per agent
)

result = swarm(
    "Proposal: migrate our monolith to microservices on EKS. "
    "$50k implementation, $10k/month to run, six months."
)

print(result.status)                                   # Status.COMPLETED, FAILED, ...
print([node.node_id for node in result.node_history])  # who handled it, in order
for node_id, node_result in result.results.items():
    print(node_id, node_result.result)
```

Swarm members cannot have their own `session_manager`; pass one to the `Swarm` instead if you need to persist swarm state.

### Parallel opinions and voting

If you want several independent opinions combined — the old "aggregator" idea — you don't need a swarm. Run the agents concurrently yourself and aggregate:

```python
import asyncio
from collections import Counter

from strands import Agent

classifiers = [
    Agent(model="us.anthropic.claude-haiku-5-5",
          system_prompt="Classify the request as BILLING, SALES or SUPPORT. Reply with one word.",
          callback_handler=None)          # silence console streaming
    for _ in range(5)
]


async def vote(text: str) -> dict:
    results = await asyncio.gather(*(a.invoke_async(text) for a in classifiers))
    votes = Counter(str(r).strip().upper() for r in results)
    label, count = votes.most_common(1)[0]
    return {"decision": label, "confidence": count / len(classifiers), "votes": dict(votes)}

print(asyncio.run(vote("I was charged twice this month")))
```

---

## Collaboration Pattern 3: Agent Graphs

`GraphBuilder` builds a directed graph whose nodes are agents (or other multi-agent systems). Edges can carry conditions, nodes with no unmet dependencies run in parallel, and cycles are allowed when bounded.

### Architecture

```
┌─────────┐     ┌──────────┐     ┌─────────┐     ┌────────┐
│ collect │────▶│ validate │────▶│ analyse │────▶│ report │
└─────────┘     └────┬─────┘     └─────────┘     └────────┘
     ▲               │ (validation failed)
     └───────────────┘
```

### Complete Example

```python
from strands import Agent
from strands.multiagent import GraphBuilder
from strands.multiagent.graph import GraphState

HAIKU = "us.anthropic.claude-haiku-5-5"
SONNET = "us.anthropic.claude-sonnet-5-5"

collector = Agent(model=HAIKU, name="collector",
                  system_prompt="Collect and structure the raw data described in the task.")
validator = Agent(model=HAIKU, name="validator",
                  system_prompt="Check the data for completeness. Start your reply with PASS or FAIL.")
analyst = Agent(model=SONNET, name="analyst",
                system_prompt="Perform the statistical analysis and identify trends.")
reporter = Agent(model=SONNET, name="reporter",
                 system_prompt="Write a concise report with the key insights.")


def text_of(state: GraphState, node_id: str) -> str:
    node = state.results.get(node_id)
    return str(node.result) if node else ""


def validation_passed(state: GraphState) -> bool:
    return text_of(state, "validate").strip().upper().startswith("PASS")


def validation_failed(state: GraphState) -> bool:
    return not validation_passed(state)


builder = GraphBuilder()
builder.add_node(collector, "collect")
builder.add_node(validator, "validate")
builder.add_node(analyst, "analyse")
builder.add_node(reporter, "report")

builder.add_edge("collect", "validate")
builder.add_edge("validate", "analyse", condition=validation_passed)
builder.add_edge("validate", "collect", condition=validation_failed)   # retry loop
builder.add_edge("analyse", "report")

builder.set_entry_point("collect")
builder.reset_on_revisit(True)          # re-run nodes cleanly when the loop revisits them
builder.set_max_node_executions(10)     # bound the loop
builder.set_execution_timeout(900)

graph = builder.build()

result = graph("Trend analysis of Q4 sales from s3://sales-bucket/q4.csv")
print(result.status, result.execution_order)
print(result.results["report"].result)
```

### State Management in Graphs

Each node receives the original task plus the outputs of the nodes that feed it. Conditions read the shared `GraphState` (`state.results`, `state.completed_nodes`, `state.execution_count`), as in `validation_passed` above. For data that should not go through the model, pass `invocation_state` when calling the graph; it reaches every node's tools via `ToolContext`.

### Parallel Execution in Graphs

Fan-out is expressed with edges; nodes whose dependencies are satisfied run concurrently:

```python
builder = GraphBuilder()
builder.add_node(Agent(model=HAIKU, name="prepare"), "prepare")
builder.add_node(Agent(model=HAIKU, name="sentiment"), "sentiment")
builder.add_node(Agent(model=HAIKU, name="entities"), "entities")
builder.add_node(Agent(model=HAIKU, name="category"), "category")
builder.add_node(Agent(model=SONNET, name="synthesise"), "synthesise")

for branch in ("sentiment", "entities", "category"):
    builder.add_edge("prepare", branch)      # fan out
    builder.add_edge(branch, "synthesise")   # fan in

builder.set_entry_point("prepare")
builder.set_execution_timeout(600)
result = builder.build()("Customer email: 'The new dashboard is great but exports keep failing.'")
```

---

## Collaboration Pattern 4: Workflows

For a fixed pipeline you know in advance, a graph without conditional edges is usually the clearest option. When you want the *model* to plan and run a multi-task workflow — with dependencies, priorities and per-task models — use the `workflow` tool from `strands-agents-tools`:

```python
from strands import Agent
from strands_tools import workflow

coordinator = Agent(model="us.anthropic.claude-sonnet-5-5", tools=[workflow])

coordinator.tool.workflow(
    action="create",
    workflow_id="document_processing",
    tasks=[
        {
            "task_id": "extract",
            "description": "Extract invoice fields from the document text provided.",
            "priority": 5,
        },
        {
            "task_id": "classify",
            "description": "Classify the document as invoice, contract, report or other.",
            "dependencies": ["extract"],
            "model_provider": "bedrock",
            "model_settings": {"model_id": "us.anthropic.claude-haiku-5-5"},
        },
        {
            "task_id": "validate",
            "description": "Validate the extracted fields for completeness.",
            "dependencies": ["extract"],
        },
    ],
)

coordinator.tool.workflow(action="start", workflow_id="document_processing")
print(coordinator.tool.workflow(action="status", workflow_id="document_processing"))
```

`agent.tool.<name>(...)` calls a tool directly, without going through the model — useful for deterministic set-up steps like these.

Durable, long-running business processes with human approval steps and S3/EventBridge triggers belong in AWS Step Functions, with each step invoking an agent (for example one hosted on AgentCore Runtime — see [AgentCore integration patterns](/amazon-bedrock-agents-guide/bedrock_agentcore_comprehensive_guide/#integration-patterns)). For pauses for human input inside an agent, Strands supports interrupts raised from hooks and tools.

### Error Handling in Workflows

Graphs and swarms report failures in their results rather than raising for every node error:

```python
from strands.multiagent.base import Status

result = graph("...")
if result.status != Status.COMPLETED:
    for node_id in result.failed_nodes:
        print("failed:", node_id)
```

Bound every multi-agent run with timeouts and execution limits (shown above) so a misbehaving loop fails fast.

---

## Tool Definition and Integration

### Creating Tools

```python
from typing import Optional

import boto3
from strands import tool

products = boto3.resource("dynamodb").Table("Products")


@tool
def get_product(sku: str, include_stock: Optional[bool] = False) -> dict:
    """Look up a product by SKU.

    Args:
        sku: Product SKU.
        include_stock: Also return the current stock level.
    """
    item = products.get_item(Key={"sku": sku}).get("Item", {})
    if not include_stock:
        item.pop("stock", None)
    return item
```

Tools can be sync or async functions. Raise an exception to report a failure; Strands returns it to the model as an error result so it can recover.

### Tools from MCP servers

```python
from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.tools.mcp import MCPClient

docs_server = MCPClient(lambda: stdio_client(
    StdioServerParameters(command="uvx", args=["awslabs.aws-documentation-mcp-server@latest"])
))

agent = Agent(model="us.anthropic.claude-sonnet-5-5", tools=[docs_server])  # client is managed by the agent
agent("How do I configure VPC networking for AgentCore Runtime?")
```

For HTTP MCP servers (including AgentCore Gateway), pass `url=` (and `headers=` or `auth=`) to `MCPClient` instead of a transport callable.

### AWS Service Integration Tools

```python
import json

import boto3
from strands import tool


@tool
def invoke_lambda(function_name: str, payload: dict) -> dict:
    """Invoke a Lambda function synchronously and return its JSON result.

    Args:
        function_name: Lambda function name or ARN.
        payload: JSON payload to send.
    """
    response = boto3.client("lambda").invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload),
    )
    return json.loads(response["Payload"].read())


@tool
def send_sns_notification(topic_arn: str, subject: str, message: str) -> str:
    """Publish a notification to an SNS topic.

    Args:
        topic_arn: SNS topic ARN.
        subject: Message subject.
        message: Message body.
    """
    return boto3.client("sns").publish(TopicArn=topic_arn, Subject=subject, Message=message)["MessageId"]
```

`strands-agents-tools` also ships a general `use_aws` tool; prefer narrow, purpose-built tools like the ones above in production so the agent's permissions stay obvious.

---

## Memory and State Management

### Conversation Memory

Within one `Agent` object, the conversation is kept in `agent.messages` and trimmed by the conversation manager. To persist it across processes, give the agent a session manager:

```python
from strands import Agent
from strands.session import S3SessionManager

session = S3SessionManager(session_id="session-123", bucket="agent-sessions", prefix="support/")

agent = Agent(
    model="us.anthropic.claude-sonnet-5-5",
    session_manager=session,
)

agent("My name is John and I need help with order #12345")
# Later, in another process, a new Agent with the same session_id resumes the conversation
agent("What's the status of my order?")
```

`FileSessionManager(session_id=..., storage_dir=...)` does the same on local disk.

### AgentCore Memory

For managed short- and long-term memory, use the session manager shipped in `bedrock-agentcore`:

```python
# pip install bedrock-agentcore
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
from strands import Agent

config = AgentCoreMemoryConfig(
    memory_id="CustomerMemory-abc123",
    session_id="session-123",
    actor_id="customer-42",
    retrieval_config={"/facts/{actorId}/": RetrievalConfig(top_k=5, relevance_score=0.3)},
)

agent = Agent(
    model="us.anthropic.claude-sonnet-5-5",
    session_manager=AgentCoreMemorySessionManager(config, region_name="us-west-2"),
)
```

Turns are written to AgentCore Memory as events; long-term records extracted by the memory's strategies are retrieved and added to context on each turn.

### Persistent State

`agent.state` is a key-value store for application data that is not sent to the model. It is saved and restored by the session manager along with the conversation:

```python
agent = Agent(
    model="us.anthropic.claude-sonnet-5-5",
    state={"onboarding_step": 1},
    session_manager=S3SessionManager(session_id="onboard-john", bucket="agent-sessions"),
)

agent.state.set("user_info", {"name": "John", "email": "john@example.com"})
step = agent.state.get("onboarding_step")
```

### Long conversations

```python
from strands.agent.conversation_manager import SummarizingConversationManager

agent = Agent(
    model="us.anthropic.claude-sonnet-5-5",
    conversation_manager=SummarizingConversationManager(
        summary_ratio=0.3,             # summarise the oldest 30% when the context fills
        preserve_recent_messages=10,
    ),
)
```

---

## AWS Integration

### S3 and other services

There is no special "integration" object: an agent reaches AWS through tools, using the IAM permissions of whatever runs it. A read-only S3 tool:

```python
import boto3
from strands import Agent, tool

s3 = boto3.client("s3")


@tool
def read_s3_text(bucket: str, key: str, max_bytes: int = 200_000) -> str:
    """Read a UTF-8 text object from S3.

    Args:
        bucket: Bucket name.
        key: Object key.
        max_bytes: Maximum number of bytes to read.
    """
    body = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{max_bytes - 1}")["Body"]
    return body.read().decode("utf-8", errors="replace")


analyst = Agent(model="us.anthropic.claude-sonnet-5-5", tools=[read_s3_text])
analyst("Summarise s3://documents/report.txt")
```

### Bedrock Guardrails

`BedrockModel` applies a guardrail to every model call:

```python
from strands.models import BedrockModel

model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-5-5",
    guardrail_id="gr-abc123",
    guardrail_version="1",
    guardrail_trace="enabled",
    guardrail_redact_output=True,
)
```

### Event-driven agents

To run an agent from EventBridge or S3 events, host it in Lambda (short tasks) or AgentCore Runtime (long or stateful tasks) and point the event rule at that function. See [Production Deployment](#production-deployment).

---

## Best Practices

### 1. Agent Design

```python
# Good: focused agent, few well-described tools
product_specialist = Agent(
    model="us.anthropic.claude-sonnet-5-5",
    name="product_specialist",
    system_prompt="You are an expert on our product line. Only discuss features, pricing and comparisons.",
    tools=[get_plan_details],
)

# Avoid: one agent with dozens of loosely related tools — split into specialists
```

### 2. Error Handling

```python
import logging

from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException

log = logging.getLogger(__name__)


def run_agent_safely(agent, text: str) -> str:
    try:
        return str(agent(text))
    except ModelThrottledException:
        log.warning("Bedrock throttled the request after retries")
        raise
    except ContextWindowOverflowException:
        log.warning("Conversation too long; consider SummarizingConversationManager")
        raise
```

Strands already retries throttled model calls (configurable with `retry_strategy=ModelRetryStrategy(...)` on the agent), so avoid wrapping every call in another blanket retry loop.

### 3. Cost Optimisation

```python
from strands import Agent
from strands.models import BedrockModel, CacheConfig

router = Agent(model=BedrockModel(model_id="us.anthropic.claude-haiku-5-5", max_tokens=256),
               system_prompt="Reply SIMPLE or COMPLEX only.", callback_handler=None)
simple = Agent(model="us.anthropic.claude-haiku-5-5")
complex_ = Agent(model=BedrockModel(model_id="us.anthropic.claude-sonnet-5-5",
                                    cache_config=CacheConfig(strategy="auto")))


def route_request(text: str) -> str:
    target = complex_ if "COMPLEX" in str(router(text)).upper() else simple
    return str(target(text))
```

Prompt caching (`cache_config=CacheConfig(...)`; the older `cache_prompt` and `cache_tools` options are deprecated) is usually the cheapest saving for agents with long system prompts and many tools.

### 4. Hooks for cross-cutting concerns

```python
from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry


class ToolAudit(HookProvider):
    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeToolCallEvent, self.log_tool)

    def log_tool(self, event: BeforeToolCallEvent) -> None:
        print("tool:", event.tool_use["name"], event.tool_use["input"])


agent = Agent(model="us.anthropic.claude-sonnet-5-5", hooks=[ToolAudit()])
```

---

## Production Deployment

### Deploying to AgentCore Runtime

The recommended home for Strands agents on AWS is Amazon Bedrock AgentCore Runtime:

```python
from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent

app = BedrockAgentCoreApp()
agent = Agent(model="us.anthropic.claude-sonnet-5-5", tools=[get_plan_details])


@app.entrypoint
async def invoke(payload, context):
    async for event in agent.stream_async(payload["prompt"]):
        if "data" in event:
            yield event["data"]


if __name__ == "__main__":
    app.run()
```

Deploy with `agentcore create` / `agentcore deploy` (npm `@aws/agentcore`). Full details are in the [AgentCore guide](/amazon-bedrock-agents-guide/bedrock_agentcore_comprehensive_guide/#getting-started).

### Deploying to Lambda

For short, stateless requests, a Lambda handler is enough (keep the function timeout above your longest expected agent run, and the package within Lambda size limits):

```python
from strands import Agent

agent = Agent(model="us.anthropic.claude-sonnet-5-5", callback_handler=None)


def handler(event, context):
    return {"result": str(agent(event["prompt"]))}
```

### Monitoring and Logging

Strands emits OpenTelemetry traces for agent invocations, model calls and tool calls. Install the extra and configure an exporter:

```python
# pip install "strands-agents[otel]"
from strands.telemetry import StrandsTelemetry

StrandsTelemetry().setup_otlp_exporter()   # honours OTEL_EXPORTER_OTLP_ENDPOINT
```

On AgentCore Runtime, use the AWS Distro for OpenTelemetry (`opentelemetry-instrument`) and the traces appear in CloudWatch. Standard Python `logging` works too: `logging.getLogger("strands").setLevel(logging.DEBUG)`.

---

## Conclusion

Strands keeps the agent itself small — model, prompt, tools — and provides the pieces you need around it: multi-agent patterns, sessions, memory, hooks and telemetry. Pair it with AgentCore Runtime for hosting.

**GitHub:** https://github.com/strands-agents
**Documentation:** https://strandsagents.com

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.62.0 | 21 October 2026 | Guide rewritten against the real SDK: removed APIs that never existed in Strands (`as_tool` function, `aggregator`, `AgentGraph`, `Workflow`, `Context`, `ConversationMemory`, `PersistentState`, `S3Integration`, `deploy_to_lambda`, `CloudWatchLogger`, `Agent(instructions=…)`, `agent.run`); replaced with `Agent.as_tool`, `Swarm`, `GraphBuilder`, the `workflow` tool, `ToolContext`/`invocation_state`, session managers, `AgentCoreMemorySessionManager`, hooks and OTEL. Model IDs moved from Claude 3/3.5 to current Claude Bedrock IDs. |
| 1.42.0 | 7 June 2026 | Version bump from 1.36.0; `__version__` replaced with `importlib.metadata.version`. |
| 1.36.0 | 17 April 2026 | Multi-agent graph workflows; interrupt propagation through nested nodes; `AgentCoreMemorySessionManager` integration; steering hooks. |
| 1.0 | March 2025 | Initial guide created. |

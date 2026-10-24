---
title: "Strands Agents (Python): Multi-Agent Systems and A2A"
description: "Coordinating several Strands agents in Python: agents as tools, Swarm handoffs, GraphBuilder graphs with conditions, loops and parallel branches, the workflow tool, orchestrator hooks and streaming, and the Agent2Agent protocol."
framework: "strands"
language: "python"
sidebar:
  label: "Multi-agent and A2A"
  order: 4
---

Checked against `strands-agents` %%version=strands-agents%% and `strands-agents-tools` %%version=strands-agents-tools%% (%%versions-checked%%). Examples were run against an offline Bedrock stub that scripts each agent's replies; the A2A examples ran a real local A2A server.

## Choosing a pattern

| Pattern | Who decides the route | Use when |
|---------|----------------------|----------|
| **Agents as tools** | A supervisor model, by calling specialists as tools | One agent stays in charge and specialists answer sub-questions |
| **Swarm** | Each agent, by handing off to another | Work moves between peers and the order is not known in advance |
| **Graph** | You, with edges and conditions | The flow is known: pipelines, fan-out and fan-in, review loops |
| **`workflow` tool** | The model, by creating a task list with dependencies | You want the model to plan a multi-step job and run it |
| **A2A** | Any of the above, across process or organisation boundaries | The other agent runs elsewhere, possibly not on Strands |

Start with agents as tools; it is the simplest to debug. The [diagrams page](/strands-agents-guide/strands_diagrams/#multi-agent-patterns) shows each shape.

## Agents as tools

Put an `Agent` in another agent's `tools`, or call `as_tool()` to set its tool name and behaviour. The wrapped agent's `name` and `description` become the tool's name and description, so write them for the supervisor model to read:

```python
from strands import Agent, tool

SONNET = "us.anthropic.%%model=anthropic%%"
HAIKU = "us.anthropic.%%model=anthropic-small%%"


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
        plan: Plan name, for example "enterprise".
    """
    return {"plan": plan, "price_per_seat": 45, "sso": True}


sales = Agent(
    model=SONNET,
    name="sales_specialist",
    description="Answers questions about plans, pricing and purchases.",
    system_prompt="You are a sales specialist. Be accurate and concise.",
    tools=[get_plan_details],
    callback_handler=None,
)

billing = Agent(
    model=HAIKU,
    name="billing_specialist",
    description="Answers questions about invoices and payments.",
    system_prompt="You are a billing specialist. Be precise about amounts and dates.",
    tools=[get_invoice],
    callback_handler=None,
)

supervisor = Agent(
    model=SONNET,
    system_prompt=(
        "You are a customer service supervisor. Send plan and pricing questions to sales_specialist "
        "and invoice questions to billing_specialist, then combine their answers."
    ),
    tools=[sales, billing.as_tool(name="billing", description="Invoices, payments and refunds.")],
    callback_handler=None,
)

print(supervisor.tool_names)   # ['sales_specialist', 'billing']
print(supervisor("Invoice INV-77 looks wrong, and what does the enterprise plan cost?"))
```

The tool takes one string argument, `input`, which becomes the sub-agent's prompt. `as_tool()` options:

| Option | Default | Effect |
|--------|---------|--------|
| `name`, `description` | The agent's own | Tool name (`[a-zA-Z0-9_-]{1,64}`) and description |
| `preserve_context` | `False` | Keep the sub-agent's conversation between calls. When `False`, its messages and state are reset to their construction-time values before every call |
| `delegate` | `False` | Return the sub-agent's answer as the final response, without another supervisor model call |

The sub-agent runs with its own `invocation_state`; the parent's per-call state is not passed down. Share request data through a tool on the sub-agent, or by building the sub-agent per request. Sub-agent progress appears in the parent's `stream_async` as tool stream events.

Hierarchies are just more of the same:

```python
from strands import Agent

HAIKU = "us.anthropic.%%model=anthropic-small%%"
SONNET = "us.anthropic.%%model=anthropic%%"

tier1 = Agent(model=HAIKU, name="tier1_support", description="Basic how-to questions.", callback_handler=None)
tier2 = Agent(model=SONNET, name="tier2_support", description="Configuration and integration issues.", callback_handler=None)
tier3 = Agent(model="us.anthropic.%%model=anthropic-flagship%%", name="tier3_support",
              description="Complex incidents and root-cause analysis.", callback_handler=None)

support_lead = Agent(
    model=SONNET,
    name="support_lead",
    description="Handles all technical support requests.",
    system_prompt="Route each request to the lowest tier that can resolve it.",
    tools=[tier1, tier2, tier3],
    callback_handler=None,
)

front_desk = Agent(model=SONNET, system_prompt="Route requests to the right department.", tools=[support_lead], callback_handler=None)
front_desk("Our deployment has been slow since yesterday's upgrade.")
```

### Parallel opinions

If you want several independent answers combined (a vote, or best-of-n), you do not need a multi-agent primitive. Run agents concurrently and aggregate:

```python
import asyncio
from collections import Counter

from strands import Agent

classifiers = [
    Agent(
        model="us.anthropic.%%model=anthropic-small%%",
        system_prompt="Classify the request as BILLING, SALES or SUPPORT. Reply with one word.",
        callback_handler=None,
    )
    for _ in range(5)
]


async def vote(text: str) -> dict:
    results = await asyncio.gather(*(agent.invoke_async(text) for agent in classifiers))
    votes = Counter(str(r).strip().upper() for r in results)
    label, count = votes.most_common(1)[0]
    return {"decision": label, "confidence": count / len(classifiers), "votes": dict(votes)}


print(asyncio.run(vote("I was charged twice this month")))
```

Each agent is a separate object because one agent cannot run two calls at once.

## Swarm

A `Swarm` is a team of agents that pass work between themselves. Each member gets a `handoff_to_agent(agent_name, message, context)` tool, and sees the original task, the agents available, the handoff history and the shared context passed so far. One agent is active at a time; the swarm ends when an agent answers without handing off, or when a limit is reached.

```python
from strands import Agent
from strands.multiagent import Swarm

SONNET = "us.anthropic.%%model=anthropic%%"

financial = Agent(
    model=SONNET,
    name="financial_analyst",
    system_prompt="Assess costs, ROI and budget impact, then hand off to technical_analyst.",
    callback_handler=None,
)
technical = Agent(
    model=SONNET,
    name="technical_analyst",
    system_prompt="Assess architecture, scalability and delivery risk, then hand off to risk_analyst.",
    callback_handler=None,
)
risk = Agent(
    model=SONNET,
    name="risk_analyst",
    system_prompt="Assess security and compliance, then write the final balanced recommendation.",
    callback_handler=None,
)

swarm = Swarm(
    [financial, technical, risk],
    entry_point=financial,
    max_handoffs=10,
    max_iterations=15,
    execution_timeout=900.0,   # seconds, whole swarm
    node_timeout=300.0,        # seconds, per agent turn
)

result = swarm(
    "Proposal: move our monolith to microservices on EKS. "
    "£40k to implement, £8k a month to run, six months."
)

print(result.status)                                   # Status.COMPLETED
print([node.node_id for node in result.node_history])  # who handled it, in order
final = result.node_history[-1].node_id
print(result.results[final].result)                    # the last agent's AgentResult
```

| `Swarm` argument | Default | Meaning |
|------------------|---------|---------|
| `nodes` | | The agents (each needs a unique `name`) |
| `entry_point` | First node | Agent that receives the task |
| `max_handoffs` | 20 | Handoffs before the swarm stops |
| `max_iterations` | 20 | Agent turns before it stops |
| `execution_timeout` | 900.0 | Seconds for the whole run |
| `node_timeout` | 300.0 | Seconds per agent turn |
| `repetitive_handoff_detection_window`, `repetitive_handoff_min_unique_agents` | 0 (off) | Stop ping-pong: for example a window of 8 handoffs must involve at least 3 agents |
| `session_manager`, `hooks`, `plugins` | | Persistence and extensions for the swarm as a whole |

`invocation_state` passed to the swarm reaches every member's tools. Members cannot have their own `session_manager` and must not already have a tool called `handoff_to_agent`. `SwarmResult` has `status`, `results` (by node ID), `node_history`, `accumulated_usage`, `execution_count` and `execution_time` (milliseconds).

## Graph

`GraphBuilder` builds a directed graph whose nodes are agents, or other multi-agent systems. A node runs when its incoming edges are satisfied; nodes whose dependencies are met at the same time run in parallel. Each node receives the original task plus the output of the nodes that feed it.

### Conditions and loops

```python
from strands import Agent
from strands.multiagent import GraphBuilder
from strands.multiagent.graph import GraphState

HAIKU = "us.anthropic.%%model=anthropic-small%%"
SONNET = "us.anthropic.%%model=anthropic%%"

collector = Agent(model=HAIKU, name="collector", system_prompt="Collect and structure the raw data described in the task.", callback_handler=None)
validator = Agent(model=HAIKU, name="validator", system_prompt="Check the data for completeness. Start your reply with PASS or FAIL.", callback_handler=None)
analyst = Agent(model=SONNET, name="analyst", system_prompt="Analyse the data and identify trends.", callback_handler=None)
reporter = Agent(model=SONNET, name="reporter", system_prompt="Write a concise report of the key findings.", callback_handler=None)


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
builder.reset_on_revisit(True)        # a revisited node starts from a clean conversation
builder.set_max_node_executions(10)   # bound the loop
builder.set_execution_timeout(900)    # seconds

graph = builder.build()
result = graph("Trend analysis of Q4 sales from s3://sales-bucket/q4.csv")

print(result.status, [node.node_id for node in result.execution_order])
print(result.results["report"].result)
```

A condition receives the `GraphState` (`results`, `completed_nodes`, `execution_count`, ...). A condition that also declares a keyword-only `invocation_state` parameter receives the per-call state, which lets you route on data the model never sees:

```python
from strands import Agent
from strands.multiagent import GraphBuilder, Swarm
from strands.multiagent.graph import GraphState


def premium_only(state: GraphState, *, invocation_state: dict, **kwargs) -> bool:
    return invocation_state.get("tier") == "premium"


def agent(name: str, prompt: str) -> Agent:
    return Agent(model="us.anthropic.%%model=anthropic%%", name=name, system_prompt=prompt, callback_handler=None)


specialists = Swarm([agent("diagnose", "Diagnose the fault."), agent("fix", "Propose a fix.")])

builder = GraphBuilder()
builder.add_node(agent("triage", "Summarise the customer's problem."), "triage")
builder.add_node(specialists, "specialists")          # a Swarm as one node
builder.add_node(agent("faq", "Answer from the FAQ."), "faq")
builder.add_edge("triage", "specialists", condition=premium_only)
builder.add_edge("triage", "faq")
builder.set_execution_timeout(600)
graph = builder.build()

result = graph("The export button does nothing", invocation_state={"tier": "premium"})
print([node.node_id for node in result.execution_order])   # triage first, then faq and specialists in parallel
```

### Fan-out and fan-in

```python
from strands import Agent
from strands.multiagent import GraphBuilder

HAIKU = "us.anthropic.%%model=anthropic-small%%"

builder = GraphBuilder()
builder.add_node(Agent(model=HAIKU, name="prepare", callback_handler=None), "prepare")
for branch in ("sentiment", "entities", "category"):
    builder.add_node(Agent(model=HAIKU, name=branch, system_prompt=f"Extract the {branch}.", callback_handler=None), branch)
builder.add_node(Agent(model="us.anthropic.%%model=anthropic%%", name="synthesise", callback_handler=None), "synthesise")

for branch in ("sentiment", "entities", "category"):
    builder.add_edge("prepare", branch)       # fan out: these three run in parallel
    builder.add_edge(branch, "synthesise")    # fan in: synthesise waits for all three

builder.set_entry_point("prepare")
builder.set_execution_timeout(600)
result = builder.build()("Customer email: 'The new dashboard is great but exports keep failing.'")
print(result.completed_nodes, "of", result.total_nodes, "nodes completed")
```

### Limits and failures

Without `set_max_node_executions` or `set_execution_timeout`, `build()` warns that execution is unbounded; always set at least one when the graph has a cycle. `set_node_timeout()` bounds each node. Agents in a graph cannot have their own `session_manager`; use `set_session_manager()` on the builder.

Graphs and swarms report failures in the result rather than raising for each node error:

```python
from strands import Agent
from strands.multiagent import GraphBuilder, Status

builder = GraphBuilder()
builder.add_node(Agent(model="us.anthropic.%%model=anthropic%%", name="step", callback_handler=None), "step")
builder.set_execution_timeout(300)
graph = builder.build()

result = graph("Process the batch")
if result.status != Status.COMPLETED:
    print(f"{result.failed_nodes} of {result.total_nodes} nodes failed")
    for node_id, node_result in result.results.items():
        if node_result.status == Status.FAILED:
            print("failed:", node_id, node_result.result)   # the exception
```

`failed_nodes`, `completed_nodes` and `total_nodes` are counts; per-node detail is in `results`. `Status` values are `PENDING`, `DISPATCHED`, `EXECUTING`, `COMPLETED`, `FAILED` and `INTERRUPTED`.

## Streaming and hooks for orchestrators

`Swarm` and `Graph` have `invoke_async` and `stream_async` as well as being callable. The stream yields dictionaries with a `type`:

| `type` | Fields |
|--------|--------|
| `multiagent_node_start` | `node_id`, `node_type` |
| `multiagent_node_stream` | `node_id`, `event` (the agent's own stream event) |
| `multiagent_handoff` | `from_node_ids`, `to_node_ids`, `message` (swarm) |
| `multiagent_node_stop` | `node_id`, `node_result` |
| `multiagent_result` | `result` |

```python
import asyncio

from strands import Agent
from strands.multiagent import Swarm

triage = Agent(model="us.anthropic.%%model=anthropic%%", name="triage",
               system_prompt="Classify the request and hand off to billing.", callback_handler=None)
billing = Agent(model="us.anthropic.%%model=anthropic%%", name="billing",
                system_prompt="Answer billing questions.", callback_handler=None)
swarm = Swarm([triage, billing], max_handoffs=4)


async def main() -> None:
    async for event in swarm.stream_async("Why was I charged twice?"):
        kind = event.get("type")
        if kind == "multiagent_node_start":
            print("start:", event["node_id"])
        elif kind == "multiagent_handoff":
            print("handoff:", event["from_node_ids"], "->", event["to_node_ids"])
        elif kind == "multiagent_node_stream" and "data" in event["event"]:
            print(event["event"]["data"], end="")
        elif kind == "multiagent_result":
            print("\nstatus:", event["result"].status)


asyncio.run(main())
```

### Hooks for orchestrators

Orchestrators take `hooks=` (`Swarm(..., hooks=[...])`, `GraphBuilder().set_hook_providers([...])`) with their own events: `MultiAgentInitializedEvent`, `BeforeMultiAgentInvocationEvent`, `BeforeNodeCallEvent`, `AfterNodeCallEvent` and `AfterMultiAgentInvocationEvent`. `BeforeNodeCallEvent.cancel_node` stops the run at that node (it does not skip it: a `Graph` raises `RuntimeError` with the message and a `Swarm` ends with `Status.FAILED`), and `BeforeNodeCallEvent.interrupt(...)` pauses the whole orchestrator for a human decision:

```python
from strands import Agent
from strands.hooks import AfterNodeCallEvent, BeforeNodeCallEvent, HookProvider, HookRegistry
from strands.multiagent import GraphBuilder


class NodeLog(HookProvider):
    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeNodeCallEvent, self.before)
        registry.add_callback(AfterNodeCallEvent, self.after)

    def before(self, event: BeforeNodeCallEvent) -> None:
        if event.node_id == "publish" and (event.invocation_state or {}).get("dry_run"):
            event.cancel_node = "dry run: not publishing"
        print("->", event.node_id)

    def after(self, event: AfterNodeCallEvent) -> None:
        print("<-", event.node_id)


builder = GraphBuilder()
builder.add_node(Agent(model="us.anthropic.%%model=anthropic%%", name="draft", callback_handler=None), "draft")
builder.add_node(Agent(model="us.anthropic.%%model=anthropic%%", name="publish", callback_handler=None), "publish")
builder.add_edge("draft", "publish")
builder.set_hook_providers([NodeLog()])
builder.set_execution_timeout(300)
graph = builder.build()

try:
    graph("Write and publish the release note", invocation_state={"dry_run": True})
except RuntimeError as exc:
    print("stopped:", exc)   # stopped: dry run: not publishing
```

## The workflow tool

For a fixed pipeline you know in advance, a graph is clearer. When you want the *model* to plan and run a multi-task job, with dependencies, priorities and a model per task, give it the `workflow` tool from `strands-agents-tools`. `agent.tool.workflow(...)` drives it directly:

```python
from strands import Agent
from strands_tools import workflow

coordinator = Agent(model="us.anthropic.%%model=anthropic%%", tools=[workflow], callback_handler=None)

coordinator.tool.workflow(
    action="create",
    workflow_id="document_processing",
    tasks=[
        {"task_id": "extract", "description": "Extract invoice fields from the document text.", "priority": 5},
        {
            "task_id": "classify",
            "description": "Classify the document as invoice, contract, report or other.",
            "dependencies": ["extract"],
            "model_provider": "bedrock",
            "model_settings": {"model_id": "us.anthropic.%%model=anthropic-small%%"},
        },
        {"task_id": "validate", "description": "Check the extracted fields are complete.", "dependencies": ["extract"]},
    ],
)

coordinator.tool.workflow(action="start", workflow_id="document_processing")
status = coordinator.tool.workflow(action="status", workflow_id="document_processing")
print(status["content"][0]["text"][:200])
```

Tasks whose dependencies are met run in parallel. Workflow state is written to `~/.strands/workflows` (`STRANDS_WORKFLOW_DIR` changes it). `strands_tools` also has `swarm`, `graph` and `agent_graph` tools that let the model build those structures itself. Durable business processes with long waits, approvals and retries across days belong in AWS Step Functions, with each step invoking an agent (for example on AgentCore Runtime).

## A2A (Agent2Agent)

A2A is an open protocol for agents to call each other over HTTP, regardless of framework. Install the `a2a` extra.

### Serving an agent

```python
from strands import Agent
from strands.multiagent.a2a import A2AServer


def make_agent(context_id: str) -> Agent:
    """One agent per A2A conversation (context)."""
    return Agent(
        model="us.anthropic.%%model=anthropic%%",
        name="geography",
        description="Answers questions about countries, capitals and borders.",
        callback_handler=None,
    )


server = A2AServer(agent_factory=make_agent, host="127.0.0.1", port=9000)
server.serve()   # Starlette app under uvicorn; the agent card is at /.well-known/agent-card.json
```

The factory is called once at start-up (with a placeholder ID) to build the agent card from the agent's `name`, `description` and tools, then once per A2A context; the server keeps up to `max_contexts` of them. Passing a single `agent=` still works but is deprecated, because one agent would serialise every caller's requests and mix their conversations. `skills=` overrides the advertised skills, `http_url=` sets the public URL behind a load balancer, `serve(app_type="fastapi")` uses FastAPI, and `to_starlette_app()` / `to_fastapi_app()` return the ASGI app to mount in your own server.

### Calling a remote agent

`A2AAgent` is a client with the same calling conventions as `Agent`:

```python
import asyncio

from strands.agent.a2a_agent import A2AAgent

remote = A2AAgent(endpoint="http://127.0.0.1:9000")


async def main() -> None:
    card = await remote.get_agent_card()
    print(card.name, "-", card.description)
    result = await remote.invoke_async("What is the capital of France?")
    print(result)


asyncio.run(main())
print(remote("And of Spain?"))   # synchronous call
```

An `A2AAgent` can be a `Graph` node. It is not a tool by itself; to give a local agent access to remote agents, use the vended `make_a2a_client` tool, which restricts the model to endpoints you list:

```python
from strands import Agent
from strands.vended_tools import make_a2a_client

agent = Agent(
    model="us.anthropic.%%model=anthropic%%",
    tools=[make_a2a_client(allowed_endpoints=["http://127.0.0.1:9000"])],
    callback_handler=None,
)
print(agent.tool_names)   # ['a2a_client']
```

Hosting A2A servers on AgentCore Runtime is covered in the [Bedrock A2A guide](/amazon-bedrock-agents-guide/python/bedrock_a2a_protocol_guide/).

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-24 | 1.62.0 | New page, from the multi-agent sections of the former Bedrock Strands page. Adds `as_tool` options (`delegate`), state propagation rules, swarm options and result fields, context-aware graph conditions, nested orchestrators, orchestrator streaming and hooks, and A2A server and client usage. |

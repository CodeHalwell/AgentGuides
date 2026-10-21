---
title: "Mistral Agents API: Agent Orchestration Guide"
description: "Coordinating multiple Mistral agents: code-driven pipelines, parallel fan-out, handoff graphs, manager-worker delegation and state across agents."
framework: mistral-agents-api
---

**Verified against `mistralai` %%version=mistralai%%** (checked %%versions-checked%%).

There are two ways to coordinate agents on Mistral, and most systems use both:

- **Code-driven orchestration.** Your Python decides which agent runs next and what it receives. Each agent has its own conversation; you pass results between them. Predictable, testable, and easy to run in parallel.
- **Handoffs.** An agent's `handoffs` list names the agents it may pass the conversation to, and the model decides when. The conversation, with its full history, moves to the next agent on Mistral's side.

One thing that does not exist: `conversations.append()` has no `agent_id`. A conversation continues with its agent (or with whichever agent a handoff moved it to); to involve another agent, use a handoff or start a separate conversation.

All examples use this setup:

```python
import os
from mistralai.client import Mistral
from mistralai.client.models import ConversationResponse, MessageOutputEntry, TextChunk

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

def text_of(response: ConversationResponse) -> str:
    for entry in reversed(response.outputs):
        if isinstance(entry, MessageOutputEntry):
            if isinstance(entry.content, str):
                return entry.content
            return "".join(c.text for c in entry.content if isinstance(c, TextChunk))
    return ""

def ask(agent_id: str, prompt: str) -> str:
    return text_of(client.beta.conversations.start(agent_id=agent_id, inputs=prompt))
```

---

## 1. Specialised agents

Small agents with one job each are easier to prompt, test and price than one agent with every tool.

```python
researcher = client.beta.agents.create(
    model="mistral-medium-latest",
    name="researcher",
    description="Finds current information on the web and cites sources.",
    instructions="Search the web. Return findings as bullet points, each with a source URL.",
    tools=[{"type": "web_search"}],
)

analyst = client.beta.agents.create(
    model="mistral-medium-latest",
    name="analyst",
    description="Analyses data with Python and produces charts.",
    instructions="Use the code interpreter for every number. Show the code you ran.",
    tools=[{"type": "code_interpreter"}],
)

writer = client.beta.agents.create(
    model="mistral-small-latest",
    name="writer",
    description="Turns notes into a short, clear report.",
    instructions="Write in plain British English. Lead with the conclusion. No more than 400 words.",
)
```

The `description` matters: it is what other agents see when deciding whether to hand off.

---

## 2. Sequential pipeline

Each step gets the previous step's output explicitly. Because every step is a separate conversation, you can log, validate or retry steps independently.

```python
def research_report(topic: str) -> dict[str, str]:
    findings = ask(researcher.id, f"Research: {topic}")
    analysis = ask(analyst.id, f"Analyse these findings and quantify the trends:\n\n{findings}")
    report = ask(writer.id, f"Write a report from this research and analysis.\n\nResearch:\n{findings}\n\nAnalysis:\n{analysis}")
    return {"findings": findings, "analysis": analysis, "report": report}

result = research_report("electric vehicle adoption in the EU, 2024-2026")
print(result["report"])
```

### Conditional routing

Route with a cheap classifier agent and structured output, then dispatch in code:

```python
import json
from pydantic import BaseModel
from mistralai.extra import response_format_from_pydantic_model

class Route(BaseModel):
    route: str        # "research", "analysis" or "writing"
    reason: str

router = client.beta.agents.create(
    model="mistral-small-latest",
    name="router",
    description="Classifies requests.",
    instructions=(
        "Classify the request. route must be one of: research (needs current facts), "
        "analysis (needs calculation), writing (needs drafting only)."
    ),
    completion_args={"response_format": response_format_from_pydantic_model(Route), "temperature": 0},
)

SPECIALISTS = {"research": researcher.id, "analysis": analyst.id, "writing": writer.id}

def handle(request: str) -> str:
    route = Route.model_validate_json(ask(router.id, request))
    agent_id = SPECIALISTS.get(route.route, writer.id)
    return ask(agent_id, request)
```

---

## 3. Parallel fan-out and fan-in

Independent work runs concurrently with the async client:

```python
import asyncio

async def ask_async(agent_id: str, prompt: str) -> str:
    response = await client.beta.conversations.start_async(agent_id=agent_id, inputs=prompt)
    return text_of(response)

async def market_scan(company: str) -> str:
    angles = [
        f"Latest financial results and guidance for {company}",
        f"Main competitors of {company} and their recent moves",
        f"Regulatory issues affecting {company} in the EU",
    ]
    results = await asyncio.gather(*(ask_async(researcher.id, a) for a in angles))
    combined = "\n\n".join(f"## {a}\n{r}" for a, r in zip(angles, results))
    return await ask_async(writer.id, f"Synthesise into one briefing:\n\n{combined}")

print(asyncio.run(market_scan("ASML")))
```

Bound concurrency with an `asyncio.Semaphore` for large fan-outs; rate limits apply per workspace.

### Competitive evaluation

Ask several agents (or models) for an answer and have a judge pick:

```python
async def best_of(question: str, candidates: list[str], judge_id: str) -> str:
    answers = await asyncio.gather(*(ask_async(a, question) for a in candidates))
    ballot = "\n\n".join(f"Answer {i + 1}:\n{a}" for i, a in enumerate(answers))
    verdict = await ask_async(judge_id, f"Question: {question}\n\n{ballot}\n\nReply with the number of the best answer and why.")
    return verdict
```

---

## 4. Handoff graphs

Handoffs let the model route mid-conversation. Configure who may hand off to whom with `handoffs`:

```python
billing = client.beta.agents.create(
    model="mistral-small-latest", name="billing",
    description="Invoices, refunds, payment methods.",
    instructions="Resolve billing questions. Ask for the invoice number if needed.",
)
technical = client.beta.agents.create(
    model="mistral-medium-latest", name="technical",
    description="Bugs, errors, configuration and integration problems.",
    instructions="Troubleshoot step by step. Search the web for error messages you do not recognise.",
    tools=[{"type": "web_search"}],
)
frontdesk = client.beta.agents.create(
    model="mistral-small-latest", name="frontdesk",
    description="First point of contact for customers.",
    instructions="Greet briefly, work out what the customer needs, and hand off to billing or technical.",
    handoffs=[billing.id, technical.id],
)

# Specialists may hand back if the question changes
for specialist in (billing, technical):
    client.beta.agents.update(agent_id=specialist.id, handoffs=[frontdesk.id])
```

### Server-side execution

```python
response = client.beta.conversations.start(
    agent_id=frontdesk.id,
    inputs="Since yesterday's update the export button returns a 500 error.",
    handoff_execution="server",
)
for entry in response.outputs:
    if entry.type == "agent.handoff":
        print(f"{entry.previous_agent_name} -> {entry.next_agent_name}")
print(text_of(response))

# The conversation stays with whoever is now handling it
follow_up = client.beta.conversations.append(
    conversation_id=response.conversation_id,
    inputs="It happens with CSV only, not with XLSX.",
)
```

### Client-side execution

With `handoff_execution="client"` the response stops at the handoff and returns it to you, so your code can check, log or block it before another agent acts:

```python
from mistralai.client.models import AgentHandoffEntry

response = client.beta.conversations.start(
    agent_id=frontdesk.id,
    inputs="Please refund invoice INV-2207.",
    handoff_execution="client",
)

last = response.outputs[-1]
if isinstance(last, AgentHandoffEntry):
    print("model wants to route to", last.next_agent_name)
    if last.next_agent_name == "billing" and not user_is_verified():   # your own check
        print("ask the customer to verify before billing acts")
    else:
        # Run the chosen specialist yourself, giving it the context it needs
        answer = ask(last.next_agent_id, "Customer request: Please refund invoice INV-2207.")
        print(answer)
```

Mistral's documentation describes `client` mode only as returning the handoff to the caller; it does not document resuming the same conversation afterwards. Running the target agent yourself, as above, or using `server` mode where you do not need the checkpoint, are the dependable options.

### Handoffs with function tools

When an agent in the graph has a function tool, the run pauses with a `function.call` entry whichever agent made it; reply with a `FunctionResultEntry` as in the [function calling section](/mistral-agents-api-guide/mistral_agents_api_comprehensive_guide/#8-function-calling), and the conversation resumes with that agent.

---

## 5. Manager-worker delegation

For open-ended tasks, give a manager agent function tools that run worker agents. Your code executes each delegation, so you keep control over cost and what the workers see:

```python
import json
from mistralai.client.models import FunctionCallEntry, FunctionResultEntry

WORKERS = {"researcher": researcher.id, "analyst": analyst.id, "writer": writer.id}

manager = client.beta.agents.create(
    model="mistral-medium-latest",
    name="manager",
    description="Plans work and delegates to specialists.",
    instructions=(
        "Break the task into steps. Use delegate() for each step, choosing the right worker. "
        "When all steps are done, give the final answer."
    ),
    tools=[{
        "type": "function",
        "function": {
            "name": "delegate",
            "description": "Give a self-contained task to a worker and get its result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "worker": {"type": "string", "enum": list(WORKERS)},
                    "task": {"type": "string", "description": "Complete instructions, including any context needed."},
                },
                "required": ["worker", "task"],
            },
        },
    }],
)

def run_manager(goal: str, max_delegations: int = 8) -> str:
    response = client.beta.conversations.start(agent_id=manager.id, inputs=goal)
    for _ in range(max_delegations):
        calls = [e for e in response.outputs if isinstance(e, FunctionCallEntry)]
        if not calls or response.outputs[-1].type != "function.call":
            break
        results = []
        for call in calls:
            args = json.loads(call.arguments) if isinstance(call.arguments, str) else call.arguments
            output = ask(WORKERS[args["worker"]], args["task"])
            results.append(FunctionResultEntry(tool_call_id=call.tool_call_id, result=output))
        response = client.beta.conversations.append(conversation_id=response.conversation_id, inputs=results)
    return text_of(response)

print(run_manager("Prepare a one-page brief on heat-pump subsidies in France, Germany and Italy."))
```

The `max_delegations` cap is the important line: it bounds cost if the manager keeps delegating.

---

## 6. State across agents

| Need | Approach |
|---|---|
| One agent's full history | the conversation itself (`get_history`) |
| Hand the whole conversation to another agent | handoffs |
| Give another agent only what it needs | pass text or structured output explicitly (pipelines, delegation) |
| State that outlives conversations | your own store (database, cache), keyed by user or task |

Explicit state passing with a typed object:

```python
from pydantic import BaseModel

class CaseState(BaseModel):
    customer_id: str
    summary: str = ""
    findings: list[str] = []
    conversation_ids: dict[str, str] = {}

def step(state: CaseState, role: str, agent_id: str, prompt: str) -> CaseState:
    context = f"Customer {state.customer_id}. Summary so far: {state.summary or 'none'}."
    response = client.beta.conversations.start(agent_id=agent_id, inputs=f"{context}\n\n{prompt}")
    state.conversation_ids[role] = response.conversation_id
    state.findings.append(text_of(response))
    return state
```

Persist `CaseState` (for example as JSON in PostgreSQL or Redis) between steps so a crashed run can resume, and store the conversation IDs so each step can be inspected or continued later with `append()`.

Reading the history of a handoff conversation shows which agent produced each entry:

```python
history = client.beta.conversations.get_history(conversation_id=response.conversation_id)
for entry in history.entries:
    print(entry.type, getattr(entry, "agent_id", None))
```

---

## 7. Event-driven orchestration

For long-running or bursty workloads, put agent steps behind a queue and let workers pick them up:

```python
import asyncio

queue: asyncio.Queue[tuple[str, str, asyncio.Future]] = asyncio.Queue()

async def worker(name: str, limit: asyncio.Semaphore) -> None:
    while True:
        agent_id, prompt, future = await queue.get()
        try:
            async with limit:
                future.set_result(await ask_async(agent_id, prompt))
        except Exception as exc:          # deliver failures to the caller
            future.set_exception(exc)
        finally:
            queue.task_done()

async def submit(agent_id: str, prompt: str) -> str:
    future: asyncio.Future = asyncio.get_running_loop().create_future()
    await queue.put((agent_id, prompt, future))
    return await future
```

In production the queue is usually external (Redis, SQS, Pub/Sub), so work survives restarts and scales across processes.

---

## 8. Production checklist

- **Pin models and agent versions.** Use dated model names and `agent_version` (or version aliases) so a change to one agent does not silently change a pipeline.
- **Cap loops.** Limit delegations, handoff round-trips and retries.
- **Log every hop.** Record conversation IDs, agent names, handoff entries and token usage per step.
- **Validate between steps.** Structured output plus Pydantic validation catches a bad step before it pollutes the next.
- **Fail soft.** If a specialist fails, return a partial result rather than failing the whole request.
- **Measure.** Track tokens (`response.usage`) and latency per agent, not just per request; one chatty agent usually dominates cost.

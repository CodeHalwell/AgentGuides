---
title: "Agno: recipes"
description: "Worked Agno examples: a support assistant with memory and confirmed refunds, a handbook Q&A over local files, a research-and-write workflow with review, a routing front desk with structured triage, a streaming FastAPI endpoint without AgentOS, batch processing with structured output, and a scheduled digest."
framework: "agno"
language: "python"
sidebar:
  label: "Recipes"
  order: 45
---

Checked against **agno %%version=agno%%** (checked %%versions-checked%%). Each recipe was run end to end offline, with provider models replaced by a stand-in that calls the first offered tool and then answers; the control flow, storage and tool execution are real. Swap in your own API keys to run them for real.

## Support assistant with memory and confirmed refunds

A single agent that looks orders up freely, asks before refunding, remembers customers across sessions and refuses prompt-injection attempts.

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.guardrails import PromptInjectionGuardrail
from agno.memory import MemoryManager
from agno.models.openai import OpenAIResponses
from agno.run import RunStatus
from agno.tools import tool

ORDERS = {"1001": {"item": "Kettle", "status": "delivered", "total": 34.0}}
db = SqliteDb(db_file="support.db")


def get_order(order_id: str) -> str:
    """Look up an order.

    Args:
        order_id: The order number, e.g. 1001.
    """
    order = ORDERS.get(order_id)
    return str(order) if order else f"No order {order_id}"


@tool(requires_confirmation=True)
def refund_order(order_id: str, reason: str) -> str:
    """Refund an order in full.

    Args:
        order_id: The order number.
        reason: Why the customer wants a refund.
    """
    ORDERS[order_id]["status"] = "refunded"
    return f"Refunded order {order_id}"


support = Agent(
    name="Support",
    model=OpenAIResponses(id="%%model=openai%%"),
    db=db,
    tools=[get_order, refund_order],
    instructions=[
        "You help customers of a small homeware shop.",
        "Always look the order up before discussing it.",
        "Only refund delivered orders.",
    ],
    add_history_to_context=True,
    num_history_runs=5,
    memory_manager=MemoryManager(model=OpenAIResponses(id="%%model=openai-small%%"), db=db),
    enable_agentic_memory=True,
    add_memories_to_context=True,
    pre_hooks=[PromptInjectionGuardrail()],
)


def handle(message: str, user_id: str, session_id: str, approve) -> str:
    run = support.run(message, user_id=user_id, session_id=session_id)
    while run.is_paused:
        for req in run.active_requirements:
            if req.needs_confirmation:
                if approve(req.tool_execution.tool_name, req.tool_execution.tool_args):
                    req.confirm()
                else:
                    req.reject(note="A member of staff declined the refund.")
        run = support.continue_run(run_id=run.run_id, requirements=run.requirements,
                                   user_id=user_id, session_id=session_id)
    if run.status == RunStatus.error:
        return "Sorry, I can't help with that request."
    return run.content


print(handle("Order 1001 arrived broken, please refund it", "cust-7", "chat-1", approve=lambda name, args: True))
print(ORDERS["1001"]["status"])
```

The `approve` callback is where a staff UI, a policy check or a second model would go. Under AgentOS the same agent pauses over HTTP and your front end posts the decision to `/agents/{id}/runs/{run_id}/continue` ([AgentOS](/agno-guide/agno_agentos/#human-in-the-loop-over-http)).

## Handbook Q&A over local files

Load a folder of Markdown and text files into a vector store and answer from it, with the source recorded on the run.

```python
from pathlib import Path

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.openai import OpenAIResponses
from agno.vectordb.lancedb import LanceDb

docs = Path("handbook")
docs.mkdir(exist_ok=True)
(docs / "leave.md").write_text("# Leave\nStaff get 25 days of annual leave plus bank holidays.\n")
(docs / "expenses.txt").write_text("Expenses must be filed within 30 days with a receipt.\n")

handbook = Knowledge(
    name="Handbook",
    vector_db=LanceDb(uri="tmp/lancedb", table_name="handbook", embedder=OpenAIEmbedder()),
    contents_db=SqliteDb(db_file="handbook.db"),
)
handbook.insert(path=str(docs), metadata={"source": "handbook"}, skip_if_exists=True)

hr = Agent(
    model=OpenAIResponses(id="%%model=openai%%"),
    knowledge=handbook,
    search_knowledge=False,          # always retrieve, rather than leaving it to the model
    add_knowledge_to_context=True,
    instructions="Answer only from the references. If they do not cover the question, say so.",
)

run = hr.run("How much annual leave do I get?")
print(run.content)
for ref in run.references or []:
    print([doc["name"] for doc in ref.references])   # e.g. ['leave.md', 'expenses.txt']
```

`skip_if_exists=True` makes the insert safe to repeat on start-up. For a larger handbook switch to agentic search (`search_knowledge=True`, the default) so the model can issue several queries, and use `PgVector` with hybrid search in production.

## Research and write, with a review loop

A workflow that searches two sources in parallel, drafts, and loops on review until the reviewer approves or three rounds pass. Short inputs skip research. Each round after the first gives the writer the notes, its previous draft and the reviewer's feedback, so it revises rather than starting again.

```python
from typing import List

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.workflow import Condition, Loop, Parallel, Step, StepInput, StepOutput, Workflow

model = OpenAIResponses(id="%%model=openai%%")
news = Agent(name="News", model=model, instructions="Find recent news on the topic. Bullet points.")
background = Agent(name="Background", model=model, instructions="Give background facts on the topic.")
writer = Agent(name="Writer", model=model, instructions="Write or revise a 150-word briefing from the notes and feedback.")
reviewer = Agent(name="Reviewer", model=model,
                 instructions="Review the briefing. Reply APPROVED if it is accurate and clear, otherwise list fixes.")


def needs_research(si: StepInput) -> bool:
    return len(str(si.input).split()) > 2


def gather_notes(si: StepInput) -> StepOutput:
    found = si.get_step_content("research") or {}
    notes = "\n\n".join(str(v) for v in found.values()) if isinstance(found, dict) else ""
    return StepOutput(content=f"Topic: {si.input}\n\nNotes:\n{notes}")


def review(si: StepInput) -> StepOutput:
    # Pass the draft explicitly: inside a loop, an agent step is handed the most
    # recently *inserted* step output, which from round two is the stale revision brief.
    draft = si.get_step_content("write")
    return StepOutput(content=reviewer.run(f"Review this briefing:\n\n{draft}").content)


def revision_brief(si: StepInput) -> StepOutput:
    # Becomes the writer's input for the next round (forward_iteration_output=True)
    return StepOutput(content=(
        f"{si.get_step_content('notes')}\n\n"
        f"Previous draft:\n{si.get_step_content('write')}\n\n"
        f"Reviewer feedback:\n{si.previous_step_content}"
    ))


def approved(outputs: List[StepOutput]) -> bool:
    return any(o.step_name == "review" and "APPROVED" in str(o.content) for o in outputs)


briefing = Workflow(
    name="Briefing",
    steps=[
        Condition(
            name="maybe_research",
            evaluator=needs_research,
            steps=[Parallel(Step(name="news", agent=news), Step(name="background", agent=background), name="research")],
        ),
        Step(name="notes", executor=gather_notes),
        Loop(
            name="draft_and_review",
            steps=[
                Step(name="write", agent=writer),
                Step(name="review", executor=review),
                Step(name="revision_brief", executor=revision_brief),
            ],
            end_condition=approved,
            max_iterations=3,
            forward_iteration_output=True,   # off by default: each round would restart from the notes
        ),
    ],
)

run = briefing.run("UK heat pump grants in 2026")
print(run.status)
print(run.content)   # the last step's output: the final revision brief, ending with the verdict

# The loop's StepOutput holds every iteration's steps in order
loop = run.step_results[-1]
drafts = [s.content for s in (loop.steps or []) if s.step_name == "write"]
print(drafts[-1] if drafts else None)   # the latest draft
```

## Front desk: classify, then route

A small, cheap model classifies the message into a schema; the triage result decides which specialist team handles it, so routing is testable without a model.

```python
from typing import Literal

from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.models.openai import OpenAIResponses


class Triage(BaseModel):
    department: Literal["billing", "technical", "other"]
    urgent: bool = Field(description="True if the customer cannot use the product at all")
    summary: str


classifier = Agent(
    model=OpenAIResponses(id="%%model=openai-small%%"),
    output_schema=Triage,
    instructions="Classify the customer message.",
)
specialists = {
    "billing": Agent(name="Billing", model=OpenAIResponses(id="%%model=openai%%"), instructions="Resolve billing questions."),
    "technical": Agent(name="Technical", model=OpenAIResponses(id="%%model=openai%%"), instructions="Resolve technical problems."),
    "other": Agent(name="General", model=OpenAIResponses(id="%%model=openai-small%%"), instructions="Help or redirect politely."),
}


def front_desk(message: str, user_id: str) -> str:
    triage: Triage = classifier.run(message).content
    agent = specialists[triage.department]
    prefix = "[URGENT] " if triage.urgent else ""
    return prefix + str(agent.run(message, user_id=user_id).content)


print(front_desk("I was charged twice for my subscription", user_id="cust-9"))
```

If you would rather let a model choose, the same specialists in a `Team(mode=TeamMode.route)` do this in one leader call ([Teams](/agno-guide/agno_teams/#route)); the explicit version above is easier to log and test.

## Streaming endpoint without AgentOS

When you only need one endpoint in an existing FastAPI service, stream the agent's events yourself:

```python
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.run.agent import RunEvent

agent = Agent(model=OpenAIResponses(id="%%model=openai%%"), db=SqliteDb(db_file="chat.db"), add_history_to_context=True)
app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str
    user_id: str


@app.post("/chat")
async def chat(req: ChatRequest):
    async def events():
        async for event in agent.arun(req.message, session_id=req.session_id, user_id=req.user_id,
                                      stream=True, stream_events=True):
            if event.event == RunEvent.run_content and event.content:
                yield f"data: {json.dumps({'text': event.content})}\n\n"
            elif event.event == RunEvent.tool_call_started:
                yield f"data: {json.dumps({'tool': event.tool.tool_name})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
```

Run it with `uvicorn chat_app:app`. AgentOS gives you this plus sessions, auth and the rest of the API; this is the minimal version.

## Batch processing with structured output

Extract structured data from many documents concurrently, with a cap on parallel requests:

```python
import asyncio

from pydantic import BaseModel

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.run import RunStatus


class Invoice(BaseModel):
    supplier: str
    total: float
    currency: str


extractor = Agent(
    model=OpenAIResponses(id="%%model=openai-small%%", retries=2, exponential_backoff=True),
    output_schema=Invoice,
    instructions="Extract the invoice fields.",
    telemetry=False,
)

texts = [
    "Invoice from Acme Ltd. Total due: GBP 120.50",
    "Northwind Traders — amount payable 99.00 EUR",
]


async def extract_all(items, limit: int = 5):
    sem = asyncio.Semaphore(limit)

    async def one(text):
        async with sem:
            run = await extractor.arun(text)
            return run.content if run.status == RunStatus.completed else None

    return await asyncio.gather(*(one(t) for t in items))


for invoice in asyncio.run(extract_all(texts)):
    print(invoice)
```

One agent object can serve concurrent `arun()` calls; give each call its own `session_id` if the agent has a `db`, or leave the agent without one for stateless batch work as here.

## A scheduled morning digest

Serve an agent with the scheduler on and store a cron schedule that calls it every weekday at 08:00 London time:

```python
from datetime import datetime, timedelta, timezone

from agno.agent import Agent
from agno.db.base import SessionType
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.os import AgentOS
from agno.scheduler import ScheduleManager
from agno.tools import tool

db = SqliteDb(db_file="digest.db")


@tool
def support_sessions(days_ago: int = 1) -> str:
    """Return the conversations the support agent started on one UTC day (1 = yesterday)."""
    day_end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1 - days_ago)
    day_start = day_end - timedelta(days=1)
    sessions = db.get_sessions(
        session_type=SessionType.AGENT,
        component_id="support",           # the support agent's id
        start_timestamp=int(day_start.timestamp()),
        end_timestamp=int(day_end.timestamp()),
    )
    lines = []
    for s in sessions:
        for run in s.runs or []:
            question = run.input.input_content if run.input else ""
            lines.append(f"[{s.session_id}] Q: {question}\nA: {run.content}")
    return "\n\n".join(lines) or "No support sessions that day."


support = Agent(id="support", name="Support", model=OpenAIResponses(id="%%model=openai%%"), db=db)
digest = Agent(id="digest", name="Digest", model=OpenAIResponses(id="%%model=openai%%"), db=db,
               tools=[support_sessions],
               instructions="Call support_sessions, then write a five-bullet digest of those conversations.")

agent_os = AgentOS(db=db, agents=[support, digest], scheduler=True)
app = agent_os.get_app()

ScheduleManager(db).create(
    name="weekday-digest",
    cron="0 8 * * 1-5",
    timezone="Europe/London",
    endpoint="/agents/digest/runs",
    payload={"message": "Write today's digest", "stream": False},
    if_exists="skip",   # safe to run on every start-up
)

if __name__ == "__main__":
    agent_os.serve(app="digest_app:app")
```

Sharing a `db` stores every run, but an agent only sees its own session's history; the digest agent reads the support agent's sessions through the `support_sessions` tool. `get_sessions` filters on when a session was created, so a conversation started earlier and continued yesterday is not included. The schedule row is created and listed in `/schedules`; the poller firing it at 08:00 was not observed for this guide. Results land in the digest agent's sessions like any other run.

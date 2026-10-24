---
title: "Strands Agents (Python): Recipes"
description: "Complete Strands Agents programs in Python: a support agent with triage and sessions, streaming over HTTP, an approval workflow that resumes in a later request, retrieval over documents, a research graph, batch extraction, cost-based model routing and AWS service tools."
framework: "strands"
language: "python"
sidebar:
  label: "Recipes"
  order: 6
---

Checked against `strands-agents` %%version=strands-agents%% (%%versions-checked%%). Each recipe is a complete program. All were run against an offline Bedrock stub (with `moto` for S3, DynamoDB and SNS), so they are known to execute; the answers you get from a real model will of course differ.

## Support agent with triage and memory

A support agent that classifies each message with structured output, answers with tools, and keeps the conversation per customer:

```python
from typing import Literal

from pydantic import BaseModel, Field
from strands import Agent, ToolContext, tool
from strands.session import FileSessionManager

ORDERS = {"ORD-1": {"status": "shipped", "carrier": "DPD"}, "ORD-2": {"status": "processing"}}


class Triage(BaseModel):
    category: Literal["order", "billing", "technical", "other"]
    urgency: Literal["low", "normal", "high"]
    summary: str = Field(description="One sentence")


@tool(context=True)
def order_status(order_id: str, tool_context: ToolContext) -> dict:
    """Look up the status of one of the customer's orders.

    Args:
        order_id: Order ID, for example ORD-1.
    """
    customer = tool_context.invocation_state["customer_id"]   # set by the server, not the model
    order = ORDERS.get(order_id)
    if order is None:
        raise ValueError(f"No order {order_id} for customer {customer}")
    return {"order_id": order_id, **order}


triage_agent = Agent(
    model="us.anthropic.%%model=anthropic-small%%",
    system_prompt="Classify customer support messages.",
    callback_handler=None,
)


def support_agent(customer_id: str) -> Agent:
    return Agent(
        model="us.anthropic.%%model=anthropic%%",
        system_prompt="You are a support agent for Example Ltd. Use tools; never guess order details.",
        tools=[order_status],
        session_manager=FileSessionManager(session_id=f"support-{customer_id}", storage_dir="./sessions"),
        callback_handler=None,
    )


def handle(customer_id: str, message: str) -> dict:
    triage = triage_agent(message, structured_output_model=Triage).structured_output
    triage_agent.messages.clear()          # the classifier keeps no history
    if triage.urgency == "high":
        return {"triage": triage.model_dump(), "answer": "Passed to a human agent."}
    result = support_agent(customer_id)(message, invocation_state={"customer_id": customer_id})
    return {"triage": triage.model_dump(), "answer": str(result).strip()}


print(handle("C-42", "Where is my order ORD-1?"))
print(handle("C-42", "And when will it arrive?"))   # same session: the agent remembers ORD-1
```

## Streaming over HTTP with FastAPI

Server-sent events with text deltas and tool notifications, one agent per request with the history in a session store:

```python
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from strands import Agent, tool
from strands.session import FileSessionManager

app = FastAPI()


@tool
def search_docs(query: str) -> str:
    """Search the product documentation.

    Args:
        query: Search terms.
    """
    return "Exports run nightly at 02:00 UTC and can be triggered from Settings > Data."


class Chat(BaseModel):
    session_id: str
    message: str


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/chat")
async def chat(body: Chat) -> StreamingResponse:
    agent = Agent(
        model="us.anthropic.%%model=anthropic%%",
        tools=[search_docs],
        session_manager=FileSessionManager(session_id=body.session_id, storage_dir="./sessions"),
        callback_handler=None,
    )

    async def events():
        async for event in agent.stream_async(body.message, limits={"turns": 8}):
            if "data" in event:
                yield sse("text", {"delta": event["data"]})
            elif "message" in event and event["message"]["role"] == "assistant":
                for block in event["message"]["content"]:
                    if "toolUse" in block:
                        yield sse("tool", {"name": block["toolUse"]["name"]})
            elif "result" in event:
                yield sse("done", {"stop_reason": event["result"].stop_reason})

    return StreamingResponse(events(), media_type="text/event-stream")


# Try it: uvicorn app:app, then
# curl -N localhost:8000/chat -H 'Content-Type: application/json' -d '{"session_id": "s1", "message": "When do exports run?"}'
```

If the client disconnects, Starlette stops iterating the generator; pass a `cancel_signal` (set from a disconnect check) if you also want the model call itself to stop promptly.

## Approval workflow across requests

A refund over £100 needs a manager. The first request stops with an interrupt and returns its ID; the approval arrives later, in a different request (or process), and a new agent with the same session ID resumes the paused tool call:

```python
from strands import Agent, ToolContext, tool
from strands.session import FileSessionManager


@tool(context=True)
def refund(order_id: str, amount: float, tool_context: ToolContext) -> str:
    """Refund an order.

    Args:
        order_id: Order to refund.
        amount: Amount in GBP.
    """
    if amount > 100:
        decision = tool_context.interrupt("manager-approval", reason={"order_id": order_id, "amount": amount})
        if decision != "approve":
            return f"Refund for {order_id} declined by manager"
    return f"Refunded £{amount:.2f} on {order_id}"


def agent_for(case_id: str) -> Agent:
    return Agent(
        model="us.anthropic.%%model=anthropic%%",
        tools=[refund],
        session_manager=FileSessionManager(session_id=case_id, storage_dir="./sessions"),
        callback_handler=None,
    )


def start(case_id: str, request: str) -> dict:
    result = agent_for(case_id)(request)
    if result.stop_reason == "interrupt":
        pending = [{"id": i.id, "name": i.name, "reason": i.reason} for i in result.interrupts]
        return {"status": "waiting_for_approval", "pending": pending}    # store or send these
    return {"status": "done", "answer": str(result).strip()}


def decide(case_id: str, interrupt_id: str, decision: str) -> dict:
    result = agent_for(case_id)([{"interruptResponse": {"interruptId": interrupt_id, "response": decision}}])
    return {"status": result.stop_reason, "answer": str(result).strip()}


ticket = start("case-981", "Please refund £250 on ORD-7, it arrived broken.")
print(ticket)
# ... hours later, in another request, the manager clicks "approve":
print(decide("case-981", ticket["pending"][0]["id"], "approve"))
```

The tool runs again on resume, so anything before `interrupt()` runs twice; keep side effects after it. With `S3SessionManager` or AgentCore Memory the pause can outlive the process and the host.

## Retrieval over your own documents

A small retrieval tool over local files. Swap the scoring for a vector store or a Bedrock knowledge base in real use; the agent code stays the same:

```python
import re
from pathlib import Path

from strands import Agent, tool

docs = Path("handbook")
docs.mkdir(exist_ok=True)
(docs / "leave.md").write_text("Annual leave: 25 days plus bank holidays. Request leave in the HR portal two weeks ahead.")
(docs / "expenses.md").write_text("Expenses: submit receipts within 30 days. Meals are capped at £25 per day when travelling.")
(docs / "security.md").write_text("Security: lock your screen when away. Report phishing to security@example.com.")


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9£]+", text.lower()))


@tool
def search_handbook(query: str, top_k: int = 2) -> list[dict]:
    """Search the staff handbook and return the most relevant passages with their source file.

    Args:
        query: What to look for.
        top_k: Number of passages to return.
    """
    q = tokens(query)
    scored = sorted(
        ((len(q & tokens(path.read_text())), path) for path in docs.glob("*.md")),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return [{"source": path.name, "text": path.read_text()} for score, path in scored[:top_k] if score > 0]


agent = Agent(
    model="us.anthropic.%%model=anthropic%%",
    system_prompt=(
        "Answer questions about the staff handbook using search_handbook. "
        "Quote the source file for each fact. If the handbook does not cover it, say so."
    ),
    tools=[search_handbook],
    callback_handler=None,
)

print(search_handbook("meal expenses limit"))
print(agent("How much can I spend on meals when travelling?"))
```

For managed retrieval on AWS, the Bedrock Knowledge Bases store (`strands.vended_memory_stores.bedrock_knowledge_base`) plugs into the agent's memory manager.

## Research graph with parallel branches

Three researchers work in parallel, a writer combines them, and the result comes back as structured output:

```python
from pydantic import BaseModel
from strands import Agent
from strands.multiagent import GraphBuilder

FAST = "us.anthropic.%%model=anthropic-small%%"
MAIN = "us.anthropic.%%model=anthropic%%"


class Brief(BaseModel):
    title: str
    key_points: list[str]
    risks: list[str]


def researcher(focus: str) -> Agent:
    return Agent(model=FAST, name=focus, system_prompt=f"Research the {focus} angle of the topic. Be factual and brief.", callback_handler=None)


builder = GraphBuilder()
builder.add_node(Agent(model=FAST, name="scope", system_prompt="Restate the research question and list what to check.", callback_handler=None), "scope")
for focus in ("market", "technical", "regulatory"):
    builder.add_node(researcher(focus), focus)
    builder.add_edge("scope", focus)
builder.add_node(Agent(model=MAIN, name="writer", system_prompt="Write a short, balanced brief from the research.", callback_handler=None), "writer")
for focus in ("market", "technical", "regulatory"):
    builder.add_edge(focus, "writer")
builder.set_entry_point("scope")
builder.set_execution_timeout(600)
builder.set_node_timeout(180)
graph = builder.build()

result = graph("Should a UK retailer adopt drone delivery in 2027?")
draft = str(result.results["writer"].result)

formatter = Agent(model=FAST, callback_handler=None)
brief = formatter(f"Turn this into a brief:\n\n{draft}", structured_output_model=Brief).structured_output
print(brief.title, brief.key_points, brief.risks)
print("tokens:", result.accumulated_usage["totalTokens"])
```

Structured output is applied in a separate step here because graph nodes return their agents' text; you could equally give the writer node `structured_output_model=Brief` at construction.

## Batch extraction with bounded concurrency

Extract fields from many documents at once. Each document gets its own agent (an agent runs one call at a time), and a semaphore keeps you under your Bedrock quota:

```python
import asyncio

from pydantic import BaseModel
from strands import Agent
from strands.types.exceptions import StructuredOutputException


class Invoice(BaseModel):
    supplier: str
    invoice_number: str
    total_gbp: float


DOCUMENTS = [
    "Invoice 1001 from Acme Ltd. Total due: £120.00",
    "Northwind Traders, invoice NW-77, amount £89.50 inc VAT",
    "Receipt from Café Nero, £4.20",
]


async def extract(text: str, limit: asyncio.Semaphore) -> Invoice | None:
    async with limit:
        agent = Agent(model="us.anthropic.%%model=anthropic-small%%", system_prompt="Extract invoice fields exactly as written.", callback_handler=None)
        try:
            result = await agent.invoke_async(text, structured_output_model=Invoice)
            return result.structured_output
        except StructuredOutputException:
            return None


async def main() -> None:
    limit = asyncio.Semaphore(4)
    results = await asyncio.gather(*(extract(doc, limit) for doc in DOCUMENTS))
    for doc, invoice in zip(DOCUMENTS, results):
        print("FAILED" if invoice is None else invoice.model_dump(), "<-", doc[:40])


asyncio.run(main())
```

For very large jobs, Bedrock batch inference is cheaper than many real-time calls, but it is outside the agent loop.

## Cost-based model routing

A small model decides whether a request needs the large one, and prompt caching cuts the cost of the large model's long system prompt:

```python
from strands import Agent
from strands.models import BedrockModel, CacheConfig

SYSTEM = "You are an expert assistant for Example Ltd's internal tools. " * 20   # long, stable prompt

router = Agent(
    model=BedrockModel(model_id="us.anthropic.%%model=anthropic-small%%", max_tokens=8),
    system_prompt="Reply SIMPLE if a short factual answer will do, otherwise COMPLEX. One word.",
    callback_handler=None,
)
small = Agent(model="us.anthropic.%%model=anthropic-small%%", system_prompt=SYSTEM, callback_handler=None)
large = Agent(
    model=BedrockModel(model_id="us.anthropic.%%model=anthropic%%", cache_config=CacheConfig(strategy="auto")),
    system_prompt=SYSTEM,
    callback_handler=None,
)


def answer(question: str) -> tuple[str, str]:
    route = str(router(question)).strip().upper()
    router.messages.clear()
    target = large if route.startswith("COMPLEX") else small
    return route, str(target(question)).strip()


print(answer("What port does the staging database use?"))
```

The agents here keep their conversations between calls; for a stateless service, clear `messages` or build agents per request. `ModelRouter` with `ClassifierStrategy` does the same choice inside a single agent.

## AWS service tools

Narrow tools around boto3 calls keep the agent's permissions obvious and easy to scope in IAM:

```python
import json

import boto3
from strands import Agent, tool

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
sns = boto3.client("sns")


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


@tool
def get_product(sku: str) -> dict:
    """Look up a product by SKU.

    Args:
        sku: Product SKU.
    """
    item = dynamodb.Table("Products").get_item(Key={"sku": sku}).get("Item")
    if item is None:
        raise ValueError(f"No product {sku}")
    return json.loads(json.dumps(item, default=str))   # DynamoDB Decimals to JSON


@tool
def notify_team(topic_arn: str, subject: str, message: str) -> str:
    """Publish a notification to the team's SNS topic.

    Args:
        topic_arn: SNS topic ARN.
        subject: Message subject.
        message: Message body.
    """
    return sns.publish(TopicArn=topic_arn, Subject=subject, Message=message)["MessageId"]


analyst = Agent(
    model="us.anthropic.%%model=anthropic%%",
    system_prompt="You analyse reports in S3 and tell the team about problems.",
    tools=[read_s3_text, get_product, notify_team],
    callback_handler=None,
)
analyst("Read s3://reports/daily.txt and notify arn:aws:sns:us-west-2:123456789012:ops if anything failed.")
```

The matching IAM policy needs only `s3:GetObject` on the report prefix, `dynamodb:GetItem` on the table and `sns:Publish` on the topic. `use_aws` from `strands-agents-tools` can call any AWS API, which is handy for exploration and hard to scope for production.

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-24 | 1.62.0 | New page: support agent with triage, SSE streaming, approvals across requests, retrieval, research graph, batch extraction, model routing and AWS tools. Parts adapted from the former Bedrock Strands page. |

---
title: "Semantic Kernel Advanced Multi‑Agent (Python)"
description: "Advanced multi-agent patterns in Semantic Kernel (Python) — custom group-chat managers, hand-offs with human input, routing, structured results and resilience."
framework: "semantic-kernel"
language: "python"
---

> Verified against `semantic-kernel` %%version=semantic-kernel%% (checked %%versions-checked%%). Orchestrations are experimental in Semantic Kernel; Microsoft Agent Framework's workflows are their supported successor — see the [comprehensive guide](/semantic-kernel-guide/python/semantic_kernel_comprehensive_python/#16-migrating-to-microsoft-agent-framework).

Earlier versions of this page used pre-1.0 APIs (`kernel.add_chat_service`, `ChatCompletionAgent(tools=...)`, `AgentGroupChat(execution_settings=...)`) that no longer exist. Everything below uses the orchestration API.

## Setup

```python
import asyncio
from typing import Annotated
from semantic_kernel.agents import (
    ChatCompletionAgent, GroupChatOrchestration, RoundRobinGroupChatManager,
    HandoffOrchestration, OrchestrationHandoffs,
)
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import AuthorRole, ChatMessageContent
from semantic_kernel.functions import kernel_function

service = OpenAIChatCompletion(ai_model_id="%%model=openai-small%%")   # reads OPENAI_API_KEY

class SearchPlugin:
    @kernel_function(description="Search the web and return result snippets.")
    async def web_search(self, query: Annotated[str, "Search query"]) -> str:
        return f"(results for {query})"   # wrap your provider; add timeouts and retries

def make_agent(name: str, description: str, instructions: str, plugins=None) -> ChatCompletionAgent:
    return ChatCompletionAgent(
        name=name, description=description, instructions=instructions,
        service=service, plugins=plugins or [],
    )

researcher = make_agent("Researcher", "Finds facts with sources",
                        "Research the topic. Cite sources. Be concise.", [SearchPlugin()])
writer = make_agent("Writer", "Drafts copy", "Write clear, short copy from the research.")
reviewer = make_agent("Reviewer", "Checks drafts", "Review the draft. Reply 'APPROVED' if it is good, otherwise list fixes.")
```

## 1. Group chat with a custom manager

`RoundRobinGroupChatManager` just takes turns. Subclass `GroupChatManager` to decide who speaks next and when to stop — here: stop as soon as the reviewer approves.

```python
from semantic_kernel.agents import GroupChatManager, BooleanResult, StringResult, MessageResult
from semantic_kernel.contents import ChatHistory

class ReviewLoopManager(RoundRobinGroupChatManager):
    async def should_terminate(self, chat_history: ChatHistory) -> BooleanResult:
        base = await super().should_terminate(chat_history)    # honours max_rounds
        if base.result:
            return base
        last = chat_history.messages[-1] if chat_history.messages else None
        approved = bool(last and last.name == "Reviewer" and "APPROVED" in str(last.content))
        return BooleanResult(result=approved, reason="Reviewer approved" if approved else "Keep going")

async def run_review_loop(task: str) -> str:
    orchestration = GroupChatOrchestration(
        members=[researcher, writer, reviewer],
        manager=ReviewLoopManager(max_rounds=9),
        agent_response_callback=lambda m: print(f"[{m.name}] {str(m.content)[:80]}"),
    )
    runtime = InProcessRuntime()
    runtime.start()
    try:
        result = await orchestration.invoke(task=task, runtime=runtime)
        return str((await result.get(timeout=300)).content)
    finally:
        await runtime.stop_when_idle()
```

The manager can also override `select_next_agent` (return a `StringResult` naming the agent) and `filter_results` (decide what the orchestration returns). An LLM-driven manager is a manager whose methods call a chat service.

## 2. Hand-off with a human in the loop

```python
triage = make_agent("Triage", "Routes requests", "Work out what the customer needs and hand off.")
billing = make_agent("Billing", "Invoices and refunds", "Handle billing questions. Never promise refunds.")
tech = make_agent("Tech", "Technical support", "Troubleshoot technical issues step by step.")

handoffs = (
    OrchestrationHandoffs()
    .add_many(source_agent=triage.name, target_agents={
        billing.name: "Billing, invoices, refunds",
        tech.name: "Errors, outages, how-to questions",
    })
    .add(source_agent=billing.name, target_agent=triage.name, description="Not a billing issue")
    .add(source_agent=tech.name, target_agent=triage.name, description="Not a technical issue")
)

def ask_human() -> ChatMessageContent:
    return ChatMessageContent(role=AuthorRole.USER, content=input("customer> "))

support = HandoffOrchestration(
    members=[triage, billing, tech],
    handoffs=handoffs,
    human_response_function=ask_human,     # called when an agent needs more input
)
```

In a web service, `human_response_function` can await a queue that your HTTP handler fills, instead of `input()`.

## 3. Deterministic routing in front of the LLM

Keyword or classifier routing is cheaper and testable. Use it to pick the orchestration (or a single agent), and fall back to the LLM triage agent when unsure:

```python
def route(message: str) -> ChatCompletionAgent | None:
    text = message.lower()
    if any(w in text for w in ("invoice", "refund", "charge")):
        return billing
    if any(w in text for w in ("error", "crash", "timeout")):
        return tech
    return None   # ambiguous: let the hand-off orchestration decide

async def answer(message: str) -> str:
    agent = route(message)
    if agent:
        return str((await agent.get_response(messages=message)).message.content)
    runtime = InProcessRuntime(); runtime.start()
    try:
        result = await support.invoke(task=message, runtime=runtime)
        return str(await result.get(timeout=300))
    finally:
        await runtime.stop_when_idle()
```

## 4. Typed results with `output_transform`

```python
from pydantic import BaseModel
from semantic_kernel.agents import ConcurrentOrchestration

class Opinions(BaseModel):
    optimist: str
    sceptic: str

optimist = make_agent("Optimist", "Sees upsides", "Give the strongest case for.")
sceptic = make_agent("Sceptic", "Sees risks", "Give the strongest case against.")

def to_opinions(messages: list[ChatMessageContent]) -> Opinions:
    by_name = {m.name: str(m.content) for m in messages}
    return Opinions(optimist=by_name["Optimist"], sceptic=by_name["Sceptic"])

debate = ConcurrentOrchestration[str, Opinions](members=[optimist, sceptic], output_transform=to_opinions)
```

## 5. Resilience

- **Timeouts.** Always pass `timeout=` to `result.get()`; an orchestration waiting on a model that never answers otherwise waits forever. `result.cancel()` stops it.
- **Retries.** Put retries in plugins (e.g. `tenacity`) for flaky tools; the OpenAI connector already retries rate-limit and 5xx responses.
- **Loop limits.** Set `max_rounds` on group-chat managers; for hand-offs, keep hand-off descriptions specific so agents do not bounce requests between each other.
- **Idempotent tools.** A retried or repeated tool call must not double-charge or double-send.

## 6. Observability and testing

- `agent_response_callback` gives every agent message — log agent name, length and latency with a request ID.
- Semantic Kernel emits OpenTelemetry spans for agent invocations and function calls; see the [middleware page](/semantic-kernel-guide/python/semantic_kernel_middleware_python/) for filters that add your own attributes.
- Test routers and managers as plain Python (they are deterministic); test agents against a stub chat service (subclass `ChatCompletionClientBase` and return canned messages).

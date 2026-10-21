---
title: "LlamaIndex Advanced Agents (Python)"
description: "Multi-agent routing, orchestration, human-in-the-loop approval, durable conversations and large tool sets with LlamaIndex workflow agents."
framework: llamaindex
language: python
---

# LlamaIndex Advanced Agents (Python)

> Verified against `llama-index-core` %%version=llama-index%% (checked %%versions-checked%%). Basics — `FunctionAgent`, tools, `Context` — are covered in the [comprehensive guide](/llamaindex-guide/python/llamaindex_comprehensive_guide/#4-agent-classes-and-types).

All examples assume:

```python
import asyncio
from llama_index.core.agent.workflow import (
    AgentWorkflow, FunctionAgent, AgentOutput, AgentStream, ToolCall, ToolCallResult,
)
from llama_index.core.workflow import Context, InputRequiredEvent, HumanResponseEvent, JsonSerializer
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-5-mini")   # any tool-calling model
```

---

## 1. Routing to specialist agents

`AgentWorkflow` gives each agent a `handoff` tool restricted to the names in `can_handoff_to`. A triage agent with no tools of its own makes a clean router.

```python
def search_docs(query: str) -> str:
    """Search the product documentation."""
    return f"(doc passages about {query})"

def get_invoice(invoice_id: str) -> dict:
    """Fetch an invoice by ID."""
    return {"id": invoice_id, "status": "paid", "amount": 120.0}

triage = FunctionAgent(
    name="triage",
    description="Decides which specialist should handle the request.",
    system_prompt=(
        "Classify the request. Hand off to 'docs' for product questions and "
        "to 'billing' for invoices and payments. Do not answer yourself."
    ),
    tools=[],
    llm=llm,
    can_handoff_to=["docs", "billing"],
)

docs = FunctionAgent(
    name="docs",
    description="Answers product questions from the documentation.",
    system_prompt="Answer from the documentation only. Cite what you used.",
    tools=[search_docs],
    llm=llm,
    can_handoff_to=["triage"],
)

billing = FunctionAgent(
    name="billing",
    description="Handles invoices and payments.",
    system_prompt="Help with invoices. Never promise refunds.",
    tools=[get_invoice],
    llm=llm,
    can_handoff_to=["triage"],
)

support = AgentWorkflow(agents=[triage, docs, billing], root_agent="triage", timeout=120)

async def main():
    handler = support.run(user_msg="Has invoice INV-7 been paid?")
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCall):
            print(f"[{ev.tool_name}] {ev.tool_kwargs}")
    print(await handler)

asyncio.run(main())
```

Routing on keywords in Python (an `if "invoice" in text` router) is cheaper and fully predictable; the LLM router earns its cost when requests are phrased unpredictably. A common compromise is a keyword pre-filter with the triage agent as the fallback.

---

## 2. Orchestrator with agents as tools

Hand-off passes control away; an orchestrator keeps it. Wrap each specialist in a tool and let one agent plan:

```python
researcher = FunctionAgent(
    name="researcher", description="Finds facts.", tools=[search_docs], llm=llm,
    system_prompt="Return concise factual notes with sources.",
)
writer = FunctionAgent(
    name="writer", description="Writes prose.", tools=[], llm=llm,
    system_prompt="Turn notes into a clear, short answer.",
)

async def research(topic: str) -> str:
    """Ask the researcher for notes on a topic."""
    return str(await researcher.run(user_msg=topic))

async def write(notes: str) -> str:
    """Ask the writer to turn notes into a final answer."""
    return str(await writer.run(user_msg=notes))

orchestrator = FunctionAgent(
    tools=[research, write],
    llm=llm,
    system_prompt="Research first, then write. Return the writer's answer verbatim.",
)
```

Each sub-agent call here starts with a fresh context, so sub-agents do not see each other's history — usually what you want, and cheaper in tokens.

---

## 3. Shared state across agents

```python
async def save_finding(ctx: Context, finding: str) -> str:
    """Record a finding for the rest of the team."""
    async with ctx.store.edit_state() as s:
        s["state"]["findings"].append(finding)
    return "Saved."

team = AgentWorkflow(
    agents=[
        FunctionAgent(name="analyst", description="Analyses data.", tools=[save_finding],
                      llm=llm, can_handoff_to=["reporter"]),
        FunctionAgent(name="reporter", description="Writes up the findings.", tools=[],
                      llm=llm),
    ],
    root_agent="analyst",
    initial_state={"findings": []},
    state_prompt="Current state:\n{state}\n\nUser message:\n{msg}",
)
```

`initial_state` is deep-copied for each run (since 0.14.23), so one run's findings never leak into the next. `state_prompt` controls how the state is shown to whichever agent is active.

---

## 4. Human-in-the-loop approval

### 4.1 Inline approval

```python
async def delete_records(ctx: Context, table: str, where: str) -> str:
    """Delete rows from a table. Requires approval."""
    reply = await ctx.wait_for_event(
        HumanResponseEvent,
        waiter_event=InputRequiredEvent(prefix=f"DELETE FROM {table} WHERE {where} — approve? "),
    )
    if reply.response.strip().lower() not in {"y", "yes"}:
        return "Rejected by the operator. Do not retry."
    return f"Deleted rows from {table}."

dba = FunctionAgent(tools=[delete_records], llm=llm)

async def run_with_console_approval(prompt: str):
    handler = dba.run(user_msg=prompt)
    async for ev in handler.stream_events():
        if isinstance(ev, InputRequiredEvent):
            handler.ctx.send_event(HumanResponseEvent(response=input(ev.prefix)))
    return await handler
```

Tell the model what a rejection means ("Do not retry") in the tool's return value; otherwise it may simply ask again.

### 4.2 Approval that outlives the process

When approval comes from a ticket or a chat message hours later, the run has to be stored and resumed in another process. A prebuilt agent paused *inside a tool* cannot currently be resumed that way — its chat memory is skipped when the context is serialised — so model the approval as a small custom workflow whose state is plain data, and call the agent (or the risky action) from its steps:

```python
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent

class ApprovalFlow(Workflow):
    @step
    async def request(self, ctx: Context, ev: StartEvent) -> InputRequiredEvent:
        await ctx.store.set("action", ev.action)
        return InputRequiredEvent(prefix=f"Approve: {ev.action}? ")

    @step
    async def execute(self, ctx: Context, ev: HumanResponseEvent) -> StopEvent:
        action = await ctx.store.get("action")
        if ev.response.strip().lower() != "yes":
            return StopEvent(result=f"Rejected: {action}")
        # ... perform the action, or run an agent to do it ...
        return StopEvent(result=f"Done: {action}")

async def start(action: str) -> dict:
    handler = ApprovalFlow(timeout=None).run(action=action)
    async for ev in handler.stream_events():
        if isinstance(ev, InputRequiredEvent):
            saved = handler.ctx.to_dict(serializer=JsonSerializer())
            await handler.cancel_run()
            return saved                      # persist; show ev.prefix to a human

async def resume(saved: dict, answer: str) -> str:
    flow = ApprovalFlow(timeout=None)
    ctx = Context.from_dict(flow, saved, serializer=JsonSerializer())
    handler = flow.run(ctx=ctx)
    handler.ctx.send_event(HumanResponseEvent(response=answer))
    return await handler
```

`timeout=None` matters: the default workflow timeout would otherwise expire while a human thinks.

---

## 5. Durable conversations

Keep one `Context` per conversation while the process is alive. Across processes, persist the chat history with a database-backed `Memory`, keyed by a stable session ID:

```python
from llama_index.core.memory import Memory

assistant = FunctionAgent(tools=[search_docs], llm=llm)

async def chat(session_id: str, message: str) -> str:
    memory = Memory.from_defaults(
        session_id=session_id,
        async_database_uri="postgresql+asyncpg://app:secret@db/agents",  # or sqlite+aiosqlite:///chat.db
        token_limit=40_000,
    )
    return str(await assistant.run(user_msg=message, memory=memory))
```

Serialising the agent's `Context` (`ctx.to_dict(serializer=JsonSerializer())`) preserves your own state keys but not the chat history, so use it for state and `Memory` for history.

---

## 6. Large tool sets: retrieve tools per query

With dozens of tools, sending every schema on every call wastes tokens and confuses the model. Index the tools and let the agent retrieve the relevant few:

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.core.tools import FunctionTool

all_tools = [FunctionTool.from_defaults(fn=f) for f in (search_docs, get_invoice)]  # imagine 50
tool_index = ObjectIndex.from_objects(all_tools, index_cls=VectorStoreIndex)

agent = FunctionAgent(
    tool_retriever=tool_index.as_retriever(similarity_top_k=3),
    llm=llm,
)
```

---

## 7. Observing multi-agent runs

```python
handler = support.run(user_msg="I was double-charged on INV-9")
async for ev in handler.stream_events():
    match ev:
        case AgentStream(delta=d):
            print(d, end="")
        case ToolCallResult(tool_name=name, tool_output=out):
            print(f"\n  ↳ {name}: {out.content[:60]}")
        case AgentOutput(current_agent_name=who):
            print(f"\n[{who} finished a turn]")
```

For production tracing (OpenTelemetry, Phoenix, Langfuse and others), use the instrumentation hooks described on the [observability page](/llamaindex-guide/python/llamaindex_observability_python/) rather than the event stream.

---

## 8. Recovery

- **Transient tool failures** — retry inside the tool (`tenacity`), and return a clear error string on final failure so the model can choose another path.
- **Model provider errors** — the OpenAI and Anthropic LLM classes take `max_retries` and `timeout`; set them explicitly.
- **Runaway loops** — pass `max_iterations=` to `run()` and set a workflow `timeout=`; use `early_stopping_method="generate"` to get a best-effort answer instead of an exception.
- **Whole-step retries in custom workflows** — `@step(retry_policy=ConstantDelayRetryPolicy(...))`; see [advanced implementations](/llamaindex-guide/python/llamaindex_advanced_implementations/).

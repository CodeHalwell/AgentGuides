---
title: "Chapter 7 — Human-in-the-Loop"
description: "Pause execution for approval, build multi-stage review workflows, multiple interrupts per node, and resume by interrupt ID."
framework: "langgraph"
language: "python"
sidebar:
  label: "7 · Human-in-the-loop"
  order: 7
---

**What you'll learn:** how to pause your graph mid-execution to collect input, approval, or corrections from a human — the `interrupt()` and `Command(resume=...)` primitives, multi-stage approval workflows, multiple interrupts inside one node, resume-by-id when parallel nodes are paused at once, and typed resume values.

Verified against **`langgraph==%%version=langgraph%%`** (module: `langgraph.types`).

**Time:** ~25 minutes.

> Prereqs: [Chapter 5 — Memory & persistence](/langgraph-guide/python/chapter-05-memory/). Interrupts require a checkpointer.

## How `interrupt` works

```
graph.stream(input, cfg)           ← first run
  → node raises GraphInterrupt
  → __interrupt__ event emitted
  → graph.get_state(cfg).interrupts shows Interrupt(value=..., id="...")

graph.stream(Command(resume=...), cfg)   ← resume
  → node re-runs from the top
  → interrupt() returns the resume value instead of raising
  → execution continues normally
```

Key rules:
- A checkpointer is **required** — interrupts persist state so the graph can resume.
- The node **re-runs from the top** on resume. Keep side-effects inside `@task`s (which are skipped on replay) rather than raw code before `interrupt()`.
- Multiple `interrupt()` calls in one node are matched to resume values **by order** within the task. Ids identify tasks, so use them to answer interrupts from different tasks that are pending at the same time.

## Example 1: Basic single interrupt + resume

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class ApprovalState(TypedDict):
    action: str
    amount: float
    decision: str


def request_approval(state: ApprovalState) -> dict:
    # interrupt() pauses here and surfaces the value to the caller.
    # On resume, it returns whatever was passed to Command(resume=...).
    answer = interrupt({
        "question": f"Approve {state['action']} for ${state['amount']:.2f}?",
        "options": ["yes", "no"],
    })
    return {"decision": "approved" if answer == "yes" else "rejected"}


def execute(state: ApprovalState) -> dict:
    return {"action": f"{state['decision'].upper()}: {state['action']}"}


builder = StateGraph(ApprovalState)
builder.add_node("request_approval", request_approval)
builder.add_node("execute", execute)
builder.add_edge(START, "request_approval")
builder.add_edge("request_approval", "execute")
builder.add_edge("execute", END)

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "t1"}}

# --- First run: graph pauses at interrupt ---
for event in graph.stream({"action": "transfer", "amount": 500.0}, cfg):
    print(event)
# {'__interrupt__': (Interrupt(value={...}, id='...', response_schema=None),)}

# Inspect the interrupt
snap = graph.get_state(cfg)
interrupt_obj = snap.interrupts[0]
print(interrupt_obj.value)   # {'question': 'Approve transfer...', 'options': [...]}
print(interrupt_obj.id)      # e.g. 'a3f2...'

# --- Resume: pass the human's answer ---
for event in graph.stream(Command(resume="yes"), cfg):
    print(event)
# {'request_approval': {'decision': 'approved'}}
# {'execute': {'action': 'APPROVED: transfer'}}
```

## Example 2: Multiple interrupts in one node, sequential resume

When a node calls `interrupt()` more than once, each call pauses the graph separately. Resume values are matched **by order** — the first resume answers the first interrupt, the second resume answers the second, and so on.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class ReviewState(TypedDict):
    content: str
    title_ok: bool
    body_ok: bool
    published: bool


def editorial_review(state: ReviewState) -> dict:
    # First interrupt: review the title
    title_answer = interrupt({
        "question": "Is the title acceptable?",
        "title": state["content"][:50],
    })

    # Second interrupt: review the body (only reached after first resume)
    body_answer = interrupt({
        "question": "Is the body acceptable?",
        "preview": state["content"][:200],
    })

    return {
        "title_ok": title_answer == "yes",
        "body_ok": body_answer == "yes",
    }


def publish(state: ReviewState) -> dict:
    if state["title_ok"] and state["body_ok"]:
        return {"published": True}
    return {"published": False}


builder = StateGraph(ReviewState)
builder.add_node("review", editorial_review)
builder.add_node("publish", publish)
builder.add_edge(START, "review")
builder.add_edge("review", "publish")
builder.add_edge("publish", END)

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "review-1"}}

# Run 1: graph pauses at the title interrupt
list(graph.stream({"content": "My Article: full body here...", "title_ok": False, "body_ok": False, "published": False}, cfg))

# Resume 1: answer the title question → graph pauses at the body interrupt
list(graph.stream(Command(resume="yes"), cfg))

# Resume 2: answer the body question → graph continues to "publish"
list(graph.stream(Command(resume="yes"), cfg))

final = graph.get_state(cfg)
print(final.values["published"])   # True
```

## Example 3: Resume by interrupt id

Every `interrupt()` inside one task shares that task's id, so ids do not distinguish the first and second interrupt of the same node (Example 2 resumes those by order). Ids matter when **several tasks are paused at once**, typically parallel nodes. Each pending interrupt then has its own id, and `Command(resume={id: value, ...})` answers them in one call.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class SignoffState(TypedDict):
    proposal: str
    legal_ok: bool
    finance_ok: bool


def legal_review(state: SignoffState) -> dict:
    answer = interrupt({"department": "legal", "item": state["proposal"]})
    return {"legal_ok": answer == "approved"}


def finance_review(state: SignoffState) -> dict:
    answer = interrupt({"department": "finance", "item": state["proposal"]})
    return {"finance_ok": answer == "approved"}


builder = StateGraph(SignoffState)
builder.add_node("legal", legal_review)
builder.add_node("finance", finance_review)
# Both reviews start from START, so they run in the same step, in parallel.
builder.add_edge(START, "legal")
builder.add_edge(START, "finance")
builder.add_edge("legal", END)
builder.add_edge("finance", END)

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "signoff-1"}}

# First run: both nodes pause, so two interrupts are pending at once.
graph.invoke({"proposal": "Acquire WidgetCo", "legal_ok": False, "finance_ok": False}, cfg)

pending = graph.get_state(cfg).interrupts
for intr in pending:
    print(intr.id, intr.value["department"])

# Answer both in one call, keyed by interrupt id.
answers = {intr.id: "approved" for intr in pending}
result = graph.invoke(Command(resume=answers), cfg)
print(result["legal_ok"], result["finance_ok"])   # True True
```

If you answer only some of the ids, those tasks finish and the rest stay pending: `get_state(cfg).interrupts` then lists only the unanswered ones, and you resume them later in the same way. A UI that shows each pending approval separately can therefore send each answer as it arrives.

## Example 4: Multi-step approval workflow

Each node in the chain owns one interrupt. Each resume drives the graph one step forward.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


def merge_dicts(x: dict, y: dict) -> dict:
    return {**x, **y}


class WorkflowState(TypedDict):
    action: str
    amount: float
    # Named function instead of a lambda — lambdas are not picklable and will
    # fail with serializing checkpointers (SqliteSaver, PostgresSaver).
    approvals: Annotated[dict, merge_dicts]
    outcome: str


def manager_review(state: WorkflowState) -> dict:
    answer = interrupt({
        "stage": "manager",
        "question": f"Approve {state['action']} for ${state['amount']:.2f}?",
    })
    return {"approvals": {"manager": answer == "yes"}}


def compliance_review(state: WorkflowState) -> dict:
    if not state["approvals"].get("manager"):
        return {"approvals": {"compliance": False}}
    answer = interrupt({"stage": "compliance", "question": "Compliance sign-off?"})
    return {"approvals": {"compliance": answer == "yes"}}


def execute_workflow(state: WorkflowState) -> dict:
    if all(state["approvals"].values()):
        return {"outcome": f"EXECUTED: {state['action']}"}
    return {"outcome": f"REJECTED: {state['action']}"}


builder = StateGraph(WorkflowState)
builder.add_node("manager", manager_review)
builder.add_node("compliance", compliance_review)
builder.add_node("execute", execute_workflow)
builder.add_edge(START, "manager")
builder.add_edge("manager", "compliance")
builder.add_edge("compliance", "execute")
builder.add_edge("execute", END)

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "workflow-1"}}

initial = {"action": "vendor-contract", "amount": 50000.0, "approvals": {}, "outcome": ""}

# Step 1: manager interrupt
list(graph.stream(initial, cfg))
# Step 2: compliance interrupt
list(graph.stream(Command(resume="yes"), cfg))
# Step 3: final execution
list(graph.stream(Command(resume="yes"), cfg))

print(graph.get_state(cfg).values["outcome"])  # EXECUTED: vendor-contract
```

## Example 5: `Command(goto=...)` after interrupt — conditional routing

Combine `interrupt` with `Command` to both resume and redirect in one step.

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class EditState(TypedDict):
    draft: str
    final: str


def review_draft(state: EditState) -> Command:
    decision = interrupt({
        "question": "Accept, reject, or edit this draft?",
        "draft": state["draft"],
        "options": ["accept", "reject", "edit"],
    })
    if decision == "accept":
        return Command(update={"final": state["draft"]}, goto="publish")
    elif decision == "edit":
        return Command(goto="revise")
    else:
        return Command(update={"final": ""}, goto=END)


def revise(state: EditState) -> dict:
    revised = interrupt({"question": "Enter your revised draft:"})
    return {"draft": revised}


def publish(state: EditState) -> dict:
    return {"final": f"[PUBLISHED] {state['final']}"}


builder = StateGraph(EditState)
builder.add_node("review", review_draft, destinations=("publish", "revise", END))
builder.add_node("revise", revise)
builder.add_node("publish", publish)
builder.add_edge(START, "review")
builder.add_edge("revise", "review")
builder.add_edge("publish", END)

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "edit-1"}}

# Start: pauses at review
list(graph.stream({"draft": "First draft text", "final": ""}, cfg))
# Human chooses "edit" → goes to revise
list(graph.stream(Command(resume="edit"), cfg))
# Human enters revised text → back to review
list(graph.stream(Command(resume="Improved draft text"), cfg))
# Human accepts → published
list(graph.stream(Command(resume="accept"), cfg))

print(graph.get_state(cfg).values["final"])   # [PUBLISHED] Improved draft text
```

## Example 6: Typed resume values with `response_schema`

`interrupt(value, response_schema=...)` declares the shape of the answer you expect. With a Pydantic model, `TypedDict` or dataclass, the resume value is validated and `interrupt` returns the validated object; the schema is also stored on the `Interrupt` as JSON Schema so a client can render a form from it.

```python
from pydantic import BaseModel
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class Decision(BaseModel):
    approve: bool
    reason: str


class RefundState(TypedDict):
    amount: float
    outcome: str


def review_refund(state: RefundState) -> dict:
    decision = interrupt({"amount": state["amount"]}, response_schema=Decision)
    # decision is a Decision instance, not a dict
    verdict = "refunded" if decision.approve else "declined"
    return {"outcome": f"{verdict}: {decision.reason}"}


graph = (
    StateGraph(RefundState)
    .add_node("review", review_refund)
    .add_edge(START, "review")
    .add_edge("review", END)
    .compile(checkpointer=InMemorySaver())
)
cfg = {"configurable": {"thread_id": "refund-1"}}

graph.invoke({"amount": 42.0, "outcome": ""}, cfg)
print(graph.get_state(cfg).interrupts[0].response_schema["required"])   # ['approve', 'reason']

try:
    graph.invoke(Command(resume={"approve": "maybe"}), cfg)   # invalid: fails validation
except Exception as exc:
    print(type(exc).__name__)                                  # ValidationError

result = graph.invoke(Command(resume={"approve": True, "reason": "damaged in transit"}), cfg)
print(result["outcome"])   # refunded: damaged in transit
```

A rejected resume value saves nothing, so the interrupt stays pending and can be answered again. A plain JSON Schema `dict` is passed to clients as-is but not used for validation. See [typed resume values](/langgraph-guide/python/reference-command-and-send/#typed-resume-values-response_schema) for the details.

## Gotchas

- **Re-execution on resume.** The node runs from the top every time. Any code before `interrupt()` executes again — make side effects idempotent or move them into `@task`s (which are cache-skipped on replay).
- **No checkpointer = no resume.** Without a checkpointer, `invoke` still stops at `interrupt()` and returns the state with an `__interrupt__` key, but nothing is saved, so `Command(resume=...)` has nothing to resume. Always compile with a checkpointer (`InMemorySaver()` in tests) and pass a `thread_id`.
- **`state.next` vs `state.interrupts`.** When paused, `graph.get_state(cfg).next` names the paused node (for example `('ask',)`); `state.interrupts` is the tuple of `Interrupt(value=..., id=...)` objects.
- **Resume by id requires the correct id.** Pass the id exactly as reported by `Interrupt.id`. A mapping whose keys are not interrupt ids is not rejected: with a single pending interrupt, the whole dict becomes the resume value (checked against %%version=langgraph%%).
- **Interrupts in one node share an id.** The id identifies the task, so a node that calls `interrupt()` twice reports the same id both times; resume those in order.
- **`Command(resume=scalar)` answers the next interrupt in order.** With interrupts from more than one task pending, a non-mapping resume value raises `RuntimeError: When there are multiple pending interrupts, you must specify the interrupt id when resuming.` Use `Command(resume={id: value})` there.



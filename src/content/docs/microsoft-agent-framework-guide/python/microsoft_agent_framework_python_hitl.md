---
title: "Microsoft Agent Framework (Python) — Human in the Loop"
description: "Pause a workflow for human input with ctx.request_info and @response_handler, or with SequentialBuilder.with_request_info. Plus tool approval gates."
framework: microsoft-agent-framework
language: python
---

# Human in the Loop — Python

Agent Framework ships three human-in-the-loop (HITL) patterns:

1. **Workflow-level `request_info`** — any `Executor` can pause and emit a typed request; the caller replies via `workflow.run(responses={...})`.
2. **Orchestration HITL** — `SequentialBuilder.with_request_info(...)`, `MagenticBuilder(enable_plan_review=True)`, `GroupChatBuilder.with_request_info(...)` bolt HITL onto prebuilt orchestrations.
3. **Tool approval** — individual tools (including MCP tools) can require approval before running.

All three surface the same `"request_info"` / `"function_approval_request"` events, so a single event loop in the caller can handle any of them.

Verified against `agent-framework-core==1.2.2`.

## Workflow-level `request_info`

An executor calls `await ctx.request_info(data, response_type)`. The workflow pauses and emits an event. The caller sends a response keyed by request ID; a matching `@response_handler` on the same executor receives it and decides what to do next.

```python
import asyncio
from dataclasses import dataclass
from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    handler,
    response_handler,
)


@dataclass
class ClarificationQuestion:
    topic: str
    options: list[str]


class Researcher(Executor):
    @handler
    async def start(self, topic: str, ctx: WorkflowContext[str, str]) -> None:
        # Ask the human which angle to take before we invest any tokens.
        await ctx.request_info(
            ClarificationQuestion(topic=topic, options=["technical", "business", "historical"]),
            response_type=str,
        )

    @response_handler
    async def on_angle_chosen(
        self,
        original_request: ClarificationQuestion,
        response: str,
        ctx: WorkflowContext[str, str],
    ) -> None:
        await ctx.yield_output(
            f"Researching '{original_request.topic}' with a {response} angle…",
        )


workflow = WorkflowBuilder(start_executor=Researcher()).build()
```

### Driving the loop from the caller

```python
async def run_with_human(topic: str) -> None:
    stream = workflow.run(topic, stream=True)
    while True:
        pending: dict[str, str] = {}
        async for event in stream:
            if event.type == "request_info":
                question: ClarificationQuestion = event.data
                print(f"{question.topic} — choose: {question.options}")
                # input() blocks the event loop; off-load to a worker thread.
                user_input = await asyncio.to_thread(input, "angle: ")
                pending[event.request_id] = user_input.strip()
            elif event.type == "output":
                print("Done:", event.data)
                return
        if not pending:
            return
        # Resume the workflow with the human's answers. Keys are request IDs.
        stream = workflow.run(responses=pending, stream=True)
```

Key points:

- `ctx.request_info(data, response_type)` works with **any** dataclass or Pydantic model as `data`. The `response_type` is used for validation when the response arrives.
- Responses are keyed by `request_id` (a UUID unless you passed one explicitly to `request_info(..., request_id=...)`).
- You can answer **some** requests and leave others pending — only the matching `response_handler` fires; unanswered requests stay in the queue.
- `workflow.run(responses=..., stream=True)` resumes; pass `checkpoint_id=...` alongside to resume from a persisted checkpoint.

### Minimal non-streaming shape

```python
result = await workflow.run("quantum sensors")
for evt in result.get_request_info_events():
    ...   # collect
result = await workflow.run(responses={"<id>": "technical"})
```

### Explicit response-handler types

`@response_handler` defaults to introspecting parameter annotations. When you're using forward references (the request/response classes are imported lazily), or you're building executors dynamically and don't want to lock the parameter types, switch to the **explicit-types** form. **All** types must come from decorator parameters in this mode — annotation-based introspection is disabled.

```python
from agent_framework import Executor, WorkflowContext, handler, response_handler


class Approver(Executor):
    @handler
    async def submit(self, draft: str, ctx: WorkflowContext[str, str]) -> None:
        await ctx.request_info(Approval(draft=draft), response_type=bool)

    # Required: request= and response=. Optional: output= (for ctx.send_message)
    # and workflow_output= (for ctx.yield_output). String forward references
    # (e.g. request="Approval") resolve against the decorated function's globals.
    @response_handler(request=Approval, response=bool, workflow_output=str)
    async def on_decision(self, original_request, approved, ctx):
        await ctx.yield_output("approved" if approved else "rejected")
```

When you mix the two modes the framework raises at registration — be explicit about which one you want. Explicit forward-reference example for a request type imported in another module:

```python
@response_handler(request="my_app.requests:BudgetCheck", response=bool)
async def on_budget(self, original_request, approved, ctx): ...
```

Inspect what handlers are registered on an executor at runtime via the `is_request_supported(request_type, response_type)` method that `RequestInfoMixin` adds:

```python
executor = Approver(id="approver")
assert executor.is_request_supported(Approval, bool)         # True
assert not executor.is_request_supported(Approval, str)      # different response type
```

Useful for unit-testing wiring before you stand up the workflow.

## Orchestration HITL

### Sequential — approve after each stage

```python
from agent_framework_orchestrations import SequentialBuilder

workflow = (
    SequentialBuilder(participants=[researcher, analyst, writer])
    .with_request_info(agents=[analyst])   # pause only after the analyst
    .build()
)
```

At each configured pause point the workflow emits a `request_info` event carrying the conversation so far. Reply with:

- a string — injected as human guidance for the next agent, or
- `None` — continue unchanged.

### GroupChat — approve a selected speaker

`GroupChatBuilder.with_request_info(agents=[...])` behaves the same way.

### Magentic — plan review & stall intervention

Magentic offers structured HITL hooks tailored to its planning loop:

```python
from agent_framework_orchestrations import MagenticBuilder

workflow = (
    MagenticBuilder(
        participants=[researcher, analyst, writer],
        manager_agent=manager_agent,
        enable_plan_review=True,        # pause after initial plan
        max_stall_count=3,              # how many stalled rounds before HITL fires
        max_round_count=20,             # absolute upper bound — fail fast on runaway
        max_reset_count=2,              # cap how often the manager replans
    )
    .with_human_input_on_stall()        # intervene instead of auto-replanning
    .build()
)
```

These emit `MagenticHumanInterventionRequest` events with `kind=PLAN_REVIEW` or `kind=STALL`. Respond with a `MagenticHumanInterventionReply` containing a decision:

- `APPROVE` — continue with the plan.
- `REVISE` — pass a revised plan.
- `REPLAN` — force the manager to replan.
- `GUIDANCE` — attach free-text guidance for the manager.

#### Driving plan review end-to-end

The shape of the loop is identical to workflow-level `request_info` — the only difference is the typed reply object:

```python
from agent_framework_orchestrations import (
    MagenticBuilder,
    MagenticPlanReviewRequest,
    MagenticPlanReviewResponse,
)


async def review_loop(workflow, task: str) -> str:
    pending: dict[str, MagenticPlanReviewResponse] = {}
    stream = workflow.run(task, stream=True)

    while True:
        async for event in stream:
            if event.type == "request_info" and isinstance(event.data, MagenticPlanReviewRequest):
                request: MagenticPlanReviewRequest = event.data
                print("Plan:\n", request.plan_text)
                choice = await ask_user(request)              # your UX

                if choice == "approve":
                    pending[event.request_id] = request.approve()
                elif choice == "revise":
                    feedback = await prompt_user("How should the plan change?")
                    pending[event.request_id] = request.revise(feedback)
                # If the user dithers, leave it pending — workflow stays paused.
            elif event.type == "output":
                return event.data

        if not pending:
            return ""
        stream = workflow.run(responses=pending, stream=True)
        pending = {}
```

`MagenticPlanReviewRequest.approve()` and `.revise(feedback)` return the matching reply — no need to construct one manually. `feedback` accepts a string, a list of strings, a `Message`, or a list of messages, so you can attach structured guidance (e.g. "Add: validate against EU regulations").

#### Custom manager prompts

The `StandardMagenticManager` accepts overrides for every prompt in the planning loop. Use them to nudge the manager toward your domain's vocabulary or to enforce a particular plan format:

```python
workflow = (
    MagenticBuilder(
        participants=[researcher, analyst, writer],
        manager_agent=manager_agent,
        task_ledger_facts_prompt=(
            "Extract verifiable facts about the engineering problem only — ignore organisational context."
        ),
        task_ledger_plan_prompt=(
            "Produce a numbered plan. Each step must name exactly one specialist and one expected artefact."
        ),
        progress_ledger_prompt=(
            "For each step, mark COMPLETED, IN_PROGRESS, or BLOCKED. If any step is BLOCKED, name the unblocker."
        ),
        final_answer_prompt=(
            "Synthesize the conversation into a one-page brief with sections: Decision, Rationale, Risks, Next steps."
        ),
        enable_plan_review=True,
    )
    .build()
)
```

Useful when the default prompts produce plans that are too generic, too verbose, or don't match the artefacts your downstream tooling expects.

#### Bring your own manager

For deterministic planning, subclass `MagenticManagerBase` and pass `manager=`. This is the right escape hatch when the LLM-driven planner makes the same mistake every time and your domain has a clear policy:

```python
from agent_framework_orchestrations import MagenticManagerBase, MagenticContext


class PolicyManager(MagenticManagerBase):
    """Hard-coded plan: researcher first, analyst second, writer last."""

    async def plan(self, context: MagenticContext) -> list[str]:
        return [
            "researcher: collect 5 reference papers",
            "analyst: extract claims and evidence",
            "writer: produce one-page brief",
        ]

    async def select_next_speaker(self, context: MagenticContext) -> str | None:
        # Round-robin in plan order — no LLM needed.
        for step in context.progress_ledger.steps:
            if not step.completed:
                return step.assignee
        return None    # all steps complete

    async def assess_progress(self, context: MagenticContext) -> bool:
        return all(s.completed for s in context.progress_ledger.steps)


workflow = MagenticBuilder(
    participants=[researcher, analyst, writer],
    manager=PolicyManager(),
).build()
```

Because the manager is your code, it can also drive HITL — emit a `request_info` from inside `plan()` to require human sign-off on the policy itself.

## Tool approval

Any tool — plain function or MCP — can require approval. Approval events fire before the tool runs.

```python
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient


@tool(approval_mode="always_require")
def delete_file(path: str) -> str:
    import os
    os.remove(path)
    return f"deleted {path}"


agent = Agent(client=OpenAIChatClient(), tools=[delete_file])
```

Driving approval from the caller:

```python
stream = agent.run("Please remove /tmp/temp.log", stream=True)
async for update in stream:
    if update.type == "function_approval_request":
        # Inspect the proposed call and decide
        proposal = update.data
        approval = proposal.to_function_approval_response(approved=True)
        await stream.send_response(approval)  # resume
    elif update.type == "message":
        print(update.text)
```

For MCP tools, combine `approval_mode="always_require"` (or the per-tool dict form) on the MCP tool constructor with the same loop — every MCP call emits the same event type.

## Patterns

**Pre-flight clarification.** Use a workflow-level `request_info` at the front of the pipeline to ask the user what they actually want before spending tokens on research.

**Cost guardrail.** Middleware tracks tokens; when a budget threshold is crossed it calls `ctx.request_info(BudgetCheck(spend=...), response_type=bool)` and halts the workflow until a human approves continuation.

**Two-stage release.** Combine Magentic's `enable_plan_review` with `SequentialBuilder.with_request_info` in a sub-workflow to get human approval both on the plan AND on the final report.

**CI resume from checkpoint.** Persist the workflow with `FileCheckpointStorage`; when the human review PR merges, a CI job calls `workflow.run(responses={...}, checkpoint_id=...)` to pick up exactly where the agent left off.

## Resume from checkpoint + responses

Checkpointing and HITL compose:

```python
from agent_framework import FileCheckpointStorage, WorkflowBuilder

storage = FileCheckpointStorage(base_path="/var/lib/agents/checkpoints")
workflow = WorkflowBuilder(start_executor=Researcher(), checkpoint_storage=storage).build()

# First run — human walks away mid-flow.
stream = workflow.run("topic", stream=True)
async for event in stream:
    ...

# Later, pick the latest checkpoint for this workflow and resume with replies.
checkpoints = await storage.list_checkpoints(workflow_name=workflow.name)
latest = checkpoints[-1]
resumed = workflow.run(
    checkpoint_id=latest.checkpoint_id,
    responses={"<request-id>": "technical"},
    stream=True,
)
```

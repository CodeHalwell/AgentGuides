---
title: "Microsoft Agent Framework (Python) — Human in the Loop"
description: "Pause a workflow for human input with ctx.request_info and @response_handler, or with SequentialBuilder.with_request_info. Plus tool approval gates."
framework: "microsoft-agent-framework"
language: "python"
sidebar:
  label: "Human in the loop"
  order: 22
---

Agent Framework ships three human-in-the-loop (HITL) patterns:

1. **Workflow-level `request_info`** — any `Executor` can pause and emit a typed request; the caller replies via `workflow.run(responses={...})`.
2. **Orchestration HITL** — `SequentialBuilder.with_request_info(...)`, `MagenticBuilder(enable_plan_review=True)`, `GroupChatBuilder.with_request_info(...)` bolt HITL onto prebuilt orchestrations.
3. **Tool approval** — individual tools (including MCP tools) can require approval before running.

All three surface the same `"request_info"` / `"function_approval_request"` events, so a single event loop in the caller can handle any of them.

Verified against `agent-framework` %%version=agent-framework%% (re-checked %%versions-checked%%; first written for 1.5.0).

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


workflow = WorkflowBuilder(start_executor=Researcher(id="researcher")).build()
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
from dataclasses import dataclass

from agent_framework import Executor, WorkflowContext, handler, response_handler


@dataclass
class Approval:
    draft: str


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

Once you pass any explicit type, parameter annotations are ignored, and both `request=` and `response=` must be given (`ValueError: Response handler on_decision with explicit type parameters must specify 'response' type` otherwise). Types can also be given as strings. They are evaluated against the module's globals when the decorator runs, so the name must already be defined or imported at that point (`"module:qualname"` paths are not accepted):

```python
from dataclasses import dataclass
from agent_framework import Executor, WorkflowContext, handler, response_handler


@dataclass
class BudgetCheck:
    spend: float


class BudgetGate(Executor):
    @handler
    async def check(self, spend: float, ctx: WorkflowContext) -> None:
        await ctx.request_info(BudgetCheck(spend), response_type=bool)

    @response_handler(request="BudgetCheck", response=bool)
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

At each configured pause point the workflow emits a `request_info` event whose `data` is the paused agent's `AgentExecutorResponse` (read `event.data.agent_response.text` to show the draft). The reply must be an `AgentRequestInfoResponse`; any other type is rejected with `ValueError: Response type mismatch`:

- `AgentRequestInfoResponse.approve()` accepts the agent's response and the pipeline continues;
- `AgentRequestInfoResponse.from_strings(["Focus on cost."])` (or `from_messages([...])`) sends the feedback back to **the same agent**, which runs again and pauses again for review.

```python
from agent_framework_orchestrations import AgentRequestInfoResponse, SequentialBuilder

workflow = SequentialBuilder(participants=[researcher, analyst, writer]).with_request_info(agents=[analyst]).build()

result = await workflow.run("Heat pumps in UK homes")
while requests := result.get_request_info_events():
    replies = {}
    for event in requests:
        print(event.data.agent_response.text)                     # the analyst's draft
        feedback = input("Feedback (empty to approve): ").strip()
        replies[event.request_id] = (
            AgentRequestInfoResponse.from_strings([feedback]) if feedback else AgentRequestInfoResponse.approve()
        )
    result = await workflow.run(responses=replies)

print(result.get_outputs()[-1].text)
```

### GroupChat — approve a selected speaker

`GroupChatBuilder.with_request_info(agents=[...])` and `ConcurrentBuilder.with_request_info(agents=[...])` use the same request and reply types.

### Magentic — plan review & stall intervention

Magentic offers structured HITL hooks tailored to its planning loop:

```python
from agent_framework_orchestrations import MagenticBuilder

workflow = (
    MagenticBuilder(
        participants=[researcher, analyst, writer],
        manager_agent=manager_agent,
        enable_plan_review=True,        # pause after the initial plan (and after each stall replan)
        max_stall_count=3,              # stalled rounds before the manager replans
        max_round_count=20,             # absolute upper bound — fail fast on runaway
        max_reset_count=2,              # cap how often the manager replans
    )
    .build()
)
```

With plan review enabled, the workflow emits a `request_info` event whose data is a `MagenticPlanReviewRequest` (fields `plan`, `current_progress`, `is_stalled`) — once after the initial plan and again whenever a stall forces a replan (`is_stalled=True`). Reply with `request.approve()` to continue or `request.revise(feedback)` to send the manager back with guidance.

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
                print("Plan:\n", request.plan.text)
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

For deterministic planning, subclass `MagenticManagerBase` and pass `manager=`. This is the right escape hatch when the LLM-driven planner makes the same mistake every time and your domain has a clear policy. The base class has four abstract methods — `plan`, `replan`, `create_progress_ledger` and `prepare_final_answer` — and the orchestrator reads the next speaker from the `MagenticProgressLedger` you return:

```python
from agent_framework import Message
from agent_framework.orchestrations import (
    MagenticBuilder,
    MagenticContext,
    MagenticManagerBase,
    MagenticProgressLedger,
    MagenticProgressLedgerItem,
)

PLAN = ["researcher", "analyst", "writer"]   # participant names, in order


class PolicyManager(MagenticManagerBase):
    """Hard-coded plan: researcher first, analyst second, writer last."""

    async def plan(self, magentic_context: MagenticContext) -> Message:
        return Message("assistant", ["Plan: researcher -> analyst -> writer."])

    async def replan(self, magentic_context: MagenticContext) -> Message:
        return await self.plan(magentic_context)

    async def create_progress_ledger(self, magentic_context: MagenticContext) -> MagenticProgressLedger:
        # Round-robin in plan order — no LLM needed.
        step = magentic_context.round_count
        done = step >= len(PLAN)
        speaker = PLAN[min(step, len(PLAN) - 1)]

        def item(answer, reason="fixed policy"):
            return MagenticProgressLedgerItem(reason=reason, answer=answer)

        return MagenticProgressLedger(
            is_request_satisfied=item(done),
            is_in_loop=item(False),
            is_progress_being_made=item(True),
            next_speaker=item(speaker, "round-robin in plan order"),
            instruction_or_question=item(f"{speaker}: complete your step of the plan."),
        )

    async def prepare_final_answer(self, magentic_context: MagenticContext) -> Message:
        history = magentic_context.chat_history
        return Message("assistant", [history[-1].text if history else ""])


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

Driving approval from the caller. The run returns with the pending calls in `response.user_input_requests`; reply on the same session with a message carrying the approval responses:

```python
from agent_framework import Message

session = agent.create_session()
response = await agent.run("Please remove /tmp/temp.log", session=session)

while response.user_input_requests:
    replies = []
    for request in response.user_input_requests:      # Content of type "function_approval_request"
        call = request.function_call
        print(f"Approve {call.name}({call.arguments})?")
        replies.append(request.to_function_approval_response(approved=True))
    response = await agent.run(Message("user", replies), session=session)

print(response.text)
```

When streaming, each `AgentResponseUpdate` exposes the same `user_input_requests` property, and `await stream.get_final_response()` gives you the joined response to run the loop above on. The stream itself has no method for answering a request; reply in a new `run()` on the same session.

For MCP tools, combine `approval_mode="always_require"` (or the per-tool dict form) on the MCP tool constructor with the same loop — every MCP call surfaces as the same `function_approval_request` content.

## Patterns

**Pre-flight clarification.** Use a workflow-level `request_info` at the front of the pipeline to ask the user what they actually want before spending tokens on research.

**Cost guardrail.** An executor between pipeline stages reads the accumulated `usage_details` and, when a budget threshold is crossed, calls `ctx.request_info(BudgetCheck(spend=...), response_type=bool)`, halting the workflow until a human approves continuation. (Middleware has no `WorkflowContext`, so the pause has to come from an executor.)

**Two-stage release.** Combine Magentic's `enable_plan_review` with `SequentialBuilder.with_request_info` in a sub-workflow to get human approval both on the plan AND on the final report.

**CI resume from checkpoint.** Persist the workflow with `FileCheckpointStorage`; when the human review PR merges, a CI job calls `workflow.run(responses={...}, checkpoint_id=...)` to pick up exactly where the agent left off.

## Resume from checkpoint + responses

Checkpointing and HITL compose:

```python
from agent_framework import FileCheckpointStorage, WorkflowBuilder

storage = FileCheckpointStorage("/var/lib/agents/checkpoints")
workflow = WorkflowBuilder(start_executor=Researcher(id="researcher"), checkpoint_storage=storage).build()

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

## Revision history

| Date | Version | Changes |
|---|---|---|
| 2026-10-24 | 1.22.0 | Corrected the orchestration review reply: `with_request_info` expects an `AgentRequestInfoResponse` (`approve()`, `from_strings()`, `from_messages()`), not a string or `None`; added a complete review loop. Corrected the cost-guardrail pattern (only executors can call `request_info`). |

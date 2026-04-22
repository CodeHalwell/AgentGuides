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

Verified against `agent-framework-core==1.1.0`.

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

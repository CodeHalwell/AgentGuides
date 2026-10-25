---
title: "OpenAI Agents SDK: Hooks, Tool Guardrails and Approvals (Python)"
description: "Lifecycle hooks (RunHooks, AgentHooks), tool input and output guardrails, human-in-the-loop approvals with RunState, and model-input filters in the Python Agents SDK."
framework: "openai-agents-sdk"
language: "python"
sidebar:
  label: "Hooks, guardrails & approvals"
  order: 2
---

> Verified against `openai-agents` %%version=openai-agents%% (checked %%versions-checked%%). Every example below was run with the SDK's `agents.testing.ScriptedModel` in place of a live model.

The SDK has no generic "middleware" chain. Instead it gives you a small number of interception points, each tied to one part of the agent loop:

| Control | Attach to | Runs | Can it stop the run? |
|---------|-----------|------|----------------------|
| `RunHooks` | `Runner.run(..., hooks=...)` | Every agent, model call, tool call and handoff in the run | No (observe only; raise to abort) |
| `AgentHooks` | `Agent(hooks=...)` | Only while that agent is active | No (observe only) |
| Input / output guardrails | `Agent(input_guardrails=..., output_guardrails=...)` or `RunConfig` | Before / after the agent's turn | Yes: raises `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered` |
| Tool input / output guardrails | `@function_tool(tool_input_guardrails=..., tool_output_guardrails=...)` | Around one function-tool call | Yes: replace the tool result, or raise |
| `needs_approval` | `@function_tool(needs_approval=...)`, `HostedMCPTool`, shell tools | Before the tool executes | Pauses the run until a person approves or rejects |
| `call_model_input_filter` | `RunConfig` | Just before every model call | Edits what the model sees |
| Handoff `input_filter` | `handoff(agent, input_filter=...)` | When control passes to another agent | Edits the history the next agent sees |

Agent-level input and output guardrails are covered in the [comprehensive guide](/openai-agents-sdk-guides/openai_agents_sdk_comprehensive_guide/#guardrails). This page covers the rest.

## Run hooks

Subclass `RunHooks` and override only the callbacks you need. Each is an `async` method. For function tools the `context` passed to `on_tool_start` / `on_tool_end` is a `ToolContext`, which carries `tool_name`, `tool_call_id` and the raw `tool_arguments`.

```python
import asyncio
import time
from typing import Any

from agents import Agent, RunContextWrapper, RunHooks, Runner, Tool, function_tool
from agents.tool_context import ToolContext


class AuditHooks(RunHooks):
    def __init__(self) -> None:
        self.started: dict[str, float] = {}

    async def on_agent_start(self, context, agent: Agent) -> None:
        print(f"[agent] {agent.name} started")

    async def on_llm_end(self, context: RunContextWrapper, agent: Agent, response) -> None:
        print(f"[llm] {agent.name}: {response.usage.total_tokens} tokens")

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        if isinstance(context, ToolContext):
            self.started[context.tool_call_id] = time.perf_counter()
            print(f"[tool] {context.tool_name}({context.tool_arguments})")

    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: Any
    ) -> None:
        if isinstance(context, ToolContext):
            elapsed = time.perf_counter() - self.started.pop(context.tool_call_id, time.perf_counter())
            print(f"[tool] {context.tool_name} finished in {elapsed:.3f}s")

    async def on_handoff(self, context: RunContextWrapper, from_agent: Agent, to_agent: Agent) -> None:
        print(f"[handoff] {from_agent.name} -> {to_agent.name}")

    async def on_agent_end(self, context, agent: Agent, output: Any) -> None:
        print(f"[agent] {agent.name} finished; run usage so far: {context.usage.total_tokens}")


@function_tool
def get_order_status(order_id: str) -> str:
    """Look up an order."""
    return f"Order {order_id} has shipped."


agent = Agent(
    name="Support",
    instructions="Answer order questions with the tool.",
    tools=[get_order_status],
)


async def main() -> None:
    result = await Runner.run(agent, "Where is order 1234?", hooks=AuditHooks())
    print(result.final_output)


asyncio.run(main())
```

Hooks are awaited inline, so keep them fast: push slow work (database writes, HTTP calls) onto a queue. Raising an exception inside a hook aborts the run, which is a blunt but effective kill switch.

## Agent hooks

`AgentHooks` has the same callbacks, named `on_start`, `on_end`, `on_tool_start` and so on, but fires only for the agent it is attached to. It suits per-agent concerns such as loading a customer record when the billing agent takes over.

```python
import asyncio
from dataclasses import dataclass, field
from typing import Any

from agents import Agent, AgentHooks, Runner


@dataclass
class SupportContext:
    customer_id: str
    notes: list[str] = field(default_factory=list)


class BillingHooks(AgentHooks[SupportContext]):
    async def on_start(self, context, agent: Agent[SupportContext]) -> None:
        # context.context is your own object; mutate it to share state with tools
        context.context.notes.append(f"billing opened for {context.context.customer_id}")

    async def on_end(self, context, agent: Agent[SupportContext], output: Any) -> None:
        context.context.notes.append("billing closed")


billing = Agent[SupportContext](
    name="Billing",
    instructions="Resolve billing questions.",
    hooks=BillingHooks(),
)


async def main() -> None:
    ctx = SupportContext(customer_id="cus_42")
    await Runner.run(billing, "Why was I charged twice?", context=ctx)
    print(ctx.notes)


asyncio.run(main())
```

## Tool guardrails

Tool guardrails wrap a single function tool. An input guardrail sees the arguments the model produced before the tool runs; an output guardrail sees the return value before the model does. Each returns a `ToolGuardrailFunctionOutput`:

- `ToolGuardrailFunctionOutput.allow()` lets the call proceed.
- `ToolGuardrailFunctionOutput.reject_content("…")` skips the tool (input) or replaces its output, and sends your message to the model instead. The run continues.
- `ToolGuardrailFunctionOutput.raise_exception()` stops the run with `ToolInputGuardrailTripwireTriggered` / `ToolOutputGuardrailTripwireTriggered`.

```python
import asyncio
import json
import re

from agents import (
    Agent,
    Runner,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
    function_tool,
    tool_input_guardrail,
    tool_output_guardrail,
)


@tool_input_guardrail
def cap_refund(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    args = json.loads(data.context.tool_arguments or "{}")
    if args.get("amount", 0) > 500:
        return ToolGuardrailFunctionOutput.reject_content(
            "Refunds over 500 need a supervisor. Tell the customer it has been escalated."
        )
    return ToolGuardrailFunctionOutput.allow()


CARD = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b")


@tool_output_guardrail
def no_card_numbers(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    if CARD.search(str(data.output)):
        return ToolGuardrailFunctionOutput.raise_exception(output_info={"reason": "card number"})
    return ToolGuardrailFunctionOutput.allow()


@function_tool(tool_input_guardrails=[cap_refund], tool_output_guardrails=[no_card_numbers])
def issue_refund(order_id: str, amount: float) -> str:
    """Refund part or all of an order."""
    return f"Refunded {amount:.2f} on {order_id}."


agent = Agent(name="Refunds", instructions="Issue refunds when asked.", tools=[issue_refund])


async def main() -> None:
    result = await Runner.run(agent, "Refund 900 on order A-17.")
    print(result.final_output)


asyncio.run(main())
```

Tool guardrails apply to `@function_tool` tools only. Hosted tools (web search, file search, code interpreter) and handoffs bypass them; use agent-level guardrails or approvals for those.

## Blocking input guardrails

Agent input guardrails run **in parallel** with the first model call by default (`run_in_parallel=True`), so a tripwire cancels the run but the model call may already have been made and billed. When a check must finish before any tokens are spent, for example a cheap classifier that rejects most traffic, run it first:

```python
from agents import Agent, GuardrailFunctionOutput, input_guardrail


@input_guardrail(run_in_parallel=False)
async def block_account_takeover(ctx, agent, input_data) -> GuardrailFunctionOutput:
    text = input_data if isinstance(input_data, str) else str(input_data)
    return GuardrailFunctionOutput(
        output_info=None,
        tripwire_triggered="reset the password for" in text.lower(),
    )


agent = Agent(name="Support", input_guardrails=[block_account_takeover])
```

Output guardrails always run after the agent has produced its final output, including in streamed runs, so they cannot stop tokens that have already been streamed to a user. Buffer the stream if that matters.

## Human-in-the-loop approvals

Mark a tool with `needs_approval=True` (or an async callable that decides per call). When the model calls it, the run stops before the tool executes and returns with `result.interruptions` populated. Convert the result to a `RunState`, approve or reject each item, and pass the state back to `Runner.run` to continue from exactly where it paused.

```python
import asyncio

from agents import Agent, RunContextWrapper, Runner, RunState, function_tool


async def large_transfers_only(ctx: RunContextWrapper, params: dict, call_id: str) -> bool:
    return params.get("amount", 0) >= 1000


@function_tool(needs_approval=large_transfers_only)
def transfer_funds(to_account: str, amount: float) -> str:
    """Move money between accounts."""
    return f"Sent {amount:.2f} to {to_account}."


agent = Agent(name="Treasury", instructions="Carry out transfers.", tools=[transfer_funds])


async def main() -> None:
    result = await Runner.run(agent, "Send 2500 to ACC-9.")

    while result.interruptions:
        state = result.to_state()

        # Persist across processes or days: store the string, reload it later.
        saved = state.to_string()
        state = await RunState.from_string(agent, saved)

        for item in state.get_interruptions():
            print(f"Approval needed: {item.tool_name} {item.arguments}")
            if input("Approve? [y/N] ").strip().lower() == "y":
                state.approve(item)
            else:
                state.reject(item, rejection_message="A treasurer declined this transfer.")

        result = await Runner.run(agent, state)

    print(result.final_output)


asyncio.run(main())
```

Points to know:

- `state.approve(item, always_approve=True)` approves that tool for the rest of the run; `always_reject=True` does the reverse.
- The serialised state contains pending tool calls and approval decisions. Store it server-side and only reload snapshots you wrote; do not accept a state string from a client.
- With streaming, read `result.interruptions` after the stream finishes, then resume with `Runner.run_streamed(agent, state)`.
- Tool input guardrails and approvals interact. By default the approval check comes first: a call that needs approval pauses the run before its input guardrails run, and the guardrails run only after a person approves. To reject a bad call (an amount over a limit, say) without bothering anyone, run the input guardrails before the pause with `RunConfig(tool_execution=ToolExecutionConfig(pre_approval_tool_input_guardrails=True))`. The guardrails still run again just before execution after approval.
- Hosted MCP tools use the same mechanism when `require_approval` is set (see [Hosted MCP tools](/openai-agents-sdk-guides/openai_agents_sdk_comprehensive_guide/#hosted-mcp-tools)).

## Filtering model input

`RunConfig(call_model_input_filter=...)` receives the instructions and input items just before every model call and returns what to send. Use it to cap history length, strip items, or add per-call instructions. It does not change what is stored in the session.

```python
import asyncio

from agents import Agent, RunConfig, Runner
from agents.run_config import CallModelData, ModelInputData


def last_twenty_items(data: CallModelData) -> ModelInputData:
    return ModelInputData(
        input=data.model_data.input[-20:],
        instructions=data.model_data.instructions,
    )


agent = Agent(name="Assistant", instructions="Be brief.")


async def main() -> None:
    result = await Runner.run(
        agent,
        "Hello",
        run_config=RunConfig(call_model_input_filter=last_twenty_items),
    )
    print(result.final_output)


asyncio.run(main())
```

## Handoff input filters

When one agent hands off to another, the receiving agent sees the conversation so far. `handoff(..., input_filter=...)` rewrites that history; `agents.extensions.handoff_filters` has ready-made filters. `remove_all_tools` drops earlier tool calls and their output, which keeps a specialist's context short and stops it re-using another agent's tool results:

```python
from agents import Agent, handoff
from agents.extensions import handoff_filters

faq_agent = Agent(name="FAQ", instructions="Answer general questions.")

triage = Agent(
    name="Triage",
    instructions="Hand off general questions to FAQ.",
    handoffs=[handoff(faq_agent, input_filter=handoff_filters.remove_all_tools)],
)
```

Set `RunConfig(handoff_input_filter=...)` to apply a filter to every handoff in a run.

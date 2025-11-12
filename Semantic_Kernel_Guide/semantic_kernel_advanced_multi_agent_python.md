---
layout: default
title: "Semantic Kernel Advanced Multi‑Agent (Python)"
description: "Advanced multi-agent patterns in Semantic Kernel (Python) with AgentGroupChat, structured handoffs, and routing."
---

# Semantic Kernel Advanced Multi‑Agent (Python)

Last verified: 2025-11 • Source of truth: https://github.com/microsoft/semantic-kernel (agents)

## Goals
- Architect robust multi-agent systems using `AgentGroupChat` and termination strategies
- Implement role routing, tool capability routing, and human-in-the-loop handoffs
- Ensure resilience with retries, timeouts, and circuit-breaking per agent

## Setup

```python
import asyncio
from typing import Optional
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import semantic_kernel as sk

async def build_kernel():
    kernel = sk.Kernel()
    kernel.add_chat_service(
        "openai-chat",
        OpenAIChatCompletion(model_id="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])  # choose current model
    )
    return kernel
```

## Roles and Capabilities

```python
class Tooling:
    @staticmethod
    async def web_search(q: str) -> str:
        # wrap your search provider; add timeouts + retries
        return f"search:{q}"

def system_prompt(role: str, capabilities: list[str]) -> str:
    return (
        f"You are the {role}. You can: {', '.join(capabilities)}. "
        "Respond concisely with citations when applicable."
    )

def create_researcher(kernel: sk.Kernel) -> ChatCompletionAgent:
    return ChatCompletionAgent(
        kernel=kernel,
        name="researcher",
        instructions=system_prompt("Researcher", ["search", "draft findings"]),
        tools={"web_search": Tooling.web_search},
    )

def create_writer(kernel: sk.Kernel) -> ChatCompletionAgent:
    return ChatCompletionAgent(
        kernel=kernel,
        name="writer",
        instructions=system_prompt("Writer", ["outline", "finalize copy"]),
    )
```

## AgentGroupChat with Termination

```python
class TokenOrTurnsTermination:
    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns

    def __call__(self, messages) -> bool:
        return len(messages) >= self.max_turns

async def run_chat(user_input: str) -> str:
    kernel = await build_kernel()
    researcher = create_researcher(kernel)
    writer = create_writer(kernel)

    chat = AgentGroupChat(
        researcher,
        writer,
        execution_settings={
            "termination_strategy": TokenOrTurnsTermination(max_turns=6),
        },
    )

    await chat.add_user_message(user_input)
    result = await chat.invoke()
    return result
```

## Routing Patterns

```python
def router(message: str) -> str:
    if "search" in message.lower() or "cite" in message.lower():
        return "researcher"
    return "writer"

async def run_routed_chat(user_input: str) -> str:
    kernel = await build_kernel()
    agents = {"researcher": create_researcher(kernel), "writer": create_writer(kernel)}
    chat = AgentGroupChat(agents["researcher"], agents["writer"])

    await chat.add_user_message(user_input)
    while True:
        current = router(chat.messages[-1].content)
        await chat.step(next_agent=agents[current])
        if chat.should_terminate():
            break
    return chat.messages[-1].content
```

## Human-in-the-Loop Handoffs

```python
async def hitl_handoff(chat: AgentGroupChat, reason: str):
    chat.add_system_message(f"Handoff to human required: {reason}")
    # persist chat state and notify via your ticketing system
```

Use guardrail checks (policy, PII) to trigger `hitl_handoff` when needed.

## Resilience

- Per-agent timeouts and retries around tool calls (use tenacity/backoff)
- Circuit breakers for flaky tools (open on repeated failures)
- Idempotent tool design; keep tool IO deterministic and side-effect aware

## Observability

- Log agent, step, selected route, and tool IO with trace IDs
- Export traces to OpenTelemetry; correlate across agents in the same chat

## Testing

- Golden prompts with snapshot tests for role prompts
- Deterministic tool mocks; property tests for router
- Conversation regression tests validating termination and handoff paths


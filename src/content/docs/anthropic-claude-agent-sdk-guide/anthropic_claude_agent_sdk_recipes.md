---
title: "Claude Agent SDK (Python) — Recipes"
description: "Copy-and-adapt Python recipes for the Claude Agent SDK: summarisation, classification, code review, test generation, DevOps, subagents, custom tools and testing."
framework: "claude-agent-sdk"
language: "python"
---

Verified against `claude-agent-sdk` %%version=claude-agent-sdk%% (checked %%versions-checked%%). Each recipe is self-contained; install with `pip install claude-agent-sdk==%%version=claude-agent-sdk%%` and set `ANTHROPIC_API_KEY`.

Most recipes reuse this small helper, which runs a query and returns the final `ResultMessage`:

```python
# agent_utils.py
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

async def run_agent(prompt: str, options: ClaudeAgentOptions | None = None) -> ResultMessage:
    result: ResultMessage | None = None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            result = message
    if result is None:
        raise RuntimeError("agent finished without a result")
    if result.is_error:
        raise RuntimeError(f"agent failed: {result.subtype} {result.errors}")
    return result
```

---

## Contents

1. [Hello world](#recipe-1-hello-world)
2. [Summarisation](#recipe-2-summarisation)
3. [Classification with structured output](#recipe-3-classification-with-structured-output)
4. [Research assistant with web tools](#recipe-4-research-assistant-with-web-tools)
5. [CSV data analyst](#recipe-5-csv-data-analyst)
6. [Read-only code review](#recipe-6-read-only-code-review)
7. [Test generator](#recipe-7-test-generator)
8. [Guarded DevOps agent](#recipe-8-guarded-devops-agent)
9. [Specialist subagents](#recipe-9-specialist-subagents)
10. [Parallel reviews from Python](#recipe-10-parallel-reviews-from-python)
11. [Custom tool: internal HTTP API](#recipe-11-custom-tool-internal-http-api)
12. [Multi-turn chat loop](#recipe-12-multi-turn-chat-loop)
13. [Audit log with hooks](#recipe-13-audit-log-with-hooks)
14. [Testing agents](#recipe-14-testing-agents)

---

## Recipe 1: Hello world

```python
import asyncio
from claude_agent_sdk import query, AssistantMessage, TextBlock

async def main() -> None:
    async for message in query(prompt="Say hello in three languages."):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text)

asyncio.run(main())
```

---

## Recipe 2: Summarisation

No tools are needed, so remove them all; this also stops the agent wandering off to read files.

```python
import asyncio
from claude_agent_sdk import ClaudeAgentOptions
from agent_utils import run_agent

async def summarise(text: str, words: int = 120) -> str:
    options = ClaudeAgentOptions(
        model="claude-sonnet-5-5",
        system_prompt=f"Summarise the user's text in at most {words} words. Plain prose, no headings.",
        tools=[],
        max_turns=1,
    )
    result = await run_agent(text, options)
    return result.result or ""

print(asyncio.run(summarise(open("report.txt").read())))
```

---

## Recipe 3: Classification with structured output

`output_format` makes the agent return JSON matching a schema, delivered on `ResultMessage.structured_output`.

```python
import asyncio
from claude_agent_sdk import ClaudeAgentOptions
from agent_utils import run_agent

LABELS = ["billing", "bug", "feature_request", "account", "other"]

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": LABELS},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["label", "confidence", "reason"],
    "additionalProperties": False,
}

async def classify(ticket: str) -> dict:
    options = ClaudeAgentOptions(
        model="claude-haiku-5-5",
        system_prompt="Classify support tickets.",
        tools=[],
        output_format={"type": "json_schema", "schema": SCHEMA},
    )
    result = await run_agent(ticket, options)
    return result.structured_output

print(asyncio.run(classify("I was charged twice for October.")))
```

---

## Recipe 4: Research assistant with web tools

```python
import asyncio
from claude_agent_sdk import ClaudeAgentOptions
from agent_utils import run_agent

async def research(topic: str) -> str:
    options = ClaudeAgentOptions(
        model="claude-opus-5-5",
        system_prompt=(
            "You are a careful researcher. Use several independent sources, "
            "cite each claim with its URL, and say when sources disagree."
        ),
        allowed_tools=["WebSearch", "WebFetch"],
        disallowed_tools=["Bash", "Write", "Edit"],
        max_turns=25,
        max_budget_usd=1.50,
    )
    result = await run_agent(f"Write a one-page briefing on: {topic}", options)
    print(f"cost ${result.total_cost_usd:.3f}, {result.num_turns} turns")
    return result.result or ""

print(asyncio.run(research("the state of post-quantum TLS deployment")))
```

---

## Recipe 5: CSV data analyst

The agent writes and runs its own analysis code. Confine it to a working directory and pre-approve only what it needs.

```python
import asyncio
from pathlib import Path
from claude_agent_sdk import ClaudeAgentOptions
from agent_utils import run_agent

async def analyse(csv_path: Path) -> str:
    options = ClaudeAgentOptions(
        cwd=csv_path.parent,
        system_prompt=(
            "You are a data analyst. Use Python with pandas via Bash. "
            "Report summary statistics, notable trends and outliers. "
            "Save any charts as PNG files in the working directory."
        ),
        allowed_tools=["Read", "Write", "Bash(python:*)", "Bash(python3:*)"],
        permission_mode="acceptEdits",
        max_turns=30,
    )
    result = await run_agent(f"Analyse {csv_path.name}.", options)
    return result.result or ""

print(asyncio.run(analyse(Path("data/sales.csv"))))
```

---

## Recipe 6: Read-only code review

```python
import asyncio
from claude_agent_sdk import ClaudeAgentOptions
from agent_utils import run_agent

REVIEW_PROMPT = """Review the code under {path}.

Report, in Markdown:
## Issues
Each with file:line, severity (high/medium/low) and a suggested fix.
## Improvements
## Overall assessment
"""

async def review(repo: str, path: str) -> str:
    options = ClaudeAgentOptions(
        cwd=repo,
        system_prompt={"type": "preset", "preset": "claude_code",
                       "append": "You are reviewing, not editing. Never modify files."},
        allowed_tools=["Read", "Glob", "Grep"],
        disallowed_tools=["Write", "Edit", "Bash"],
        effort="high",
    )
    result = await run_agent(REVIEW_PROMPT.format(path=path), options)
    return result.result or ""

print(asyncio.run(review("/srv/repos/api", "src/auth/")))
```

---

## Recipe 7: Test generator

```python
import asyncio
from claude_agent_sdk import ClaudeAgentOptions
from agent_utils import run_agent

async def generate_tests(repo: str, module: str) -> str:
    options = ClaudeAgentOptions(
        cwd=repo,
        system_prompt=(
            "Write focused, deterministic pytest tests. Cover edge cases and "
            "error paths. Run the tests and fix them until they pass."
        ),
        allowed_tools=["Read", "Glob", "Grep", "Write", "Edit", "Bash(pytest:*)"],
        permission_mode="acceptEdits",
        max_turns=40,
    )
    result = await run_agent(f"Add tests for {module} under tests/.", options)
    return result.result or ""

print(asyncio.run(generate_tests("/srv/repos/api", "src/api/pagination.py")))
```

---

## Recipe 8: Guarded DevOps agent

A permission callback lets read-only commands through, refuses destructive ones, and denies everything else unless a human approves.

```python
import asyncio
from typing import Any
from claude_agent_sdk import (
    ClaudeAgentOptions, PermissionResultAllow, PermissionResultDeny, ToolPermissionContext,
)
from agent_utils import run_agent

READ_ONLY = ("kubectl get", "kubectl describe", "kubectl logs", "df ", "uptime", "systemctl status")
FORBIDDEN = ("kubectl delete", "rm -rf", "shutdown", "reboot", "mkfs")

async def guard(tool_name: str, tool_input: dict[str, Any], ctx: ToolPermissionContext):
    if tool_name != "Bash":
        return PermissionResultAllow()
    command = tool_input.get("command", "")
    if any(f in command for f in FORBIDDEN):
        return PermissionResultDeny(message="Destructive command refused", interrupt=True)
    if command.startswith(READ_ONLY):
        return PermissionResultAllow()
    answer = await asyncio.to_thread(input, f"Allow `{command}`? [y/N] ")
    if answer.strip().lower() == "y":
        return PermissionResultAllow()
    return PermissionResultDeny(message="Operator declined")

async def main() -> None:
    options = ClaudeAgentOptions(
        system_prompt="You are an SRE. Diagnose before acting; explain each command.",
        allowed_tools=["Read", "Grep"],   # Bash is deliberately not pre-approved
        can_use_tool=guard,
        max_turns=30,
    )
    result = await run_agent("The checkout pods keep restarting. Find out why.", options)
    print(result.result)

asyncio.run(main())
```

---

## Recipe 9: Specialist subagents

Let the main agent decide when to delegate. Each subagent has its own context window, tools and (cheaper) model.

```python
import asyncio
from claude_agent_sdk import ClaudeAgentOptions, AgentDefinition
from agent_utils import run_agent

options = ClaudeAgentOptions(
    model="claude-opus-5-5",
    cwd="/srv/repos/api",
    allowed_tools=["Read", "Glob", "Grep", "Agent"],
    agents={
        "security": AgentDefinition(
            description="Security review of code. Use for vulnerabilities and auth issues.",
            prompt="You are an application security reviewer. Report issues with severity and fix.",
            tools=["Read", "Glob", "Grep"],
            model="sonnet",
        ),
        "performance": AgentDefinition(
            description="Performance review of code. Use for latency, memory and query issues.",
            prompt="You are a performance engineer. Quantify impact where you can.",
            tools=["Read", "Glob", "Grep"],
            model="sonnet",
        ),
        "style": AgentDefinition(
            description="Readability and maintainability review.",
            prompt="You review naming, structure and documentation.",
            tools=["Read", "Glob"],
            model="haiku",
        ),
    },
)

async def main() -> None:
    result = await run_agent(
        "Review src/orders/ using the security, performance and style subagents, "
        "then give me one prioritised list.",
        options,
    )
    print(result.result)
    for model, usage in (result.model_usage or {}).items():
        print(model, usage.get("costUSD"))

asyncio.run(main())
```

---

## Recipe 10: Parallel reviews from Python

When the workflow is fixed, orchestrate in code. `asyncio.gather` runs independent agents concurrently; a semaphore keeps the number of CLI processes sensible.

```python
import asyncio
from claude_agent_sdk import ClaudeAgentOptions
from agent_utils import run_agent

ROLES = {
    "security": "You are a security reviewer.",
    "performance": "You are a performance engineer.",
    "quality": "You review code quality and maintainability.",
}
limit = asyncio.Semaphore(3)

async def one_review(role: str, system: str, path: str) -> tuple[str, str]:
    async with limit:
        options = ClaudeAgentOptions(
            model="claude-sonnet-5-5",
            system_prompt=system,
            allowed_tools=["Read", "Glob", "Grep"],
        )
        result = await run_agent(f"Review {path}. Be specific.", options)
        return role, result.result or ""

async def main(path: str) -> None:
    reviews = dict(await asyncio.gather(*(one_review(r, s, path) for r, s in ROLES.items())))
    combined = "\n\n".join(f"### {r}\n{text}" for r, text in reviews.items())
    summary = await run_agent(
        f"Merge these reviews into one prioritised action list:\n\n{combined}",
        ClaudeAgentOptions(model="claude-opus-5-5", tools=[]),
    )
    print(summary.result)

asyncio.run(main("src/payments/"))
```

---

## Recipe 11: Custom tool: internal HTTP API

Wrap an internal service as an in-process MCP tool. The handler is ordinary async Python, so use whatever HTTP client you already have (here `httpx`).

```python
import asyncio
from typing import Annotated, Any
import httpx
from claude_agent_sdk import tool, create_sdk_mcp_server, ClaudeAgentOptions
from agent_utils import run_agent

@tool(
    "lookup_order",
    "Fetch an order from the orders service by ID",
    {"order_id": Annotated[str, "Order ID, e.g. ORD-12345"]},
)
async def lookup_order(args: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url="https://orders.internal", timeout=10) as http:
        response = await http.get(f"/orders/{args['order_id']}")
    if response.status_code == 404:
        return {"content": [{"type": "text", "text": "No such order"}], "is_error": True}
    response.raise_for_status()
    return {"content": [{"type": "text", "text": response.text}]}

orders = create_sdk_mcp_server(name="orders", version="1.0.0", tools=[lookup_order])

async def main() -> None:
    options = ClaudeAgentOptions(
        mcp_servers={"orders": orders},
        allowed_tools=["mcp__orders__lookup_order"],
        tools=[],   # no built-in tools: this agent only talks to the orders API
    )
    result = await run_agent("Where is order ORD-12345 and when will it arrive?", options)
    print(result.result)

asyncio.run(main())
```

---

## Recipe 12: Multi-turn chat loop

```python
import asyncio
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock

async def chat() -> None:
    options = ClaudeAgentOptions(model="claude-sonnet-5-5", system_prompt="You are a helpful assistant.")
    async with ClaudeSDKClient(options=options) as client:
        while True:
            user = await asyncio.to_thread(input, "you> ")
            if user.strip() in {"exit", "quit"}:
                break
            await client.query(user)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print("claude>", block.text)

asyncio.run(chat())
```

Use `await client.interrupt()` from another task to stop a long answer.

---

## Recipe 13: Audit log with hooks

```python
import json
import time
from typing import Any
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, HookContext

async def log_tool(input_data: dict[str, Any], tool_use_id: str | None,
                   context: HookContext) -> dict[str, Any]:
    record = {
        "ts": time.time(),
        "event": input_data["hook_event_name"],
        "session": input_data["session_id"],
        "tool": input_data.get("tool_name"),
        "input": input_data.get("tool_input"),
    }
    with open("agent_audit.jsonl", "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return {}

options = ClaudeAgentOptions(
    hooks={
        "PreToolUse": [HookMatcher(hooks=[log_tool])],
        "PostToolUseFailure": [HookMatcher(hooks=[log_tool])],
    },
)
```

---

## Recipe 14: Testing agents

Unit-test your own logic (permission callbacks, tool handlers, result parsing) directly; they are plain async functions. Keep end-to-end tests that call the model few, cheap and tolerant.

```python
# test_agent.py — run with: pytest -q (requires pytest-asyncio)
import os
import pytest
from claude_agent_sdk import ClaudeAgentOptions, PermissionResultDeny, ToolPermissionContext

from recipes import guard, lookup_order   # the functions from recipes 8 and 11
from agent_utils import run_agent

@pytest.mark.asyncio
async def test_guard_blocks_destructive_commands():
    ctx = ToolPermissionContext(suggestions=[])
    decision = await guard("Bash", {"command": "rm -rf /"}, ctx)
    assert isinstance(decision, PermissionResultDeny)

@pytest.mark.asyncio
async def test_tool_handler_is_callable_directly():
    # SdkMcpTool keeps the original coroutine on .handler
    assert callable(lookup_order.handler)

@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs API access")
async def test_end_to_end_arithmetic():
    options = ClaudeAgentOptions(model="claude-haiku-5-5", tools=[], max_turns=1)
    result = await run_agent("What is 17 * 3? Reply with the number only.", options)
    assert "51" in (result.result or "")
    assert result.total_cost_usd is not None
```

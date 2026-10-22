---
title: "Claude Agent SDK (Python) — Recipes"
description: "Copy-and-adapt Python recipes for the Claude Agent SDK: summarisation, classification, code review, test generation, DevOps, subagents, custom tools and testing."
framework: "claude-agent-sdk"
language: "python"
---

Verified against `claude-agent-sdk` %%version=claude-agent-sdk%% (checked %%versions-checked%%). Each recipe is self-contained; install with `pip install claude-agent-sdk==%%version=claude-agent-sdk%%` and set `ANTHROPIC_API_KEY`.

Most recipes reuse this small helper, which runs a query to the end and returns the final `ResultMessage`. It deliberately keeps reading after the `ResultMessage`: when a run fails, the SDK yields the error result first and then raises `ResultError` (with `subtype`, `terminal_reason` and `api_error_status`), and that exception is what the caller sees.

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
    return result
```

For retries, use `run_resilient` from the [production guide](/anthropic-claude-agent-sdk-guide/anthropic_claude_agent_sdk_production_guide/#retry-only-what-is-worth-retrying).

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
15. [Streaming to a browser (FastAPI and SSE)](#recipe-15-streaming-to-a-browser-fastapi-and-sse)

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
        model="%%model=anthropic%%",
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
        model="%%model=anthropic-small%%",
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
        model="%%model=anthropic-flagship%%",
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
    print(f"cost ${result.total_cost_usd or 0:.3f}, {result.num_turns} turns")
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

The SDK emits a `CanUseToolShadowedWarning` for `Read` and `Grep` here: they are pre-approved, so `guard` never sees them. That is intended; the warning exists to catch the case where it is not.

---

## Recipe 9: Specialist subagents

Let the main agent decide when to delegate. Each subagent has its own context window, tools and (cheaper) model.

```python
import asyncio
from claude_agent_sdk import ClaudeAgentOptions, AgentDefinition
from agent_utils import run_agent

options = ClaudeAgentOptions(
    model="%%model=anthropic-flagship%%",
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
            model="%%model=anthropic%%",
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
        ClaudeAgentOptions(model="%%model=anthropic-flagship%%", tools=[]),
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
    options = ClaudeAgentOptions(model="%%model=anthropic%%", system_prompt="You are a helpful assistant.")
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

Three layers, cheapest first:

1. **Your own functions.** Permission callbacks, hooks and tool handlers are plain async functions; call them directly.
2. **Your message handling, offline.** Point `cli_path` at a stub executable that speaks the CLI's stream-json protocol and replays a canned result. This tests success and failure paths (including the `ResultError` that follows an error result) without an API key.
3. **A few end-to-end runs** against the real model, cheap and tolerant of wording.

The stub answers the SDK's control requests and ends every run with the result in `$STUB_RESULT`:

```python
#!/usr/bin/env python3
# stub_claude.py — make it executable: chmod +x stub_claude.py
import json, os, sys

def send(obj):
    print(json.dumps(obj), flush=True)

result = {"type": "result", "subtype": "success", "is_error": False, "result": "stub answer",
          "session_id": "stub", "num_turns": 1, "duration_ms": 1, "duration_api_ms": 1,
          "total_cost_usd": 0.0, **json.loads(os.environ.get("STUB_RESULT", "{}"))}

for line in sys.stdin:
    message = json.loads(line)
    if message["type"] == "control_request":           # initialize and other control calls
        send({"type": "control_response", "response": {
            "subtype": "success", "request_id": message["request_id"], "response": {}}})
    elif message["type"] == "user":
        send(result)
        sys.exit(1 if result["is_error"] else 0)        # the real CLI exits non-zero after an error result
```

```python
# test_agent.py — run with: pytest -q (requires pytest-asyncio)
import json
import os
from pathlib import Path
import httpx
import pytest
from claude_agent_sdk import (
    ClaudeAgentOptions, PermissionResultDeny, ResultError, ToolPermissionContext,
)

from recipes import guard, lookup_order   # the functions from recipes 8 and 11
from agent_utils import run_agent

STUB = str(Path(__file__).with_name("stub_claude.py"))

@pytest.mark.asyncio
async def test_guard_blocks_destructive_commands():
    decision = await guard("Bash", {"command": "rm -rf /"}, ToolPermissionContext(suggestions=[]))
    assert isinstance(decision, PermissionResultDeny)

@pytest.mark.asyncio
async def test_lookup_order_reports_missing_orders(monkeypatch):
    # Serve the orders API from a mock transport; SdkMcpTool keeps the coroutine on .handler.
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(404))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=transport, **kw))
    result = await lookup_order.handler({"order_id": "ORD-1"})
    assert result["is_error"] is True

@pytest.mark.asyncio
async def test_successful_run_returns_the_result():
    result = await run_agent("hi", ClaudeAgentOptions(cli_path=STUB))
    assert result.result == "stub answer"

@pytest.mark.asyncio
async def test_turn_limit_raises_result_error(monkeypatch):
    monkeypatch.setenv("STUB_RESULT", json.dumps(
        {"subtype": "error_max_turns", "is_error": True, "terminal_reason": "max_turns"}))
    with pytest.raises(ResultError) as failure:
        await run_agent("hi", ClaudeAgentOptions(cli_path=STUB))
    assert failure.value.subtype == "error_max_turns"

@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs API access")
async def test_end_to_end_arithmetic():
    options = ClaudeAgentOptions(model="%%model=anthropic-small%%", tools=[], max_turns=1)
    result = await run_agent("What is 17 * 3? Reply with the number only.", options)
    assert "51" in (result.result or "")
    assert result.total_cost_usd is not None
```

The stub covers the protocol only as far as these tests need it. To exercise tool calls or hooks offline, extend it to emit `assistant` messages and `control_request`s of its own.

---

## Recipe 15: Streaming to a browser (FastAPI and SSE)

Run an agent per request and forward text deltas, tool activity and the final result as Server-Sent Events. Starlette cancels the response when the browser disconnects, and cancelling the iteration stops the CLI process.

```python
# sse_server.py — run with: uvicorn sse_server:app
import json
from typing import Any, AsyncIterator
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from claude_agent_sdk import (
    query, ClaudeAgentOptions, AssistantMessage, ResultMessage, StreamEvent, ToolUseBlock,
    ClaudeSDKError,
)

app = FastAPI()

def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

@app.get("/stream")
async def stream(q: str, session_id: str | None = None) -> StreamingResponse:
    options = ClaudeAgentOptions(
        model="%%model=anthropic%%",
        cwd="/srv/knowledge-base",
        setting_sources=[],
        allowed_tools=["Read", "Glob", "Grep"],
        disallowed_tools=["Bash", "Write", "Edit"],
        include_partial_messages=True,
        max_turns=15,
        max_budget_usd=0.5,
        verbatim_prompts=True,      # user text cannot trigger @file reads or slash commands
        resume=session_id,
    )

    async def events() -> AsyncIterator[str]:
        try:
            async for message in query(prompt=q[:4000], options=options):
                if isinstance(message, StreamEvent):
                    event = message.event
                    if event.get("type") == "content_block_delta" and event["delta"].get("type") == "text_delta":
                        yield sse("delta", event["delta"]["text"])
                elif isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, ToolUseBlock):
                            yield sse("tool", {"name": block.name})
                elif isinstance(message, ResultMessage):
                    yield sse("done", {"ok": not message.is_error, "session_id": message.session_id,
                                       "cost_usd": message.total_cost_usd})
        except ClaudeSDKError as e:     # includes the ResultError that follows a failed run
            yield sse("error", {"message": str(e)})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```

The browser side is an `EventSource` listening for `delta`, `tool`, `done` and `error`, as in the [TypeScript version](/anthropic-claude-agent-sdk-typescript-guide/claude_streaming_server_express/). Return `session_id` to the client and pass it back for follow-up questions. Authenticate the endpoint and rate-limit it per user: each request starts a process and costs money. Proxies must not buffer `text/event-stream` responses.

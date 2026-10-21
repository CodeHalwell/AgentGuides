---
title: "Claude Agent SDK (Python) — Comprehensive Guide"
description: "Reference for building agents with the Python claude-agent-sdk: query(), ClaudeSDKClient, tools, MCP, subagents, hooks, permissions, sessions and thinking."
framework: anthropic-claude-agent-sdk
---

**Verified against `claude-agent-sdk` %%version=claude-agent-sdk%%** (checked %%versions-checked%%). Every import, class, option and method signature on this page was checked against an installed copy of that release.

The TypeScript package has its own guide: [Claude Agent SDK (TypeScript)](/anthropic-claude-agent-sdk-typescript-guide/).

---

## Table of contents

1. [Introduction](#introduction)
2. [Installation & authentication](#installation--authentication)
3. [Core architecture & concepts](#core-architecture--concepts)
4. [Simple agents](#simple-agents)
5. [Tools & integration](#tools--integration)
6. [Permissions](#permissions)
7. [Hooks](#hooks)
8. [Multi-agent systems](#multi-agent-systems)
9. [Model Context Protocol (MCP)](#model-context-protocol-mcp)
10. [Structured output](#structured-output)
11. [Session management](#session-management)
12. [Thinking and effort](#thinking-and-effort)
13. [File checkpointing and rewind](#file-checkpointing-and-rewind)
14. [Context usage, budgets and limits](#context-usage-budgets-and-limits)
15. [Error handling](#error-handling)
16. [API reference](#api-reference)

---

## Introduction

### What the SDK is

The Claude Agent SDK is Claude Code packaged as a library. It was called the Claude Code SDK until version 0.1.0, when it was renamed to reflect that the same harness is useful well beyond coding.

Under the hood the Python package starts the Claude Code CLI as a subprocess (a copy is bundled in the platform wheels) and talks to it over a JSON stream. That gives you, without writing a tool loop yourself:

- the agent loop, context management and automatic compaction;
- built-in tools — `Read`, `Write`, `Edit`, `Bash`, `Glob`, `Grep`, `WebSearch`, `WebFetch`, the `Agent` tool for subagents, and others;
- in-process custom tools via MCP, plus external MCP servers;
- permission modes, a `can_use_tool` callback and lifecycle hooks;
- persistent, resumable and forkable sessions;
- subagents, skills, plugins and sandboxing.

### When to use it

Use the Agent SDK when you want a batteries-included agent that works on files, runs commands and calls tools on your own infrastructure. If you only need a single model call, or a tool loop over tools you define yourself, the plain `anthropic` client (with its tool runner) is lighter. The SDK does not host anything for you: you deploy it like any other Python service.

### Two entry points

| | `query()` | `ClaudeSDKClient` |
|---|---|---|
| Shape | async generator, one task | async context manager, a conversation |
| Follow-up turns | resume via `options.resume` | call `client.query()` again |
| Interrupts, model switch, MCP status | no | yes |
| `rewind_files()`, `get_context_usage()` | no | yes |
| Typical use | scripts, batch jobs, CI | chat UIs, long-lived workers |

There is no `Agent` class; everything goes through these two.

---

## Installation & authentication

### Requirements

- Python 3.10 or later.
- The platform wheels on PyPI bundle the Claude Code CLI, so no separate Node.js install is needed. If you want to use a CLI installed elsewhere, pass `cli_path=` in the options.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install claude-agent-sdk==%%version=claude-agent-sdk%%
```

A minimal `requirements.txt`:

```txt
claude-agent-sdk==%%version=claude-agent-sdk%%
python-dotenv>=1.0
```

Check what you have installed:

```python
import claude_agent_sdk
print(claude_agent_sdk.__version__)  # e.g. %%version=claude-agent-sdk%%
```

### API key

The CLI reads credentials from the environment. Create a key in the [Claude Console](https://platform.claude.com/) and export it:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

You can also pass environment variables to the CLI process explicitly through `env=`:

```python
import os
from claude_agent_sdk import ClaudeAgentOptions

options = ClaudeAgentOptions(env={"ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"]})
```

### Amazon Bedrock, Google Vertex AI and Microsoft Foundry

There are no `api_provider`-style options. Provider selection is done with the same environment variables Claude Code uses:

```bash
# Amazon Bedrock (credentials from the usual AWS chain)
export CLAUDE_CODE_USE_BEDROCK=1
export AWS_REGION=us-east-1

# Google Vertex AI (credentials from Application Default Credentials)
export CLAUDE_CODE_USE_VERTEX=1
export CLOUD_ML_REGION=global
export ANTHROPIC_VERTEX_PROJECT_ID=my-gcp-project
```

Set them in the process environment or in `ClaudeAgentOptions(env={...})`. Model IDs differ per provider (Bedrock IDs carry an `anthropic.` prefix); check the provider console for the exact string.

---

## Core architecture & concepts

```
your code ──► query() / ClaudeSDKClient ──► Claude Code CLI subprocess ──► Claude API
                    ▲                              │
                    │      JSON message stream     │  built-in tools, MCP servers,
                    └──────────────────────────────┘  hooks, permission checks
```

**Options.** All configuration lives in one dataclass, `ClaudeAgentOptions`. The fields you will use most are `model`, `system_prompt`, `allowed_tools` / `disallowed_tools`, `permission_mode`, `cwd`, `max_turns`, `max_budget_usd`, `mcp_servers`, `agents`, `hooks`, `can_use_tool`, `resume` and `setting_sources`. The full list is in the [API reference](#api-reference).

**Messages.** Both entry points yield typed dataclasses, not dictionaries:

| Class | When you see it | Useful fields |
|---|---|---|
| `SystemMessage` | `subtype="init"` at the start; also other system events | `subtype`, `data` (a dict; `data["session_id"]` on init) |
| `AssistantMessage` | each model turn | `content` (list of blocks), `model`, `usage`, `session_id` |
| `UserMessage` | tool results fed back to the model | `content`, `uuid`, `tool_use_result` |
| `ResultMessage` | always last | `result`, `is_error`, `subtype`, `total_cost_usd`, `usage`, `num_turns`, `session_id`, `structured_output`, `terminal_reason` |
| `StreamEvent` | only with `include_partial_messages=True` | `event` (raw API stream event) |
| `RateLimitEvent` | when a rate-limit state changes | `rate_limit_info` |

Task-related messages (`TaskStartedMessage`, `TaskProgressMessage`, `TaskNotificationMessage`) are subclasses of `SystemMessage`, emitted for background tasks and subagents.

**Content blocks** inside `AssistantMessage.content` and `UserMessage.content`: `TextBlock(text)`, `ThinkingBlock(thinking, signature)`, `ToolUseBlock(id, name, input)`, `ToolResultBlock(tool_use_id, content, is_error)`, plus `ServerToolUseBlock` / `ServerToolResultBlock`.

**Settings isolation.** Since 0.1.0 the SDK does not inject the Claude Code system prompt — you get a minimal prompt unless you ask for the preset (see [System prompts](#system-prompts)). `setting_sources` controls whether `settings.json`, `CLAUDE.md`, project slash commands and file-based subagents are read: `None` (the default) leaves the CLI's defaults in place, `[]` loads nothing, and a list such as `["project"]` loads only what you name. For reproducible services, pass an explicit list.

### Models

The model is any Claude model ID, or one of the aliases the CLI understands (`"opus"`, `"sonnet"`, `"haiku"`). When `model` is omitted the CLI's default is used. The current family:

| Model | ID | Notes |
|---|---|---|
| Claude Opus 5.5 | `claude-opus-5-5` | Default choice for agentic and coding work |
| Claude Sonnet 5.5 | `claude-sonnet-5-5` | Faster and cheaper; good for most subagents |
| Claude Haiku 5.5 | `claude-haiku-5-5` | Cheapest; classification, routing, simple subagents |
| Claude Fable 5.1 | `claude-fable-5-1` | Most capable and most expensive; long-horizon reasoning |

All four have a 1M-token context window. `fallback_model=` names a model to switch to if the primary is overloaded.

---

## Simple agents

### First agent

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage, TextBlock

async def main() -> None:
    options = ClaudeAgentOptions(model="claude-opus-5-5")
    async for message in query(prompt="What is the capital of France?", options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text)
        elif isinstance(message, ResultMessage):
            print(f"[{message.num_turns} turns, ${message.total_cost_usd or 0:.4f}]")

asyncio.run(main())
```

Note that `prompt` and `options` are keyword-only: `query("...")` raises a `TypeError`.

If you only want the final answer, `ResultMessage.result` already holds it:

```python
from claude_agent_sdk import query, ResultMessage

async def ask(prompt: str) -> str:
    async for message in query(prompt=prompt):
        if isinstance(message, ResultMessage):
            if message.is_error:
                raise RuntimeError(f"{message.subtype}: {message.errors}")
            return message.result or ""
    return ""
```

### System prompts

`system_prompt` takes a plain string, or a preset dictionary that keeps Claude Code's own prompt and appends to it:

```python
from claude_agent_sdk import ClaudeAgentOptions

# Your own prompt only
reviewer = ClaudeAgentOptions(
    system_prompt=(
        "You are a performance engineer. Identify bottlenecks, propose concrete "
        "changes with code, and estimate the gain. Prefer readability over "
        "micro-optimisation."
    ),
)

# Claude Code's prompt plus your additions
coder = ClaudeAgentOptions(
    system_prompt={
        "type": "preset",
        "preset": "claude_code",
        "append": "Always run the test suite before declaring a task finished.",
    },
)
```

The preset also accepts `"exclude_dynamic_sections": True`, which strips per-machine details (working directory, git status) so the prompt stays identical across runs and caches well.

### Streaming partial output

By default you receive complete messages. Set `include_partial_messages=True` to also receive `StreamEvent`s carrying raw API deltas, which is what a chat UI wants:

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, StreamEvent, ResultMessage

async def stream(prompt: str) -> None:
    options = ClaudeAgentOptions(include_partial_messages=True)
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, StreamEvent):
            event = message.event
            if event.get("type") == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    print(delta["text"], end="", flush=True)
        elif isinstance(message, ResultMessage):
            print()

asyncio.run(stream("Explain the CAP theorem in three paragraphs."))
```

### Interactive sessions with `ClaudeSDKClient`

```python
import asyncio
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock

async def main() -> None:
    options = ClaudeAgentOptions(
        model="claude-sonnet-5-5",
        system_prompt="You are a concise assistant.",
    )
    async with ClaudeSDKClient(options=options) as client:
        for question in ["Name three sorting algorithms.", "Which is stable?"]:
            await client.query(question)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(block.text)

asyncio.run(main())
```

`receive_response()` stops after the next `ResultMessage`; `receive_messages()` keeps going until you break. Other client methods: `interrupt()`, `set_model()`, `set_permission_mode()`, `get_mcp_status()`, `toggle_mcp_server()`, `reconnect_mcp_server()`, `get_context_usage()`, `rewind_files()`, `stop_task()` and `get_server_info()`.

### Token usage and cost

`ResultMessage` carries the authoritative totals for the run, so there is no need to hard-code prices:

```python
from claude_agent_sdk import query, ResultMessage

async def run_and_report(prompt: str) -> None:
    async for message in query(prompt=prompt):
        if isinstance(message, ResultMessage):
            usage = message.usage or {}
            print("input tokens :", usage.get("input_tokens"))
            print("output tokens:", usage.get("output_tokens"))
            print("cache reads  :", usage.get("cache_read_input_tokens"))
            print("cost (USD)   :", message.total_cost_usd)
            for model, mu in (message.model_usage or {}).items():
                print(f"  {model}: {mu}")
```

`model_usage` breaks the totals down per model, which matters once subagents run on a different model from the main loop.

---

## Tools & integration

### Built-in tools

| Tool | Purpose |
|---|---|
| `Read`, `Write`, `Edit` | Read, create and modify files |
| `Glob`, `Grep` | Find files and search contents |
| `Bash` | Run shell commands (subject to permissions and sandboxing) |
| `WebSearch`, `WebFetch` | Search the web and fetch pages |
| `Agent` | Delegate to a subagent |
| `TodoWrite`, `NotebookEdit`, `Skill` | Task lists, Jupyter notebooks, skills (enable skills with the `skills=` option rather than listing `"Skill"` in `allowed_tools`, which is deprecated) |

Three options shape the tool set:

- `tools=` sets the base set the model can see (a list of names, or a preset).
- `allowed_tools=` lists tools that are **pre-approved** — they run without a permission check.
- `disallowed_tools=` removes tools entirely.

```python
from claude_agent_sdk import ClaudeAgentOptions

read_only = ClaudeAgentOptions(
    cwd="/srv/repos/payments",
    allowed_tools=["Read", "Glob", "Grep"],
    disallowed_tools=["Bash", "Write", "Edit"],
)
```

Tool permission rules use the same syntax as Claude Code settings, so you can pre-approve narrowly, for example `"Bash(npm test:*)"` or `"Read(./src/**)"`.

### Custom tools with an in-process MCP server

Define a tool with the `@tool` decorator and serve it with `create_sdk_mcp_server()`. The server runs inside your Python process; there is no subprocess or network hop.

```python
import asyncio
from typing import Any
from claude_agent_sdk import (
    tool, create_sdk_mcp_server, query, ClaudeAgentOptions, ResultMessage,
)

@tool("get_exchange_rate", "Return the exchange rate between two ISO currency codes",
      {"base": str, "quote": str})
async def get_exchange_rate(args: dict[str, Any]) -> dict[str, Any]:
    rates = {("GBP", "EUR"): 1.17, ("EUR", "GBP"): 0.85}
    rate = rates.get((args["base"].upper(), args["quote"].upper()))
    if rate is None:
        return {"content": [{"type": "text", "text": "Unknown currency pair"}], "is_error": True}
    return {"content": [{"type": "text", "text": f"1 {args['base']} = {rate} {args['quote']}"}]}

fx_server = create_sdk_mcp_server(name="fx", version="1.0.0", tools=[get_exchange_rate])

async def main() -> None:
    options = ClaudeAgentOptions(
        mcp_servers={"fx": fx_server},
        allowed_tools=["mcp__fx__get_exchange_rate"],
    )
    async for message in query(prompt="How many euros is 250 pounds?", options=options):
        if isinstance(message, ResultMessage):
            print(message.result)

asyncio.run(main())
```

Points worth knowing:

- Tool names exposed to the model are `mcp__<server key>__<tool name>`; the server key is the dictionary key in `mcp_servers`, not the `name=` argument.
- The handler receives a single `args` dict and returns an MCP result: `{"content": [...]}`, optionally with `"is_error": True`.
- `input_schema` can be a `{name: type}` mapping, a `TypedDict`, or a full JSON Schema dict.
- Add per-parameter descriptions with `typing.Annotated`:

```python
from typing import Annotated, Any
from claude_agent_sdk import tool

@tool(
    "search_codebase",
    "Search the project for a pattern",
    {
        "pattern": Annotated[str, "Regex or literal string to search for"],
        "max_results": Annotated[int, "Maximum matches to return (1-100)"],
    },
)
async def search_codebase(args: dict[str, Any]) -> dict[str, Any]:
    ...
    return {"content": [{"type": "text", "text": "no matches"}]}
```

- `annotations=ToolAnnotations(readOnlyHint=True)` advertises MCP hints to Claude; `ToolAnnotations(maxResultSizeChars=...)` raises the size up to which a result stays inline.

---

## Permissions

Permission checks run in this order: hooks → deny rules → permission mode → allow rules → the `can_use_tool` callback.

### Permission modes

| Mode | Behaviour |
|---|---|
| `"default"` | Unapproved tools go to `can_use_tool` (or are denied if there is none) |
| `"acceptEdits"` | File edits are auto-approved; other tools follow the default path |
| `"plan"` | Planning only; no edits or commands are executed |
| `"dontAsk"` | Anything not pre-approved is denied rather than asked about |
| `"auto"` | The CLI's classifier decides whether a call is safe to run |
| `"bypassPermissions"` | Everything runs. Only inside a disposable sandbox |

The mode can be changed mid-session with `await client.set_permission_mode("acceptEdits")`.

### `can_use_tool`

The callback receives the tool name, its input and a `ToolPermissionContext`, and returns `PermissionResultAllow` or `PermissionResultDeny`:

```python
from typing import Any
from claude_agent_sdk import (
    ClaudeAgentOptions, PermissionResultAllow, PermissionResultDeny, ToolPermissionContext,
)

PROTECTED = ("/etc", ".env", "secrets")

async def can_use_tool(
    tool_name: str, tool_input: dict[str, Any], context: ToolPermissionContext
) -> PermissionResultAllow | PermissionResultDeny:
    if tool_name in ("Write", "Edit"):
        path = tool_input.get("file_path", "")
        if any(p in path for p in PROTECTED):
            return PermissionResultDeny(message=f"{path} is protected")
    if tool_name == "Bash":
        command = tool_input.get("command", "")
        if "rm -rf" in command:
            return PermissionResultDeny(message="Destructive command refused", interrupt=True)
        # Rewrite the input before it runs
        if command.startswith("pytest") and "-q" not in command:
            return PermissionResultAllow(updated_input={**tool_input, "command": command + " -q"})
    return PermissionResultAllow()

options = ClaudeAgentOptions(permission_mode="default", can_use_tool=can_use_tool)
```

`ToolPermissionContext` carries `tool_use_id` and `agent_id` (to tell parallel calls and subagents apart), `suggestions` (permission updates the CLI proposes), plus `blocked_path`, `decision_reason`, `title`, `display_name` and `description`.

Two traps:

- A tool listed in `allowed_tools`, or any tool under `bypassPermissions`, never reaches the callback. Since 0.2.111 the SDK logs a `CanUseToolShadowedWarning` when this happens.
- Since 0.2.140 `can_use_tool` works with `query()` and string prompts, not just `ClaudeSDKClient`.

### Sandboxing

`sandbox=` runs Bash commands in an OS-level sandbox with optional network restrictions:

```python
from claude_agent_sdk import ClaudeAgentOptions

options = ClaudeAgentOptions(
    sandbox={
        "enabled": True,
        "autoAllowBashIfSandboxed": True,
        "excludedCommands": ["docker"],
        "network": {"allowedDomains": ["pypi.org", "files.pythonhosted.org"]},
    },
)
```

Since 0.2.166 `ignoreViolations` is a `dict[str, list[str]]` mapping a command-pattern substring to violation-text substrings, for example `{"*": ["/tmp"]}`.

---

## Hooks

Hooks are async callbacks that run at fixed points in the loop. Register them per event with `HookMatcher`; `matcher` filters by tool name (a regex such as `"Write|Edit"`), and `None` matches everything.

Supported events: `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `UserPromptSubmit`, `Stop`, `SubagentStart`, `SubagentStop`, `PreCompact`, `Notification` and `PermissionRequest`.

```python
import json
import time
from typing import Any
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, HookContext

async def block_env_files(input_data: dict[str, Any], tool_use_id: str | None,
                          context: HookContext) -> dict[str, Any]:
    path = input_data["tool_input"].get("file_path", "")
    if path.endswith(".env"):
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "Environment files are off limits",
            }
        }
    return {}

async def audit(input_data: dict[str, Any], tool_use_id: str | None,
                context: HookContext) -> dict[str, Any]:
    with open("tool_audit.jsonl", "a") as f:
        f.write(json.dumps({"ts": time.time(), "tool": input_data["tool_name"],
                            "id": tool_use_id}) + "\n")
    return {}

options = ClaudeAgentOptions(
    hooks={
        "PreToolUse": [HookMatcher(matcher="Read|Write|Edit", hooks=[block_env_files])],
        "PostToolUse": [HookMatcher(hooks=[audit], timeout=10)],
    },
)
```

A hook returns a dict. Common keys are `continue_` (set `False` to stop the run; the trailing underscore is converted for you), `stopReason`, `systemMessage`, `reason`, and `hookSpecificOutput`. For `PreToolUse`, `permissionDecision` may be `"allow"`, `"deny"`, `"ask"` or `"defer"`, and `updatedInput` rewrites the tool input. For `PostToolUse`, `additionalContext` adds text for the model and `updatedToolOutput` replaces the result. Since 0.2.166 these output shapes have typed classes (`PreToolUseHookSpecificOutput` and friends).

All matchers registered for an event fire concurrently, so do not rely on ordering between them.

---

## Multi-agent systems

### Programmatic subagents

Define subagents in `agents=`. The main agent delegates to them through the `Agent` tool, and each runs with its own context window, prompt, tools and model:

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition, ResultMessage

options = ClaudeAgentOptions(
    model="claude-opus-5-5",
    allowed_tools=["Read", "Glob", "Grep", "Agent"],
    agents={
        "security-reviewer": AgentDefinition(
            description="Reviews code for security vulnerabilities. Use for any security question.",
            prompt="You are an application security specialist. Report issues with severity and a fix.",
            tools=["Read", "Glob", "Grep"],
            model="sonnet",
        ),
        "test-writer": AgentDefinition(
            description="Writes pytest tests for a given module.",
            prompt="You write focused, deterministic pytest tests.",
            tools=["Read", "Write", "Bash"],
            model="claude-sonnet-5-5",
            maxTurns=20,
        ),
    },
)

async def main() -> None:
    async for message in query(
        prompt="Review src/auth/ for security problems, then add tests for anything you fix.",
        options=options,
    ):
        if isinstance(message, ResultMessage):
            print(message.result)

asyncio.run(main())
```

`AgentDefinition` takes `description` and `prompt` (required), and optionally `tools`, `disallowedTools`, `model` (an alias, `"inherit"`, or a full ID), `skills`, `mcpServers`, `maxTurns`, `effort`, `permissionMode`, `memory`, `initialPrompt` and `background`. Note the camelCase field names: they mirror the CLI's schema.

Messages produced inside a subagent carry `parent_tool_use_id`, pointing at the `Agent` tool call that spawned it. Set `forward_subagent_text=True` to receive the subagent's text and thinking blocks in the stream as well. Background subagents emit `TaskStartedMessage`, `TaskProgressMessage` and `TaskNotificationMessage`; `client.stop_task(task_id)` cancels one.

### Orchestrating from your own code

When the pipeline is fixed, plain Python is clearer than asking a model to orchestrate. Independent steps can run concurrently:

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

async def run(prompt: str, system: str, model: str = "claude-sonnet-5-5") -> str:
    options = ClaudeAgentOptions(model=model, system_prompt=system, allowed_tools=["Read", "Grep"])
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            return message.result or ""
    return ""

async def review(path: str) -> dict[str, str]:
    security, performance = await asyncio.gather(
        run(f"Review {path} for security issues.", "You are a security reviewer."),
        run(f"Review {path} for performance issues.", "You are a performance engineer."),
    )
    summary = await run(
        f"Merge these reviews into one prioritised list:\n\n{security}\n\n{performance}",
        "You write short, prioritised engineering summaries.",
        model="claude-opus-5-5",
    )
    return {"security": security, "performance": performance, "summary": summary}
```

Each `query()` call starts its own CLI process, so concurrency is bounded by memory and your rate limits; use an `asyncio.Semaphore` for large fan-outs.

---

## Model Context Protocol (MCP)

`mcp_servers` accepts a dictionary of server configs, or a path to an MCP JSON config file. Four server types are supported:

```python
import os
from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server

in_process = create_sdk_mcp_server(name="calc", tools=[])  # tools defined with @tool

options = ClaudeAgentOptions(
    mcp_servers={
        # In-process (type "sdk")
        "calc": in_process,
        # Local subprocess over stdio
        "filesystem": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/srv/data"],
        },
        # Remote over streamable HTTP
        "tickets": {
            "type": "http",
            "url": "https://mcp.example.com/mcp",
            "headers": {"Authorization": f"Bearer {os.environ.get('TICKETS_TOKEN', '')}"},
        },
        # Remote over SSE (older servers)
        "legacy": {"type": "sse", "url": "https://legacy.example.com/sse"},
    },
    allowed_tools=["mcp__filesystem__read_file", "mcp__tickets__search"],
    strict_mcp_config=True,   # ignore MCP servers from settings files
)
```

At runtime, `ClaudeSDKClient` can report and manage servers:

```python
from claude_agent_sdk import ClaudeSDKClient

async def show_mcp(client: ClaudeSDKClient) -> None:
    status = await client.get_mcp_status()
    for server in status["mcpServers"]:
        print(server["name"], server["status"])
    await client.toggle_mcp_server("legacy", enabled=False)
```

Since 0.2.140 the SDK supports both `mcp` 1.x and 2.x for in-process servers, and you can pass a hand-built `mcp.server.Server` with resources and prompts at full fidelity.

---

## Structured output

`output_format` asks the agent to finish with JSON that matches a schema. The parsed value arrives on `ResultMessage.structured_output`:

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "risk": {"type": "string", "enum": ["low", "medium", "high"]},
        "files": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "risk", "files"],
    "additionalProperties": False,
}

async def main() -> None:
    options = ClaudeAgentOptions(
        output_format={"type": "json_schema", "schema": schema},
        allowed_tools=["Read", "Glob", "Grep"],
    )
    async for message in query(prompt="Assess the risk of upgrading requests in this repo.",
                               options=options):
        if isinstance(message, ResultMessage) and not message.is_error:
            print(message.structured_output)  # a dict matching the schema

asyncio.run(main())
```

With Pydantic, generate the schema with `Model.model_json_schema()` and validate with `Model.model_validate(message.structured_output)`.

---

## Session management

Every run is a session, persisted by the CLI as a transcript under `~/.claude/projects/`. The session ID is on the init `SystemMessage` and on `ResultMessage`.

### Resume and fork

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

async def last_result(prompt: str, options: ClaudeAgentOptions) -> ResultMessage:
    result = None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            result = message
    assert result is not None
    return result

async def main() -> None:
    first = await last_result("Read README.md and summarise the architecture.", ClaudeAgentOptions())
    sid = first.session_id

    # Continue the same session
    await last_result("Now list the three riskiest modules.", ClaudeAgentOptions(resume=sid))

    # Branch: a new session that starts from the same history
    alt = await last_result("Propose a microservices split instead.",
                            ClaudeAgentOptions(resume=sid, fork_session=True))
    print("forked to", alt.session_id)

asyncio.run(main())
```

`continue_conversation=True` resumes the most recent session in `cwd` without an ID. `resume_session_at=<message uuid>` resumes from an earlier point, and `resume_drops_turn` guards against silently dropping messages you have not seen (both 0.2.137+).

### Inspecting stored sessions

Module-level helpers read and manage transcripts without starting the CLI:

```python
from claude_agent_sdk import list_sessions, get_session_messages, rename_session, tag_session

for info in list_sessions(directory="/srv/repos/payments", limit=10):
    print(info.session_id, info.custom_title or info.summary, info.git_branch)

messages = get_session_messages("SESSION_ID", directory="/srv/repos/payments")
rename_session("SESSION_ID", "Payments risk review")
tag_session("SESSION_ID", "reviewed")
```

There are also `get_session_info`, `fork_session`, `delete_session`, `list_subagents` and `get_subagent_messages`.

### Session stores

For multi-host deployments, `session_store=` mirrors every transcript line to an external store, and `resume` can rebuild a session from it when the local file is missing. `InMemorySessionStore` is provided for tests; implement the `SessionStore` protocol for Redis, S3 or a database. The `*_via_store` / `*_from_store` variants of the helpers above (`list_sessions_from_store`, `fork_session_via_store`, and so on) work against a store instead of the local disk.

```python
from claude_agent_sdk import ClaudeAgentOptions, InMemorySessionStore

store = InMemorySessionStore()
options = ClaudeAgentOptions(session_store=store, session_store_flush="eager")
```

---

## Thinking and effort

Two options control reasoning:

- `effort=` — `"low"`, `"medium"`, `"high"`, `"xhigh"` or `"max"`. This is the main lever on current models.
- `thinking=` — a `ThinkingConfigAdaptive`, `ThinkingConfigEnabled` or `ThinkingConfigDisabled` dict.

```python
from claude_agent_sdk import ClaudeAgentOptions

options = ClaudeAgentOptions(
    model="claude-opus-5-5",
    thinking={"type": "adaptive", "display": "summarized"},
    effort="high",
)
```

On the Claude 5 family thinking is adaptive: the model decides how much to think, and `effort` sets how hard it tries. Fixed budgets (`{"type": "enabled", "budget_tokens": N}`) are rejected by Claude Opus 5.5, Claude Sonnet 5.5, Claude Haiku 5.5 and Claude Fable 5.1, and Opus 5.5 does not allow thinking to be disabled either; use a lower `effort` to make it cheaper. `display: "summarized"` returns readable summaries in `ThinkingBlock`s; the default on these models leaves the thinking text empty.

Claude Opus 5.5 defaults to `medium` effort, so set it explicitly when you want more. `xhigh` is a good setting for long coding and agentic tasks; `low` suits simple subagents.

`max_thinking_tokens` is deprecated in favour of `thinking` and should not be used in new code.

---

## File checkpointing and rewind

With `enable_file_checkpointing=True`, the CLI backs up files before modifying them, and `ClaudeSDKClient.rewind_files()` restores them to their state at any earlier user message. You need the user message UUIDs, which the CLI only sends when asked to replay user messages:

```python
import asyncio
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, UserMessage

async def main() -> None:
    options = ClaudeAgentOptions(
        enable_file_checkpointing=True,
        permission_mode="acceptEdits",
        extra_args={"replay-user-messages": None},
    )
    async with ClaudeSDKClient(options=options) as client:
        checkpoint = None
        await client.query("Refactor utils.py to use dataclasses.")
        async for message in client.receive_response():
            if isinstance(message, UserMessage) and message.uuid and checkpoint is None:
                checkpoint = message.uuid

        # Unhappy with the result: put the files back
        if checkpoint:
            await client.rewind_files(checkpoint)

asyncio.run(main())
```

Rewinding restores files only; the conversation history is unchanged. To rewind the conversation as well, start a new run with `resume=` and `resume_session_at=`.

---

## Context usage, budgets and limits

### Context usage

`get_context_usage()` returns a `ContextUsageResponse` (a `TypedDict` with camelCase keys) describing what is occupying the context window:

```python
from claude_agent_sdk import ClaudeSDKClient

async def report_context(client: ClaudeSDKClient) -> None:
    usage = await client.get_context_usage()
    print(f"{usage['totalTokens']:,} / {usage['maxTokens']:,} tokens ({usage['percentage']:.1f}%)")
    for category in usage["categories"]:
        print(f"  {category['name']}: {category['tokens']:,}")
    if usage["isAutoCompactEnabled"]:
        print("auto-compact at", usage["autoCompactThreshold"])
```

### Limits

| Option | Effect |
|---|---|
| `max_turns` | Stop after N agent turns (`ResultMessage.subtype == "error_max_turns"`) |
| `max_budget_usd` | Stop when the run's cost passes the limit (`"error_max_budget_usd"`) |
| `task_budget={"total": N}` | Tell the model its token budget so it paces itself (advisory, not enforced) |

`ResultMessage.terminal_reason` (0.2.126+) says why the loop ended: `"completed"`, `"max_turns"`, `"aborted_streaming"`, `"aborted_tools"` and so on.

---

## Error handling

Exceptions all derive from `ClaudeSDKError`:

| Exception | Raised when |
|---|---|
| `CLINotFoundError` | The CLI binary cannot be found (no bundled wheel for your platform and nothing on `PATH`) |
| `CLIConnectionError` | The SDK cannot talk to the CLI process |
| `ProcessError` | The CLI exited with an error; has `exit_code` and `stderr` |
| `ResultError` | Subclass of `ProcessError` raised after a terminal error result (0.2.140+); has `subtype`, `errors`, `terminal_reason`, `api_error_status`, `session_id` |
| `CLIJSONDecodeError` | A line from the CLI could not be parsed |

```python
import asyncio
from claude_agent_sdk import (
    query, ClaudeAgentOptions, ResultMessage,
    ClaudeSDKError, CLINotFoundError, ResultError,
)

async def run_with_retry(prompt: str, attempts: int = 3) -> str:
    options = ClaudeAgentOptions(max_turns=30, max_budget_usd=2.0)
    for attempt in range(1, attempts + 1):
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    return message.result or ""
        except CLINotFoundError:
            raise  # configuration problem: retrying will not help
        except ResultError as e:
            retryable = e.terminal_reason == "api_error" or e.api_error_status in (429, 500, 529)
            if not retryable or attempt == attempts:
                raise
            await asyncio.sleep(2 ** attempt)
        except ClaudeSDKError:
            if attempt == attempts:
                raise
            await asyncio.sleep(2 ** attempt)
    return ""
```

An `AssistantMessage` can also carry an `error` field (for example `"rate_limit"` or `"authentication_failed"`) when a single model call fails but the loop continues.

---

## API reference

### `query()`

```python
async def query(
    *,
    prompt: str | AsyncIterable[dict[str, Any]],
    options: ClaudeAgentOptions | None = None,
    transport: Transport | None = None,
) -> AsyncIterator[Message]: ...
```

Passing an async iterable of message dicts as `prompt` streams input (for images or multi-part prompts).

### `ClaudeSDKClient`

```python
class ClaudeSDKClient:
    def __init__(self, options: ClaudeAgentOptions | None = None, transport: Transport | None = None): ...
    async def connect(self, prompt=None) -> None: ...
    async def query(self, prompt, session_id: str = "default") -> None: ...
    def receive_messages(self) -> AsyncIterator[Message]: ...
    def receive_response(self) -> AsyncIterator[Message]: ...
    async def interrupt(self) -> None: ...
    async def set_model(self, model: str | None = None) -> None: ...
    async def set_permission_mode(self, mode) -> None: ...
    async def get_context_usage(self) -> ContextUsageResponse: ...
    async def get_mcp_status(self) -> McpStatusResponse: ...
    async def toggle_mcp_server(self, server_name: str, enabled: bool) -> None: ...
    async def reconnect_mcp_server(self, server_name: str) -> None: ...
    async def rewind_files(self, user_message_id: str) -> None: ...
    async def stop_task(self, task_id: str) -> None: ...
    async def disconnect(self) -> None: ...
```

### `ClaudeAgentOptions` (selected fields)

| Field | Type | Notes |
|---|---|---|
| `model`, `fallback_model` | `str` | Full ID or alias |
| `system_prompt` | `str` or preset dict | See [System prompts](#system-prompts) |
| `tools`, `allowed_tools`, `disallowed_tools` | `list[str]` | Tool set and pre-approvals |
| `permission_mode` | `"default" \| "acceptEdits" \| "plan" \| "bypassPermissions" \| "dontAsk" \| "auto"` | |
| `can_use_tool` | async callback | Returns `PermissionResultAllow` / `PermissionResultDeny` |
| `hooks` | `dict[event, list[HookMatcher]]` | |
| `mcp_servers`, `strict_mcp_config` | dict or path; `bool` | |
| `agents` | `dict[str, AgentDefinition]` | Programmatic subagents |
| `cwd`, `add_dirs` | path; list of paths | Working directory and extra readable directories |
| `max_turns`, `max_budget_usd`, `task_budget` | `int`; `float`; `TaskBudget` | Limits |
| `effort`, `thinking` | see [Thinking and effort](#thinking-and-effort) | |
| `output_format` | `dict` | JSON schema output |
| `resume`, `continue_conversation`, `fork_session`, `session_id` | | Sessions |
| `resume_session_at`, `resume_drops_turn` | `str` | Truncating resume |
| `session_store`, `session_store_flush` | | External transcript mirror |
| `setting_sources` | `list["user" \| "project" \| "local"]` | Filesystem settings to load |
| `skills` | `list[str] \| "all"` | Enable skills without hand-editing `allowed_tools` |
| `plugins` | `list[SdkPluginConfig]` | Local plugins: `{"type": "local", "path": "..."}` |
| `sandbox` | `SandboxSettings` | Bash sandboxing |
| `enable_file_checkpointing` | `bool` | Needed for `rewind_files()` |
| `include_partial_messages`, `include_hook_events`, `forward_subagent_text` | `bool` | Extra stream content |
| `verbatim_prompts` | `bool` | Deliver prompts literally: no `@file` expansion or slash commands (0.2.158+) |
| `env`, `extra_args`, `cli_path`, `settings`, `user` | | Process-level configuration |
| `stderr` | `Callable[[str], None]` | Receives CLI stderr lines |
| `betas` | `list[str]` | e.g. `["context-1m-2025-08-07"]` |
| `load_timeout_ms` | `int` | CLI start-up timeout (default 60 000) |

### What changed recently

The full history is in the SDK's [CHANGELOG](https://github.com/anthropics/claude-agent-sdk-python/blob/main/CHANGELOG.md). Changes since 0.2.93 that affect code on this page:

- **0.2.111** — warning when `can_use_tool` is shadowed by `allowed_tools` or `bypassPermissions`.
- **0.2.126** — `ResultMessage.terminal_reason`; typed `model_usage`.
- **0.2.129 (breaking)** — skill names in `skills=` are validated; replace `skills=["*"]` or `["plugin:*"]` with `skills="all"`.
- **0.2.137** — `ConversationResetMessage` joins the `Message` union (exhaustive `match` statements need a new case); `origin` on `UserMessage`/`ResultMessage`; `resume_session_at` and `resume_drops_turn`.
- **0.2.140** — `ResultError`; `can_use_tool` works with `query()`; `forward_subagent_text`; `mcp` 2.x support.
- **0.2.158** — `verbatim_prompts`.
- **0.2.166** — typed hook outputs and `SystemInitData`; `SandboxSettings.ignoreViolations` changed to `dict[str, list[str]]`.
- **0.2.170** — `get_session_messages()` now returns every branch of parallel tool calls.

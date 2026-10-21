---
title: "Claude Agent SDK (Python) — Production Guide"
description: "Running Python Claude Agent SDK services in production: resilience, cost control, observability, containers, security and scaling."
framework: anthropic-claude-agent-sdk
---

# Claude Agent SDK (Python) — Production Guide

Verified against `claude-agent-sdk` %%version=claude-agent-sdk%% (checked %%versions-checked%%).

The Agent SDK runs the Claude Code CLI as a child process for every `query()` call or `ClaudeSDKClient` session. Most production concerns follow from that: each run costs a process, a working directory and some memory; each run can execute tools on the host; and each run has a cost you can measure precisely from its `ResultMessage`.

---

## Contents

1. [Resilience](#resilience)
2. [Cost control](#cost-control)
3. [Observability](#observability)
4. [Deployment](#deployment)
5. [Security](#security)
6. [Scaling](#scaling)
7. [Troubleshooting](#troubleshooting)

---

## Resilience

### Know what can fail

| Failure | How it surfaces |
|---|---|
| Model API error (overload, 5xx, rate limit) on one call | The CLI retries; if it gives up, `ResultMessage.is_error` and then `ResultError` with `terminal_reason == "api_error"` and `api_error_status` |
| Turn or budget limit reached | `ResultMessage.subtype` of `"error_max_turns"` / `"error_max_budget_usd"`, then `ResultError` |
| CLI missing | `CLINotFoundError` before anything runs |
| CLI crashed | `ProcessError` with `exit_code` and `stderr` |
| Garbled output line | `CLIJSONDecodeError` |

### Retry only what is worth retrying

```python
import asyncio
import random
from claude_agent_sdk import (
    query, ClaudeAgentOptions, ResultMessage,
    CLINotFoundError, CLIConnectionError, ProcessError, ResultError,
)

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 529}

async def run_resilient(prompt: str, options: ClaudeAgentOptions, attempts: int = 4) -> ResultMessage:
    for attempt in range(1, attempts + 1):
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    return message
            raise RuntimeError("no result message")
        except CLINotFoundError:
            raise                                   # deployment problem
        except ResultError as e:
            if e.subtype in ("error_max_turns", "error_max_budget_usd"):
                raise                               # a limit you set: do not retry blindly
            if e.api_error_status not in RETRYABLE_STATUS and e.terminal_reason != "api_error":
                raise
            last = e
        except (CLIConnectionError, ProcessError) as e:
            last = e
        if attempt < attempts:
            await asyncio.sleep(min(30, 2 ** attempt) + random.random())
    raise last
```

Retrying an agent run is not free: it repeats tool calls, which may have side effects. For agents that write files or call external systems, prefer resuming the failed session (`resume=result.session_id`) with a prompt such as "continue from where you stopped", or make tools idempotent.

### Timeouts

Wrap runs in `asyncio.timeout()` (Python 3.11+) or `asyncio.wait_for()`. Cancelling the task tears down the CLI process; since 0.2.111 the SDK shields that clean-up from cancellation so no orphaned `claude` processes are left behind.

```python
import asyncio
from claude_agent_sdk import ClaudeAgentOptions

async def run_with_deadline(prompt: str, seconds: float = 300):
    async with asyncio.timeout(seconds):
        return await run_resilient(prompt, ClaudeAgentOptions(max_turns=40))
```

### Model fallback

`fallback_model="claude-sonnet-5-5"` lets the CLI switch models when the primary is overloaded, rather than failing the run.

---

## Cost control

### Hard limits per run

```python
from claude_agent_sdk import ClaudeAgentOptions

options = ClaudeAgentOptions(
    max_turns=30,              # stop runaway loops
    max_budget_usd=2.00,       # stop when the run's cost passes $2
    task_budget={"total": 200_000},  # tell the model its token budget so it paces itself
)
```

`max_budget_usd` is enforced by the SDK; `task_budget` is advisory and helps the model finish gracefully before a hard limit cuts it off.

### Measure from `ResultMessage`

Do not hard-code prices. Each run reports its own cost, and `model_usage` splits it by model:

```python
from claude_agent_sdk import ResultMessage

def record_cost(result: ResultMessage, tenant: str) -> None:
    metrics.counter("agent_cost_usd", result.total_cost_usd or 0.0, tags={"tenant": tenant})
    for model, usage in (result.model_usage or {}).items():
        metrics.counter("agent_tokens_in", usage.get("inputTokens", 0), tags={"model": model})
        metrics.counter("agent_tokens_out", usage.get("outputTokens", 0), tags={"model": model})
        metrics.counter("agent_cache_read", usage.get("cacheReadInputTokens", 0), tags={"model": model})
```

(`metrics` stands for whatever client you use: StatsD, Prometheus, OpenTelemetry metrics.)

### Spend less per task

In rough order of impact:

1. **Lower effort before changing model.** `effort="low"` or `"medium"` on a capable model often matches a cheaper model at higher effort, and keeps one cache namespace.
2. **Use cheaper models for subagents.** Give `AgentDefinition(model="sonnet")` or `"haiku"` to reading-heavy or routine subagents and keep the main loop on `claude-opus-5-5`.
3. **Keep the system prompt stable.** Use `{"type": "preset", "preset": "claude_code", "exclude_dynamic_sections": True}` or a fixed string, so prompt caching works across runs. Timestamps or request IDs in the system prompt defeat it.
4. **Trim the tool set.** Fewer tools means a smaller prompt; `tools=[]` for pure text tasks.
5. **Cap turns.** Many runaway costs are loops; `max_turns` is the cheapest guard.

A simple routing helper:

```python
from claude_agent_sdk import ClaudeAgentOptions

PROFILES = {
    "simple":   dict(model="claude-haiku-5-5",  effort="low"),
    "standard": dict(model="claude-sonnet-5-5", effort="medium"),
    "complex":  dict(model="claude-opus-5-5",   effort="high"),
}

def options_for(kind: str, **overrides) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(**{**PROFILES[kind], **overrides})
```

---

## Observability

### Structured logs from the message stream

Log one line per tool call and one per result. Everything you need is on the messages:

```python
import json
import logging
import time
from claude_agent_sdk import (
    query, ClaudeAgentOptions, AssistantMessage, UserMessage, ResultMessage, SystemMessage,
    ToolUseBlock, ToolResultBlock,
)

log = logging.getLogger("agent")

async def run_logged(prompt: str, options: ClaudeAgentOptions, request_id: str) -> ResultMessage | None:
    started = time.monotonic()
    result = None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, SystemMessage) and message.subtype == "init":
            log.info(json.dumps({"rid": request_id, "event": "init",
                                 "session": message.data.get("session_id"),
                                 "model": message.data.get("model")}))
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    log.info(json.dumps({"rid": request_id, "event": "tool_use",
                                         "tool": block.name, "id": block.id}))
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ToolResultBlock) and block.is_error:
                    log.warning(json.dumps({"rid": request_id, "event": "tool_error",
                                            "id": block.tool_use_id}))
        elif isinstance(message, ResultMessage):
            result = message
            log.info(json.dumps({
                "rid": request_id, "event": "result", "ok": not message.is_error,
                "subtype": message.subtype, "terminal_reason": message.terminal_reason,
                "turns": message.num_turns, "cost_usd": message.total_cost_usd,
                "duration_ms": message.duration_ms, "wall_ms": int((time.monotonic() - started) * 1000),
            }))
    return result
```

Pass `stderr=lambda line: log.debug("cli: %s", line)` in the options to capture the CLI's own diagnostics.

### Hooks for policy-level auditing

Logging in hooks catches tool calls made by subagents too, and runs before permission decisions. See [Recipe 13](/anthropic-claude-agent-sdk-guide/anthropic_claude_agent_sdk_recipes/#recipe-13-audit-log-with-hooks).

### Tracing

If `opentelemetry-api` is installed and a span is active when a run starts, the SDK passes the W3C trace context (`TRACEPARENT`) to the CLI, so the CLI's telemetry joins your trace. Enable the CLI's exporter with its standard environment variables:

```python
from claude_agent_sdk import ClaudeAgentOptions

options = ClaudeAgentOptions(env={
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "OTEL_METRICS_EXPORTER": "otlp",
    "OTEL_LOGS_EXPORTER": "otlp",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel-collector:4317",
    "CLAUDE_AGENT_SDK_CLIENT_APP": "orders-agent/1.4.0",   # shows up in the User-Agent
})
```

### What to alert on

- error rate by `subtype` / `terminal_reason`;
- p95 cost and turns per run (a sudden rise usually means a prompt or tool regression);
- `RateLimitEvent`s in the stream;
- tool error rate per tool.

---

## Deployment

### Container image

The platform wheels bundle the CLI, so a slim Python image is enough. Run as a non-root user, and give the agent a dedicated working directory.

```dockerfile
FROM python:3.12-slim

RUN useradd --create-home agent \
 && apt-get update && apt-get install -y --no-install-recommends git ripgrep \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt   # pins claude-agent-sdk==%%version=claude-agent-sdk%%

COPY --chown=agent:agent . .
USER agent
ENV HOME=/home/agent PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
```

Session transcripts are written under `$HOME/.claude/`. Mount a volume there if sessions must survive restarts, or use a `SessionStore` (below).

### A small HTTP service

```python
# service.py
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from claude_agent_sdk import ClaudeAgentOptions, ResultError

app = FastAPI()

class Task(BaseModel):
    prompt: str
    session_id: str | None = None

@app.post("/run")
async def run(task: Task) -> dict:
    options = ClaudeAgentOptions(
        model="claude-sonnet-5-5",
        cwd="/work",
        setting_sources=[],              # do not read settings from the image
        allowed_tools=["Read", "Glob", "Grep"],
        max_turns=20,
        max_budget_usd=1.0,
        resume=task.session_id,
    )
    try:
        result = await run_logged(task.prompt, options, request_id=str(uuid.uuid4()))
    except ResultError as e:
        raise HTTPException(status_code=502, detail={"subtype": e.subtype, "session": e.session_id})
    return {"answer": result.result, "session_id": result.session_id, "cost_usd": result.total_cost_usd}

@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}
```

(`run_logged` is the function from [Observability](#observability).)

### Kubernetes notes

- Requests and limits: budget a few hundred MB of memory per concurrent run, and size replicas by concurrent runs, not requests per second.
- Put `ANTHROPIC_API_KEY` in a `Secret` and expose it as an environment variable; never bake it into the image.
- Use a `readOnlyRootFilesystem` with writable `emptyDir` volumes for `/work` and `$HOME`.
- Set `terminationGracePeriodSeconds` longer than your longest expected run, and stop accepting work on `SIGTERM`.

---

## Security

An agent with `Bash` and `Write` can do anything the process user can. Layer the controls:

1. **Least tool privilege.** Start from `tools=[...]` with only what the task needs. Pre-approve narrowly (`"Bash(pytest:*)"`, not `"Bash"`).
2. **Explicit permission mode.** Use `"default"` with a `can_use_tool` callback, or `"dontAsk"` to deny anything not pre-approved. Reserve `bypassPermissions` for disposable sandboxes.
3. **Sandbox Bash.** `sandbox={"enabled": True, "network": {"allowedDomains": [...]}}` adds OS-level filesystem and network isolation.
4. **Isolate settings.** `setting_sources=[]` (or an explicit list) so a stray `~/.claude/settings.json` or a repository's `.claude/` cannot change behaviour.
5. **Treat inputs as untrusted.** Content fetched from the web, files in a repository and tool results can contain prompt injection. Set `verbatim_prompts=True` when you inline untrusted text into prompts, so `@path` expansions and slash commands in that text are not acted on.
6. **Hooks as a policy layer.** A `PreToolUse` hook that denies writes outside the working directory, or network commands, is easy to test and applies to subagents as well.
7. **Secrets.** Pass credentials through `env=` or the process environment, not the prompt. Deny `Read` on `.env` and credential paths.

```python
from claude_agent_sdk import ClaudeAgentOptions

hardened = ClaudeAgentOptions(
    cwd="/work/job-123",
    setting_sources=[],
    tools=["Read", "Glob", "Grep", "Edit", "Bash"],
    allowed_tools=["Read", "Glob", "Grep", "Bash(pytest:*)"],
    disallowed_tools=["WebFetch", "WebSearch", "Read(./.env)"],
    permission_mode="dontAsk",
    sandbox={"enabled": True, "network": {"allowedDomains": []}},
    verbatim_prompts=True,
    max_turns=40,
    max_budget_usd=3.0,
)
```

---

## Scaling

- **Concurrency is processes.** Bound it with an `asyncio.Semaphore` per worker and a queue (Celery, RQ, SQS, Pub/Sub) in front of the workers.
- **Rate limits are per organisation and model.** Spread load across models where quality allows, and watch for `RateLimitEvent` in the stream.
- **Sessions across hosts.** `session_store=` mirrors transcripts to shared storage so any worker can `resume` a session. Implement the `SessionStore` protocol over Redis, S3 or a database; `InMemorySessionStore` is for tests.
- **Workspaces.** Give each run its own directory (`cwd`), and clean it up afterwards. Use `add_dirs` for read-only shared inputs.

---

## Troubleshooting

**429s or `RateLimitEvent`s.** Reduce concurrency, add jittered back-off, move subagents to a different model, or request higher limits.

**Context grows until runs slow down.** Check `await client.get_context_usage()`; large tool results are the usual cause. Narrow tools (`Grep` with patterns rather than `Read` on whole directories), give reading-heavy work to subagents (each has its own context), and keep auto-compaction on.

**Runs hang after a background subagent finishes.** Fixed in 0.2.127 and 0.2.160; upgrade.

**Hooks or `can_use_tool` never fire.** The tool is in `allowed_tools`, or the mode is `bypassPermissions`. The SDK logs `CanUseToolShadowedWarning` for this.

**Different behaviour between laptop and server.** Filesystem settings differ. Set `setting_sources` explicitly.

**Results vary run to run.** Agents are not deterministic. Use `output_format` for machine-read results, pin the model ID, set `effort` explicitly, and test with assertions that tolerate wording changes.

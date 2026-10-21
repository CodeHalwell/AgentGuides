---
title: "Claude Agent SDK (Python) — Architecture Diagrams"
description: "How the Python Claude Agent SDK works: the CLI subprocess, the message and control streams, permission order, sessions, compaction, failure surfaces and deployment."
framework: "claude-agent-sdk"
language: "python"
sidebar:
  label: "Diagrams"
---

Verified against `claude-agent-sdk` %%version=claude-agent-sdk%% (checked %%versions-checked%%). Each diagram shows a mechanism the [comprehensive guide](/anthropic-claude-agent-sdk-guide/anthropic_claude_agent_sdk_comprehensive_guide/) describes in prose.

## Processes

The SDK is a thin client. The agent loop, the built-in tools, permission rules, compaction and the session files all live in the Claude Code CLI, which the SDK starts as a child process (a copy is bundled in the platform wheels).

```
┌──────────────── your Python process ────────────────┐
│  query() / ClaudeSDKClient                          │
│  ClaudeAgentOptions → CLI flags                     │
│  callbacks the CLI calls back into:                 │
│    can_use_tool · hooks · in-process @tool servers  │
└───────────────┬─────────────────────▲───────────────┘
   stdin: prompts, control requests   │ stdout: messages, control requests
┌───────────────▼─────────────────────┴───────────────┐
│  Claude Code CLI (child process)                    │
│  agent loop · built-in tools · permission rules     │
│  compaction · transcripts under ~/.claude/projects/ │
│  stdio / HTTP / SSE MCP servers                     │
└───────────────┬─────────────────────────────────────┘
                │ HTTPS
        Claude API · Amazon Bedrock · Google Vertex AI
```

Every `query()` call, and every `ClaudeSDKClient` connection, is one CLI process. Capacity planning is therefore about processes and memory, not requests per second.

## One `query()` call

```
SDK                                   CLI
 │── spawn with flags ──────────────▶ │
 │── control: initialize ───────────▶ │  registers hooks, agents, SDK MCP servers
 │── user message (the prompt) ─────▶ │
 │◀──────────── SystemMessage(init) ──│  session_id, model, tools
 │◀────────────── AssistantMessage ───│  text, thinking, tool_use blocks
 │◀── control: can_use_tool ──────────│  only for tools that are not pre-approved
 │── PermissionResultAllow/Deny ────▶ │
 │◀────────────────── UserMessage ────│  tool_result blocks
 │            … more turns …          │
 │◀───────────────── ResultMessage ───│  cost, model_usage, terminal_reason
 │                                    │  process exits
```

`ResultMessage` ends each turn. With `query()` the process then exits; with `ClaudeSDKClient` it waits for the next `client.query()`.

## A failed run

```
CLI emits ResultMessage(is_error=True)        SDK yields it to your loop
        │                                              │
CLI exits non-zero                             SDK raises ResultError
                                               (subtype, terminal_reason,
                                                api_error_status, errors, result)
```

| Cause | `subtype` | Other fields |
|---|---|---|
| Model API failed after the CLI's retries | `"success"` (with `is_error=True`) | `result="API Error: …"`, `api_error_status`, `terminal_reason="api_error"` |
| `max_turns` | `"error_max_turns"` | `terminal_reason="max_turns"` |
| `max_budget_usd` | `"error_max_budget_usd"` | |
| Structured output never validated | `"error_max_structured_output_retries"` | |
| CLI crashed, no result | none | `ProcessError` with `exit_code` |
| CLI not installed | none | `CLINotFoundError` before anything runs |

A loop that returns on the first `ResultMessage` never reaches the exception. See [Error handling](/anthropic-claude-agent-sdk-guide/anthropic_claude_agent_sdk_comprehensive_guide/#error-handling).

## Control requests

Both sides can send control requests over the same pipes.

```
CLI → SDK                              SDK → CLI (ClaudeSDKClient methods)
  can_use_tool   → can_use_tool()        interrupt            → interrupt()
  hook_callback  → your hook             set_model            → set_model()
  mcp_message    → in-process @tool      set_permission_mode  → set_permission_mode()
                                         get_context_usage    → get_context_usage()
                                         mcp_status / mcp_toggle / mcp_reconnect
                                         rewind_files         → rewind_files()
                                         stop_task            → stop_task()
```

## Permission order

```
tool call requested
  │
  ├─ 1. PreToolUse hooks            allow · deny · ask · defer · rewrite input
  ├─ 2. deny rules                  disallowed_tools, settings
  ├─ 3. permission mode             bypassPermissions runs everything
  │                                 acceptEdits auto-approves file edits
  │                                 plan executes nothing
  │                                 dontAsk denies whatever is not pre-approved
  ├─ 4. allow rules                 allowed_tools, settings → runs; step 5 is skipped
  └─ 5. can_use_tool                PermissionResultAllow(updated_input=…) / Deny
  │
tool runs (Bash optionally inside the sandbox)
  │
PostToolUse / PostToolUseFailure hooks   (PostToolUse may replace the output)
```

Hooks run even for pre-approved tools, so they are the place for checks that must see every call.

## Subagents

```
main loop ── Agent tool call (subagent_type="security-reviewer") ──┐
                                                                   ▼
                                     subagent: own context window, prompt,
                                     tools and model from AgentDefinition
                                                                   │
           messages it produces carry parent_tool_use_id ◀─────────┤
main loop ◀─────────────── one tool_result: the subagent's summary ┘
```

Hooks and `can_use_tool` apply to subagent tool calls as well; `context.agent_id` tells you which subagent asked.

## Sessions and session stores

```
local (always)                              external (session_store=…)
~/.claude/projects/<project>/<id>.jsonl ──▶ store.append(key, entries)
one JSON line per transcript entry          batched per turn, or "eager"

resume=<id>
  local file present  → CLI resumes from it
  local file missing  → SDK calls store.load(key), writes a temporary JSONL,
                        and the CLI resumes from that (bounded by load_timeout_ms)
fork_session=True     → same history, new session ID
```

## Compaction

```
each turn: CLI tracks context size (client.get_context_usage())
  │
  ├─ below autoCompactThreshold → continue
  └─ at the threshold
        ├─ PreCompact hook (trigger="auto")
        ├─ CLI summarises the earlier conversation with the model
        ├─ SystemMessage(subtype="compact_boundary") in the stream
        └─ the loop continues on the summary plus recent messages
```

The cost of the summarisation call appears in `model_usage`. A manual `/compact` prompt follows the same path with `trigger="manual"`.

## Deployment

```
queue ─▶ worker (bounded by a semaphore)
           │  one CLI process per active run
           │  cwd: a fresh directory per job
           ▼
         Claude Code CLI ──▶ Claude API
           │
           └─ transcripts: $HOME/.claude volume, or a SessionStore
              (Postgres, S3, …) so any worker can resume
```

Size workers by concurrent runs, keep the API key in a secret, and give each run its own working directory. The [production guide](/anthropic-claude-agent-sdk-guide/anthropic_claude_agent_sdk_production_guide/) has the details.

---
title: "Claude Agent SDK (TypeScript) — Architecture Diagrams"
description: "How the TypeScript Claude Agent SDK works: the Claude Code child process, the message and control streams, permission order, failure surfaces, sessions, compaction and deployment."
framework: "claude-agent-sdk"
language: "typescript"
sidebar:
  label: "Diagrams"
---

Verified against `@anthropic-ai/claude-agent-sdk` %%version=@anthropic-ai/claude-agent-sdk%% (checked %%versions-checked%%). Each diagram shows a mechanism the [comprehensive guide](/anthropic-claude-agent-sdk-typescript-guide/claude_agent_sdk_typescript_comprehensive_guide/) describes in prose.

## Processes

`query()` starts Claude Code as a child process (a platform-specific binary installed as an optional dependency). The agent loop, built-in tools, permission rules, compaction and session files live there, not in your Node.js process.

```
┌──────────────── your Node.js process ───────────────┐
│  query({ prompt, options }) → Query (AsyncGenerator) │
│  Options → CLI flags                                 │
│  callbacks Claude Code calls back into:              │
│    canUseTool · hooks · createSdkMcpServer() tools   │
└───────────────┬──────────────────────▲───────────────┘
   stdin: prompts, control requests    │ stdout: SDKMessages, control requests
┌───────────────▼──────────────────────┴───────────────┐
│  Claude Code (child process)                         │
│  agent loop · built-in tools · permission rules      │
│  compaction · transcripts under ~/.claude/projects/  │
│  stdio / HTTP / SSE MCP servers                      │
└───────────────┬──────────────────────────────────────┘
                │ HTTPS
        Claude API · Amazon Bedrock · Google Vertex AI
```

Each `query()` is one process, so capacity is planned in concurrent processes and memory.

## One `query()` call

```
SDK                                   Claude Code
 │── spawn with flags ──────────────▶ │
 │── control: initialize ───────────▶ │  hooks, agents, SDK MCP servers
 │── user message (the prompt) ─────▶ │
 │◀──── system (subtype: 'init') ─────│  session_id, model, tools, mcp_servers
 │◀───────────────────── assistant ───│  message.message.content: text, thinking, tool_use
 │◀── control: can_use_tool ──────────│  only for tools that are not pre-approved
 │── { behavior: 'allow' | 'deny' } ▶ │
 │◀─────────────────────────── user ──│  tool_result blocks
 │            … more turns …          │
 │◀───────────────────────── result ──│  total_cost_usd, modelUsage, terminal_reason
 │                                    │  single prompt: the process exits
```

With an `AsyncIterable<SDKUserMessage>` prompt the process stays up after `result` and waits for the next message.

## A failed run

```
Claude Code emits result (is_error: true)     the iterator yields it
        │                                              │
Claude Code exits non-zero                     the iterator throws
                                               Error('Claude Code returned an error result: …')
```

| Cause | `subtype` | Other fields |
|---|---|---|
| Model API failed after Claude Code's retries | `'success'` (with `is_error: true`) | `result: 'API Error: …'`, `api_error_status`, `terminal_reason: 'api_error'` |
| `maxTurns` | `'error_max_turns'` | `terminal_reason: 'max_turns'`, `errors` |
| `maxBudgetUsd` | `'error_max_budget_usd'` | `errors` |
| Structured output never validated | `'error_max_structured_output_retries'` | `errors` |
| Crash, failed start | no result | the iterator throws |
| `abortController.abort()` | no result | `AbortError` |

Capture the result inside the loop and decide in `catch`; see [Errors and cancellation](/anthropic-claude-agent-sdk-typescript-guide/claude_agent_sdk_typescript_comprehensive_guide/#errors-and-cancellation).

## Permission order

```
tool call requested
  │
  ├─ 1. PreToolUse hooks            allow · deny · ask · defer · updatedInput
  ├─ 2. deny rules                  disallowedTools, settings
  ├─ 3. permissionMode              bypassPermissions runs everything
  │                                 acceptEdits auto-approves file edits
  │                                 plan executes nothing
  │                                 dontAsk denies whatever is not pre-approved
  ├─ 4. allow rules                 allowedTools, settings → runs; step 5 is skipped
  └─ 5. canUseTool                  { behavior: 'allow', updatedInput } / { behavior: 'deny' }
  │
tool runs (Bash optionally inside the sandbox)
  │
PostToolUse / PostToolUseFailure hooks   (PostToolUse may set updatedToolOutput)
```

Hooks run even for pre-approved tools, so they are the place for checks that must see every call.

## Subagents

```
main loop ── Agent tool call (subagent_type: 'security-reviewer') ──┐
                                                                    ▼
                                    subagent: own context window, prompt,
                                    tools and model from options.agents
                                                                    │
     its messages carry parent_tool_use_id and agent_id ◀───────────┤
main loop ◀─────────────── one tool_result: the subagent's summary ─┘
```

Background subagents report through `system` messages with subtype `task_started`, `task_progress` and `task_notification`.

## Sessions and session stores

```
local (unless persistSession: false)        external (sessionStore)
~/.claude/projects/<project>/<id>.jsonl ──▶ store.append(key, entries)
one JSON line per transcript entry          batched per turn, or 'eager'

resume: <id>
  local file present  → Claude Code resumes from it
  local file missing  → SDK calls store.load(key), writes a temporary JSONL,
                        and Claude Code resumes from that (bounded by loadTimeoutMs)
forkSession: true     → same history, new session ID
```

## Compaction

```
each turn: Claude Code tracks context size (q.getContextUsage())
  │
  ├─ below the auto-compact threshold → continue
  └─ at the threshold
        ├─ PreCompact hook (trigger: 'auto')
        ├─ the earlier conversation is summarised by the model
        ├─ system message { subtype: 'compact_boundary', compact_metadata: { pre_tokens, … } }
        ├─ PostCompact hook
        └─ the loop continues on the summary plus recent messages
```

The summarisation call is included in `modelUsage`.

## Deployment

```
queue ─▶ worker (bounded by a semaphore)
           │  one Claude Code process per active run
           │  cwd: a fresh directory per job
           ▼
         Claude Code ──▶ Claude API
           │
           └─ transcripts: $HOME/.claude volume, or a sessionStore
              (Postgres, S3, …) so any worker can resume
```

Do not cache agent runs by prompt: a run reads files and calls tools, so the same prompt can legitimately produce a different result. Keep the system prompt stable instead, so the API's prompt caching applies across runs. The [production guide](/anthropic-claude-agent-sdk-typescript-guide/claude_agent_sdk_typescript_production_guide/) has the details.

---
title: "Claude Agent SDK (TypeScript) — Comprehensive Guide"
description: "Reference for building agents with @anthropic-ai/claude-agent-sdk: query(), streaming input, tools, MCP, subagents, hooks, permissions, sessions and thinking."
framework: anthropic-claude-agent-sdk-typescript
---

# Claude Agent SDK (TypeScript) — Comprehensive Guide

**Verified against `@anthropic-ai/claude-agent-sdk` %%version=@anthropic-ai/claude-agent-sdk%%** (checked %%versions-checked%%). Every code block on this page was type-checked against the installed package with TypeScript and Zod 4.

The Python package has its own guide: [Claude Agent SDK (Python)](/anthropic-claude-agent-sdk-guide/).

---

## Table of contents

1. [Core fundamentals](#core-fundamentals)
2. [Messages](#messages)
3. [Simple agents](#simple-agents)
4. [Multi-turn conversations](#multi-turn-conversations)
5. [Tools integration](#tools-integration)
6. [Permissions](#permissions)
7. [Hooks](#hooks)
8. [Multi-agent systems](#multi-agent-systems)
9. [Model Context Protocol (MCP)](#model-context-protocol-mcp)
10. [Structured output](#structured-output)
11. [Session management](#session-management)
12. [Thinking and effort](#thinking-and-effort)
13. [File checkpointing](#file-checkpointing)
14. [Limits, cost and context](#limits-cost-and-context)
15. [Errors and cancellation](#errors-and-cancellation)
16. [API reference](#api-reference)

---

## Core fundamentals

### What the SDK is

`@anthropic-ai/claude-agent-sdk` is Claude Code as a library. `query()` starts a Claude Code process (a platform-specific binary is installed as an optional dependency), streams typed messages back, and handles the agent loop, built-in tools, permissions, compaction and session persistence. Your code supplies a prompt, options, and optionally custom tools, hooks and permission callbacks.

It was published as `@anthropic-ai/claude-code` until the rename; see the [migration guide](/anthropic-claude-agent-sdk-typescript-guide/claude_agent_sdk_typescript_migration_guide/).

Use it when you want an agent that reads and edits files, runs commands and calls tools on your own infrastructure. For a single model call, or a tool loop over tools you define yourself, the plain `@anthropic-ai/sdk` client is lighter.

### Installation

Node.js 18 or later. Zod 4 is a peer dependency, used to define custom tool schemas.

```bash
npm install @anthropic-ai/claude-agent-sdk@%%version=@anthropic-ai/claude-agent-sdk%% zod
npm install --save-dev typescript @types/node tsx
```

```json
{
  "type": "module",
  "dependencies": {
    "@anthropic-ai/claude-agent-sdk": "%%version=@anthropic-ai/claude-agent-sdk%%",
    "zod": "^4.0.0"
  }
}
```

The package is ESM-only; set `"type": "module"` (or use `.mts` files) so top-level `await` works.

### Authentication

Set `ANTHROPIC_API_KEY` in the environment. For other providers use Claude Code's environment variables, either in the process environment or through `options.env`:

```bash
# Amazon Bedrock
export CLAUDE_CODE_USE_BEDROCK=1 AWS_REGION=us-east-1

# Google Vertex AI
export CLAUDE_CODE_USE_VERTEX=1 CLOUD_ML_REGION=global ANTHROPIC_VERTEX_PROJECT_ID=my-project
```

There is no `apiKey` or `apiProvider` option. Note that `options.env` replaces the child process's environment rather than merging with it, so write `env: { ...process.env, CLAUDE_CODE_USE_BEDROCK: '1' }`.

### First agent

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';

for await (const message of query({
  prompt: 'What is the capital of France?',
  options: { model: 'claude-opus-5-5' },
})) {
  if (message.type === 'assistant') {
    for (const block of message.message.content) {
      if (block.type === 'text') console.log(block.text);
    }
  } else if (message.type === 'result' && message.subtype === 'success') {
    console.log(`[${message.num_turns} turns, $${message.total_cost_usd.toFixed(4)}]`);
  }
}
```

Note `message.message.content`: an assistant message wraps the API's `BetaMessage`, so the content blocks are one level down.

### Models

`model` takes a full model ID or an alias (`'opus'`, `'sonnet'`, `'haiku'`). When omitted, Claude Code's default is used.

| Model | ID | Use for |
|---|---|---|
| Claude Opus 5.5 | `claude-opus-5-5` | Default for agentic and coding work |
| Claude Sonnet 5.5 | `claude-sonnet-5-5` | Faster, cheaper; most subagents |
| Claude Haiku 5.5 | `claude-haiku-5-5` | Classification, routing, simple subagents |
| Claude Fable 5.1 | `claude-fable-5-1` | Hardest long-horizon reasoning; most expensive |

All have a 1M-token context window. `fallbackModel` names a model to use if the primary is overloaded.

---

## Messages

`query()` returns a `Query`: an `AsyncGenerator<SDKMessage>` with control methods. `SDKMessage` is a discriminated union on `type` (and, for system messages, `subtype`). The ones you will handle most:

| `type` | `subtype` | Type | Contents |
|---|---|---|---|
| `system` | `init` | `SDKSystemMessage` | `session_id`, `model`, `tools`, `mcp_servers`, `permissionMode`, `skills` |
| `assistant` | — | `SDKAssistantMessage` | `message` (a `BetaMessage`), `parent_tool_use_id`, `agent_id` |
| `user` | — | `SDKUserMessage` | tool results fed back to the model; `tool_use_result` |
| `stream_event` | — | `SDKPartialAssistantMessage` | raw API stream event (with `includePartialMessages`) |
| `result` | `success` or `error_*` | `SDKResultMessage` | `result`, `total_cost_usd`, `usage`, `modelUsage`, `num_turns`, `structured_output`, `errors` |
| `system` | `compact_boundary` | `SDKCompactBoundaryMessage` | compaction happened |
| `system` | `task_started`, `task_progress`, `task_notification` | | background tasks and subagents |
| `rate_limit_event` | — | `SDKRateLimitEvent` | rate-limit state changes |

The result is always the last message of a turn. Its `subtype` is `'success'`, or one of `'error_during_execution'`, `'error_max_turns'`, `'error_max_budget_usd'`, `'error_max_structured_output_retries'`; only the success variant has `result` and `structured_output`.

A helper that most examples below reuse:

```typescript
// run.ts
import { query, type Options, type SDKResultMessage } from '@anthropic-ai/claude-agent-sdk';

export async function run(prompt: string, options: Options = {}): Promise<SDKResultMessage> {
  let result: SDKResultMessage | undefined;
  for await (const message of query({ prompt, options })) {
    if (message.type === 'result') result = message;
  }
  if (!result) throw new Error('agent finished without a result');
  return result;
}

export function text(result: SDKResultMessage): string {
  if (result.subtype !== 'success') {
    throw new Error(`agent failed: ${result.subtype} ${result.errors.join('; ')}`);
  }
  return result.result;
}
```

---

## Simple agents

### System prompts

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';

// Your own prompt
const reviewer = query({
  prompt: 'Review src/cache.ts for performance problems.',
  options: {
    systemPrompt:
      'You are a performance engineer. Identify bottlenecks, propose concrete changes ' +
      'with code, and estimate the gain. Prefer readability over micro-optimisation.',
    allowedTools: ['Read', 'Grep', 'Glob'],
  },
});

// Claude Code's prompt, plus your additions
const coder = query({
  prompt: 'Add input validation to the signup handler.',
  options: {
    systemPrompt: {
      type: 'preset',
      preset: 'claude_code',
      append: 'Always run the test suite before declaring a task finished.',
    },
  },
});
```

Without a `systemPrompt`, the SDK uses a minimal prompt, not Claude Code's. Ask for the preset when you want Claude Code's coding behaviour.

### Streaming partial output

With `includePartialMessages: true` you also receive `stream_event` messages carrying raw API deltas:

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';

for await (const message of query({
  prompt: 'Explain the CAP theorem in three paragraphs.',
  options: { includePartialMessages: true },
})) {
  if (message.type === 'stream_event') {
    const event = message.event;
    if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
      process.stdout.write(event.delta.text);
    }
  } else if (message.type === 'result') {
    process.stdout.write('\n');
  }
}
```

### Token usage and cost

The result message is authoritative; do not hard-code prices:

```typescript
import { run } from './run.js';

const result = await run('Summarise README.md in five bullet points.', {
  allowedTools: ['Read'],
});
console.log('cost   :', result.total_cost_usd);
console.log('input  :', result.usage.input_tokens);
console.log('output :', result.usage.output_tokens);
console.log('cached :', result.usage.cache_read_input_tokens);
for (const [model, usage] of Object.entries(result.modelUsage)) {
  console.log(`  ${model}: $${usage.costUSD.toFixed(4)} (${usage.outputTokens} output tokens)`);
}
```

---

## Multi-turn conversations

TypeScript has no client class; there are two ways to hold a conversation.

**Streaming input.** Pass an `AsyncIterable<SDKUserMessage>` as the prompt. The process stays alive between turns, and you can push new messages as they arrive:

```typescript
import { query, type SDKUserMessage } from '@anthropic-ai/claude-agent-sdk';

function userMessage(text: string): SDKUserMessage {
  return { type: 'user', message: { role: 'user', content: text }, parent_tool_use_id: null };
}

async function* conversation(): AsyncGenerator<SDKUserMessage> {
  yield userMessage('Name three sorting algorithms.');
  // In a real app, wait for the next user input here.
  yield userMessage('Which of them is stable?');
}

for await (const message of query({
  prompt: conversation(),
  options: { model: 'claude-sonnet-5-5' },
})) {
  if (message.type === 'assistant') {
    for (const block of message.message.content) {
      if (block.type === 'text') console.log(block.text);
    }
  }
}
```

Streaming input is also how you send images, and it is required for the control methods that act mid-session (`interrupt()`, `setModel()`, `setPermissionMode()`, `rewindFiles()`).

**Resume.** Run separate `query()` calls and pass the previous `session_id` as `options.resume` (see [Session management](#session-management)).

The `unstable_v2_*` session API was removed in 0.3.142; use one of the two patterns above.

---

## Tools integration

### Built-in tools

| Tools | Purpose |
|---|---|
| `Read`, `Write`, `Edit`, `NotebookEdit` | Files and notebooks |
| `Glob`, `Grep` | Find files and search contents |
| `Bash` | Shell commands (subject to permissions and sandboxing) |
| `WebSearch`, `WebFetch` | The web |
| `Agent` | Delegate to a subagent |
| `TaskCreate`, `TaskUpdate`, `TaskGet`, `TaskList` | Task tracking (replaced `TodoWrite` for SDK sessions in 0.3.142) |
| `Skill` | Skills; enable with the `skills` option |

`BUILTIN_TOOL_NAMES` exports the full list. Three options shape what the agent may use:

- `tools` — the base set the model sees (a list of names, or `{ type: 'preset', preset: 'claude_code' }`);
- `allowedTools` — pre-approved tools that run without a permission check;
- `disallowedTools` — tools removed entirely.

Rules use Claude Code's permission syntax, for example `'Bash(npm test:*)'` or `'Read(./src/**)'`.

```typescript
import type { Options } from '@anthropic-ai/claude-agent-sdk';

const readOnly: Options = {
  cwd: '/srv/repos/payments',
  allowedTools: ['Read', 'Glob', 'Grep'],
  disallowedTools: ['Bash', 'Write', 'Edit'],
};
```

### Custom tools

Define tools with `tool()` and a Zod shape, and serve them in-process with `createSdkMcpServer()`:

```typescript
import { query, tool, createSdkMcpServer } from '@anthropic-ai/claude-agent-sdk';
import { z } from 'zod';

const rates: Record<string, number> = { 'GBP:EUR': 1.17, 'EUR:GBP': 0.85 };

const exchangeRate = tool(
  'get_exchange_rate',
  'Return the exchange rate between two ISO currency codes',
  {
    base: z.string().length(3).describe('ISO code to convert from'),
    quote: z.string().length(3).describe('ISO code to convert to'),
  },
  async ({ base, quote }) => {
    const rate = rates[`${base.toUpperCase()}:${quote.toUpperCase()}`];
    if (rate === undefined) {
      return { content: [{ type: 'text', text: 'Unknown currency pair' }], isError: true };
    }
    return { content: [{ type: 'text', text: `1 ${base} = ${rate} ${quote}` }] };
  },
  { annotations: { readOnlyHint: true } },
);

const fx = createSdkMcpServer({ name: 'fx', version: '1.0.0', tools: [exchangeRate] });

for await (const message of query({
  prompt: 'How many euros is 250 pounds?',
  options: {
    mcpServers: { fx },
    allowedTools: ['mcp__fx__get_exchange_rate'],
  },
})) {
  if (message.type === 'result' && message.subtype === 'success') console.log(message.result);
}
```

- Tool names are `mcp__<server key>__<tool name>`, where the server key is the key in `mcpServers`.
- The handler's arguments are typed from the Zod shape. Return an MCP `CallToolResult`; set `isError: true` for failures the model should see.
- `createSdkMcpServer({ timeout })` overrides the tool-call timeout for that server (0.3.248+).

---

## Permissions

Order of evaluation: hooks → deny rules → permission mode → allow rules → `canUseTool`.

| `permissionMode` | Behaviour |
|---|---|
| `'default'` | Unapproved tools go to `canUseTool` |
| `'acceptEdits'` | File edits are auto-approved |
| `'plan'` | Planning only; nothing executes |
| `'dontAsk'` | Anything not pre-approved is denied |
| `'auto'` | Claude Code's classifier decides whether a call is safe |
| `'bypassPermissions'` | Everything runs; also requires `allowDangerouslySkipPermissions: true` |

### `canUseTool`

```typescript
import type { CanUseTool } from '@anthropic-ai/claude-agent-sdk';

const PROTECTED = ['/etc', '.env', 'secrets'];

export const canUseTool: CanUseTool = async (toolName, input, { toolUseID, agentID }) => {
  if (toolName === 'Write' || toolName === 'Edit') {
    const path = String(input.file_path ?? '');
    if (PROTECTED.some((p) => path.includes(p))) {
      return { behavior: 'deny', message: `${path} is protected` };
    }
  }
  if (toolName === 'Bash') {
    const command = String(input.command ?? '');
    if (command.includes('rm -rf')) {
      return { behavior: 'deny', message: 'Destructive command refused', interrupt: true };
    }
    if (command.startsWith('npm test') && !command.includes('--silent')) {
      return { behavior: 'allow', updatedInput: { ...input, command: `${command} --silent` } };
    }
  }
  console.log(`allow ${toolName} (${toolUseID}${agentID ? `, subagent ${agentID}` : ''})`);
  return { behavior: 'allow', updatedInput: input };
};
```

The third argument also carries `signal`, `suggestions` (permission updates to persist an "always allow"), `blockedPath`, `decisionReason`, `mcpServer` and display hints. A tool in `allowedTools`, or anything under `bypassPermissions`, never reaches the callback.

For unattended sessions, `permissionPrompts: 'none'` auto-denies prompts without turning off the `auto` classifier.

### Sandboxing

```typescript
import type { Options } from '@anthropic-ai/claude-agent-sdk';

const sandboxed: Options = {
  sandbox: {
    enabled: true,
    autoAllowBashIfSandboxed: true,
    network: { allowedDomains: ['registry.npmjs.org'] },
  },
};
```

---

## Hooks

Hooks are async callbacks keyed by event. Each event takes a list of `HookCallbackMatcher`s; `matcher` filters by tool name (a regex such as `'Write|Edit'`). Events include `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `UserPromptSubmit`, `SessionStart`, `SessionEnd`, `Stop`, `SubagentStart`, `SubagentStop`, `PreCompact`, `PostCompact`, `Notification` and `PermissionRequest`; `HOOK_EVENTS` exports the full list.

```typescript
import { appendFile } from 'node:fs/promises';
import type { HookCallback, Options } from '@anthropic-ai/claude-agent-sdk';

const blockEnvFiles: HookCallback = async (input) => {
  if (input.hook_event_name !== 'PreToolUse') return {};
  const path = String((input.tool_input as Record<string, unknown>).file_path ?? '');
  if (path.endsWith('.env')) {
    return {
      hookSpecificOutput: {
        hookEventName: 'PreToolUse',
        permissionDecision: 'deny',
        permissionDecisionReason: 'Environment files are off limits',
      },
    };
  }
  return {};
};

const audit: HookCallback = async (input, toolUseID) => {
  if (input.hook_event_name === 'PostToolUse') {
    await appendFile(
      'tool_audit.jsonl',
      JSON.stringify({ ts: Date.now(), tool: input.tool_name, id: toolUseID }) + '\n',
    );
  }
  return {};
};

export const options: Options = {
  hooks: {
    PreToolUse: [{ matcher: 'Read|Write|Edit', hooks: [blockEnvFiles] }],
    PostToolUse: [{ hooks: [audit], timeout: 10 }],
  },
};
```

Common output fields: `continue` (set `false` to stop), `stopReason`, `systemMessage`, `decision`, `reason`, and `hookSpecificOutput`. For `PreToolUse`, `permissionDecision` is `'allow' | 'deny' | 'ask' | 'defer'` and `updatedInput` rewrites the input; for `PostToolUse`, `additionalContext` adds text for the model. All matchers for an event run concurrently.

---

## Multi-agent systems

### Subagents

```typescript
import { run, text } from './run.js';

const result = await run(
  'Review src/auth/ for security problems, then add tests for anything you fix.',
  {
    model: 'claude-opus-5-5',
    allowedTools: ['Read', 'Glob', 'Grep', 'Agent'],
    agents: {
      'security-reviewer': {
        description: 'Reviews code for security vulnerabilities. Use for any security question.',
        prompt: 'You are an application security specialist. Report issues with severity and a fix.',
        tools: ['Read', 'Glob', 'Grep'],
        model: 'sonnet',
      },
      'test-writer': {
        description: 'Writes Vitest tests for a given module.',
        prompt: 'You write focused, deterministic Vitest tests.',
        tools: ['Read', 'Write', 'Bash'],
        model: 'claude-sonnet-5-5',
        maxTurns: 20,
      },
    },
  },
);
console.log(text(result));
```

`AgentDefinition` requires `description` and `prompt`; optional fields include `tools`, `disallowedTools`, `model`, `mcpServers`, `skills`, `maxTurns`, `effort`, `permissionMode` and `omitClaudeMd`. Messages produced inside a subagent carry `parent_tool_use_id` and `agent_id`. Set `forwardSubagentText: true` to receive a subagent's text and thinking blocks too. No more than 20 subagents run at once by default (`CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS` changes that).

### Orchestrating in code

When the workflow is fixed, `Promise.all` over independent `query()` calls is simpler and cheaper than asking a model to orchestrate:

```typescript
import { run, text } from './run.js';

async function reviewer(path: string, role: string): Promise<string> {
  return text(
    await run(`Review ${path} for ${role} issues.`, {
      model: 'claude-sonnet-5-5',
      systemPrompt: `You are a ${role} reviewer. Be specific: file, line, fix.`,
      allowedTools: ['Read', 'Glob', 'Grep'],
    }),
  );
}

export async function review(path: string) {
  const [security, performance] = await Promise.all([
    reviewer(path, 'security'),
    reviewer(path, 'performance'),
  ]);
  const summary = text(
    await run(`Merge into one prioritised list:\n\n${security}\n\n${performance}`, {
      model: 'claude-opus-5-5',
      tools: [],
    }),
  );
  return { security, performance, summary };
}
```

---

## Model Context Protocol (MCP)

```typescript
import { createSdkMcpServer, type Options } from '@anthropic-ai/claude-agent-sdk';

const local = createSdkMcpServer({ name: 'local', tools: [] });

export const options: Options = {
  mcpServers: {
    local,                                                 // in-process
    filesystem: {                                          // stdio subprocess
      type: 'stdio',
      command: 'npx',
      args: ['-y', '@modelcontextprotocol/server-filesystem', '/srv/data'],
    },
    tickets: {                                             // streamable HTTP
      type: 'http',
      url: 'https://mcp.example.com/mcp',
      headers: { Authorization: `Bearer ${process.env.TICKETS_TOKEN ?? ''}` },
    },
    legacy: { type: 'sse', url: 'https://legacy.example.com/sse' },
  },
  allowedTools: ['mcp__filesystem__read_file', 'mcp__tickets__search'],
  strictMcpConfig: true,
};
```

Since 0.3.142 MCP servers connect in the background: the session starts immediately and slow servers show `status: 'pending'` in the init message. `CLAUDE_CODE_MCP_STARTUP_WAIT_MS` (via `env`) bounds the first-turn wait. At runtime, `Query` offers `mcpServerStatus()`, `toggleMcpServer()`, `reconnectMcpServer()` and `setMcpServers()`.

---

## Structured output

```typescript
import { z } from 'zod';
import { run } from './run.js';

const Assessment = z.object({
  summary: z.string(),
  risk: z.enum(['low', 'medium', 'high']),
  files: z.array(z.string()),
});

const result = await run('Assess the risk of upgrading express in this repo.', {
  allowedTools: ['Read', 'Glob', 'Grep'],
  outputFormat: { type: 'json_schema', schema: z.toJSONSchema(Assessment) },
});

if (result.subtype === 'success') {
  const assessment = Assessment.parse(result.structured_output);
  console.log(assessment.risk, assessment.files);
}
```

If the model cannot produce valid output after retries, the result subtype is `'error_max_structured_output_retries'`.

---

## Session management

Sessions are persisted under `~/.claude/projects/` unless `persistSession: false`.

```typescript
import { run, text } from './run.js';

const first = await run('Read README.md and summarise the architecture.');
const sessionId = first.session_id;

// Continue the same session
await run('Now list the three riskiest modules.', { resume: sessionId });

// Branch into a new session with the same history
const alt = await run('Propose a microservices split instead.', {
  resume: sessionId,
  forkSession: true,
});
console.log('forked to', alt.session_id, text(alt).slice(0, 80));
```

`continue: true` resumes the most recent session in `cwd`. `resumeSessionAt` resumes from an earlier message, and `resumeDropsTurn` guards against silently discarding messages.

Stored sessions can be inspected without starting a process:

```typescript
import { listSessions, getSessionMessages, renameSession, tagSession } from '@anthropic-ai/claude-agent-sdk';

for (const info of await listSessions({ dir: '/srv/repos/payments', limit: 10 })) {
  console.log(info.sessionId, info.customTitle ?? info.summary);
}
const messages = await getSessionMessages('SESSION_ID', { dir: '/srv/repos/payments' });
await renameSession('SESSION_ID', 'Payments risk review');
await tagSession('SESSION_ID', 'reviewed');
```

`forkSession()`, `deleteSession()`, `getSessionInfo()`, `listSubagents()` and `getSubagentMessages()` are also exported. For multiple hosts, `sessionStore` mirrors transcripts to an external store (`InMemorySessionStore` is provided for tests), and `importSessionToStore()` copies a local session into one.

---

## Thinking and effort

```typescript
import type { Options } from '@anthropic-ai/claude-agent-sdk';

export const options: Options = {
  model: 'claude-opus-5-5',
  thinking: { type: 'adaptive', display: 'summarized' },
  effort: 'high',     // 'low' | 'medium' | 'high' | 'xhigh' | 'max'
};
```

On the Claude 5 family thinking is adaptive and `effort` is the main control. Fixed budgets (`{ type: 'enabled', budgetTokens }`) are rejected by Claude Opus 5.5, Sonnet 5.5, Haiku 5.5 and Fable 5.1, and Opus 5.5 cannot have thinking disabled; lower the effort instead. Claude Opus 5.5 defaults to `medium` effort, so set it explicitly. `display: 'summarized'` makes thinking blocks readable; by default their text is empty. `maxThinkingTokens` is deprecated.

---

## File checkpointing

With `enableFileCheckpointing: true`, Claude Code backs up files before editing them, and `Query.rewindFiles(userMessageId)` restores them. The user message UUIDs come from the replayed user messages the CLI sends back:

```typescript
import { query, type SDKUserMessage } from '@anthropic-ai/claude-agent-sdk';

async function* prompts(): AsyncGenerator<SDKUserMessage> {
  yield { type: 'user', message: { role: 'user', content: 'Refactor utils.ts to use classes.' }, parent_tool_use_id: null };
}

const q = query({
  prompt: prompts(),
  options: {
    enableFileCheckpointing: true,
    permissionMode: 'acceptEdits',
    extraArgs: { 'replay-user-messages': null },
  },
});

let checkpoint: string | undefined;
for await (const message of q) {
  if (message.type === 'user' && 'uuid' in message && message.uuid && !checkpoint) {
    checkpoint = message.uuid;
  }
  if (message.type === 'result') break;
}

if (checkpoint) {
  const outcome = await q.rewindFiles(checkpoint, { dryRun: true });
  console.log(outcome);
  await q.rewindFiles(checkpoint);
}
q.close();
```

Rewinding restores files only; the conversation is unchanged.

---

## Limits, cost and context

| Option | Effect |
|---|---|
| `maxTurns` | Stop after N turns (`error_max_turns`) |
| `maxBudgetUsd` | Stop when cost passes the limit (`error_max_budget_usd`) |
| `taskBudget: { total }` | Tell the model its token budget so it paces itself |

`getContextUsage()` on a running query shows what fills the window:

```typescript
import type { Query } from '@anthropic-ai/claude-agent-sdk';

export async function reportContext(q: Query): Promise<void> {
  const usage = await q.getContextUsage({ detail: 'summary' });
  console.log(`${usage.totalTokens} / ${usage.maxTokens} tokens (${usage.percentage.toFixed(1)}%)`);
  for (const category of usage.categories) console.log(`  ${category.name}: ${category.tokens}`);
}
```

`detail: 'summary'` (0.3.257+) answers from local estimates without extra API calls.

---

## Errors and cancellation

Failures inside the agent loop arrive as a result message with an error subtype and `errors: string[]`, and `terminal_reason` says why the loop ended. A failure to start, or an aborted query, throws. Cancel with an `AbortController`, or call `interrupt()` on a streaming-input query to stop the current turn only:

```typescript
import { query, AbortError } from '@anthropic-ai/claude-agent-sdk';

const controller = new AbortController();
const timer = setTimeout(() => controller.abort(), 120_000);

try {
  for await (const message of query({
    prompt: 'Audit every dependency in package.json.',
    options: { abortController: controller, maxTurns: 40 },
  })) {
    if (message.type === 'result') {
      if (message.subtype === 'success') console.log(message.result);
      else console.error(message.subtype, message.terminal_reason, message.errors);
    }
  }
} catch (err) {
  if (err instanceof AbortError) console.error('timed out');
  else throw err;
} finally {
  clearTimeout(timer);
}
```

An assistant message can also carry an `error` field (for example `'rate_limit'`) when one model call fails but the loop continues.

---

## API reference

### `query()`

```typescript
import type { Options, Query, SDKUserMessage } from '@anthropic-ai/claude-agent-sdk';

declare function query(params: {
  prompt: string | AsyncIterable<SDKUserMessage>;
  options?: Options;
}): Query;
```

`Query` methods (selected): `interrupt()`, `setPermissionMode()`, `setModel()`, `supportedModels()`, `supportedCommands()`, `mcpServerStatus()`, `toggleMcpServer()`, `reconnectMcpServer()`, `setMcpServers()`, `getContextUsage()`, `rewindFiles()`, `streamInput()`, `stopTask()`, `backgroundTasks()`, `accountInfo()` and `close()`.

### `Options` (selected)

| Option | Notes |
|---|---|
| `model`, `fallbackModel` | Full ID or alias |
| `systemPrompt` | string, or `{ type: 'preset', preset: 'claude_code', append? }` |
| `tools`, `allowedTools`, `disallowedTools` | Tool set and pre-approvals |
| `permissionMode`, `canUseTool`, `allowDangerouslySkipPermissions`, `permissionPrompts` | Permissions |
| `hooks` | `Partial<Record<HookEvent, HookCallbackMatcher[]>>` |
| `mcpServers`, `strictMcpConfig` | MCP |
| `agents`, `agent` | Subagent definitions; `agent` runs the main thread as a named agent |
| `cwd`, `additionalDirectories`, `env`, `executable`, `extraArgs`, `pathToClaudeCodeExecutable` | Process |
| `maxTurns`, `maxBudgetUsd`, `taskBudget` | Limits |
| `thinking`, `effort` | Reasoning |
| `outputFormat` | `{ type: 'json_schema', schema }` |
| `resume`, `continue`, `forkSession`, `sessionId`, `resumeSessionAt`, `resumeDropsTurn`, `persistSession` | Sessions |
| `sessionStore`, `sessionStoreFlush` | External transcript mirror |
| `settingSources` | `('user' \| 'project' \| 'local')[]`; omitted loads all, `[]` loads none |
| `skills`, `plugins` | Skills and local plugins |
| `sandbox` | Bash sandboxing |
| `enableFileCheckpointing` | Needed for `rewindFiles()` |
| `includePartialMessages`, `includeHookEvents`, `forwardSubagentText` | Extra stream content |
| `verbatimPrompts` | No `@file` expansion or slash commands in prompts (0.3.280+) |
| `abortController`, `stderr`, `debug` | Control and diagnostics |

### Notable changes since 0.3.165

From the upstream [CHANGELOG](https://github.com/anthropics/claude-agent-sdk-typescript/blob/main/CHANGELOG.md):

- **0.3.223** — `resumeDropsTurn`.
- **0.3.234** — unused `bypass_permissions_disabled` removed from `ExitReason` (compile error if you `case` on it).
- **0.3.248** — per-server `timeout` for `createSdkMcpServer`.
- **0.3.257** — `getContextUsage({ detail })`; `thinkingTokens` in `ModelUsage`.
- **0.3.259** — `permissionPrompts: 'none'`.
- **0.3.271** — `omitClaudeMd` on `AgentDefinition`.
- **0.3.274** — `mcpServer` in `canUseTool` options; `CLAUDE_CODE_MCP_STARTUP_WAIT_MS`.
- **0.3.280** — `verbatimPrompts`.
- **0.3.282** — `@anthropic-ai/claude-agent-sdk/core`, a smaller entry point for bundled apps; `prewarm()` (alpha).
- **0.3.301** — `BUILTIN_TOOL_NAMES` and `LEGACY_TOOL_NAME_ALIASES`; per-tool permission policies on SDK MCP servers.

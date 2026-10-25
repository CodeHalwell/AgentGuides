---
title: "Claude Agent SDK (TypeScript) — Comprehensive Guide"
description: "Reference for building agents with @anthropic-ai/claude-agent-sdk: query(), streaming input, tools, MCP, subagents, hooks, permissions, sessions and thinking."
framework: "claude-agent-sdk"
language: "typescript"
---

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
8. [Skills, slash commands and plugins](#skills-slash-commands-and-plugins)
9. [User input and approvals](#user-input-and-approvals)
10. [Multi-agent systems](#multi-agent-systems)
11. [Model Context Protocol (MCP)](#model-context-protocol-mcp)
12. [Structured output](#structured-output)
13. [Session management](#session-management)
14. [Thinking and effort](#thinking-and-effort)
15. [File checkpointing](#file-checkpointing)
16. [Limits, cost and context](#limits-cost-and-context)
17. [Errors and cancellation](#errors-and-cancellation)
18. [API reference](#api-reference)

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
  options: { model: '%%model=anthropic-flagship%%' },
})) {
  if (message.type === 'assistant') {
    for (const block of message.message.content) {
      if (block.type === 'text') console.log(block.text);
    }
  } else if (message.type === 'result' && !message.is_error) {
    console.log(`[${message.num_turns} turns, $${message.total_cost_usd.toFixed(4)}]`);
  }
}
```

If the run fails, the loop receives a `result` with `is_error: true` and then throws; see [Errors and cancellation](#errors-and-cancellation).

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
| `result` | `success` or `error_*` | `SDKResultMessage` | `is_error`, `result`, `total_cost_usd`, `usage`, `modelUsage`, `num_turns`, `structured_output`, `errors`, `terminal_reason` |
| `system` | `compact_boundary` | `SDKCompactBoundaryMessage` | compaction happened |
| `system` | `task_started`, `task_progress`, `task_notification` | | background tasks and subagents |
| `rate_limit_event` | — | `SDKRateLimitEvent` | rate-limit state changes |

The result is the last message of a turn. Its `subtype` is `'success'`, or one of `'error_during_execution'`, `'error_max_turns'`, `'error_max_budget_usd'`, `'error_max_structured_output_retries'`; only the success variant has `result`, `structured_output` and `api_error_status`. Check `is_error` as well as `subtype`: when the last model call fails, the result has `subtype: 'success'`, `is_error: true` and the API error text in `result`.

After an error result, the `query()` iterator **throws** (`Claude Code returned an error result: …`) when the process exits. A helper that most examples below reuse catches that, so callers always get the result to inspect:

```typescript
// run.ts
import { query, type Options, type SDKResultMessage } from '@anthropic-ai/claude-agent-sdk';

export async function run(prompt: string, options: Options = {}): Promise<SDKResultMessage> {
  let result: SDKResultMessage | undefined;
  try {
    for await (const message of query({ prompt, options })) {
      if (message.type === 'result') result = message;
    }
  } catch (err) {
    // The iterator throws after an error result. Rethrow anything else
    // (failed start, crash, abort).
    if (!result?.is_error) throw err;
  }
  if (!result) throw new Error('agent finished without a result');
  return result;
}

export function text(result: SDKResultMessage): string {
  if (result.subtype !== 'success') {
    throw new Error(`agent failed: ${result.subtype} ${result.errors.join('; ')}`);
  }
  if (result.is_error) throw new Error(`agent failed: ${result.result}`);
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

The result message reports what the run cost; do not hard-code prices. Use `total_cost_usd` and `modelUsage` for accounting: they cover every model call (main loop, subagents, compaction) and are cumulative. `usage` covers the main loop only, and with streaming input only the latest turn.

```typescript
import { run } from './run.js';

const result = await run('Summarise README.md in five bullet points.', {
  allowedTools: ['Read'],
});
console.log('cost:', result.total_cost_usd);
for (const [model, usage] of Object.entries(result.modelUsage)) {
  console.log(
    `  ${model}: $${usage.costUSD.toFixed(4)} in=${usage.inputTokens} out=${usage.outputTokens} ` +
      `cache_read=${usage.cacheReadInputTokens}`,
  );
}
```

With streaming input, each result carries the running total so far: read the latest one rather than adding them up.

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
  options: { model: '%%model=anthropic%%' },
})) {
  if (message.type === 'assistant') {
    for (const block of message.message.content) {
      if (block.type === 'text') console.log(block.text);
    }
  }
}
```

The process stays alive between turns, so a failed turn arrives as a `result` with `is_error: true`; check it on every result.

Streaming input is also how you send images, and it is required for the control methods that act mid-session (`interrupt()`, `setModel()`, `setPermissionMode()`, `rewindFiles()`).

Images go in as Messages API content blocks:

```typescript
import { readFile } from 'node:fs/promises';
import { query, type SDKUserMessage } from '@anthropic-ai/claude-agent-sdk';

async function* withImage(path: string, question: string): AsyncGenerator<SDKUserMessage> {
  const data = (await readFile(path)).toString('base64');
  yield {
    type: 'user',
    message: {
      role: 'user',
      content: [
        { type: 'image', source: { type: 'base64', media_type: 'image/png', data } },
        { type: 'text', text: question },
      ],
    },
    parent_tool_use_id: null,
  };
}

for await (const message of query({ prompt: withImage('chart.png', 'What trend does this chart show?') })) {
  if (message.type === 'result' && message.subtype === 'success') console.log(message.result);
}
```

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
| `AskUserQuestion` | Ask the user a multiple-choice question (see [User input and approvals](#user-input-and-approvals)) |
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

The third argument also carries `signal`, `suggestions` (permission updates to persist an "always allow"), `blockedPath`, `decisionReason`, `mcpServer` and display hints. A tool in `allowedTools`, or anything under `bypassPermissions`, never reaches the callback; Node prints a `CLAUDE_SDK_CAN_USE_TOOL_SHADOWED` warning when that happens. `PreToolUse` hooks still run for those tools, because hooks come first, so use a hook when every call must be checked.

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

Common output fields: `continue` (set `false` to stop), `stopReason`, `systemMessage`, `decision`, `reason`, and `hookSpecificOutput`. For `PreToolUse`, `permissionDecision` is `'allow' | 'deny' | 'ask' | 'defer'` and `updatedInput` rewrites the input; for `PostToolUse`, `additionalContext` adds text for the model and `updatedToolOutput` replaces the tool's output before the model sees it (see [Middleware](/anthropic-claude-agent-sdk-typescript-guide/claude_agent_sdk_typescript_middleware/#redacting-tool-output)). All matchers for an event run concurrently.


### Hook patterns

The hooks above only observe or deny. These three change what the agent does: they rewrite a tool's input, screen and enrich prompts, and push back when the agent tries to finish early. Narrow `input` with `hook_event_name` and TypeScript gives you the right fields for each event. They type-check under `tsc --strict`.

```typescript
import type { HookCallback, Options } from '@anthropic-ai/claude-agent-sdk';

const SECRET = /(api[_-]?key|token|password)\s*[=:]\s*\S+/i;

const addDryRun: HookCallback = async (input) => {
  if (input.hook_event_name !== 'PreToolUse') return {};
  const toolInput = input.tool_input as { command?: string };
  const command = toolInput.command ?? '';
  if (command.startsWith('terraform apply')) {
    return {
      hookSpecificOutput: {
        hookEventName: 'PreToolUse',
        permissionDecision: 'allow',
        updatedInput: { ...toolInput, command: command.replace('apply', 'plan') },
        additionalContext: 'apply was downgraded to plan; ask the user before applying.',
      },
    };
  }
  return {};
};

const screenPrompt: HookCallback = async (input) => {
  if (input.hook_event_name !== 'UserPromptSubmit') return {};
  if (SECRET.test(input.prompt)) {
    return { decision: 'block', reason: 'Remove credentials from the prompt and try again.' };
  }
  return {
    hookSpecificOutput: {
      hookEventName: 'UserPromptSubmit',
      additionalContext: 'Current ticket: PAY-142 (refund rounding).',
    },
  };
};

const requireTests: HookCallback = async (input) => {
  if (input.hook_event_name !== 'Stop') return {};
  if (input.stop_hook_active) return {};
  if (!(input.last_assistant_message ?? '').includes('vitest')) {
    return { decision: 'block', reason: 'Run vitest and report the result before finishing.' };
  }
  return {};
};

export const options: Options = {
  allowedTools: ['Read', 'Edit', 'Bash'],
  hooks: {
    PreToolUse: [{ matcher: 'Bash', hooks: [addDryRun] }],
    UserPromptSubmit: [{ hooks: [screenPrompt] }],
    Stop: [{ hooks: [requireTests] }],
  },
};
```

- `PreToolUse` with `updatedInput` changes the arguments the tool receives. The model is not told unless you also set `additionalContext`.
- `UserPromptSubmit` can block a prompt (`decision: 'block'`) or add context.
- `Stop` fires when the agent is about to finish. `decision: 'block'` with a `reason` keeps the loop going with that reason as feedback. `stop_hook_active` is true when the agent is already continuing because of a previous block, so check it or the hook can keep objecting indefinitely.
- `decision` is only for blocking. To allow or deny a tool call, use `permissionDecision` inside `hookSpecificOutput`.
- `includeHookEvents: true` adds hook lifecycle messages (`hook_started`, `hook_progress`, `hook_response`) to the stream, which helps when a hook does not seem to fire.


---

## Skills, slash commands and plugins

Skills live on disk rather than in code. A skill is a directory with a `SKILL.md` whose frontmatter `description` tells the model when to use it; there is no API for registering one programmatically (subagents can be defined in `agents`).

```text
.claude/skills/security-check/SKILL.md
---
name: security-check
description: Run a security vulnerability scan
---
Analyse the codebase for SQL injection, XSS, exposed credentials and insecure configuration.
```

- `settingSources` decides where skills, `.claude/commands/` files and `CLAUDE.md` are read from. Omitted, it loads user, project and local sources. If you set it, keep `'project'` for `<cwd>/.claude/skills/` (and parent directories up to the repository root) and `'user'` for `~/.claude/skills/`. `settingSources: []` finds no skills.
- `skills` scopes which discovered skills the model may invoke: `'all'`, a list of names (the directory name, or `plugin:skill` for a plugin's skill), or `[]` for none. Wildcards such as `'docs:*'` are rejected. If you also pass `tools`, include `'Skill'` in it.

```typescript
import { query, type Options } from '@anthropic-ai/claude-agent-sdk';

const options: Options = {
  cwd: '/srv/repos/payments',            // .claude/skills/ here or in a parent directory
  settingSources: ['user', 'project'],
  skills: ['security-check'],
  allowedTools: ['Read', 'Grep', 'Glob'],
  maxTurns: 10,
};

for await (const message of query({ prompt: 'Check this project for security issues', options })) {
  if (message.type === 'system' && message.subtype === 'init') {
    console.log('skills:', message.skills);
    console.log('commands:', message.slash_commands);
    console.log('plugins:', message.plugins.map((p) => p.name));
  } else if (message.type === 'result' && message.subtype === 'success') {
    console.log(message.result);
  }
}

export const withPlugin: Options = {
  plugins: [{ type: 'local', path: './plugins/payments-tools' }],
  skills: ['payments-tools:reconcile'],
};
```

The `skills` list in the `init` message holds user-invocable skills only; a skill with `user-invocable: false` still loads for the model but is not listed.

**Slash commands.** A prompt that starts with `/<name>` dispatches a command: a built-in such as `/compact` or `/clear`, a bundled skill, one of your user-invocable skills, or a Markdown file in `.claude/commands/`. Dispatch does not depend on the `skills` list. A name that matches nothing is not an error: the text goes to the model as an ordinary prompt with a note that the command did not run. Use `verbatimPrompts: true` (0.3.280+) when a prompt may contain untrusted text, so neither `/command` nor `@file` is interpreted. Commands that act on history need history, so in TypeScript send them as a later message in a streaming-input query (or with `resume`):

```typescript
import { query, type SDKUserMessage } from '@anthropic-ai/claude-agent-sdk';

const user = (text: string): SDKUserMessage => ({
  type: 'user', message: { role: 'user', content: text }, parent_tool_use_id: null,
});

async function* turns(): AsyncGenerator<SDKUserMessage> {
  yield user('Explain what this project does.');
  yield user('/compact');                  // needs history to summarise
}

for await (const message of query({ prompt: turns(), options: { maxTurns: 5 } })) {
  if (message.type === 'system' && message.subtype === 'compact_boundary') {
    console.log('compacted from', message.compact_metadata.pre_tokens, 'tokens, trigger:', message.compact_metadata.trigger);
  }
}
```

**Plugins.** A plugin bundles skills, commands, agents, hooks and MCP servers in a folder. `plugins: [{ type: 'local', path }]` (shown above) loads one; plugin skills are namespaced `plugin:skill`. The `init` message lists loaded plugins in `plugins`.

These snippets were type-checked with `tsc --strict`. Running them needs a real Claude Code process and skills on disk, so they were not run against a model; the behaviour described is from the SDK type definitions and the official skills documentation.

---

## User input and approvals

Two things reach your code when the agent needs a person: permission prompts for tools that are not pre-approved, and questions the model asks with the built-in `AskUserQuestion` tool. Both arrive through `canUseTool`.

For `AskUserQuestion`, the input holds up to four `questions`, each with a `question`, a short `header`, two to four `options` (`label`, `description`) and `multiSelect`. Collect the answers, put them in the input under `answers` (question text → chosen label; join several labels with `', '` for multi-select), and allow the call. The model receives the answers as the tool result.

For other tools, `suggestions` holds the permission rules Claude Code proposes for an "always allow" choice. Returning them as `updatedPermissions` applies them, so later matching calls no longer reach the callback.

```typescript
import { createInterface } from 'node:readline/promises';
import type { CanUseTool, Options } from '@anthropic-ai/claude-agent-sdk';

const rl = createInterface({ input: process.stdin, output: process.stdout });

interface Question {
  question: string;
  options: { label: string; description: string }[];
  multiSelect: boolean;
}

export const approvals: CanUseTool = async (toolName, input, { suggestions }) => {
  if (toolName === 'AskUserQuestion') {
    const answers: Record<string, string> = {};
    for (const q of input.questions as Question[]) {
      const labels = q.options.map((option) => option.label);
      const menu = labels.map((label, i) => `  ${i + 1}. ${label}`).join('\n');
      const reply = (await rl.question(`${q.question}\n${menu}\n> `)).trim();
      // A number picks an option; anything else is passed on as a free-text answer.
      const choice = labels[Number(reply) - 1];
      answers[q.question] = /^\d+$/.test(reply) && choice !== undefined ? choice : reply;
    }
    return { behavior: 'allow', updatedInput: { ...input, answers } };
  }

  const reply = (await rl.question(`Allow ${toolName} ${JSON.stringify(input)}? [y]es / [a]lways / [n]o `)).trim();
  if (reply === 'a') return { behavior: 'allow', updatedInput: input, updatedPermissions: suggestions };
  if (reply === 'y') return { behavior: 'allow', updatedInput: input };
  return { behavior: 'deny', message: 'The user declined this action.' };
};

export const options: Options = {
  canUseTool: approvals,
  allowedTools: ['Read', 'Glob', 'Grep'],   // never prompt for these
};
```

If you set `tools`, include `'AskUserQuestion'` in it, or the model cannot ask. In a web app, `canUseTool` is where you hand the question to the browser and await the reply; use the `signal` in the third argument and a timeout, and deny when it expires. For unattended jobs, `permissionMode: 'dontAsk'` or `permissionPrompts: 'none'` makes sure nothing waits for a person.

---

## Multi-agent systems

### Subagents

```typescript
import { run, text } from './run.js';

const result = await run(
  'Review src/auth/ for security problems, then add tests for anything you fix.',
  {
    model: '%%model=anthropic-flagship%%',
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
        model: '%%model=anthropic%%',
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
      model: '%%model=anthropic%%',
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
      model: '%%model=anthropic-flagship%%',
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

if (result.subtype === 'success' && !result.is_error) {
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

`forkSession()`, `deleteSession()`, `getSessionInfo()`, `listSubagents()` and `getSubagentMessages()` are also exported.

### Session stores

For multiple hosts, `sessionStore` mirrors every transcript line to an external store, and `resume` rebuilds a session from it when the local file is missing (on another host, or in a fresh container). Claude Code still writes its local copy first; the store receives a second copy in batches, once per turn by default or after every frame with `sessionStoreFlush: 'eager'`. `InMemorySessionStore` is provided for tests, and `importSessionToStore()` copies an existing local session into a store.

A store needs `append(key, entries)` and `load(key)`; `listSessions`, `listSessionSummaries`, `delete` and `listSubkeys` are optional. `key` has `projectKey` (derived from `cwd` by default), `sessionId` and, for subagent transcripts, `subpath`. Treat entries as opaque JSON, and use their `uuid` (when present) to ignore duplicates, because failed batches are retried. A SQLite version:

```typescript
// sqlite-store.ts — node:sqlite needs Node 22.13 or later; the same queries work on Postgres
import { DatabaseSync } from 'node:sqlite';
import type { SessionKey, SessionStore, SessionStoreEntry } from '@anthropic-ai/claude-agent-sdk';

const SCHEMA = `
CREATE TABLE IF NOT EXISTS transcript (
  seq         INTEGER PRIMARY KEY AUTOINCREMENT,
  project_key TEXT NOT NULL,
  session_id  TEXT NOT NULL,
  subpath     TEXT NOT NULL,          -- '' for the main transcript
  uuid        TEXT,                   -- idempotency key; NULL entries are never deduplicated
  entry       TEXT NOT NULL,
  written_ms  INTEGER NOT NULL,
  UNIQUE (project_key, session_id, subpath, uuid)
)`;

export class SQLiteSessionStore implements SessionStore {
  private db: DatabaseSync;

  constructor(path: string) {
    this.db = new DatabaseSync(path);
    this.db.exec(SCHEMA);
  }

  async append(key: SessionKey, entries: SessionStoreEntry[]): Promise<void> {
    const insert = this.db.prepare(
      'INSERT OR IGNORE INTO transcript (project_key, session_id, subpath, uuid, entry, written_ms) ' +
        'VALUES (?, ?, ?, ?, ?, ?)',
    );
    const now = Date.now();
    this.db.exec('BEGIN');
    try {
      for (const entry of entries) {
        insert.run(key.projectKey, key.sessionId, key.subpath ?? '', entry.uuid ?? null, JSON.stringify(entry), now);
      }
      this.db.exec('COMMIT');
    } catch (err) {
      this.db.exec('ROLLBACK');
      throw err;
    }
  }

  async load(key: SessionKey): Promise<SessionStoreEntry[] | null> {
    const rows = this.db
      .prepare('SELECT entry FROM transcript WHERE project_key = ? AND session_id = ? AND subpath = ? ORDER BY seq')
      .all(key.projectKey, key.sessionId, key.subpath ?? '') as { entry: string }[];
    return rows.length ? rows.map((row) => JSON.parse(row.entry) as SessionStoreEntry) : null;
  }

  async listSessions(projectKey: string): Promise<{ sessionId: string; mtime: number }[]> {
    return this.db
      .prepare(
        "SELECT session_id AS sessionId, MAX(written_ms) AS mtime FROM transcript " +
          "WHERE project_key = ? AND subpath = '' GROUP BY session_id",
      )
      .all(projectKey) as { sessionId: string; mtime: number }[];
  }

  async delete(key: SessionKey): Promise<void> {
    if (key.subpath !== undefined) {   // one subagent transcript
      this.db
        .prepare('DELETE FROM transcript WHERE project_key = ? AND session_id = ? AND subpath = ?')
        .run(key.projectKey, key.sessionId, key.subpath);
    } else {                           // the session and everything under it
      this.db
        .prepare('DELETE FROM transcript WHERE project_key = ? AND session_id = ?')
        .run(key.projectKey, key.sessionId);
    }
  }

  async listSubkeys(key: { projectKey: string; sessionId: string }): Promise<string[]> {
    const rows = this.db
      .prepare("SELECT DISTINCT subpath FROM transcript WHERE project_key = ? AND session_id = ? AND subpath != ''")
      .all(key.projectKey, key.sessionId) as { subpath: string }[];
    return rows.map((row) => row.subpath);
  }
}
```

Use the same store on every host and resume by ID wherever the next request lands:

```typescript
import type { Options } from '@anthropic-ai/claude-agent-sdk';
import { SQLiteSessionStore } from './sqlite-store.js';

const store = new SQLiteSessionStore('/data/sessions.db');

export const optionsFor = (sessionId?: string): Options => ({ cwd: '/work', sessionStore: store, resume: sessionId });
```

`loadTimeoutMs` bounds each `load()` / `listSubkeys()` call during a store-backed resume, so a slow store fails the query instead of hanging it. Retention is up to you: the SDK only deletes from a store when you ask it to. The Python SDK ships a conformance suite for adapters (`claude_agent_sdk.testing.run_session_store_conformance`); a port of its append, load, ordering, isolation, delete and subkey checks passes against the TypeScript store above.

---

## Thinking and effort

```typescript
import type { Options } from '@anthropic-ai/claude-agent-sdk';

export const options: Options = {
  model: '%%model=anthropic-flagship%%',
  thinking: { type: 'adaptive', display: 'summarized' },
  effort: 'high',     // 'low' | 'medium' | 'high' | 'xhigh' | 'max'
};
```

On the Claude 5 family thinking is adaptive and `effort` is the main control. Fixed budgets (`{ type: 'enabled', budgetTokens }`) are rejected by Claude Opus 5.5, Sonnet 5.5, Haiku 5.5 and Fable 5.1. Do not rely on disabling thinking either; to make a run cheaper, lower the effort. Claude Opus 5.5 defaults to `medium` effort, so set it explicitly. `display: 'summarized'` makes thinking blocks readable; by default their text is empty. `maxThinkingTokens` is deprecated.

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

A failed run reaches you twice: the iterator first yields a `result` with `is_error: true`, then **throws** `Error('Claude Code returned an error result: …')` when the process exits. A failure to start, a crash and an abort throw without a result.

| What failed | `result` fields |
|---|---|
| The model API, after Claude Code's own retries (overload, 5xx, auth) | `subtype: 'success'`, `is_error: true`, `result: 'API Error: …'`, `api_error_status` (e.g. 529, 401), `terminal_reason: 'api_error'` |
| `maxTurns` reached | `subtype: 'error_max_turns'`, `terminal_reason: 'max_turns'` |
| `maxBudgetUsd` reached | `subtype: 'error_max_budget_usd'` |
| Structured output never validated | `subtype: 'error_max_structured_output_retries'` |

So capture the result inside the loop and decide in `catch`. With streaming input the process keeps running after a failed turn, so check `is_error` on every result as well. Cancel with an `AbortController`, or call `interrupt()` on a streaming-input query to stop the current turn only:

```typescript
import { query, AbortError, type SDKResultMessage } from '@anthropic-ai/claude-agent-sdk';

const controller = new AbortController();
const timer = setTimeout(() => controller.abort(), 120_000);
let result: SDKResultMessage | undefined;

try {
  for await (const message of query({
    prompt: 'Audit every dependency in package.json.',
    options: { abortController: controller, maxTurns: 40 },
  })) {
    if (message.type === 'result') result = message;
  }
} catch (err) {
  if (err instanceof AbortError) console.error('timed out');
  else if (!result?.is_error) throw err;     // crashed or failed to start
} finally {
  clearTimeout(timer);
}

if (result?.subtype === 'success' && !result.is_error) console.log(result.result);
else if (result?.subtype === 'success') console.error('API error', result.api_error_status, result.result);
else if (result) console.error(result.subtype, result.terminal_reason, result.errors);
```

The [production guide](/anthropic-claude-agent-sdk-typescript-guide/claude_agent_sdk_typescript_production_guide/#error-handling) has a retry wrapper built on this, tested against each failure. An assistant message can also carry an `error` field (for example `'rate_limit'`) when one model call fails but the loop continues.

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
| `sessionStore`, `sessionStoreFlush`, `loadTimeoutMs` | External transcript mirror; `loadTimeoutMs` bounds each store `load()` during resume |
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

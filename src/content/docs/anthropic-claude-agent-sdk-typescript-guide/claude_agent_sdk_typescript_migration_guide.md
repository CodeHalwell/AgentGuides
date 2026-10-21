---
title: "Claude Agent SDK (TypeScript) — Migration Guide"
description: "Moving from the SDK in @anthropic-ai/claude-code to @anthropic-ai/claude-agent-sdk, and upgrading across the 0.1, 0.2 and 0.3 series."
framework: "claude-agent-sdk"
language: "typescript"
---

**Target:** `@anthropic-ai/claude-agent-sdk` %%version=@anthropic-ai/claude-agent-sdk%% (checked %%versions-checked%%).

Two parts:

1. Moving code from the SDK that used to ship inside `@anthropic-ai/claude-code` to `@anthropic-ai/claude-agent-sdk` (renamed at 0.1.0).
2. Upgrading an existing `@anthropic-ai/claude-agent-sdk` project to the current release, including the breaking 0.3.142 release.

---

## Part 1 — from `@anthropic-ai/claude-code`

### What changed

| Area | Before | After |
|---|---|---|
| Package | `@anthropic-ai/claude-code` | `@anthropic-ai/claude-agent-sdk` |
| System prompt options | `customSystemPrompt`, `appendSystemPrompt` | one `systemPrompt` (string or preset) |
| Default system prompt | Claude Code's | minimal; request the `claude_code` preset explicitly |
| Filesystem settings | always loaded | controlled by `settingSources` |
| Subagents | `.claude/agents/*.md` files only | also inline via `agents` |

`query()`, `tool()`, `createSdkMcpServer()`, `Options`, `canUseTool` and `hooks` kept their names.

### Step 1: dependencies

```bash
npm uninstall @anthropic-ai/claude-code
npm install @anthropic-ai/claude-agent-sdk@%%version=@anthropic-ai/claude-agent-sdk%% zod@^4
```

The current release lists `zod` ^4 as a peer dependency, and `@anthropic-ai/sdk` and `@modelcontextprotocol/sdk` as peers (since 0.3.143; npm installs them automatically).

### Step 2: imports

```typescript
// Before
// import { query, type Options } from '@anthropic-ai/claude-code';

// After
import { query, type Options } from '@anthropic-ai/claude-agent-sdk';
```

```bash
grep -rl --include='*.ts' --include='*.tsx' "@anthropic-ai/claude-code'" src \
  | xargs sed -i "s#@anthropic-ai/claude-code'#@anthropic-ai/claude-agent-sdk'#g"
```

### Step 3: system prompt

```typescript
import type { Options } from '@anthropic-ai/claude-agent-sdk';

// Before: { appendSystemPrompt: 'Prefer small, reviewable diffs.' }
// After:
const options: Options = {
  systemPrompt: {
    type: 'preset',
    preset: 'claude_code',
    append: 'Prefer small, reviewable diffs.',
  },
};

// Before: { customSystemPrompt: 'You are a release-notes writer.' }
// After:
const writer: Options = { systemPrompt: 'You are a release-notes writer.' };
```

### Step 4: settings sources

Pass `settingSources` explicitly so behaviour does not depend on whatever `~/.claude` or `.claude/` happens to contain on the host:

```typescript
import type { Options } from '@anthropic-ai/claude-agent-sdk';

const withProjectFiles: Options = { settingSources: ['project'] };   // CLAUDE.md, .claude/settings.json
const isolated: Options = { settingSources: [] };                   // nothing from disk
```

In current releases, omitting `settingSources` loads all sources (the CLI default). `'project'` is required for `CLAUDE.md`.

### Step 5: message handling

Messages are a discriminated union. Assistant content is in `message.message.content`, not `message.content`, and there are no `tool_call`, `tool_result` or `error` message types: tool calls are `tool_use` blocks inside assistant messages, tool results are `tool_result` blocks inside user messages, and failures arrive as a `result` message with an error subtype.

```typescript
import type { SDKMessage } from '@anthropic-ai/claude-agent-sdk';

export function handle(message: SDKMessage): void {
  switch (message.type) {
    case 'system':
      if (message.subtype === 'init') console.log('session', message.session_id);
      break;
    case 'assistant':
      for (const block of message.message.content) {
        if (block.type === 'text') console.log(block.text);
        if (block.type === 'tool_use') console.log('tool:', block.name);
      }
      break;
    case 'result':
      if (message.subtype === 'success') console.log('done:', message.result);
      else console.error('failed:', message.subtype, message.errors);
      break;
  }
}
```

---

## Part 2 — upgrading within `@anthropic-ai/claude-agent-sdk`

Most releases track Claude Code and change nothing in the API. These ones matter. Full detail is in the [CHANGELOG](https://github.com/anthropics/claude-agent-sdk-typescript/blob/main/CHANGELOG.md).

### 0.1.x → 0.2.x

| Version | Change | Action |
|---|---|---|
| 0.1.71 | Zod 4 supported alongside Zod 3 | Move to Zod 4; the current peer range is `^4.0.0` |
| 0.2.133 | `'Skill'` in `allowedTools` deprecated; v2 session API deprecated | Use `skills`; use `query()` |
| 0.2.136 | `TodoWrite` deprecated | See 0.3.142 |

### 0.2.x → 0.3.x (breaking)

0.3.142 removed and changed three things:

1. **The `unstable_v2_*` session API is gone** (`unstable_v2_createSession`, `unstable_v2_resumeSession`, `unstable_v2_prompt`, `SDKSession`). Use `query()` with an `AsyncIterable<SDKUserMessage>` prompt for a live conversation, or `options.resume` to continue a session.

   ```typescript
   import { query, type SDKUserMessage } from '@anthropic-ai/claude-agent-sdk';

   async function* turns(inputs: AsyncIterable<string>): AsyncGenerator<SDKUserMessage> {
     for await (const text of inputs) {
       yield { type: 'user', message: { role: 'user', content: text }, parent_tool_use_id: null };
     }
   }

   export function chat(inputs: AsyncIterable<string>) {
     return query({ prompt: turns(inputs), options: { model: 'claude-sonnet-5-5' } });
   }
   ```

2. **MCP servers connect in the background.** The first turn no longer waits for slow servers; they report `status: 'pending'` in the init message. Mark a server `alwaysLoad: true` if turn 1 needs it, or set `MCP_CONNECTION_NONBLOCKING=0` in `env` for the old behaviour.

3. **Task tools replace `TodoWrite`** in SDK sessions. If you render the agent's to-do list, consume `TaskCreate` / `TaskUpdate` tool calls and accumulate by task ID.

### Within 0.3.x

| Version | Change | Action |
|---|---|---|
| 0.3.143 | `@anthropic-ai/sdk` and `@modelcontextprotocol/sdk` became peer dependencies | Yarn classic users add them explicitly |
| 0.3.207 | `canUseTool` returning `{ behavior: 'allow' }` without `updatedInput` now runs the tool with the original input | Older code that always echoed `updatedInput: input` still works |
| 0.3.234 | `bypass_permissions_disabled` removed from `ExitReason` | Delete that `case` |
| 0.3.271 | `persistent` removed from the `MonitorInput` tool type | Only affects code typing Monitor tool inputs |

### Options that never existed

Older drafts of this guide used options that `Options` does not have. TypeScript will reject them; here is what to use instead.

| Wrong | Use |
|---|---|
| `apiKey`, `apiProvider` | `ANTHROPIC_API_KEY`; `CLAUDE_CODE_USE_BEDROCK` / `CLAUDE_CODE_USE_VERTEX` |
| `workingDirectory` | `cwd` |
| `temperature`, `maxTokens` | not configurable; use `effort` |
| `QueryOptions` type | `Options` |
| `enableSubagents` | define `agents` |

### Model IDs

Replace `claude-3-5-sonnet-*`, `claude-3-7-sonnet-*` and `claude-sonnet-4-5` with the current family — `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`, `claude-fable-5-1` — or an alias. On these models fixed thinking budgets are rejected (use `thinking: { type: 'adaptive' }` and `effort`), and Claude Opus 5.5 defaults to `medium` effort.

---

## Troubleshooting

**`Property 'content' does not exist on type 'SDKAssistantMessage'`** — use `message.message.content`.

**`Module '"@anthropic-ai/claude-agent-sdk"' has no exported member 'unstable_v2_createSession'`** — removed in 0.3.142; see above.

**`ERR_REQUIRE_ESM` or top-level `await` errors** — the package is ESM-only. Set `"type": "module"` and `"module": "nodenext"` in `tsconfig.json`.

**Custom tool parameter descriptions are missing** — upgrade past 0.2.89 and use Zod 4's `.describe()`.

**The agent ignores `CLAUDE.md`** — include `'project'` in `settingSources`.

**`bypassPermissions` has no effect** — also set `allowDangerouslySkipPermissions: true`.

---

## Checklist

- [ ] `@anthropic-ai/claude-code` removed; `@anthropic-ai/claude-agent-sdk` and `zod@^4` installed
- [ ] Imports updated
- [ ] `customSystemPrompt` / `appendSystemPrompt` → `systemPrompt`
- [ ] `settingSources` set explicitly
- [ ] `unstable_v2_*` replaced with `query()`
- [ ] Message handling uses `message.message.content` and `result` subtypes
- [ ] Model IDs updated
- [ ] `tsc --noEmit` passes

---
title: "Claude Agent SDK (TypeScript) — Recipes"
description: "Type-checked TypeScript recipes for the Claude Agent SDK: code review, research, QA, documentation, performance and security agents."
framework: "claude-agent-sdk"
language: "typescript"
---

Verified against `@anthropic-ai/claude-agent-sdk` %%version=@anthropic-ai/claude-agent-sdk%% (checked %%versions-checked%%); every block type-checks with `strict: true`. Install with:

```bash
npm install @anthropic-ai/claude-agent-sdk@%%version=@anthropic-ai/claude-agent-sdk%% zod@^4
```

All recipes are ES modules (`"type": "module"`) and reuse this helper. It returns the final `result` message even when the run failed: after an error result the `query()` iterator throws, and the helper turns that back into the result so callers can check `is_error` and `subtype`.

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

For retries, use `runResilient` from the [production guide](/anthropic-claude-agent-sdk-typescript-guide/claude_agent_sdk_typescript_production_guide/#error-handling).

---

## Recipe 1: Multi-turn code review

A reviewer that keeps context across follow-up questions. The prompt is an async queue, so the process stays alive between turns.

```typescript
import { query, type SDKUserMessage } from '@anthropic-ai/claude-agent-sdk';

class Inbox implements AsyncIterable<SDKUserMessage> {
  private queue: SDKUserMessage[] = [];
  private waiting?: (m: IteratorResult<SDKUserMessage>) => void;
  private closed = false;

  send(text: string): void {
    const message: SDKUserMessage = {
      type: 'user', message: { role: 'user', content: text }, parent_tool_use_id: null,
    };
    if (this.waiting) { this.waiting({ value: message, done: false }); this.waiting = undefined; }
    else this.queue.push(message);
  }

  close(): void {
    this.closed = true;
    this.waiting?.({ value: undefined, done: true });
  }

  [Symbol.asyncIterator](): AsyncIterator<SDKUserMessage> {
    return {
      next: () => {
        const next = this.queue.shift();
        if (next) return Promise.resolve({ value: next, done: false });
        if (this.closed) return Promise.resolve({ value: undefined, done: true });
        return new Promise((resolve) => { this.waiting = resolve; });
      },
    };
  }
}

const inbox = new Inbox();
const session = query({
  prompt: inbox,
  options: {
    cwd: process.cwd(),
    systemPrompt: { type: 'preset', preset: 'claude_code', append: 'You are reviewing, not editing.' },
    allowedTools: ['Read', 'Glob', 'Grep'],
    disallowedTools: ['Write', 'Edit', 'Bash'],
  },
});

const followUps = [
  'Review src/auth/ and list issues with file:line and severity.',
  'Which of those would you fix first, and why?',
  'Show the patch for the top issue.',
];
inbox.send(followUps.shift()!);

for await (const message of session) {
  if (message.type === 'assistant') {
    for (const block of message.message.content) if (block.type === 'text') console.log(block.text);
  }
  if (message.type === 'result') {
    const next = followUps.shift();
    if (next) inbox.send(next);
    else inbox.close();
  }
}
```

---

## Recipe 2: Research pipeline

Research with web tools, then a structured summary in a second, tool-free step.

```typescript
import { z } from 'zod';
import { run, text } from './run.js';

const Briefing = z.object({
  headline: z.string(),
  keyFindings: z.array(z.object({ claim: z.string(), source: z.string().url() })),
  openQuestions: z.array(z.string()),
});

export async function research(topic: string) {
  const notes = text(await run(`Research: ${topic}. Use several independent sources and cite URLs.`, {
    model: '%%model=anthropic-flagship%%',
    allowedTools: ['WebSearch', 'WebFetch'],
    disallowedTools: ['Bash', 'Write', 'Edit'],
    maxTurns: 25,
    maxBudgetUsd: 1.5,
  }));

  const structured = await run(`Turn these notes into a briefing:\n\n${notes}`, {
    model: '%%model=anthropic%%',
    tools: [],
    outputFormat: { type: 'json_schema', schema: z.toJSONSchema(Briefing) },
  });
  if (structured.subtype !== 'success' || structured.is_error) throw new Error(`briefing failed: ${structured.subtype}`);
  return Briefing.parse(structured.structured_output);
}

console.log(await research('post-quantum TLS deployment'));
```

---

## Recipe 3: Testing and QA agent

Writes tests, runs them, and fixes them until they pass. Only the test runner is pre-approved for Bash.

```typescript
import { run, text } from './run.js';

export async function addTests(repo: string, modulePath: string): Promise<string> {
  const result = await run(`Add Vitest tests for ${modulePath}. Run them and fix failures.`, {
    cwd: repo,
    systemPrompt:
      'Write focused, deterministic tests. Cover edge cases and error paths. ' +
      'Never change production code to make a test pass without saying so.',
    allowedTools: ['Read', 'Glob', 'Grep', 'Write', 'Edit', 'Bash(npx vitest:*)'],
    permissionMode: 'acceptEdits',
    maxTurns: 40,
  });
  return text(result);
}

console.log(await addTests('/srv/repos/api', 'src/pagination.ts'));
```

---

## Recipe 4: Documentation generator

```typescript
import { run, text } from './run.js';

export async function documentModule(repo: string, dir: string): Promise<string> {
  return text(await run(
    `Write docs/${dir.replace(/\//g, '-')}.md describing the public API of ${dir}: ` +
      'purpose, each exported function with a usage example, and error behaviour.',
    {
      cwd: repo,
      allowedTools: ['Read', 'Glob', 'Grep', 'Write(./docs/**)'],
      permissionMode: 'dontAsk',   // anything outside the allow list is refused
      effort: 'medium',
    },
  ));
}
```

---

## Recipe 5: Performance analysis with a custom tool

Expose your profiler output to the agent as an in-process MCP tool.

```typescript
import { readFile } from 'node:fs/promises';
import { tool, createSdkMcpServer } from '@anthropic-ai/claude-agent-sdk';
import { z } from 'zod';
import { run, text } from './run.js';

const hotspots = tool(
  'get_hotspots',
  'Return the top N functions by self time from the latest CPU profile',
  { limit: z.number().int().min(1).max(50).describe('How many functions to return') },
  async ({ limit }) => {
    const profile = JSON.parse(await readFile('profile.json', 'utf8')) as {
      functions: { name: string; file: string; selfMs: number }[];
    };
    const top = [...profile.functions].sort((a, b) => b.selfMs - a.selfMs).slice(0, limit);
    return { content: [{ type: 'text', text: JSON.stringify(top, null, 2) }] };
  },
  { annotations: { readOnlyHint: true } },
);

const profiler = createSdkMcpServer({ name: 'profiler', version: '1.0.0', tools: [hotspots] });

console.log(text(await run('Find the three biggest hotspots and propose concrete fixes with code.', {
  mcpServers: { profiler },
  allowedTools: ['mcp__profiler__get_hotspots', 'Read', 'Grep'],
  effort: 'high',
})));
```

---

## Recipe 6: Security audit with specialist subagents

```typescript
import { run, text } from './run.js';

const result = await run(
  'Audit this repository. Use the subagents, then give one prioritised list with fixes.',
  {
    model: '%%model=anthropic-flagship%%',
    allowedTools: ['Read', 'Glob', 'Grep', 'Agent'],
    agents: {
      'dependency-auditor': {
        description: 'Checks package.json and lockfiles for risky or outdated dependencies.',
        prompt: 'You audit npm dependencies. Report package, issue and recommended version.',
        tools: ['Read', 'Glob', 'Bash(npm audit:*)'],
        model: 'haiku',
      },
      'code-auditor': {
        description: 'Finds injection, authz and secret-handling flaws in source code.',
        prompt: 'You are an application security reviewer. Report file:line, severity and fix.',
        tools: ['Read', 'Glob', 'Grep'],
        model: 'sonnet',
      },
    },
  },
);
console.log(text(result));
for (const [model, usage] of Object.entries(result.modelUsage)) {
  console.log(model, usage.costUSD.toFixed(4));
}
```

---

## Recipe 7: Human approval for risky commands

```typescript
import { createInterface } from 'node:readline/promises';
import type { CanUseTool } from '@anthropic-ai/claude-agent-sdk';
import { run, text } from './run.js';

const READ_ONLY = ['kubectl get', 'kubectl describe', 'kubectl logs', 'df ', 'uptime'];
const FORBIDDEN = ['kubectl delete', 'rm -rf', 'shutdown', 'reboot'];
const rl = createInterface({ input: process.stdin, output: process.stdout });

const guard: CanUseTool = async (toolName, input) => {
  if (toolName !== 'Bash') return { behavior: 'allow', updatedInput: input };
  const command = String(input.command ?? '');
  if (FORBIDDEN.some((f) => command.includes(f))) {
    return { behavior: 'deny', message: 'Destructive command refused', interrupt: true };
  }
  if (READ_ONLY.some((p) => command.startsWith(p))) return { behavior: 'allow', updatedInput: input };
  const answer = await rl.question(`Allow \`${command}\`? [y/N] `);
  return answer.trim().toLowerCase() === 'y'
    ? { behavior: 'allow', updatedInput: input }
    : { behavior: 'deny', message: 'Operator declined' };
};

console.log(text(await run('The checkout pods keep restarting. Find out why.', {
  systemPrompt: 'You are an SRE. Diagnose before acting and explain each command.',
  allowedTools: ['Read', 'Grep'],
  canUseTool: guard,
  maxTurns: 30,
})));
rl.close();
```

---

## Recipe 8: Testing agents

Three layers, cheapest first:

1. **Your own functions.** `canUseTool` policies, hooks and `tool()` handlers are plain async functions; call them directly.
2. **Your message handling, offline.** Point `pathToClaudeCodeExecutable` at a stub that speaks Claude Code's stream-json protocol and replays a canned result. This tests the success and failure paths (including the throw after an error result) without an API key.
3. **A few end-to-end runs** against the real model, cheap and tolerant of wording.

The stub answers the SDK's control requests and ends every run with the result in `$STUB_RESULT`:

```javascript
#!/usr/bin/env node
// stub-claude.mjs — make it executable: chmod +x stub-claude.mjs
import { createInterface } from 'node:readline';

const send = (obj) => process.stdout.write(JSON.stringify(obj) + '\n');
const result = {
  type: 'result', subtype: 'success', is_error: false, result: 'stub answer', session_id: 'stub',
  num_turns: 1, duration_ms: 1, duration_api_ms: 1, total_cost_usd: 0, usage: {}, modelUsage: {},
  permission_denials: [], ...JSON.parse(process.env.STUB_RESULT ?? '{}'),
};

for await (const line of createInterface({ input: process.stdin })) {
  const message = JSON.parse(line);
  if (message.type === 'control_request') {          // initialize and other control calls
    send({ type: 'control_response', response: { subtype: 'success', request_id: message.request_id, response: {} } });
  } else if (message.type === 'user') {
    send(result);
    process.exit(result.is_error ? 1 : 0);           // Claude Code exits non-zero after an error result
  }
}
```

```typescript
// agent.test.ts — run with: npx tsx --test agent.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { tool } from '@anthropic-ai/claude-agent-sdk';
import { z } from 'zod';
import { run, text } from './run.js';
import { canUseTool } from './permissions.js';   // the policy from the comprehensive guide
import { isTransient } from './resilient.js';    // from the production guide

const STUB = fileURLToPath(new URL('./stub-claude.mjs', import.meta.url));

test('destructive commands are refused', async () => {
  const decision = await canUseTool('Bash', { command: 'rm -rf /' }, {
    signal: new AbortController().signal,
    toolUseID: 'toolu_test',
    requestId: 'req_test',
  });
  assert.equal(decision?.behavior, 'deny');
});

test('tool handlers can be called directly', async () => {
  const lookup = tool('lookup_order', 'Fetch an order', { id: z.string() }, async ({ id }) => ({
    content: [{ type: 'text', text: id === 'ORD-1' ? 'shipped' : 'not found' }],
  }));
  const output = await lookup.handler({ id: 'ORD-1' }, {});
  assert.deepEqual(output.content[0], { type: 'text', text: 'shipped' });
});

test('a successful run returns its text', async () => {
  assert.equal(text(await run('hi', { pathToClaudeCodeExecutable: STUB })), 'stub answer');
});

test('a turn limit comes back as a non-retryable error result', async (t) => {
  process.env.STUB_RESULT = JSON.stringify({
    subtype: 'error_max_turns', is_error: true, errors: ['Reached maximum number of turns'], terminal_reason: 'max_turns',
  });
  t.after(() => { delete process.env.STUB_RESULT; });
  const result = await run('hi', { pathToClaudeCodeExecutable: STUB });
  assert.equal(result.subtype, 'error_max_turns');
  assert.equal(isTransient(result), false);
  assert.throws(() => text(result), /error_max_turns/);
});

test('end to end', { skip: !process.env.ANTHROPIC_API_KEY }, async () => {
  const result = await run('What is 17 * 3? Reply with the number only.', {
    model: '%%model=anthropic-small%%', tools: [], maxTurns: 1,
  });
  assert.match(text(result), /51/);
});
```

The stub covers the protocol only as far as these tests need. To exercise tool calls or hooks offline, extend it to emit `assistant` messages and `control_request`s of its own.

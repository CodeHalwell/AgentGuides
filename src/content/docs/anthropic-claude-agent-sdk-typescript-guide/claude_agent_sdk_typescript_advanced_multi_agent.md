---
title: "Claude Agent SDK Advanced Multi-Agent (TypeScript)"
description: "Routing, specialist subagents, guardrails and human-in-the-loop approval with the TypeScript Claude Agent SDK."
framework: anthropic-claude-agent-sdk-typescript
---

Verified against `@anthropic-ai/claude-agent-sdk` %%version=@anthropic-ai/claude-agent-sdk%% (checked %%versions-checked%%).

## Patterns

- **Route in code when you can.** A cheap classifier (or a regular expression) choosing which agent to run is faster, cheaper and easier to test than a coordinator model.
- **Delegate with subagents when the split depends on the task.** Each subagent has its own context window, tools and model; the parent sees only its summary.
- **Put guardrails at the boundary.** Enforce policy in `canUseTool` and hooks, which apply to subagent tool calls too, not only in prompts.
- **Keep a human in the loop for irreversible actions**, and persist the session ID so a reviewer can inspect or resume the run.

## Router with specialist agents

```typescript
import { z } from 'zod';
import { query, type Options, type SDKResultMessage } from '@anthropic-ai/claude-agent-sdk';

async function run(prompt: string, options: Options): Promise<SDKResultMessage> {
  let result: SDKResultMessage | undefined;
  for await (const m of query({ prompt, options })) if (m.type === 'result') result = m;
  if (!result) throw new Error('no result');
  return result;
}

const Route = z.object({ route: z.enum(['billing', 'code', 'general']) });

const SPECIALISTS: Record<z.infer<typeof Route>['route'], Options> = {
  billing: {
    model: 'claude-sonnet-5-5',
    systemPrompt: 'You answer billing questions using only the billing tools.',
    tools: [],
  },
  code: {
    model: 'claude-opus-5-5',
    systemPrompt: { type: 'preset', preset: 'claude_code' },
    allowedTools: ['Read', 'Glob', 'Grep'],
    cwd: '/srv/repos/app',
  },
  general: { model: 'claude-haiku-5-5', tools: [] },
};

export async function handle(input: string): Promise<string> {
  const routed = await run(input, {
    model: 'claude-haiku-5-5',
    systemPrompt: 'Classify the request: billing, code (about our source code) or general.',
    tools: [],
    effort: 'low',
    outputFormat: { type: 'json_schema', schema: z.toJSONSchema(Route) },
  });
  const route = routed.subtype === 'success' ? Route.parse(routed.structured_output).route : 'general';

  const answer = await run(input, SPECIALISTS[route]);
  return answer.subtype === 'success' ? answer.result : `Sorry, that failed (${answer.subtype}).`;
}
```

## Delegation with guardrails and human approval

The coordinator delegates to subagents; a single `canUseTool` callback enforces policy for all of them and asks a human before anything is written.

```typescript
import { createInterface } from 'node:readline/promises';
import { query, type CanUseTool } from '@anthropic-ai/claude-agent-sdk';

const rl = createInterface({ input: process.stdin, output: process.stdout });

const policy: CanUseTool = async (toolName, input, { agentID }) => {
  const who = agentID ? `subagent ${agentID}` : 'coordinator';
  if (toolName === 'Write' || toolName === 'Edit') {
    const path = String(input.file_path ?? '');
    if (!path.includes('/docs/')) return { behavior: 'deny', message: 'Only docs/ may be modified.' };
    const ok = (await rl.question(`${who} wants to ${toolName} ${path}. Allow? [y/N] `)).trim() === 'y';
    return ok ? { behavior: 'allow', updatedInput: input } : { behavior: 'deny', message: 'Reviewer declined.' };
  }
  return { behavior: 'allow', updatedInput: input };
};

let sessionId: string | undefined;
for await (const message of query({
  prompt: 'Document the public API of src/payments/ in docs/payments.md.',
  options: {
    model: 'claude-opus-5-5',
    cwd: '/srv/repos/app',
    allowedTools: ['Read', 'Glob', 'Grep', 'Agent'],   // Write/Edit deliberately go through policy
    canUseTool: policy,
    agents: {
      analyst: {
        description: 'Reads source code and extracts the public API with signatures and behaviour.',
        prompt: 'List every exported function, its parameters, return value and errors.',
        tools: ['Read', 'Glob', 'Grep'],
        model: 'sonnet',
      },
      writer: {
        description: 'Writes clear Markdown documentation from an API summary.',
        prompt: 'Write concise reference documentation with one example per function.',
        tools: ['Read', 'Write', 'Edit'],
        model: 'sonnet',
      },
    },
  },
})) {
  if (message.type === 'system' && message.subtype === 'init') sessionId = message.session_id;
  if (message.type === 'system' && message.subtype === 'task_started') {
    console.log('started:', message.description);
  }
  if (message.type === 'result') {
    console.log(message.subtype, `$${message.total_cost_usd.toFixed(3)}`, 'session', sessionId);
  }
}
rl.close();
```

## Persisting state for review

Save the session ID with the job. A reviewer can then read the transcript without re-running anything, or resume with corrections:

```typescript
import { getSessionMessages, listSubagents, getSubagentMessages, tagSession } from '@anthropic-ai/claude-agent-sdk';

export async function reviewRun(sessionId: string, dir: string) {
  const messages = await getSessionMessages(sessionId, { dir });
  const subagents = await listSubagents(sessionId, { dir });
  const perAgent = await Promise.all(
    subagents.map(async (id) => ({ id, messages: await getSubagentMessages(sessionId, id, { dir }) })),
  );
  await tagSession(sessionId, 'reviewed', { dir });
  return { messages, perAgent };
}
```

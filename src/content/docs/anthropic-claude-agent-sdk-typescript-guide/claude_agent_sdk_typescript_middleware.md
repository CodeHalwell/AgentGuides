---
title: "Claude Agent SDK Middleware (TypeScript)"
description: "Guardrails, redaction, auditing and tracing for the TypeScript Claude Agent SDK, built from hooks, canUseTool and stream wrappers."
framework: anthropic-claude-agent-sdk-typescript
---

Verified against `@anthropic-ai/claude-agent-sdk` %%version=@anthropic-ai/claude-agent-sdk%% (checked %%versions-checked%%).

The Agent SDK has no middleware class. Three mechanisms cover what middleware usually does:

| Need | Mechanism | Runs |
|---|---|---|
| Inspect or rewrite the prompt | `UserPromptSubmit` hook, or your own code before `query()` | before the model sees it |
| Allow, deny or rewrite a tool call | `PreToolUse` hook or `canUseTool` | before the tool runs, including in subagents |
| Inspect or replace tool output | `PostToolUse` hook | after the tool runs, before the model sees it |
| Transform what your app receives | a wrapper around the `query()` iterator | in your process only |

## Input policy

Block prompts that contain data you must not send to the model:

```typescript
import type { HookCallback } from '@anthropic-ai/claude-agent-sdk';

const CARD = /\b(?:\d[ -]?){13,16}\b/;

export const promptPolicy: HookCallback = async (input) => {
  if (input.hook_event_name === 'UserPromptSubmit' && CARD.test(input.prompt)) {
    return { decision: 'block', reason: 'Prompt appears to contain a card number.' };
  }
  return {};
};
```

## Tool guardrails

```typescript
import type { HookCallback } from '@anthropic-ai/claude-agent-sdk';

export const toolPolicy: HookCallback = async (input) => {
  if (input.hook_event_name !== 'PreToolUse') return {};
  const args = input.tool_input as Record<string, unknown>;
  if (input.tool_name === 'Bash' && /\b(curl|wget)\b/.test(String(args.command ?? ''))) {
    return {
      hookSpecificOutput: {
        hookEventName: 'PreToolUse',
        permissionDecision: 'deny',
        permissionDecisionReason: 'Network access from Bash is not allowed; use WebFetch.',
      },
    };
  }
  return {};
};
```

## Redacting tool output

`PostToolUse` can add context for the model. To keep secrets out of the model's view entirely, deny the read in `PreToolUse` instead; redaction after the fact is a second line of defence.

```typescript
import type { HookCallback } from '@anthropic-ai/claude-agent-sdk';

export const flagSecrets: HookCallback = async (input) => {
  if (input.hook_event_name !== 'PostToolUse') return {};
  const output = JSON.stringify(input.tool_response ?? '');
  if (/sk-ant-[A-Za-z0-9_-]{10,}/.test(output)) {
    return {
      systemMessage: 'A tool result contained an API key.',
      hookSpecificOutput: {
        hookEventName: 'PostToolUse',
        additionalContext: 'The previous tool output contained a secret. Do not repeat or use it.',
      },
    };
  }
  return {};
};
```

## Wiring it together, with an audit log

```typescript
import { appendFile } from 'node:fs/promises';
import { query, type HookCallback, type Options } from '@anthropic-ai/claude-agent-sdk';
import { promptPolicy, toolPolicy, flagSecrets } from './policies.js';

const audit: HookCallback = async (input, toolUseID) => {
  await appendFile('audit.jsonl', JSON.stringify({
    ts: new Date().toISOString(),
    event: input.hook_event_name,
    session: input.session_id,
    tool: 'tool_name' in input ? input.tool_name : undefined,
    toolUseID,
  }) + '\n');
  return {};
};

const options: Options = {
  hooks: {
    UserPromptSubmit: [{ hooks: [promptPolicy] }],
    PreToolUse: [{ hooks: [toolPolicy, audit] }],
    PostToolUse: [{ hooks: [flagSecrets, audit] }],
    PostToolUseFailure: [{ hooks: [audit] }],
  },
};

for await (const message of query({ prompt: 'Summarise the open TODOs in src/.', options })) {
  if (message.type === 'result' && message.subtype === 'success') console.log(message.result);
}
```

All matchers registered for an event run concurrently, so do not rely on their order.

## Stream wrappers

For transformations that only affect your application, wrap the iterator. This one redacts US Social Security numbers from assistant text before your code sees it:

```typescript
import type { SDKMessage } from '@anthropic-ai/claude-agent-sdk';

const SSN = /\b\d{3}-\d{2}-\d{4}\b/g;

export async function* redactText(source: AsyncIterable<SDKMessage>): AsyncGenerator<SDKMessage> {
  for await (const message of source) {
    if (message.type === 'assistant') {
      for (const block of message.message.content) {
        if (block.type === 'text') block.text = block.text.replace(SSN, '***-**-****');
      }
    }
    if (message.type === 'result' && message.subtype === 'success') {
      yield { ...message, result: message.result.replace(SSN, '***-**-****') };
      continue;
    }
    yield message;
  }
}
```

Use it as `for await (const m of redactText(query({ prompt, options })))`.

## Tracing

If your service already uses OpenTelemetry, enable Claude Code's exporter and identify your app:

```typescript
import type { Options } from '@anthropic-ai/claude-agent-sdk';

export const traced: Options = {
  env: {
    ...process.env,
    CLAUDE_CODE_ENABLE_TELEMETRY: '1',
    OTEL_METRICS_EXPORTER: 'otlp',
    OTEL_LOGS_EXPORTER: 'otlp',
    CLAUDE_AGENT_SDK_CLIENT_APP: 'support-bot/2.3.0',
  },
};
```

---
title: "OpenAI Agents SDK: Lifecycle Events, Tool Guardrails and Approvals (TypeScript)"
description: "Agent and runner lifecycle events, tool input and output guardrails, human-in-the-loop approvals with RunState, and model-input filters in the TypeScript Agents SDK."
framework: "openai-agents-sdk"
language: "typescript"
sidebar:
  label: "Events, guardrails & approvals"
  order: 2
---

> Verified against `@openai/agents` %%version=@openai/agents%% (checked %%versions-checked%%). Every block compiles under `tsc --strict` and was run with `ScriptedModel` from `@openai/agents/testing`.

The SDK has no generic middleware chain. It has interception points tied to parts of the agent loop:

| Control | Attach to | Runs | Can it stop the run? |
|---------|-----------|------|----------------------|
| Runner events | `runner.on(...)` on a `Runner` | Every agent, tool call and handoff in runs on that runner | No (observe only) |
| Agent events | `agent.on(...)` | Only while that agent is active | No (observe only) |
| Input / output guardrails | `inputGuardrails` / `outputGuardrails` on the agent or `Runner` | Before / after the agent's turn | Yes: throws a tripwire error |
| Tool input / output guardrails | `inputGuardrails` / `outputGuardrails` on `tool()` | Around one function-tool call | Yes: replace the tool result, or throw |
| `needsApproval` | `tool()`, `agent.asTool()`, `hostedMcpTool()`, shell and computer tools | Before the tool executes | Pauses the run until a person decides |
| `callModelInputFilter` | `run()` options or `Runner` | Just before every model call | Edits what the model sees |
| Handoff `inputFilter` | `handoff(agent, { inputFilter })` | When control moves to another agent | Edits the history the next agent sees |

Agent-level guardrails are covered in the [comprehensive guide](/openai-agents-sdk-typescript-guide/openai_agents_sdk_typescript_comprehensive_guide/#guardrails).

## Lifecycle events

`Agent` and `Runner` are event emitters. Runner events carry the agent as an argument; agent events fire only for that agent. Listeners are called synchronously as the loop reaches each point, so keep them quick.

```typescript
import { Agent, Runner, tool } from '@openai/agents';
import { z } from 'zod';

const getOrderStatus = tool({
  name: 'get_order_status',
  description: 'Look up an order',
  parameters: z.object({ orderId: z.string() }),
  execute: async ({ orderId }) => `Order ${orderId} has shipped.`,
});

const support = new Agent({
  name: 'Support',
  instructions: 'Answer order questions with the tool.',
  tools: [getOrderStatus],
});

const runner = new Runner();
const started = new Map<string, number>();

runner.on('agent_start', (_ctx, agent) => console.log(`[agent] ${agent.name} started`));
runner.on('agent_tool_start', (_ctx, _agent, tool, { toolCall }) => {
  const id = 'callId' in toolCall ? toolCall.callId : tool.name;
  started.set(id, performance.now());
  console.log(`[tool] ${tool.name} started`);
});
runner.on('agent_tool_end', (_ctx, _agent, tool, result, { toolCall }) => {
  const id = 'callId' in toolCall ? toolCall.callId : tool.name;
  const ms = performance.now() - (started.get(id) ?? performance.now());
  console.log(`[tool] ${tool.name} -> ${result} (${ms.toFixed(1)} ms)`);
});
runner.on('agent_handoff', (_ctx, from, to) => console.log(`[handoff] ${from.name} -> ${to.name}`));
runner.on('agent_end', (ctx, agent) =>
  console.log(`[agent] ${agent.name} done; ${ctx.usage.totalTokens} tokens so far`),
);

// Agent-scoped listener
support.on('agent_start', () => console.log('support desk opened'));

const result = await runner.run(support, 'Where is order 1234?');
console.log(result.finalOutput);
```

For per-model-call metrics, read `result.rawResponses` (each has `usage`) or attach a [tracing processor](/openai-agents-sdk-typescript-guide/openai_agents_sdk_typescript_comprehensive_guide/#tracing), which receives a span for every model call.

## Tool guardrails

Tool guardrails wrap one function tool. An input guardrail sees the call the model made before the tool runs; an output guardrail sees the result before the model does. Each returns a behaviour, built with `ToolGuardrailFunctionOutputFactory`:

- `allow()` lets the call proceed.
- `rejectContent(message)` skips the tool (input) or replaces its result (output), and sends `message` to the model instead; the run continues.
- `throwException()` stops the run with `ToolInputGuardrailTripwireTriggered` / `ToolOutputGuardrailTripwireTriggered`. Unlike agent-level guardrails, these reach `run()`'s caller wrapped in a `ToolCallError`; the tripwire is its `error` property.

```typescript
import {
  Agent,
  ToolCallError,
  ToolGuardrailFunctionOutputFactory,
  ToolOutputGuardrailTripwireTriggered,
  defineToolInputGuardrail,
  defineToolOutputGuardrail,
  run,
  tool,
} from '@openai/agents';
import { z } from 'zod';

const capRefund = defineToolInputGuardrail({
  name: 'cap_refund',
  run: async ({ toolCall }) => {
    const args = JSON.parse(toolCall.arguments) as { amount?: number };
    if ((args.amount ?? 0) > 500) {
      return ToolGuardrailFunctionOutputFactory.rejectContent(
        'Refunds over 500 need a supervisor. Tell the customer it has been escalated.',
      );
    }
    return ToolGuardrailFunctionOutputFactory.allow();
  },
});

const CARD = /\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b/;

const noCardNumbers = defineToolOutputGuardrail({
  name: 'no_card_numbers',
  run: async ({ output }) =>
    CARD.test(String(output))
      ? ToolGuardrailFunctionOutputFactory.throwException({ reason: 'card number in output' })
      : ToolGuardrailFunctionOutputFactory.allow(),
});

const issueRefund = tool({
  name: 'issue_refund',
  description: 'Refund part or all of an order',
  parameters: z.object({ orderId: z.string(), amount: z.number() }),
  inputGuardrails: [capRefund],
  outputGuardrails: [noCardNumbers],
  execute: async ({ orderId, amount }) => `Refunded ${amount.toFixed(2)} on ${orderId}.`,
});

const agent = new Agent({ name: 'Refunds', instructions: 'Issue refunds.', tools: [issueRefund] });

try {
  const result = await run(agent, 'Refund 900 on order A-17.');
  console.log(result.finalOutput);
} catch (error) {
  // A thrown tool guardrail reaches your code wrapped in a ToolCallError
  if (error instanceof ToolCallError && error.error instanceof ToolOutputGuardrailTripwireTriggered) {
    console.error('Blocked tool output');
  } else {
    throw error;
  }
}
```

Tool guardrails apply to function tools created with `tool()`. Hosted tools and handoffs bypass them; use agent-level guardrails or approvals for those.

## Human-in-the-loop approvals

Mark a tool with `needsApproval` (a boolean or an async predicate). When the model calls it, the run stops before the tool executes and returns the pending calls in `result.interruptions`. Approve or reject each one on `result.state`, then pass the state back to `run()` to continue where it paused:

```typescript
import { Agent, RunToolApprovalItem, run, tool } from '@openai/agents';
import { z } from 'zod';

const deleteUser = tool({
  name: 'delete_user',
  description: 'Delete a user account permanently',
  parameters: z.object({ userId: z.string(), reason: z.string() }),
  needsApproval: true,
  execute: async ({ userId }) => ({ status: 'deleted', userId }),
});

const banUser = tool({
  name: 'ban_user',
  description: 'Ban a user from the platform',
  parameters: z.object({ userId: z.string(), duration: z.enum(['temporary', 'permanent']) }),
  // Only permanent bans need a person
  needsApproval: async (_ctx, { duration }) => duration === 'permanent',
  execute: async ({ userId, duration }) => ({ status: 'banned', userId, duration }),
});

const admin = new Agent({
  name: 'Admin',
  instructions: 'Manage users with care.',
  tools: [deleteUser, banUser],
});

// Replace with your UI, a Slack button, a ticket queue...
async function askHuman(item: RunToolApprovalItem): Promise<boolean> {
  console.log(`Approve ${item.name ?? 'tool'} with ${item.arguments ?? '{}'}?`);
  return true;
}

let result = await run(admin, 'Delete user_123: repeated terms-of-service violations.');

while (result.interruptions.length > 0) {
  for (const item of result.interruptions) {
    if (await askHuman(item)) result.state.approve(item);
    else result.state.reject(item, { message: 'An administrator declined this action.' });
  }
  result = await run(admin, result.state);
}

console.log(result.finalOutput);
```

### Pausing across requests

Approvals often arrive minutes or days later, in another process. Serialise the state with `toString()`, store it, and restore it with `RunState.fromString()` when the decision comes in:

```typescript
import { Agent, RunState, run, tool } from '@openai/agents';
import { z } from 'zod';

const sendEmail = tool({
  name: 'send_email',
  description: 'Send an email to a customer',
  parameters: z.object({ to: z.string(), body: z.string() }),
  needsApproval: true,
  execute: async ({ to }) => `Sent to ${to}`,
});

const agent = new Agent({ name: 'Mailer', instructions: 'Send emails when asked.', tools: [sendEmail] });
const db = new Map<string, string>(); // stand-in for your database

// Request 1: the run pauses
const paused = await run(agent, 'Email bob@example.com to say his order shipped.');
if (paused.interruptions.length > 0) {
  db.set('run-1', paused.state.toString());
}

// Request 2, later: a reviewer clicks "approve"
const saved = db.get('run-1');
if (saved) {
  const state = await RunState.fromString(agent, saved);
  for (const item of state.getInterruptions()) state.approve(item);
  const resumed = await run(agent, state);
  console.log(resumed.finalOutput);
}
```

Tool input guardrails and approvals interact. By default the approval check comes first: a call that needs approval pauses the run before its input guardrails run, and they run only after a person approves. To reject a bad call without bothering anyone, set `toolExecution: { preApprovalInputGuardrails: true }` in the `run()` options or on the `Runner`; the guardrails still run again just before execution after approval.

Treat stored state as sensitive and trusted-only: it contains the conversation and the pending tool arguments, and loading it resumes execution. `approve(item, { alwaysApprove: true })` approves that tool for the rest of the run. With streaming, wait for `stream.completed`, then read `stream.interruptions` and resume with `run(agent, stream.state, { stream: true })`.

## Filtering model input

`callModelInputFilter` receives the instructions and input items just before each model call and returns what to send. Use it to cap history or strip items; it does not change what the session stores.

```typescript
import { Agent, CallModelInputFilter, run } from '@openai/agents';

const lastTwenty: CallModelInputFilter = ({ modelData }) => ({
  input: modelData.input.slice(-20),
  instructions: modelData.instructions,
});

const agent = new Agent({ name: 'Assistant', instructions: 'Be brief.' });
const result = await run(agent, 'Hello', { callModelInputFilter: lastTwenty });
console.log(result.finalOutput);
```

Handoff `inputFilter` does the same for the history a receiving agent sees; see [message filtering during handoffs](/openai-agents-sdk-typescript-guide/openai_agents_sdk_typescript_comprehensive_guide/#message-filtering-during-handoffs).

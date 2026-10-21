---
title: "OpenAI Agents SDK with TypeScript: Production Guide"
description: "Running TypeScript Agents SDK apps in production: run limits, timeouts and retries, guardrails, approvals, sessions, tracing and data controls, and offline tests with ScriptedModel."
framework: "openai-agents-sdk"
language: "typescript"
sidebar:
  label: "Production"
  order: 5
---

> Verified against `@openai/agents` %%version=@openai/agents%% (checked %%versions-checked%%). Every TypeScript block compiles under `tsc --strict` and was run with `ScriptedModel` from `@openai/agents/testing`.

This page covers the controls the SDK gives you for running agents in production. Hosting is ordinary Node.js: an agent runs inside whatever server, worker or function you already deploy. For an HTTP streaming endpoint with a Dockerfile and Kubernetes manifest, see [Streaming server (Express)](/openai-agents-sdk-typescript-guide/openai_agents_streaming_server_express/).

## Pin the release

All `@openai/agents*` packages share one version, which is `0.Y.Z`: a minor bump may break public interfaces. Pin an exact version and read the [release notes](/openai-agents-sdk-typescript-guide/#whats-new) before moving it:

```bash
npm install --save-exact @openai/agents@%%version=@openai/agents%% zod
```

Set `model` on every agent. The default model changed in 0.15 and changes behaviour and cost when you upgrade.

## Bound every run

| Limit | Where | Default | On breach |
|-------|-------|---------|-----------|
| Turns (model calls) per run | `run(..., { maxTurns })` | 10 | `MaxTurnsExceededError`, or your `errorHandlers.maxTurns` |
| Whole run | `run(..., { signal: AbortSignal.timeout(ms) })` | none | the run is aborted |
| One model-call attempt | `modelSettings.timeoutMs` | none | the call is aborted (and retried if a policy allows it) |
| Model-call retries | `modelSettings.retry` | off | the last error is thrown |
| One function-tool call | `tool({ timeoutMs, timeoutBehavior })` | none | an error result goes to the model (`'error_as_result'`) or the run throws (`'raise_exception'`) |
| Output tokens per call | `modelSettings.maxTokens` | model limit | the response is truncated |

```typescript
import { Agent, ModelSettings, Runner, retryPolicies, tool } from '@openai/agents';
import { z } from 'zod';

const resilient: ModelSettings = {
  timeoutMs: 60_000,
  maxTokens: 2_000,
  retry: {
    maxRetries: 3,
    backoff: { initialDelayMs: 1_000, maxDelayMs: 20_000, multiplier: 2, jitter: true },
    policy: retryPolicies.any(
      retryPolicies.providerSuggested(), // honour the API's retry advice
      retryPolicies.networkError(),      // connection resets and timeouts
      retryPolicies.httpStatus([429, 500, 502, 503]),
    ),
  },
};

const lookupAccount = tool({
  name: 'lookup_account',
  description: 'Fetch an account summary',
  parameters: z.object({ accountId: z.string() }),
  timeoutMs: 10_000,
  execute: async ({ accountId }) => `Account ${accountId}: active`,
});

const agent = new Agent({
  name: 'Support',
  instructions: 'Answer account questions.',
  model: '%%model=openai%%',
  modelSettings: resilient,
  tools: [lookupAccount],
});

const runner = new Runner({ workflowName: 'support' });

const result = await runner.run(agent, 'Is account 77 active?', {
  maxTurns: 6,
  signal: AbortSignal.timeout(120_000),
  errorHandlers: {
    maxTurns: () => ({
      finalOutput: "Sorry, I couldn't finish that. A colleague will follow up.",
      includeInHistory: false,
    }),
  },
});
console.log(result.finalOutput);
```

Retries happen inside one `run()` call and only when the request is safe to replay. Do not wrap `run()` in your own retry loop: a re-run repeats tool calls that may already have had side effects. `errorHandlers` also accepts `modelRefusal` and `invalidFinalOutput`.

## Guardrails and approvals

- Input guardrails run **in parallel** with the first model call unless `runInParallel: false`; use blocking mode for checks that should prevent spend. A small model with a structured verdict is more reliable than keyword lists ([example](/openai-agents-sdk-typescript-guide/openai_agents_sdk_typescript_comprehensive_guide/#guardrails)).
- Output guardrails see only the final output. They cannot recall text that has already been streamed.
- Put `needsApproval` on tools that move money, send messages or change records. Store `result.state.toString()` server-side while a person decides, then resume with `RunState.fromString(agent, saved)` ([flow](/openai-agents-sdk-typescript-guide/openai_agents_sdk_middleware_typescript/#human-in-the-loop-approvals)). Never accept a serialised state from a client.
- [Tool guardrails](/openai-agents-sdk-typescript-guide/openai_agents_sdk_middleware_typescript/#tool-guardrails) cap arguments and redact tool output per tool.

## Sessions

The SDK ships `MemorySession` (in-process), `OpenAIConversationsSession` (stored by OpenAI) and `OpenAIResponsesCompactionSession` (wraps another session and compacts long histories). For Redis, Postgres or DynamoDB, implement the five-method `Session` interface; the [comprehensive guide](/openai-agents-sdk-typescript-guide/openai_agents_sdk_typescript_comprehensive_guide/#sessions) has a complete example.

Key sessions by conversation, for example `` `${tenantId}:${conversationId}` ``. A key that changes per request (a timestamp, a random UUID) gives the user a new conversation every time. Keep secrets out of the run `context` unless tools need them: the context is passed to every tool, guardrail and event listener.

## Tracing and data controls

Tracing is on by default and exports to the OpenAI dashboard using your OpenAI key. Since 0.14 the SDK does not log model and tool payloads to its debug logger unless you opt in.

| Need | Setting |
|------|---------|
| Keep prompts, tool arguments and outputs out of traces | `new Runner({ traceIncludeSensitiveData: false })` |
| No tracing (e.g. Zero Data Retention organisations) | `setTracingDisabled(true)`, `new Runner({ tracingDisabled: true })` or `OPENAI_AGENTS_DISABLE_TRACING=1` |
| Group runs from one conversation | `new Runner({ workflowName, groupId, traceMetadata })`, or `withTrace(name, fn, { groupId })` |
| Export elsewhere as well / instead | `addTraceProcessor(processor)` / `setTraceProcessors([processor])` |
| Short-lived processes (serverless, jobs) | `await getGlobalTraceProvider().forceFlush()` before returning |
| Tracing key differs from the model key | `setTracingExportApiKey('sk-...')` |

```typescript
import { Agent, Runner, getGlobalTraceProvider, withTrace } from '@openai/agents';

const agent = new Agent({ name: 'Assistant', instructions: 'Be brief.', model: '%%model=openai%%' });

const runner = new Runner({ traceIncludeSensitiveData: false });

export async function handler(conversationId: string, text: string): Promise<string> {
  try {
    // One trace per request, grouped by conversation in the dashboard
    return await withTrace(
      'support-chat',
      async () => {
        const result = await runner.run(agent, text);
        return result.finalOutput ?? '';
      },
      { groupId: conversationId },
    );
  } finally {
    // In a serverless function, make sure spans leave before the instance freezes
    await getGlobalTraceProvider().forceFlush();
  }
}

console.log(await handler('conversation-123', 'Hello'));
```

Vendors such as Langfuse, Braintrust and Arize/OpenInference publish `TracingProcessor` implementations for this SDK; prefer those over generic HTTP instrumentation, which sees only the outer request.

## Usage and cost

`result.runContext.usage` holds the run's totals (`requests`, `inputTokens`, `outputTokens`, `totalTokens`, plus detail breakdowns). Since 0.21 detail values must be narrowed to numbers before arithmetic. Per model call, each entry of `result.rawResponses` has its own `usage`:

```typescript
import { Agent, run } from '@openai/agents';

const agent = new Agent({ name: 'Assistant', instructions: 'Be brief.', model: '%%model=openai%%' });
const result = await run(agent, 'Hello');

const { requests, inputTokens, outputTokens, totalTokens } = result.runContext.usage;
console.log({ requests, inputTokens, outputTokens, totalTokens });

for (const response of result.rawResponses) {
  // send to your metrics system, labelled by agent and model
  console.log('call', response.responseId, response.usage.totalTokens);
}
```

Route simple work to a smaller model (`model: '%%model=openai-small%%'`) and keep stable instructions and tool definitions at the start of the prompt so OpenAI's prompt caching applies.

## Testing without the API

`@openai/agents/testing` (since 0.16) provides `ScriptedModel`, a deterministic model that returns the steps you script and records each call. Tests run offline and can assert on the tools that ran and on what the model was sent. This example uses Node's built-in test runner (`node --test`):

```typescript
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Agent, retryPolicies, run, tool } from '@openai/agents';
import {
  ScriptedModel,
  assistantMessage,
  functionCall,
  modelError,
} from '@openai/agents/testing';
import { z } from 'zod';

const refunds: Array<[string, number]> = [];

const issueRefund = tool({
  name: 'issue_refund',
  description: 'Refund an order',
  parameters: z.object({ orderId: z.string(), amount: z.number() }),
  execute: async ({ orderId, amount }) => {
    refunds.push([orderId, amount]);
    return 'refunded';
  },
});

test('calls the refund tool with the scripted arguments', async () => {
  refunds.length = 0;
  const model = new ScriptedModel([
    [functionCall('issue_refund', { orderId: 'A-1', amount: 20 }, { callId: 'call_1' })],
    [assistantMessage('Your refund of 20.00 is on its way.')],
  ]);
  const agent = new Agent({ name: 'Refunds', instructions: 'Issue refunds.', tools: [issueRefund], model });

  const result = await run(agent, 'Refund 20 on A-1');

  assert.deepEqual(refunds, [['A-1', 20]]);
  assert.equal(result.finalOutput, 'Your refund of 20.00 is on its way.');
  assert.equal(model.calls.length, 2);
  assert.match(model.firstCall?.request.systemInstructions ?? '', /Issue refunds/);
  model.assertComplete();
});

test('retries a network error', async () => {
  const model = new ScriptedModel([
    modelError(Object.assign(new Error('socket hang up'), { code: 'ECONNRESET' })),
    [assistantMessage('ok')],
  ]);
  const agent = new Agent({
    name: 'Assistant',
    model,
    modelSettings: {
      retry: { maxRetries: 1, backoff: { initialDelayMs: 0 }, policy: retryPolicies.networkError() },
    },
  });
  const result = await run(agent, 'hi');
  assert.equal(result.finalOutput, 'ok');
  assert.equal(model.calls.length, 2);
});

test('an aborted run rejects', async () => {
  const agent = new Agent({ name: 'Slow', model: new ScriptedModel([[assistantMessage('late')]]) });
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(run(agent, 'hi', { signal: controller.signal }));
});
```

Other helpers: `modelResponder(fn)` builds the response from the recorded request (useful for structured output), `modelStream(events)` scripts a streamed response, and `scriptedSandboxSession` scripts sandbox calls. To swap the model for every agent without changing definitions, pass `new Runner({ model: scripted })` or `setDefaultModelProvider()`.

Keep a small set of live "smoke" tests against the real API for prompts whose wording matters, and run them separately from the offline suite.

## Checklist

- Pin the `@openai/agents*` packages to one exact version and set `model` on every agent.
- Set `maxTurns`, a run `signal`, `modelSettings.timeoutMs` / `retry` and tool `timeoutMs` for every production agent.
- Put `needsApproval` on tools with side effects; store paused `RunState` server-side.
- Key sessions by conversation, not by request.
- Decide what may appear in traces (`traceIncludeSensitiveData`) and where they go; flush in short-lived processes.
- Test with `ScriptedModel` in CI; no API key needed.

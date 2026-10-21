---
title: "Claude Agent SDK (TypeScript) — Production Guide"
description: "Running @anthropic-ai/claude-agent-sdk services in production: error handling, rate limits, cost, logging, containers, security and scaling."
framework: anthropic-claude-agent-sdk-typescript
---

# Claude Agent SDK (TypeScript) — Production Guide

Verified against `@anthropic-ai/claude-agent-sdk` %%version=@anthropic-ai/claude-agent-sdk%% (checked %%versions-checked%%).

Each `query()` starts a Claude Code process. Plan capacity in processes, give every run its own working directory, and treat every run as code execution on your host.

---

## Error handling

Three kinds of failure:

| Failure | Surfaces as |
|---|---|
| Loop ends unsuccessfully (API errors after retries, turn or budget limit, structured-output retries exhausted) | `result` message with `subtype` `'error_during_execution'`, `'error_max_turns'`, `'error_max_budget_usd'` or `'error_max_structured_output_retries'`, plus `errors` and `terminal_reason` |
| One model call fails but the loop continues | `assistant` message with an `error` field |
| Process cannot start, or the query is aborted | the async iterator throws (`AbortError` on abort) |

A wrapper that retries transient failures with jittered back-off and leaves limit errors alone:

```typescript
import { query, AbortError, type Options, type SDKResultMessage } from '@anthropic-ai/claude-agent-sdk';

const LIMIT_SUBTYPES = new Set(['error_max_turns', 'error_max_budget_usd']);

export async function runResilient(
  prompt: string,
  options: Options,
  { attempts = 4, timeoutMs = 300_000 } = {},
): Promise<SDKResultMessage> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      let result: SDKResultMessage | undefined;
      for await (const message of query({ prompt, options: { ...options, abortController: controller } })) {
        if (message.type === 'result') result = message;
      }
      if (!result) throw new Error('no result message');
      if (result.subtype === 'success' || LIMIT_SUBTYPES.has(result.subtype)) return result;
      lastError = new Error(`${result.subtype}: ${result.errors.join('; ')}`);
    } catch (err) {
      if (err instanceof AbortError) throw err;   // our own timeout: do not retry blindly
      lastError = err;
    } finally {
      clearTimeout(timer);
    }
    if (attempt < attempts) {
      await new Promise((r) => setTimeout(r, Math.min(30_000, 1000 * 2 ** attempt) + Math.random() * 1000));
    }
  }
  throw lastError;
}
```

Retrying repeats tool calls. For agents with side effects, resume the failed session (`resume: result.session_id`) rather than starting again, or make tools idempotent. `fallbackModel: 'claude-sonnet-5-5'` lets Claude Code switch model on overload instead of failing.

---

## Rate limits

Rate limits apply per organisation and per model. The stream tells you when you are close:

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';

for await (const message of query({ prompt: 'Summarise CHANGELOG.md', options: { allowedTools: ['Read'] } })) {
  if (message.type === 'rate_limit_event') {
    console.warn('rate limit state:', message.rate_limit_info.status, message.rate_limit_info.resetsAt);
  }
}
```

Bound concurrency per worker so a burst of requests does not become a burst of processes:

```typescript
export class Semaphore {
  private waiters: (() => void)[] = [];
  constructor(private available: number) {}

  async use<T>(fn: () => Promise<T>): Promise<T> {
    if (this.available === 0) await new Promise<void>((r) => this.waiters.push(r));
    else this.available--;
    try {
      return await fn();
    } finally {
      const next = this.waiters.shift();
      if (next) next();
      else this.available++;
    }
  }
}
```

---

## Cost control

Per-run limits:

```typescript
import type { Options } from '@anthropic-ai/claude-agent-sdk';

export const limits: Options = {
  maxTurns: 30,
  maxBudgetUsd: 2,
  taskBudget: { total: 200_000 },  // advisory: the model paces itself
};
```

Record what each run actually cost — `total_cost_usd` and `modelUsage` are on every result, so there is no need to maintain a price table:

```typescript
import type { SDKResultMessage } from '@anthropic-ai/claude-agent-sdk';

export function costRecord(result: SDKResultMessage, tenant: string) {
  return {
    tenant,
    session: result.session_id,
    ok: result.subtype === 'success',
    costUsd: result.total_cost_usd,
    turns: result.num_turns,
    byModel: Object.fromEntries(
      Object.entries(result.modelUsage).map(([model, u]) => [
        model,
        { costUsd: u.costUSD, input: u.inputTokens, output: u.outputTokens, cacheRead: u.cacheReadInputTokens },
      ]),
    ),
  };
}
```

Ways to spend less, roughly in order of impact:

1. Lower `effort` on a capable model before switching to a weaker one.
2. Run reading-heavy or routine subagents on `'sonnet'` or `'haiku'`.
3. Keep `systemPrompt` stable between runs so prompt caching works; never put timestamps or request IDs in it.
4. Give each agent only the tools it needs (`tools: []` for pure text tasks).
5. Cap `maxTurns`.

---

## Logging and monitoring

Log one structured line per tool call and per result:

```typescript
import { query, type Options } from '@anthropic-ai/claude-agent-sdk';

type Log = (event: string, fields: Record<string, unknown>) => void;
const log: Log = (event, fields) => console.log(JSON.stringify({ ts: new Date().toISOString(), event, ...fields }));

export async function runLogged(prompt: string, options: Options, requestId: string) {
  const started = Date.now();
  for await (const message of query({ prompt, options: { ...options, stderr: (line) => log('cli_stderr', { requestId, line }) } })) {
    switch (message.type) {
      case 'system':
        if (message.subtype === 'init') log('init', { requestId, session: message.session_id, model: message.model });
        break;
      case 'assistant':
        for (const block of message.message.content) {
          if (block.type === 'tool_use') log('tool_use', { requestId, tool: block.name, id: block.id, agent: message.agent_id });
        }
        break;
      case 'result':
        log('result', {
          requestId,
          subtype: message.subtype,
          terminalReason: message.terminal_reason,
          turns: message.num_turns,
          costUsd: message.total_cost_usd,
          durationMs: message.duration_ms,
          wallMs: Date.now() - started,
        });
        return message;
    }
  }
  return undefined;
}
```

For OpenTelemetry, turn on Claude Code's exporter through `env`:

```typescript
import type { Options } from '@anthropic-ai/claude-agent-sdk';

export const telemetry: Options = {
  env: {
    ...process.env,
    CLAUDE_CODE_ENABLE_TELEMETRY: '1',
    OTEL_METRICS_EXPORTER: 'otlp',
    OTEL_LOGS_EXPORTER: 'otlp',
    OTEL_EXPORTER_OTLP_ENDPOINT: 'http://otel-collector:4317',
  },
};
```

Note that `env` replaces the child's environment, so spread `process.env` if the process still needs `PATH`, `HOME` and your API key.

Alert on: result error rate by `subtype`; p95 cost and turns per run; rate-limit events; per-tool error rate. Hooks (`PreToolUse`, `PostToolUseFailure`) give an audit trail that includes subagent tool calls — see the [middleware page](/anthropic-claude-agent-sdk-typescript-guide/claude_agent_sdk_typescript_middleware/).

---

## Deployment

### Docker

The platform binary is installed as an optional npm dependency, so install on the same OS and architecture you run on (build inside the image, not on your laptop).

```dockerfile
FROM node:22-slim

RUN apt-get update && apt-get install -y --no-install-recommends git ripgrep ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home agent

WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev        # package.json pins @anthropic-ai/claude-agent-sdk
COPY --chown=agent:agent dist ./dist

USER agent
ENV HOME=/home/agent NODE_ENV=production
EXPOSE 8080
CMD ["node", "dist/server.js"]
```

Transcripts go to `$HOME/.claude/projects/`. Mount a volume there if sessions must survive restarts, set `persistSession: false` if you never resume, or mirror to shared storage with `sessionStore`.

### Kubernetes

- Size replicas by concurrent runs, not requests per second; budget a few hundred MB of memory per concurrent run.
- Mount `ANTHROPIC_API_KEY` from a `Secret`.
- Use a read-only root filesystem with writable `emptyDir` volumes for the work directory and `$HOME`.
- Set `terminationGracePeriodSeconds` above your longest expected run; on `SIGTERM` stop taking work and let in-flight runs finish or abort them.

---

## Environment and configuration

Validate configuration at start-up rather than on the first request:

```typescript
import { z } from 'zod';

const Env = z.object({
  ANTHROPIC_API_KEY: z.string().startsWith('sk-ant-'),
  AGENT_MODEL: z.string().default('claude-sonnet-5-5'),
  AGENT_MAX_BUDGET_USD: z.coerce.number().positive().default(2),
  AGENT_CONCURRENCY: z.coerce.number().int().positive().default(4),
});

export const config = Env.parse(process.env);
```

---

## Security hardening

```typescript
import type { Options } from '@anthropic-ai/claude-agent-sdk';

export function hardened(workDir: string): Options {
  return {
    cwd: workDir,
    settingSources: [],                          // ignore ~/.claude and .claude/ on the host
    tools: ['Read', 'Glob', 'Grep', 'Edit', 'Bash'],
    allowedTools: ['Read', 'Glob', 'Grep', 'Bash(npm test:*)'],
    disallowedTools: ['WebFetch', 'WebSearch', 'Read(./.env)'],
    permissionMode: 'dontAsk',                   // anything not pre-approved is refused
    sandbox: { enabled: true, network: { allowedDomains: [] } },
    verbatimPrompts: true,                       // untrusted text cannot trigger @file reads or slash commands
    maxTurns: 40,
    maxBudgetUsd: 3,
  };
}
```

- **Least privilege.** Start from an explicit `tools` list; pre-approve narrowly.
- **Permission mode.** `default` with `canUseTool`, or `dontAsk`. `bypassPermissions` only inside a disposable sandbox (it also needs `allowDangerouslySkipPermissions: true`).
- **Sandbox Bash** with network allow-lists.
- **Untrusted input.** Web pages, repository files and tool results can carry prompt injection. Use `verbatimPrompts` when inlining untrusted text, and enforce policy in hooks or `canUseTool`, not only in the system prompt.
- **Secrets.** Pass them via `env`, never in prompts; deny `Read` on credential files.
- **MCP trust.** `canUseTool` options and hook inputs carry the MCP server name and `source`, so you can trust in-process (`source === 'sdk'`) servers differently from third-party ones.

---

## Load balancing and high availability

- Put a queue in front of workers (BullMQ, SQS, Pub/Sub) and have each worker take only as many jobs as its semaphore allows.
- Make the HTTP layer stateless: return `session_id` to the client and resume with it. Any worker can resume when transcripts are on a shared volume or a `SessionStore`.
- Health checks should not start an agent. Check that the process is alive and the queue is reachable.

### Persisting results

Store the result record (from `costRecord` above) and the session ID with your domain object, so you can audit, bill and resume later:

```typescript
import type { SDKResultMessage } from '@anthropic-ai/claude-agent-sdk';

interface AgentRunRow {
  id: string;
  sessionId: string;
  status: 'success' | 'error';
  subtype: string;
  costUsd: number;
  turns: number;
  output: string | null;
  createdAt: Date;
}

export function toRow(id: string, result: SDKResultMessage): AgentRunRow {
  return {
    id,
    sessionId: result.session_id,
    status: result.subtype === 'success' ? 'success' : 'error',
    subtype: result.subtype,
    costUsd: result.total_cost_usd,
    turns: result.num_turns,
    output: result.subtype === 'success' ? result.result : null,
    createdAt: new Date(),
  };
}
```

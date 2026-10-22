---
title: "Mistral Agents API (TypeScript): Production"
description: "Errors, retries, timeouts, cancellation, request hooks, configuration, concurrency limits and an SSE streaming server for the Mistral Agents API with @mistralai/mistralai."
framework: "mistral-agents-api"
language: "typescript"
sidebar:
  label: "Production"
  order: 3
---

**Verified against `@mistralai/mistralai` %%version=@mistralai/mistralai%%** (checked %%versions-checked%%). Examples that import from `./common.js` use the module on the [agents and conversations page](/mistral-agents-api-guide/typescript/agents-and-conversations/).

The agent loop, built-in tools and conversation state run on Mistral's side, so a production service is mostly a careful API client. The operational advice in the [Python production guide](/mistral-agents-api-guide/mistral_agents_api_production_guide/) — storing conversation references, rate-limiting your own users, observability, security, CI/CD — applies unchanged; this page covers the TypeScript specifics.

---

## 1. Configuration and deployment

Validate configuration once at start-up:

```typescript
import { z } from "zod";

const Settings = z.object({
  AGENT_ID_SUPPORT: z.string().startsWith("ag_"),
  AGENT_VERSION_SUPPORT: z.string().default("prod"),
  MISTRAL_SERVER: z.enum(["eu", "us", "global"]).default("eu"),
  MISTRAL_TIMEOUT_MS: z.coerce.number().int().positive().default(60_000),
  MISTRAL_MAX_CONCURRENCY: z.coerce.number().int().positive().default(16),
});

export const settings = Settings.parse(process.env);
```

Credentials come from `MISTRAL_API_KEY`, or from a service-account token file named by `MISTRAL_SA_TOKEN_PATH`, which takes precedence and is re-read on every request so rotated tokens are picked up. An unreadable or empty token file fails the request with `UnexpectedClientError`.

**Create agents in a deploy step, not per request.** A script updates the agent, moves a version alias, and the service always starts conversations with `agentVersion: "prod"`. A rollback is moving the alias back.

```typescript
// deploy-agents.ts
import { readFile } from "node:fs/promises";
import { Mistral } from "@mistralai/mistralai";

const client = new Mistral({ apiKey: process.env["MISTRAL_API_KEY"] ?? "" });
const agentId = process.env["AGENT_ID_SUPPORT"] ?? "";

const agent = await client.beta.agents.update({
  agentId,
  updateAgentRequest: {
    instructions: await readFile("prompts/support.md", "utf8"),
    versionMessage: process.env["GIT_SHA"] ?? "manual",
  },
});
await client.beta.agents.createVersionAlias({ agentId: agent.id, alias: "prod", version: agent.version });
console.log("support agent now at version", agent.version);
```

---

## 2. Client, retries and timeouts

Create one client per process and reuse it. **Retries are off unless you configure them**: the default strategy is `"none"`. With a backoff strategy, the SDK retries on 429, 500, 502, 503 and 504 by default.

```typescript
import { Mistral } from "@mistralai/mistralai";

export const client = new Mistral({
  apiKey: process.env["MISTRAL_API_KEY"] ?? "",
  server: "eu", // "global" (default), "eu" or "us"
  timeoutMs: 60_000,
  retryConfig: {
    strategy: "backoff",
    backoff: { initialInterval: 500, maxInterval: 30_000, exponent: 1.8, maxElapsedTime: 120_000 },
    retryConnectionErrors: true,
  },
});
```

Intervals are in milliseconds. `serverURL` overrides the endpoint entirely (for a proxy, say).

Every method takes a second argument with per-call options: `timeoutMs`, `retries`, `retryCodes`, extra `headers`, and an `AbortSignal`:

```typescript
import { client } from "./common.js";

const controller = new AbortController();
setTimeout(() => controller.abort(), 10_000);

const response = await client.beta.conversations.start(
  { agentId: "ag_0123456789", inputs: "Summarise today's headlines." },
  {
    timeoutMs: 30_000,
    retries: { strategy: "backoff", backoff: { initialInterval: 1_000, maxInterval: 10_000, exponent: 2, maxElapsedTime: 60_000 } },
    retryCodes: ["429", "503"],
    headers: { "X-Request-Id": crypto.randomUUID() },
    signal: controller.signal,
  },
);
console.log(response.conversationId);
```

Be careful with retries on `start()` and `append()`: if a request reached Mistral but the response was lost, a retry runs the turn again (and, for `append()`, adds it to the conversation twice). Retrying on 429 and 503 is safe; retrying on timeouts is a judgement call.

---

## 3. Errors

```typescript
import * as errors from "@mistralai/mistralai/models/errors";
import { client } from "./common.js";

try {
  await client.beta.conversations.start({ agentId: "ag_missing", inputs: "Hello" });
} catch (err) {
  if (err instanceof errors.HTTPValidationError) {
    console.error("bad request:", err.data$.detail);
  } else if (err instanceof errors.MistralError) {
    // Any other HTTP error response
    if (err.statusCode === 429) console.error("rate limited; retry later");
    else if (err.statusCode === 404) console.error("no such agent");
    else throw err;
  } else if (err instanceof errors.SDKValidationError) {
    // The request (or response) did not match the SDK's Zod schema
    console.error(err.pretty());
  } else if (err instanceof errors.RequestTimeoutError || err instanceof errors.ConnectionError) {
    console.error("network problem:", err.message);
  } else {
    throw err;
  }
}
```

| Class | When |
|---|---|
| `MistralError` | Base class for HTTP error responses: `statusCode`, `body`, `headers`, `rawResponse` |
| `HTTPValidationError` | 422; structured details on `data$.detail` |
| `SDKError` | Other HTTP errors (subclass of `MistralError`) |
| `ResponseValidationError` | The response did not match the SDK's schema; `rawValue` and `pretty()` help diagnose |
| `SDKValidationError` | A request failed the SDK's Zod validation; nothing was sent |
| `ConnectionError`, `RequestTimeoutError`, `RequestAbortedError` | Network failure, timeout, or your `AbortSignal` |

Once a stream is open, errors the platform reports arrive as `conversation.response.error` events rather than exceptions, so handle them in the event loop too; network failures and aborts still throw.

### Errors as values

For code that prefers not to throw, or bundles that should include only the calls they use, every method also exists as a standalone function taking a `MistralCore` and returning a result object:

```typescript
import { MistralCore } from "@mistralai/mistralai/core.js";
import { betaConversationsStart } from "@mistralai/mistralai/funcs/betaConversationsStart.js";

const core = new MistralCore({ apiKey: process.env["MISTRAL_API_KEY"] ?? "" });

const result = await betaConversationsStart(core, { agentId: "ag_0123456789", inputs: "Hello" });
if (result.ok) {
  console.log(result.value.conversationId);
} else {
  console.error(result.error.name, result.error.message);
}
```

---

## 4. Logging and request hooks

`debugLogger: console` (or `MISTRAL_DEBUG=true`) logs every request and response, including headers, so keep it out of production. For metrics, pass your own `HTTPClient` with hooks:

```typescript
import { Mistral } from "@mistralai/mistralai";
import { HTTPClient } from "@mistralai/mistralai/lib/http.js";

const httpClient = new HTTPClient();
const started = new WeakMap<Request, number>();

httpClient.addHook("beforeRequest", (req) => {
  started.set(req, performance.now());
});
httpClient.addHook("response", (res, req) => {
  const ms = performance.now() - (started.get(req) ?? performance.now());
  console.log(`${req.method} ${new URL(req.url).pathname} ${res.status} ${ms.toFixed(0)}ms`);
});
httpClient.addHook("requestError", (err, req) => {
  console.error(`${req.method} ${req.url} failed`, err);
});

export const client = new Mistral({ apiKey: process.env["MISTRAL_API_KEY"] ?? "", httpClient });
```

The SDK has optional OpenTelemetry support as well; see the package README.

### Testing

`new HTTPClient({ fetcher })` takes a custom `fetch`, so tests can script the API's replies, for example a `function.call` followed by the final `message.output`, and assert on what your code sent:

```typescript
import assert from "node:assert/strict";
import { Mistral } from "@mistralai/mistralai";
import { HTTPClient } from "@mistralai/mistralai/lib/http.js";

const now = "2026-10-21T00:00:00Z";
const turns: object[][] = [
  [{ object: "entry", type: "function.call", id: "f1", created_at: now,
     tool_call_id: "tc1", name: "get_order_status", arguments: '{"orderId":"ORD-1234"}' }],
  [{ object: "entry", type: "message.output", id: "m1", created_at: now,
     role: "assistant", content: "ORD-1234 has shipped." }],
];
const sent: unknown[] = [];

// Every request gets the next scripted turn; request bodies are kept for assertions.
const httpClient = new HTTPClient({
  fetcher: async (input, init) => {
    const req = input instanceof Request ? input : new Request(input, init);
    sent.push(JSON.parse(await req.text()));
    return Response.json({
      object: "conversation.response",
      conversation_id: "conv_test",
      outputs: turns.shift(),
      usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
    });
  },
});
const client = new Mistral({ apiKey: "test", httpClient });

const first = await client.beta.conversations.start({ agentId: "ag_test", inputs: "Where is ORD-1234?" });
assert.equal(first.outputs.at(-1)?.type, "function.call");

const second = await client.beta.conversations.append({
  conversationId: first.conversationId,
  conversationAppendRequest: {
    inputs: [{ type: "function.result", toolCallId: "tc1", result: '{"status":"shipped"}' }],
  },
});
assert.equal(second.outputs.at(-1)?.type, "message.output");
assert.equal(sent.length, 2);
console.log("ok");
```

This is how the examples on these pages were exercised without calling the API. The [Python production guide](/mistral-agents-api-guide/mistral_agents_api_production_guide/#4-testing) does the same with `httpx2.MockTransport`.

---

## 5. Concurrency

Bound how many conversations your process runs at once, so a burst of users becomes queueing rather than a wall of 429s:

```typescript
import { client, textOf } from "./common.js";

/** Run at most `limit` tasks at once. */
function limiter(limit: number) {
  let active = 0;
  const queue: (() => void)[] = [];
  return async function run<T>(task: () => Promise<T>): Promise<T> {
    if (active < limit) active++;
    else await new Promise<void>((resolve) => queue.push(resolve)); // slot handed over below
    try {
      return await task();
    } finally {
      const next = queue.shift();
      if (next) next();
      else active--;
    }
  };
}

const limit = limiter(16);

export async function ask(agentId: string, inputs: string): Promise<string> {
  return limit(async () => textOf(await client.beta.conversations.start({ agentId, agentVersion: "prod", inputs })));
}

const answers = await Promise.all(["Question 1", "Question 2", "Question 3"].map((q) => ask("ag_0123456789", q)));
console.log(answers);
```

Libraries such as `p-limit` do the same job. Across several processes, rate-limit at your API gateway instead.

---

## 6. Streaming to a browser

A dependency-free Node server that relays a conversation stream as server-sent events, and stops the upstream request if the browser disconnects:

```typescript
import { createServer } from "node:http";
import { client } from "./common.js";

const AGENT_ID = process.env["AGENT_ID"] ?? "";

createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", "http://localhost");
  const message = url.searchParams.get("q");
  if (url.pathname !== "/chat" || !message) {
    res.writeHead(400).end("expected /chat?q=...");
    return;
  }
  const conversationId = url.searchParams.get("conversation");

  // Abort the upstream request if the browser goes away
  const controller = new AbortController();
  res.on("close", () => controller.abort());

  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });
  const send = (event: string, data: unknown) => res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

  try {
    const stream = conversationId
      ? await client.beta.conversations.appendStream(
          { conversationId, conversationAppendStreamRequest: { inputs: message } },
          { signal: controller.signal },
        )
      : await client.beta.conversations.startStream(
          { agentId: AGENT_ID, inputs: message },
          { signal: controller.signal },
        );

    for await (const { data } of stream) {
      if (data.type === "conversation.response.started") send("conversation", { id: data.conversationId });
      else if (data.type === "message.output.delta" && typeof data.content === "string") send("delta", data.content);
      else if (data.type === "tool.execution.started") send("tool", { name: data.name });
      else if (data.type === "conversation.response.error") send("error", { message: data.message });
      else if (data.type === "conversation.response.done") send("done", data.usage);
    }
  } catch (err) {
    if (!controller.signal.aborted) send("error", { message: err instanceof Error ? err.message : String(err) });
  } finally {
    res.end();
  }
}).listen(8000, () => console.log("listening on http://localhost:8000"));
```

The browser reads it with `new EventSource("/chat?q=...")`, keeps the `conversation` ID it receives, and sends it back on the next turn. In a real service, check that the conversation belongs to the signed-in user before appending to it, and put authentication in front of the endpoint. The same loop works in Express, Fastify or a framework route handler that returns a `ReadableStream`.

---

## 7. Checklist

- One client per process, with an explicit `retryConfig` and `timeoutMs`.
- Agents created and versioned by a deploy script; requests use a version alias.
- Concurrency bounded; 429s retried with backoff.
- Turn caps on every function-calling loop.
- `requiresConfirmation` on connector tools that write; your own approval step for function tools that do.
- Conversation IDs stored against your users, and ownership checked before `append()`.
- `debugLogger` off in production; API keys or service-account tokens from your secret store.

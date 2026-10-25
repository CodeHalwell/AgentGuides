---
title: "Mistral Agents API (TypeScript): Agents and Conversations"
description: "Creating and versioning agents, starting, continuing and branching conversations, streaming, structured output and guardrails with the @mistralai/mistralai TypeScript SDK."
framework: "mistral-agents-api"
language: "typescript"
sidebar:
  label: "Agents and conversations"
  order: 1
---

**Verified against `@mistralai/mistralai` %%version=@mistralai/mistralai%%** (checked %%versions-checked%%).

The examples share a small module with the client and a helper that extracts the answer text from a response:

```typescript
// common.ts
import { Mistral } from "@mistralai/mistralai";
import type { ConversationResponse } from "@mistralai/mistralai/models/components";

export const client = new Mistral({ apiKey: process.env["MISTRAL_API_KEY"] ?? "" });

/** Return the text of the last message in a conversation response. */
export function textOf(response: ConversationResponse): string {
  for (const entry of [...response.outputs].reverse()) {
    if (entry.type === "message.output") {
      if (typeof entry.content === "string") return entry.content;
      return entry.content.map((chunk) => (chunk.type === "text" ? chunk.text : "")).join("");
    }
  }
  return "";
}
```

`new Mistral()` resolves credentials in this order: an `Authorization` header passed per request, the `apiKey` option (a string or an async function returning one), the service-account token file named by `MISTRAL_SA_TOKEN_PATH` (re-read on every request, so rotation is picked up), then `MISTRAL_API_KEY`. The examples pass `apiKey` explicitly for clarity; because that outranks the token file, omit it (`new Mistral()`) when you rely on `MISTRAL_SA_TOKEN_PATH`. Pass `server: "eu"` or `server: "us"` for a regional endpoint.

Model names are the same as in Python: `%%model=mistral%%` (Mistral Medium 3.5) as the default for agentic work, `%%model=mistral-small%%` for routers and cheap agents, and dated names such as `mistral-medium-3-5` when you need to pin. See the [models table](/mistral-agents-api-guide/mistral_agents_api_comprehensive_guide/#3-models).

---

## 1. Agents

### Create

```typescript
import { client } from "./common.js";

const agent = await client.beta.agents.create({
  model: "%%model=mistral%%",
  name: "research-assistant",
  description: "Answers questions using current information from the web.",
  instructions: "Search before answering anything time-sensitive. Cite sources inline.",
  tools: [{ type: "web_search" }],
  completionArgs: { temperature: 0.3, topP: 0.95 },
  metadata: { team: "docs" },
});
console.log(agent.id, agent.version);
```

`create()` requires `model` and `name`; `instructions`, `tools`, `completionArgs`, `guardrails`, `description`, `handoffs`, `metadata` and `versionMessage` are optional. The result is an `Agent` with `id`, `version` and `versions`.

### Read, list, delete

```typescript
import { client } from "./common.js";

const agentId = "ag_0123456789";

const agent = await client.beta.agents.get({ agentId });
console.log(agent.name, agent.model, agent.version);

// Cursor-paginated listing: iterate over pages
for await (const page of await client.beta.agents.listPages({ pageSize: 50, name: "research-assistant" })) {
  for (const a of page.result.data) console.log(a.id, a.name, a.version);
}

await client.beta.agents.delete({ agentId });
```

`agents.list()` still exists but is marked deprecated in favour of `listPages()`.

### Versions and aliases

```typescript
import { client } from "./common.js";

const agentId = "ag_0123456789";

// Every update creates a new version. The changes go in updateAgentRequest.
const updated = await client.beta.agents.update({
  agentId,
  updateAgentRequest: {
    instructions: "Search before answering. Cite sources as [n] with a list at the end.",
    versionMessage: "Numbered citations",
  },
});
console.log("now at version", updated.version);

const versions = await client.beta.agents.listVersions({ agentId });
console.log(versions.map((v) => `${v.version}: ${v.versionMessage ?? ""}`));

await client.beta.agents.updateVersion({ agentId, version: 0 }); // roll back
await client.beta.agents.createVersionAlias({ agentId, alias: "prod", version: 1 });

// Start a conversation on a specific version or alias
const response = await client.beta.conversations.start({
  agentId,
  agentVersion: "prod",
  inputs: "What changed in EU AI rules this month?",
});
console.log(response.conversationId);
```

`getVersion()`, `listVersionAliases()` and `deleteVersionAlias()` complete the set.

---

## 2. Conversations

### Start

```typescript
import { client, textOf } from "./common.js";

const response = await client.beta.conversations.start({
  agentId: "ag_0123456789",
  inputs: "Who won the 2026 Tour de France?",
});
console.log(response.conversationId);
for (const entry of response.outputs) {
  console.log(entry.type); // e.g. "tool.execution", then "message.output"
}
console.log(textOf(response));

// Without a stored agent: pass model (and optionally instructions and tools)
const adHoc = await client.beta.conversations.start({
  model: "%%model=mistral-small%%",
  instructions: "Answer in one sentence.",
  inputs: [{ role: "user", content: "What is the capital of Slovenia?" }],
});
console.log(textOf(adHoc));
```

`inputs` is a string or an array of entries. `outputs` can hold `tool.execution`, `function.call`, `agent.handoff` and `message.output` entries; a message's `content` is either a string or an array of chunks (`text`, `tool_reference` for citations, `tool_file` for generated files, and others).

### Continue

```typescript
import { client, textOf } from "./common.js";

const conversationId = "conv_0123456789";

const response = await client.beta.conversations.append({
  conversationId,
  conversationAppendRequest: { inputs: "And who came second?" },
});
console.log(textOf(response));
```

`append()` continues with the same agent; there is no `agentId` on an append. To involve another agent, use [handoffs](/mistral-agents-api-guide/typescript/tools-and-orchestration/#handoffs) or start a new conversation with it.

### Restart from an earlier entry

```typescript
import { client, textOf } from "./common.js";

const conversationId = "conv_0123456789";

const history = await client.beta.conversations.getHistory({ conversationId });
const firstEntry = history.entries[0];
if (!firstEntry?.id) throw new Error("conversation has no entries");

const branch = await client.beta.conversations.restart({
  conversationId,
  conversationRestartRequest: {
    fromEntryId: firstEntry.id,
    inputs: "Answer again, but only for the women's race.",
  },
});
console.log(branch.conversationId, textOf(branch)); // a new conversation ID
```

`restart()` creates a new conversation that shares history up to `fromEntryId`; the original is left as it was.

### Read, list, delete and usage

```typescript
import { client } from "./common.js";

const conversationId = "conv_0123456789";

const conv = await client.beta.conversations.get({ conversationId });
const messages = await client.beta.conversations.getMessages({ conversationId });
const recent = await client.beta.conversations.list({ pageSize: 20, metadata: { team: "docs" } });
console.log(conv, messages.messages.length, recent.length);

await client.beta.conversations.delete({ conversationId });

// One-off request that is not persisted (cannot be appended to afterwards)
const ephemeral = await client.beta.conversations.start({
  model: "%%model=mistral-small%%",
  inputs: "Translate 'good morning' into Welsh.",
  store: false,
});
const { promptTokens, completionTokens, totalTokens, connectorTokens } = ephemeral.usage;
console.log(promptTokens, completionTokens, totalTokens, connectorTokens);
```

---

## 3. Streaming

`startStream()`, `appendStream()` and `restartStream()` return an async iterable of server-sent events. Each event's `data` is a union you can narrow on `type`:

```typescript
import { client } from "./common.js";

const stream = await client.beta.conversations.startStream({
  agentId: "ag_0123456789",
  inputs: "Summarise this week's ECB announcements.",
});

for await (const event of stream) {
  const data = event.data;
  switch (data.type) {
    case "message.output.delta":
      if (typeof data.content === "string") process.stdout.write(data.content);
      break;
    case "tool.execution.started":
      console.log(`\n[tool: ${data.name}]`);
      break;
    case "agent.handoff.started":
      console.log(`\n[handoff from ${data.previousAgentName}]`);
      break;
    case "conversation.response.done":
      console.log(`\n[${data.usage.totalTokens} tokens]`);
      break;
    case "conversation.response.error":
      console.error(`\n[error ${data.code}: ${data.message}]`);
      break;
  }
}
```

Event types: `conversation.response.started` (carries `conversationId`), `message.output.delta`, `tool.execution.started` / `.delta` / `.done`, `function.call.delta`, `agent.handoff.started` / `.done`, `conversation.response.done` and `conversation.response.error`. Unknown future event types are passed through rather than rejected, so keep a default branch harmless.

The append and restart variants nest their body in the same way as the non-streaming calls:

```typescript
import { client } from "./common.js";

const stream = await client.beta.conversations.appendStream({
  conversationId: "conv_0123456789",
  conversationAppendStreamRequest: { inputs: "Now in French, please." },
});
for await (const { data } of stream) {
  if (data.type === "message.output.delta" && typeof data.content === "string") {
    process.stdout.write(data.content);
  }
}
```

---

## 4. Structured output

Pass a JSON schema through `completionArgs.responseFormat`. The SDK ships a helper that turns a Zod object into one:

```typescript
import { z } from "zod";
import { responseFormatFromZodObject } from "@mistralai/mistralai/extra/structChat.js";
import { client, textOf } from "./common.js";

const Ticket = z.object({
  category: z.enum(["billing", "bug", "feature", "other"]),
  priority: z.number().int().min(1).max(5),
  summary: z.string(),
});

const classifier = await client.beta.agents.create({
  model: "%%model=mistral-small%%",
  name: "ticket-classifier",
  instructions: "Classify the support ticket.",
  completionArgs: { responseFormat: responseFormatFromZodObject(Ticket) },
});

const response = await client.beta.conversations.start({
  agentId: classifier.id,
  inputs: "The app crashes on login since today.",
});
const ticket = Ticket.parse(JSON.parse(textOf(response)));
console.log(ticket.category, ticket.priority);
```

`responseFormatFromZodObject` produces a strict `json_schema` format (`additionalProperties: false`, all properties required). Parse the answer with the same schema, as above, so a malformed reply fails loudly.

For a single call without an agent, `chat.parse()` takes the Zod schema directly and returns the parsed object on `choices[0].message.parsed`:

```typescript
import { z } from "zod";
import { client } from "./common.js";

const Ticket = z.object({ category: z.string(), priority: z.number(), summary: z.string() });

const result = await client.chat.parse({
  model: "%%model=mistral-small%%",
  messages: [{ role: "user", content: "Classify: the app crashes on login since today." }],
  responseFormat: Ticket,
});
console.log(result.choices?.[0]?.message?.parsed);
```

---

## 5. Guardrails

```typescript
import { client } from "./common.js";

const safe = await client.beta.agents.create({
  model: "%%model=mistral%%",
  name: "public-assistant",
  instructions: "You are a helpful assistant for the public website.",
  guardrails: [{ blockOnError: true, moderationLlmV2: {} }],
});

const response = await client.beta.conversations.start({ agentId: safe.id, inputs: "Hello!" });
console.log(response.guardrails);
```

An empty `moderationLlmV2` object uses the SDK's default, `mistral-moderation-2603` (Mistral Moderation 2); it also accepts `customCategoryThresholds`, `ignoreOtherCategories` and `action`. `moderationLlmV1` selects the retired `mistral-moderation-2411` and should not be used for new work. `guardrails` can also be passed to `conversations.start()` and `restart()` for a single conversation.

---

Next: [tools and orchestration](/mistral-agents-api-guide/typescript/tools-and-orchestration/).

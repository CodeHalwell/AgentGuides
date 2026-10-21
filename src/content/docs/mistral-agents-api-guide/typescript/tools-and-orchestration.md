---
title: "Mistral Agents API (TypeScript): Tools and Orchestration"
description: "Built-in tools, function calling with Zod, tool approvals, handoffs, manager-worker delegation and MCP connectors with the @mistralai/mistralai TypeScript SDK."
framework: "mistral-agents-api"
language: "typescript"
sidebar:
  label: "Tools and orchestration"
  order: 2
---

# Tools and orchestration

**Verified against `@mistralai/mistralai` %%version=@mistralai/mistralai%%** (checked %%versions-checked%%). The examples import `client` and `textOf()` from the `common.ts` module on the [agents and conversations page](/mistral-agents-api-guide/typescript/agents-and-conversations/).

---

## Built-in tools

These run on Mistral's side. Declare them in `tools` on an agent or on `conversations.start()`; what they did appears as `tool.execution` entries in `outputs`.

| `type` | What it does | Fields besides `type` |
|---|---|---|
| `web_search` | Web search with citations | `toolConfiguration` |
| `web_search_premium` | Web search plus news-agency sources | `toolConfiguration` |
| `code_interpreter` | Runs Python in a sandbox | `toolConfiguration` |
| `image_generation` | Generates images (returned as files) | `toolConfiguration` |
| `document_library` | Retrieval over your Libraries | `libraryIds`, `toolConfiguration` |
| `connector` | A registered MCP connector | `connectorId`, `authorization`, `toolConfiguration` |
| `function` | Your own function | `function` |

`toolConfiguration` has three optional string arrays: `include`, `exclude` and `requiresConfirmation`. There is no `mcp` or `document_retrieval` tool type, and no per-tool settings such as result counts or timeouts; the SDK's types reject them.

### Code interpreter

```typescript
import { writeFile } from "node:fs/promises";
import { client, textOf } from "./common.js";

const analyst = await client.beta.agents.create({
  model: "mistral-medium-latest",
  name: "data-analyst",
  description: "Writes and runs Python to answer quantitative questions.",
  instructions: "Use the code interpreter for any calculation. Show the code you ran and explain the result.",
  tools: [{ type: "code_interpreter" }],
});

const response = await client.beta.conversations.start({
  agentId: analyst.id,
  inputs: "What is the compound value of 10,000 EUR at 3.2% for 15 years? Plot it year by year.",
});

for (const entry of response.outputs) {
  if (entry.type === "tool.execution") console.log("ran", entry.name, entry.arguments.slice(0, 200));
}
console.log(textOf(response));

// Files the code produced (charts, CSVs) arrive as tool_file chunks
for (const entry of response.outputs) {
  if (entry.type !== "message.output" || typeof entry.content === "string") continue;
  for (const chunk of entry.content) {
    if (chunk.type !== "tool_file") continue;
    const stream = await client.files.download({ fileId: chunk.fileId });
    const bytes = new Uint8Array(await new Response(stream).arrayBuffer());
    await writeFile(chunk.fileName ?? `${chunk.fileId}.png`, bytes);
  }
}
```

`files.download()` returns a `ReadableStream<Uint8Array>`; wrapping it in a `Response` is the simplest way to collect it. The sandbox has no access to your network or files, so pass data in the prompt.

### Image generation

```typescript
import { client } from "./common.js";

const illustrator = await client.beta.agents.create({
  model: "mistral-medium-latest",
  name: "illustrator",
  description: "Creates illustrations from descriptions.",
  instructions: "Generate one image per request. Describe what you generated in one sentence.",
  tools: [{ type: "image_generation" }],
});

const response = await client.beta.conversations.start({
  agentId: illustrator.id,
  inputs: "A flat, two-colour illustration of a lighthouse at dusk, for a blog header.",
});
console.log(response.outputs.length);
```

The image arrives as a `tool_file` chunk; download it as in the code interpreter example.

### Web search with citations

```typescript
import { client } from "./common.js";

const researcher = await client.beta.agents.create({
  model: "mistral-medium-latest",
  name: "researcher",
  description: "Answers questions with current, cited information.",
  instructions: "Search before answering anything time-sensitive. Prefer primary sources.",
  tools: [{ type: "web_search" }], // or { type: "web_search_premium" } for news-agency sources
});

const response = await client.beta.conversations.start({
  agentId: researcher.id,
  inputs: "What did the ECB decide at its most recent rate meeting?",
});

for (const entry of response.outputs) {
  if (entry.type !== "message.output" || typeof entry.content === "string") continue;
  for (const chunk of entry.content) {
    if (chunk.type === "text") process.stdout.write(chunk.text);
    else if (chunk.type === "tool_reference") process.stdout.write(` [${chunk.title}](${chunk.url ?? ""})`);
  }
}
console.log();
```

### Document libraries

```typescript
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { client, textOf } from "./common.js";

const library = await client.beta.libraries.create({
  name: "product-handbook",
  description: "Product manuals and support policies",
});

const uploaded: string[] = [];
for (const name of (await readdir("handbook")).filter((n) => n.endsWith(".pdf"))) {
  const doc = await client.beta.libraries.documents.upload({
    libraryId: library.id,
    requestBody: { file: { fileName: name, content: await readFile(join("handbook", name)) } },
  });
  uploaded.push(doc.id);
}

// Processing is asynchronous: wait until every document is done (or failed)
for (const documentId of uploaded) {
  for (;;) {
    const { processStatus } = await client.beta.libraries.documents.status({ libraryId: library.id, documentId });
    if (processStatus === "done" || processStatus === "error") break;
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

const support = await client.beta.agents.create({
  model: "mistral-medium-latest",
  name: "handbook-support",
  description: "Answers questions from the product handbook.",
  instructions: "Answer only from the handbook. If it does not cover the question, say so.",
  tools: [{ type: "document_library", libraryIds: [library.id] }],
});

const response = await client.beta.conversations.start({
  agentId: support.id,
  inputs: "What is the refund window for annual plans?",
});
console.log(textOf(response));
```

`processStatus` can also be `todo`, `in_progress`, `waiting_for_capacity` and a few others. `documents.list()`, `textContent()`, `reprocess()` and `delete()` manage documents afterwards; `libraries.create()` also accepts `chunkSize`.

---

## Function calling

The TypeScript SDK has no `RunContext`, so you run the loop: declare the function, and while the response contains `function.call` entries, execute them and append `function.result` entries. A Zod schema gives you both the JSON schema for the model and validation of the arguments it sends.

```typescript
import { z } from "zod";
import type { FunctionCallEntry, FunctionResultEntry } from "@mistralai/mistralai/models/components";
import { client, textOf } from "./common.js";

// One Zod schema gives both the JSON schema the model sees and runtime validation of its arguments
const OrderStatusArgs = z.object({
  orderId: z.string().describe("The order ID, e.g. ORD-1234"),
});

async function getOrderStatus({ orderId }: z.infer<typeof OrderStatusArgs>) {
  return { orderId, status: "shipped", eta: "2026-10-24" };
}

const support = await client.beta.agents.create({
  model: "mistral-small-latest",
  name: "order-support",
  instructions: "Use get_order_status for any question about an order.",
  tools: [
    {
      type: "function",
      function: {
        name: "get_order_status",
        description: "Look up the status of an order by its ID.",
        parameters: z.toJSONSchema(OrderStatusArgs),
      },
    },
  ],
});

let response = await client.beta.conversations.start({
  agentId: support.id,
  inputs: "Where is ORD-1234?",
});

for (let turn = 0; turn < 5; turn++) {
  const calls = response.outputs.filter((e): e is FunctionCallEntry => e.type === "function.call");
  if (calls.length === 0) break;

  const results: FunctionResultEntry[] = [];
  for (const call of calls) {
    const raw = typeof call.arguments === "string" ? JSON.parse(call.arguments) : call.arguments;
    const parsed = OrderStatusArgs.safeParse(raw);
    const result = parsed.success
      ? await getOrderStatus(parsed.data)
      : { error: `invalid arguments: ${parsed.error.message}` };
    results.push({ type: "function.result", toolCallId: call.toolCallId, result: JSON.stringify(result) });
  }

  response = await client.beta.conversations.append({
    conversationId: response.conversationId,
    conversationAppendRequest: { inputs: results },
  });
}
console.log(textOf(response));
```

Notes:

- `z.toJSONSchema()` is Zod 4. With Zod 3, use the `zod-to-json-schema` package.
- `call.arguments` may be a JSON string or an already-parsed object; handle both.
- Returning a validation error as the result, rather than throwing, lets the model correct itself.
- The turn cap bounds cost if the model keeps calling tools.

### Approving tool calls

For connector tools, list the tools that need a person's approval in `toolConfiguration.requiresConfirmation`. Mistral then stops before running them and returns a `function.call` entry whose `confirmationStatus` is `"pending"`; reply with `toolConfirmations` on the next `append()`:

```typescript
import type { FunctionCallEntry, ToolCallConfirmation } from "@mistralai/mistralai/models/components";
import { client, textOf } from "./common.js";

const response = await client.beta.conversations.start({
  model: "mistral-medium-latest",
  inputs: "Add a comment to ticket T-42 saying the fix is deployed.",
  tools: [
    {
      type: "connector",
      connectorId: "tickets",
      toolConfiguration: { requiresConfirmation: ["post_comment"] },
    },
  ],
});

const pending = response.outputs.filter(
  (e): e is FunctionCallEntry => e.type === "function.call" && e.confirmationStatus === "pending",
);

if (pending.length > 0) {
  const toolConfirmations: ToolCallConfirmation[] = pending.map((call) => {
    console.log("needs approval:", call.name, call.arguments);
    const approved = true; // ask a person here
    return { toolCallId: call.toolCallId, confirmation: approved ? "allow" : "deny" };
  });

  const resumed = await client.beta.conversations.append({
    conversationId: response.conversationId,
    conversationAppendRequest: { toolConfirmations },
  });
  console.log(textOf(resumed));
} else {
  console.log(textOf(response));
}
```

This is the same mechanism the Python `RunContext` uses for server-side deferrals. Because the conversation is stored, the approval can arrive in a later request: persist `conversationId` and the pending `toolCallId`s. For your own function tools no flag is needed; your loop simply does not execute a call until it has been approved.

---

## Handoffs

An agent can pass the conversation to the agents listed in its `handoffs`:

```typescript
import { client, textOf } from "./common.js";

const billing = await client.beta.agents.create({
  model: "mistral-small-latest",
  name: "billing",
  description: "Invoices, refunds, payment methods.",
  instructions: "Resolve billing questions. Ask for the invoice number if needed.",
});
const technical = await client.beta.agents.create({
  model: "mistral-medium-latest",
  name: "technical",
  description: "Bugs, errors, configuration and integration problems.",
  instructions: "Troubleshoot step by step. Search the web for error messages you do not recognise.",
  tools: [{ type: "web_search" }],
});
const frontdesk = await client.beta.agents.create({
  model: "mistral-small-latest",
  name: "frontdesk",
  description: "First point of contact for customers.",
  instructions: "Work out what the customer needs and hand off to billing or technical. Do not answer yourself.",
  handoffs: [billing.id, technical.id],
});

// Specialists may hand back if the question changes
for (const specialist of [billing, technical]) {
  await client.beta.agents.update({
    agentId: specialist.id,
    updateAgentRequest: { handoffs: [frontdesk.id] },
  });
}

const response = await client.beta.conversations.start({
  agentId: frontdesk.id,
  inputs: "Since yesterday's update the export button returns a 500 error.",
  handoffExecution: "server", // the platform runs the target agent in the same request
});
for (const entry of response.outputs) {
  if (entry.type === "agent.handoff") console.log(`${entry.previousAgentName} -> ${entry.nextAgentName}`);
}
console.log(textOf(response));

// The conversation stays with whichever agent is now handling it
const followUp = await client.beta.conversations.append({
  conversationId: response.conversationId,
  conversationAppendRequest: { inputs: "It happens with CSV only, not with XLSX." },
});
console.log(textOf(followUp));
```

The `description` of each target is what the routing agent sees, so make it specific.

### Client-side handoffs

With `handoffExecution: "client"` the response stops at the `agent.handoff` entry, so your code can check or block the route before another agent acts:

```typescript
import { client, textOf } from "./common.js";

const frontdeskId = "ag_frontdesk";
const userIsVerified = (): boolean => false; // your own check

const request = "Please refund invoice INV-2207.";
const response = await client.beta.conversations.start({
  agentId: frontdeskId,
  inputs: request,
  handoffExecution: "client",
});

const last = response.outputs.at(-1);
if (last?.type === "agent.handoff") {
  console.log("model wants to route to", last.nextAgentName);
  if (last.nextAgentName === "billing" && !userIsVerified()) {
    console.log("ask the customer to verify before billing acts");
  } else {
    // Run the chosen specialist yourself, giving it the context it needs
    const answer = await client.beta.conversations.start({
      agentId: last.nextAgentId,
      inputs: `Customer request: ${request}`,
    });
    console.log(textOf(answer));
  }
} else {
  console.log(textOf(response));
}
```

Mistral documents `client` mode only as returning the handoff to the caller, not how to resume the same conversation afterwards, so running the target yourself (as above) is the dependable option. If an agent in a handoff graph has a function tool, the run pauses with a `function.call` entry whichever agent made it; answer it as in [function calling](#function-calling).

---

## Code-driven orchestration

### Parallel fan-out

When the steps are known in advance, plain `Promise.all` over separate conversations is simpler and cheaper than handoffs:

```typescript
import { client, textOf } from "./common.js";

async function ask(agentId: string, input: string): Promise<string> {
  return textOf(await client.beta.conversations.start({ agentId, inputs: input, store: false }));
}

const countries = ["France", "Germany", "Italy"];
const findings = await Promise.all(
  countries.map((c) => ask("ag_researcher", `Summarise current heat-pump subsidies in ${c}.`)),
);
const brief = await ask(
  "ag_writer",
  countries.map((c, i) => `## ${c}\n${findings[i]}`).join("\n\n") + "\n\nWrite a one-page comparison.",
);
console.log(brief);
```

### Manager-worker delegation

For open-ended tasks, give a manager agent a `delegate` function; your code runs each delegation against a worker agent, so you keep control of cost and of what each worker sees:

```typescript
import { z } from "zod";
import type { FunctionCallEntry, FunctionResultEntry } from "@mistralai/mistralai/models/components";
import { client, textOf } from "./common.js";

const WORKERS = {
  researcher: "ag_researcher",
  analyst: "ag_analyst",
  writer: "ag_writer",
} as const;

const DelegateArgs = z.object({
  worker: z.enum(["researcher", "analyst", "writer"]),
  task: z.string().describe("Complete instructions, including any context needed."),
});

async function ask(agentId: string, task: string): Promise<string> {
  return textOf(await client.beta.conversations.start({ agentId, inputs: task, store: false }));
}

const manager = await client.beta.agents.create({
  model: "mistral-medium-latest",
  name: "manager",
  description: "Plans work and delegates to specialists.",
  instructions:
    "Break the task into steps. Use delegate for each step, choosing the right worker. " +
    "When all steps are done, give the final answer.",
  tools: [
    {
      type: "function",
      function: {
        name: "delegate",
        description: "Give a self-contained task to a worker and get its result.",
        parameters: z.toJSONSchema(DelegateArgs),
      },
    },
  ],
});

async function runManager(goal: string, maxDelegations = 8): Promise<string> {
  let response = await client.beta.conversations.start({ agentId: manager.id, inputs: goal });
  for (let i = 0; i < maxDelegations; i++) {
    const calls = response.outputs.filter((e): e is FunctionCallEntry => e.type === "function.call");
    if (calls.length === 0) break;

    // Run the delegations in parallel
    const results: FunctionResultEntry[] = await Promise.all(
      calls.map(async (call) => {
        const raw = typeof call.arguments === "string" ? JSON.parse(call.arguments) : call.arguments;
        const args = DelegateArgs.parse(raw);
        const output = await ask(WORKERS[args.worker], args.task);
        return { type: "function.result" as const, toolCallId: call.toolCallId, result: output };
      }),
    );
    response = await client.beta.conversations.append({
      conversationId: response.conversationId,
      conversationAppendRequest: { inputs: results },
    });
  }
  return textOf(response);
}

console.log(await runManager("Prepare a one-page brief on heat-pump subsidies in France, Germany and Italy."));
```

More patterns (sequential pipelines, conditional routing, competitive evaluation, shared state) are in the [orchestration guide](/mistral-agents-api-guide/mistral_agents_api_orchestration_guide/); they translate directly.

---

## MCP

There are two ways to use an MCP server, and the TypeScript SDK supports the first fully and the second only by hand.

### Registered connectors

Register a remote MCP server once; any agent or conversation can then use its tools by name, and Mistral calls the server for you:

```typescript
import { client, textOf } from "./common.js";

await client.beta.connectors.create({
  protocol: "mcp",
  name: "deepwiki",
  description: "DeepWiki MCP server for exploring public code repositories",
  server: "https://mcp.deepwiki.com/mcp",
  visibility: "private",
});

const tools = await client.beta.connectors.listTools({ connectorIdOrName: "deepwiki" });
console.log(tools.length, "tools");

const response = await client.beta.conversations.start({
  model: "mistral-medium-latest",
  inputs: "Using deepwiki, explain how the sqlite/sqlite repository is organised.",
  tools: [
    {
      type: "connector",
      connectorId: "deepwiki",
      toolConfiguration: { include: ["read_wiki_structure", "ask_question"] },
    },
  ],
});
console.log(textOf(response));
```

For servers that need credentials, store them with Mistral, or send the user through an OAuth flow:

```typescript
import { client } from "./common.js";

await client.beta.connectors.createCredentials({
  connectorIdOrName: "internal-tickets",
  consumerScope: "workspace", // "user", "workspace" or "organization"
  credentialsCreateOrUpdate: {
    name: "default",
    isDefault: true,
    credentials: { headers: { Authorization: `Bearer ${process.env["TICKETS_TOKEN"] ?? ""}` } },
  },
});

const auth = await client.beta.connectors.getAuthUrl({
  connectorIdOrName: "gmail",
  appReturnUrl: "https://myapp.example.com/oauth/callback",
});
console.log("send the user to", auth.authUrl);
```

To call a registered connector's tools directly, without the model (useful in tests), `connectors.mcpClient()` returns an official `@modelcontextprotocol/sdk` client connected through Mistral's Connectors Gateway, using the Mistral client's key, base URL and timeout. `connectors.callTool()` also exists but is deprecated.

```typescript
import { client } from "./common.js";

// An official @modelcontextprotocol/sdk Client, connected through Mistral's Connectors Gateway.
// It reuses the Mistral client's API key, base URL and timeout.
const mcp = await client.beta.connectors.mcpClient("deepwiki");
try {
  const { tools } = await mcp.listTools();
  console.log(tools.map((t) => t.name));

  const result = await mcp.callTool({
    name: "read_wiki_structure",
    arguments: { repoName: "sqlite/sqlite" },
  });
  console.log(result.content);
} finally {
  await mcp.close();
}
```

Gateway failures (unknown connector, missing credentials, upstream unreachable) throw `ConnectorsGatewayError` from `@mistralai/mistralai/extra/connectors.js`; errors from the MCP server itself surface as the MCP SDK's own errors.

### Local or private MCP servers

The Python SDK can drive a local stdio server through `RunContext`; the TypeScript SDK cannot. If the server must stay inside your process or network, bridge it yourself with `@modelcontextprotocol/sdk` (`npm install @modelcontextprotocol/sdk`): advertise its tools as function tools and forward the calls.

```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import type { FunctionCallEntry, FunctionResultEntry, FunctionTool } from "@mistralai/mistralai/models/components";
import { client, textOf } from "./common.js";

// 1. Connect to a local MCP server over stdio
const mcp = new Client({ name: "mistral-bridge", version: "1.0.0" });
await mcp.connect(new StdioClientTransport({ command: "node", args: ["tickets-server.js"] }));

try {
  // 2. Advertise its tools to the model as function tools
  const { tools } = await mcp.listTools();
  const functionTools: FunctionTool[] = tools.map((t) => ({
    type: "function",
    function: { name: t.name, description: t.description ?? "", parameters: t.inputSchema },
  }));

  let response = await client.beta.conversations.start({
    model: "mistral-medium-latest",
    instructions: "Use the ticket tools to answer.",
    inputs: "Are there open tickets about login failures on Android?",
    tools: functionTools,
  });

  // 3. Execute the calls the model makes and send the results back
  for (let turn = 0; turn < 8; turn++) {
    const calls = response.outputs.filter((e): e is FunctionCallEntry => e.type === "function.call");
    if (calls.length === 0) break;
    const results: FunctionResultEntry[] = [];
    for (const call of calls) {
      const args = typeof call.arguments === "string" ? JSON.parse(call.arguments) : call.arguments;
      const result = await mcp.callTool({ name: call.name, arguments: args });
      results.push({ type: "function.result", toolCallId: call.toolCallId, result: JSON.stringify(result.content) });
    }
    response = await client.beta.conversations.append({
      conversationId: response.conversationId,
      conversationAppendRequest: { inputs: results },
    });
  }
  console.log(textOf(response));
} finally {
  await mcp.close();
}
```

Add an approval step before `callTool()` for tools that change anything. The [MCP guide](/mistral-agents-api-guide/mistral_agents_api_mcp_guide/) covers writing MCP servers and the security considerations, which apply unchanged.

---

## Practical advice

- **Tell the agent when to use each tool.** "Search before answering anything time-sensitive" is more reliable than leaving it to the model.
- **Give each agent few tools.** Several narrow agents joined by handoffs or code are easier to test than one with everything.
- **Log `tool.execution` and `function.call` entries.** They show what ran and with which arguments.
- **Treat tool output as untrusted input.** Web pages and MCP results can contain instructions; use `requiresConfirmation` for connector tools that write or send anything, and `include` to expose only the tools you need.
- **Watch usage.** `response.usage.connectorTokens` and `response.usage.connectors` report connector consumption separately from model tokens.

Next: [production](/mistral-agents-api-guide/typescript/production/).

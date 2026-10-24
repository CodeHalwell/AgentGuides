---
title: "Google ADK for TypeScript - Comprehensive Guide"
description: "Building agents with ADK for TypeScript: LlmAgent, Zod-typed tools, callbacks, sessions and state, memory, artifacts, workflows, human approval, plugins, streaming, MCP, A2A, custom models and testing."
framework: "google-adk"
language: "typescript"
sidebar:
  label: "Comprehensive guide"
  order: 10
---

**Verified against:** `@google/adk` %%version=@google/adk%% (checked %%versions-checked%%). Every example on this page passes `tsc --strict` and was run under Node.js 22 against a scripted stand-in for Gemini, so the calls, events and state changes are real but the model's words are not. The MCP example ran against a local MCP server, the A2A example against a local A2A server, and the database example against SQLite.

ADK for TypeScript versions independently of the Python and Go kits. Concepts carry across, but this page sticks to what the TypeScript package itself exports. For deployment, the dev tools and telemetry see the [production guide](/google-adk-guide/typescript/google_adk_typescript_production_guide/).

## Contents

1. [Installation and setup](#installation-and-setup)
2. [Your first agent](#your-first-agent)
3. [Agents](#agents)
4. [Models](#models)
5. [Tools](#tools)
6. [Callbacks](#callbacks)
7. [Multi-agent systems](#multi-agent-systems)
8. [Workflows](#workflows)
9. [Sessions and state](#sessions-and-state)
10. [Memory](#memory)
11. [Artifacts](#artifacts)
12. [Human approval for tool calls](#human-approval-for-tool-calls)
13. [Plugins](#plugins)
14. [Streaming](#streaming)
15. [MCP](#mcp)
16. [A2A](#a2a)
17. [Code execution and planners](#code-execution-and-planners)
18. [Testing](#testing)
19. [What TypeScript does not have](#what-typescript-does-not-have)

## Installation and setup

```bash
npm install @google/adk zod
npm install -D @google/adk-devtools typescript tsx
```

`zod` is how you describe tool parameters and structured output; the kit accepts Zod 3 or Zod 4 schemas (and plain `@google/genai` `Schema` objects). `@google/adk-devtools` provides the `adk` command: a terminal runner, the dev UI, an API server, evaluation and deployment. The examples here use top-level `await`, so set `"type": "module"` in `package.json` or name the files `.mts`.

Optional features need optional peer packages, installed only when you use them:

| Feature | Install |
|---|---|
| MCP toolsets | `@modelcontextprotocol/sdk` |
| `DatabaseSessionService` | one of `@mikro-orm/sqlite`, `@mikro-orm/postgresql`, `@mikro-orm/mysql`, `@mikro-orm/mariadb`, `@mikro-orm/mssql` |
| Serving an agent over A2A (`toA2a`) | `express` |
| `GcsArtifactService` | `@google-cloud/storage` |
| Cloud Trace and Cloud Monitoring export | `@google-cloud/opentelemetry-cloud-trace-exporter`, `@google-cloud/opentelemetry-cloud-monitoring-exporter` |

### Credentials

For the Gemini API, set one of `GOOGLE_GENAI_API_KEY`, `GOOGLE_API_KEY` or `GEMINI_API_KEY` (checked in that order). For Vertex AI, set `GOOGLE_GENAI_USE_ENTERPRISE=1` (the older `GOOGLE_GENAI_USE_VERTEXAI` still works), `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`, and authenticate with Application Default Credentials (`gcloud auth application-default login`, or the service account on Cloud Run and GKE).

A missing key or project is an error when the `Gemini` model is created, not a silent fallback: `new Gemini(...)` throws `API key must be provided via constructor or the GOOGLE_GENAI_API_KEY, GOOGLE_API_KEY or GEMINI_API_KEY environment variable.` The `adk` commands load a `.env` file for you; in your own scripts, export the variables or load them with `dotenv`.

## Your first agent

An agent is configuration; a runner executes it. `InMemoryRunner` bundles in-memory session, artifact and memory services, which is what you want for scripts and tests.

```typescript
import {FunctionTool, InMemoryRunner, LlmAgent, isFinalResponse, stringifyContent} from '@google/adk';
import {z} from 'zod';

const getWeather = new FunctionTool({
  name: 'get_weather',
  description: 'Returns the current weather for a city.',
  parameters: z.object({city: z.string().describe('City name')}),
  execute: async ({city}) => ({city, forecast: 'sunny', tempC: 21}),
});

export const rootAgent = new LlmAgent({
  name: 'assistant',
  model: '%%model=gemini%%',
  instruction: 'You are a helpful assistant. Use get_weather for weather questions.',
  tools: [getWeather],
});

const runner = new InMemoryRunner({agent: rootAgent, appName: 'demo'});

for await (const event of runner.runEphemeral({
  userId: 'u1',
  newMessage: {role: 'user', parts: [{text: 'What is the weather in Leeds?'}]},
})) {
  if (isFinalResponse(event)) {
    console.log(stringifyContent(event));
  }
}
```

What happens on one turn:

1. `runEphemeral` creates a throwaway session, appends the user message and starts an invocation. Use `runAsync` with a session you created to keep the conversation (see [Sessions and state](#sessions-and-state)).
2. The agent builds a request: its instruction, the conversation so far and the declarations of its tools.
3. The model asks for `get_weather`; ADK validates the arguments against the Zod schema, runs `execute`, and sends the result back to the model.
4. The model answers. Every step is yielded as an `Event`; `isFinalResponse(event)` marks the one to show the user, and `stringifyContent(event)` joins its text parts.

Exporting the agent as `rootAgent` lets the dev tools load the same file: `npx adk run agent.ts` for a terminal chat or `npx adk web` for the browser UI. `npx adk create my_agent` scaffolds a folder with an `agent.ts` and a `.env`. Install `@google/adk-devtools` in the project first; on a machine without it, `npx adk` downloads an unrelated npm package called `adk`, so name the package (`npx @google/adk-devtools run agent.ts`) when in doubt.

## Agents

`LlmAgent` takes a configuration object. The fields you will use most:

| Field | Purpose |
|---|---|
| `name` | Identifier, unique within the agent tree. Used for transfers and as the event author |
| `description` | What the agent does. Parents read it when deciding whether to transfer to it |
| `model` | A model name string or a `BaseLlm` instance (see [Models](#models)). Inherited from the parent when omitted |
| `instruction` | The system instruction: a template string or a function of the context |
| `globalInstruction` | Applied to every agent in the tree when set on the root |
| `tools` | Tools, toolsets and workflow nodes the model may call |
| `subAgents` | Agents the model may transfer the conversation to |
| `generateContentConfig` | `@google/genai` settings: `temperature`, `maxOutputTokens`, `safetySettings`, `thinkingConfig` |
| `outputSchema` | A Zod object (or genai `Schema`) the final reply must match |
| `outputKey` | Save the final reply to this session-state key |
| `inputSchema` | The argument schema when the agent is called as a tool or used as a workflow node |
| `includeContents` | `'none'` sends only the instruction and the current input, not the history |
| `disallowTransferToParent`, `disallowTransferToPeers` | Restrict where the model can transfer |
| `beforeAgentCallback`, `afterAgentCallback`, `beforeModelCallback`, `afterModelCallback`, `beforeToolCallback`, `afterToolCallback` | Hooks around each step (see [Callbacks](#callbacks)) |
| `codeExecutor`, `planner` | Code execution and planning (see [Code execution and planners](#code-execution-and-planners)) |
| `mode` | How the agent behaves as a workflow node: `'single_turn'` (default) or `'task'` |

Instructions, generation settings and structured output together:

```typescript
import {InMemoryRunner, LlmAgent, type ReadonlyContext} from '@google/adk';
import {z} from 'zod';

// 1. A templated instruction: {key} reads session state, {key?} is optional.
const writer = new LlmAgent({
  name: 'writer',
  model: '%%model=gemini%%',
  description: 'Writes product descriptions.',
  instruction: 'Write for a {user:audience?} audience. Current draft, if any: {draft?}',
  generateContentConfig: {temperature: 0.3, maxOutputTokens: 1024},
  outputKey: 'draft', // the final reply is saved to state["draft"]
});

// 2. An instruction built in code. No {placeholder} substitution happens here.
function supportInstruction(ctx: ReadonlyContext): string {
  const tier = ctx.state.get<string>('user:tier') ?? 'standard';
  return `You are a support agent. The customer is on the ${tier} plan.`;
}
const support = new LlmAgent({name: 'support', model: '%%model=gemini%%', instruction: supportInstruction});

// 3. Structured output: the reply must match the Zod schema.
const Triage = z.object({
  category: z.enum(['billing', 'technical', 'account']),
  urgency: z.number().int().min(1).max(5).describe('1 = low, 5 = critical'),
  summary: z.string(),
});
const triage = new LlmAgent({
  name: 'triage',
  model: '%%model=gemini-small%%',
  instruction: 'Classify the support ticket.',
  outputSchema: Triage,
  outputKey: 'triage',
});

const runner = new InMemoryRunner({agent: triage, appName: 'triage_app'});
const session = await runner.sessionService.createSession({appName: 'triage_app', userId: 'u1'});
for await (const _event of runner.runAsync({
  userId: 'u1',
  sessionId: session.id,
  newMessage: {role: 'user', parts: [{text: 'I was charged twice this month.'}]},
})) {
  // consume the run
}
const saved = await runner.sessionService.getSession({appName: 'triage_app', userId: 'u1', sessionId: session.id});
const result = Triage.parse(saved?.state['triage']);
console.log(result.category, result.urgency);
void writer;
void support;
```

- **Templates.** `{key}` in a string instruction is replaced with `state[key]`, and a missing key is an error; `{key?}` inserts nothing when the key is missing. `{user:name}` and `{app:name}` read the scoped state described under [Sessions and state](#sessions-and-state).
- **Instruction functions** receive a `ReadonlyContext` (state, user ID, session ID, agent name) and may be async. Their result is used as is, with no placeholder substitution, so literal braces are safe there.
- **Structured output.** With `outputSchema`, the model is asked for JSON matching the schema and `outputKey` stores the parsed object rather than the text. If the reply does not validate, ADK logs an error and still stores the parsed value, so validate it again where it matters, as `Triage.parse` does above. An agent can have both `outputSchema` and `tools`; for models that cannot combine the two in one request, ADK adds a `set_model_response` tool that the model calls with the final answer.
- **No default model.** If neither the agent nor any ancestor sets `model`, the first run throws `No model found for <name>`.
- **`includeContents: 'none'`** suits helper agents that should judge only the input they are given.

## Models

A model string is resolved through `LLMRegistry`. The package registers three providers: `Gemini` for `gemini-*` names (and Vertex AI endpoint paths), `ApigeeLlm` for `apigee/...` names routed through an Apigee proxy, and `ChromeBuiltInLlm` for Chrome's on-device model in the browser build. Pass a `Gemini` instance to set the connection explicitly:

```typescript
import {Gemini, LlmAgent} from '@google/adk';

// A model string is resolved by LLMRegistry: gemini-* names become Gemini instances.
const simple = new LlmAgent({name: 'simple', model: '%%model=gemini%%'});

// Configure the connection explicitly with a Gemini instance.
const onVertex = new LlmAgent({
  name: 'on_vertex',
  model: new Gemini({
    model: '%%model=gemini%%',
    vertexai: true,
    project: 'my-gcp-project',
    location: 'europe-west2',
  }),
});

const withKey = new Gemini({model: '%%model=gemini-small%%', apiKey: process.env.GEMINI_API_KEY});
void simple;
void onVertex;
void withKey;
```

There is no built-in client for Anthropic, OpenAI or other providers in the published 2.3.0 packages. A `Claude` class for Claude on Vertex AI exists in the `adk-js` repository's `@google/adk-integrations` package, but the 2.3.0 release of that package exports only its version number, so it is not usable from npm yet. To use another provider, subclass `BaseLlm` and register it. This minimal provider talks to any OpenAI-compatible chat endpoint (Ollama, vLLM and similar), text only:

```typescript
import {BaseLlm, LLMRegistry, LlmAgent, type BaseLlmConnection, type LlmRequest, type LlmResponse} from '@google/adk';

// A minimal provider for an OpenAI-compatible chat endpoint (text only, no tools).
export class ChatCompletionsLlm extends BaseLlm {
  static override readonly supportedModels = [/^local\/.+/];

  async *generateContentAsync(request: LlmRequest): AsyncGenerator<LlmResponse, void> {
    const system = String(request.config?.systemInstruction ?? '');
    const messages = [
      ...(system ? [{role: 'system', content: system}] : []),
      ...request.contents.map((c) => ({
        role: c.role === 'model' ? 'assistant' : 'user',
        content: (c.parts ?? []).map((p) => p.text ?? '').join(''),
      })),
    ];
    const res = await fetch(`${process.env.LOCAL_LLM_URL ?? 'http://localhost:11434'}/v1/chat/completions`, {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify({model: this.model.replace(/^local\//, ''), messages}),
    });
    if (!res.ok) {
      yield {errorCode: String(res.status), errorMessage: await res.text()};
      return;
    }
    const body = (await res.json()) as {choices: Array<{message: {content: string}}>};
    yield {content: {role: 'model', parts: [{text: body.choices[0].message.content}]}};
  }

  async connect(): Promise<BaseLlmConnection> {
    throw new Error('live sessions are not supported');
  }
}

LLMRegistry.register(ChatCompletionsLlm); // now model: 'local/<name>' resolves to it

export const localAgent = new LlmAgent({
  name: 'local_assistant',
  model: 'local/llama3.2',
  instruction: 'Answer briefly.',
});
```

`generateContentAsync` receives the whole `LlmRequest` (contents, `config.systemInstruction`, `config.tools` and the `toolsDict` of ADK tools) and yields `LlmResponse` objects; yield `errorCode` and `errorMessage` rather than throwing for provider errors you want surfaced as events. A production provider would also translate tool declarations and `functionCall` parts. `connect` is only used for live (bidirectional) sessions.

## Tools

### Function tools

`FunctionTool` wraps an async function. The Zod schema is turned into the declaration the model sees, the `describe()` text becomes each parameter's description, and the arguments are validated before `execute` runs, so `execute` receives typed input. The optional second argument is the `ToolContext`: session state (`state.get` / `state.set`), artifacts (`saveArtifact`, `loadArtifact`, `listArtifacts`), memory search (`searchMemory`), confirmation (`requestConfirmation`, `toolConfirmation`) and the event actions.

```typescript
import {FunctionTool, InMemoryRunner, LlmAgent, LongRunningFunctionTool, isFinalResponse, stringifyContent, type ToolContext} from '@google/adk';
import {z} from 'zod';

// A typed tool. The Zod schema becomes the declaration the model sees and
// validates the arguments before execute runs.
const getOrder = new FunctionTool({
  name: 'get_order',
  description: 'Look up an order by its reference.',
  parameters: z.object({orderId: z.string().describe('Order reference, such as A123')}),
  execute: async ({orderId}, toolContext?: ToolContext) => {
    if (!/^A\d+$/.test(orderId)) {
      // A thrown error is caught and returned to the model as {error: "..."}.
      throw new Error(`No order called ${orderId}. References look like A123.`);
    }
    toolContext?.state.set('last_order', orderId);
    return {orderId, status: 'shipped', etaDays: 2};
  },
});

// A long-running tool returns at once; the result arrives later as a function response.
const requestRefund = new LongRunningFunctionTool({
  name: 'request_refund',
  description: 'Start a refund. A person approves it later.',
  parameters: z.object({orderId: z.string(), reason: z.string()}),
  execute: async ({orderId}) => ({status: 'pending', ticket: `RF-${orderId}`}),
});

const agent = new LlmAgent({
  name: 'support',
  model: '%%model=gemini%%',
  instruction: 'Answer order questions with get_order. Use request_refund for refunds.',
  tools: [getOrder, requestRefund],
});

const runner = new InMemoryRunner({agent, appName: 'support_app'});
for await (const event of runner.runEphemeral({
  userId: 'u1',
  newMessage: {role: 'user', parts: [{text: 'Where is order A123?'}]},
})) {
  if (isFinalResponse(event)) console.log(stringifyContent(event));
}
```

- **Errors.** An exception thrown in `execute` does not end the run: ADK catches it and sends the model a response of `{error: "Error in tool 'get_order': ..."}`, so write messages the model can act on. Add `ReflectAndRetryToolPlugin` (see [Plugins](#plugins)) to turn failures into a structured retry prompt with a retry limit.
- **Return values.** Return a plain object; anything else (a string, a number) reaches the model as `{result: value}`.
- **State writes** from tools are recorded on the event and saved with the session when the event is appended.
- **Long-running tools.** `LongRunningFunctionTool` marks the call as long-running: its ID is listed in the event's `longRunningToolIds`, and the declaration tells the model not to call it again while it is pending. Return the pending status at once; when the job finishes, your application sends the final result as a `functionResponse` with the same call ID.

### Agents as tools and built-in tools

`AgentTool` wraps another agent so the caller can use it like a function and get its answer back, staying in control. `GOOGLE_SEARCH`, `URL_CONTEXT` and `GOOGLE_MAPS_GROUNDING` are Gemini's built-in tools: they run inside the model call and need no `execute`. `VertexAiSearchTool`, `EnterpriseWebSearchTool` and `VertexRagRetrievalTool` cover Vertex AI Search, enterprise web grounding and RAG corpora.

```typescript
import {AgentTool, FunctionTool, GOOGLE_SEARCH, LlmAgent, ToolContext} from '@google/adk';
import {z} from 'zod';

const addToBasket = new FunctionTool({
  name: 'add_to_basket',
  description: "Add an item to the customer's basket.",
  parameters: z.object({
    sku: z.string().describe('The product SKU'),
    quantity: z.number().int().min(1).describe('How many to add'),
  }),
  execute: async ({sku, quantity}, toolContext?: ToolContext) => {
    const basket = toolContext?.state.get<Record<string, number>>('basket', {}) ?? {};
    basket[sku] = (basket[sku] ?? 0) + quantity;
    toolContext?.state.set('basket', basket); // saved with the session
    return {basket};
  },
});

const researcher = new LlmAgent({
  name: 'researcher',
  model: '%%model=gemini%%',
  description: 'Searches the web and summarises what it finds.',
  instruction: 'Answer using Google Search. Cite sources.',
  tools: [GOOGLE_SEARCH],
});

export const shopAssistant = new LlmAgent({
  name: 'shop_assistant',
  model: '%%model=gemini%%',
  instruction: 'Help customers shop. Use the researcher for product questions.',
  tools: [addToBasket, new AgentTool({agent: researcher})],
});
```

Gemini does not accept every built-in tool alongside function tools in one request. Wrapping the searching agent in `AgentTool`, as here, is the dependable way to combine them. `AgentToolConfig` also accepts `skipSummarization`, but 2.3.0 stores the flag without acting on it.

Other ready-made tools in the package: `LOAD_MEMORY` and `PRELOAD_MEMORY` ([Memory](#memory)), `LoadArtifactsTool` ([Artifacts](#artifacts)), `EXIT_LOOP` (for the deprecated `LoopAgent`), `LOAD_WEB_PAGE`, `OpenAPIToolset` (tools from an OpenAPI spec), `APIHubToolset`, `ApplicationIntegrationToolset`, the Google API toolsets (`bigqueryToolSet`, `calendarToolSet`, `gmailToolSet` and others), `MCPToolset` ([MCP](#mcp)) and `SkillToolset` for skill folders.

### Toolsets

A toolset chooses its tools at request time. Subclass `BaseToolset` and return the tools the current user should see; `getTools` receives a `ReadonlyContext`, so the decision can depend on state:

```typescript
import {BaseToolset, FunctionTool, LlmAgent, type BaseTool, type ReadonlyContext} from '@google/adk';
import {z} from 'zod';

const refund = new FunctionTool({
  name: 'issue_refund',
  description: 'Refund an order in full.',
  parameters: z.object({orderId: z.string()}),
  execute: async ({orderId}) => ({orderId, status: 'refunded'}),
});
const closeAccount = new FunctionTool({
  name: 'close_account',
  description: "Close the customer's account.",
  parameters: z.object({reason: z.string()}),
  execute: async ({reason}) => ({closed: true, reason}),
});

// A toolset decides which tools to offer on each model call.
class StaffToolset extends BaseToolset {
  constructor() {
    super([]); // no name filter
  }
  async getTools(context?: ReadonlyContext): Promise<BaseTool[]> {
    const role = context?.state.get<string>('user:role');
    if (role === 'admin') return [refund, closeAccount];
    if (role === 'agent') return [refund];
    return [];
  }
  async close(): Promise<void> {}
}

export const staffAssistant = new LlmAgent({
  name: 'staff_assistant',
  model: '%%model=gemini%%',
  instruction: 'Help support staff with customer accounts.',
  tools: [new StaffToolset()],
});
```

The constructor's first argument is a tool filter (a list of tool names or a predicate) applied by toolsets that use `isToolSelected`; the second is an optional name prefix. `close()` is where a toolset releases connections; `MCPToolset` uses it to stop its server.

## Callbacks

Callbacks attach to one agent. Each receives an object with the context and the step's data; returning `undefined` lets the step proceed, returning a value replaces it.

| Callback | Receives | Return a value to |
|---|---|---|
| `beforeAgentCallback` | `Context` | Skip the agent and reply with this `Content` |
| `afterAgentCallback` | `Context` | Replace the agent's final reply |
| `beforeModelCallback` | `{context, request}` | Skip the model call and use this `LlmResponse` |
| `afterModelCallback` | `{context, response}` | Replace the model's response |
| `beforeToolCallback` | `{tool, args, context}` | Skip the tool and use this object as its result |
| `afterToolCallback` | `{tool, args, context, response}` | Replace the tool's result |

Each field also accepts an array of callbacks, run in order until one returns a value.

```typescript
import {FunctionTool, InMemoryRunner, LlmAgent, isFinalResponse, stringifyContent, type SingleBeforeModelCallback, type SingleAfterToolCallback} from '@google/adk';
import {z} from 'zod';

const BLOCKED = ['password', 'credit card number'];

// Runs before every model call. Returning a response skips the model.
const screenInput: SingleBeforeModelCallback = ({request}) => {
  const last = request.contents.at(-1);
  const text = (last?.parts ?? []).map((p) => p.text ?? '').join(' ').toLowerCase();
  if (BLOCKED.some((term) => text.includes(term))) {
    return {content: {role: 'model', parts: [{text: "I can't help with that."}]}};
  }
  return undefined; // carry on to the model
};

// Runs after every tool call. Returning an object replaces the tool's result.
const redact: SingleAfterToolCallback = ({tool, response}) => {
  if (tool.name !== 'lookup_customer') return undefined;
  const {internalId: _dropped, ...rest} = response;
  return rest;
};

const lookupCustomer = new FunctionTool({
  name: 'lookup_customer',
  description: 'Look up a customer by email address.',
  parameters: z.object({email: z.string().describe('The customer email')}),
  execute: async ({email}) => ({email, name: 'Sam Taylor', plan: 'pro', internalId: 'cus_9f2'}),
});

const agent = new LlmAgent({
  name: 'account_helper',
  model: '%%model=gemini%%',
  instruction: 'Help with account questions. Use lookup_customer to find the customer.',
  tools: [lookupCustomer],
  beforeModelCallback: screenInput,
  afterToolCallback: redact,
  // Agent callbacks receive the context; returning content ends the agent's turn with it.
  beforeAgentCallback: (ctx) =>
    ctx.state.get<boolean>('user:suspended')
      ? {role: 'model', parts: [{text: 'This account is suspended. Please contact support.'}]}
      : undefined,
});

const runner = new InMemoryRunner({agent, appName: 'accounts'});
for (const text of ['What is my password?', 'Look up sam@example.com']) {
  for await (const event of runner.runEphemeral({userId: 'u1', newMessage: {role: 'user', parts: [{text}]}})) {
    if (isFinalResponse(event)) console.log(stringifyContent(event));
  }
}
```

The first message is blocked before it reaches the model; on the second, `internalId` is removed before the model sees the tool's result. Use callbacks for checks that belong to one agent. For checks that should apply to every agent and tool in an app, write a [plugin](#plugins).

## Multi-agent systems

There are three ways to combine agents. Choose by who should be in control:

| Pattern | Mechanism | Control |
|---|---|---|
| Delegation | `subAgents: [...]` on an `LlmAgent` | The model transfers the conversation to a sub-agent, which then talks to the user |
| Agent as a tool | `tools: [new AgentTool({agent})]` | The caller stays in charge and receives the answer |
| Workflow | `new Workflow({edges: [...]})` | Your graph fixes the order; agents and functions are nodes |

Delegation, with each specialist's `description` telling the coordinator when to use it:

```typescript
import {LlmAgent} from '@google/adk';

const billing = new LlmAgent({
  name: 'billing',
  model: '%%model=gemini%%',
  description: 'Handles invoices, refunds and payment problems.',
  instruction: 'Resolve billing questions. Ask for the invoice number.',
});

const technical = new LlmAgent({
  name: 'technical',
  model: '%%model=gemini%%',
  description: 'Handles bugs, errors and how-to questions about the product.',
  instruction: 'Diagnose technical problems step by step.',
});

// The coordinator transfers the conversation to whichever specialist fits.
export const coordinator = new LlmAgent({
  name: 'coordinator',
  model: '%%model=gemini-small%%',
  instruction: 'Work out what the customer needs and transfer to the right specialist.',
  subAgents: [billing, technical],
});
```

- ADK adds a `transfer_to_agent` tool to agents with sub-agents. After a transfer, later turns go straight to the sub-agent until it transfers back.
- An agent can have only one parent: passing it to a second parent's `subAgents` throws `Agent "billing" already has a parent agent`.
- Set `disallowTransferToParent` or `disallowTransferToPeers` on a sub-agent to restrict where it can go.
- `SequentialAgent`, `ParallelAgent` and `LoopAgent` still work but log a deprecation warning in 2.x; the replacement is `Workflow`.

## Workflows

A `Workflow` is a graph of nodes. Agents, tools and plain functions can all be nodes, and `FunctionNode` names a function. Edges are arrays: a chain of nodes, with a nested array for a parallel fan-out and an object for routing. `Workflow` is marked experimental in 2.3.0 and logs a warning when constructed.

Each node receives the previous node's output as its input and produces an output of its own; an agent node's output is its final reply. The first node after `START` receives the user's message text. Events from nodes carry the output in `event.output`.

### Fan-out and join

```typescript
import {App, FunctionNode, InMemoryRunner, JoinNode, LlmAgent, START, Workflow} from '@google/adk';

const security = new LlmAgent({
  name: 'security',
  model: '%%model=gemini%%',
  instruction: 'List security issues in the code. Bullet points only.',
});
const style = new LlmAgent({
  name: 'style',
  model: '%%model=gemini-small%%',
  instruction: 'List readability issues in the code. Bullet points only.',
});

// A JoinNode passes {branchName: output} to the next node.
const report = new FunctionNode('report', (_ctx, input: Record<string, unknown>) =>
  Object.entries(input)
    .map(([name, text]) => `## ${name}\n${String(text)}`)
    .join('\n\n'),
);

const review = new Workflow({
  name: 'code_review',
  edges: [[START, [security, style], new JoinNode({name: 'join'}), report]],
});

const runner = new InMemoryRunner({app: new App({name: 'review_app', rootAgent: review})});

let result: unknown;
for await (const event of runner.runEphemeral({
  userId: 'u1',
  newMessage: {role: 'user', parts: [{text: "def f(q): return db.execute(f'SELECT {q}')"}]},
})) {
  if (event.output !== undefined) {
    result = event.output; // each node reports its output; the last one is the report
  }
}
console.log(result);
```

A `JoinNode` waits for all its predecessors and passes `{nodeName: output}` to the next node. `maxConcurrency` on the `Workflow` caps how many nodes run at once.

### Routing

A node chooses the next edge by emitting a route. A `FunctionNode` returns `createEvent({output, route})`; the routing object maps route values (strings, numbers or booleans) to the next node:

```typescript
import {App, FunctionNode, InMemoryRunner, LlmAgent, START, Workflow, createEvent} from '@google/adk';

// A function node routes by returning an event with a route.
const classify = new FunctionNode('classify', (_ctx, input: string) =>
  createEvent({output: input, route: /refund|invoice|charge/i.test(input) ? 'billing' : 'general'}),
);

const billing = new LlmAgent({
  name: 'billing',
  model: '%%model=gemini%%',
  instruction: 'Answer the billing question.',
});
const general = new LlmAgent({
  name: 'general',
  model: '%%model=gemini-small%%',
  instruction: 'Answer the general question briefly.',
});

// A routing map: the key is the route value, the value is the next node.
const triage = new Workflow({
  name: 'triage',
  edges: [[START, classify, {billing, general}]],
});

const runner = new InMemoryRunner({app: new App({name: 'triage_app', rootAgent: triage})});
for await (const event of runner.runEphemeral({
  userId: 'u1',
  newMessage: {role: 'user', parts: [{text: 'Why was I charged twice?'}]},
})) {
  if (event.output !== undefined) console.log(`${event.author}:`, event.output);
}
```

`DEFAULT_ROUTE` as a key catches any value with no matching entry.

### Loops, retries and timeouts

A loop is an edge back to an earlier node. Nodes take `retryConfig` and `timeout` (both in seconds), and node state writes are saved with the session:

```typescript
import {App, FunctionNode, InMemoryRunner, START, Workflow, createEvent} from '@google/adk';

const draft = new FunctionNode('draft', (ctx) => {
  const round = (ctx.state.get<number>('round') ?? 0) + 1;
  ctx.state.set('round', round); // node state writes are saved with the session
  return `draft v${round}`;
});

// Route back to draft until the check passes: a loop is just an edge to an earlier node.
const review = new FunctionNode('review', (ctx, input: string) =>
  createEvent({output: input, route: (ctx.state.get<number>('round') ?? 0) >= 3 ? 'approve' : 'revise'}),
);

let calls = 0;
const publish = new FunctionNode(
  'publish',
  (_ctx, input: string) => {
    calls += 1;
    if (calls === 1) throw new Error('CMS unavailable'); // fails once, then succeeds
    return `published ${input}`;
  },
  // Retries are per node. Delays are in seconds.
  {retryConfig: {maxAttempts: 3, initialDelay: 0.5, backoffFactor: 2}, timeout: 30},
);

const editorial = new Workflow({
  name: 'editorial',
  edges: [[START, draft, review, {revise: draft, approve: publish}]],
});

const runner = new InMemoryRunner({app: new App({name: 'editorial_app', rootAgent: editorial})});
for await (const event of runner.runEphemeral({userId: 'u1', newMessage: {role: 'user', parts: [{text: 'Write the launch post'}]}})) {
  if (event.output !== undefined) console.log(`${event.author}:`, event.output);
}
```

`retryConfig` accepts `maxAttempts`, `initialDelay`, `maxDelay`, `backoffFactor`, `jitter` and `exceptions` (error class names or classes to retry; all errors when omitted). A node that runs past `timeout` fails with `NodeTimeoutError`.

### Pausing for human input

A node pauses the workflow by yielding a `RequestInput`. The run ends with an `adk_request_input` function call; the caller answers it later with a function response carrying the same ID, and the workflow resumes from that node:

```typescript
import {App, FunctionNode, InMemoryRunner, REQUEST_INPUT_FUNCTION_CALL_NAME, RequestInput, START, Workflow, getFunctionCalls} from '@google/adk';

const prepare = new FunctionNode('prepare', (_ctx, input: string) => ({request: input, amount: 250}));

// Yielding a RequestInput pauses the workflow until the caller answers it.
const approve = new FunctionNode(
  'approve',
  async function* (ctx, input: {request: string; amount: number}) {
    const decision = ctx.resumeInputs['manager_approval'] as {approved?: boolean} | undefined;
    if (decision === undefined) {
      yield new RequestInput({
        interruptId: 'manager_approval',
        message: `Approve a refund of ${input.amount}?`,
        payload: input,
      });
      return;
    }
    yield decision.approved ? `refund of ${input.amount} approved` : 'refund rejected';
  },
  {rerunOnResume: true}, // run the node again, with resumeInputs filled in
);

const refunds = new Workflow({name: 'refunds', edges: [[START, prepare, approve]]});
const runner = new InMemoryRunner({app: new App({name: 'refund_app', rootAgent: refunds})});
const session = await runner.sessionService.createSession({appName: 'refund_app', userId: 'u1'});

let interruptId: string | undefined;
for await (const event of runner.runAsync({
  userId: 'u1',
  sessionId: session.id,
  newMessage: {role: 'user', parts: [{text: 'Refund order A1'}]},
})) {
  for (const call of getFunctionCalls(event)) {
    if (call.name === REQUEST_INPUT_FUNCTION_CALL_NAME) {
      interruptId = call.id;
      console.log('paused:', call.args?.['message']);
    }
  }
}

// Later (another request, even another process with a persistent session service):
for await (const event of runner.runAsync({
  userId: 'u1',
  sessionId: session.id,
  newMessage: {
    role: 'user',
    parts: [{functionResponse: {id: interruptId, name: REQUEST_INPUT_FUNCTION_CALL_NAME, response: {approved: true}}}],
  },
})) {
  if (event.output !== undefined) console.log(`${event.author}:`, event.output);
}
```

With `rerunOnResume: true` the node runs again and finds the answer in `ctx.resumeInputs[interruptId]`. Without it (the default), the answer itself becomes the node's output and the workflow moves on to the next node. Pausing only works with `runAsync` on a stored session, because the resumed run reads the paused one from the session's events; use a persistent session service if the answer can arrive after a restart.

Two limits in 2.3.0: a `Workflow` cannot be an `LlmAgent` sub-agent (use it as the root of a `Runner` or `App`, or as a node in another workflow), and agent nodes in `mode: 'task'` get a `finish_task` tool and keep going until they call it, which is how you get multi-turn work out of a node.

## Sessions and state

A session holds one conversation: its events and its state. `Runner` takes the services explicitly; `InMemoryRunner` is a `Runner` with in-memory services built in.

```typescript
import {InMemorySessionService, LlmAgent, Runner, isFinalResponse, stringifyContent} from '@google/adk';

const agent = new LlmAgent({
  name: 'assistant',
  model: '%%model=gemini%%',
  instruction: 'You are a helpful assistant. The customer is called {user:name?}.',
});

const sessionService = new InMemorySessionService();
const runner = new Runner({appName: 'support', agent, sessionService});

// Sessions must exist before runAsync, unless the runner has autoCreateSession: true.
const session = await sessionService.createSession({
  appName: 'support',
  userId: 'u1',
  state: {'user:name': 'Sam'}, // the user: prefix shares this value across the user's sessions
});

for (const text of ['My order number is A123.', 'What was my order number?']) {
  for await (const event of runner.runAsync({
    userId: 'u1',
    sessionId: session.id,
    newMessage: {role: 'user', parts: [{text}]},
  })) {
    if (isFinalResponse(event)) {
      console.log(stringifyContent(event));
    }
  }
}

const saved = await sessionService.getSession({appName: 'support', userId: 'u1', sessionId: session.id});
console.log(saved?.events.length, 'events stored');
```

State keys are scoped by prefix:

| Prefix | Scope |
|---|---|
| none | This session |
| `user:` | Every session of this user in this app |
| `app:` | Every session of every user in this app |
| `temp:` | The current invocation only; never stored |

Write state from tools and callbacks (`context.state.set(key, value)`), with `outputKey`, or with `stateDelta` on `runAsync`. Never put secrets in state: it is stored with the session and can reach the model through instruction templates.

`runAsync` throws `Session not found: <id>` for an unknown session unless the runner was built with `autoCreateSession: true`. For storage that survives a restart, use `DatabaseSessionService`:

```typescript
import {DatabaseSessionService, LlmAgent, Runner, isFinalResponse, stringifyContent} from '@google/adk';

// npm install @mikro-orm/sqlite  (or @mikro-orm/postgresql, /mysql, /mariadb, /mssql)
// Tables are created on first use.
const sessionService = new DatabaseSessionService('sqlite://sessions.db');

const agent = new LlmAgent({
  name: 'assistant',
  model: '%%model=gemini%%',
  instruction: 'You are a helpful assistant.',
});

const runner = new Runner({appName: 'support', agent, sessionService, autoCreateSession: true});

// With autoCreateSession, an unknown sessionId is created on first use.
for await (const event of runner.runAsync({
  userId: 'u1',
  sessionId: 'order-A123',
  newMessage: {role: 'user', parts: [{text: 'My order A123 has not arrived.'}]},
  stateDelta: {'user:tier': 'gold'}, // applied with the user message
})) {
  if (isFinalResponse(event)) console.log(stringifyContent(event));
}

const {sessions} = await sessionService.listSessions({appName: 'support', userId: 'u1'});
console.log(sessions.map((s) => s.id));
const session = await sessionService.getSession({
  appName: 'support',
  userId: 'u1',
  sessionId: 'order-A123',
  config: {numRecentEvents: 10}, // load only the last ten events
});
console.log(session?.state['user:tier'], session?.events.length);
```

`DatabaseSessionService` takes a connection string (`sqlite://path`, `sqlite://:memory:`, `postgresql://...`, `mysql://...`, `mariadb://...` or `mssql://...`) or MikroORM options, and creates its tables on first use. `VertexAiSessionService` stores sessions in Vertex AI Agent Engine, and `getSessionServiceFromUri` picks a service from a URI (`memory://`, a database URI, or `vertexai://`, which reads `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` and `GOOGLE_CLOUD_AGENT_ENGINE_ID`). The dev tools use it for `--session_service_uri`.

## Memory

Session state lives with one conversation. Memory is searchable knowledge across a user's conversations: you add finished sessions to a memory service, and agents search it.

```typescript
import {InMemoryMemoryService, InMemorySessionService, LlmAgent, LOAD_MEMORY, Runner} from '@google/adk';

const sessionService = new InMemorySessionService();
const memoryService = new InMemoryMemoryService();

const agent = new LlmAgent({
  name: 'assistant',
  model: '%%model=gemini%%',
  instruction: 'Use load_memory when the user refers to earlier conversations.',
  tools: [LOAD_MEMORY], // or PRELOAD_MEMORY to search on every turn
});

const runner = new Runner({appName: 'mem_app', agent, sessionService, memoryService, autoCreateSession: true});

// Conversation 1.
for await (const _ of runner.runAsync({
  userId: 'u1',
  sessionId: 'monday',
  newMessage: {role: 'user', parts: [{text: 'My favourite colour is green.'}]},
})) {
  // consume
}

// When a conversation ends, copy it into memory.
const monday = await sessionService.getSession({appName: 'mem_app', userId: 'u1', sessionId: 'monday'});
if (monday) await memoryService.addSessionToMemory(monday);

// Any later session of the same user can search it.
const found = await memoryService.searchMemory({appName: 'mem_app', userId: 'u1', query: 'favourite colour'});
console.log(found.memories.length, 'memories found');
```

- `LOAD_MEMORY` lets the model search when it decides to; `PRELOAD_MEMORY` searches on every turn with the user's message and adds the results to the request.
- Tools and callbacks can search directly with `context.searchMemory(query)`.
- `InMemoryMemoryService` matches on words and keeps nothing after the process exits. `VertexAiMemoryBankService` uses Vertex AI Memory Bank, which extracts and consolidates memories, and `VertexAiRagMemoryService` stores sessions in a Vertex AI RAG corpus.

## Artifacts

Artifacts are versioned files (reports, images, uploads) stored per session by an artifact service. Tools save and load them through the context:

```typescript
import {FunctionTool, InMemoryArtifactService, InMemorySessionService, LlmAgent, Runner, isFinalResponse, stringifyContent} from '@google/adk';
import {z} from 'zod';

const saveReport = new FunctionTool({
  name: 'save_report',
  description: 'Save a report as a Markdown file for the user to download.',
  parameters: z.object({title: z.string(), body: z.string()}),
  execute: async ({title, body}, toolContext) => {
    if (!toolContext) throw new Error('no tool context');
    const filename = `${title.toLowerCase().replace(/\W+/g, '-')}.md`;
    // Each save creates a new version, numbered from 0.
    const version = await toolContext.saveArtifact(filename, {
      inlineData: {mimeType: 'text/markdown', data: Buffer.from(`# ${title}\n\n${body}`).toString('base64')},
    });
    return {filename, version};
  },
});

const artifactService = new InMemoryArtifactService(); // GcsArtifactService / FileArtifactService in production
const sessionService = new InMemorySessionService();
const agent = new LlmAgent({
  name: 'reporter',
  model: '%%model=gemini%%',
  instruction: 'Write the report the user asks for and save it with save_report.',
  tools: [saveReport],
});
const runner = new Runner({appName: 'reports', agent, sessionService, artifactService, autoCreateSession: true});

for await (const event of runner.runAsync({
  userId: 'u1',
  sessionId: 's1',
  newMessage: {role: 'user', parts: [{text: 'Write a short status report.'}]},
})) {
  if (isFinalResponse(event)) console.log(stringifyContent(event));
}

// Outside the agent, read artifacts through the service.
const key = {appName: 'reports', userId: 'u1', sessionId: 's1'};
for (const filename of await artifactService.listArtifactKeys(key)) {
  const part = await artifactService.loadArtifact({...key, filename});
  const text = Buffer.from(part?.inlineData?.data ?? '', 'base64').toString();
  console.log(filename, '->', text.split('\n')[0]);
}
```

- Each save of a filename creates a new version, starting at 0; `loadArtifact(filename)` returns the latest, `loadArtifact(filename, version)` a specific one.
- A filename starting with `user:` is shared across all of the user's sessions.
- `InMemoryArtifactService` is for tests; `FileArtifactService` writes to a local directory; `GcsArtifactService` (also importable from `@google/adk/artifacts/gcs`) uses a Cloud Storage bucket.
- `new LoadArtifactsTool()` lets the model list and open artifacts itself.

## Human approval for tool calls

Mark a tool with `requireConfirmation` and ADK pauses before running it. The run ends with an `adk_request_confirmation` function call that carries the original call; your application shows it to a person and replies with a function response to that call's ID:

```typescript
import {
  FunctionTool,
  InMemoryRunner,
  LlmAgent,
  REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
  getFunctionCalls,
  isFinalResponse,
  stringifyContent,
  type Event,
} from '@google/adk';
import {z} from 'zod';

const issueRefund = new FunctionTool({
  name: 'issue_refund',
  description: 'Refund an order.',
  parameters: z.object({orderId: z.string(), amount: z.number()}),
  // true asks every time; a function decides per call.
  requireConfirmation: ({amount}) => amount > 100,
  execute: async ({orderId, amount}) => ({status: 'refunded', orderId, amount}),
});

const agent = new LlmAgent({
  name: 'support',
  model: '%%model=gemini%%',
  instruction: 'Issue refunds when the customer asks for one.',
  tools: [issueRefund],
});

const runner = new InMemoryRunner({agent, appName: 'shop'});
const session = await runner.sessionService.createSession({appName: 'shop', userId: 'u1'});

async function turn(message: Parameters<typeof runner.runAsync>[0]['newMessage']): Promise<Event[]> {
  const events: Event[] = [];
  for await (const event of runner.runAsync({userId: 'u1', sessionId: session.id, newMessage: message})) {
    events.push(event);
    if (isFinalResponse(event) && event.content?.parts?.some((p) => p.text)) {
      console.log('agent:', stringifyContent(event));
    }
  }
  return events;
}

// Turn 1: the model calls issue_refund; ADK pauses and asks for confirmation instead.
const events = await turn({role: 'user', parts: [{text: 'Please refund order A1, it was 250 pounds.'}]});
const request = events
  .flatMap((e) => getFunctionCalls(e))
  .find((call) => call.name === REQUEST_CONFIRMATION_FUNCTION_CALL_NAME);

if (request?.id) {
  console.log('approval needed for', JSON.stringify(request.args?.['originalFunctionCall']));
  // Turn 2: answer the request. confirmed: false rejects the call instead.
  await turn({
    role: 'user',
    parts: [{functionResponse: {id: request.id, name: REQUEST_CONFIRMATION_FUNCTION_CALL_NAME, response: {confirmed: true}}}],
  });
}
```

- `requireConfirmation: true` asks every time; a function (sync or async) decides per call from the validated arguments.
- Reply with `{confirmed: false}` to reject: the tool does not run and the model is told it was rejected.
- Inside a tool, `toolContext.requestConfirmation({hint, payload})` asks for approval from your own logic, and `toolContext.toolConfirmation` holds the answer when the tool runs again.
- For one policy across all tools, use `SecurityPlugin` with a policy engine that returns allow, confirm or deny; see the [production guide](/google-adk-guide/typescript/google_adk_typescript_production_guide/#tool-policies).

## Plugins

A plugin is a class extending `BasePlugin`, registered on the `Runner` (or `App`) with `plugins: [...]`. It receives callbacks for every agent, model call and tool call in the runner, plus run-level hooks: `onUserMessageCallback`, `beforeRunCallback`, `onEventCallback`, `afterRunCallback`, `beforeAgentCallback`, `afterAgentCallback`, `beforeModelCallback`, `afterModelCallback`, `onModelErrorCallback`, `beforeToolCallback`, `afterToolCallback`, `onToolErrorCallback`, and for workflows `beforeNodeCallback` and `afterNodeCallback`.

```typescript
import {
  BasePlugin,
  InMemoryRunner,
  LlmAgent,
  LoggingPlugin,
  ReflectAndRetryToolPlugin,
  type BaseTool,
  type Context,
  type LlmResponse,
} from '@google/adk';

// A plugin sees every agent, model call and tool call in the runner.
class UsagePlugin extends BasePlugin {
  private tokens = 0;
  private toolCalls = new Map<string, number>();

  constructor() {
    super('usage');
  }

  override async afterModelCallback({llmResponse}: {callbackContext: Context; llmResponse: LlmResponse}) {
    this.tokens += llmResponse.usageMetadata?.totalTokenCount ?? 0;
    return undefined; // keep the response unchanged
  }

  override async beforeToolCallback({tool}: {tool: BaseTool; toolArgs: Record<string, unknown>; toolContext: Context}) {
    this.toolCalls.set(tool.name, (this.toolCalls.get(tool.name) ?? 0) + 1);
    return undefined; // returning an object would skip the tool and use it as the result
  }

  report() {
    return {tokens: this.tokens, toolCalls: Object.fromEntries(this.toolCalls)};
  }
}

const usage = new UsagePlugin();
const agent = new LlmAgent({name: 'assistant', model: '%%model=gemini%%', instruction: 'Be helpful.'});

const runner = new InMemoryRunner({
  agent,
  appName: 'demo',
  plugins: [
    usage,
    new ReflectAndRetryToolPlugin({maxRetries: 2}), // turns tool errors into a retry prompt
    new LoggingPlugin(), // logs every callback; for development
  ],
});

for await (const _ of runner.runEphemeral({userId: 'u1', newMessage: {role: 'user', parts: [{text: 'Hello'}]}})) {
  // consume
}
console.log(usage.report());
```

Plugin callbacks run before the agent's own callbacks, and a value returned by a plugin short-circuits the rest. Built-in plugins:

| Plugin | What it does |
|---|---|
| `LoggingPlugin` | Logs every callback through ADK's logger; verbose, for development |
| `DebugLoggingPlugin` | Writes every request, response and event to a YAML file (`adk_debug.yaml` by default) |
| `ReflectAndRetryToolPlugin` | Turns tool errors into a structured retry prompt, up to `maxRetries` |
| `ReflectAndRetryModelPlugin` | Retries failed model calls with reflection guidance (by default, responses that end in `MALFORMED_FUNCTION_CALL`) |
| `GlobalInstructionPlugin` | Adds an instruction to every agent in the app |
| `SecurityPlugin` | Allows, confirms or denies each tool call through a policy engine |
| `ToolCallIntegrityPlugin` | Stamps each stored function call with an HMAC and rejects calls whose stamp does not verify, so a tampered session cannot trigger tool calls |

## Streaming

Set `streamingMode: StreamingMode.SSE` in the run config to receive the reply in chunks. Chunk events have `partial: true`; the complete reply follows as a normal, non-partial event, and only that one is saved to the session.

```typescript
import {InMemoryRunner, LlmAgent, StreamingMode, isFinalResponse} from '@google/adk';

const agent = new LlmAgent({name: 'writer', model: '%%model=gemini%%', instruction: 'Write clearly.'});
const runner = new InMemoryRunner({agent, appName: 'stream_demo'});

const session = await runner.sessionService.createSession({appName: 'stream_demo', userId: 'u1'});
for await (const event of runner.runAsync({
  userId: 'u1',
  sessionId: session.id,
  newMessage: {role: 'user', parts: [{text: 'Explain HTTP caching in three paragraphs.'}]},
  runConfig: {streamingMode: StreamingMode.SSE, maxLlmCalls: 10},
  abortSignal: AbortSignal.timeout(60_000), // give up after a minute
})) {
  if (event.partial) {
    // A chunk of the reply as it arrives.
    process.stdout.write(event.content?.parts?.map((p) => p.text ?? '').join('') ?? '');
  } else if (isFinalResponse(event)) {
    // The complete reply arrives once more, unpartial, at the end.
    process.stdout.write('\n');
  }
}
```

`abortSignal` stops the run, including the model call in flight; `AbortSignal.timeout` is a simple way to bound a turn. `maxLlmCalls` caps model calls per invocation (500 by default). For voice and video, `runner.runLive({userId, sessionId, liveRequestQueue})` keeps a connection to a Gemini Live model open; you push input into a `LiveRequestQueue` and read events as the model responds.

## MCP

`MCPToolset` connects to a Model Context Protocol server and offers its tools to an agent. It takes the connection parameters, an optional tool filter (names or a predicate) and an optional name prefix. The server below is a 20-line MCP server written with the official SDK, run as a child process over stdio:

```typescript
// stock_server.ts: a small MCP server to connect to (run with: npx tsx stock_server.ts)
import {McpServer} from '@modelcontextprotocol/sdk/server/mcp.js';
import {StdioServerTransport} from '@modelcontextprotocol/sdk/server/stdio.js';
import {z} from 'zod';

const server = new McpServer({name: 'stock', version: '1.0.0'});
server.registerTool(
  'check_stock',
  {description: 'Units in stock for a SKU.', inputSchema: {sku: z.string()}},
  async ({sku}) => ({content: [{type: 'text', text: JSON.stringify({sku, units: 12})}]}),
);
server.registerTool(
  'delete_sku',
  {description: 'Delete a SKU.', inputSchema: {sku: z.string()}},
  async ({sku}) => ({content: [{type: 'text', text: `deleted ${sku}`}]}),
);
await server.connect(new StdioServerTransport());
```

```typescript
import {InMemoryRunner, LlmAgent, MCPToolset, isFinalResponse, stringifyContent} from '@google/adk';

// npm install @modelcontextprotocol/sdk
const stock = new MCPToolset(
  {
    type: 'StdioConnectionParams',
    serverParams: {command: 'npx', args: ['tsx', 'stock_server.ts']},
  },
  ['check_stock'], // offer only this tool to the model
);

// A remote server over Streamable HTTP, with a bearer token.
const remote = new MCPToolset({
  type: 'StreamableHTTPConnectionParams',
  url: 'https://mcp.example.com/mcp',
  transportOptions: {requestInit: {headers: {Authorization: `Bearer ${process.env.MCP_TOKEN ?? ''}`}}},
});
void remote; // add it to tools: [...] once the server exists

const agent = new LlmAgent({
  name: 'inventory',
  model: '%%model=gemini%%',
  instruction: 'Answer stock questions with check_stock.',
  tools: [stock],
});

const runner = new InMemoryRunner({agent, appName: 'inventory'});
try {
  for await (const event of runner.runEphemeral({
    userId: 'u1',
    newMessage: {role: 'user', parts: [{text: 'How many SKU-42 do we have?'}]},
  })) {
    if (isFinalResponse(event)) console.log(stringifyContent(event));
  }
} finally {
  await stock.close(); // stops the server process
}
```

- Connection types are `StdioConnectionParams` (a local process), `StreamableHTTPConnectionParams` (the current HTTP transport) and `SseConnectionParams` (the older SSE transport). Pass headers for remote servers through `transportOptions.requestInit` or `header`.
- The filter matters: offer the model only the tools it should use. Here `delete_sku` exists on the server but is never shown to the model.
- Call `close()` when you are done, or the child process keeps running.
- `MCPToolset` also exposes the server's resources (`listResources`, `readResource`) and `LoadMcpResourceTool` lets the model read them. ADK for TypeScript has no helper for exposing an agent *as* an MCP server; write one with the MCP SDK if you need it.

## A2A

The Agent2Agent protocol connects agents across processes and frameworks. `toA2a` (from `@google/adk/a2a`) turns an agent into an Express app with an agent card at `/.well-known/agent-card.json` and JSON-RPC and REST endpoints:

```typescript
import {LlmAgent} from '@google/adk';
import {bearerTokenUserBuilder, toA2a} from '@google/adk/a2a';

// npm install express
const weather = new LlmAgent({
  name: 'weather',
  model: '%%model=gemini%%',
  description: 'Answers questions about the weather.',
  instruction: 'Answer weather questions briefly.',
});

const token = process.env.A2A_TOKEN;
if (!token) throw new Error('Set A2A_TOKEN');

// Builds an Express app with the agent card, a JSON-RPC endpoint and a REST endpoint.
const app = await toA2a(weather, {
  host: 'localhost',
  port: 8001, // used to write the URLs in the agent card
  authentication: bearerTokenUserBuilder(token), // or allowUnauthenticated: true for local tests
});
export const server = app.listen(8001, () => console.log('A2A agent on http://localhost:8001'));
```

`toA2a` refuses to start without `authentication` unless you set `allowUnauthenticated: true`, because the endpoint lets any caller run the agent and its tools. `bearerTokenUserBuilder(token)` checks a shared bearer token; for anything stronger, pass your own user builder (OIDC, mTLS at the proxy).

On the calling side, `RemoteA2AAgent` is an ordinary agent, so it can be a sub-agent, a workflow node or wrapped in `AgentTool`. Authentication is added through the A2A client's `fetch`:

```typescript
import {InMemoryRunner, LlmAgent, isFinalResponse, stringifyContent} from '@google/adk';
import {RemoteA2AAgent} from '@google/adk/a2a';
import {ClientFactory, JsonRpcTransportFactory} from '@a2a-js/sdk/client';

// Add the bearer token to every request the A2A client sends.
const authFetch: typeof fetch = (input, init) => {
  const headers = new Headers(init?.headers);
  headers.set('Authorization', `Bearer ${process.env.A2A_TOKEN}`);
  return fetch(input, {...init, headers});
};

const weather = new RemoteA2AAgent({
  name: 'weather',
  description: 'Remote agent that answers weather questions.',
  agentCard: 'http://localhost:8001', // the card is read from /.well-known/agent-card.json
  clientFactory: new ClientFactory({transports: [new JsonRpcTransportFactory({fetchImpl: authFetch})]}),
});

// A remote agent is an ordinary agent: a sub-agent here.
const planner = new LlmAgent({
  name: 'travel_planner',
  model: '%%model=gemini%%',
  instruction: 'Plan trips. Transfer weather questions to the weather agent.',
  subAgents: [weather],
});

const runner = new InMemoryRunner({agent: weather, appName: 'client'}); // or agent: planner
for await (const event of runner.runEphemeral({
  userId: 'u1',
  newMessage: {role: 'user', parts: [{text: 'Will it rain in Leeds tomorrow?'}]},
})) {
  if (isFinalResponse(event)) console.log(stringifyContent(event));
}
void planner;
```

- `agentCard` is the base URL of the remote agent (the card path is added) or an `AgentCard` object. The card is fetched on first use, not at construction.
- The card's RPC URL must share the origin it was fetched from, and plain `http://` is accepted only for loopback hosts; `resolveAgentCardOptions: {allowInsecureRpc, allowCrossOriginRpc}` relaxes these checks when you have a reason to.
- `beforeRequestCallbacks` and `afterRequestCallbacks` see each A2A request and response; `A2AAgentExecutor` is the lower-level piece if you want to mount A2A in your own server.
- Install `@a2a-js/sdk` directly if you import from it, as above; it is already a dependency of `@google/adk`.

## Code execution and planners

`BuiltInCodeExecutor` lets Gemini write and run Python in Google's sandbox; the response then carries `executableCode` and `codeExecutionResult` parts beside the answer. A planner shapes how the model reasons before it answers: `BuiltInPlanner` sets Gemini's thinking configuration on every request, and `PlanReActPlanner` adds a prompt that asks for planning, reasoning and action sections, for models without native thinking.

```typescript
import {BuiltInCodeExecutor, BuiltInPlanner, LlmAgent, PlanReActPlanner} from '@google/adk';
import {ThinkingLevel} from '@google/genai';

// The model writes Python and Gemini runs it server-side.
export const analyst = new LlmAgent({
  name: 'analyst',
  model: '%%model=gemini%%',
  instruction: 'Answer numeric questions by writing and running Python. Print every result.',
  codeExecutor: new BuiltInCodeExecutor(),
});

// Gemini's own thinking, with the thoughts returned as parts marked thought: true.
export const thinker = new LlmAgent({
  name: 'thinker',
  model: '%%model=gemini%%',
  planner: new BuiltInPlanner({thinkingConfig: {includeThoughts: true, thinkingLevel: ThinkingLevel.LOW}}),
});

// A prompt-based plan / reason / act loop for models without native thinking.
export const reasoner = new LlmAgent({
  name: 'reasoner',
  model: '%%model=gemini-small%%',
  planner: new PlanReActPlanner(),
});
```

With `includeThoughts: true`, thought parts (`part.thought === true`) appear in the events, so filter them out before showing text to a user. `ThinkingLevel` comes from `@google/genai`, which `@google/adk` already depends on; add it to your own `package.json` if you import it.

The other executors run code on your side: `ContainerCodeExecutor` in a Docker container, `AgentEngineSandboxCodeExecutor` in a Vertex AI Agent Engine sandbox, and `UnsafeLocalCodeExecutor` in the local process, which, as the name says, gives model-written code your process's permissions. Use it only for trusted experiments.

## Testing

Build agents in functions that take the model as a parameter, then drive them through a real runner with a scripted `BaseLlm`. Tests run without credentials and assert on tool calls and replies:

```typescript
// agent.ts
import {FunctionTool, LlmAgent, type BaseLlm} from '@google/adk';
import {z} from 'zod';

export const getOrder = new FunctionTool({
  name: 'get_order',
  description: 'Look up an order.',
  parameters: z.object({orderId: z.string()}),
  execute: async ({orderId}) => ({orderId, status: 'shipped'}),
});

// Take the model as a parameter so tests can pass a fake.
export function buildAgent(model: string | BaseLlm = '%%model=gemini%%') {
  return new LlmAgent({
    name: 'support',
    model,
    instruction: 'Answer order questions with get_order.',
    tools: [getOrder],
  });
}
```

```typescript
// agent.test.ts: run with  npx tsx --test agent.test.ts
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {BaseLlm, InMemoryRunner, getFunctionCalls, isFinalResponse, stringifyContent, type BaseLlmConnection, type LlmRequest, type LlmResponse} from '@google/adk';
import {buildAgent} from './agent.js';

// A scripted model: returns the queued responses in order and records each request.
class ScriptedLlm extends BaseLlm {
  readonly requests: LlmRequest[] = [];
  constructor(private readonly script: LlmResponse[]) {
    super({model: 'scripted'});
  }
  async *generateContentAsync(request: LlmRequest): AsyncGenerator<LlmResponse, void> {
    this.requests.push(request);
    const next = this.script.shift();
    if (!next) throw new Error('script exhausted');
    yield next;
  }
  async connect(): Promise<BaseLlmConnection> {
    throw new Error('live mode is not supported by ScriptedLlm');
  }
}

test('looks up the order, then answers', async () => {
  const model = new ScriptedLlm([
    {content: {role: 'model', parts: [{functionCall: {name: 'get_order', args: {orderId: 'A123'}}}]}},
    {content: {role: 'model', parts: [{text: 'Order A123 has shipped.'}]}},
  ]);
  const runner = new InMemoryRunner({agent: buildAgent(model), appName: 'test'});

  let answer = '';
  const toolCalls: string[] = [];
  for await (const event of runner.runEphemeral({
    userId: 'u1',
    newMessage: {role: 'user', parts: [{text: 'Where is A123?'}]},
  })) {
    toolCalls.push(...getFunctionCalls(event).map((c) => c.name ?? ''));
    if (isFinalResponse(event)) answer = stringifyContent(event);
  }

  assert.deepEqual(toolCalls, ['get_order']);
  assert.equal(answer, 'Order A123 has shipped.');
  assert.equal(model.requests.length, 2);
  // The second request carries the tool's result back to the model.
  const last = model.requests[1].contents.at(-1);
  assert.deepEqual(last?.parts?.[0].functionResponse?.response, {orderId: 'A123', status: 'shipped'});
});
```

```bash
npx tsx --test agent.test.ts
```

Script function calls as well as text so the tool path is exercised, and assert on what the model was sent (`model.requests`): that is where instruction templating and tool results show up. The dev tools also run evaluation sets against the real model (`npx adk eval agent.ts my.evalset.json`); see the [production guide](/google-adk-guide/typescript/google_adk_typescript_production_guide/#evaluation).

## What TypeScript does not have

Compared with the Python kit, as of %%version=@google/adk%%:

- **Model providers.** Only Gemini, Apigee-proxied models and Chrome's built-in model are registered; there is no LiteLLM, Anthropic or OpenAI class in a published release. See [Models](#models) for writing your own.
- **Agent as an MCP server.** There is an MCP client (`MCPToolset`) but no `toMcpServer`.
- **Workflow as a sub-agent.** A `Workflow` can be a root or a node, not an `LlmAgent` sub-agent.
- **Firestore and other community session services.** The session services are in-memory, database (MikroORM) and Vertex AI.

The workflow engine, tool confirmation, plugins, memory, artifacts, A2A and live sessions are all present.

---

## Revision history

| Date | Version | Changes |
|---|---|---|
| 2026-10-24 | current | New page. Covers agents, models (including a custom `BaseLlm`), tools, toolsets, callbacks, multi-agent, workflows (routing, loops, retries, human input), sessions, memory, artifacts, tool confirmation, plugins, streaming, MCP, A2A, code execution and testing. Every example type-checked with `tsc --strict` and run. |

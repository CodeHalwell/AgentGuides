---
title: "OpenAI Agents SDK with TypeScript: Comprehensive Technical Guide"
description: "Agents, the runner, handoffs, agents as tools, function and hosted tools, MCP, structured output, guardrails, sessions, tracing, realtime voice and sandbox agents in the TypeScript OpenAI Agents SDK."
framework: "openai-agents-sdk"
language: "typescript"
sidebar:
  label: "Comprehensive guide"
  order: 1
---

> Verified against `@openai/agents` %%version=@openai/agents%% (checked %%versions-checked%%). Node.js 22+ (also Deno and Bun; Cloudflare Workers with `nodejs_compat` is experimental), Zod 4, `openai` v7. Every TypeScript block on this page compiles under `tsc --strict` as written and was run with the SDK's `ScriptedModel` in place of a live model.

## Table of Contents

1. [Core Fundamentals](#core-fundamentals)
2. [Simple Agents](#simple-agents)
3. [Multi-Agent Systems](#multi-agent-systems)
4. [Tools Integration](#tools-integration)
5. [Model Context Protocol](#model-context-protocol)
6. [Structured Output](#structured-output)
7. [Guardrails](#guardrails)
8. [Sessions](#sessions)
9. [Tracing](#tracing)
10. [Realtime Voice Agents](#realtime-voice-agents)
11. [Sandbox Agents](#sandbox-agents)
12. [Model Providers](#model-providers)
13. [Summary](#summary)

Elsewhere in this guide: [Lifecycle events, guardrails and approvals](/openai-agents-sdk-typescript-guide/openai_agents_sdk_middleware_typescript/) · [Production](/openai-agents-sdk-typescript-guide/openai_agents_sdk_typescript_production_guide/) (limits, retries, tracing controls, testing) · [Streaming server (Express)](/openai-agents-sdk-typescript-guide/openai_agents_streaming_server_express/) · [Recipes](/openai-agents-sdk-typescript-guide/openai_agents_sdk_typescript_recipes/).

---

## Core Fundamentals

### Installation and Setup

`@openai/agents` bundles the core, the OpenAI provider and realtime support; Zod is a peer dependency:

```bash
npm install @openai/agents@%%version=@openai/agents%% zod
# Optional: non-OpenAI models via the Vercel AI SDK, extra realtime transports, hosted sandbox providers
npm install @openai/agents-extensions@%%version=@openai/agents%% @ai-sdk/anthropic
```

The `@openai/agents*` packages are versioned together; keep them on the same version. Set `OPENAI_API_KEY` in the environment (or call `setDefaultOpenAIKey()`).

### TypeScript Configuration

The SDK ships ES modules. A `tsconfig.json` that matches the examples:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": true,
    "skipLibCheck": true,
    "outDir": "./dist",
    "rootDir": "./src"
  },
  "include": ["src/**/*"]
}
```

Set `"type": "module"` in `package.json` so top-level `await` works.

### Design Philosophy

The SDK has a small set of primitives and leaves orchestration to ordinary TypeScript:

- **Agent**: a model with instructions, tools, handoffs, guardrails and an optional output type
- **run() / Runner**: the loop that calls the model, runs tools and follows handoffs until there is a final output
- **Handoff**: transfers the conversation to another agent
- **Guardrail**: checks input, output or individual tool calls, and can stop the run
- **Session**: stores conversation history between runs
- **Tool**: a function (with a Zod schema), a hosted OpenAI tool, an MCP server, or another agent

It replaced OpenAI's experimental Swarm library, which was Python-only; see [Coming from Swarm](/openai-agents-sdk-typescript-guide/openai_agents_sdk_typescript_swarm_migration_guide/).

### Core Primitives

#### Agent

An `Agent` is configuration: a name, instructions, a model, and optionally tools, handoffs, guardrails and an output type. Sampling parameters go in `modelSettings`:

```typescript
import { Agent } from '@openai/agents';

export const researchAgent = new Agent({
  name: 'Research Assistant',
  instructions: `You are a thorough research assistant.
    - Provide well-sourced information
    - Acknowledge uncertainty when appropriate`,
  model: '%%model=openai%%',
  modelSettings: { maxTokens: 2048 },
});
```

If `model` is omitted the SDK uses its default (`gpt-5.6-luna` since 0.15.0) or the `OPENAI_DEFAULT_MODEL` environment variable. Set it explicitly in production: the default has changed between releases.

#### Runner

`run()` executes the agent loop until a final output is produced. A `Runner` instance holds defaults shared by many runs, and the result carries the output, the last agent and token usage:

```typescript
import { Agent, MaxTurnsExceededError, Runner, run } from '@openai/agents';

const agent = new Agent({ name: 'Researcher', instructions: 'Answer briefly.' });

// One-off run with the default runner
const result = await run(agent, 'What are the main uses of TypeScript?', {
  maxTurns: 10, // the default; null disables the limit
});
console.log(result.finalOutput);
console.log(result.lastAgent?.name);
console.log(result.runContext.usage.totalTokens);

// Shared configuration for many runs
const runner = new Runner({
  model: '%%model=openai-small%%', // overrides every agent's model
  workflowName: 'Research',
  traceIncludeSensitiveData: false,
});

try {
  const r = await runner.run(agent, 'Summarise the field in three bullets.');
  console.log(r.finalOutput);
} catch (error) {
  if (error instanceof MaxTurnsExceededError) console.error('Agent looped');
  else throw error;
}
```

#### Handoff

Handoffs let one agent delegate the conversation to another. List agents directly, or wrap them with `handoff()` to override the tool name, accept input data or filter history:

```typescript
import { Agent, handoff } from '@openai/agents';
import { z } from 'zod';

const billingAgent = new Agent({
  name: 'Billing Specialist',
  handoffDescription: 'Billing, invoices and payments',
  instructions: 'Handle billing enquiries accurately.',
});

const technicalAgent = new Agent({
  name: 'Technical Support',
  handoffDescription: 'Bugs, errors and crashes',
  instructions: 'Resolve technical issues step by step.',
});

export const triageAgent = Agent.create({
  name: 'Customer Service Triage',
  instructions: 'Route customers to the appropriate specialist.',
  handoffs: [
    billingAgent,
    handoff(technicalAgent, {
      toolNameOverride: 'transfer_to_technical',
      toolDescriptionOverride: 'Transfer the customer to technical support',
      inputType: z.object({ summary: z.string() }),
      onHandoff: (_ctx, input) => console.log('Escalation summary:', input?.summary),
    }),
  ],
});
```

`Agent.create()` (rather than `new Agent()`) keeps the result's output type correct when handoff targets have different output types.

#### Guardrail

An input guardrail checks the first agent's input; an output guardrail checks the final output. Each returns `{ tripwireTriggered, outputInfo }`, and a tripwire throws:

```typescript
import {
  Agent,
  InputGuardrail,
  InputGuardrailTripwireTriggered,
  OutputGuardrail,
  OutputGuardrailTripwireTriggered,
  run,
} from '@openai/agents';
import { z } from 'zod';

const noEmailsGuardrail: InputGuardrail = {
  name: 'no_email_addresses',
  execute: async ({ input }) => {
    const text = typeof input === 'string' ? input : JSON.stringify(input);
    const found = /[\w.+-]+@[\w-]+\.[\w.]+/.test(text);
    return { tripwireTriggered: found, outputInfo: { found } };
  },
};

const responseSchema = z.object({
  status: z.enum(['success', 'error', 'pending']),
  message: z.string(),
});

const statusGuardrail: OutputGuardrail<typeof responseSchema> = {
  name: 'no_error_status',
  execute: async ({ agentOutput }) => ({
    tripwireTriggered: agentOutput.status === 'error',
    outputInfo: { status: agentOutput.status },
  }),
};

const agent = new Agent({
  name: 'Support',
  instructions: 'Answer support questions.',
  outputType: responseSchema,
  inputGuardrails: [noEmailsGuardrail],
  outputGuardrails: [statusGuardrail],
});

try {
  await run(agent, 'My email is jane@example.com, please reset my password');
} catch (e) {
  if (e instanceof InputGuardrailTripwireTriggered) console.log('Input blocked');
  else if (e instanceof OutputGuardrailTripwireTriggered) console.log('Output blocked');
  else throw e;
}
```

More on guardrails, including blocking mode and model-based checks, in [Guardrails](#guardrails).

#### Session

Pass a session to `run()` and the SDK loads prior items before the turn and stores new ones afterwards:

```typescript
import { Agent, MemorySession, run } from '@openai/agents';

const agent = new Agent({ name: 'Tutor', instructions: 'Explain concepts simply.' });
const session = new MemorySession();

const first = await run(agent, 'What is machine learning?', { session });
console.log('Turn 1:', first.finalOutput);

const second = await run(agent, 'Explain neural networks in that context.', { session });
console.log('Turn 2:', second.finalOutput);

console.log('Stored items:', (await session.getItems()).length);
```

#### Tool

`tool()` builds a function tool from a Zod schema; arguments are validated before `execute` runs, and `execute`'s parameter is typed from the schema:

```typescript
import { tool } from '@openai/agents';
import { z } from 'zod';

export const calculateTool = tool({
  name: 'calculate',
  description: 'Performs basic arithmetic',
  parameters: z.object({
    operation: z.enum(['add', 'subtract', 'multiply', 'divide']),
    a: z.number(),
    b: z.number(),
  }),
  execute: async ({ operation, a, b }) => {
    switch (operation) {
      case 'add': return a + b;
      case 'subtract': return a - b;
      case 'multiply': return a * b;
      case 'divide': return b !== 0 ? a / b : 'Error: division by zero';
    }
  },
});
```

---

## Simple Agents

### Creating Basic Agents

```typescript
import { Agent, run } from '@openai/agents';

const agent = new Agent({
  name: 'General Assistant',
  instructions: 'You are a helpful, friendly assistant. Give clear, concise answers.',
  model: '%%model=openai%%',
});

const result = await run(agent, 'What is the capital of France?');
console.log(result.finalOutput);
```

### Context and Dynamic Instructions

The `context` you pass to `run()` is your own object. It is never sent to the model, but instructions functions, tools, guardrails and lifecycle events all receive it as `RunContext<T>`:

```typescript
import { Agent, RunContext, run, tool } from '@openai/agents';
import { z } from 'zod';

interface UserContext {
  userId: string;
  expertise: 'beginner' | 'expert';
}

const getAccount = tool({
  name: 'get_account',
  description: 'Look up the current user account',
  parameters: z.object({}),
  execute: async (_args, runContext?: RunContext<UserContext>) => {
    return { userId: runContext?.context.userId, plan: 'pro' };
  },
});

const tutor = new Agent<UserContext>({
  name: 'Tutor',
  instructions: (runContext) =>
    runContext.context.expertise === 'beginner'
      ? 'Explain simply and avoid jargon.'
      : 'Assume background knowledge and be precise.',
  tools: [getAccount],
});

const result = await run(tutor, 'Explain recursion.', {
  context: { userId: 'u_42', expertise: 'beginner' },
});
console.log(result.finalOutput);
```

### Multi-Turn Dialogues

Either pass the previous turn's history back in, or let a session hold it:

```typescript
import { Agent, AgentInputItem, MemorySession, run } from '@openai/agents';

const agent = new Agent({ name: 'Assistant', instructions: 'Remember what the user tells you.' });

// Manual history
let history: AgentInputItem[] = [{ role: 'user', content: 'My name is Alex.' }];
const turn1 = await run(agent, history);
history = [...turn1.history, { role: 'user', content: 'What is my name?' }];
const turn2 = await run(agent, history);
console.log(turn2.finalOutput);

// Session-managed history
const session = new MemorySession();
await run(agent, 'My name is Alex.', { session });
const answer = await run(agent, 'What is my name?', { session });
console.log(answer.finalOutput);
```

For server-side state on OpenAI models, pass `previousResponseId: result.lastResponseId` or a `conversationId` instead.

### Streaming Outputs

`run(agent, input, { stream: true })` returns a streamed result. `toTextStream()` yields only text; iterating the result yields every event:

```typescript
import { Agent, run } from '@openai/agents';

const agent = new Agent({ name: 'Writer', instructions: 'Write vivid prose.' });

// Text only
const stream = await run(agent, 'Describe a thunderstorm in two sentences.', { stream: true });
for await (const chunk of stream.toTextStream()) {
  process.stdout.write(chunk);
}
await stream.completed;
console.log('\nFinal:', stream.finalOutput);

// All events
const events = await run(agent, 'Now describe the morning after.', { stream: true });
for await (const event of events) {
  if (event.type === 'raw_model_stream_event' && event.data.type === 'output_text_delta') {
    process.stdout.write(event.data.delta);
  } else if (event.type === 'run_item_stream_event') {
    console.log(`\n[${event.name}]`);
  } else if (event.type === 'agent_updated_stream_event') {
    console.log(`\n[agent: ${event.agent.name}]`);
  }
}
await events.completed;
```

### Error Handling and Timeouts

The SDK throws typed errors. Use an `AbortSignal` to put a deadline on a whole run; unlike `Promise.race`, it cancels the in-flight model request and leaves no timer behind:

```typescript
import {
  Agent,
  InputGuardrailTripwireTriggered,
  MaxTurnsExceededError,
  ModelBehaviorError,
  OutputGuardrailTripwireTriggered,
  run,
} from '@openai/agents';

const agent = new Agent({ name: 'Assistant', instructions: 'Answer briefly.' });

async function answer(question: string): Promise<string> {
  try {
    const result = await run(agent, question, {
      maxTurns: 5,
      signal: AbortSignal.timeout(30_000),
    });
    return result.finalOutput ?? '';
  } catch (error) {
    if (error instanceof MaxTurnsExceededError) return 'That took too many steps.';
    if (error instanceof InputGuardrailTripwireTriggered) return 'I cannot help with that.';
    if (error instanceof OutputGuardrailTripwireTriggered) return 'The answer was withheld.';
    if (error instanceof ModelBehaviorError) return 'The model returned something unusable.';
    if (error instanceof Error && error.name === 'AbortError') return 'Timed out.';
    throw error;
  }
}

console.log(await answer('What is TypeScript?'));
```

To retry transient model errors inside the run, configure `modelSettings.retry` rather than wrapping `run()` in a loop; see [Production](/openai-agents-sdk-typescript-guide/openai_agents_sdk_typescript_production_guide/#bound-every-run).

---

## Multi-Agent Systems

### Handoff Mechanisms Between Agents

The triage agent sees each specialist as a `transfer_to_<name>` tool. After a handoff the specialist owns the conversation and produces the final output:

```typescript
import { Agent, run } from '@openai/agents';

const weatherAgent = new Agent({
  name: 'Weather Expert',
  handoffDescription: 'Weather questions and forecasts',
  instructions: 'Give weather information and forecasts.',
});

const financeAgent = new Agent({
  name: 'Finance Advisor',
  handoffDescription: 'Questions about money, budgets and investments',
  instructions: 'Give general financial information. Do not give personal advice.',
});

const triage = Agent.create({
  name: 'Universal Assistant',
  instructions: 'Route each question to the most appropriate specialist.',
  handoffs: [weatherAgent, financeAgent],
});

const result = await run(triage, 'Will it rain in Leeds tomorrow?');
console.log(result.lastAgent?.name, '->', result.finalOutput);
```

### Message Filtering During Handoffs

Control what history the receiving agent sees with `inputFilter` on `handoff()`. A filter takes and returns `HandoffInputData`. `removeAllTools` strips earlier tool calls and their results; it is exported from `@openai/agents-core/extensions`, so add `@openai/agents-core` (same version) to your dependencies if you import it, or write the filter yourself:

```typescript
import { Agent, HandoffInputData, handoff } from '@openai/agents';
import { removeAllTools } from '@openai/agents-core/extensions';

const MAX_HISTORY = 10;

function trimHistory(data: HandoffInputData): HandoffInputData {
  const withoutTools = removeAllTools(data);
  const history = Array.isArray(withoutTools.inputHistory)
    ? withoutTools.inputHistory.slice(-MAX_HISTORY)
    : withoutTools.inputHistory;
  return { ...withoutTools, inputHistory: history };
}

const specialist = new Agent({
  name: 'Specialist',
  instructions: 'Answer using only the recent conversation.',
});

export const frontDesk = new Agent({
  name: 'Front Desk',
  instructions: 'Hand complex questions to the specialist.',
  handoffs: [handoff(specialist, { inputFilter: trimHistory })],
});
```

### Agents as Tools

When a manager should stay in charge and only consult specialists, turn the specialists into tools with `agent.asTool()`. The specialist runs as a nested run and its final output becomes the tool result; the conversation does not move:

```typescript
import { Agent, run } from '@openai/agents';

const summariser = new Agent({
  name: 'Summariser',
  instructions: 'Summarise the text you are given in at most three sentences.',
});

const translator = new Agent({
  name: 'Translator',
  instructions: 'Translate the text you are given into the requested language.',
});

const editor = new Agent({
  name: 'Editor',
  instructions: 'Use your tools to summarise documents and translate them on request.',
  tools: [
    summariser.asTool({
      toolName: 'summarise_text',
      toolDescription: 'Summarise a piece of text',
    }),
    translator.asTool({
      toolName: 'translate_text',
      toolDescription: 'Translate text; say which language in the input',
    }),
  ],
});

const result = await run(editor, 'Summarise this and translate it into French: TypeScript adds static types to JavaScript...');
console.log(result.finalOutput);
```

`asTool()` also accepts `customOutputExtractor` (choose what the manager sees from the nested run), `parameters` (a Zod schema for structured tool input), `needsApproval` and `isEnabled`.

### Parallel Agent Execution

Independent runs are just promises. `Promise.allSettled` keeps one failure from discarding the other results:

```typescript
import { Agent, run } from '@openai/agents';

const analysts = [
  new Agent({ name: 'Market', instructions: 'Analyse market trends.' }),
  new Agent({ name: 'Risk', instructions: 'Analyse risks.' }),
  new Agent({ name: 'Finance', instructions: 'Analyse financial projections.' }),
];

const settled = await Promise.allSettled(
  analysts.map((agent) => run(agent, 'Assess a launch into the UK market.', {
    signal: AbortSignal.timeout(60_000),
  })),
);

settled.forEach((outcome, i) => {
  const name = analysts[i]?.name ?? `agent ${i}`;
  if (outcome.status === 'fulfilled') console.log(name, outcome.value.finalOutput);
  else console.error(name, 'failed:', outcome.reason);
});
```

---

## Tools Integration

### Function Tools

Each tool has a name, a description the model reads, a Zod schema and an `execute` function. Useful options:

- `needsApproval`: `true` or an async predicate; pauses the run for a human ([approvals](/openai-agents-sdk-typescript-guide/openai_agents_sdk_middleware_typescript/#human-in-the-loop-approvals))
- `timeoutMs` and `timeoutBehavior`: bound a slow tool; by default the timeout is reported to the model as an error result
- `errorFunction`: format a thrown error for the model, or `null` to let errors propagate
- `inputGuardrails` / `outputGuardrails`: [tool guardrails](/openai-agents-sdk-typescript-guide/openai_agents_sdk_middleware_typescript/#tool-guardrails)
- `isEnabled`: hide the tool per run, based on context

```typescript
import { Agent, run, tool } from '@openai/agents';
import { z } from 'zod';

const getWeather = tool({
  name: 'get_weather',
  description: 'Current weather for a city',
  parameters: z.object({
    city: z.string().describe('City name, e.g. "Leeds"'),
    units: z.enum(['celsius', 'fahrenheit']).default('celsius'),
  }),
  timeoutMs: 5_000,
  execute: async ({ city, units }) => {
    // Call your weather API here
    return { city, units, temperature: 14, conditions: 'light rain' };
  },
});

const queryDatabase = tool({
  name: 'query_database',
  description: 'Run a SQL query against the reporting database',
  parameters: z.object({ sql: z.string() }),
  // Anything other than a SELECT waits for a human
  needsApproval: async (_ctx, { sql }) => !sql.trim().toUpperCase().startsWith('SELECT'),
  errorFunction: (_ctx, error) =>
    `Query failed: ${error instanceof Error ? error.message : String(error)}`,
  execute: async ({ sql }) => ({ sql, rows: [] as unknown[] }),
});

const agent = new Agent({
  name: 'Assistant',
  instructions: 'Use the tools to answer.',
  tools: [getWeather, queryDatabase],
});

const result = await run(agent, 'What is the weather in Leeds?');
console.log(result.finalOutput);
```

### Schemas That Convert to JSON Schema

Tool parameters are sent to the model as JSON Schema, and `tool()` throws at definition time if a Zod schema cannot be represented. Two common traps:

- Dates and other transforms (`z.date()`, `z.coerce.date()`, `.transform()`): accept a string and parse it inside `execute`.
- A union at the root: the root must be an object, so wrap the union in a property.

```typescript
import { tool } from '@openai/agents';
import { z } from 'zod';

export const searchEvents = tool({
  name: 'search_events',
  description: 'Search events by keyword or by date range',
  parameters: z.object({
    query: z.discriminatedUnion('kind', [
      z.object({ kind: z.literal('keyword'), text: z.string() }),
      z.object({
        kind: z.literal('dateRange'),
        from: z.string().describe('ISO 8601 date, e.g. 2026-03-01'),
        to: z.string().describe('ISO 8601 date'),
      }),
    ]),
  }),
  execute: async ({ query }) => {
    if (query.kind === 'keyword') return `Events matching "${query.text}"`;
    const from = new Date(query.from);
    const to = new Date(query.to);
    if (Number.isNaN(from.getTime()) || Number.isNaN(to.getTime())) return 'Invalid date';
    return `Events between ${from.toDateString()} and ${to.toDateString()}`;
  },
});
```

### OpenAI Hosted Tools

Hosted tools run on OpenAI's side and need an OpenAI Responses model. They are imported from `@openai/agents`:

```typescript
import {
  Agent,
  codeInterpreterTool,
  fileSearchTool,
  imageGenerationTool,
  run,
  webSearchTool,
} from '@openai/agents';

const researcher = new Agent({
  name: 'Researcher',
  instructions: 'Answer with sources. Use the documents first, then the web.',
  model: '%%model=openai%%',
  tools: [
    webSearchTool(),
    fileSearchTool(['vs_your_vector_store_id'], { maxNumResults: 5 }),
    codeInterpreterTool(), // runs Python in an OpenAI-managed container
    imageGenerationTool({ quality: 'low' }),
  ],
});

const result = await run(researcher, 'Chart UK inflation over the last five years.');
console.log(result.finalOutput);
```

Tools that act on your own machine are also built in: `computerTool({ computer })` (you implement the `Computer` interface over a browser or VM), `shellTool()` and `applyPatchTool()`. Each accepts `needsApproval`.

### Programmatic Tool Calling

Supported Responses models can write a short JavaScript program that calls several of your tools in one step (in a hosted V8 sandbox with no filesystem or network access). Tools opt in with `allowedCallers`:

```typescript
import { Agent, programmaticToolCallingTool, run, tool } from '@openai/agents';
import { z } from 'zod';

const getInventory = tool({
  name: 'get_inventory',
  description: 'Stock level for one SKU',
  parameters: z.object({ sku: z.string() }),
  allowedCallers: ['programmatic'],
  execute: async ({ sku }) => ({ sku, available: 42 }),
});

const planner = new Agent({
  name: 'Inventory planner',
  model: '%%model=openai%%',
  modelSettings: { toolChoice: 'programmatic_tool_calling' },
  tools: [getInventory, programmaticToolCallingTool()],
});

const result = await run(planner, 'Check stock for lamp-1, lamp-2 and lamp-3 and summarise.');
console.log(result.finalOutput);
```

---

## Model Context Protocol

### Local and Remote MCP Servers

`MCPServerStdio` starts a local server as a subprocess; `MCPServerStreamableHttp` (or the older `MCPServerSSE`) connects to a remote one. Connect before the run and close afterwards; the agent lists the server's tools on each run unless `cacheToolsList: true`:

```typescript
import { Agent, MCPServerStdio, MCPServerStreamableHttp, run } from '@openai/agents';

const filesystem = new MCPServerStdio({
  name: 'Filesystem',
  fullCommand: 'npx -y @modelcontextprotocol/server-filesystem ./project',
  cacheToolsList: true,
});

const git = new MCPServerStdio({
  name: 'Git',
  command: 'uvx', // the reference Git server is a Python package
  args: ['mcp-server-git', '--repository', './project'],
});

const github = new MCPServerStreamableHttp({
  name: 'GitHub',
  url: 'https://api.githubcopilot.com/mcp/',
  requestInit: { headers: { Authorization: `Bearer ${process.env.GITHUB_TOKEN ?? ''}` } },
});

await Promise.all([filesystem.connect(), git.connect(), github.connect()]);
try {
  const agent = new Agent({
    name: 'Code assistant',
    instructions: 'Use the MCP tools to answer questions about the project.',
    mcpServers: [filesystem, git, github],
  });
  const result = await run(agent, 'Summarise the latest three commits.');
  console.log(result.finalOutput);
} finally {
  await Promise.all([filesystem.close(), git.close(), github.close()]);
}
```

Since 0.20 automatic tool discovery stops after 64 pages of tool listings; raise `maxListPages` for very large servers. `toolFilter` limits which tools an agent sees.

### Hosted MCP Tools

`hostedMcpTool()` gives the server to the OpenAI Responses API, which lists and calls its tools itself; your process never connects. Use it for public HTTPS servers with OpenAI models:

```typescript
import { Agent, hostedMcpTool, run } from '@openai/agents';

const agent = new Agent({
  name: 'Docs assistant',
  instructions: 'Answer questions about the repository.',
  model: '%%model=openai%%',
  tools: [
    hostedMcpTool({
      serverLabel: 'gitmcp',
      serverUrl: 'https://gitmcp.io/openai/codex',
      requireApproval: 'never',
    }),
  ],
});

const result = await run(agent, 'Which language is this repository written in?');
console.log(result.finalOutput);
```

With `requireApproval: 'always'` (or a per-tool mapping), calls surface as `result.interruptions` and use the same approve/reject flow as function tools.

---

## Structured Output

### Output Schema Definition

Give the agent an `outputType` (a Zod object schema) and `result.finalOutput` is parsed, validated and typed:

```typescript
import { Agent, run } from '@openai/agents';
import { z } from 'zod';

const Analysis = z.object({
  summary: z.string(),
  keyPoints: z.array(z.string()),
  sentiment: z.enum(['positive', 'negative', 'neutral']),
  confidence: z.number().min(0).max(1),
});

const analyst = new Agent({
  name: 'Analyst',
  instructions: 'Analyse the text you are given.',
  outputType: Analysis,
});

const result = await run(analyst, 'The launch went well, though support queues grew.');
const analysis = result.finalOutput; // typed as z.infer<typeof Analysis> | undefined
if (analysis) {
  console.log(analysis.sentiment, analysis.keyPoints.length);
}
```

### Complex Nested Structures

Nested objects, arrays, enums, optional fields and discriminated unions (inside an object) all work. Optional fields are sent as nullable in strict mode, so use `.nullable()` or `.optional()` as your code prefers:

```typescript
import { Agent, run } from '@openai/agents';
import { z } from 'zod';

const Address = z.object({ street: z.string(), city: z.string(), postcode: z.string() });

const Company = z.object({
  name: z.string(),
  founded: z.number().int(),
  headquarters: Address,
  website: z.string().nullable(),
  contacts: z.array(
    z.object({
      name: z.string(),
      role: z.enum(['sales', 'support', 'press']),
      email: z.string(),
    }),
  ),
  status: z.discriminatedUnion('kind', [
    z.object({ kind: z.literal('active') }),
    z.object({ kind: z.literal('acquired'), acquirer: z.string() }),
  ]),
});

const extractor = new Agent({
  name: 'Company extractor',
  instructions: 'Extract company details from the text.',
  outputType: Company,
});

const result = await run(extractor, 'Acme Ltd, founded 2010, 1 High Street, Leeds LS1 1AA...');
const company = result.finalOutput;
if (company) {
  console.log(company.name, company.headquarters.city);
  if (company.status.kind === 'acquired') console.log('Acquired by', company.status.acquirer);
}
```

### Structured Output with Handoffs

When the agents in a handoff chain have different output types, create the entry agent with `Agent.create()` so the result type is the union of the possible outputs, then narrow on `result.lastAgent`:

```typescript
import { Agent, run } from '@openai/agents';
import { z } from 'zod';

const Refund = z.object({ orderId: z.string(), amount: z.number() });

const refunds = new Agent({
  name: 'Refunds',
  handoffDescription: 'Refund requests',
  instructions: 'Extract the refund request.',
  outputType: Refund,
});

const triage = Agent.create({
  name: 'Triage',
  instructions: 'Hand refund requests to Refunds; answer anything else yourself.',
  handoffs: [refunds],
});

const result = await run(triage, 'Please refund 20 pounds on order A-17.');
const output = result.finalOutput; // string | { orderId: string; amount: number } | undefined
if (typeof output === 'string') console.log('Text answer:', output);
else if (output) console.log('Refund for', output.orderId, output.amount);
```

---

## Guardrails

### Blocking Versus Parallel Input Guardrails

Input guardrails run alongside the first model call by default, so a tripwire stops the run but the model call may already have been made. Set `runInParallel: false` to make the run wait for the guardrail first:

```typescript
import { Agent, InputGuardrail, InputGuardrailTripwireTriggered, run } from '@openai/agents';
import { z } from 'zod';

const Screening = z.object({ offTopic: z.boolean(), reason: z.string() });

const screener = new Agent({
  name: 'Screener',
  instructions: 'Decide whether the message is about our banking products. Be strict.',
  model: '%%model=openai-small%%',
  outputType: Screening,
});

const onTopic: InputGuardrail = {
  name: 'on_topic',
  runInParallel: false,
  execute: async ({ input, context }) => {
    const verdict = await run(screener, input, { context: context.context });
    return {
      tripwireTriggered: verdict.finalOutput?.offTopic ?? true,
      outputInfo: verdict.finalOutput,
    };
  },
};

const assistant = new Agent({
  name: 'Banking assistant',
  instructions: 'Help with accounts and cards.',
  inputGuardrails: [onTopic],
});

try {
  const result = await run(assistant, 'How do I freeze my card?');
  console.log(result.finalOutput);
} catch (error) {
  if (error instanceof InputGuardrailTripwireTriggered) {
    console.log('Blocked:', error.result.output.outputInfo);
  } else {
    throw error;
  }
}
```

A small, fast model with a structured verdict is more reliable than keyword lists, which block legitimate messages and miss rephrased ones.

### Output Guardrails and Streaming

Output guardrails run on the final output only, including in streamed runs, so they cannot recall text that has already been streamed. Buffer the stream if unchecked text must never reach a user. Guardrails for every agent in a run go on the `Runner` (`new Runner({ inputGuardrails, outputGuardrails })`); checks on individual tool calls are [tool guardrails](/openai-agents-sdk-typescript-guide/openai_agents_sdk_middleware_typescript/#tool-guardrails).

---

## Sessions

| Session | Package | Stores history |
|---------|---------|----------------|
| `MemorySession` | `@openai/agents` | In process memory (tests, single-process apps) |
| `OpenAIConversationsSession` | `@openai/agents` | On OpenAI, via the Conversations API |
| `OpenAIResponsesCompactionSession` | `@openai/agents` | Wraps another session and compacts long histories |
| Your own class implementing `Session` | – | Redis, Postgres, DynamoDB… |

The `Session` interface is five methods. A minimal Redis-style store, shown here with a `Map` so it runs as is:

```typescript
import { Agent, AgentInputItem, Session, run } from '@openai/agents';

class KeyValueSession implements Session {
  constructor(
    private readonly id: string,
    private readonly store: Map<string, AgentInputItem[]>,
  ) {}

  async getSessionId(): Promise<string> {
    return this.id;
  }

  async getItems(limit?: number): Promise<AgentInputItem[]> {
    const items = this.store.get(this.id) ?? [];
    return limit === undefined ? [...items] : items.slice(-limit);
  }

  async addItems(items: AgentInputItem[]): Promise<void> {
    this.store.set(this.id, [...(this.store.get(this.id) ?? []), ...items]);
  }

  async popItem(): Promise<AgentInputItem | undefined> {
    const items = this.store.get(this.id) ?? [];
    const last = items.pop();
    this.store.set(this.id, items);
    return last;
  }

  async clearSession(): Promise<void> {
    this.store.delete(this.id);
  }
}

const store = new Map<string, AgentInputItem[]>();
const agent = new Agent({ name: 'Assistant', instructions: 'Be brief.' });
const session = new KeyValueSession('tenant-1:conversation-9', store);

await run(agent, 'Remember: my order is A-17.', { session });
const result = await run(agent, 'Which order did I mention?', { session });
console.log(result.finalOutput, (await session.getItems()).length);
```

Key sessions by conversation, not by request: a key that changes on every call starts a new conversation each time.

---

## Tracing

Every run is traced and exported to the OpenAI dashboard by default. `withTrace()` groups several runs into one trace; `addTraceProcessor()` sends spans somewhere else as well, and `setTraceProcessors()` replaces the default exporter:

```typescript
import {
  Agent,
  Span,
  Trace,
  TracingProcessor,
  addTraceProcessor,
  run,
  withTrace,
} from '@openai/agents';

class ConsoleProcessor implements TracingProcessor {
  async onTraceStart(trace: Trace): Promise<void> {
    console.log('trace start', trace.name);
  }
  async onTraceEnd(trace: Trace): Promise<void> {
    console.log('trace end', trace.traceId);
  }
  async onSpanStart(_span: Span<any>): Promise<void> {}
  async onSpanEnd(span: Span<any>): Promise<void> {
    console.log('span', span.spanData.type, span.endedAt);
  }
  async shutdown(): Promise<void> {}
  async forceFlush(): Promise<void> {}
}

addTraceProcessor(new ConsoleProcessor());

const agent = new Agent({ name: 'Processor', instructions: 'Process data.' });

await withTrace(
  'Customer data pipeline',
  async () => {
    await run(agent, 'Validate the customer records.');
    await run(agent, 'Summarise the validation results.');
  },
  { groupId: 'batch-2026-10-21', metadata: { environment: 'staging' } },
);
```

What appears in traces (`traceIncludeSensitiveData`), disabling tracing, and flushing in short-lived processes are covered in [Production](/openai-agents-sdk-typescript-guide/openai_agents_sdk_typescript_production_guide/#tracing-and-data-controls). Observability vendors (Langfuse, Braintrust, Arize/OpenInference and others) publish processors for this SDK.

---

## Realtime Voice Agents

Realtime agents live in `@openai/agents/realtime` and run on a speech-to-speech Realtime model (the default has been `gpt-realtime-2.1` since 0.13.0). A `RealtimeSession` connects over WebRTC in the browser, where it also wires up the microphone and speaker, or over WebSocket on a server:

```typescript
import { RealtimeAgent, RealtimeSession, tool } from '@openai/agents/realtime';
import { z } from 'zod';

const getWeather = tool({
  name: 'get_weather',
  description: 'Current weather for a city',
  parameters: z.object({ city: z.string() }),
  execute: async ({ city }) => `It is dry and mild in ${city}.`,
});

const agent = new RealtimeAgent({
  name: 'Voice assistant',
  instructions: 'Speak in short, friendly sentences.',
  tools: [getWeather],
});

const session = new RealtimeSession(agent, {
  transport: 'websocket', // server-side; browsers default to WebRTC
});

session.on('audio', (event) => {
  // event.data is an ArrayBuffer of PCM16 audio: play it or forward it to a client
  console.log('audio bytes', event.data.byteLength);
});
session.on('agent_tool_start', (_ctx, _agent, tool) => console.log('tool', tool.name));
session.on('error', (event) => console.error(event.error));

await session.connect({ apiKey: process.env.OPENAI_API_KEY ?? '' });
session.sendMessage('What is the weather in Leeds?');
// For live audio: session.sendAudio(pcm16Chunk)
```

In the browser, mint a short-lived client secret on your server and pass that as `apiKey`; never ship your API key to the client. Realtime sessions support tool approvals (`tool_approval_requested`), output guardrails on the transcript, and handoffs between realtime agents. `@openai/agents/realtime/testing` has a scripted transport for tests.

---

## Sandbox Agents

A `SandboxAgent` is an agent with a workspace: a manifest of files, directories and Git repositories to stage, and sandbox-native tools (shell, filesystem) to work in it. It runs through the normal `run()`, with a sandbox client chosen per run:

```typescript
import { run } from '@openai/agents';
import { Capabilities, SandboxAgent, gitRepo, localDir } from '@openai/agents/sandbox';
import { UnixLocalSandboxClient } from '@openai/agents/sandbox/local';

const agent = new SandboxAgent({
  name: 'Workspace assistant',
  model: '%%model=openai%%',
  instructions: 'Inspect the repository before answering. Write any reports to output/.',
  defaultManifest: {
    entries: {
      repo: gitRepo({ repo: 'openai/openai-agents-js' }),
      data: localDir({ src: './data' }),
    },
  },
  capabilities: Capabilities.default(),
});

const result = await run(agent, 'Summarise what the repository README says the project does.', {
  sandbox: { client: new UnixLocalSandboxClient() },
});
console.log(result.finalOutput);
```

`UnixLocalSandboxClient` (macOS and Linux) runs commands as your own user with no isolation, so keep it for trusted local work. `DockerSandboxClient` (also in `@openai/agents/sandbox/local`) isolates the workspace in a container, and `@openai/agents-extensions` provides hosted providers (E2B, Modal, Daytona, Vercel, Cloudflare, Blaxel). `scriptedSandboxSession` in `@openai/agents/testing` scripts sandbox calls for tests.

---

## Model Providers

OpenAI models are referenced by name and use the Responses API. Call `setOpenAIAPI('chat_completions')` at start-up to use Chat Completions instead. Other providers go through the Vercel AI SDK adapter in `@openai/agents-extensions`:

```typescript
import { Agent, run } from '@openai/agents';
import { aisdk } from '@openai/agents-extensions/ai-sdk';
import { anthropic } from '@ai-sdk/anthropic';
import { google } from '@ai-sdk/google';

const claudeAgent = new Agent({
  name: 'Claude assistant',
  instructions: 'Be concise.',
  model: aisdk(anthropic('%%model=anthropic%%')),
  modelSettings: { temperature: 0.7 },
});

const geminiAgent = new Agent({
  name: 'Gemini assistant',
  instructions: 'Be concise.',
  model: aisdk(google('%%model=gemini%%')),
});

for (const agent of [claudeAgent, geminiAgent]) {
  const result = await run(agent, 'Explain recursion in one sentence.');
  console.log(agent.name, result.finalOutput);
}
```

Hosted OpenAI tools (web search, file search, code interpreter, hosted MCP) need an OpenAI Responses model. Tracing still exports to OpenAI unless you disable it or set a separate key with `setTracingExportApiKey()`.

---

## Summary

1. Start with one agent and `run()`; add tools before adding agents.
2. Use handoffs when a specialist should own the reply, and `agent.asTool()` when a manager should keep control.
3. Give agents an `outputType` when code consumes the result.
4. Store history in a session keyed by conversation.
5. Add guardrails, tool guardrails and `needsApproval` where input is untrusted or tools have side effects.
6. Bound runs with `maxTurns`, `signal` and model retry settings, and test with `ScriptedModel`.
7. Read traces when an agent misbehaves; decide early what may appear in them.

The release table, with breaking changes, is on the [overview page](/openai-agents-sdk-typescript-guide/#whats-new).

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| %%version=@openai/agents%% | %%versions-checked%% | Rewritten. Every block now compiles under `tsc --strict` with its own imports and runs against `ScriptedModel`. Added `agent.asTool()`, hosted tools (the old section showed hand-written look-alikes), programmatic tool calling, MCP (moved from the features page, with the correct Git server), guardrails, sessions with a custom `Session`, tracing processors, realtime and sandbox agents. Fixed a tool schema that threw at definition time (dates, root unions) and typed-output examples that ignored `outputType`. Removed generic router, factory and retry classes. Moved the breaking-changes table to the overview. |
| %%version=@openai/agents%% | 2026-10-21 | Earlier pass: rewrote the core primitives against the real API; fixed install commands; replaced non-existent APIs; Zod 4 fixes. |
| 0.11.6 | 2026-05-29 | Version bump; key symbols re-checked. |
| 0.8.0 | 2026-03 | Realtime section added. |
| 0.3.2 | 2025-11 | First version of this guide. |

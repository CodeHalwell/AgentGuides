---
title: "Google ADK for TypeScript - Production Guide"
description: "Running ADK for TypeScript agents in production: the adk dev tools, the API server, deploying to Cloud Run and Agent Engine, your own Express server, persistence, tool policies, A2A authentication, telemetry and evaluation."
framework: "google-adk"
language: "typescript"
sidebar:
  label: "Production"
  order: 30
---

**Verified against:** `@google/adk` and `@google/adk-devtools` %%version=@google/adk%% (checked %%versions-checked%%). The TypeScript examples pass `tsc --strict` and were run under Node.js 22 with a scripted model; command-line options are from the dev tools' own `--help` and source. Deployment commands were not run against a live Google Cloud project.

Concepts (agents, tools, sessions, plugins) are in the [comprehensive guide](/google-adk-guide/typescript/google_adk_typescript_comprehensive_guide/).

## Contents

1. [The adk command](#the-adk-command)
2. [The API server](#the-api-server)
3. [Deploying with adk deploy](#deploying-with-adk-deploy)
4. [Your own server](#your-own-server)
5. [Persistence](#persistence)
6. [Tool policies](#tool-policies)
7. [Serving A2A safely](#serving-a2a-safely)
8. [Telemetry](#telemetry)
9. [Evaluation](#evaluation)
10. [Checklist](#checklist)

## The adk command

`@google/adk-devtools` installs an `adk` command. Install it in the project (`npm install -D @google/adk-devtools`) and call it with `npx adk`.

| Command | What it does |
|---|---|
| `npx adk create my_agent` | Scaffold a folder with an `agent.ts` exporting `rootAgent` and a `.env`; `--model`, `--api_key`, `--project`, `--region`, `--language ts\|js` and `-y` skip the prompts |
| `npx adk run agent.ts` | Chat in the terminal. `--save_session` writes the session to JSON on exit, `--resume file.json` continues one, `--replay file.json` runs scripted queries |
| `npx adk web [dir]` | The dev UI and API server on `http://localhost:8000` |
| `npx adk api_server [dir]` | The same server without the UI |
| `npx adk eval agent.ts set.json` | Run evaluation cases; see [Evaluation](#evaluation) |
| `npx adk deploy cloud_run [dir]` | Build and deploy to Cloud Run |
| `npx adk deploy agent_engine [dir]` | Build and deploy to Vertex AI Agent Engine |

The agent file must export `rootAgent`: an agent or a `Workflow`. `web`, `api_server` and `deploy` take either one file or a directory of agents laid out as `dir/<name>.ts` or `dir/<name>/agent.ts`. TypeScript files are compiled with esbuild before they run (`--compile false` to skip, `--bundle false` to keep dependencies external). A `.env` next to the agent is loaded automatically.

## The API server

`adk web` and `adk api_server` serve the same HTTP API the dev UI uses:

| Route | Purpose |
|---|---|
| `GET /list-apps` | Agents found in the directory |
| `POST /apps/{app}/users/{user}/sessions`, `GET`/`DELETE .../sessions/{id}` | Create, read and delete sessions |
| `POST /run` | Run one turn and return all its events |
| `POST /run_sse` | Run one turn and stream events as server-sent events |
| `GET .../sessions/{id}/artifacts[/{name}[/versions/{v}]]` | List and fetch artifacts |
| `GET /health`, `GET /version` | Health and version checks |

Options for `web` and `api_server` (`run` accepts the session, artifact and telemetry ones too):

- `--host` (default `localhost`) and `--port` (default `8000`).
- `--session_service_uri`: `memory://` by default, or a database URI such as `postgresql://...` or `sqlite://sessions.db` (with the matching `@mikro-orm/*` driver installed). The `DATABASE_URL` environment variable is used when the option is absent.
- `--artifact_service_uri`: `gs://<bucket>` for Cloud Storage.
- `--allow_origins` for CORS, and `--allowed_hosts` to widen the DNS-rebinding guard for a reverse proxy in front of a loopback-bound server.
- `--otel_to_cloud` to export traces to Google Cloud.
- `--a2a` to serve each agent over A2A as well, with `--a2a_auth_token` to require a bearer token on it.

The server has no user authentication: whoever can reach it can run the agent as any user ID. Keep it on `localhost` for development, and in production put it behind authentication (Cloud Run IAM, Identity-Aware Proxy, an API gateway) or use [your own server](#your-own-server).

## Deploying with adk deploy

### Cloud Run

```bash
npx adk deploy cloud_run ./agents \
  --project my-gcp-project \
  --region europe-west2 \
  --service_name support-agent \
  --session_service_uri "$DATABASE_URI" \
  --artifact_service_uri gs://my-agent-artifacts \
  --otel_to_cloud \
  --no-allow-unauthenticated
```

What it does: writes a Dockerfile (Node.js LTS on Alpine, a non-root user, `GOOGLE_GENAI_USE_ENTERPRISE=1` with your project and region so the agent uses Vertex AI), copies the agent and `package.json`, and runs `gcloud run deploy --source` on the result. The container runs `adk api_server` (or `adk web` with `--with_ui`). Project and region default to your `gcloud config` values.

- Options the dev tools do not recognise are passed through to `gcloud run deploy`, which is how `--no-allow-unauthenticated`, `--service-account` or `--set-secrets` reach it above. Pass `--no-allow-unauthenticated` explicitly: the dev tools do not add it.
- The service account needs `roles/aiplatform.user` for Gemini on Vertex AI, plus whatever your tools and storage use.
- `--adk_version` pins the dev tools version installed in the image (the default is `latest`); pin it so a rebuild does not change behaviour under you.
- `--a2a` adds the A2A endpoint. Pair it with `--a2a_auth_token`, which is sent to Cloud Run as the `ADK_A2A_AUTH_TOKEN` environment variable; without it the deployed A2A endpoint has no authentication.

### Vertex AI Agent Engine

```bash
npx adk deploy agent_engine ./agents \
  --project my-gcp-project \
  --region europe-west2 \
  --repository my-docker-repo \
  --display_name "Support agent"
```

`--repository` (an Artifact Registry Docker repository) is required; create it with `gcloud artifacts repositories create my-docker-repo --repository-format=docker --location=europe-west2`. `--staging_bucket` sets where Cloud Build logs go. Agent Engine provides managed sessions, so use `VertexAiSessionService` (or `vertexai://`) rather than a database for state.

## Your own server

When you need your own authentication, routes or response format, run the agent inside your HTTP server. Build the runner once and share it; take the user ID from your authentication layer, never from the request body:

```typescript
import express from 'express';
import {DatabaseSessionService, LlmAgent, Runner, StreamingMode} from '@google/adk';

const agent = new LlmAgent({name: 'assistant', model: '%%model=gemini%%', instruction: 'Be helpful and brief.'});

// Build these once at start-up and share them across requests.
const runner = new Runner({
  appName: 'assistant',
  agent,
  sessionService: new DatabaseSessionService(process.env.SESSION_DB ?? 'sqlite://sessions.db'),
  autoCreateSession: true,
});

const app = express();
app.use(express.json());

// POST /chat {"sessionId": "...", "message": "..."} streams the reply as server-sent events.
app.post('/chat', async (req, res) => {
  const userId = String(req.header('x-user-id') ?? ''); // take this from your auth layer, not the client
  const {sessionId, message} = req.body as {sessionId?: string; message?: string};
  if (!userId || !sessionId || !message) {
    res.status(400).json({error: 'x-user-id, sessionId and message are required'});
    return;
  }

  res.setHeader('content-type', 'text/event-stream');
  res.setHeader('cache-control', 'no-cache');
  const controller = new AbortController();
  res.on('close', () => controller.abort()); // stop the run if the client goes away

  try {
    for await (const event of runner.runAsync({
      userId,
      sessionId,
      newMessage: {role: 'user', parts: [{text: message}]},
      runConfig: {streamingMode: StreamingMode.SSE, maxLlmCalls: 20},
      abortSignal: controller.signal,
    })) {
      const text = event.content?.parts?.map((p) => p.text ?? '').join('') ?? '';
      if (event.partial && text) res.write(`data: ${JSON.stringify({delta: text})}\n\n`);
      if (event.errorMessage) res.write(`data: ${JSON.stringify({error: event.errorMessage})}\n\n`);
    }
    res.write('data: {"done": true}\n\n');
  } catch (err) {
    if (!controller.signal.aborted) res.write(`data: ${JSON.stringify({error: (err as Error).message})}\n\n`);
  }
  res.end();
});

export const server = app.listen(Number(process.env.PORT ?? 8080));
```

- Each request uses the same `Runner`; concurrency is per session, so do not run two turns on one session at once.
- Abort the run when the client disconnects so a cancelled request does not keep calling the model.
- `maxLlmCalls` bounds each turn's model calls (the default is 500).
- `event.errorMessage` carries model errors (quota, safety blocks) as events rather than exceptions.

## Persistence

| What | Development | Production |
|---|---|---|
| Sessions | `InMemorySessionService` | `DatabaseSessionService` (PostgreSQL, MySQL, MariaDB, SQL Server, SQLite) or `VertexAiSessionService` |
| Artifacts | `InMemoryArtifactService`, `FileArtifactService` | `GcsArtifactService` |
| Memory | `InMemoryMemoryService` | `VertexAiMemoryBankService`, `VertexAiRagMemoryService` |

`DatabaseSessionService` creates its tables on first use and checks the schema version on start. With several replicas, point them all at the same database; no session affinity is needed. Keep secrets out of session state: it is stored as is and can reach the model through instruction templates.

## Tool policies

`SecurityPlugin` puts one policy in front of every tool call in the app. A policy engine returns `ALLOW`, `CONFIRM` (pause for a person, through the same `adk_request_confirmation` flow as `requireConfirmation`) or `DENY` (the model is told the call was rejected and why):

```typescript
import {
  FunctionTool,
  InMemoryRunner,
  LlmAgent,
  PolicyOutcome,
  SecurityPlugin,
  getFunctionCalls,
  isFinalResponse,
  stringifyContent,
  type BasePolicyEngine,
  type PolicyCheckResult,
  type ToolCallPolicyContext,
} from '@google/adk';
import {z} from 'zod';

// One place to decide, for every tool call in the app: allow, ask a person, or refuse.
class ShopPolicy implements BasePolicyEngine {
  async evaluate({tool, toolArgs}: ToolCallPolicyContext): Promise<PolicyCheckResult> {
    if (tool.name === 'delete_customer') {
      return {outcome: PolicyOutcome.DENY, reason: 'Deleting customers is not allowed from chat.'};
    }
    if (tool.name === 'issue_refund' && Number(toolArgs['amount']) > 100) {
      return {outcome: PolicyOutcome.CONFIRM, reason: 'Refunds over 100 need approval.'};
    }
    return {outcome: PolicyOutcome.ALLOW};
  }
}

const issueRefund = new FunctionTool({
  name: 'issue_refund',
  description: 'Refund an order.',
  parameters: z.object({orderId: z.string(), amount: z.number()}),
  execute: async ({orderId, amount}) => ({status: 'refunded', orderId, amount}),
});
const deleteCustomer = new FunctionTool({
  name: 'delete_customer',
  description: 'Delete a customer record.',
  parameters: z.object({customerId: z.string()}),
  execute: async ({customerId}) => ({deleted: customerId}),
});

const agent = new LlmAgent({
  name: 'support',
  model: '%%model=gemini%%',
  instruction: 'Help customers with orders.',
  tools: [issueRefund, deleteCustomer],
});

const runner = new InMemoryRunner({agent, appName: 'shop', plugins: [new SecurityPlugin({policyEngine: new ShopPolicy()})]});
for await (const event of runner.runEphemeral({userId: 'u1', newMessage: {role: 'user', parts: [{text: 'Delete customer C9'}]}})) {
  for (const call of getFunctionCalls(event)) console.log('call:', call.name);
  if (isFinalResponse(event)) console.log(stringifyContent(event));
}
```

Without a `policyEngine`, `SecurityPlugin` uses `InMemoryPolicyEngine`, which allows everything; it exists so you can wire the plugin before writing the policy. Keep the policy in code you can test, and log denials.

Other defences worth having:

- Offer each agent only the tools it needs; `MCPToolset` filters and per-user [toolsets](/google-adk-guide/typescript/google_adk_typescript_comprehensive_guide/#toolsets) help.
- Validate tool arguments beyond their types (ownership of an order, amount limits) inside `execute`.
- `ToolCallIntegrityPlugin` stamps stored function calls with an HMAC so a tampered session store cannot replay or forge tool calls.

## Serving A2A safely

`toA2a` refuses to mount without authentication unless you pass `allowUnauthenticated: true`. Use `bearerTokenUserBuilder(token)` for a shared secret between services, keep the token in Secret Manager, and rotate it. On the client side, `RemoteA2AAgent` checks that the agent card's RPC URL uses HTTPS (plain HTTP only on loopback) and matches the origin the card came from. Both checks can be relaxed with `resolveAgentCardOptions`, but only do so for a reason you can write down. Examples are in the [comprehensive guide](/google-adk-guide/typescript/google_adk_typescript_comprehensive_guide/#a2a).

## Telemetry

ADK emits OpenTelemetry spans for invocations, agent runs, model calls, tool calls and workflow nodes. `maybeSetOtelProviders` installs the providers; call it once at start-up:

```typescript
import {LlmAgent, maybeSetOtelProviders} from '@google/adk';
import {getGcpExporters, getGcpResource} from '@google/adk/telemetry/gcp';
import {BatchSpanProcessor, ConsoleSpanExporter} from '@opentelemetry/sdk-trace-base';

// Call once at start-up, before any agent runs.
if (process.env.GOOGLE_CLOUD_PROJECT) {
  // Cloud Trace and Cloud Monitoring. npm install @google-cloud/opentelemetry-cloud-trace-exporter
  // @google-cloud/opentelemetry-cloud-monitoring-exporter
  maybeSetOtelProviders([await getGcpExporters({enableTracing: true, enableMetrics: true})], getGcpResource());
} else {
  // Anywhere else: any OpenTelemetry span processor. OTEL_EXPORTER_OTLP_ENDPOINT also works with no code.
  maybeSetOtelProviders([{spanProcessors: [new BatchSpanProcessor(new ConsoleSpanExporter())]}]);
}

export const rootAgent = new LlmAgent({name: 'assistant', model: '%%model=gemini%%', instruction: 'Be helpful.'});
```

- `getGcpExporters` (from `@google/adk/telemetry/gcp`) supports traces and metrics; it does not export logs. It finds the project from Application Default Credentials and returns no exporters, with a warning, when there is none.
- Setting `OTEL_EXPORTER_OTLP_ENDPOINT` (or the per-signal `_TRACES_`, `_METRICS_` and `_LOGS_` variants) adds an OTLP HTTP exporter with no code change.
- Spans include prompts, responses and tool arguments by default. Set `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=false` to replace them with `{}` when your trace store should not hold user data.
- `maybeSetOtelProviders` does not replace a provider that is already registered; under `adk web` the dev server registers its own first, so use its trace view there.
- `npm install @opentelemetry/sdk-trace-base` if you import span processors directly, as above.

## Evaluation

An evaluation file for `adk eval` is a JSON array of cases. Each case has a `name` and a list of turns, each with the user `query`, the tools you expect it to call (`expected_tool_use`) and a `reference` answer:

```json
[
  {
    "name": "order_status",
    "data": [
      {
        "query": "Where is order A123?",
        "expected_tool_use": [{"tool_name": "get_order", "tool_input": {"orderId": "A123"}}],
        "expected_intermediate_agent_responses": [],
        "reference": "Order A123 has shipped."
      }
    ]
  }
]
```

```bash
npx adk eval agent.ts orders.evalset.json --config_file_path test_config.json --print_detailed_results
```

`test_config.json` sets the pass thresholds: `{"criteria": {"tool_trajectory_avg_score": 1.0, "response_match_score": 0.8}}`, which are also the defaults. `tool_trajectory_avg_score` is 1 for a turn whose tool calls match the expected names and arguments exactly, and `response_match_score` is the ROUGE-1 F1 overlap between the reply and the reference, so set its threshold with the wording variation of a real model in mind. Add `:case_a,case_b` after a file name to run only those cases. The same runner is available in code as `runEvals`, which is how to call it from a test suite.

Evaluations call the real model, so they cost money and vary between runs; keep fast, deterministic tests with a scripted model ([Testing](/google-adk-guide/typescript/google_adk_typescript_comprehensive_guide/#testing)) for CI, and run evaluation sets before releases.

## Checklist

- Pin `@google/adk` and `@google/adk-devtools` versions, and `--adk_version` in deployments.
- Use Vertex AI with a service account in production rather than an API key.
- Persistent session and artifact services; secrets out of state.
- Authentication in front of the API server, and on any A2A endpoint.
- `maxLlmCalls`, timeouts (`AbortSignal.timeout`) and client-disconnect aborts on every run.
- A tool policy (`SecurityPlugin` or `requireConfirmation`) for tools that spend money or change data.
- Telemetry with message capture set deliberately.
- Scripted-model tests in CI, evaluation sets before release.

---

## Revision history

| Date | Version | Changes |
|---|---|---|
| 2026-10-24 | current | New page: dev tools commands and API server, `adk deploy cloud_run` and `agent_engine`, an Express server with streaming, persistence, `SecurityPlugin` policies, A2A authentication, telemetry and evaluation. Examples type-checked and run; command options checked against the dev tools' help and source. |

---
title: "Deploying TypeScript Agents to AgentCore Runtime"
description: "Deploy a Node.js Strands agent to Amazon Bedrock AgentCore Runtime with the AgentCore CLI, as a container image or as a NODE_22 code bundle, and invoke it from TypeScript."
framework: "amazon-bedrock-agents"
language: "typescript"
sidebar:
  label: "Deploying to Runtime"
  order: 3
---

AgentCore Runtime runs Node.js agents in the same session-isolated microVMs as Python ones. There are two ways to ship one: the **AgentCore CLI** (`@aws/agentcore`), which scaffolds the project, builds it and deploys it with CDK; or **by hand** with the AWS SDK for JavaScript control-plane client, as either a container image in ECR or a zipped code bundle on the managed `NODE_22` runtime.

> **Verified against:** `bedrock-agentcore` %%version=bedrock-agentcore-ts%%, `@strands-agents/sdk` %%version=@strands-agents/sdk%% and the AgentCore CLI (`@aws/agentcore`) %%version=@aws/agentcore%% (checked %%versions-checked%%). TypeScript examples compile under `tsc --strict`; the CLI commands were checked against its `--help` output and a generated project.

The server code itself is covered in [Runtime: BedrockAgentCoreApp](/amazon-bedrock-agents-guide/typescript/agentcore_typescript_guide/#runtime-bedrockagentcoreapp). The Python equivalents of the deployment steps are in the [AgentCore guide](/amazon-bedrock-agents-guide/python/bedrock_agentcore_comprehensive_guide/#getting-started) and the [production guide](/amazon-bedrock-agents-guide/python/bedrock_agents_production_guide/).

---

## Table of Contents

1. [The Runtime contract](#the-runtime-contract)
2. [Option 1: the AgentCore CLI](#option-1-the-agentcore-cli)
3. [Option 2: container image](#option-2-container-image)
4. [Option 3: direct code deployment](#option-3-direct-code-deployment)
5. [Invoking a deployed runtime](#invoking-a-deployed-runtime)
6. [Production checklist](#production-checklist)

---

## The Runtime contract

Whatever the packaging, the process must:

- listen on `0.0.0.0:8080` (HTTP protocol; MCP uses 8000 and A2A 9000);
- answer `GET /ping` with `{"status": "Healthy"}` or `{"status": "HealthyBusy"}`;
- answer `POST /invocations`, reading the session from `X-Amzn-Bedrock-AgentCore-Runtime-Session-Id`;
- run on `linux/arm64` if it is a container image.

`BedrockAgentCoreApp` from `bedrock-agentcore/runtime` does all of this, so a typical `main.ts` ends with `app.run()`. You can also write the two routes yourself with Express or Fastify; the official Strands TypeScript deployment sample does exactly that.

---

## Option 1: the AgentCore CLI

The CLI is an npm package (Node.js 20.12 or later). It can scaffold a TypeScript agent from a template, or register your own code as a runtime.

```bash
npm install -g @aws/agentcore

agentcore create --name OrderSupport --template agent-typescript-strands
cd OrderSupport

agentcore dev                                   # local server with hot reload (tsx watch)
agentcore invoke --local --runtime agent --payload '{"prompt": "What can you do?"}'   # in a second terminal
agentcore deploy                                # CDK deploy to your AWS account
agentcore invoke --runtime agent --payload '{"prompt": "Where is order ORD-7?", "actorId": "user-123"}'
```

`agentcore create --help` lists the TypeScript templates: `agent-typescript-strands` (Strands, with AgentCore Memory) and `agent-typescript-vercel` (Vercel AI SDK on Bedrock). Run `agentcore create` with no flags to choose a different model provider (Anthropic, OpenAI, an OpenAI-compatible endpoint or Gemini); the wizard keeps an API key in `agentcore/.env.local` and provisions it as an AgentCore Identity credential on deploy.

To use code you already have, create an empty project and register a runtime:

```bash
agentcore create --name OrderSupport
cd OrderSupport
agentcore add runtime --name agent --language typescript
```

`add runtime` writes no code. It creates `app/agent/` with a `README.md` and `AGENTS.md` that state what the folder must provide: a `main.ts` entry (TypeScript is always bundled from `main.ts`), a `package.json`, and a server that follows the Runtime contract (the `BedrockAgentCoreApp` examples in the [AgentCore SDK guide](/amazon-bedrock-agents-guide/typescript/agentcore_typescript_guide/#runtime-bedrockagentcoreapp) do). `engines.node` in `package.json` constrains the managed runtime version. `--protocol` selects `HTTP` (the default), `MCP`, `A2A` or `AGUI`.

The generated project looks like this:

```text
OrderSupport/
├── agentcore/
│   ├── agentcore.json        # project definition: runtimes, memories, credentials, gateways...
│   ├── aws-targets.json      # account and Region to deploy to
│   ├── .env.local            # local-only secrets for agentcore dev
│   └── cdk/                  # generated CDK app
└── app/agent/
    ├── main.ts               # BedrockAgentCoreApp + Strands agent (streams text deltas)
    ├── model/load.ts         # model provider setup
    ├── memory/memory.ts      # AgentCore Memory stores for Strands
    ├── package.json
    └── tsconfig.json
```

The runtime entry in `agentcore/agentcore.json` records how the agent is built:

```json
{
  "name": "agent",
  "build": "CodeZip",
  "modelProvider": "Bedrock",
  "entrypoint": "main.js",
  "codeLocation": "app/agent",
  "runtimeVersion": "NODE_22",
  "protocol": "HTTP"
}
```

`CodeZip` (the default) has the CLI install dependencies, bundle the code and upload it to the managed Node.js runtime; adding a `Dockerfile` to the folder switches the runtime to a container build. Network mode, lifecycle limits, the request-header allowlist and the authorizer are options of `agentcore add runtime` (`--network-mode VPC` with `--network-config`, `--lifecycle-configuration`, `--request-header-allowlist`, `--authorizer-type CUSTOM_JWT`).

The template also declares an AgentCore Memory resource, `agentMemory`, and the generated `memory/memory.ts` reads its ID from `MEMORY_AGENTMEMORY_ID`, which the CLI injects on deploy. Do not hard-code it.

Two things to change in the generated code before you ship it:

1. **Know which model it uses.** The template's `model/load.ts` sets `global.anthropic.claude-sonnet-5-5`, a global cross-Region inference profile. Choose the profile you want (for example `new BedrockModel({ modelId: 'us.anthropic.%%model=anthropic%%' })` to keep inference in US Regions) and enable model access in the deployment Region.
2. **Check the dependency ranges.** The template's `package.json` pins old minor versions with tilde ranges (for example `bedrock-agentcore` `~0.3.0` and `@strands-agents/sdk` `~1.5.0`), which do not admit newer releases. Set them to the versions you tested, such as `bedrock-agentcore@%%version=bedrock-agentcore-ts%%` and `@strands-agents/sdk@%%version=@strands-agents/sdk%%`, and commit the lock file. The template imports the memory helper from `bedrock-agentcore/experimental/memory/strands`; in newer SDK releases the path is `bedrock-agentcore/memory/strands`.

More resources are added to the project and deployed together. For example, `agentcore add memory` declares an AgentCore Memory resource; on deploy the CLI grants the runtime access and injects the memory ID as `MEMORY_<NAME>_ID`. `agentcore add credentials api-key|oauth` and `agentcore add gateway` work the same way. After any `add` or `remove`, run `agentcore deploy` again. `agentcore logs runtime`, `agentcore traces runtime list` and `agentcore status` cover day-two operations. A project created with a 0.x CLI is converted with `agentcore migrate`.

---

## Option 2: container image

A container gives you full control over the Node.js version and system packages. This Dockerfile is a starting point for a Node.js agent that compiles TypeScript at build time. (With the CLI, adding a `Dockerfile` to a runtime's folder switches it to a container build.)

```dockerfile
FROM --platform=linux/arm64 public.ecr.aws/docker/library/node:22-slim

WORKDIR /app
ENV NODE_ENV=production \
    DOCKER_CONTAINER=1

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npx tsc && npm prune --omit=dev

RUN useradd -m -u 1000 agent
USER agent

EXPOSE 8080
CMD ["node", "dist/main.js"]
```

`DOCKER_CONTAINER=1` matters for A2A servers built with `serveA2A`, which otherwise bind to loopback because Runtime does not create `/.dockerenv`; `BedrockAgentCoreApp` binds `0.0.0.0` regardless. Build for arm64 and push to ECR:

```bash
export AWS_REGION=us-west-2
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export REPO=order-support

aws ecr create-repository --repository-name $REPO --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

docker buildx build --platform linux/arm64 \
  -t $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:v1 --push .
```

Then create the runtime with the control-plane client. The execution role needs a trust policy for `bedrock-agentcore.amazonaws.com`, ECR pull, CloudWatch Logs, X-Ray and `bedrock:InvokeModel*` permissions (see [Security and IAM](/amazon-bedrock-agents-guide/python/bedrock_security_iam_examples/)).

```typescript
import {
  BedrockAgentCoreControlClient,
  CreateAgentRuntimeCommand,
  GetAgentRuntimeCommand,
} from '@aws-sdk/client-bedrock-agentcore-control'

const control = new BedrockAgentCoreControlClient({ region: 'us-west-2' })

const created = await control.send(
  new CreateAgentRuntimeCommand({
    agentRuntimeName: 'order_support',
    agentRuntimeArtifact: {
      containerConfiguration: {
        containerUri: '111122223333.dkr.ecr.us-west-2.amazonaws.com/order-support:v1',
      },
    },
    roleArn: 'arn:aws:iam::111122223333:role/AgentCoreRuntimeRole',
    networkConfiguration: { networkMode: 'PUBLIC' },
    protocolConfiguration: { serverProtocol: 'HTTP' },
    lifecycleConfiguration: { idleRuntimeSessionTimeout: 900, maxLifetime: 28800 },
    requestHeaderConfiguration: {
      requestHeaderAllowlist: ['X-Amzn-Bedrock-AgentCore-Runtime-Custom-Actor-Id'],
    },
    environmentVariables: { NODE_OPTIONS: '--enable-source-maps' },
  })
)

console.log(created.agentRuntimeArn, created.status)

// Poll until the runtime is READY
let status = created.status
while (status === 'CREATING' || status === 'UPDATING') {
  await new Promise((resolve) => setTimeout(resolve, 10_000))
  const current = await control.send(new GetAgentRuntimeCommand({ agentRuntimeId: created.agentRuntimeId }))
  status = current.status
}
console.log('final status:', status)
```

To roll out a new image, push a new tag and call `UpdateAgentRuntimeCommand` with the new `containerUri`; each update creates a new runtime version, and the `DEFAULT` endpoint moves to it.

---

## Option 3: direct code deployment

Without a container, upload a zip of the built application to S3 and point the runtime at it with `codeConfiguration` and the `NODE_22` managed runtime. The zip must contain the compiled JavaScript and its production `node_modules`, built for Linux arm64 if any dependency has native code.

```bash
npm ci && npx tsc && npm prune --omit=dev
zip -qr deployment.zip dist node_modules package.json
aws s3 cp deployment.zip s3://my-agent-artifacts/order-support/deployment.zip
```

```typescript
import { BedrockAgentCoreControlClient, CreateAgentRuntimeCommand } from '@aws-sdk/client-bedrock-agentcore-control'

const control = new BedrockAgentCoreControlClient({ region: 'us-west-2' })

const response = await control.send(
  new CreateAgentRuntimeCommand({
    agentRuntimeName: 'order_support_zip',
    agentRuntimeArtifact: {
      codeConfiguration: {
        code: { s3: { bucket: 'my-agent-artifacts', prefix: 'order-support/deployment.zip' } },
        runtime: 'NODE_22',
        entryPoint: ['dist/main.js'],
      },
    },
    roleArn: 'arn:aws:iam::111122223333:role/AgentCoreRuntimeRole',
    networkConfiguration: { networkMode: 'PUBLIC' },
  })
)

console.log(response.agentRuntimeArn)
```

The execution role also needs `s3:GetObject` on the artefact.

---

## Invoking a deployed runtime

Callers use the data-plane client `@aws-sdk/client-bedrock-agentcore`. The payload is bytes; `runtimeSessionId` must be at least 33 characters, and reusing it routes the call to the same session microVM. For a streaming handler, ask for Server-Sent Events with `accept`.

```typescript
import { randomUUID } from 'node:crypto'
import { BedrockAgentCoreClient, InvokeAgentRuntimeCommand } from '@aws-sdk/client-bedrock-agentcore'

const client = new BedrockAgentCoreClient({ region: 'us-west-2' })
const sessionId = `order-support-${randomUUID()}` // 50 characters

const response = await client.send(
  new InvokeAgentRuntimeCommand({
    agentRuntimeArn: 'arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/order_support-abc123',
    qualifier: 'DEFAULT',
    runtimeSessionId: sessionId,
    contentType: 'application/json',
    accept: 'text/event-stream', // omit (or use application/json) for non-streaming handlers
    payload: new TextEncoder().encode(JSON.stringify({ prompt: 'Where is order ORD-7?' })),
  })
)

if (!response.response) throw new Error('empty response')

if (response.contentType?.startsWith('text/event-stream')) {
  // Read SSE frames as they arrive
  const decoder = new TextDecoder()
  for await (const chunk of response.response.transformToWebStream()) {
    process.stdout.write(decoder.decode(chunk, { stream: true }))
  }
} else {
  console.log(JSON.parse(await response.response.transformToString()))
}
```

Callers need `bedrock-agentcore:InvokeAgentRuntime` on the runtime (and its endpoint). For end-user access without AWS credentials, configure a JWT `authorizerConfiguration` on the runtime and call the HTTPS endpoint with a bearer token; `RuntimeClient` in `bedrock-agentcore/runtime` can produce presigned WebSocket URLs for browser clients.

---

## Production checklist

- **Pin versions.** Both SDKs are moving quickly, and `bedrock-agentcore` is pre-1.0; pin exact versions and commit `package-lock.json`.
- **Set the model ID explicitly.** The Strands default model may change between releases; use a current Claude inference profile such as `us.anthropic.%%model=anthropic%%` and make sure model access is enabled in the deployment Region.
- **Validate input.** Give `BedrockAgentCoreApp` a Zod `requestSchema` and pass only the prompt text to the agent.
- **One agent per session.** Runtime gives each session its own microVM; keep one `Agent` per `sessionId` and never share an agent between concurrent requests.
- **Flush memory writes.** With AgentCore Memory, `await agent.memoryManager?.flush()` at the end of each invocation so writes are not lost when an idle session is reclaimed.
- **Keep busy sessions alive.** Track background work with `app.asyncTask()` or `addAsyncTask()` so `/ping` reports `HealthyBusy`.
- **Bound sessions.** Set `idleRuntimeSessionTimeout` and `maxLifetime` in `lifecycleConfiguration`; defaults are 15 minutes idle and 8 hours maximum.
- **Least-privilege roles.** Scope `bedrock:InvokeModel*` to the models and inference profiles you use, and grant Memory, Code Interpreter, Browser and Identity actions only where needed.
- **Sandboxed tools.** Prefer the managed Code Interpreter and Browser to the vended `bash`, `shell` or `python-repl` tools, which run inside the runtime microVM.
- **Observability.** Runtime sends stdout and stderr to CloudWatch Logs. For traces, configure OpenTelemetry for Strands with `setupTracer()` from `@strands-agents/sdk/telemetry` and an OTLP exporter, or use `agentcore traces` with a CLI-deployed project; the TypeScript AgentCore SDK has no observability module yet.

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-21 | `bedrock-agentcore` 0.4.6 · AgentCore CLI 0.31.1 | New page. |

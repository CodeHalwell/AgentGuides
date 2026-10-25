---
title: "Amazon Bedrock Agents: Production Deployment Guide"
description: "Running agents in production on Amazon Bedrock AgentCore: deployment targets and CI/CD with the AgentCore CLI, versions and endpoints, inbound and outbound identity, memory, observability, evaluations, least-privilege IAM and cost. A short section covers Agents Classic."
framework: "amazon-bedrock-agents"
language: "python"
sidebar:
  label: "Production"
  order: 10
---

> **Verified against:** AgentCore CLI (`@aws/agentcore`) %%version=@aws/agentcore%%, `bedrock-agentcore` %%version=bedrock-agentcore%% and `boto3` %%version=boto3%% (checked %%versions-checked%%). CLI commands were run against a generated project (`create`, `add`, `export`) or checked against `--help` (`deploy`, `invoke`, `logs`, `traces`, `eval`). boto3 calls were validated against the service models.

This page is about **AgentCore**: code-defined agents on Runtime and managed harnesses. Writing the agent is covered in the [Strands Agents guide](/strands-agents-guide/python/) and the [AgentCore guide](/amazon-bedrock-agents-guide/python/bedrock_agentcore_comprehensive_guide/). Existing Agents Classic deployments are covered briefly [at the end](#agents-classic-in-production).

## Deployment options

| Route | What it does | Use it when |
|-------|--------------|-------------|
| **AgentCore CLI** (`agentcore deploy`) | Synthesises and deploys a CDK app from `agentcore/agentcore.json`: runtimes, harnesses, endpoints, memories, gateways, credentials, evaluators | The default. Your project, IAM roles and resources live in one config file |
| **CDK constructs** (`@aws/agentcore-cdk`) | The constructs the CLI generates, used in your own CDK app | AgentCore is part of a larger CDK estate |
| **boto3 / CloudFormation** (`create_agent_runtime`, `AWS::BedrockAgentCore::Runtime`) | Individual resources | You already manage infrastructure another way |

A code runtime is built either as a **CodeZip** (the default: your source and dependencies zipped and run on a managed `PYTHON_3_x` or `NODE_22` runtime) or as a **container** (the CLI builds your `Dockerfile`; Runtime runs ARM64 images). Add a `Dockerfile` to the runtime's folder to switch. Choose a container when you need system packages or a custom base image.

## Environments and CI/CD

Deployment targets (account and Region) live in `agentcore/aws-targets.json`. The first `agentcore deploy` creates a target called `default` from your current credentials and Region; add more entries for environments:

```json
[
  { "name": "staging", "account": "111122223333", "region": "us-west-2" },
  { "name": "prod", "account": "444455556666", "region": "us-west-2" }
]
```

`agentcore build` builds the deployable artifacts without touching AWS, which is the right pre-deploy check in a pull request. `agentcore deploy --target staging` deploys, and `agentcore status --target staging` shows what exists. In CI, pass flags and the commands run without prompts. `--json` makes output machine-readable, and `--region` overrides the Region. (In CLI 1.0, `--yes` on `deploy` only confirms removing a target's stack when the project declares nothing to deploy.) A GitHub Actions workflow using OIDC role assumption:

```yaml
name: deploy-agent
on:
  push:
    branches: [main]

permissions:
  id-token: write
  contents: read

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: npm install -g @aws/agentcore && pip install uv

      - run: agentcore build

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::111122223333:role/agent-deploy-staging
          aws-region: us-west-2
      - run: agentcore deploy --target staging --json
      - name: Smoke test
        run: agentcore invoke --runtime SalesAnalyst --target staging --payload '{"prompt":"Reply with the word OK."}' --json | tee smoke.json

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::444455556666:role/agent-deploy-prod
          aws-region: us-west-2
      - run: agentcore deploy --target prod --json
```

Pin the CLI version in CI (`npm install -g @aws/agentcore@%%version=@aws/agentcore%%`) so a new release cannot change a pipeline. Put an approval gate (a GitHub environment with required reviewers) on the production step, and run your unit tests for tools and entrypoint logic before any deploy. Those tests need no AWS access, because the entrypoint is a plain function.

**Deploy-role permissions.** The CLI deploys through CloudFormation and creates IAM roles, S3 objects, ECR repositories and CodeBuild projects. AWS's [CLI policy](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-permissions.html#runtime-permissions-cli) is intended for development; for CI, pre-bootstrap CDK in the target account and scope the deploy role to the project's stack (`AgentCore-<project>-<target>`) and the resources it declares. `agentcore deploy` also tries to enable CloudWatch Transaction Search and skips that step if the role lacks the X-Ray, CloudWatch Logs and Application Signals permissions it needs (AWS lists them on the page above).

### A smoke test that actually checks something

`agentcore invoke` in CI fails on transport errors, but it does not tell you whether the agent behaved. A small boto3 check that fails the build on a bad answer:

```python
import json
import sys
import uuid

import boto3


def smoke_test(runtime_arn: str, qualifier: str = "DEFAULT") -> None:
    client = boto3.client("bedrock-agentcore")
    response = client.invoke_agent_runtime(
        agentRuntimeArn=runtime_arn,
        qualifier=qualifier,
        runtimeSessionId=f"smoke-{uuid.uuid4()}",  # at least 33 characters
        payload=json.dumps({"prompt": "Reply with exactly the word OK."}),
    )
    if response["statusCode"] != 200:
        sys.exit(f"runtime returned HTTP {response['statusCode']}")
    body = response["response"].read().decode()
    if "OK" not in body:
        sys.exit(f"unexpected answer: {body[:200]}")
    print("smoke test passed")


if __name__ == "__main__":
    smoke_test(sys.argv[1], *sys.argv[2:3])
```

For behavioural checks beyond "it answers", use [evaluations](#evaluations).

## Versions and endpoints

Every deploy that changes a runtime creates a new immutable **version**. The `DEFAULT` endpoint follows the latest version. Named endpoints pin a version, so callers that use `qualifier="prod"` only move when you move the endpoint.

With the CLI, the endpoint is part of the project config:

```bash
agentcore add runtime-endpoint --runtime SalesAnalyst --name prod --version 3
agentcore deploy --target prod
# promote later: change "endpoints": {"prod": {"version": 4}} in agentcore.json and deploy again
```

With boto3, the same blue/green move:

```python
import boto3

control = boto3.client("bedrock-agentcore-control")
RUNTIME_ID = "SalesAnalyst-abc123"

control.create_agent_runtime_endpoint(agentRuntimeId=RUNTIME_ID, name="canary", agentRuntimeVersion="4")
# ... run the smoke test and evaluations with qualifier="canary" ...
control.update_agent_runtime_endpoint(agentRuntimeId=RUNTIME_ID, endpointName="prod", agentRuntimeVersion="4")
# roll back by pointing "prod" at "3" again
```

For a gradual rollout between two variants, the CLI has A/B tests over configuration bundles or gateway targets (`agentcore eval ab-test config-based run` and `agentcore eval ab-test target-based run`, with `get`, `pause`, `resume` and `stop` to manage them). Harnesses have the same model: `agentcore harness version list` and `agentcore harness endpoint create|update` pin and move versions.

## Sessions, limits and scaling

- **Sessions.** A `runtimeSessionId` maps to one microVM with its own memory and filesystem. Reuse the ID for a conversation and use a new one per user conversation. Never share a session between users.
- **Lifecycle.** `idleRuntimeSessionTimeout` (default 900 seconds) and `maxLifetime` (default 28,800 seconds) bound how long a session lives and bills. The API accepts 60 to 1,209,600 seconds for each. Set them with `--lifecycle-configuration` on `agentcore add runtime` or in `lifecycleConfiguration` on `create_agent_runtime`. Long background work should report `HealthyBusy` through `@app.async_task` (see [Resource management](/amazon-bedrock-agents-guide/python/bedrock_agentcore_comprehensive_guide/#resource-management)).
- **Quotas.** Concurrent sessions, invocation rates and payload sizes are per account and Region; check [AgentCore quotas](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/bedrock-agentcore-limits.html) and request increases before a launch rather than load-testing into them. Bedrock model quotas (tokens per minute) are usually the first limit an agent hits.
- **Agent state.** Build the agent once per session, not per request, and keep anything that must survive a session ending in [Memory](#memory) or your own store, not in process memory.

## Identity

**Inbound.** IAM (SigV4) suits service-to-service callers. For end users, configure a JWT authorizer on the runtime (`authorizerConfiguration.customJWTAuthorizer`, shown in the [AgentCore guide](/amazon-bedrock-agents-guide/python/bedrock_agentcore_comprehensive_guide/#inbound-authentication)). JWT callers invoke over HTTPS with a bearer token instead of the SigV4 SDK call:

```python
import json
import urllib.parse
import urllib.request
import uuid


def invoke_with_jwt(region: str, runtime_arn: str, token: str, prompt: str, qualifier: str = "DEFAULT") -> str:
    url = (
        f"https://bedrock-agentcore.{region}.amazonaws.com/runtimes/"
        f"{urllib.parse.quote(runtime_arn, safe='')}/invocations?qualifier={qualifier}"
    )
    request = urllib.request.Request(
        url,
        data=json.dumps({"prompt": prompt}).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": str(uuid.uuid4()),
        },
    )
    with urllib.request.urlopen(request) as response:
        return response.read().decode()
```

From the CLI, `agentcore invoke --runtime SalesAnalyst --bearer-token "$TOKEN" --payload '{"prompt": "..."}'` does the same. Test both a valid token and an expired or wrong-audience token before launch. The second must fail with 401/403, not reach your code.

**Outbound.** Tools that call third-party APIs should get credentials from AgentCore Identity (`@requires_access_token`, `@requires_api_key`, or `agentcore add credentials`), not from environment variables baked into the image. With JWT inbound auth, user-federated OAuth tokens are scoped to the calling user. AWS recommends that production execution roles **deny** `bedrock-agentcore:GetWorkloadAccessTokenForUserId`, which issues tokens for a caller-supplied user ID without verifying a token, and allow only `GetWorkloadAccessTokenForJWT`.

**Payload trust.** Runtime passes the payload to your entrypoint unchanged. Validate that `prompt` is a string, and strip `toolUse` blocks from any message lists before they reach the agent. Take user and tenant IDs from the verified token or `context`, not from the payload.

## Memory

Add memory to the project rather than creating it by hand, so it is versioned with the agent:

```bash
agentcore add memory --name SalesMemory --strategies SEMANTIC,SUMMARIZATION --event-expiry-duration 30
```

- Scope every read and write by **actor ID** (the end user) and **session ID**, both taken from the verified identity. Namespace templates such as `/users/{actorId}/facts` keep users' long-term records apart.
- Each long-term strategy runs extraction on new events, which costs model tokens. Enable only the strategies you query.
- Set the event expiry to your retention policy, and use a customer-managed KMS key (`encryptionKeyArn`) where your data classification requires it.
- With Strands, `AgentCoreMemorySessionManager` writes and reads memory for you (see the [Strands guide](/strands-agents-guide/python/strands_comprehensive_guide/#agentcore-memory)).

## Observability

1. **CloudWatch Transaction Search** must be enabled once per account and Region so spans are stored and appear in the GenAI Observability dashboards. `agentcore deploy` enables it for you; opt out with `agentcore config transactionSearch false` if your organisation manages it centrally.
2. CLI-generated Python projects already depend on `aws-opentelemetry-distro`. For hand-built containers, see [instrumenting an agent](/amazon-bedrock-agents-guide/python/bedrock_agents_observability_python/).
3. Use the CLI to look at a deployed agent:

```bash
agentcore status --target prod
agentcore logs runtime --name SalesAnalyst --since 1h --level error
agentcore traces runtime list --name SalesAnalyst --since 30m
agentcore traces runtime get <traceId>      # writes <traceId>.json
```

`agentcore logs runtime --tail` follows new records, and `agentcore logs harness` and `agentcore traces harness` do the same for harnesses. Alarm on Runtime errors and latency, on Bedrock throttling for the models you use, and on any business metric your tools publish.

## Evaluations

AgentCore Evaluations scores agent traces with built-in or custom evaluators (LLM-as-a-judge or a Lambda function). The built-in evaluators include `Builtin.Correctness`, `Builtin.Helpfulness`, `Builtin.Faithfulness`, `Builtin.GoalSuccessRate`, `Builtin.ToolSelectionAccuracy`, `Builtin.ToolParameterAccuracy`, `Builtin.Coherence`, `Builtin.Conciseness`, `Builtin.InstructionFollowing`, `Builtin.ResponseRelevance`, `Builtin.Refusal`, `Builtin.Harmfulness` and `Builtin.Stereotyping`, plus trajectory matchers for tool-call order. There are two ways to run them.

**Continuous (online) evaluation** samples live traffic. Declare it in the project so it is deployed with the agent:

```bash
agentcore add online-eval --name SalesQuality --agent SalesAnalyst \
  --evaluators Builtin.Helpfulness Builtin.Correctness --sampling-rate 10
agentcore deploy --target prod
```

`--sampling-rate` is a percentage of sessions (0.01 to 100). Evaluation is enabled when the config deploys; pass `--enable-on-create false` to add it paused. `agentcore eval online-eval pause|resume|list|get|logs` manages it afterwards.

**On-demand evaluation** scores sessions that already exist, for example the last day of traffic, or one session against a known answer:

```bash
agentcore eval ondemand evaluate --agent <runtime-id> \
  --evaluators Builtin.GoalSuccessRate --lookback-days 1

agentcore eval ondemand evaluate --agent <runtime-id> \
  --evaluators Builtin.Correctness --session-ids <session-id> \
  --ground-truth '[{"context":{"spanContext":{"sessionId":"<session-id>","traceId":"<trace-id>"}},"expectedResponse":{"text":"Q4 revenue rose 12%"}}]'
```

From Python, `EvaluationClient` in the SDK reads the session's spans from CloudWatch and calls the evaluation API. Use it in tests or notebooks:

```python
from datetime import timedelta

from bedrock_agentcore.evaluation import EvaluationClient

client = EvaluationClient(region_name="us-west-2")

results = client.run(
    evaluator_ids=["Builtin.Helpfulness", "Builtin.Correctness"],
    session_id="SESSION_ID",
    agent_id="SalesAnalyst-abc123",     # the runtime ID; its log group is derived from it
    look_back_time=timedelta(days=2),
)

for result in results:
    print(result["evaluatorId"], result.get("value"), result.get("explanation"))
```

`agentcore add evaluator llm-as-a-judge` defines a custom judge (with `--level SESSION|TRACE|TOOL_CALL`, instructions and a rating scale such as `pass-fail` or `1-5-quality`), and `add evaluator code-based` uses a Lambda function. For a pre-promotion gate, replay a stored dataset against the `canary` endpoint with `agentcore eval ondemand simulate` (or `eval batch-evaluation simulate` for a service-side batch), and move `prod` only if the scores hold. `agentcore eval dataset` creates and versions the datasets.

## IAM least privilege

Three kinds of principal need permissions. Keep them separate:

| Principal | Needs | Scope to |
|-----------|-------|----------|
| **Deploy role** (CI) | CloudFormation, IAM role creation for `*BedrockAgentCore*` roles, S3/ECR/CodeBuild for `bedrock-agentcore-*` | One account per environment |
| **Execution role** (assumed by Runtime) | Logs, X-Ray, `cloudwatch:PutMetricData` (namespace `bedrock-agentcore`), workload-identity tokens, `bedrock:InvokeModel*` on the models used, and the Memory, Gateway, Lambda or S3 resources the agent's tools reach | The specific inference profiles and foundation models, and the runtime's own log group |
| **Callers** | `bedrock-agentcore:InvokeAgentRuntime` (plus `InvokeHarness` for a harness) | The runtime or harness ARN, and endpoint |

The trust policy on the execution role should allow only `bedrock-agentcore.amazonaws.com`, conditioned on your account (`aws:SourceAccount`) and AgentCore resources (`aws:SourceArn`). The [IAM examples](/amazon-bedrock-agents-guide/python/bedrock_security_iam_examples/) page has the full policies. The CLI generates execution roles for you. Review them before production, because the defaults favour getting started over least privilege.

## Cost

AgentCore bills by consumption: Runtime by active CPU and memory per session-second, Memory by events and extracted records, Gateway by calls and indexed tools, Code Interpreter and Browser by session time. There is no separate harness orchestration charge. Model tokens are billed by Bedrock as usual. See [AgentCore pricing](https://aws.amazon.com/bedrock/agentcore/pricing/) and filter Cost Explorer on the AgentCore service.

- Short idle timeouts stop idle sessions accruing Runtime charges.
- Use a small model (Haiku class) for routing and extraction steps, and prompt caching (`CacheConfig` on Strands `BedrockModel`) for long, stable system prompts and tool lists.
- Enable only the memory strategies you read.
- Always stop Code Interpreter and Browser sessions (the `code_session` / `browser_session` context managers do this).
- Set `maxIterations`, `maxTokens` and `timeoutSeconds` on a harness, and timeouts and node limits on Strands multi-agent runs, so a looping agent fails cheaply.

## Production checklist

- [ ] Entrypoint validates the payload; identity comes from the token or `context`, not the payload
- [ ] Tools and entrypoint covered by unit tests; smoke test and evaluations run against a staging or canary endpoint
- [ ] `prod` endpoint pinned to a tested version; rollback is "move the endpoint back"
- [ ] Execution role scoped to specific models and resources; trust policy conditioned on account and source ARN
- [ ] Inbound auth chosen and tested with bad tokens; `GetWorkloadAccessTokenForUserId` denied where JWTs are available
- [ ] Lifecycle timeouts and harness or multi-agent limits set
- [ ] Transaction Search enabled; alarms on errors, latency and throttling
- [ ] Memory expiry, strategies and KMS reviewed; actor and session IDs scoped per user
- [ ] Quotas checked for expected peak concurrency

## Agents Classic in production

For existing Agents Classic deployments (maintenance mode since 30 July 2026):

- **Versions and aliases.** Test against `DRAFT` (with `TSTALIASID`), then create an alias, which snapshots a numbered version. Roll forward or back by pointing the alias at another version with `update_agent_alias(agentId=..., agentAliasId=..., agentAliasName=..., routingConfiguration=[{"agentVersion": "N"}])`. Callers always use an alias ID, never `DRAFT`.
- **Smoke test.** After each change, invoke the alias and check the answer, as below.
- **Monitoring.** `AWS/Bedrock/Agents` CloudWatch metrics and `enableTrace=True` traces; see the [observability page](/amazon-bedrock-agents-guide/python/bedrock_agents_observability_python/#a-enable-traces-and-logging-for-bedrock-agents-classic).
- **IAM.** A service role trusted by `bedrock.amazonaws.com` for your account's agents, plus `bedrock:InvokeAgent` on the alias ARN for callers; see the [IAM examples](/amazon-bedrock-agents-guide/python/bedrock_security_iam_examples/).
- **Plan the move.** No new models will reach Classic. See [Migrating from Bedrock Agents Classic](/amazon-bedrock-agents-guide/python/bedrock_agentcore_comprehensive_guide/#migrating-from-bedrock-agents-classic).

```python
import uuid

import boto3


def classic_smoke_test(agent_id: str, alias_id: str) -> None:
    runtime = boto3.client("bedrock-agent-runtime")
    response = runtime.invoke_agent(
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId=str(uuid.uuid4()),
        inputText="Reply with exactly the word OK.",
    )
    answer = "".join(e["chunk"]["bytes"].decode() for e in response["completion"] if "chunk" in e)
    if "OK" not in answer:
        raise SystemExit(f"unexpected answer: {answer[:200]}")
```

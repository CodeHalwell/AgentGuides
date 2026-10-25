---
title: "Amazon Bedrock Agents: IAM Examples"
description: "Least-privilege IAM policies for AgentCore Runtime and harness execution roles and callers, and for Bedrock Agents Classic service roles, knowledge bases and callers."
framework: "amazon-bedrock-agents"
language: "python"
sidebar:
  label: "IAM examples"
  order: 8
---

Least-privilege starting points. Replace `REGION`, `ACCOUNT`, IDs and names with your own, and prefer specific ARNs over wildcards. The AgentCore policies follow AWS's [Runtime permissions](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-permissions.html) and [harness security](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness-security.html) pages, scoped down further. The [production guide](/amazon-bedrock-agents-guide/python/bedrock_agents_production_guide/#iam-least-privilege) explains which principal needs which.

## AgentCore Runtime: execution role

Trust policy. Only AgentCore, only for your account's resources:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "bedrock-agentcore.amazonaws.com" },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": { "aws:SourceAccount": "ACCOUNT" },
        "ArnLike": { "aws:SourceArn": "arn:aws:bedrock-agentcore:REGION:ACCOUNT:*" }
      }
    }
  ]
}
```

Permissions for an agent named `AGENT_NAME` that calls one Claude inference profile. Drop the ECR statements for CodeZip deployments, and add statements for any Memory, Gateway or Lambda resources the agent uses:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "EcrPull",
      "Effect": "Allow",
      "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
      "Resource": "arn:aws:ecr:REGION:ACCOUNT:repository/AGENT_REPOSITORY"
    },
    {
      "Sid": "EcrToken",
      "Effect": "Allow",
      "Action": "ecr:GetAuthorizationToken",
      "Resource": "*"
    },
    {
      "Sid": "Logs",
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup", "logs:DescribeLogStreams"],
      "Resource": "arn:aws:logs:REGION:ACCOUNT:log-group:/aws/bedrock-agentcore/runtimes/*"
    },
    {
      "Sid": "LogStreams",
      "Effect": "Allow",
      "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
      "Resource": "arn:aws:logs:REGION:ACCOUNT:log-group:/aws/bedrock-agentcore/runtimes/*:log-stream:*"
    },
    {
      "Sid": "DescribeLogGroups",
      "Effect": "Allow",
      "Action": "logs:DescribeLogGroups",
      "Resource": "arn:aws:logs:REGION:ACCOUNT:log-group:*"
    },
    {
      "Sid": "Tracing",
      "Effect": "Allow",
      "Action": ["xray:PutTraceSegments", "xray:PutTelemetryRecords", "xray:GetSamplingRules", "xray:GetSamplingTargets"],
      "Resource": "*"
    },
    {
      "Sid": "Metrics",
      "Effect": "Allow",
      "Action": "cloudwatch:PutMetricData",
      "Resource": "*",
      "Condition": { "StringEquals": { "cloudwatch:namespace": "bedrock-agentcore" } }
    },
    {
      "Sid": "WorkloadIdentity",
      "Effect": "Allow",
      "Action": ["bedrock-agentcore:GetWorkloadAccessToken", "bedrock-agentcore:GetWorkloadAccessTokenForJWT"],
      "Resource": [
        "arn:aws:bedrock-agentcore:REGION:ACCOUNT:workload-identity-directory/default",
        "arn:aws:bedrock-agentcore:REGION:ACCOUNT:workload-identity-directory/default/workload-identity/AGENT_NAME-*"
      ]
    },
    {
      "Sid": "DenyUnverifiedUserTokens",
      "Effect": "Deny",
      "Action": "bedrock-agentcore:GetWorkloadAccessTokenForUserId",
      "Resource": "*"
    },
    {
      "Sid": "InvokeModel",
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
      "Resource": [
        "arn:aws:bedrock:REGION:ACCOUNT:inference-profile/us.anthropic.%%model=anthropic%%",
        "arn:aws:bedrock:*::foundation-model/anthropic.%%model=anthropic%%"
      ]
    },
    {
      "Sid": "Memory",
      "Effect": "Allow",
      "Action": [
        "bedrock-agentcore:CreateEvent",
        "bedrock-agentcore:ListEvents",
        "bedrock-agentcore:GetEvent",
        "bedrock-agentcore:RetrieveMemoryRecords"
      ],
      "Resource": "arn:aws:bedrock-agentcore:REGION:ACCOUNT:memory/MEMORY_ID"
    }
  ]
}
```

AWS's template allows all foundation models and `arn:aws:bedrock:REGION:ACCOUNT:*`. The version above names one inference profile and its model. Keep the `DenyUnverifiedUserTokens` statement only when callers authenticate with JWTs: it blocks issuing outbound-credential tokens for a caller-supplied user ID.

## AgentCore Runtime: invoke a runtime

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "bedrock-agentcore:InvokeAgentRuntime",
      "Resource": [
        "arn:aws:bedrock-agentcore:REGION:ACCOUNT:runtime/RUNTIME_ID",
        "arn:aws:bedrock-agentcore:REGION:ACCOUNT:runtime/RUNTIME_ID/runtime-endpoint/prod"
      ]
    }
  ]
}
```

Callers that also pass `runtimeUserId` (the `X-Amzn-Bedrock-AgentCore-Runtime-User-Id` header) to identify an end user need a second action on the same resources:

```json
{
  "Effect": "Allow",
  "Action": ["bedrock-agentcore:InvokeAgentRuntime", "bedrock-agentcore:InvokeAgentRuntimeForUser"],
  "Resource": "arn:aws:bedrock-agentcore:REGION:ACCOUNT:runtime/RUNTIME_ID/runtime-endpoint/prod"
}
```

`bedrock-agentcore:InvokeAgentRuntimeCommand` is a separate action that runs a shell command inside a runtime or harness session without going through the model. Do not grant it to callers who should only be able to chat.

## AgentCore Runtime: using Code Interpreter and Browser

Add statements to the execution role for the sandboxed tools the agent calls. The session actions match the API operations; the managed interpreter and browser are owned by the `aws` account.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "CodeInterpreter",
      "Effect": "Allow",
      "Action": [
        "bedrock-agentcore:StartCodeInterpreterSession",
        "bedrock-agentcore:InvokeCodeInterpreter",
        "bedrock-agentcore:GetCodeInterpreterSession",
        "bedrock-agentcore:StopCodeInterpreterSession"
      ],
      "Resource": "arn:aws:bedrock-agentcore:REGION:aws:code-interpreter/aws.codeinterpreter.v1"
    },
    {
      "Sid": "Browser",
      "Effect": "Allow",
      "Action": [
        "bedrock-agentcore:StartBrowserSession",
        "bedrock-agentcore:GetBrowserSession",
        "bedrock-agentcore:StopBrowserSession"
      ],
      "Resource": "arn:aws:bedrock-agentcore:REGION:aws:browser/aws.browser.v1"
    }
  ]
}
```

Driving the browser over its WebSocket stream (`generate_ws_headers()`) needs additional stream permissions; check the AgentCore Browser documentation for the current action names before locking the role down. A custom interpreter or browser has its own ARN in your account.

## AgentCore harness

The harness execution role uses the same trust policy. On the public network, it pulls the managed harness image from ECR Public, so it needs `ecr-public:GetAuthorizationToken` and `sts:GetServiceBearerToken` instead of the private ECR statements. In VPC mode, it needs `ecr:BatchGetImage`, `ecr:GetDownloadUrlForLayer` and `ecr:BatchCheckLayerAvailability` on `arn:aws:ecr:REGION:*:repository/harness-*`. The logs, tracing, metrics and model statements are the same as for Runtime, with workload identities named `harness_HARNESS_NAME-*`. Add statements per tool: `bedrock-agentcore:InvokeGateway` on a SigV4 gateway, Code Interpreter or Browser session actions on `arn:aws:bedrock-agentcore:REGION:aws:code-interpreter/*` or `browser/*`, and memory actions on the harness's memory.

Callers of a harness need both actions on the harness ARN:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["bedrock-agentcore:InvokeHarness", "bedrock-agentcore:InvokeAgentRuntime"],
      "Resource": "arn:aws:bedrock-agentcore:REGION:ACCOUNT:harness/HARNESS_ID"
    }
  ]
}
```

## Agents Classic: invoke one agent alias

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "bedrock:InvokeAgent",
      "Resource": "arn:aws:bedrock:REGION:ACCOUNT:agent-alias/AGENT_ID/ALIAS_ID"
    }
  ]
}
```

## Agents Classic: agent service role

Attached to the role the agent assumes (trusted by `bedrock.amazonaws.com`). When the agent uses a cross-Region inference profile such as `us.anthropic.%%model=anthropic%%`, allow both the profile and the underlying foundation model in the Regions the profile routes to.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:GetInferenceProfile"
      ],
      "Resource": [
        "arn:aws:bedrock:REGION:ACCOUNT:inference-profile/us.anthropic.%%model=anthropic%%",
        "arn:aws:bedrock:*::foundation-model/anthropic.%%model=anthropic%%"
      ]
    }
  ]
}
```

Trust policy for that role, scoped to your account's agents:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "bedrock.amazonaws.com" },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": { "aws:SourceAccount": "ACCOUNT" },
        "ArnLike": { "aws:SourceArn": "arn:aws:bedrock:REGION:ACCOUNT:agent/*" }
      }
    }
  ]
}
```

## Knowledge base access

For the agent service role (or an application calling `Retrieve` / `RetrieveAndGenerate` directly):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:Retrieve",
        "bedrock:RetrieveAndGenerate"
      ],
      "Resource": "arn:aws:bedrock:REGION:ACCOUNT:knowledge-base/KB_ID"
    }
  ]
}
```

## KMS decrypt for encrypted secrets

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [ "kms:Decrypt" ],
      "Resource": "arn:aws:kms:REGION:ACCOUNT:key/KEY_ID"
    }
  ]
}
```

---
title: "Amazon Bedrock Agents - IAM Examples"
description: "Least-privilege IAM policies for Bedrock Agents, agent service roles, knowledge bases and AgentCore Runtime."
framework: amazon-bedrock-agents
---

# Amazon Bedrock Agents - IAM Examples

Least-privilege starting points. Replace `REGION`, `ACCOUNT`, IDs and names with your own, and prefer specific ARNs over wildcards. More policies appear in the [comprehensive guide](/amazon-bedrock-agents-guide/bedrock_agents_comprehensive_guide/#aws-bedrock-setup-and-iam-permissions) and the [AgentCore guide](/amazon-bedrock-agents-guide/bedrock_agentcore_comprehensive_guide/#security-and-compliance).

## Application: invoke one agent alias (Bedrock Agents Classic)

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

## Agent service role: model invocation

Attached to the role the agent assumes (trusted by `bedrock.amazonaws.com`). When the agent uses a cross-Region inference profile such as `us.anthropic.claude-sonnet-5-5`, allow both the profile and the underlying foundation model in the Regions the profile routes to.

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
        "arn:aws:bedrock:REGION:ACCOUNT:inference-profile/us.anthropic.claude-sonnet-5-5",
        "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-5-5"
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

## Knowledge Base Access

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

## AgentCore Runtime: invoke a runtime

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "bedrock-agentcore:InvokeAgentRuntime",
      "Resource": "arn:aws:bedrock-agentcore:REGION:ACCOUNT:runtime/RUNTIME_ID*"
    }
  ]
}
```

## KMS Decrypt for Encrypted Secrets

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

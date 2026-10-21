---
title: "Amazon Bedrock Agents Classic: Technical Reference"
description: "Reference for Amazon Bedrock Agents (Agents Classic, in maintenance mode) for existing users: IAM, agents and aliases, action groups and return of control, knowledge bases, guardrails, sessions and memory, prompt overrides, inline agents, Flows and infrastructure as code, with boto3 examples."
framework: "amazon-bedrock-agents"
language: "python"
sidebar:
  label: "Agents Classic reference"
  order: 12
---

> **Verified against:** `boto3` %%version=boto3%% (checked %%versions-checked%%). Every request below was validated against the botocore service models for `bedrock`, `bedrock-agent`, `bedrock-agent-runtime` and `bedrock-runtime`.

> **Maintenance mode.** Amazon Bedrock Agents, now called *Agents Classic*, stopped accepting new customers on 30 July 2026. Accounts with Bedrock Agents activity in the previous 12 months keep full access, while other accounts get `AccessDeniedException` from `CreateAgent` and `InvokeInlineAgent`. The model catalogue for Classic is frozen and no new features are planned. For new work AWS recommends [Amazon Bedrock AgentCore](/amazon-bedrock-agents-guide/python/bedrock_agentcore_comprehensive_guide/). See [migrating from Classic](/amazon-bedrock-agents-guide/python/bedrock_agentcore_comprehensive_guide/#migrating-from-bedrock-agents-classic).

**Which boto3 client?**

| Client | Used for |
|--------|----------|
| `bedrock-agent` | Build time: agents, action groups, aliases, knowledge bases, data sources, flows |
| `bedrock-agent-runtime` | Run time: `invoke_agent`, `invoke_inline_agent`, `retrieve`, `retrieve_and_generate`, `invoke_flow`, agent memory |
| `bedrock` | Guardrails, model listing, inference profiles |
| `bedrock-runtime` | Direct model calls and `apply_guardrail` |

---

## Table of Contents

1. [How an agent works](#how-an-agent-works)
2. [IAM](#iam)
3. [Creating an agent](#creating-an-agent)
4. [Action groups](#action-groups)
5. [Return of control](#return-of-control)
6. [Code interpretation](#code-interpretation)
7. [Knowledge bases](#knowledge-bases)
8. [Guardrails](#guardrails)
9. [Sessions and memory](#sessions-and-memory)
10. [Prompt overrides](#prompt-overrides)
11. [Inline agents](#inline-agents)
12. [Flows](#flows)
13. [Models and Regions](#models-and-regions)
14. [Creating agents (console, CloudFormation, Terraform)](#creating-agents-console-cloudformation-terraform)

---

## How an agent works

An agent is a model, an instruction, and optional action groups, knowledge bases and a guardrail. `invoke_agent` runs a managed loop: optional pre-processing, then orchestration (the model reasons, calls action groups or queries knowledge bases, and observes the results), then optional post-processing. The answer comes back as an event stream of `chunk` events, plus `trace` events when `enableTrace=True`, and `returnControl` or `files` events when relevant.

Agents are edited as a `DRAFT` version, **prepared**, then published through an **alias**, which snapshots a numbered version. Applications invoke an alias. Multi-agent collaboration (a supervisor with collaborator agents) is covered in [multi-agent patterns](/amazon-bedrock-agents-guide/python/bedrock_agents_advanced_multi_agent_python/#4-multi-agent-collaboration-ga-march-2025), and the flow is drawn in the [diagrams](/amazon-bedrock-agents-guide/python/bedrock_agents_diagrams/#bedrock-agents-classic).

## IAM

Three separate sets of permissions are involved. Don't merge them into one role:

1. **The agent service role** (`agentResourceRoleArn`), assumed by Bedrock while the agent runs. Trust only `bedrock.amazonaws.com`, conditioned on your account and agents:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "bedrock.amazonaws.com" },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": { "aws:SourceAccount": "111122223333" },
        "ArnLike": { "aws:SourceArn": "arn:aws:bedrock:us-east-1:111122223333:agent/*" }
      }
    }
  ]
}
```

   Grant it model invocation on the specific inference profile **and** the foundation model it routes to. Foundation-model ARNs have no account ID, so `arn:aws:bedrock:*:ACCOUNT:*` does not match them. Add knowledge-base retrieval and guardrail use as needed:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "InvokeModel",
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream", "bedrock:GetInferenceProfile"],
      "Resource": [
        "arn:aws:bedrock:us-east-1:111122223333:inference-profile/us.anthropic.claude-sonnet-5-5",
        "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-5-5"
      ]
    },
    {
      "Sid": "KnowledgeBase",
      "Effect": "Allow",
      "Action": ["bedrock:Retrieve"],
      "Resource": "arn:aws:bedrock:us-east-1:111122223333:knowledge-base/KB_ID"
    },
    {
      "Sid": "Guardrail",
      "Effect": "Allow",
      "Action": ["bedrock:ApplyGuardrail"],
      "Resource": "arn:aws:bedrock:us-east-1:111122223333:guardrail/GUARDRAIL_ID"
    }
  ]
}
```

2. **Action-group Lambda functions** are not called through the service role. Each function needs a resource-based policy allowing `bedrock.amazonaws.com` to invoke it, with `SourceArn` set to the agent ARN (see the [CloudFormation template](#cloudformation)).

3. **Callers.** Applications need `bedrock:InvokeAgent` on the alias ARN only. Build pipelines need the `bedrock:CreateAgent`, `bedrock:UpdateAgent`, `bedrock:PrepareAgent` and alias actions, plus `iam:PassRole` on the service role.

More policies, including knowledge-base and AgentCore roles, are on the [IAM examples](/amazon-bedrock-agents-guide/python/bedrock_security_iam_examples/) page.

## Creating an agent

```python
import time

import boto3

agents = boto3.client("bedrock-agent", region_name="us-east-1")

agent = agents.create_agent(
    agentName="SupportAgent",
    foundationModel="us.anthropic.claude-sonnet-5-5",
    agentResourceRoleArn="arn:aws:iam::111122223333:role/BedrockAgentServiceRole",
    instruction="You are a support agent for Example Corp. Look up orders before answering questions about them.",
    idleSessionTTLInSeconds=900,
    memoryConfiguration={"enabledMemoryTypes": ["SESSION_SUMMARY"], "storageDays": 30},
    guardrailConfiguration={"guardrailIdentifier": "gr-abc123", "guardrailVersion": "1"},
)["agent"]
agent_id = agent["agentId"]


def wait_until_settled(agent_id: str) -> str:
    while (status := agents.get_agent(agentId=agent_id)["agent"]["agentStatus"]) in ("CREATING", "UPDATING", "PREPARING"):
        time.sleep(2)
    return status


wait_until_settled(agent_id)
# ... add action groups and knowledge bases here (below) ...
agents.prepare_agent(agentId=agent_id)
assert wait_until_settled(agent_id) == "PREPARED"

alias_id = agents.create_agent_alias(agentId=agent_id, agentAliasName="live")["agentAlias"]["agentAliasId"]
```

The alias is usable once `get_agent_alias(...)["agentAlias"]["agentAliasStatus"]` is `PREPARED`. To release a change, prepare again and either create a new alias or point an existing one at the new version with `update_agent_alias(agentId=..., agentAliasId=..., agentAliasName=..., routingConfiguration=[{"agentVersion": "N"}])`. The [recipes](/amazon-bedrock-agents-guide/python/bedrock_agents_recipes/) page has a `prepare_and_alias` helper that waits correctly.

Invoking the alias:

```python
runtime = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

response = runtime.invoke_agent(
    agentId=agent_id,
    agentAliasId=alias_id,
    sessionId="customer-42-2026-10-21",
    inputText="Where is order ORD-7?",
    sessionState={"sessionAttributes": {"customerId": "C-42"}},  # reaches action-group Lambdas, not the model
)
answer = "".join(e["chunk"]["bytes"].decode() for e in response["completion"] if "chunk" in e)
```

## Action groups

An action group is a set of operations the agent can call, described either by a **function schema** (simplest) or an **OpenAPI schema**, and executed by a Lambda function or returned to your application ([return of control](#return-of-control)).

```python
agents.create_agent_action_group(
    agentId=agent_id,
    agentVersion="DRAFT",
    actionGroupName="Orders",
    description="Look up and cancel customer orders",
    actionGroupExecutor={"lambda": "arn:aws:lambda:us-east-1:111122223333:function:orders-action-group"},
    functionSchema={
        "functions": [
            {
                "name": "get_order",
                "description": "Get the status of an order",
                "parameters": {"order_id": {"type": "string", "description": "Order ID, e.g. ORD-7", "required": True}},
            },
            {
                "name": "cancel_order",
                "description": "Cancel an order that has not shipped",
                "parameters": {"order_id": {"type": "string", "description": "Order ID", "required": True}},
                "requireConfirmation": "ENABLED",  # ask the user before running it
            },
        ]
    },
)
```

The Lambda receives the function name and parameters, plus the session attributes, and must answer in the agent's response envelope:

```python
import json


def lambda_handler(event, context):
    params = {p["name"]: p["value"] for p in event.get("parameters", [])}
    customer_id = event.get("sessionAttributes", {}).get("customerId")

    if event["function"] == "get_order":
        body = {"order_id": params["order_id"], "customer": customer_id, "status": "SHIPPED"}
    elif event["function"] == "cancel_order":
        body = {"order_id": params["order_id"], "cancelled": True}
    else:
        body = {"error": f"unknown function {event['function']}"}

    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": event["actionGroup"],
            "function": event["function"],
            "functionResponse": {"responseBody": {"TEXT": {"body": json.dumps(body)}}},
        },
    }
```

For an OpenAPI action group, pass `apiSchema={"payload": json.dumps(openapi_spec)}` (or an S3 location) instead of `functionSchema`. The Lambda then receives `apiPath`, `httpMethod` and `requestBody`, and returns `apiPath`, `httpMethod`, `httpStatusCode` and `responseBody` in the envelope, as in the [CloudFormation example](#cloudformation). Bedrock Agents has no built-in HTTP executor: to call an external API, call it from the Lambda or use return of control.

## Return of control

With `actionGroupExecutor={"customControl": "RETURN_CONTROL"}`, the agent does not run the action. It ends the stream with a `returnControl` event, and your application runs the call and sends the result back in the next `invoke_agent` on the same session:

```python
import json

import boto3

agents = boto3.client("bedrock-agent")
runtime = boto3.client("bedrock-agent-runtime")

agents.create_agent_action_group(
    agentId="AGENT_ID",
    agentVersion="DRAFT",
    actionGroupName="LocalTools",
    actionGroupExecutor={"customControl": "RETURN_CONTROL"},
    functionSchema={
        "functions": [
            {
                "name": "get_weather",
                "description": "Current weather for a city",
                "parameters": {"city": {"type": "string", "required": True}},
            }
        ]
    },
)


def run_local(function: str, params: dict) -> dict:
    if function == "get_weather":
        return {"city": params["city"], "forecast": "light rain, 12 °C"}
    raise ValueError(function)


def ask(agent_id: str, alias_id: str, session_id: str, text: str) -> str:
    kwargs = {"inputText": text}
    while True:
        response = runtime.invoke_agent(agentId=agent_id, agentAliasId=alias_id, sessionId=session_id, **kwargs)
        answer, control = "", None
        for event in response["completion"]:
            if "chunk" in event:
                answer += event["chunk"]["bytes"].decode()
            elif "returnControl" in event:
                control = event["returnControl"]
        if control is None:
            return answer

        results = []
        for item in control["invocationInputs"]:
            call = item["functionInvocationInput"]
            params = {p["name"]: p["value"] for p in call.get("parameters", [])}
            results.append({
                "functionResult": {
                    "actionGroup": call["actionGroup"],
                    "function": call["function"],
                    "responseBody": {"TEXT": {"body": json.dumps(run_local(call["function"], params))}},
                }
            })
        kwargs = {"sessionState": {"invocationId": control["invocationId"], "returnControlInvocationResults": results}}
```

The built-in `AMAZON.UserInput` signature (an action group with `parentActionGroupSignature="AMAZON.UserInput"`) lets the agent ask the user for missing information instead of guessing.

## Code interpretation

Code interpretation is enabled by adding an action group with the built-in `AMAZON.CodeInterpreter` signature. There is no flag on the agent:

```python
agents.create_agent_action_group(
    agentId=agent_id,
    agentVersion="DRAFT",
    actionGroupName="CodeInterpreterAction",
    parentActionGroupSignature="AMAZON.CodeInterpreter",
    actionGroupState="ENABLED",
)
```

Files are passed in `sessionState`, and generated files come back as `files` events:

```python
response = runtime.invoke_agent(
    agentId=agent_id,
    agentAliasId=alias_id,
    sessionId="analysis-1",
    inputText="Summarise the attached CSV and plot revenue by month.",
    sessionState={
        "files": [
            {
                "name": "sales.csv",
                "source": {"sourceType": "S3", "s3Location": {"uri": "s3://my-bucket/sales.csv"}},
                "useCase": "CODE_INTERPRETER",
            }
        ]
    },
)

for event in response["completion"]:
    if "chunk" in event:
        print(event["chunk"]["bytes"].decode(), end="")
    elif "files" in event:
        for f in event["files"]["files"]:
            with open(f["name"], "wb") as out:
                out.write(f["bytes"])
```

## Knowledge bases

Knowledge bases are not affected by maintenance mode. They are a separate Bedrock feature, and AgentCore agents reach them through Gateway or a retrieval tool. A vector knowledge base on OpenSearch Serverless, with an S3 data source:

```python
kb = agents.create_knowledge_base(
    name="support-docs",
    roleArn="arn:aws:iam::111122223333:role/BedrockKnowledgeBaseRole",
    knowledgeBaseConfiguration={
        "type": "VECTOR",
        "vectorKnowledgeBaseConfiguration": {
            "embeddingModelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0"
        },
    },
    storageConfiguration={
        "type": "OPENSEARCH_SERVERLESS",
        "opensearchServerlessConfiguration": {
            "collectionArn": "arn:aws:aoss:us-east-1:111122223333:collection/abc123",
            "vectorIndexName": "support-docs-index",
            "fieldMapping": {
                "vectorField": "bedrock-vector",
                "textField": "AMAZON_BEDROCK_TEXT_CHUNK",
                "metadataField": "AMAZON_BEDROCK_METADATA",
            },
        },
    },
)["knowledgeBase"]
```

**Chunking happens server-side** during ingestion, configured on the data source. There are four strategies:

```python
CHUNKING = {
    # Fixed windows: predictable, a good default for uniform prose
    "fixed": {"chunkingStrategy": "FIXED_SIZE",
              "fixedSizeChunkingConfiguration": {"maxTokens": 512, "overlapPercentage": 20}},
    # Parent/child chunks: retrieve small chunks, return their larger parents for context
    "hierarchical": {"chunkingStrategy": "HIERARCHICAL",
                     "hierarchicalChunkingConfiguration": {
                         "levelConfigurations": [{"maxTokens": 1500}, {"maxTokens": 300}],
                         "overlapTokens": 60}},
    # Split where the embedding distance between sentences jumps
    "semantic": {"chunkingStrategy": "SEMANTIC",
                 "semanticChunkingConfiguration": {
                     "maxTokens": 300, "bufferSize": 1, "breakpointPercentileThreshold": 95}},
    # Each file is one chunk (pre-chunked content)
    "none": {"chunkingStrategy": "NONE"},
}

ds = agents.create_data_source(
    knowledgeBaseId=kb["knowledgeBaseId"],
    name="support-docs-s3",
    dataSourceConfiguration={
        "type": "S3",
        "s3Configuration": {"bucketArn": "arn:aws:s3:::support-docs", "inclusionPrefixes": ["published/"]},
    },
    vectorIngestionConfiguration={"chunkingConfiguration": CHUNKING["hierarchical"]},
)["dataSource"]

agents.start_ingestion_job(knowledgeBaseId=kb["knowledgeBaseId"], dataSourceId=ds["dataSourceId"])
```

The chunking strategy cannot be changed after the data source is created; create a new data source to switch.

Attach the knowledge base to an agent (then prepare it again):

```python
agents.associate_agent_knowledge_base(
    agentId=agent_id,
    agentVersion="DRAFT",
    knowledgeBaseId=kb["knowledgeBaseId"],
    description="Product manuals and support articles. Use for how-to and troubleshooting questions.",
    knowledgeBaseState="ENABLED",
)
```

Or query it directly, without an agent. `retrieve` returns chunks, and `retrieve_and_generate` returns a cited answer:

```python
runtime = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

chunks = runtime.retrieve(
    knowledgeBaseId=kb["knowledgeBaseId"],
    retrievalQuery={"text": "How do I reset the device?"},
    retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 5, "overrideSearchType": "HYBRID"}},
)["retrievalResults"]

answer = runtime.retrieve_and_generate(
    input={"text": "How do I reset the device?"},
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": kb["knowledgeBaseId"],
            "modelArn": "arn:aws:bedrock:us-east-1:111122223333:inference-profile/us.anthropic.claude-sonnet-5-5",
        },
    },
)
print(answer["output"]["text"])
for citation in answer["citations"]:
    for ref in citation["retrievedReferences"]:
        print(ref["location"])
```

Pass the returned `sessionId` to the next `retrieve_and_generate` call for follow-up questions.

## Guardrails

Guardrails are managed with the `bedrock` client and attached to an agent with `guardrailConfiguration`. They also work on direct model calls and AgentCore agents.

```python
bedrock = boto3.client("bedrock", region_name="us-east-1")

guardrail = bedrock.create_guardrail(
    name="support-guardrail",
    blockedInputMessaging="Sorry, I can't help with that request.",
    blockedOutputsMessaging="Sorry, I can't provide that response.",
    topicPolicyConfig={
        "topicsConfig": [
            {
                "name": "InvestmentAdvice",
                "definition": "Recommendations about buying or selling specific financial products.",
                "examples": ["Should I buy shares in Example Corp?"],
                "type": "DENY",
            }
        ]
    },
    contentPolicyConfig={
        "filtersConfig": [
            {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
            {"type": "INSULTS", "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"},
            {"type": "VIOLENCE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
            {"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "NONE"},
        ]
    },
    wordPolicyConfig={
        "wordsConfig": [{"text": "Project Nightjar"}],      # blocked on input and output by default
        "managedWordListsConfig": [{"type": "PROFANITY"}],
    },
    sensitiveInformationPolicyConfig={
        "piiEntitiesConfig": [
            {"type": "EMAIL", "action": "ANONYMIZE"},
            {"type": "PHONE", "action": "ANONYMIZE"},
            {"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "BLOCK"},
            {"type": "INTERNATIONAL_BANK_ACCOUNT_NUMBER", "action": "BLOCK"},
            {"type": "AWS_ACCESS_KEY", "action": "BLOCK"},
        ],
        "regexesConfig": [
            {"name": "InternalTicketId", "pattern": r"TCK-\d{6}", "action": "ANONYMIZE"},
        ],
    },
)
version = bedrock.create_guardrail_version(guardrailIdentifier=guardrail["guardrailId"])["version"]
```

Each word and filter entry can also set `inputAction`/`outputAction` (`BLOCK` or `NONE`) and `inputEnabled`/`outputEnabled` separately. `PROMPT_ATTACK` filters input only, so its output strength must be `NONE`.

Check text against a guardrail without calling a model, for example before forwarding user input to an agent:

```python
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

result = bedrock_runtime.apply_guardrail(
    guardrailIdentifier=guardrail["guardrailId"],
    guardrailVersion=version,
    source="INPUT",                       # or "OUTPUT" for model responses
    content=[{"text": {"text": "My card is 4111 1111 1111 1111, should I buy Example Corp shares?"}}],
)

if result["action"] == "GUARDRAIL_INTERVENED":
    print("blocked:", [o["text"] for o in result["outputs"]])
```

## Sessions and memory

Agents Classic keeps the conversation for a `sessionId` inside the service until `idleSessionTTLInSeconds` (60 to 5,400 seconds) passes without a call. You don't store it yourself. Per-session data:

- `sessionState.sessionAttributes`: key-value pairs passed to action-group Lambdas for the whole session. Use them for user and tenant IDs.
- `sessionState.promptSessionAttributes`: added to the orchestration prompt for this turn (for example, the current date).
- `sessionState.conversationHistory`: seed a new session with earlier turns.
- `endSession=True`: end the session now.

**Memory across sessions.** With `memoryConfiguration={"enabledMemoryTypes": ["SESSION_SUMMARY"], "storageDays": 30}` on the agent, each session is summarised when it ends. Later sessions that pass the same `memoryId` (one per end user) get those summaries in context:

```python
runtime.invoke_agent(
    agentId=agent_id,
    agentAliasId=alias_id,
    sessionId="customer-42-2026-10-22",
    memoryId="customer-42",
    inputText="Any update on what we discussed yesterday?",
)

summaries = runtime.get_agent_memory(
    agentId=agent_id, agentAliasId=alias_id, memoryId="customer-42", memoryType="SESSION_SUMMARY",
)["memoryContents"]

# Right to erasure
runtime.delete_agent_memory(agentId=agent_id, agentAliasId=alias_id, memoryId="customer-42")
```

## Prompt overrides

Each orchestration stage (`PRE_PROCESSING`, `ORCHESTRATION`, `KNOWLEDGE_BASE_RESPONSE_GENERATION`, `POST_PROCESSING`, `MEMORY_SUMMARIZATION`) uses a base prompt template you can override or switch on and off. Start from the defaults that `get_agent` returns for the agent, edit them, and set `promptCreationMode="OVERRIDDEN"`:

```python
POST_PROCESSING_TEMPLATE = open("post_processing_prompt.txt").read()  # your edited copy of the default

agents.update_agent(
    agentId=agent_id,
    agentName="SupportAgent",
    foundationModel="us.anthropic.claude-sonnet-5-5",
    agentResourceRoleArn="arn:aws:iam::111122223333:role/BedrockAgentServiceRole",
    instruction="You are a support agent for Example Corp.",
    promptOverrideConfiguration={
        "promptConfigurations": [
            {
                "promptType": "PRE_PROCESSING",
                "promptCreationMode": "DEFAULT",
                "promptState": "ENABLED",          # classify and reject malicious or off-topic input first
            },
            {
                "promptType": "POST_PROCESSING",
                "promptCreationMode": "OVERRIDDEN",
                "promptState": "ENABLED",
                "basePromptTemplate": POST_PROCESSING_TEMPLATE,
                "inferenceConfiguration": {"temperature": 0.0, "topP": 1.0, "maximumLength": 1024},
                "parserMode": "DEFAULT",
            },
        ]
    },
)
```

`overrideLambda` together with `parserMode="OVERRIDDEN"` lets a Lambda function parse the model's raw output for a stage. Prompt overrides are the main Classic feature without a direct harness equivalent, so agents that rely on them migrate to code-defined agents.

## Inline agents

`invoke_inline_agent` defines and runs an agent in one call, with no create, prepare or alias steps. This is useful when the instruction or tools vary per request. Like `CreateAgent`, it is only available to accounts with recent usage.

```python
response = runtime.invoke_inline_agent(
    sessionId="inline-session-1",
    foundationModel="us.anthropic.claude-sonnet-5-5",
    instruction="You are a travel assistant for Example Corp employees. Use the policy tool before booking.",
    actionGroups=[
        {
            "actionGroupName": "TravelPolicy",
            "actionGroupExecutor": {"customControl": "RETURN_CONTROL"},
            "functionSchema": {
                "functions": [
                    {"name": "max_hotel_rate", "description": "Nightly hotel limit for a city",
                     "parameters": {"city": {"type": "string", "required": True}}}
                ]
            },
        }
    ],
    knowledgeBases=[{"knowledgeBaseId": "KB_ID", "description": "Travel policy documents"}],
    inputText="Book me a hotel in Leeds for two nights next week.",
)

for event in response["completion"]:
    if "chunk" in event:
        print(event["chunk"]["bytes"].decode(), end="")
    elif "returnControl" in event:
        ...  # handle as in Return of control, then call again with inlineSessionState
```

## Flows

Amazon Bedrock Flows link prompts, agents, knowledge bases, Lambda functions and conditions into a graph of typed nodes. Like knowledge bases, Flows is a separate Bedrock feature. A minimal flow (Input → Prompt → Output):

```python
flow = agents.create_flow(
    name="summarise-ticket",
    executionRoleArn="arn:aws:iam::111122223333:role/BedrockFlowsRole",
    definition={
        "nodes": [
            {
                "name": "FlowInputNode",
                "type": "Input",
                "configuration": {"input": {}},
                "outputs": [{"name": "document", "type": "String"}],
            },
            {
                "name": "Summarise",
                "type": "Prompt",
                "configuration": {
                    "prompt": {
                        "sourceConfiguration": {
                            "inline": {
                                "modelId": "us.anthropic.claude-haiku-5-5",
                                "templateType": "TEXT",
                                "inferenceConfiguration": {"text": {"temperature": 0.0, "maxTokens": 300}},
                                "templateConfiguration": {
                                    "text": {
                                        "text": "Summarise this support ticket in two sentences:\n\n{{ticket}}",
                                        "inputVariables": [{"name": "ticket"}],
                                    }
                                },
                            }
                        }
                    }
                },
                "inputs": [{"name": "ticket", "type": "String", "expression": "$.data"}],
                "outputs": [{"name": "modelCompletion", "type": "String"}],
            },
            {
                "name": "FlowOutputNode",
                "type": "Output",
                "configuration": {"output": {}},
                "inputs": [{"name": "document", "type": "String", "expression": "$.data"}],
            },
        ],
        "connections": [
            {"name": "InputToPrompt", "type": "Data", "source": "FlowInputNode", "target": "Summarise",
             "configuration": {"data": {"sourceOutput": "document", "targetInput": "ticket"}}},
            {"name": "PromptToOutput", "type": "Data", "source": "Summarise", "target": "FlowOutputNode",
             "configuration": {"data": {"sourceOutput": "modelCompletion", "targetInput": "document"}}},
        ],
    },
)

agents.prepare_flow(flowIdentifier=flow["id"])
version = agents.create_flow_version(flowIdentifier=flow["id"])["version"]
alias = agents.create_flow_alias(
    flowIdentifier=flow["id"], name="live", routingConfiguration=[{"flowVersion": version}],
)

result = runtime.invoke_flow(
    flowIdentifier=flow["id"],
    flowAliasIdentifier=alias["id"],
    inputs=[{"nodeName": "FlowInputNode", "nodeOutputName": "document",
             "content": {"document": "Customer reports the export button fails on large files..."}}],
)
for event in result["responseStream"]:
    if "flowOutputEvent" in event:
        print(event["flowOutputEvent"]["content"]["document"])
```

Other node types are `Condition`, `KnowledgeBase`, `Retrieval`, `Agent`, `LambdaFunction`, `InlineCode`, `Iterator`/`Collector`, `Loop`, `Storage` and `Lex`. See the [Flows node reference](https://docs.aws.amazon.com/bedrock/latest/userguide/flows-nodes.html).

## Models and Regions

The model catalogue for Agents Classic was frozen on 30 July 2026, so models released after that date are not available to Classic agents. The Claude models current at that date:

| Model | Bedrock model ID | Inference profile example | Typical use |
|-------|------------------|---------------------------|-------------|
| Claude Opus 5.5 | `anthropic.claude-opus-5-5` | `us.anthropic.claude-opus-5-5` | Hardest reasoning, long-horizon work |
| Claude Sonnet 5.5 | `anthropic.claude-sonnet-5-5` | `us.anthropic.claude-sonnet-5-5` | Default for most agents |
| Claude Haiku 5.5 | `anthropic.claude-haiku-5-5` | `us.anthropic.claude-haiku-5-5` | Routing, classification, extraction |

Agents should usually call a cross-Region **inference profile** rather than the base model ID. Which geography prefixes (`us.`, `eu.`, `apac.`, `global.` and others) exist differs by model, so check the [model cards](https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html) and `aws bedrock list-inference-profiles` for your Region. Regions that support Agents are listed in [Supported Regions for Amazon Bedrock Agents](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-supported.html).

## Creating agents (console, CloudFormation, Terraform)

**Console** (existing Classic accounts): Bedrock console → **Agents** → **Create agent**. Choose a model and service role, write the instruction, add action groups and knowledge bases, then **Save** and **Prepare**. Test against `DRAFT` in the test pane, and create an alias for applications.

### CloudFormation

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: Bedrock agent with a function-schema action group

Parameters:
  AgentName:
    Type: String
    Default: SupportAgent
  InferenceProfileId:
    Type: String
    Default: us.anthropic.claude-sonnet-5-5
  FoundationModelId:
    Type: String
    Default: anthropic.claude-sonnet-5-5

Resources:
  AgentRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal: { Service: bedrock.amazonaws.com }
            Action: sts:AssumeRole
            Condition:
              StringEquals: { 'aws:SourceAccount': !Ref 'AWS::AccountId' }
              ArnLike: { 'aws:SourceArn': !Sub 'arn:aws:bedrock:${AWS::Region}:${AWS::AccountId}:agent/*' }
      Policies:
        - PolicyName: InvokeModel
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action: [bedrock:InvokeModel, bedrock:InvokeModelWithResponseStream, bedrock:GetInferenceProfile]
                Resource:
                  - !Sub 'arn:aws:bedrock:${AWS::Region}:${AWS::AccountId}:inference-profile/${InferenceProfileId}'
                  - !Sub 'arn:aws:bedrock:*::foundation-model/${FoundationModelId}'

  OrdersFunctionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal: { Service: lambda.amazonaws.com }
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

  OrdersFunction:
    Type: AWS::Lambda::Function
    Properties:
      Runtime: python3.13
      Handler: index.handler
      Role: !GetAtt OrdersFunctionRole.Arn
      Code:
        ZipFile: |
          import json

          def handler(event, context):
              params = {p['name']: p['value'] for p in event.get('parameters', [])}
              body = {'order_id': params.get('order_id'), 'status': 'SHIPPED'}
              return {
                  'messageVersion': '1.0',
                  'response': {
                      'actionGroup': event['actionGroup'],
                      'function': event['function'],
                      'functionResponse': {'responseBody': {'TEXT': {'body': json.dumps(body)}}},
                  },
              }

  Agent:
    Type: AWS::Bedrock::Agent
    Properties:
      AgentName: !Ref AgentName
      FoundationModel: !Ref InferenceProfileId
      AgentResourceRoleArn: !GetAtt AgentRole.Arn
      AutoPrepare: true
      IdleSessionTTLInSeconds: 900
      Instruction: You are a support agent for Example Corp. Look up orders before answering questions about them.
      ActionGroups:
        - ActionGroupName: Orders
          ActionGroupExecutor:
            Lambda: !GetAtt OrdersFunction.Arn
          FunctionSchema:
            Functions:
              - Name: get_order
                Description: Get the status of an order
                Parameters:
                  order_id:
                    Type: string
                    Description: Order ID
                    Required: true

  OrdersFunctionPermission:
    Type: AWS::Lambda::Permission
    Properties:
      FunctionName: !Ref OrdersFunction
      Action: lambda:InvokeFunction
      Principal: bedrock.amazonaws.com
      SourceAccount: !Ref 'AWS::AccountId'
      SourceArn: !GetAtt Agent.AgentArn

  AgentAlias:
    Type: AWS::Bedrock::AgentAlias
    Properties:
      AgentId: !GetAtt Agent.AgentId
      AgentAliasName: live

Outputs:
  AgentId:
    Value: !GetAtt Agent.AgentId
  AgentAliasId:
    Value: !GetAtt AgentAlias.AgentAliasId
```

### Terraform

The AWS provider has `aws_bedrockagent_agent`, `aws_bedrockagent_agent_action_group`, `aws_bedrockagent_agent_alias`, `aws_bedrockagent_knowledge_base` and `aws_bedrockagent_agent_knowledge_base_association`:

```hcl
resource "aws_bedrockagent_agent" "support" {
  agent_name              = "SupportAgent"
  agent_resource_role_arn = aws_iam_role.agent.arn   # trust and model policy as in the IAM section
  foundation_model        = "us.anthropic.claude-sonnet-5-5"
  idle_session_ttl_in_seconds = 900
  instruction             = "You are a support agent for Example Corp. Look up orders before answering questions about them."
}

resource "aws_bedrockagent_agent_action_group" "orders" {
  agent_id          = aws_bedrockagent_agent.support.agent_id
  agent_version     = "DRAFT"
  action_group_name = "Orders"
  action_group_executor {
    lambda = aws_lambda_function.orders.arn
  }
  function_schema {
    member_functions {
      functions {
        name        = "get_order"
        description = "Get the status of an order"
        parameters {
          map_block_key = "order_id"
          type          = "string"
          required      = true
        }
      }
    }
  }
}

resource "aws_lambda_permission" "bedrock" {
  statement_id  = "AllowBedrockAgent"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.orders.function_name
  principal     = "bedrock.amazonaws.com"
  source_arn    = aws_bedrockagent_agent.support.agent_arn
}

resource "aws_bedrockagent_agent_alias" "live" {
  agent_id         = aws_bedrockagent_agent.support.agent_id
  agent_alias_name = "live"
  depends_on       = [aws_bedrockagent_agent_action_group.orders]
}
```

IaC templates keep working for allowlisted accounts. In accounts without prior Bedrock Agents usage, `CreateAgent` fails, so new environments should provision AgentCore resources instead.

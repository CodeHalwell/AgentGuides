---
title: "Advanced Error Handling and Recovery for Amazon Bedrock Agents (Python)"
description: "Error handling for Amazon Bedrock agents in Python: which errors to retry, reading event-stream errors, action-group error envelopes, failed-request queues and fallbacks."
framework: "amazon-bedrock-agents"
language: "python"
sidebar:
  label: "Error handling"
  order: 6
---

> Examples use `boto3` %%version=boto3%% (checked %%versions-checked%%).

## 1. Common Errors in Bedrock Agents

*   **ValidationException:** Invalid input parameters.
*   **AccessDeniedException:** Insufficient IAM permissions.
*   **ResourceNotFoundException:** The specified agent, knowledge base, or action group does not exist.
*   **ThrottlingException:** The request rate exceeds the limit.
*   **ServiceQuotaExceededException:** A quota (for example concurrent sessions) is exhausted.
*   **ModelNotReadyException:** The model is not ready to serve requests yet; retry after a short wait.
*   **InternalServerException / BadGatewayException:** A transient service-side error occurred.
*   **DependencyFailedException:** An error occurred in a downstream service, such as a Lambda function or a knowledge base.

## 2. Error Handling Patterns

### a. Retry with Exponential Backoff

Implement a retry mechanism with exponential backoff to handle transient errors like `ThrottlingException` and `InternalServerException`.

```python
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EventStreamError

# botocore already retries throttling with its own backoff; 'adaptive' mode adds client-side rate limiting
agent_runtime = boto3.client(
    "bedrock-agent-runtime",
    config=Config(retries={"max_attempts": 5, "mode": "adaptive"}, read_timeout=300),
)

RETRYABLE = {
    "ThrottlingException",
    "ServiceQuotaExceededException",
    "ModelNotReadyException",
    "BadGatewayException",
    "InternalServerException",
}


def invoke_agent_with_retry(agent_id, alias_id, session_id, input_text, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = agent_runtime.invoke_agent(
                agentId=agent_id,
                agentAliasId=alias_id,
                sessionId=session_id,
                inputText=input_text,
            )
            # Errors can also arrive mid-stream, so read the stream inside the try block
            return "".join(
                e["chunk"]["bytes"].decode() for e in response["completion"] if "chunk" in e
            )
        except (ClientError, EventStreamError) as e:
            code = e.response["Error"]["Code"]
            if code in RETRYABLE and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
```

### b. Queue requests that still fail

When retries are exhausted, keep the request for later replay or review instead of dropping it. A small SQS hand-off:

```python
import json
import os

import boto3

sqs = boto3.client("sqs")


def invoke_or_park(agent_id, alias_id, session_id, input_text):
    try:
        return invoke_agent_with_retry(agent_id, alias_id, session_id, input_text)
    except (ClientError, EventStreamError) as e:
        sqs.send_message(
            QueueUrl=os.environ["FAILED_REQUESTS_QUEUE_URL"],
            MessageBody=json.dumps({
                "agentId": agent_id, "aliasId": alias_id, "sessionId": session_id,
                "inputText": input_text, "error": e.response["Error"]["Code"],
            }),
        )
        raise
```

For asynchronous Lambda invocations (for example from EventBridge), configure an on-failure destination or a dead-letter queue on the function instead.

### c. Custom Error Handling in Action Groups

Implement custom error handling in your action group Lambda functions to provide more meaningful error messages to the agent.

Return errors inside the normal response envelope so the agent can read them and recover (ask for missing input, try another approach) instead of the whole invocation failing with `DependencyFailedException`:

```python
import json


def lambda_handler(event, context):
    try:
        result = handle(event)  # your action group logic
        status, body = 200, result
    except ValueError as e:      # bad input the model can fix
        status, body = 400, {"error": "InvalidInput", "message": str(e)}
    except Exception:
        status, body = 500, {"error": "InternalError", "message": "The booking system is unavailable; try later."}

    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": event["actionGroup"],
            "apiPath": event["apiPath"],
            "httpMethod": event["httpMethod"],
            "httpStatusCode": status,
            "responseBody": {"application/json": {"body": json.dumps(body)}},
        },
    }
```

## 3. Recovery Patterns

### a. Session State and Recovery

Bedrock Agents keeps conversation state per `sessionId` for the agent's idle session TTL, so a retried request with the same session continues where it left off; enable agent memory (`memoryConfiguration`) to summarise sessions for later ones. For code-defined agents, persist state yourself — Strands session managers (S3, file or AgentCore Memory) restore the conversation and `agent.state` after a failure, and AgentCore Runtime sessions keep in-memory state between calls with the same `runtimeSessionId`.

### b. Fallback Agents

Implement a fallback mechanism to a simpler agent or a human agent if the primary agent fails to complete a task.

```python
try:
    answer = invoke_agent_with_retry(PRIMARY_AGENT_ID, PRIMARY_ALIAS_ID, session_id, request)
except ClientError:
    answer = invoke_agent_with_retry(FALLBACK_AGENT_ID, FALLBACK_ALIAS_ID, session_id, request)
```

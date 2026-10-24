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

These are the errors raised by the Bedrock Agents Classic runtime (`bedrock-agent-runtime`). [Section 4](#4-errors-on-agentcore) covers AgentCore.

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

## 4. Errors on AgentCore

AgentCore failures reach the caller in three different ways, and each needs its own handling.

**1. The API call fails.** `invoke_agent_runtime` and `invoke_harness` raise `ClientError` with a code from the service model. Retry `ThrottlingException`, `InternalServerException`, `RetryableConflictException` and (with care) `ServiceQuotaExceededException`. Do not retry `ValidationException`, `AccessDeniedException`, `ResourceNotFoundException` or `RuntimeClientError`. The [AgentCore guide](/amazon-bedrock-agents-guide/python/bedrock_agentcore_comprehensive_guide/#2-error-handling) has a retry wrapper that is tested with botocore's `Stubber`.

**2. Your container returns an error.** An exception in a non-streaming entrypoint is logged with its stack trace and returned as HTTP 500 with `{"error": "<message>"}`. Check `response["statusCode"]` before parsing the body, because boto3 does not raise for a non-200 status from your container. Keep messages free of secrets and internal detail, since they go back to the caller.

**3. A stream fails part-way.** Once a streaming response has started, the HTTP status is already 200. If the generator raises, the SDK ends the stream with a last frame such as `{"error": "model failed", "error_type": "ValueError", "message": "An error occurred during streaming"}`. A caller that only concatenates text will miss it, so parse each frame:

```python
import json

import boto3

agentcore = boto3.client("bedrock-agentcore", region_name="us-west-2")


class AgentStreamError(RuntimeError):
    """The runtime started streaming, then reported a failure in a final frame."""


def stream_agent(runtime_arn: str, session_id: str, prompt: str):
    response = agentcore.invoke_agent_runtime(
        agentRuntimeArn=runtime_arn,
        runtimeSessionId=session_id,
        payload=json.dumps({"prompt": prompt}),
    )
    if response["statusCode"] != 200:
        raise RuntimeError(f"runtime answered HTTP {response['statusCode']}")
    for line in response["response"].iter_lines():
        if not line.startswith(b"data: "):
            continue  # blank separator lines between frames
        frame = json.loads(line[len(b"data: "):])
        if isinstance(frame, dict) and "error" in frame:
            raise AgentStreamError(f"{frame.get('error_type')}: {frame['error']}")
        yield frame
```

Harness streams report failures as events instead: `internalServerException`, `validationException` and `runtimeClientError` appear in the event stream, and the `stopReason` on the final `messageStop` event says why the run ended (`tool_use` means an inline function is waiting for a result). Log it and check it instead of assuming the answer is complete, especially when you set `maxIterations`, `timeoutSeconds` or `maxTokens`.

**Design for retries.** A retried call to the same `runtimeSessionId` runs in the same microVM, with whatever state the failed attempt left behind. Make tools idempotent (use a request ID as the key for writes) and keep a conversation's history in AgentCore Memory or your own store, not only in process memory.

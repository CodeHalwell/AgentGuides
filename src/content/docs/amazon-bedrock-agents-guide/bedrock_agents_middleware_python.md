---
title: "Implementing Middleware for Amazon Bedrock Agents (Python)"
description: "This guide explains how to implement middleware for Amazon Bedrock Agents to add custom logic for tasks like authentication, logging, and caching."
framework: amazon-bedrock-agents
---

This guide explains how to implement middleware for Amazon Bedrock Agents to add custom logic for tasks like authentication, logging, and caching.

## 1. What is Middleware?

Middleware is software that sits between your application and the Bedrock Agent, allowing you to intercept and process requests and responses.

**Use Cases:**

*   **Authentication and Authorization:** Verify user identity and permissions before invoking an agent.
*   **Logging and Auditing:** Log all agent interactions for security and compliance.
*   **Caching:** Cache frequent requests to reduce latency and cost.
*   **Request/Response Transformation:** Modify requests before they reach the agent and responses before they are returned to the user.

## 2. Middleware Architecture

```
                                  ┌───────────────────┐
                                  │ Middleware        │
                                  ├───────────────────┤
┌──────────┐      ┌───────────┐   │ - Authentication  │   ┌────────────────┐
│          │      │           │   │ - Logging         │   │                │
│  User    ├─────►│  API      ├─► │ - Caching         ├─► │  Bedrock Agent │
│          │      │  Gateway  │   │ - Transformation  │   │                │
└──────────┘      └───────────┘   │                   │   └────────────────┘
                                  └───────────────────┘
```

## 3. Implementation with API Gateway and Lambda

You can implement middleware using Amazon API Gateway and AWS Lambda.

**Steps:**

1.  **Create a Lambda function** that contains your middleware logic.
2.  **Create an API Gateway** that triggers the Lambda function.
3.  **Configure the API Gateway** to forward requests to the Bedrock Agent.

**Example Lambda Middleware (Python):**

An API Gateway (REST, Lambda proxy integration) handler that authenticates via a Cognito authorizer, logs, caches in DynamoDB and calls the agent through `bedrock-agent-runtime`:

```python
import hashlib
import json
import logging
import os
import time

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

agent_runtime = boto3.client("bedrock-agent-runtime")
cache = boto3.resource("dynamodb").Table(os.environ["CACHE_TABLE"])

AGENT_ID = os.environ["AGENT_ID"]
AGENT_ALIAS_ID = os.environ["AGENT_ALIAS_ID"]
CACHE_TTL_SECONDS = 300


def middleware_handler(event, context):
    # 1. Authentication: API Gateway's Cognito authorizer has already validated the token
    claims = event["requestContext"]["authorizer"]["claims"]
    user_id = claims["sub"]
    if "agent-users" not in claims.get("cognito:groups", ""):
        return {"statusCode": 403, "body": json.dumps({"error": "forbidden"})}

    body = json.loads(event["body"])
    question, session_id = body["question"], body.get("sessionId", user_id)

    # 2. Logging (never log tokens or full PII)
    logger.info(json.dumps({"event": "request", "user": user_id, "session": session_id}))

    # 3. Caching: only for stateless, repeatable questions
    key = hashlib.sha256(question.strip().lower().encode()).hexdigest()
    cached = cache.get_item(Key={"pk": key}).get("Item")
    if cached and cached["expires_at"] > time.time():
        return {"statusCode": 200, "body": json.dumps({"answer": cached["answer"], "cached": True})}

    # 4. Invoke the Bedrock Agent (pass user context as session attributes, not in the prompt)
    response = agent_runtime.invoke_agent(
        agentId=AGENT_ID,
        agentAliasId=AGENT_ALIAS_ID,
        sessionId=session_id,
        inputText=question,
        sessionState={"sessionAttributes": {"userId": user_id}},
    )
    answer = "".join(
        e["chunk"]["bytes"].decode() for e in response["completion"] if "chunk" in e
    )

    # 5. Logging and 6. caching the response
    logger.info(json.dumps({"event": "response", "user": user_id, "chars": len(answer)}))
    cache.put_item(Item={"pk": key, "answer": answer, "expires_at": int(time.time()) + CACHE_TTL_SECONDS})

    # 7. Return the response to the user
    return {"statusCode": 200, "body": json.dumps({"answer": answer, "cached": False})}
```

`invoke_agent` returns an event stream, so the response must be read before it can be serialised. Session attributes reach action-group Lambdas in `event["sessionAttributes"]`, which is the safe way to pass identity to tools.

For code-defined agents (Strands on AgentCore Runtime), the same concerns are handled with **Strands hooks** (for example `BeforeToolCallEvent` / `AfterInvocationEvent`) inside the agent and **AgentCore Identity** JWT authorisation in front of it — see the [Strands guide](/amazon-bedrock-agents-guide/bedrock_strands_sdk_guide/#best-practices) and [AgentCore security](/amazon-bedrock-agents-guide/bedrock_agentcore_comprehensive_guide/#security-and-compliance).

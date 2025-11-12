# Advanced Error Handling and Recovery for Amazon Bedrock Agents (Python)

This guide covers advanced error handling and recovery patterns for building resilient and reliable Bedrock Agents.

## 1. Common Errors in Bedrock Agents

*   **ValidationException:** Invalid input parameters.
*   **AccessDeniedException:** Insufficient IAM permissions.
*   **ResourceNotFoundException:** The specified agent, knowledge base, or action group does not exist.
*   **ThrottlingException:** The request rate exceeds the limit.
*   **ServiceUnavailableException:** An internal server error occurred.
*   **DependencyFailedException:** An error occurred in a downstream service, such as a Lambda function or a knowledge base.

## 2. Error Handling Patterns

### a. Retry with Exponential Backoff

Implement a retry mechanism with exponential backoff to handle transient errors like `ThrottlingException` and `ServiceUnavailableException`.

```python
import time
from botocore.exceptions import ClientError

def invoke_agent_with_retry(agent_id, input_text, max_retries=3):
    for i in range(max_retries):
        try:
            response = bedrock_runtime.invoke_agent(
                agentId=agent_id,
                inputText=input_text,
            )
            return response
        except ClientError as e:
            if e.response['Error']['Code'] in ['ThrottlingException', 'ServiceUnavailableException']:
                time.sleep(2 ** i)
            else:
                raise
```

### b. Dead-Letter Queues (DLQs)

Use a Dead-Letter Queue (DLQ) to capture and analyze failed agent invocations. You can configure a DLQ on your Lambda functions or use an Amazon SQS queue.

### c. Custom Error Handling in Action Groups

Implement custom error handling in your action group Lambda functions to provide more meaningful error messages to the agent.

```python
def lambda_handler(event, context):
    try:
        # Your action group logic here
        pass
    except Exception as e:
        return {
            'error': {
                'type': 'CustomError',
                'message': str(e)
            }
        }
```

## 3. Recovery Patterns

### a. Checkpointing and Recovery

For long-running tasks, use the built-in checkpointing and recovery feature of AgentCore to save the agent's state and resume the task from the last checkpoint in case of a failure.

### b. Fallback Agents

Implement a fallback mechanism to a simpler agent or a human agent if the primary agent fails to complete a task.

```python
try:
    response = primary_agent.invoke(request)
except Exception as e:
    response = fallback_agent.invoke(request)
```
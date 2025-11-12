# Observability and Monitoring for Amazon Bedrock Agents (Python)

This guide provides a deep dive into the new AgentCore Observability solution and how to use it to monitor and debug your Bedrock Agents.

## 1. Amazon Bedrock AgentCore Observability

AgentCore Observability is a dedicated solution for monitoring, analyzing, and auditing AI agent interactions.

**Key Features:**

*   **OpenTelemetry (OTEL) Compatibility:** Emits telemetry data in a standardized OpenTelemetry-compatible format.
*   **CloudWatch Integration:** Provides native support for CloudWatch metrics, logs, and traces.
*   **Third-Party Integrations:** Integrates with third-party observability platforms like Datadog and Dynatrace.

## 2. Setting up Observability

### a. Enable CloudWatch Logs

Enable CloudWatch Logs for your agent to capture detailed information about each invocation.

### b. Use CloudWatch Metrics

Monitor key performance indicators (KPIs) such as:

*   `InvocationCount`
*   `InvocationErrors`
*   `InvocationLatency`
*   `InputTokenCount`
*   `OutputTokenCount`

### c. Enable X-Ray Tracing

Enable AWS X-Ray to trace agent requests as they travel through your application and downstream services.

## 3. Advanced Observability Patterns

### a. Custom Metrics

Publish custom metrics from your action group Lambda functions to CloudWatch to monitor business-specific KPIs.

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

def lambda_handler(event, context):
    # Your action group logic here
    
    cloudwatch.put_metric_data(
        Namespace='BedrockAgents',
        MetricData=[
            {
                'MetricName': 'SuccessfulBooking',
                'Value': 1,
                'Unit': 'Count'
            },
        ]
    )
```

### b. Structured Logging

Use structured logging in your Lambda functions to make your logs more searchable and analyzable.

```python
import json

def lambda_handler(event, context):
    log_data = {
        'agent_id': event['agent']['id'],
        'session_id': event['sessionId'],
        'action': event['actionGroup'],
        # ... other relevant data ...
    }
    print(json.dumps(log_data))
```

### c. Distributed Tracing with OpenTelemetry

Instrument your application with the OpenTelemetry SDK to create custom spans and traces for a more detailed view of your agent's performance.
```
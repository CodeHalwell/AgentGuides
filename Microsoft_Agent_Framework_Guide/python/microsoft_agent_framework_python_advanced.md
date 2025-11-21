---
layout: default
title: "Microsoft Agent Framework Advanced (Python)"
description: "Advanced multi-agent, resilience, and Azure deployment."
---

# Microsoft Agent Framework Advanced (Python)


Latest: 1.1.0
Upstream: https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/ai/azure-ai-agents | https://learn.microsoft.com/python/api/overview/azure/ai-agents-readme

## Focus
- Multi-agent orchestration with HITL
- Resilience: retries/timeouts/circuit breakers
- Azure: App Service/Functions, Key Vault, monitoring

## Azure AI Agents (Python) Example

```python
from azure.ai.agents import AgentsClient
from azure.identity import DefaultAzureCredential
import os

endpoint = os.environ["AZURE_AI_ENDPOINT"]
client = AgentsClient(endpoint=endpoint, credential=DefaultAzureCredential())

agent = client.create_agent(
    name="writer",
    model="gpt-4o-mini",
    instructions="You are a helpful writing assistant."
)

thread = client.create_thread()
client.create_message(thread.id, role="user", content="Summarize LangGraph in one paragraph.")
client.create_run(thread.id, agent.id)
messages = client.get_messages(thread.id)
print(messages[0].content[0].text)
```

## Production
- Managed identity and Key Vault for secrets
- OpenTelemetry instrumentation for client calls

## Deployment (Gunicorn + Kubernetes)

Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:8080", "--workers", "2"]
```

deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ms-agent-python
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ms-agent-python
  template:
    metadata:
      labels:
        app: ms-agent-python
    spec:
      containers:
        - name: app
          image: ghcr.io/yourorg/ms-agent-python:latest
          env:
            - name: AZURE_AI_ENDPOINT
              valueFrom:
                secretKeyRef:
                  name: azure-secrets
                  key: endpoint
          ports:
            - containerPort: 8080
```
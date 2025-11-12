---
layout: default
title: "Microsoft Agent Framework Advanced (.NET)"
description: "Advanced agent orchestration, resilience, and deployment patterns in .NET."
---

# Microsoft Agent Framework Advanced (.NET)


Latest: 1.1.0
Upstream: https://github.com/Azure/azure-sdk-for-net/tree/main/sdk/ai/Azure.AI.Agents.Persistent | https://learn.microsoft.com/dotnet/api/overview/azure/ai.agents.persistent-readme

## Topics
- Agent orchestration and HITL
- Polly resilience, telemetry with OTEL
- Azure deployments (AKS/App Service), identity, Key Vault

## Azure AI Persistent Agents (.NET) Example

```csharp
using Azure.AI.Agents.Persistent;
using Azure.Identity;

var endpoint = new Uri(Environment.GetEnvironmentVariable("AZURE_AI_ENDPOINT")!);
var cred = new DefaultAzureCredential();
var client = new AgentsClient(endpoint, cred);

// Create or get agent
var agent = await client.CreateAgentAsync(new CreateAgentOptions(
    name: "writer",
    model: "gpt-4o-mini",
    instructions: "You are a helpful writing assistant."
));

// Send message and get response
var thread = await client.CreateThreadAsync();
await client.CreateMessageAsync(thread.Value.Id, MessageRole.User, "Draft a summary of LangGraph.");
await client.CreateRunAsync(thread.Value.Id, agent.Value.Id);
var messages = await client.GetMessagesAsync(thread.Value.Id);
Console.WriteLine(messages.Value[0].Content[0].Text);
```

## Resilience & Telemetry
- Wrap client calls with Polly (retry + circuit-breaker)
- Add OpenTelemetry spans: agent.create, thread.create, run.create, messages.get

## Deployment (Kubernetes)

deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ms-agent-dotnet
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ms-agent-dotnet
  template:
    metadata:
      labels:
        app: ms-agent-dotnet
    spec:
      containers:
        - name: app
          image: ghcr.io/yourorg/ms-agent-dotnet:latest
          env:
            - name: AZURE_AI_ENDPOINT
              valueFrom:
                secretKeyRef:
                  name: azure-secrets
                  key: endpoint
          ports:
            - containerPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: ms-agent-dotnet
spec:
  selector:
    app: ms-agent-dotnet
  ports:
    - port: 80
      targetPort: 8080
```

## CI/CD (GitHub Actions)

```yaml
name: build-and-deploy
on:
  push:
    branches: [ main ]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-dotnet@v4
        with:
          dotnet-version: 8.0.x
      - run: dotnet publish -c Release -o out
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:latest
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: azure/k8s-deploy@v5
        with:
          action: deploy
          manifests: |
            deployment.yaml
          images: ghcr.io/${{ github.repository }}:latest
          namespace: default
```

## Security Best Practices
- Store keys in cloud secret managers (Key Vault/Secret Manager/Secrets Manager)
- Use least-privileged IAM roles; separate read/write roles for tools
- Network egress allowlisting for tool calls
- Encrypt logs; avoid sensitive data in traces
- Rotate keys regularly; monitor for anomalies

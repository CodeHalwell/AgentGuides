---
title: "Microsoft Agent Framework Advanced (.NET)"
description: "Advanced agent orchestration, resilience, and deployment patterns in .NET."
framework: "microsoft-agent-framework"
language: "dotnet"
---


Checked against `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% on %%versions-checked%%. The service-agent example below goes through `Microsoft.Agents.AI.AzureAI.Persistent`, which wraps the Azure AI Agents SDK (`Azure.AI.Agents.Persistent`).
Upstream: https://github.com/Azure/azure-sdk-for-net/tree/main/sdk/ai/Azure.AI.Agents.Persistent | https://learn.microsoft.com/dotnet/api/overview/azure/ai.agents.persistent-readme

## Topics
- Agent orchestration and HITL
- Polly resilience, telemetry with OTEL
- Azure deployments (AKS/App Service), identity, Key Vault

## Azure AI Persistent Agents (.NET) Example

Service-managed agents live in Azure AI Foundry; `Microsoft.Agents.AI.AzureAI.Persistent` (preview) bridges them into the framework as an ordinary `AIAgent`, so the session and streaming APIs are the same as for `ChatClientAgent`:

```bash
dotnet add package Microsoft.Agents.AI.AzureAI.Persistent --prerelease
dotnet add package Azure.Identity
```

```csharp
using Azure.AI.Agents.Persistent;
using Azure.Identity;
using Microsoft.Agents.AI;

var endpoint = Environment.GetEnvironmentVariable("AZURE_AI_PROJECT_ENDPOINT")!;
var client = new PersistentAgentsClient(endpoint, new DefaultAzureCredential());

// Create the server-side agent and get an AIAgent wrapper for it in one call.
// (Use client.GetAIAgentAsync(agentId) to attach to an agent that already exists.)
AIAgent agent = await client.CreateAIAgentAsync(
    model: "gpt-4o-mini",
    name: "writer",
    instructions: "You are a helpful writing assistant.");

AgentSession session = await agent.CreateSessionAsync();   // backed by a service-side thread
var response = await agent.RunAsync("Draft a summary of LangGraph.", session);
Console.WriteLine(response.Text);
```

An earlier draft of this page used an `AgentsClient` / `CreateAgentOptions` surface from a pre-release Azure SDK; the current client is `PersistentAgentsClient`.
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


---
title: "Microsoft Agent Framework (.NET) — Azure RBAC"
description: "Least-privilege Azure role assignments for agents that use Azure OpenAI, Microsoft Foundry, Key Vault, Cosmos DB, Storage and AI Search through managed identity."
framework: "microsoft-agent-framework"
language: "dotnet"
sidebar:
  label: "Azure RBAC"
  order: 86
---

With managed identity there are no keys to rotate: the agent's host (App Service, Container Apps, Functions, AKS workload identity, or a Foundry project) has an identity, and `DefaultAzureCredential` / `ManagedIdentityCredential` obtains tokens for it. What remains is giving that identity the smallest set of roles it needs. The same assignments apply to Python services.

## Which role for which call

| What the agent does | Resource | Role | Notes |
|---|---|---|---|
| Model calls through `AzureOpenAIClient` (`AsAIAgent`, `AsIChatClient`) | Azure OpenAI or Foundry resource | **Cognitive Services OpenAI User** | Inference only; cannot change deployments. |
| Model calls and agents through `AIProjectClient` / `Microsoft.Agents.AI.Foundry` | Foundry resource or project | **Foundry User** (formerly *Azure AI User*) | Build and run agents in the project. Assign by role ID while the rename rolls out. |
| Only calling published Foundry agents (Responses API) | Foundry project or a single agent | **Foundry Agent Consumer** | Least privilege for callers that don't create or modify agents. |
| Read secrets for third-party tools | Key Vault | **Key Vault Secrets User** | Vault must use the Azure RBAC permission model. |
| Store sessions or chat history (`Microsoft.Agents.AI.CosmosNoSql`) | Cosmos DB account | **Cosmos DB Built-in Data Contributor** | A Cosmos DB data-plane role, assigned with `az cosmosdb sql role assignment`. |
| Read documents for RAG | Storage account | **Storage Blob Data Reader** | Use *Contributor* only if the agent writes blobs. |
| Query an Azure AI Search index (`TextSearchProvider`, search tools) | AI Search service | **Search Index Data Reader** | Turn off key authentication on the service once RBAC works. |
| Content Safety / Prompt Shields from middleware | Content Safety or Foundry resource | **Cognitive Services User** | |
| Send telemetry with Entra authentication | Application Insights | **Monitoring Metrics Publisher** | Only if you disable local (key-based) ingestion. |

People need roles too: developers usually get **Foundry User** on the project, and only the platform team needs **Foundry Owner** or **Foundry Account Owner**.

## Assigning the roles with the Azure CLI

```bash
# The identity of the host, e.g. a Container App with a system-assigned identity
PRINCIPAL_ID=$(az containerapp show -n my-agent -g my-rg --query identity.principalId -o tsv)

# Azure OpenAI inference
az role assignment create --assignee-object-id "$PRINCIPAL_ID" --assignee-principal-type ServicePrincipal \
  --role "Cognitive Services OpenAI User" --scope "$OPENAI_RESOURCE_ID"

# Foundry User, by role definition ID (stable across the Azure AI -> Foundry rename)
az role assignment create --assignee-object-id "$PRINCIPAL_ID" --assignee-principal-type ServicePrincipal \
  --role "53ca6127-db72-4b80-b1b0-d745d6d5456d" --scope "$FOUNDRY_PROJECT_ID"

# Key Vault secrets
az role assignment create --assignee-object-id "$PRINCIPAL_ID" --assignee-principal-type ServicePrincipal \
  --role "Key Vault Secrets User" --scope "$KEYVAULT_ID"

# Cosmos DB data plane (built-in data contributor, role ID ...0002), scoped to one database
az cosmosdb sql role assignment create --account-name my-cosmos -g my-rg \
  --principal-id "$PRINCIPAL_ID" \
  --role-definition-id 00000000-0000-0000-0000-000000000002 \
  --scope "/dbs/agents"
```

Scope each assignment to the single resource (or project, or Cosmos DB database) rather than the resource group. Role assignments take a few minutes to propagate, so a `403` straight after deployment is usually timing.

## In code

Nothing in the agent changes between environments: `DefaultAzureCredential` uses your `az login` locally and the managed identity in Azure. In production, prefer `ManagedIdentityCredential` (pass the client ID for a user-assigned identity) so the credential chain doesn't probe other sources.

```csharp
using Azure.AI.OpenAI;
using Azure.Core;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

TokenCredential credential = Environment.GetEnvironmentVariable("AZURE_CLIENT_ID") is { } clientId
    ? new ManagedIdentityCredential(ManagedIdentityId.FromUserAssignedClientId(clientId))
    : new DefaultAzureCredential();

AIAgent agent = new AzureOpenAIClient(new Uri(Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")!), credential)
    .GetChatClient("%%model=openai-small%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You are a helpful assistant.");
```

Then enforce it: the Azure Policy definition *Azure AI services resources should have key access disabled* stops anyone falling back to API keys. Identity patterns for users (on-behalf-of, multi-tenant) are on the [enterprise page](/microsoft-agent-framework-guide/microsoft_agent_framework_enterprise_2025/), and the full role reference is Microsoft's [Foundry RBAC page](https://learn.microsoft.com/azure/foundry/concepts/rbac-foundry).

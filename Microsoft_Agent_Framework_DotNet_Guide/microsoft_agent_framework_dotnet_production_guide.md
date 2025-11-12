# Microsoft Agent Framework .NET - Production Guide

This guide provides enterprise-level best practices for deploying, managing, and scaling applications built with the Microsoft Agent Framework for .NET.

**Target Audience:** DevOps Engineers, Infrastructure Architects, Senior .NET Developers  
**Platform:** .NET 8.0+

---

## 1. Production Deployment

### Recommended Host: Azure Container Apps

For most scenarios, **Azure Container Apps (ACA)** provides the best balance of scalability, ease of use, and power for hosting agent applications. It offers serverless containers, built-in Dapr integration for microservices, and KEDA-based scaling.

### Strategy 1: Docker & Azure Container Apps

**Step 1: Dockerize Your Application**

Create a `Dockerfile` in your project root.

```dockerfile
# Use the official .NET 8 SDK image to build
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY ["MyAgentApp.csproj", "."]
RUN dotnet restore "./MyAgentApp.csproj"
COPY . .
RUN dotnet build "MyAgentApp.csproj" -c Release -o /app/build

# Create the final runtime image
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS publish
WORKDIR /app
COPY --from=build /app/build .
ENTRYPOINT ["dotnet", "MyAgentApp.dll"]
```

**Step 2: Build and Push to Azure Container Registry (ACR)**

```bash
# Log in to Azure
az login

# Create an Azure Container Registry (if you don't have one)
az acr create --resource-group MyResourceGroup --name myagentregistry --sku Basic

# Build and push the image
az acr build --registry myagentregistry --image my-agent-app:v1 .
```

**Step 3: Deploy to Azure Container Apps**

Use a Bicep or ARM template for infrastructure-as-code.

```bicep
// deploy.bicep
param location string = resourceGroup().location
param acrName string = 'myagentregistry'
param appName string = 'my-agent-service'

resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' existing = {
  name: acrName
}

resource containerAppEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: '${appName}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
    }
  }
}

resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: appName
  location: location
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8080
      }
      secrets: [
        {
          name: 'openai-key'
          value: 'YOUR_OPENAI_KEY' // Use Key Vault reference in production
        }
      ]
      registries: [
        {
          server: acr.properties.loginServer
          identity: 'system' // Use managed identity to pull from ACR
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'agent-container'
          image: '${acr.properties.loginServer}/my-agent-app:v1'
          resources: {
            cpu: json('1.0')
            memory: '2.0Gi'
          }
          env: [
            {
              name: 'ASPNETCORE_URLS'
              value: 'http://+:8080'
            }
            {
              name: 'AZURE_OPENAI_KEY'
              secretRef: 'openai-key'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 10
        rules: [
          {
            name: 'http-scaling-rule'
            http: {
              metadata: {
                concurrentRequests: '100' // Scale up when 100 concurrent requests are hit
              }
            }
          }
        ]
      }
    }
  }
  identity: {
    type: 'SystemAssigned'
  }
}
```

### Strategy 2: CI/CD with Azure DevOps or GitHub Actions

Automate your deployment process to ensure consistency and reliability.

**GitHub Actions Workflow Example:**

{% raw %}
```yaml
# .github/workflows/deploy.yml
name: Deploy Agent to ACA

on:
  push:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: 'Az CLI login'
      uses: azure/login@v1
      with:
        creds: ${{ secrets.AZURE_CREDENTIALS }}

    - name: 'Build and push image'
      run: |
        az acr build --registry myagentregistry --image my-agent-app:${{ github.sha }} .

    - name: 'Deploy to Azure Container Apps'
      uses: azure/container-apps-deploy-action@v1
      with:
        imageToDeploy: myagentregistry.azurecr.io/my-agent-app:${{ github.sha }}
        containerAppName: my-agent-service
        resourceGroup: MyResourceGroup
```
{% endraw %}

---

## 2. Scaling Strategies

### Horizontal Scaling (Scale Out)

-   **Stateless Agents:** Design your agents to be as stateless as possible. Persist conversation history and state to an external store (like Cosmos DB or Redis). This allows you to add or remove agent instances without losing data.
-   **KEDA-Based Autoscaling:** Use the scaling rules in Azure Container Apps to automatically scale based on metrics like HTTP requests, CPU/memory usage, or queue depth.

### Caching Strategies

-   **LLM Response Caching:** Cache responses for identical prompts to reduce latency and cost. Use a distributed cache like Azure Cache for Redis.
-   **Memory Caching:** Cache frequently accessed user profiles or memory objects to reduce database load.

```csharp
// Program.cs - Adding Redis for distributed caching
builder.Services.AddStackExchangeRedisCache(options =>
{
    options.Configuration = builder.Configuration.GetConnectionString("Redis");
    options.InstanceName = "AgentCache_";
});
```

---

## 3. Monitoring & Observability

### Structured Logging with Application Insights

Configure your application to use structured logging. This allows for powerful querying and alerting in Application Insights.

```csharp
// Program.cs
builder.Logging.AddApplicationInsights(
    configureTelemetryConfiguration: (config) => 
        config.ConnectionString = builder.Configuration["APPLICATIONINSIGHTS_CONNECTION_STRING"],
    configureApplicationInsightsLoggerOptions: (options) => { }
);
```

**Log with context in your agents:**

```csharp
// Inside an agent or tool
private readonly ILogger<MyTool> _logger;

public MyTool(ILogger<MyTool> logger) { _logger = logger; }

public void DoWork(string threadId)
{
    using (_logger.BeginScope(new Dictionary<string, object> { ["ThreadId"] = threadId }))
    {
        _logger.LogInformation("Starting work for thread.");
        // ...
        _logger.LogWarning("An issue occurred but was handled.");
    }
}
```

### Custom Metrics and Tracing

-   **Custom Metrics:** Use Application Insights or Prometheus to track key business and performance metrics (e.g., `agent_invocations_total`, `tool_calls_per_minute`, `average_token_count`).
-   **Distributed Tracing:** The framework is designed to integrate with OpenTelemetry. Tracing allows you to visualize the entire lifecycle of a request as it flows through multiple agents and services.

---

## 4. Security Best Practices

### Secrets Management with Azure Key Vault

Never store secrets (API keys, connection strings) in code or configuration files.

```csharp
// Program.cs - Integrating Azure Key Vault
var keyVaultEndpoint = new Uri(Environment.GetEnvironmentVariable("KEY_VAULT_ENDPOINT"));
builder.Configuration.AddAzureKeyVault(keyVaultEndpoint, new DefaultAzureCredential());

// Now you can access secrets via IConfiguration
var apiKey = builder.Configuration["AzureOpenAIApiKey"];
```

### Network Security

-   **VNet Integration:** Deploy your Container Apps and dependent services (like databases and Key Vault) into a virtual network to isolate them from the public internet.
-   **Private Endpoints:** Use private endpoints to access Azure services securely over the Azure backbone network.

### Authentication and Authorization

-   **Managed Identity:** Use managed identities for your Azure resources to authenticate with other Azure services without needing to manage credentials.
-   **API Authentication:** Secure your agent's public endpoints using OAuth 2.0 (e.g., with Microsoft Entra ID).

---

## 5. High Availability & Disaster Recovery

### Multi-Region Deployment

For critical applications, deploy your agent services to multiple Azure regions. Use Azure Traffic Manager or Azure Front Door to route traffic to the nearest or healthiest region.

### Database and State Replication

-   **Cosmos DB:** Use the globally distributed nature of Cosmos DB to replicate your agent state across multiple regions automatically.
-   **Azure SQL:** Configure active geo-replication to maintain a readable secondary database in a different region.

### Resiliency with Polly

Use the Polly library to implement resiliency patterns like retries, circuit breakers, and fallbacks for external dependencies (LLM calls, tool APIs).

```csharp
// Add Polly to DI container
builder.Services.AddHttpClient("OpenAI")
    .AddPolicyHandler(
        HttpPolicyExtensions
            .HandleTransientHttpError()
            .OrResult(msg => msg.StatusCode == System.Net.HttpStatusCode.TooManyRequests)
            .WaitAndRetryAsync(3, retryAttempt => TimeSpan.FromSeconds(Math.Pow(2, retryAttempt)))
    )
    .AddPolicyHandler(
        HttpPolicyExtensions
            .HandleTransientHttpError()
            .CircuitBreakerAsync(5, TimeSpan.FromSeconds(30))
    );

// Use the named HttpClient in your services
public class MyLLMService
{
    private readonly HttpClient _httpClient;
    public MyLLMService(IHttpClientFactory httpClientFactory)
    {
        _httpClient = httpClientFactory.CreateClient("OpenAI");
    }
}
```

---

## 6. Cost Optimization

-   **Model Selection:** Use smaller, cheaper models (e.g., GPT-4o-mini) for simple tasks like classification or routing, and reserve larger models for complex reasoning.
-   **Prompt Engineering:** Optimize your prompts to be as concise as possible while still achieving the desired output.
-   **Caching:** Aggressively cache responses from LLMs and tools.
-   **Monitor Token Usage:** Log the token count for every LLM call. Create alerts in Application Insights to notify you of unusually high token usage, which could indicate a bug or prompt injection attack.

---

## 7. Performance Tuning

-   **Asynchronous Programming:** Use `async` and `await` for all I/O-bound operations (LLM calls, database queries, API calls) to keep your application responsive.
-   **Batching:** When processing multiple requests, batch them together where possible. This is especially effective for calls to embedding models.
-   **Connection Pooling:** Ensure your database connections are being pooled correctly to avoid the overhead of establishing new connections for every request.

---

## 8. Enterprise Governance

-   **Azure Policy:** Use Azure Policy to enforce organizational standards, such as requiring that all resources have specific tags, are deployed in certain regions, or have diagnostic logging enabled.
-   **Auditing:** Log all agent actions and decisions to a secure, immutable store for auditing and compliance purposes.
-   **Data Governance:** Use tools like Microsoft Purview to classify and govern the data that your agents interact with, especially in RAG scenarios.

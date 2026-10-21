---
title: "Semantic Kernel Production Guide (.NET)"
description: "Production deployment, resilience, monitoring and testing for Semantic Kernel .NET applications."
framework: "semantic-kernel"
language: "dotnet"
sidebar:
  label: "Production"
  order: 3
---

> Targets `Microsoft.SemanticKernel` %%version=Microsoft.SemanticKernel%% on .NET 8 or .NET 10 (checked %%versions-checked%%). Semantic Kernel is supported but its successor, [Microsoft Agent Framework](/microsoft-agent-framework-guide/dotnet/), is where new agent features land; the hosting, resilience and telemetry patterns here apply to both.

---

This page is the .NET code for the [production guide](/semantic-kernel-guide/semantic_kernel_production_guide/), which covers deployment shape, security, scaling, cost and the release checklist. The hosting pattern itself (singleton services, transient `Kernel`) is in [recipe 1](/semantic-kernel-guide/dotnet/semantic_kernel_recipes_dotnet/#1-aspnet-core-minimal-api-with-a-transient-kernel).

---

## Container

### Dockerfile

```dockerfile
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

COPY ["SkApp/SkApp.csproj", "SkApp/"]
RUN dotnet restore "SkApp/SkApp.csproj"

COPY . .
WORKDIR "/src/SkApp"
RUN dotnet publish "SkApp.csproj" -c Release -o /app/publish --no-restore

FROM mcr.microsoft.com/dotnet/aspnet:10.0
WORKDIR /app
COPY --from=build /app/publish .

# .NET 8+ images listen on 8080 and run as the non-root 'app' user by default
USER app
EXPOSE 8080

ENTRYPOINT ["dotnet", "SkApp.dll"]
```

---

## Resilience

The OpenAI and Azure OpenAI connectors sit on the official OpenAI .NET SDK, which already retries 429 and 5xx responses with back-off. Add policy on top only where you need it, using Polly v8 via `Microsoft.Extensions.Http.Resilience` rather than the Polly v7 `Policy.Handle<...>()` API.

### HTTP-level: the standard resilience handler

```csharp
// dotnet add package Microsoft.Extensions.Http.Resilience
builder.Services.ConfigureHttpClientDefaults(http => http.AddStandardResilienceHandler(o =>
{
    o.AttemptTimeout.Timeout = TimeSpan.FromSeconds(60);       // LLM calls are slow
    o.TotalRequestTimeout.Timeout = TimeSpan.FromMinutes(3);
    o.CircuitBreaker.SamplingDuration = TimeSpan.FromMinutes(2);
}));
```

### Call-level: a resilience pipeline around a kernel function

```csharp
// dotnet add package Polly.Core
using Polly;
using Polly.Retry;
using Microsoft.SemanticKernel;

public sealed class ResilientKernelService(Kernel kernel, ILogger<ResilientKernelService> logger)
{
    private readonly ResiliencePipeline _pipeline = new ResiliencePipelineBuilder()
        .AddRetry(new RetryStrategyOptions
        {
            ShouldHandle = new PredicateBuilder().Handle<HttpOperationException>(ex =>
                ex.StatusCode is System.Net.HttpStatusCode.TooManyRequests or >= System.Net.HttpStatusCode.InternalServerError),
            MaxRetryAttempts = 3,
            BackoffType = DelayBackoffType.Exponential,
            UseJitter = true,
            OnRetry = args =>
            {
                logger.LogWarning("Retry {Attempt} after {Delay}", args.AttemptNumber, args.RetryDelay);
                return default;
            },
        })
        .AddCircuitBreaker(new Polly.CircuitBreaker.CircuitBreakerStrategyOptions
        {
            FailureRatio = 0.5,
            MinimumThroughput = 10,
            BreakDuration = TimeSpan.FromSeconds(30),
        })
        .Build();

    public async Task<string?> InvokeAsync(KernelFunction function, KernelArguments arguments, CancellationToken ct = default) =>
        await _pipeline.ExecuteAsync(async token => await kernel.InvokeAsync<string>(function, arguments, token), ct);
}
```

Do not retry on 400-class errors other than 429 — they will fail again and cost tokens.

---

## Monitoring with OpenTelemetry

Semantic Kernel emits OpenTelemetry traces, metrics and logs (sources and meters named `Microsoft.SemanticKernel*`), including GenAI semantic-convention spans for model calls when the diagnostics switch is on. Application Insights consumes them through the Azure Monitor distro; there is no need to hand-roll `TelemetryClient` operations.

```csharp
// dotnet add package Azure.Monitor.OpenTelemetry.AspNetCore
using Azure.Monitor.OpenTelemetry.AspNetCore;

AppContext.SetSwitch("Microsoft.SemanticKernel.Experimental.GenAI.EnableOTelDiagnostics", true);
// "...EnableOTelDiagnosticsSensitive" also records prompts and completions — not in production

builder.Services.AddOpenTelemetry()
    .UseAzureMonitor()                                   // reads APPLICATIONINSIGHTS_CONNECTION_STRING
    .WithTracing(t => t.AddSource("Microsoft.SemanticKernel*"))
    .WithMetrics(m => m.AddMeter("Microsoft.SemanticKernel*"));
```

Useful signals: token usage per function (`semantic_kernel.function.invocation.token_usage.prompt` / `.completion`, tagged with function name and model ID; the Python SDK does not emit these), function duration (`semantic_kernel.function.invocation.duration`), model latency, and failed function calls. Alert on sudden token-usage growth — it is the usual first sign of a prompt or loop regression.

---

## Testing with xUnit

```csharp
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Moq;
using Xunit;

public class KernelServiceTests
{
    [Fact]
    public async Task InvokeAsync_ReturnsExpectedResult()
    {
        // Arrange
        var mockService = new Mock<IChatCompletionService>();
        mockService.Setup(s => s.GetChatMessageContentsAsync(
            It.IsAny<ChatHistory>(),
            It.IsAny<PromptExecutionSettings>(),
            It.IsAny<Kernel>(),
            It.IsAny<CancellationToken>()
        )).ReturnsAsync(new List<ChatMessageContent> {
            new(AuthorRole.Assistant, "Test response")
        });

        var builder = Kernel.CreateBuilder();
        builder.Services.AddSingleton(mockService.Object);
        var kernel = builder.Build();

        var function = kernel.CreateFunctionFromPrompt("Test");

        // Act
        var result = await kernel.InvokeAsync<string>(function);

        // Assert
        Assert.Equal("Test response", result);
    }
}
```

Mocking `IChatCompletionService` keeps unit tests offline. For agent tests, the same mocked service can back a `ChatCompletionAgent` through its `Kernel`.

---

## Security settings worth checking

- **Template encoding stays on.** Do not set `AllowDangerouslySetContent` on templates or variables that carry user input, documents or tool output ([details](/semantic-kernel-guide/dotnet/semantic_kernel_comprehensive_dotnet/#43-input-encoding)).
- **Advertise only the tools a call needs**: `FunctionChoiceBehavior.Auto(functions: [kernel.Plugins.GetFunction("orders", "Status")])`.
- **Gate destructive tools** with an `IAutoFunctionInvocationFilter` that returns a refusal instead of calling `next` ([example](/semantic-kernel-guide/dotnet/semantic_kernel_comprehensive_dotnet/#12-filters)).
- **Sensitive diagnostics off**: leave `Microsoft.SemanticKernel.Experimental.GenAI.EnableOTelDiagnosticsSensitive` unset outside development.

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-21 | 1.81.0 | Overlap with the language-neutral production guide removed; test usings completed; security settings added. All blocks compile and the test passes against 1.81.0. |

**[Back to .NET overview](/semantic-kernel-guide/dotnet/)**


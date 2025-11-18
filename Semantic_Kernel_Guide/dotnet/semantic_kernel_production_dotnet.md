# Semantic Kernel Production Guide (.NET)

**Production Deployment, Monitoring, and Best Practices for .NET**

Last Updated: November 2025
.NET Version: 6.0+
Semantic Kernel .NET: 1.38.0+

---

## Overview

Production deployment of Semantic Kernel .NET applications including Docker/Kubernetes deployment, monitoring, security, performance optimization.

**See Also:** [../semantic_kernel_production_guide.md](../semantic_kernel_production_guide.md) for language-agnostic patterns.

---

## Docker Containerization

### Dockerfile

```dockerfile
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

COPY ["SkApp/SkApp.csproj", "SkApp/"]
RUN dotnet restore "SkApp/SkApp.csproj"

COPY . .
WORKDIR "/src/SkApp"
RUN dotnet build "SkApp.csproj" -c Release -o /app/build
RUN dotnet publish "SkApp.csproj" -c Release -o /app/publish

FROM mcr.microsoft.com/dotnet/aspnet:8.0
WORKDIR /app
COPY --from=build /app/publish .

EXPOSE 80
EXPOSE 443

ENTRYPOINT ["dotnet", "SkApp.dll"]
```

---

## Polly Resilience Policies

```csharp
using Polly;
using Polly.CircuitBreaker;
using Polly.Retry;

public class ResilientKernelService
{
    private readonly Kernel _kernel;
    private readonly AsyncRetryPolicy _retryPolicy;
    private readonly AsyncCircuitBreakerPolicy _circuitBreakerPolicy;

    public ResilientKernelService(Kernel kernel)
    {
        _kernel = kernel;

        // Retry policy
        _retryPolicy = Policy
            .Handle<HttpRequestException>()
            .Or<TimeoutException>()
            .WaitAndRetryAsync(
                retryCount: 3,
                sleepDurationProvider: attempt => TimeSpan.FromSeconds(Math.Pow(2, attempt)),
                onRetry: (exception, timeSpan, retryCount, context) =>
                {
                    Console.WriteLine($"Retry {retryCount} after {timeSpan.TotalSeconds}s");
                }
            );

        // Circuit breaker
        _circuitBreakerPolicy = Policy
            .Handle<HttpRequestException>()
            .CircuitBreakerAsync(
                exceptionsAllowedBeforeBreaking: 5,
                durationOfBreak: TimeSpan.FromMinutes(1),
                onBreak: (exception, duration) =>
                {
                    Console.WriteLine($"Circuit broken for {duration.TotalSeconds}s");
                },
                onReset: () => Console.WriteLine("Circuit reset")
            );
    }

    public async Task<string> InvokeWithResilienceAsync(
        KernelFunction function,
        KernelArguments arguments)
    {
        return await _circuitBreakerPolicy.ExecuteAsync(async () =>
            await _retryPolicy.ExecuteAsync(async () =>
                await _kernel.InvokeAsync<string>(function, arguments)
            )
        );
    }
}
```

---

## Application Insights Integration

```csharp
using Microsoft.ApplicationInsights;
using Microsoft.ApplicationInsights.DataContracts;

public class MonitoredKernelService
{
    private readonly Kernel _kernel;
    private readonly TelemetryClient _telemetry;

    public MonitoredKernelService(Kernel kernel, TelemetryClient telemetry)
    {
        _kernel = kernel;
        _telemetry = telemetry;
    }

    public async Task<string> InvokeWithMonitoringAsync(
        KernelFunction function,
        KernelArguments arguments)
    {
        var operation = _telemetry.StartOperation<RequestTelemetry>("SK_Invoke");
        operation.Telemetry.Properties["function"] = function.Name;

        try
        {
            var result = await _kernel.InvokeAsync<string>(function, arguments);

            operation.Telemetry.Success = true;
            return result;
        }
        catch (Exception ex)
        {
            operation.Telemetry.Success = false;
            _telemetry.TrackException(ex);
            throw;
        }
        finally
        {
            _telemetry.StopOperation(operation);
        }
    }
}
```

---

## Testing with xUnit

```csharp
using Xunit;
using Moq;

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
        Assert.NotNull(result);
        Assert.Contains("response", result);
    }
}
```

---

For complete production patterns, see [../semantic_kernel_production_guide.md](../semantic_kernel_production_guide.md)

**[Back to .NET README](README.md)** | **[Complete Index](GUIDE_INDEX.md)**

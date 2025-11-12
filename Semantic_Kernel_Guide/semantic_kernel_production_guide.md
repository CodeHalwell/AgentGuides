
# Semantic Kernel: Production Deployment and Operations Guide

**Version:** 1.0  
**Focus:** Deployment, Performance, Monitoring, Security, and Production Patterns

---

## Table of Contents

1. [Deployment Strategies](#1-deployment-strategies)
2. [Performance Optimization](#2-performance-optimization)
3. [Monitoring and Observability](#3-monitoring-and-observability)
4. [Security Best Practices](#4-security-best-practices)
5. [Error Handling and Resilience](#5-error-handling-and-resilience)
6. [Scaling and Load Management](#6-scaling-and-load-management)
7. [Cost Management](#7-cost-management)
8. [Testing Strategies](#8-testing-strategies)
9. [Production Patterns](#9-production-patterns)

---

## 1. Deployment Strategies

### 1.1 .NET Deployment

#### Azure Container Instances (ACI)

```dockerfile
# Dockerfile
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app
COPY --from=build /app/publish .

ENV AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
ENV AZURE_OPENAI_DEPLOYMENT=gpt-4-deployment
ENV ASPNETCORE_URLS=http://+:8080

EXPOSE 8080
ENTRYPOINT ["dotnet", "SemanticKernelApp.dll"]
```

```bash
# Deploy to Azure Container Instances
az container create \
  --resource-group myResourceGroup \
  --name sk-app \
  --image myregistry.azurecr.io/sk-app:latest \
  --environment-variables \
    AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT \
    AZURE_OPENAI_API_KEY=$AZURE_OPENAI_API_KEY \
  --ports 8080 \
  --dns-name-label sk-app
```

#### Azure App Service

```csharp
// Program.cs for App Service deployment
using Microsoft.SemanticKernel;
using Azure.Identity;
using Microsoft.Extensions.Azure;

var builder = WebApplication.CreateBuilder(args);

// Configure services
builder.Services.AddControllers();
builder.Services.AddSemanticKernel();

// Add Azure services
builder.Services.AddAzureClients(clientBuilder =>
{
    clientBuilder.AddOpenAIClient();
    clientBuilder.AddAzureSearchDocumentIndexClient();
});

// Build Semantic Kernel
var kernelBuilder = Kernel.CreateBuilder();
kernelBuilder.AddAzureOpenAIChatCompletion(
    deploymentName: builder.Configuration["Azure:OpenAI:Deployment"]!,
    endpoint: new Uri(builder.Configuration["Azure:OpenAI:Endpoint"]!),
    credential: new DefaultAzureCredential()
);

var app = builder.Build();
app.MapControllers();
app.Run();
```

#### Azure Functions

```csharp

// Azure Function with Semantic Kernel
using Azure.Identity;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel;

public class SemanticKernelFunction
{
    private readonly Kernel _kernel;
    private readonly ILogger<SemanticKernelFunction> _logger;

    public SemanticKernelFunction(Kernel kernel, ILogger<SemanticKernelFunction> logger)
    {
        _kernel = kernel;
        _logger = logger;
    }

    [Function("ProcessText")]
    public async Task<string> Run(
        [HttpTrigger(AuthorizationLevel.Function, "post", Route = null)] HttpRequestData req,
        FunctionContext executionContext)
    {
        _logger.LogInformation("Processing request...");

        var requestBody = await req.Body.ReadAsAsync<dynamic>();
        var text = requestBody?.text ?? "";

        try
        {
            var function = _kernel.CreateFunctionFromPrompt(
                prompt: "Summarise the following: {{$input}}",
                functionName: "Summarise"
            );

            var result = await _kernel.InvokeAsync(function, new() { ["input"] = text });

            var response = req.CreateResponse(System.Net.HttpStatusCode.OK);
            await response.WriteAsJsonAsync(new { result = result.ToString() });
            return response;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing request");
            var response = req.CreateResponse(System.Net.HttpStatusCode.InternalServerError);
            await response.WriteAsJsonAsync(new { error = ex.Message });
            return response;
        }
    }
}

```

### 1.2 Python Deployment

#### Container Deployment

```dockerfile
# Dockerfile for Python
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
ENV AZURE_OPENAI_API_KEY=your-api-key
ENV FLASK_APP=app.py

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
```

```python

# app.py - Production Flask app with SK
from flask import Flask, request, jsonify
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.azure_open_ai import AzureOpenAIChatCompletion
import logging
import os
from functools import wraps

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialise Kernel
def create_kernel():
    kernel = Kernel()
    kernel.add_service(
        AzureOpenAIChatCompletion(
            deployment_id=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
            endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY")
        )
    )
    return kernel

kernel = create_kernel()

# Error handling decorator
def handle_errors(f):
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        try:
            return await f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    return decorated_function

@app.route("/api/process", methods=["POST"])
def process_text():
    """Process text using Semantic Kernel"""
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "Text is required"}), 400

        function = kernel.create_function_from_prompt(
            prompt="Summarise: {{$input}}",
            function_name="Summarise"
        )

        result = kernel.invoke_async(function, input=text)

        return jsonify({"result": str(result)})

    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

```

---

## 2. Performance Optimization

### 2.1 Response Caching

```csharp
// .NET Response Caching Strategy
using Microsoft.SemanticKernel;
using System.Collections.Concurrent;

public class CachedKernelService
{
    private readonly Kernel _kernel;
    private readonly ConcurrentDictionary<string, string> _cache;
    private readonly ILogger _logger;

    public CachedKernelService(Kernel kernel, ILogger logger)
    {
        _kernel = kernel;
        _logger = logger;
        _cache = new ConcurrentDictionary<string, string>();
    }

    public async Task<string> InvokeWithCacheAsync(
        string pluginName,
        string functionName,
        string cacheKey,
        Dictionary<string, object> arguments,
        TimeSpan? cacheDuration = null)
    {
        // Check cache
        if (_cache.TryGetValue(cacheKey, out var cachedResult))
        {
            _logger.LogInformation($"Cache hit for {cacheKey}");
            return cachedResult;
        }

        // Invoke function
        var result = await _kernel.InvokeAsync(
            pluginName,
            functionName,
            new KernelArguments(arguments)
        );

        var resultString = result.ToString() ?? "";

        // Store in cache
        _cache.TryAdd(cacheKey, resultString);

        // Optionally expire after duration
        if (cacheDuration.HasValue)
        {
            _ = Task.Delay(cacheDuration.Value).ContinueWith(_ =>
            {
                _cache.TryRemove(cacheKey, out _);
            });
        }

        return resultString;
    }
}
```

```python
# Python Response Caching
from functools import lru_cache
from datetime import datetime, timedelta
from typing import Dict
import hashlib

class CachedKernelService:
    def __init__(self, kernel, cache_ttl_seconds: int = 300):
        self.kernel = kernel
        self.cache_ttl = cache_ttl_seconds
        self.cache: Dict[str, tuple] = {}  # key -> (value, expires_at)
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def invoke_with_cache(
        self,
        plugin_name: str,
        function_name: str,
        **kwargs
    ) -> str:
        """Invoke with caching"""
        cache_key = self._generate_cache_key(plugin_name, function_name, **kwargs)
        
        # Check cache
        if cache_key in self.cache:
            result, expires_at = self.cache[cache_key]
            if datetime.now() < expires_at:
                return result
            else:
                del self.cache[cache_key]
        
        # Invoke
        result = await self.kernel.invoke_async(
            plugin_name,
            function_name,
            **kwargs
        )
        
        # Store in cache
        expires_at = datetime.now() + timedelta(seconds=self.cache_ttl)
        self.cache[cache_key] = (str(result), expires_at)
        
        return str(result)
```

### 2.2 Batch Processing

```csharp
// .NET Batch Processing
public class BatchKernelProcessor
{
    private readonly Kernel _kernel;
    private readonly int _batchSize;
    private readonly ILogger _logger;

    public BatchKernelProcessor(Kernel kernel, int batchSize = 10, ILogger logger = null)
    {
        _kernel = kernel;
        _batchSize = batchSize;
        _logger = logger;
    }

    public async Task<List<string>> ProcessBatchAsync<T>(
        List<T> items,
        string pluginName,
        string functionName,
        Func<T, KernelArguments> argumentsBuilder)
    {
        var results = new List<string>();
        var batches = items
            .Select((item, index) => new { item, index })
            .GroupBy(x => x.index / _batchSize)
            .Select(g => g.Select(x => x.item).ToList())
            .ToList();

        foreach (var batch in batches)
        {
            _logger?.LogInformation($"Processing batch of {batch.Count} items");

            var tasks = batch.Select(async item =>
            {
                try
                {
                    var arguments = argumentsBuilder(item);
                    var result = await _kernel.InvokeAsync(
                        pluginName,
                        functionName,
                        arguments
                    );
                    return result.ToString() ?? "";
                }
                catch (Exception ex)
                {
                    _logger?.LogError($"Error processing item: {ex.Message}");
                    return $"Error: {ex.Message}";
                }
            });

            var batchResults = await Task.WhenAll(tasks);
            results.AddRange(batchResults);

            // Add delay between batches to avoid throttling
            await Task.Delay(100);
        }

        return results;
    }
}
```

---

## 3. Monitoring and Observability

### 3.1 Logging Configuration

```csharp
// .NET Logging with Application Insights
using Microsoft.ApplicationInsights.Extensibility;
using Microsoft.Extensions.Logging;

var builder = WebApplication.CreateBuilder(args);

// Add Application Insights
builder.Services.AddApplicationInsightsTelemetry(options =>
{
    options.EnableAdaptiveSampling = false;
});

// Add logging
builder.Logging.AddApplicationInsights(telemetryConfiguration =>
{
    telemetryConfiguration.TelemetryInitializers.Add(
        new HttpClientTelemetryInitializer()
    );
});

builder.Logging.SetMinimumLevel(LogLevel.Information);

// Custom telemetry tracking
public class SemanticKernelTelemetryService
{
    private readonly TelemetryClient _telemetryClient;

    public SemanticKernelTelemetryClient(TelemetryClient telemetryClient)
    {
        _telemetryClient = telemetryClient;
    }

    public async Task<T> TrackFunctionInvocationAsync<T>(
        string pluginName,
        string functionName,
        Func<Task<T>> operation)
    {
        var properties = new Dictionary<string, string>
        {
            { "PluginName", pluginName },
            { "FunctionName", functionName }
        };

        using (var operation = _telemetryClient.StartOperation<RequestTelemetry>(
            $"{pluginName}/{functionName}"))
        {
            try
            {
                var result = await operation();
                _telemetryClient.TrackEvent("FunctionInvocationSuccess", properties);
                return result;
            }
            catch (Exception ex)
            {
                _telemetryClient.TrackException(ex, properties);
                throw;
            }
        }
    }
}
```

```python
# Python Logging with Azure Monitor
import logging
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace

# Configure Azure Monitor
configure_azure_monitor(
    connection_string="InstrumentationKey=your-key"
)

# Get tracer
tracer = trace.get_tracer(__name__)

logger = logging.getLogger(__name__)

class SemanticKernelTelemetry:
    def __init__(self):
        self.tracer = tracer
    
    async def track_function_invocation(self, plugin_name, function_name, operation):
        """Track function invocation with span"""
        with self.tracer.start_as_current_span(
            f"{plugin_name}/{function_name}"
        ) as span:
            span.set_attribute("plugin", plugin_name)
            span.set_attribute("function", function_name)
            
            try:
                result = await operation()
                span.set_attribute("status", "success")
                return result
            except Exception as e:
                span.set_attribute("status", "error")
                span.set_attribute("error", str(e))
                logger.error(f"Error in {plugin_name}/{function_name}: {str(e)}")
                raise
```

### 3.2 Performance Metrics

```csharp
// .NET Performance Metrics
public class PerformanceMetrics
{
    private readonly TelemetryClient _telemetryClient;

    public async Task<T> MeasureInvocationAsync<T>(
        string operationName,
        Func<Task<T>> operation)
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        try
        {
            var result = await operation();
            stopwatch.Stop();

            var metrics = new Dictionary<string, double>
            {
                { "DurationMs", stopwatch.ElapsedMilliseconds }
            };

            _telemetryClient.TrackEvent(
                $"{operationName}Completed",
                properties: new Dictionary<string, string>
                {
                    { "Status", "Success" }
                },
                metrics: metrics
            );

            return result;
        }
        catch (Exception ex)
        {
            stopwatch.Stop();

            var metrics = new Dictionary<string, double>
            {
                { "DurationMs", stopwatch.ElapsedMilliseconds }
            };

            _telemetryClient.TrackEvent(
                $"{operationName}Failed",
                properties: new Dictionary<string, string>
                {
                    { "Error", ex.Message }
                },
                metrics: metrics
            );

            throw;
        }
    }
}
```

---

## 4. Security Best Practices

### 4.1 Credential Management

```csharp
// .NET Credential Management with Key Vault
using Azure.Identity;
using Azure.Security.KeyVault.Secrets;
using Microsoft.SemanticKernel;

public class SecureCredentialManager
{
    private readonly SecretClient _secretClient;
    private readonly Dictionary<string, string> _credentialCache;

    public SecureCredentialManager(string keyVaultUri)
    {
        var credential = new DefaultAzureCredential();
        _secretClient = new SecretClient(new Uri(keyVaultUri), credential);
        _credentialCache = new Dictionary<string, string>();
    }

    public async Task<Kernel> CreateSecureKernelAsync()
    {
        var builder = Kernel.CreateBuilder();

        // Get credentials from Key Vault
        var openaiApiKey = await GetSecretAsync("OpenAI-ApiKey");
        var azureEndpoint = await GetSecretAsync("Azure-OpenAI-Endpoint");
        var azureDeployment = await GetSecretAsync("Azure-OpenAI-Deployment");

        // Use DefaultAzureCredential for Azure services
        builder.AddAzureOpenAIChatCompletion(
            deploymentName: azureDeployment,
            endpoint: new Uri(azureEndpoint),
            credential: new DefaultAzureCredential()
        );

        return builder.Build();
    }

    private async Task<string> GetSecretAsync(string secretName)
    {
        if (_credentialCache.TryGetValue(secretName, out var cached))
        {
            return cached;
        }

        var secret = await _secretClient.GetSecretAsync(secretName);
        _credentialCache[secretName] = secret.Value.Value;
        return secret.Value.Value;
    }
}
```

```python
# Python Credential Management with Key Vault
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.azure_open_ai import AzureOpenAIChatCompletion
import os

class SecureCredentialManager:
    def __init__(self, key_vault_uri: str):
        self.credential = DefaultAzureCredential()
        self.secret_client = SecretClient(
            vault_url=key_vault_uri,
            credential=self.credential
        )
        self.credential_cache = {}
    
    async def create_secure_kernel(self) -> Kernel:
        """Create kernel with credentials from Key Vault"""
        kernel = Kernel()
        
        # Get credentials
        azure_endpoint = await self.get_secret_async("Azure-OpenAI-Endpoint")
        azure_deployment = await self.get_secret_async("Azure-OpenAI-Deployment")
        
        # Use DefaultAzureCredential
        kernel.add_service(
            AzureOpenAIChatCompletion(
                deployment_id=azure_deployment,
                endpoint=azure_endpoint,
                credential=self.credential
            )
        )
        
        return kernel
    
    async def get_secret_async(self, secret_name: str) -> str:
        """Get secret from cache or Key Vault"""
        if secret_name in self.credential_cache:
            return self.credential_cache[secret_name]
        
        secret = self.secret_client.get_secret(secret_name)
        self.credential_cache[secret_name] = secret.value
        return secret.value
```

### 4.2 Input Validation and Sanitisation

```csharp
// .NET Input Validation
using System.Text.RegularExpressions;

public class InputValidator
{
    public static void ValidatePromptInput(string input, int maxLength = 5000)
    {
        if (string.IsNullOrWhiteSpace(input))
            throw new ArgumentException("Input cannot be empty");

        if (input.Length > maxLength)
            throw new ArgumentException($"Input exceeds maximum length of {maxLength}");

        // Check for potentially harmful patterns
        var dangerousPatterns = new[]
        {
            @"(?i)(drop|delete|truncate|execute|exec|script|javascript|eval)",
            @"(?i)(<script|javascript:|onerror=|onclick=)"
        };

        foreach (var pattern in dangerousPatterns)
        {
            if (Regex.IsMatch(input, pattern))
                throw new ArgumentException("Input contains potentially harmful content");
        }
    }

    public static string SanitiseInput(string input)
    {
        // Remove leading/trailing whitespace
        input = input.Trim();

        // Remove control characters
        input = Regex.Replace(input, @"[\x00-\x1F\x7F]", "");

        return input;
    }
}
```

---

## 5. Error Handling and Resilience

### 5.1 Retry Policies

```csharp
// .NET Retry with Polly
using Polly;
using Polly.CircuitBreaker;
using Polly.Retry;

public class ResilientKernelService
{
    private readonly Kernel _kernel;
    private readonly IAsyncPolicy<object> _retryPolicy;
    private readonly IAsyncPolicy<object> _circuitBreakerPolicy;
    private readonly IAsyncPolicy<object> _combinedPolicy;

    public ResilientKernelService(Kernel kernel)
    {
        _kernel = kernel;

        // Retry policy: exponential backoff
        _retryPolicy = Policy<object>
            .Handle<HttpRequestException>()
            .Or<OperationCanceledException>()
            .OrResult<object>(r => r == null)
            .WaitAndRetryAsync(
                retryCount: 3,
                sleepDurationProvider: retryAttempt =>
                    TimeSpan.FromSeconds(Math.Pow(2, retryAttempt)),
                onRetry: (outcome, timespan, retryCount, context) =>
                {
                    // Log retry
                    Console.WriteLine($"Retry {retryCount} after {timespan.TotalSeconds}s");
                }
            );

        // Circuit breaker policy
        _circuitBreakerPolicy = Policy<object>
            .Handle<HttpRequestException>()
            .OrResult<object>(r => r == null)
            .CircuitBreakerAsync(
                handledEventsAllowedBeforeBreaking: 3,
                durationOfBreak: TimeSpan.FromSeconds(30),
                onBreak: (outcome, duration) =>
                {
                    Console.WriteLine($"Circuit breaker open for {duration.TotalSeconds}s");
                }
            );

        // Combine policies
        _combinedPolicy = Policy.WrapAsync(_retryPolicy, _circuitBreakerPolicy);
    }

    public async Task<string> InvokeWithResilienceAsync(
        string pluginName,
        string functionName,
        KernelArguments? arguments = null)
    {
        return (await _combinedPolicy.ExecuteAsync(async () =>
        {
            var result = await _kernel.InvokeAsync(
                pluginName,
                functionName,
                arguments ?? new KernelArguments()
            );
            return result;
        })).ToString() ?? "";
    }
}
```

---

## 6. Scaling and Load Management

### 6.1 Horizontal Scaling

```yaml
# Kubernetes Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: semantic-kernel-app
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sk-app
  template:
    metadata:
      labels:
        app: sk-app
    spec:
      containers:
      - name: sk-app
        image: myregistry.azurecr.io/sk-app:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: "250m"
            memory: "512Mi"
          limits:
            cpu: "500m"
            memory: "1Gi"
        env:
        - name: AZURE_OPENAI_ENDPOINT
          valueFrom:
            secretKeyRef:
              name: sk-secrets
              key: azure-endpoint
        - name: AZURE_OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: sk-secrets
              key: azure-api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: semantic-kernel-service
  namespace: production
spec:
  selector:
    app: sk-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sk-app-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: semantic-kernel-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 7. Cost Management

### 7.1 Token Tracking

```csharp
// .NET Token Cost Tracking
public class TokenCostTracker
{
    private readonly string _modelName;
    private readonly decimal _inputCostPer1kTokens;
    private readonly decimal _outputCostPer1kTokens;

    public TokenCostTracker(
        string modelName,
        decimal inputCost,
        decimal outputCost)
    {
        _modelName = modelName;
        _inputCostPer1kTokens = inputCost;
        _outputCostPer1kTokens = outputCost;
    }

    public decimal CalculateCost(
        int inputTokens,
        int outputTokens)
    {
        var inputCost = (inputTokens / 1000m) * _inputCostPer1kTokens;
        var outputCost = (outputTokens / 1000m) * _outputCostPer1kTokens;
        return inputCost + outputCost;
    }

    public void LogCostMetrics(
        string operationName,
        int inputTokens,
        int outputTokens,
        TelemetryClient telemetryClient)
    {
        var totalCost = CalculateCost(inputTokens, outputTokens);

        var properties = new Dictionary<string, string>
        {
            { "Operation", operationName },
            { "Model", _modelName }
        };

        var metrics = new Dictionary<string, double>
        {
            { "InputTokens", inputTokens },
            { "OutputTokens", outputTokens },
            { "TotalCostUSD", (double)totalCost }
        };

        telemetryClient.TrackEvent("TokenUsage", properties, metrics);
    }
}
```

---

## 8. Testing Strategies

### 8.1 Unit Testing

```csharp
// .NET Unit Tests with xUnit
using Xunit;
using Moq;
using Microsoft.SemanticKernel;

public class SemanticKernelTests
{
    [Fact]
    public async Task InvokeAsync_WithValidInput_ReturnsResult()
    {
        // Arrange
        var mockKernel = new Mock<Kernel>();
        mockKernel.Setup(k => k.InvokeAsync(
            It.IsAny<string>(),
            It.IsAny<string>(),
            It.IsAny<KernelArguments>()))
            .ReturnsAsync("Expected Result");

        // Act
        var result = await mockKernel.Object.InvokeAsync(
            "Math",
            "Add",
            new KernelArguments { ["a"] = 5, ["b"] = 3 }
        );

        // Assert
        Assert.Equal("Expected Result", result.ToString());
    }

    [Theory]
    [InlineData(0, 0, 0)]
    [InlineData(5, 3, 8)]
    [InlineData(-5, 5, 0)]
    public async Task Add_WithVariousInputs_ReturnsCorrectSum(
        int a,
        int b,
        int expected)
    {
        // Test implementation
        var mathPlugin = new MathPlugin();
        var result = mathPlugin.Add(a, b);

        Assert.Equal(expected, result);
    }
}
```

```python
# Python Unit Tests with pytest
import pytest
from unittest.mock import AsyncMock, patch
from semantic_kernel import Kernel

class TestSemanticKernel:
    @pytest.mark.asyncio
    async def test_invoke_with_valid_input(self):
        """Test kernel invocation with valid input"""
        # Arrange
        kernel = Kernel()
        
        # Mock the invocation
        with patch.object(kernel, 'invoke_async', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = "Expected Result"
            
            # Act
            result = await kernel.invoke_async("Math", "add", a=5, b=3)
            
            # Assert
            assert result == "Expected Result"
            mock_invoke.assert_called_once()
    
    @pytest.mark.parametrize("a,b,expected", [
        (0, 0, 0),
        (5, 3, 8),
        (-5, 5, 0)
    ])
    def test_add_with_various_inputs(self, a, b, expected):
        """Test addition with various inputs"""
        from my_plugins import MathPlugin
        
        plugin = MathPlugin()
        result = plugin.add(a, b)
        
        assert result == expected
```

### 8.2 Integration Testing

```csharp

// .NET Integration Tests
[Collection("Integration")]
public class IntegrationTests : IAsyncLifetime
{
    private Kernel _kernel;

    public async Task InitializeAsync()
    {
        _kernel = Kernel.CreateBuilder()
            .AddAzureOpenAIChatCompletion(
                deploymentName: "test-deployment",
                endpoint: new Uri("https://test.openai.azure.com/"),
                apiKey: "test-key"
            )
            .Build();
    }

    public Task DisposeAsync()
    {
        return Task.CompletedTask;
    }

    [Fact]
    public async Task EndToEnd_ProcessingWorkflow_Succeeds()
    {
        // Arrange
        var inputText = "Test input for semantic kernel";

        // Act
        var function = _kernel.CreateFunctionFromPrompt(
            prompt: "Process: {{$input}}"
        );
        var result = await _kernel.InvokeAsync(function, new() { ["input"] = inputText });

        // Assert
        Assert.NotNull(result);
        Assert.NotEmpty(result.ToString());
    }
}

```

---

## 9. Production Patterns

### 9.1 API Gateway Pattern

```csharp
// ASP.NET Core API with Rate Limiting
using AspNetCoreRateLimit;
using Microsoft.SemanticKernel;

var builder = WebApplication.CreateBuilder(args);

// Add rate limiting
builder.Services.AddMemoryCache();
builder.Services.Configure<IpRateLimitOptions>(options =>
{
    options.GeneralRules = new List<RateLimitRule>
    {
        new RateLimitRule
        {
            Endpoint = "*",
            Limit = 100,
            Period = "1m"
        }
    };
});
builder.Services.AddSingleton<IRateLimitCounterStore, MemoryCacheRateLimitCounterStore>();
builder.Services.AddSingleton<IIpPolicyStore, MemoryCacheIpPolicyStore>();
builder.Services.AddInMemoryRateLimiting();
builder.Services.AddHttpContextAccessor();

var app = builder.Build();

app.UseIpRateLimiting();

app.MapPost("/api/process", async (
    ProcessRequest request,
    ISemanticKernelService service) =>
{
    try
    {
        var result = await service.ProcessAsync(request.Text);
        return Results.Ok(new { result });
    }
    catch (Exception ex)
    {
        return Results.BadRequest(new { error = ex.Message });
    }
});

app.Run();

public record ProcessRequest(string Text);
```

### 9.2 Asynchronous Processing Pattern

```csharp

// .NET Background Job Processing
using Hangfire;

public class BackgroundJobService
{
    private readonly Kernel _kernel;
    private readonly IBackgroundJobClient _backgroundJobs;

    public BackgroundJobService(Kernel kernel, IBackgroundJobClient backgroundJobs)
    {
        _kernel = kernel;
        _backgroundJobs = backgroundJobs;
    }

    public string ScheduleLargeProcessingJob(List<string> items)
    {
        // Schedule background job
        var jobId = _backgroundJobs.Enqueue(() => ProcessItemsAsync(items));
        return jobId;
    }

    [AutomaticRetry(Attempts = 3)]
    public async Task ProcessItemsAsync(List<string> items)
    {
        foreach (var item in items)
        {
            try
            {
                var function = _kernel.CreateFunctionFromPrompt(
                    prompt: "Process: {{$input}}"
                );
                await _kernel.InvokeAsync(function, new() { ["input"] = item });
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error processing item: {ex.Message}");
            }
        }
    }
}

```

This production guide covers essential operations patterns for deploying Semantic Kernel applications in enterprise environments with emphasis on reliability, scalability, security, and cost management.





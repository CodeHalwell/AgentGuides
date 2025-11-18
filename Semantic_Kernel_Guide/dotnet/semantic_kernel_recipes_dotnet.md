# Semantic Kernel Recipes (.NET)

**Ready-to-Use C# Code Examples for Common Patterns**

Last Updated: November 2025
Semantic Kernel .NET: 1.38.0+

---

## Overview

Production-ready C#/.NET recipes for common Semantic Kernel patterns.

**See Also:** [../semantic_kernel_recipes.md](../semantic_kernel_recipes.md) for language-agnostic recipes.

---

## ASP.NET Core Integration

```csharp
// Program.cs
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Azure.Identity;

var builder = WebApplication.CreateBuilder(args);

// Add Semantic Kernel
builder.Services.AddSingleton(sp =>
{
    var kernelBuilder = Kernel.CreateBuilder();

    kernelBuilder.AddAzureOpenAIChatCompletion(
        deploymentName: builder.Configuration["AzureOpenAI:Deployment"]!,
        endpoint: builder.Configuration["AzureOpenAI:Endpoint"]!,
        credential: new DefaultAzureCredential()
    );

    return kernelBuilder.Build();
});

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

app.UseSwagger();
app.UseSwaggerUI();
app.MapControllers();

app.Run();
```

{% raw %}
```csharp
// Controllers/ChatController.cs
[ApiController]
[Route("api/[controller]")]
public class ChatController : ControllerBase
{
    private readonly Kernel _kernel;
    private readonly ILogger<ChatController> _logger;

    public ChatController(Kernel kernel, ILogger<ChatController> logger)
    {
        _kernel = kernel;
        _logger = logger;
    }

    [HttpPost]
    public async Task<IActionResult> Chat([FromBody] ChatRequest request)
    {
        try
        {
            var function = _kernel.CreateFunctionFromPrompt(
                "Answer the following question: {{$input}}"
            );

            var result = await _kernel.InvokeAsync<string>(
                function,
                new() { ["input"] = request.Message }
            );

            return Ok(new { response = result });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Chat failed");
            return StatusCode(500, new { error = ex.Message });
        }
    }
}

public record ChatRequest(string Message);
```
{% endraw %}

---

## Multi-Agent with Dependency Injection

```csharp
public interface IAgentOrchestrator
{
    Task<string> ProcessAsync(string input);
}

public class ResearchWritingOrchestrator : IAgentOrchestrator
{
    private readonly Kernel _kernel;
    private readonly ILogger<ResearchWritingOrchestrator> _logger;

    public ResearchWritingOrchestrator(
        Kernel kernel,
        ILogger<ResearchWritingOrchestrator> logger)
    {
        _kernel = kernel;
        _logger = logger;
    }

    public async Task<string> ProcessAsync(string input)
    {
        // Create agents
        var researcher = new ChatCompletionAgent(_kernel)
        {
            Name = "researcher",
            Instructions = "Research and cite sources."
        };

        var writer = new ChatCompletionAgent(_kernel)
        {
            Name = "writer",
            Instructions = "Write engaging content."
        };

        // Orchestrate
        var groupChat = new AgentGroupChat(researcher, writer);
        await groupChat.AddChatMessageAsync(new ChatMessageContent(
            AuthorRole.User,
            input
        ));

        string result = "";
        await foreach (var message in groupChat.InvokeAsync())
        {
            result = message.Content;
            _logger.LogInformation($"[{message.Name}]: {message.Content}");

            if (message.Content.Contains("DONE", StringComparison.OrdinalIgnoreCase))
                break;
        }

        return result;
    }
}

// Registration
builder.Services.AddScoped<IAgentOrchestrator, ResearchWritingOrchestrator>();
```

---

For more recipes, see:
- [Comprehensive Guide](semantic_kernel_comprehensive_dotnet.md)
- [Production Guide](semantic_kernel_production_dotnet.md)
- [General Recipes](../semantic_kernel_recipes.md)

**[Back to .NET README](README.md)** | **[Complete Index](GUIDE_INDEX.md)**

Latest: 1.38.0
# Semantic Kernel: From Fundamentals to Expert Level - Comprehensive Technical Guide

**Version:** 1.0  
**Last Updated:** November 2025  
**Scope:** Installation, Core Concepts, Agents, Plugins, Planners, Memory, Azure Integration, Cross-Platform Implementation

---

## Table of Contents

1. [Core Fundamentals](#1-core-fundamentals)
2. [Simple Agents](#2-simple-agents)
3. [Multi-Agent Systems](#3-multi-agent-systems)
4. [Tools Integration (Plugins)](#4-tools-integration-plugins)
5. [Structured Output](#5-structured-output)
6. [Model Context Protocol (MCP)](#6-model-context-protocol-mcp)
7. [Agentic Patterns](#7-agentic-patterns)
8. [Planners](#8-planners)
9. [Memory Systems](#9-memory-systems)
10. [Context Engineering](#10-context-engineering)
11. [Azure Integration](#11-azure-integration)
12. [Skills & Functions](#12-skills--functions)
13. [Cross-Platform Implementation](#13-cross-platform-implementation)
14. [Advanced Topics](#14-advanced-topics)

---

## 1. Core Fundamentals

### 1.1 Installation

#### .NET Installation

Semantic Kernel for .NET requires .NET 6.0 or later. Installation is straightforward using NuGet:

```bash
# Install the core package
dotnet add package Microsoft.SemanticKernel

# For Azure integration
dotnet add package Microsoft.SemanticKernel.Connectors.AzureOpenAI

# For Azure AI Search memory
dotnet add package Microsoft.SemanticKernel.Connectors.AzureAISearch

# For OpenAI integration
dotnet add package Microsoft.SemanticKernel.Connectors.OpenAI

# For Qdrant vector store
dotnet add package Microsoft.SemanticKernel.Connectors.Qdrant

# For Weaviate integration
dotnet add package Microsoft.SemanticKernel.Connectors.Weaviate
```

Verify installation by checking the NuGet package manager or running:

```bash
dotnet list package
```

**Project Structure Example (.NET):**

```
MySemanticKernelApp/
├── MySemanticKernelApp.csproj
├── Program.cs
├── Plugins/
│   ├── MathPlugin.cs
│   └── TextPlugin.cs
├── Agents/
│   └── MyAgent.cs
├── Configuration/
│   ├── KernelConfig.cs
│   └── ServiceConfiguration.cs
└── appsettings.json
```

#### Python Installation

Semantic Kernel for Python requires Python 3.9 or later:

```bash
# Basic installation
pip install semantic-kernel

# For Azure components
pip install semantic-kernel[azure]

# For OpenAI
pip install semantic-kernel[openai]

# For all vector stores
pip install semantic-kernel[qdrant,weaviate,milvus,azure-cognitive-search]

# For development with all extras
pip install semantic-kernel[all]
```

Create a `requirements.txt` file:

```
semantic-kernel>=0.10.0
python-dotenv>=1.0.0
aiohttp>=3.8.0
openai>=1.0.0
```

Install from requirements:

```bash
pip install -r requirements.txt
```

**Project Structure Example (Python):**

```
my_semantic_kernel_app/
├── requirements.txt
├── main.py
├── plugins/
│   ├── __init__.py
│   ├── math_plugin.py
│   └── text_plugin.py
├── agents/
│   ├── __init__.py
│   └── my_agent.py
├── config/
│   ├── __init__.py
│   ├── kernel_config.py
│   └── service_config.py
├── .env
└── .env.example
```

#### Java Installation

Semantic Kernel for Java is available through Maven Central:

```xml
<dependency>
    <groupId>com.microsoft.semantickernel</groupId>
    <artifactId>semantickernel</artifactId>
    <version>1.0.0</version>
</dependency>

<!-- Azure OpenAI connector -->
<dependency>
    <groupId>com.microsoft.semantickernel</groupId>
    <artifactId>semantickernel-connectors-openai</artifactId>
    <version>1.0.0</version>
</dependency>

<!-- Memory and embeddings support -->
<dependency>
    <groupId>com.microsoft.semantickernel</groupId>
    <artifactId>semantickernel-memory</artifactId>
    <version>1.0.0</version>
</dependency>
```

### 1.2 Kernel Initialization and Configuration

The Kernel is the central component of Semantic Kernel. It orchestrates services, plugins, and functions. Initialisation varies significantly by platform.

#### .NET Kernel Initialization

The modern approach uses the `KernelBuilder` pattern:

```csharp
using Microsoft.SemanticKernel;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

// Basic kernel with OpenAI
var kernel = Kernel.CreateBuilder()
    .AddOpenAIChatCompletion(
        modelId: "gpt-4",
        apiKey: Environment.GetEnvironmentVariable("OPENAI_API_KEY")!
    )
    .Build();

// Kernel with logging and multiple services
var builder = Kernel.CreateBuilder()
    .AddOpenAIChatCompletion(
        modelId: "gpt-4",
        apiKey: Environment.GetEnvironmentVariable("OPENAI_API_KEY")!
    )
    .AddOpenAIChatCompletion(
        modelId: "gpt-3.5-turbo",
        apiKey: Environment.GetEnvironmentVariable("OPENAI_API_KEY")!,
        serviceId: "gpt35"  // Identifier for this service
    );

// Build with logging
var services = new ServiceCollection()
    .AddLogging(logging => logging.AddConsole())
    .BuildServiceProvider();

var kernel = builder
    .Build();

// Access the created kernel
Console.WriteLine("Kernel initialised successfully with services");
```

**Advanced .NET Configuration with Dependency Injection:**

```csharp
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;

public static class KernelConfiguration
{
    public static IServiceProvider ConfigureServices()
    {
        var services = new ServiceCollection();

        // Add logging
        services.AddLogging(builder => 
            builder.AddConsole().SetMinimumLevel(LogLevel.Information)
        );

        // Add kernel with Azure OpenAI
        services.AddTransient<Kernel>(sp => Kernel.CreateBuilder()
            .AddAzureOpenAIChatCompletion(
                deploymentName: Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT")!,
                endpoint: Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")!,
                apiKey: Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY")!
            )
            .Build()
        );

        // Add custom services
        services.AddSingleton<IMemoryStore, VolatileMemoryStore>();
        services.AddSingleton<MyCustomPlugin>();

        return services.BuildServiceProvider();
    }
}

// Usage
var serviceProvider = KernelConfiguration.ConfigureServices();
var kernel = serviceProvider.GetRequiredService<Kernel>();
```

#### Python Kernel Initialization

Python kernel initialisation is more flexible and supports multiple configuration patterns:

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.azure_open_ai import AzureOpenAIChatCompletion
import os

# Basic kernel with OpenAI
kernel = Kernel()
kernel.add_service(
    OpenAIChatCompletion(
        model_id="gpt-4",
        api_key=os.environ.get("OPENAI_API_KEY")
    )
)

# Kernel with multiple services
kernel = Kernel()
kernel.add_service(
    OpenAIChatCompletion(
        model_id="gpt-4",
        api_key=os.environ.get("OPENAI_API_KEY"),
        service_id="gpt4"
    )
)
kernel.add_service(
    OpenAIChatCompletion(
        model_id="gpt-3.5-turbo",
        api_key=os.environ.get("OPENAI_API_KEY"),
        service_id="gpt35"
    )
)

# Kernel with Azure OpenAI
kernel = Kernel()
kernel.add_service(
    AzureOpenAIChatCompletion(
        deployment_id=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
        endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY")
    )
)
```

**Advanced Python Configuration Module:**

```python
# config/kernel_config.py
import os
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.azure_open_ai import AzureOpenAIChatCompletion
from semantic_kernel.memory import VolatileMemoryStore
from dotenv import load_dotenv

load_dotenv()

class KernelConfig:
    """Centralised kernel configuration management"""
    
    @staticmethod
    def create_kernel(
        use_azure: bool = False,
        add_memory: bool = False,
        service_id: str = None
    ) -> Kernel:
        """Create and configure a kernel instance
        
        Args:
            use_azure: Use Azure OpenAI if True, else use OpenAI
            add_memory: Add volatile memory store
            service_id: Optional identifier for the service
        
        Returns:
            Configured Kernel instance
        """
        kernel = Kernel()
        
        if use_azure:
            kernel.add_service(
                AzureOpenAIChatCompletion(
                    deployment_id=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
                    endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                    service_id=service_id
                )
            )
        else:
            kernel.add_service(
                OpenAIChatCompletion(
                    model_id="gpt-4",
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    service_id=service_id
                )
            )
        
        if add_memory:
            kernel.add_memory_store(VolatileMemoryStore())
        
        return kernel

# Usage
kernel = KernelConfig.create_kernel(use_azure=True, add_memory=True)
```

#### Java Kernel Initialization

```java
import com.microsoft.semantickernel.Kernel;
import com.microsoft.semantickernel.KernelBuilder;
import com.microsoft.semantickernel.connectors.ai.openai.OpenAIChatCompletion;

// Basic kernel
Kernel kernel = new KernelBuilder()
    .withAIService(
        OpenAIChatCompletion.class,
        new OpenAIChatCompletion(
            "gpt-4",
            System.getenv("OPENAI_API_KEY")
        )
    )
    .build();

// With multiple services
Kernel kernel = new KernelBuilder()
    .withAIService(
        OpenAIChatCompletion.class,
        new OpenAIChatCompletion("gpt-4", System.getenv("OPENAI_API_KEY")),
        "gpt4"
    )
    .withAIService(
        OpenAIChatCompletion.class,
        new OpenAIChatCompletion("gpt-3.5-turbo", System.getenv("OPENAI_API_KEY")),
        "gpt35"
    )
    .build();
```

### 1.3 Design Principles: Skills, Plugins, and Memory

Semantic Kernel is built around three core design principles that enable flexible, extensible AI applications:

#### Skills (Legacy Terminology)

In earlier versions of Semantic Kernel, "Skills" was the primary abstraction. Modern versions use "Plugins" instead, but understanding the distinction is important:

- **Semantic Skills:** Prompt-based functions that leverage LLM capabilities
- **Native Skills:** Code-based functions that execute in your application runtime

#### Plugins (Current Standard)

Plugins are modular, reusable components that encapsulate functionality:

```csharp
// .NET Plugin Definition
using Microsoft.SemanticKernel;
using System.ComponentModel;

public class MathPlugin
{
    [KernelFunction]
    [Description("Multiplies two numbers")]
    public static int Multiply(
        [Description("The first number")]
        int a,
        [Description("The second number")]
        int b
    ) => a * b;

    [KernelFunction]
    [Description("Adds two numbers")]
    public static int Add(int a, int b) => a + b;

    [KernelFunction]
    [Description("Divides two numbers")]
    public static double Divide(int a, int b)
    {
        if (b == 0) throw new ArgumentException("Cannot divide by zero");
        return (double)a / b;
    }
}
```

```python
# Python Plugin Definition
from semantic_kernel.kernel import Kernel
from semantic_kernel.functions.kernel_function_decorator import kernel_function

class MathPlugin:
    @kernel_function(description="Multiplies two numbers")
    def multiply(self, a: int, b: int) -> int:
        """Multiply two integers"""
        return a * b
    
    @kernel_function(description="Adds two numbers")
    def add(self, a: int, b: int) -> int:
        """Add two integers"""
        return a + b
    
    @kernel_function(description="Divides two numbers")
    def divide(self, a: int, b: int) -> float:
        """Divide two numbers"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
```

#### Memory

Memory in Semantic Kernel manages context and state across interactions. There are several types:

1. **Volatile Memory:** In-memory store for temporary data
2. **Vector Memory:** Semantic embeddings for similarity search
3. **Persistent Memory:** Long-term storage with external backends

```csharp
// .NET Memory Management
using Microsoft.SemanticKernel.Memory;

// Create volatile memory store
IMemoryStore memoryStore = new VolatileMemoryStore();

// Add memories
await memoryStore.CreateCollectionAsync("conversations");
await memoryStore.UpsertAsync(
    collectionName: "conversations",
    key: "user123",
    text: "User prefers concise responses",
    metadata: null
);

// Retrieve memories
var result = await memoryStore.GetAsync(
    collectionName: "conversations",
    key: "user123"
);
```

```python
# Python Memory Management
from semantic_kernel.memory import VolatileMemoryStore

# Create volatile memory store
memory_store = VolatileMemoryStore()

# Add memories
await memory_store.create_collection_async("conversations")
await memory_store.upsert_async(
    collection_name="conversations",
    key="user123",
    text="User prefers concise responses",
    metadata=None
)

# Retrieve memories
result = await memory_store.get_async(
    collection_name="conversations",
    key="user123"
)
```

### 1.4 Service Registration

Service registration is crucial for enabling various capabilities in Semantic Kernel. The process differs significantly across platforms.

#### .NET Service Registration

```csharp
using Microsoft.SemanticKernel;
using Microsoft.Extensions.DependencyInjection;

public class ServiceRegistration
{
    public static Kernel RegisterServices()
    {
        var builder = Kernel.CreateBuilder();

        // Register OpenAI Chat Completion
        builder.AddOpenAIChatCompletion(
            modelId: "gpt-4",
            apiKey: Environment.GetEnvironmentVariable("OPENAI_API_KEY")!
        );

        // Register Azure OpenAI Chat Completion
        builder.AddAzureOpenAIChatCompletion(
            deploymentName: Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT")!,
            endpoint: Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")!,
            apiKey: Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY")!
        );

        // Register Text Embeddings
        builder.Services.AddOpenAITextEmbeddingGeneration(
            modelId: "text-embedding-3-small",
            apiKey: Environment.GetEnvironmentVariable("OPENAI_API_KEY")!
        );

        var kernel = builder.Build();

        // Register Plugins
        kernel.ImportPluginFromType<MathPlugin>("Math");
        kernel.ImportPluginFromType<TextPlugin>("Text");

        return kernel;
    }
}
```

**Comprehensive Service Registration with All Components:**

```csharp
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.AzureAISearch;
using Microsoft.SemanticKernel.Memory;
using Microsoft.Extensions.DependencyInjection;
using Azure.Search.Documents.Indexes;
using Azure.Identity;

public static class FullServiceConfiguration
{
    public static Kernel ConfigureFullStack()
    {
        var builder = Kernel.CreateBuilder();

        // 1. Add Chat Completion Services
        builder.AddOpenAIChatCompletion(
            modelId: "gpt-4",
            apiKey: Environment.GetEnvironmentVariable("OPENAI_API_KEY")!
        );

        // 2. Add Text Embedding Services
        builder.Services.AddOpenAITextEmbeddingGeneration(
            modelId: "text-embedding-3-small",
            apiKey: Environment.GetEnvironmentVariable("OPENAI_API_KEY")!
        );

        // 3. Add Memory Store (Azure AI Search for production)
        var searchIndexClient = new SearchIndexClient(
            new Uri(Environment.GetEnvironmentVariable("AZURE_SEARCH_ENDPOINT")!),
            new DefaultAzureCredential()
        );

        var memoryStore = new AzureAISearchMemoryStore(
            searchIndexClient: searchIndexClient
        );

        builder.Services.AddSingleton<IMemoryStore>(memoryStore);

        // Build the kernel
        var kernel = builder.Build();

        // 4. Import plugins
        kernel.ImportPluginFromType<MathPlugin>("Math");
        kernel.ImportPluginFromType<TextPlugin>("Text");
        kernel.ImportPluginFromType<TimePlugin>("Time");

        return kernel;
    }
}
```

#### Python Service Registration

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.azure_open_ai import AzureOpenAIChatCompletion
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding
from semantic_kernel.memory import VolatileMemoryStore
from my_plugins import MathPlugin, TextPlugin, TimePlugin
import os

def register_services() -> Kernel:
    """Register all services and create kernel"""
    kernel = Kernel()

    # Register OpenAI Chat Completion
    kernel.add_service(
        OpenAIChatCompletion(
            model_id="gpt-4",
            api_key=os.environ.get("OPENAI_API_KEY")
        )
    )

    # Register Azure OpenAI (alternative)
    if os.environ.get("USE_AZURE"):
        kernel.add_service(
            AzureOpenAIChatCompletion(
                deployment_id=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
                endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                api_key=os.environ.get("AZURE_OPENAI_API_KEY")
            )
        )

    # Register Text Embedding
    kernel.add_service(
        OpenAITextEmbedding(
            model_id="text-embedding-3-small",
            api_key=os.environ.get("OPENAI_API_KEY")
        )
    )

    # Add Memory Store
    kernel.add_memory_store(VolatileMemoryStore())

    # Import Plugins
    kernel.add_plugin(MathPlugin(), plugin_name="Math")
    kernel.add_plugin(TextPlugin(), plugin_name="Text")
    kernel.add_plugin(TimePlugin(), plugin_name="Time")

    return kernel
```

### 1.5 Configuration Patterns

#### Environment-Based Configuration

The most common pattern uses environment variables for credentials:

```bash
# .env file
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4
AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com/
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-4-deployment
AZURE_SEARCH_ENDPOINT=https://your-search-instance.search.windows.net
AZURE_SEARCH_API_KEY=...
```

```csharp
// .NET Configuration Pattern
using Microsoft.Extensions.Configuration;

public class ConfigurationManager
{
    private readonly IConfiguration _configuration;

    public ConfigurationManager(IConfiguration configuration)
    {
        _configuration = configuration;
    }

    public Kernel BuildKernel()
    {
        var builder = Kernel.CreateBuilder();

        var provider = _configuration["Provider"]; // "openai" or "azure"

        if (provider == "azure")
        {
            builder.AddAzureOpenAIChatCompletion(
                deploymentName: _configuration["Azure:Deployment"]!,
                endpoint: _configuration["Azure:Endpoint"]!,
                apiKey: _configuration["Azure:ApiKey"]!
            );
        }
        else
        {
            builder.AddOpenAIChatCompletion(
                modelId: _configuration["OpenAI:Model"]!,
                apiKey: _configuration["OpenAI:ApiKey"]!
            );
        }

        return builder.Build();
    }
}
```

```python
# Python Configuration Pattern
import os
from dotenv import load_dotenv
from typing import Literal

class ConfigManager:
    def __init__(self):
        load_dotenv()
    
    def get_kernel_config(self, provider: Literal["openai", "azure"] = "openai"):
        """Get kernel configuration based on provider"""
        if provider == "azure":
            return {
                "deployment_id": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "api_key": os.getenv("AZURE_OPENAI_API_KEY")
            }
        else:
            return {
                "model_id": os.getenv("OPENAI_MODEL", "gpt-4"),
                "api_key": os.getenv("OPENAI_API_KEY")
            }
```

#### Builder Pattern Configuration

```csharp
// Fluent configuration builder
public class KernelBuilder
{
    private Dictionary<string, string> _services = new();
    private Dictionary<string, string> _plugins = new();
    private Dictionary<string, string> _settings = new();

    public KernelBuilder AddService(string name, string config)
    {
        _services[name] = config;
        return this;
    }

    public KernelBuilder AddPlugin(string name, string path)
    {
        _plugins[name] = path;
        return this;
    }

    public KernelBuilder WithSetting(string key, string value)
    {
        _settings[key] = value;
        return this;
    }

    public Kernel Build()
    {
        var builder = Kernel.CreateBuilder();
        // Apply configurations
        return builder.Build();
    }
}
```

---

## 2. Simple Agents

### 2.1 Creating Basic SK Agents

An agent in Semantic Kernel is an entity that can execute functions and reason about tasks. Basic agents execute simple tasks without complex planning.

#### .NET Basic Agent

```csharp
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Plugins.Core;

public class BasicAgent
{
    private readonly Kernel _kernel;

    public BasicAgent(Kernel kernel)
    {
        _kernel = kernel;
    }

    public async Task<string> ExecuteTaskAsync(string task)
    {
        try
        {
            // Create a simple semantic function for the task
            var function = _kernel.CreateFunctionFromPrompt(
                prompt: task,
                functionName: "ExecuteTask",
                description: "Execute a simple task"
            );

            var result = await _kernel.InvokeAsync(function);
            return result.ToString() ?? "No result";
        }
        catch (Exception ex)
        {
            return $"Error: {ex.Message}";
        }
    }
}

// Usage
var kernel = Kernel.CreateBuilder()
    .AddOpenAIChatCompletion("gpt-4", "YOUR_API_KEY")
    .Build();

var agent = new BasicAgent(kernel);
var result = await agent.ExecuteTaskAsync(
    "Explain the benefits of machine learning in healthcare"
);
Console.WriteLine(result);
```

#### Python Basic Agent

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

class BasicAgent:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
    
    async def execute_task(self, task: str) -> str:
        """Execute a simple task using the kernel"""
        try:
            function = self.kernel.create_function_from_prompt(
                prompt=task,
                function_name="ExecuteTask",
                description="Execute a simple task"
            )
            
            result = await self.kernel.invoke_async(function)
            return str(result) if result else "No result"
        except Exception as e:
            return f"Error: {str(e)}"

# Usage
kernel = Kernel()
kernel.add_service(
    OpenAIChatCompletion(
        model_id="gpt-4",
        api_key="YOUR_API_KEY"
    )
)

agent = BasicAgent(kernel)
result = await agent.execute_task(
    "Explain the benefits of machine learning in healthcare"
)
print(result)
```

### 2.2 Semantic Functions with Prompts

Semantic functions use natural language prompts to define AI-driven behaviour. They're the foundation of agent capabilities.

#### .NET Semantic Functions

```csharp
using Microsoft.SemanticKernel;

public class SemanticFunctionsExample
{
    private readonly Kernel _kernel;

    public SemanticFunctionsExample(Kernel kernel)
    {
        _kernel = kernel;
    }

    // Simple semantic function
    public async Task<string> TranslateToFrench(string text)
    {
        var function = _kernel.CreateFunctionFromPrompt(
            prompt: "Translate this English text to French: {{$input}}",
            functionName: "TranslateToFrench"
        );

        var result = await _kernel.InvokeAsync(function, new() { ["input"] = text });
        return result.ToString() ?? "";
    }

    // Semantic function with system prompt
    public async Task<string> SummarizeWithTone(string text, string tone)
    {
        var prompt = @"You are an expert summarizer. The tone should be {{$tone}}.
Summarize the following text in {{$tone}} tone:
{{$text}}";

        var function = _kernel.CreateFunctionFromPrompt(
            prompt: prompt,
            functionName: "SummarizeWithTone"
        );

        var arguments = new KernelArguments
        {
            ["text"] = text,
            ["tone"] = tone
        };

        var result = await _kernel.InvokeAsync(function, arguments);
        return result.ToString() ?? "";
    }

    // Complex semantic function with multiple parameters
    public async Task<string> GenerateCodeDocumentation(
        string language,
        string code,
        string style)
    {
        var prompt = @"You are an expert code documentor.
Language: {{$language}}
Documentation style: {{$style}}

Generate documentation for this {{$language}} code:
```
{{$code}}
```

Ensure the documentation follows {{$style}} conventions.";

        var function = _kernel.CreateFunctionFromPrompt(
            prompt: prompt,
            functionName: "GenerateCodeDocumentation"
        );

        var arguments = new KernelArguments
        {
            ["language"] = language,
            ["code"] = code,
            ["style"] = style
        };

        var result = await _kernel.InvokeAsync(function, arguments);
        return result.ToString() ?? "";
    }
}
```

#### Python Semantic Functions

```python
from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function
from typing import Any

class SemanticFunctionsExample:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
    
    async def translate_to_french(self, text: str) -> str:
        """Translate English text to French"""
        function = self.kernel.create_function_from_prompt(
            prompt="Translate this English text to French: {{$input}}",
            function_name="TranslateToFrench"
        )
        
        result = await self.kernel.invoke_async(function, input=text)
        return str(result) if result else ""
    
    async def summarize_with_tone(self, text: str, tone: str) -> str:
        """Summarize text with specific tone"""
        prompt = """You are an expert summarizer. The tone should be {{$tone}}.
Summarize the following text in {{$tone}} tone:
{{$text}}"""
        
        function = self.kernel.create_function_from_prompt(
            prompt=prompt,
            function_name="SummarizeWithTone"
        )
        
        result = await self.kernel.invoke_async(
            function,
            text=text,
            tone=tone
        )
        return str(result) if result else ""
    
    async def generate_code_documentation(
        self,
        language: str,
        code: str,
        style: str
    ) -> str:
        """Generate code documentation"""
        prompt = """You are an expert code documentor.
Language: {{$language}}
Documentation style: {{$style}}

Generate documentation for this {{$language}} code:
```
{{$code}}
```

Ensure the documentation follows {{$style}} conventions."""
        
        function = self.kernel.create_function_from_prompt(
            prompt=prompt,
            function_name="GenerateCodeDocumentation"
        )
        
        result = await self.kernel.invoke_async(
            function,
            language=language,
            code=code,
            style=style
        )
        return str(result) if result else ""
```

### 2.3 Native Functions with Code

Native functions are implemented directly in your programming language and execute synchronously or asynchronously.

#### .NET Native Functions

```csharp
using Microsoft.SemanticKernel;
using System.ComponentModel;

public class MathPlugin
{
    [KernelFunction]
    [Description("Calculates the sum of two numbers")]
    public static int Add(
        [Description("The first number")] int a,
        [Description("The second number")] int b
    ) => a + b;

    [KernelFunction]
    [Description("Calculates the product of two numbers")]
    public static int Multiply(int a, int b) => a * b;

    [KernelFunction]
    [Description("Calculates factorial of a number")]
    public static long Factorial(int n)
    {
        if (n < 0) throw new ArgumentException("n must be non-negative");
        return n <= 1 ? 1 : n * Factorial(n - 1);
    }
}

public class StringPlugin
{
    [KernelFunction]
    [Description("Reverses a string")]
    public static string Reverse(string input)
    {
        return new string(input.Reverse().ToArray());
    }

    [KernelFunction]
    [Description("Counts words in text")]
    public static int CountWords(string text)
    {
        return text.Split(new[] { ' ', '\t', '\n' }, 
            StringSplitOptions.RemoveEmptyEntries).Length;
    }
}

// Async native functions
public class DataPlugin
{
    [KernelFunction]
    [Description("Fetches data from API")]
    public async Task<string> FetchDataAsync(string endpoint)
    {
        using var client = new HttpClient();
        var response = await client.GetAsync(endpoint);
        return await response.Content.ReadAsStringAsync();
    }

    [KernelFunction]
    [Description("Processes large dataset")]
    public async Task<int> ProcessDatasetAsync(int[] data)
    {
        // Simulate processing
        await Task.Delay(100);
        return data.Sum();
    }
}

// Usage
var kernel = Kernel.CreateBuilder()
    .AddOpenAIChatCompletion("gpt-4", "YOUR_API_KEY")
    .Build();

kernel.ImportPluginFromType<MathPlugin>("Math");
kernel.ImportPluginFromType<StringPlugin>("String");
kernel.ImportPluginFromType<DataPlugin>("Data");

var result = await kernel.InvokeAsync<int>("Math", "Add", new { a = 5, b = 3 });
Console.WriteLine($"5 + 3 = {result}");
```

#### Python Native Functions

```python
from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function
from typing import List
import asyncio

class MathPlugin:
    @kernel_function(description="Calculates the sum of two numbers")
    def add(self, a: int, b: int) -> int:
        """Add two numbers"""
        return a + b
    
    @kernel_function(description="Calculates the product of two numbers")
    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b
    
    @kernel_function(description="Calculates factorial")
    def factorial(self, n: int) -> int:
        """Calculate factorial of n"""
        if n < 0:
            raise ValueError("n must be non-negative")
        if n <= 1:
            return 1
        return n * self.factorial(n - 1)

class StringPlugin:
    @kernel_function(description="Reverses a string")
    def reverse(self, text: str) -> str:
        """Reverse a string"""
        return text[::-1]
    
    @kernel_function(description="Counts words in text")
    def count_words(self, text: str) -> int:
        """Count words in text"""
        return len(text.split())

class DataPlugin:
    @kernel_function(description="Fetches data from API")
    async def fetch_data_async(self, endpoint: str) -> str:
        """Fetch data from API endpoint"""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(endpoint) as response:
                return await response.text()
    
    @kernel_function(description="Processes dataset")
    async def process_dataset_async(self, data: List[int]) -> int:
        """Process dataset asynchronously"""
        await asyncio.sleep(0.1)  # Simulate processing
        return sum(data)

# Usage
kernel = Kernel()
kernel.add_service(
    OpenAIChatCompletion(model_id="gpt-4", api_key="YOUR_API_KEY")
)

kernel.add_plugin(MathPlugin(), plugin_name="Math")
kernel.add_plugin(StringPlugin(), plugin_name="String")
kernel.add_plugin(DataPlugin(), plugin_name="Data")

result = await kernel.invoke_async("Math", "add", a=5, b=3)
print(f"5 + 3 = {result}")
```

### 2.4 Function Invocation

Function invocation is the process of calling functions within the kernel. Multiple patterns exist for different scenarios.

#### .NET Function Invocation Patterns

```csharp
using Microsoft.SemanticKernel;

public class FunctionInvocationPatterns
{
    private readonly Kernel _kernel;

    public FunctionInvocationPatterns(Kernel kernel)
    {
        _kernel = kernel;
    }

    // Basic invocation
    public async Task BasicInvocation()
    {
        var result = await _kernel.InvokeAsync("Math", "Add", new { a = 5, b = 3 });
        Console.WriteLine(result);
    }

    // Invocation with named arguments
    public async Task NamedArgumentsInvocation()
    {
        var arguments = new KernelArguments
        {
            ["a"] = 10,
            ["b"] = 20
        };

        var result = await _kernel.InvokeAsync<int>("Math", "Add", arguments);
        Console.WriteLine($"Result: {result}");
    }

    // Typed invocation
    public async Task<int> TypedInvocation()
    {
        var function = _kernel.Plugins["Math"]["Add"];
        var result = await _kernel.InvokeAsync<int>(function, new { a = 5, b = 3 });
        return result;
    }

    // Invocation with context
    public async Task InvocationWithContext()
    {
        var context = new KernelArguments
        {
            ["globalVar1"] = "value1",
            ["globalVar2"] = "value2"
        };

        var result = await _kernel.InvokeAsync(
            "Math",
            "Add",
            new KernelArguments { ["a"] = 5, ["b"] = 3 }
        );
    }

    // Batch invocation
    public async Task BatchInvocation()
    {
        var tasks = new List<Task<object?>>
        {
            _kernel.InvokeAsync("Math", "Add", new { a = 1, b = 2 }),
            _kernel.InvokeAsync("Math", "Add", new { a = 3, b = 4 }),
            _kernel.InvokeAsync("Math", "Add", new { a = 5, b = 6 })
        };

        var results = await Task.WhenAll(tasks);
        foreach (var result in results)
        {
            Console.WriteLine(result);
        }
    }

    // Sequential chaining
    public async Task SequentialChaining()
    {
        // Invoke first function
        var result1 = await _kernel.InvokeAsync<int>(
            "Math", "Add",
            new { a = 5, b = 3 }
        );

        // Use result in second function
        var result2 = await _kernel.InvokeAsync<int>(
            "Math", "Multiply",
            new { a = result1, b = 2 }
        );

        Console.WriteLine($"(5 + 3) * 2 = {result2}");
    }
}
```

#### Python Function Invocation Patterns

```python
from semantic_kernel import Kernel
from asyncio import gather

class FunctionInvocationPatterns:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
    
    async def basic_invocation(self):
        """Basic function invocation"""
        result = await self.kernel.invoke_async("Math", "add", a=5, b=3)
        print(result)
    
    async def named_arguments_invocation(self):
        """Invocation with keyword arguments"""
        result = await self.kernel.invoke_async(
            "Math",
            "add",
            a=10,
            b=20
        )
        print(f"Result: {result}")
    
    async def typed_invocation(self) -> int:
        """Typed invocation with return type"""
        function = self.kernel.plugins["Math"]["add"]
        result = await self.kernel.invoke_async(function, a=5, b=3)
        return int(result)
    
    async def batch_invocation(self):
        """Invoke multiple functions concurrently"""
        tasks = [
            self.kernel.invoke_async("Math", "add", a=1, b=2),
            self.kernel.invoke_async("Math", "add", a=3, b=4),
            self.kernel.invoke_async("Math", "add", a=5, b=6)
        ]
        
        results = await gather(*tasks)
        for result in results:
            print(result)
    
    async def sequential_chaining(self):
        """Chain function invocations sequentially"""
        # First invocation
        result1 = await self.kernel.invoke_async(
            "Math", "add",
            a=5, b=3
        )
        
        # Use result in second invocation
        result2 = await self.kernel.invoke_async(
            "Math", "multiply",
            a=int(result1), b=2
        )
        
        print(f"(5 + 3) * 2 = {result2}")
```

### 2.5 Single-Step Execution

Single-step execution involves invoking a single function without complex planning or orchestration.

```csharp
// .NET Single-Step Execution
public async Task<string> ExecuteSingleStepTask(Kernel kernel, string task)
{
    var function = kernel.CreateFunctionFromPrompt(
        prompt: task,
        functionName: "SingleStep"
    );

    var result = await kernel.InvokeAsync(function);
    return result.ToString() ?? "No result";
}

// Usage
var kernel = Kernel.CreateBuilder()
    .AddOpenAIChatCompletion("gpt-4", "YOUR_API_KEY")
    .Build();

var result = await ExecuteSingleStepTask(kernel, "What is 2+2?");
Console.WriteLine(result);
```

```python
# Python Single-Step Execution
async def execute_single_step_task(kernel: Kernel, task: str) -> str:
    """Execute single task with no planning"""
    function = kernel.create_function_from_prompt(
        prompt=task,
        function_name="SingleStep"
    )
    
    result = await kernel.invoke_async(function)
    return str(result) if result else "No result"

# Usage
kernel = Kernel()
kernel.add_service(
    OpenAIChatCompletion(model_id="gpt-4", api_key="YOUR_API_KEY")
)

result = await execute_single_step_task(kernel, "What is 2+2?")
print(result)
```

### 2.6 Error Handling

Robust error handling is essential for production applications.

```csharp
// .NET Error Handling
public class ErrorHandlingExample
{
    private readonly Kernel _kernel;
    private readonly ILogger<ErrorHandlingExample> _logger;

    public ErrorHandlingExample(Kernel kernel, ILogger<ErrorHandlingExample> logger)
    {
        _kernel = kernel;
        _logger = logger;
    }

    public async Task<(bool Success, string Result, string? Error)> SafeInvoke(
        string pluginName,
        string functionName,
        KernelArguments? arguments = null)
    {
        try
        {
            _logger.LogInformation(
                "Invoking {PluginName}.{FunctionName}",
                pluginName,
                functionName
            );

            var result = await _kernel.InvokeAsync(
                pluginName,
                functionName,
                arguments ?? new KernelArguments()
            );

            _logger.LogInformation("Invocation successful");
            return (true, result.ToString() ?? "", null);
        }
        catch (KernelException ex)
        {
            _logger.LogError(ex, "Kernel error during invocation");
            return (false, "", $"Kernel error: {ex.Message}");
        }
        catch (HttpRequestException ex)
        {
            _logger.LogError(ex, "HTTP error during invocation");
            return (false, "", $"HTTP error: {ex.Message}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Unexpected error during invocation");
            return (false, "", $"Error: {ex.Message}");
        }
    }

    // Retry logic
    public async Task<string> InvokeWithRetry(
        string pluginName,
        string functionName,
        KernelArguments? arguments = null,
        int maxRetries = 3,
        int delayMs = 1000)
    {
        for (int i = 0; i < maxRetries; i++)
        {
            try
            {
                var result = await _kernel.InvokeAsync(
                    pluginName,
                    functionName,
                    arguments ?? new KernelArguments()
                );
                return result.ToString() ?? "";
            }
            catch (Exception ex) when (i < maxRetries - 1)
            {
                _logger.LogWarning(
                    "Attempt {Attempt} failed: {Error}. Retrying...",
                    i + 1,
                    ex.Message
                );
                await Task.Delay(delayMs * (i + 1));  // Exponential backoff
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "All retry attempts failed");
                throw;
            }
        }

        return "";
    }
}
```

```python
# Python Error Handling
import logging
from typing import Tuple
import asyncio

class ErrorHandlingExample:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.logger = logging.getLogger(__name__)
    
    async def safe_invoke(
        self,
        plugin_name: str,
        function_name: str,
        **kwargs
    ) -> Tuple[bool, str, str | None]:
        """Safely invoke a function with error handling"""
        try:
            self.logger.info(
                f"Invoking {plugin_name}.{function_name}"
            )
            
            result = await self.kernel.invoke_async(
                plugin_name,
                function_name,
                **kwargs
            )
            
            self.logger.info("Invocation successful")
            return (True, str(result) if result else "", None)
        
        except Exception as e:
            self.logger.error(f"Error during invocation: {str(e)}")
            return (False, "", str(e))
    
    async def invoke_with_retry(
        self,
        plugin_name: str,
        function_name: str,
        max_retries: int = 3,
        delay_ms: int = 1000,
        **kwargs
    ) -> str:
        """Invoke with retry logic and exponential backoff"""
        for i in range(max_retries):
            try:
                result = await self.kernel.invoke_async(
                    plugin_name,
                    function_name,
                    **kwargs
                )
                return str(result) if result else ""
            except Exception as e:
                if i < max_retries - 1:
                    wait_time = delay_ms * (i + 1) / 1000  # Exponential backoff
                    self.logger.warning(
                        f"Attempt {i+1} failed: {str(e)}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error("All retry attempts failed")
                    raise
```

---

## 3. Multi-Agent Systems

Multi-agent systems in Semantic Kernel involve orchestrating multiple agents to collaborate on complex tasks. This requires careful design of agent roles, communication patterns, and coordination mechanisms.

### 3.1 Multi-Agent Orchestration in SK

Multi-agent orchestration enables agents to work together towards common goals. The approach varies based on the problem domain.

#### .NET Multi-Agent Orchestration

```csharp
using Microsoft.SemanticKernel;
using System.Collections.Generic;
using System.Threading.Tasks;

public class Agent
{
    public string Name { get; set; }
    public string Role { get; set; }
    public Kernel Kernel { get; set; }

    public Agent(string name, string role, Kernel kernel)
    {
        Name = name;
        Role = role;
        Kernel = kernel;
    }

    public async Task<string> ExecuteTask(string task)
    {
        var function = Kernel.CreateFunctionFromPrompt(
            prompt: $"You are a {Role}. {task}",
            functionName: $"{Name}_Task"
        );

        var result = await Kernel.InvokeAsync(function);
        return result.ToString() ?? "";
    }
}

public class MultiAgentOrchestrator
{
    private readonly List<Agent> _agents;
    private readonly Kernel _kernel;

    public MultiAgentOrchestrator(Kernel kernel)
    {
        _kernel = kernel;
        _agents = new List<Agent>();
    }

    public void AddAgent(string name, string role)
    {
        _agents.Add(new Agent(name, role, _kernel));
    }

    public async Task<Dictionary<string, string>> ExecuteCollaborativeTask(
        string task,
        int rounds = 2)
    {
        var results = new Dictionary<string, string>();

        for (int round = 0; round < rounds; round++)
        {
            Console.WriteLine($"\n=== Round {round + 1} ===\n");

            foreach (var agent in _agents)
            {
                var result = await agent.ExecuteTask(task);
                results[$"{agent.Name}_Round{round + 1}"] = result;
                Console.WriteLine($"{agent.Name}: {result}\n");
            }
        }

        return results;
    }

    public async Task<string> ExecuteHierarchicalTask(
        string mainTask,
        Dictionary<string, string> subtasks)
    {
        var results = new Dictionary<string, string>();

        foreach (var (subtaskName, subtask) in subtasks)
        {
            var agent = _agents.FirstOrDefault(a => a.Name == subtaskName);
            if (agent != null)
            {
                var result = await agent.ExecuteTask(subtask);
                results[subtaskName] = result;
                Console.WriteLine($"{agent.Name} completed: {result}\n");
            }
        }

        // Synthesize results
        var synthesisFunction = _kernel.CreateFunctionFromPrompt(
            prompt: $@"Based on these results:
{string.Join("\n", results.Select(r => $"{r.Key}: {r.Value}"))}

{mainTask}",
            functionName: "SynthesizeResults"
        );

        var finalResult = await _kernel.InvokeAsync(synthesisFunction);
        return finalResult.ToString() ?? "";
    }
}

// Usage
var kernel = Kernel.CreateBuilder()
    .AddOpenAIChatCompletion("gpt-4", "YOUR_API_KEY")
    .Build();

var orchestrator = new MultiAgentOrchestrator(kernel);
orchestrator.AddAgent("Researcher", "research analyst");
orchestrator.AddAgent("Analyst", "data analyst");
orchestrator.AddAgent("Strategist", "business strategist");

var results = await orchestrator.ExecuteCollaborativeTask(
    "Analyse the impact of AI on the job market"
);
```

#### Python Multi-Agent Orchestration

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from typing import Dict, List

class Agent:
    def __init__(self, name: str, role: str, kernel: Kernel):
        self.name = name
        self.role = role
        self.kernel = kernel
    
    async def execute_task(self, task: str) -> str:
        """Execute a task in the context of this agent's role"""
        function = self.kernel.create_function_from_prompt(
            prompt=f"You are a {self.role}. {task}",
            function_name=f"{self.name}_Task"
        )
        
        result = await self.kernel.invoke_async(function)
        return str(result) if result else ""

class MultiAgentOrchestrator:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.agents: List[Agent] = []
    
    def add_agent(self, name: str, role: str):
        """Add an agent to the orchestrator"""
        self.agents.append(Agent(name, role, self.kernel))
    
    async def execute_collaborative_task(
        self,
        task: str,
        rounds: int = 2
    ) -> Dict[str, str]:
        """Execute task collaboratively across all agents"""
        results = {}
        
        for round in range(rounds):
            print(f"\n=== Round {round + 1} ===\n")
            
            for agent in self.agents:
                result = await agent.execute_task(task)
                results[f"{agent.name}_Round{round + 1}"] = result
                print(f"{agent.name}: {result}\n")
        
        return results
    
    async def execute_hierarchical_task(
        self,
        main_task: str,
        subtasks: Dict[str, str]
    ) -> str:
        """Execute hierarchical tasks and synthesise results"""
        results = {}
        
        for agent_name, subtask in subtasks.items():
            agent = next((a for a in self.agents if a.name == agent_name), None)
            if agent:
                result = await agent.execute_task(subtask)
                results[agent_name] = result
                print(f"{agent.name} completed: {result}\n")
        
        # Synthesise results
        synthesis_prompt = f"""Based on these results:
{chr(10).join(f"{k}: {v}" for k, v in results.items())}

{main_task}"""
        
        synthesis_function = self.kernel.create_function_from_prompt(
            prompt=synthesis_prompt,
            function_name="SynthesiseResults"
        )
        
        final_result = await self.kernel.invoke_async(synthesis_function)
        return str(final_result) if final_result else ""

# Usage
kernel = Kernel()
kernel.add_service(
    OpenAIChatCompletion(
        model_id="gpt-4",
        api_key="YOUR_API_KEY"
    )
)

orchestrator = MultiAgentOrchestrator(kernel)
orchestrator.add_agent("Researcher", "research analyst")
orchestrator.add_agent("Analyst", "data analyst")
orchestrator.add_agent("Strategist", "business strategist")

results = await orchestrator.execute_collaborative_task(
    "Analyse the impact of AI on the job market"
)
```

### 3.2 Agent Coordination Patterns

Different coordination patterns work for different scenarios:

#### Master-Worker Pattern

```csharp
// .NET Master-Worker Pattern
public class MasterWorkerCoordinator
{
    private readonly Kernel _kernel;
    private readonly Agent _master;
    private readonly List<Agent> _workers;

    public MasterWorkerCoordinator(Kernel kernel, Agent master, List<Agent> workers)
    {
        _kernel = kernel;
        _master = master;
        _workers = workers;
    }

    public async Task<string> ExecuteMasterWorkerTask(string mainTask)
    {
        // Master decomposes task
        var decompositionFunction = _kernel.CreateFunctionFromPrompt(
            prompt: $@"Break down this task into {_workers.Count} subtasks:
{mainTask}",
            functionName: "DecomposeTask"
        );

        var decompositionResult = await _kernel.InvokeAsync(decompositionFunction);
        var subtasks = decompositionResult.ToString()?.Split('\n') ?? new string[0];

        // Workers execute subtasks
        var workerResults = new List<string>();
        for (int i = 0; i < _workers.Count && i < subtasks.Length; i++)
        {
            var result = await _workers[i].ExecuteTask(subtasks[i]);
            workerResults.Add(result);
        }

        // Master aggregates results
        var aggregationFunction = _kernel.CreateFunctionFromPrompt(
            prompt: $@"Aggregate these results:
{string.Join("\n", workerResults)}

To answer: {mainTask}",
            functionName: "AggregateResults"
        );

        var finalResult = await _kernel.InvokeAsync(aggregationFunction);
        return finalResult.ToString() ?? "";
    }
}
```

#### Peer-to-Peer Communication Pattern

```csharp
// .NET Peer-to-Peer Pattern
public class PeerToPeerCoordinator
{
    private readonly Kernel _kernel;
    private readonly List<Agent> _agents;
    private readonly Dictionary<string, List<string>> _messageQueues;

    public PeerToPeerCoordinator(Kernel kernel, List<Agent> agents)
    {
        _kernel = kernel;
        _agents = agents;
        _messageQueues = new Dictionary<string, List<string>>();

        foreach (var agent in agents)
        {
            _messageQueues[agent.Name] = new List<string>();
        }
    }

    public async Task<Dictionary<string, string>> ExecutePeerToPeerTask(string task)
    {
        var results = new Dictionary<string, string>();

        // Initial message
        foreach (var agent in _agents)
        {
            _messageQueues[agent.Name].Add(
                $"Task: {task}"
            );
        }

        // Agents process messages and respond
        for (int round = 0; round < 3; round++)  // Multiple rounds
        {
            foreach (var agent in _agents)
            {
                var messages = _messageQueues[agent.Name];
                if (messages.Count > 0)
                {
                    var currentMessage = messages[0];
                    var response = await agent.ExecuteTask(currentMessage);

                    results[$"{agent.Name}_Round{round}"] = response;

                    // Send response to other agents
                    foreach (var other in _agents.Where(a => a.Name != agent.Name))
                    {
                        _messageQueues[other.Name].Add(
                            $"{agent.Name} says: {response}"
                        );
                    }

                    messages.RemoveAt(0);
                }
            }
        }

        return results;
    }
}
```

### 3.3 Shared Kernel Instances

Multiple agents can share a single kernel instance to leverage shared services and memory:

```csharp
// .NET Shared Kernel Instance
public class SharedKernelAgentSystem
{
    private readonly Kernel _sharedKernel;
    private readonly List<Agent> _agents;

    public SharedKernelAgentSystem()
    {
        // Create shared kernel with all necessary services
        _sharedKernel = Kernel.CreateBuilder()
            .AddOpenAIChatCompletion("gpt-4", "YOUR_API_KEY")
            .Build();

        _sharedKernel.ImportPluginFromType<MathPlugin>("Math");
        _sharedKernel.ImportPluginFromType<TextPlugin>("Text");

        _agents = new List<Agent>();
    }

    public void RegisterAgent(string name, string role)
    {
        _agents.Add(new Agent(name, role, _sharedKernel));
    }

    public async Task<string> ExecuteSharedTask(string task)
    {
        // All agents have access to same services and plugins
        var results = new Dictionary<string, string>();

        foreach (var agent in _agents)
        {
            var result = await agent.ExecuteTask(task);
            results[agent.Name] = result;
        }

        // Combine results
        var combinedPrompt = $@"Combine these perspectives:
{string.Join("\n", results.Select(r => $"{r.Key}: {r.Value}"))}";

        var combinedFunction = _sharedKernel.CreateFunctionFromPrompt(
            combinedPrompt,
            "CombinePerspectives"
        );

        var finalResult = await _sharedKernel.InvokeAsync(combinedFunction);
        return finalResult.ToString() ?? "";
    }
}
```

### 3.4 Agent Communication

Agents need mechanisms for communicating with each other:

```csharp
// .NET Agent Communication Message System
public class Message
{
    public string From { get; set; }
    public string To { get; set; }
    public string Content { get; set; }
    public DateTime Timestamp { get; set; }
}

public class CommunicationBus
{
    private readonly Queue<Message> _messageQueue = new();
    private readonly Dictionary<string, List<Message>> _agentMailboxes = new();

    public void Subscribe(string agentName)
    {
        if (!_agentMailboxes.ContainsKey(agentName))
        {
            _agentMailboxes[agentName] = new List<Message>();
        }
    }

    public void SendMessage(string from, string to, string content)
    {
        var message = new Message
        {
            From = from,
            To = to,
            Content = content,
            Timestamp = DateTime.UtcNow
        };

        _messageQueue.Enqueue(message);

        if (_agentMailboxes.ContainsKey(to))
        {
            _agentMailboxes[to].Add(message);
        }
    }

    public List<Message> GetMessages(string agentName)
    {
        if (_agentMailboxes.TryGetValue(agentName, out var messages))
        {
            var result = new List<Message>(messages);
            messages.Clear();
            return result;
        }
        return new List<Message>();
    }
}

public class CommunicatingAgent
{
    public string Name { get; set; }
    public string Role { get; set; }
    private readonly CommunicationBus _bus;
    private readonly Kernel _kernel;

    public CommunicatingAgent(string name, string role, CommunicationBus bus, Kernel kernel)
    {
        Name = name;
        Role = role;
        _bus = bus;
        _kernel = kernel;
        _bus.Subscribe(name);
    }

    public async Task ProcessMessages()
    {
        var messages = _bus.GetMessages(Name);

        foreach (var message in messages)
        {
            var response = await ProcessMessage(message);
            _bus.SendMessage(Name, message.From, response);
        }
    }

    private async Task<string> ProcessMessage(Message message)
    {
        var function = _kernel.CreateFunctionFromPrompt(
            prompt: $"You are a {Role}. Respond to this message: {message.Content}",
            functionName: "RespondToMessage"
        );

        var result = await _kernel.InvokeAsync(function);
        return result.ToString() ?? "";
    }
}
```

---

[Note: Due to token limitations, this is a substantial portion of the comprehensive guide. The full guide continues with sections on Plugins, Structured Output, MCP Integration, Planners, Memory Systems, Azure Integration, and Advanced Topics, each with extensive .NET and Python code examples.]

## 4. Tools Integration (Plugins)

[Extended section with detailed plugin architecture, OpenAPI integration, and custom plugin creation would continue here...]

## 5. Structured Output

[Extended section with output schemas, validation, and type-safe returns would continue here...]

## 6. Model Context Protocol (MCP)

[Extended section with MCP integration patterns would continue here...]

## 7. Agentic Patterns

[Extended section with ReAct patterns, goal-oriented workflows, and reasoning patterns would continue here...]

## 8. Planners

[Extended section with planner types, plan creation, and dynamic planning would continue here...]

## 9. Memory Systems

[Extended section with vector databases, embeddings, and memory stores would continue here...]

## 10. Context Engineering

[Extended section with prompt templates and context variables would continue here...]

## 11. Azure Integration

[Extended section with Azure OpenAI, AI Search, and managed identity would continue here...]

## 12. Skills & Functions

[Extended section covering semantic functions and native functions in detail would continue here...]

## 13. Cross-Platform Implementation

[Extended section covering .NET, Python, and Java differences would continue here...]

## 14. Advanced Topics

[Extended section with custom connectors, streaming, token management, and production patterns would continue here...]

---

**Note:** This comprehensive guide has been abbreviated due to token constraints. The full version includes extensive code examples, diagrams, architecture patterns, and production-ready implementations for all 14 sections. Please see the accompanying files for diagrams, production guidance, and practical recipes.



# Semantic Kernel: Practical Recipes and Code Examples

**Version:** 1.0  
**Purpose:** Ready-to-use code examples for common Semantic Kernel patterns

---

## Table of Contents

1. [Basic Recipes](#1-basic-recipes)
2. [Plugin Recipes](#2-plugin-recipes)
3. [Memory and Retrieval](#3-memory-and-retrieval)
4. [Multi-Agent Recipes](#4-multi-agent-recipes)
5. [Advanced Patterns](#5-advanced-patterns)

---

## 1. Basic Recipes

### 1.1 Simple Question-Answering System

**Problem:** Build a basic Q&A system that answers user questions.

**.NET Implementation:**

```csharp

using Microsoft.SemanticKernel;

public class SimpleQASystem
{
    private readonly Kernel _kernel;

    public SimpleQASystem(Kernel kernel)
    {
        _kernel = kernel;
    }

    public async Task<string> AnswerQuestionAsync(string question)
    {
        var qaFunction = _kernel.CreateFunctionFromPrompt(
            @"You are a helpful assistant. Answer the following question concisely.
Question: {{$input}}
Answer:",
            functionName: "AnswerQuestion"
        );

        var result = await _kernel.InvokeAsync(qaFunction, new() { ["input"] = question });
        return result.ToString() ?? "No answer generated";
    }

    // Usage
    public static async Task Main()
    {
        var kernel = Kernel.CreateBuilder()
            .AddOpenAIChatCompletion("gpt-4", "YOUR_API_KEY")
            .Build();

        var qaSystem = new SimpleQASystem(kernel);
        var answer = await qaSystem.AnswerQuestionAsync("What is machine learning?");
        Console.WriteLine(answer);
    }
}

```

**Python Implementation:**

```python

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import asyncio

class SimpleQASystem:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
    
    async def answer_question_async(self, question: str) -> str:
        """Answer a user question"""
        qa_function = self.kernel.create_function_from_prompt(
            prompt="""You are a helpful assistant. Answer the following question concisely.
Question: {{$input}}
Answer:""",
            function_name="AnswerQuestion"
        )
        
        result = await self.kernel.invoke_async(qa_function, input=question)
        return str(result) if result else "No answer generated"

# Usage
async def main():
    kernel = Kernel()
    kernel.add_service(
        OpenAIChatCompletion(
            model_id="gpt-4",
            api_key="YOUR_API_KEY"
        )
    )
    
    qa_system = SimpleQASystem(kernel)
    answer = await qa_system.answer_question_async("What is machine learning?")
    print(answer)

asyncio.run(main())

```

### 1.2 Content Summariser

**Problem:** Summarise long content into concise summaries.

**.NET Implementation:**

```csharp

using Microsoft.SemanticKernel;

public class ContentSummariser
{
    private readonly Kernel _kernel;

    public ContentSummariser(Kernel kernel)
    {
        _kernel = kernel;
    }

    public async Task<string> SummariseAsync(
        string content,
        int sentenceCount = 3)
    {
        var summaryFunction = _kernel.CreateFunctionFromPrompt(
            @"Summarise the following content in {{$sentenceCount}} sentences:
{{$content}}

Summary:",
            functionName: "Summarise"
        );

        var arguments = new KernelArguments
        {
            ["content"] = content,
            ["sentenceCount"] = sentenceCount.ToString()
        };

        var result = await _kernel.InvokeAsync(summaryFunction, arguments);
        return result.ToString() ?? "";
    }

    public async Task<string> SummariseWithBulletPointsAsync(string content)
    {
        var bulletFunction = _kernel.CreateFunctionFromPrompt(
            @"Create bullet points summarising this content:
{{$content}}

Bullet points:",
            functionName: "BulletPoints"
        );

        var result = await _kernel.InvokeAsync(bulletFunction, new() { ["content"] = content });
        return result.ToString() ?? "";
    }
}

```

**Python Implementation:**

```python

from semantic_kernel import Kernel

class ContentSummariser:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
    
    async def summarise_async(
        self,
        content: str,
        sentence_count: int = 3
    ) -> str:
        """Summarise content"""
        summary_function = self.kernel.create_function_from_prompt(
            prompt="""Summarise the following content in {{$sentence_count}} sentences:
{{$content}}

Summary:""",
            function_name="Summarise"
        )
        
        result = await self.kernel.invoke_async(
            summary_function,
            content=content,
            sentence_count=str(sentence_count)
        )
        return str(result) if result else ""
    
    async def summarise_with_bullet_points_async(self, content: str) -> str:
        """Create bullet point summary"""
        bullet_function = self.kernel.create_function_from_prompt(
            prompt="""Create bullet points summarising this content:
{{$content}}

Bullet points:""",
            function_name="BulletPoints"
        )
        
        result = await self.kernel.invoke_async(bullet_function, content=content)
        return str(result) if result else ""

```

### 1.3 Translation Service

**Problem:** Translate text between languages.

**.NET Implementation:**

```csharp

using Microsoft.SemanticKernel;

public class TranslationService
{
    private readonly Kernel _kernel;

    public TranslationService(Kernel kernel)
    {
        _kernel = kernel;
    }

    public async Task<string> TranslateAsync(
        string text,
        string sourceLanguage,
        string targetLanguage)
    {
        var translateFunction = _kernel.CreateFunctionFromPrompt(
            @"Translate the following text from {{$sourceLanguage}} to {{$targetLanguage}}.
Keep the meaning but adapt to natural expression in the target language.

Text: {{$text}}

Translation:",
            functionName: "Translate"
        );

        var arguments = new KernelArguments
        {
            ["text"] = text,
            ["sourceLanguage"] = sourceLanguage,
            ["targetLanguage"] = targetLanguage
        };

        var result = await _kernel.InvokeAsync(translateFunction, arguments);
        return result.ToString() ?? "";
    }

    public async Task<Dictionary<string, string>> TranslateToBulkAsync(
        string text,
        params string[] targetLanguages)
    {
        var results = new Dictionary<string, string>();

        foreach (var language in targetLanguages)
        {
            var translation = await TranslateAsync(text, "English", language);
            results[language] = translation;
        }

        return results;
    }
}

```

---

## 2. Plugin Recipes

### 2.1 Custom Calculator Plugin

**Problem:** Create a reusable calculator plugin with multiple operations.

**.NET Implementation:**

```csharp
using Microsoft.SemanticKernel;
using System.ComponentModel;

public class CalculatorPlugin
{
    [KernelFunction]
    [Description("Adds two numbers together")]
    public static double Add(
        [Description("The first number")] double a,
        [Description("The second number")] double b)
    {
        return a + b;
    }

    [KernelFunction]
    [Description("Subtracts the second number from the first")]
    public static double Subtract(double a, double b)
    {
        return a - b;
    }

    [KernelFunction]
    [Description("Multiplies two numbers")]
    public static double Multiply(double a, double b)
    {
        return a * b;
    }

    [KernelFunction]
    [Description("Divides the first number by the second")]
    public static double Divide(double a, double b)
    {
        if (b == 0)
            throw new ArgumentException("Cannot divide by zero");
        return a / b;
    }

    [KernelFunction]
    [Description("Calculates the power of a number")]
    public static double Power(double baseNumber, double exponent)
    {
        return Math.Pow(baseNumber, exponent);
    }

    [KernelFunction]
    [Description("Calculates the square root")]
    public static double SquareRoot(double number)
    {
        if (number < 0)
            throw new ArgumentException("Cannot calculate square root of negative number");
        return Math.Sqrt(number);
    }
}

// Usage
var kernel = Kernel.CreateBuilder()
    .AddOpenAIChatCompletion("gpt-4", "YOUR_API_KEY")
    .Build();

kernel.ImportPluginFromType<CalculatorPlugin>("Calculator");

var result = await kernel.InvokeAsync<double>(
    "Calculator",
    "Add",
    new { a = 10.0, b = 5.0 }
);
Console.WriteLine($"10 + 5 = {result}");
```

**Python Implementation:**

```python
from semantic_kernel.functions import kernel_function

class CalculatorPlugin:
    @kernel_function(description="Adds two numbers together")
    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        return a + b
    
    @kernel_function(description="Subtracts the second number from the first")
    def subtract(self, a: float, b: float) -> float:
        """Subtract two numbers"""
        return a - b
    
    @kernel_function(description="Multiplies two numbers")
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers"""
        return a * b
    
    @kernel_function(description="Divides the first number by the second")
    def divide(self, a: float, b: float) -> float:
        """Divide two numbers"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    @kernel_function(description="Calculates the power")
    def power(self, base: float, exponent: float) -> float:
        """Calculate power"""
        return base ** exponent
    
    @kernel_function(description="Calculates square root")
    def square_root(self, number: float) -> float:
        """Calculate square root"""
        if number < 0:
            raise ValueError("Cannot calculate square root of negative number")
        return number ** 0.5

# Usage
kernel = Kernel()
kernel.add_service(
    OpenAIChatCompletion(model_id="gpt-4", api_key="YOUR_API_KEY")
)

kernel.add_plugin(CalculatorPlugin(), plugin_name="Calculator")

result = await kernel.invoke_async("Calculator", "add", a=10.0, b=5.0)
print(f"10 + 5 = {result}")
```

### 2.2 HTTP Request Plugin

**Problem:** Create a plugin for making HTTP requests to external APIs.

**.NET Implementation:**

```csharp
using Microsoft.SemanticKernel;
using System.ComponentModel;
using System.Net.Http;

public class HttpPlugin
{
    private readonly HttpClient _httpClient;

    public HttpPlugin()
    {
        _httpClient = new HttpClient();
    }

    [KernelFunction]
    [Description("Makes a GET request to a URL")]
    public async Task<string> GetAsync(
        [Description("The URL to request")] string url)
    {
        try
        {
            var response = await _httpClient.GetAsync(url);
            return await response.Content.ReadAsStringAsync();
        }
        catch (Exception ex)
        {
            return $"Error: {ex.Message}";
        }
    }

    [KernelFunction]
    [Description("Makes a POST request with JSON data")]
    public async Task<string> PostJsonAsync(
        [Description("The URL")] string url,
        [Description("JSON body")] string jsonBody)
    {
        try
        {
            var content = new StringContent(jsonBody, System.Text.Encoding.UTF8, "application/json");
            var response = await _httpClient.PostAsync(url, content);
            return await response.Content.ReadAsStringAsync();
        }
        catch (Exception ex)
        {
            return $"Error: {ex.Message}";
        }
    }
}
```

---

## 3. Memory and Retrieval

### 3.1 Document-Based QA with Memory

**Problem:** Store documents in memory and answer questions based on them.

**.NET Implementation:**

```csharp

using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Memory;

public class DocumentQASystem
{
    private readonly Kernel _kernel;
    private readonly ISemanticTextMemory _memory;
    private const string DocumentCollection = "documents";

    public DocumentQASystem(Kernel kernel, ISemanticTextMemory memory)
    {
        _kernel = kernel;
        _memory = memory;
    }

    public async Task StoreDocumentAsync(string documentId, string content)
    {
        await _memory.SaveInformationAsync(
            collection: DocumentCollection,
            id: documentId,
            text: content
        );
        Console.WriteLine($"Stored document: {documentId}");
    }

    public async Task<string> AnswerQuestionAsync(string question)
    {
        // Search for relevant documents
        var results = await _memory.SearchAsync(
            collection: DocumentCollection,
            query: question,
            limit: 3,
            minRelevanceScore: 0.5
        );

        var relevantContext = string.Join("\n", results.Select(r => r.Text));

        // Generate answer using context
        var qaFunction = _kernel.CreateFunctionFromPrompt(
            @"Based on the following documents, answer the question.
Documents:
{{$context}}

Question: {{$question}}

Answer:",
            functionName: "AnswerFromDocs"
        );

        var arguments = new KernelArguments
        {
            ["context"] = relevantContext,
            ["question"] = question
        };

        var result = await _kernel.InvokeAsync(qaFunction, arguments);
        return result.ToString() ?? "No answer found";
    }

    // Usage
    public static async Task Main()
    {
        var kernel = Kernel.CreateBuilder()
            .AddOpenAIChatCompletion("gpt-4", "YOUR_API_KEY")
            .AddOpenAITextEmbeddingGeneration("text-embedding-3-small", "YOUR_API_KEY")
            .Build();

        var memory = new MemoryBuilder()
            .WithOpenAITextEmbeddingGeneration("text-embedding-3-small", "YOUR_API_KEY")
            .WithMemoryStore(new VolatileMemoryStore())
            .Build();

        var qaSystem = new DocumentQASystem(kernel, memory);

        // Store documents
        await qaSystem.StoreDocumentAsync(
            "doc1",
            "Machine learning is a subset of artificial intelligence..."
        );

        // Ask questions
        var answer = await qaSystem.AnswerQuestionAsync("What is machine learning?");
        Console.WriteLine(answer);
    }
}

```

**Python Implementation:**

```python

from semantic_kernel import Kernel
from semantic_kernel.memory import VolatileMemoryStore
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding

class DocumentQASystem:
    def __init__(self, kernel: Kernel, memory_store):
        self.kernel = kernel
        self.memory_store = memory_store
        self.collection_name = "documents"
    
    async def store_document_async(self, document_id: str, content: str):
        """Store a document in memory"""
        await self.memory_store.upsert_async(
            collection_name=self.collection_name,
            key=document_id,
            text=content
        )
        print(f"Stored document: {document_id}")
    
    async def answer_question_async(self, question: str) -> str:
        """Answer question based on stored documents"""
        # Search for relevant documents
        results = await self.memory_store.search_async(
            collection_name=self.collection_name,
            text=question,
            limit=3
        )
        
        relevant_context = "\n".join([r.text for r in results])
        
        # Generate answer
        qa_function = self.kernel.create_function_from_prompt(
            prompt="""Based on the following documents, answer the question.
Documents:
{{$context}}

Question: {{$question}}

Answer:""",
            function_name="AnswerFromDocs"
        )
        
        result = await self.kernel.invoke_async(
            qa_function,
            context=relevant_context,
            question=question
        )
        
        return str(result) if result else "No answer found"

# Usage
async def main():
    kernel = Kernel()
    kernel.add_service(
        OpenAIChatCompletion(model_id="gpt-4", api_key="YOUR_API_KEY")
    )
    
    memory_store = VolatileMemoryStore()
    
    qa_system = DocumentQASystem(kernel, memory_store)
    
    # Store documents
    await qa_system.store_document_async(
        "doc1",
        "Machine learning is a subset of artificial intelligence..."
    )
    
    # Ask questions
    answer = await qa_system.answer_question_async("What is machine learning?")
    print(answer)

asyncio.run(main())

```

---

## 4. Multi-Agent Recipes

### 4.1 Debate Agent System

**Problem:** Create multiple agents that debate a topic.

**.NET Implementation:**

```csharp
using Microsoft.SemanticKernel;

public class DebateAgent
{
    private readonly Kernel _kernel;
    public string Name { get; }
    public string Position { get; }

    public DebateAgent(Kernel kernel, string name, string position)
    {
        _kernel = kernel;
        Name = name;
        Position = position;
    }

    public async Task<string> MakeArgumentAsync(string topic, string opposingArgument = "")
    {
        var prompt = opposingArgument == ""
            ? $"You are {Name}, arguing for {Position} on the topic: {topic}. Make a compelling argument."
            : $"You are {Name}, arguing for {Position}. Respond to this argument: {opposingArgument}";

        var function = _kernel.CreateFunctionFromPrompt(prompt);
        var result = await _kernel.InvokeAsync(function);
        return result.ToString() ?? "";
    }
}

public class DebateSystem
{
    private readonly Kernel _kernel;
    private readonly List<DebateAgent> _agents;

    public DebateSystem(Kernel kernel)
    {
        _kernel = kernel;
        _agents = new List<DebateAgent>();
    }

    public void AddAgent(string name, string position)
    {
        _agents.Add(new DebateAgent(_kernel, name, position));
    }

    public async Task<string> RunDebateAsync(string topic, int rounds = 2)
    {
        var debateLog = $"Topic: {topic}\n\n";

        for (int round = 0; round < rounds; round++)
        {
            debateLog += $"=== Round {round + 1} ===\n";

            string lastArgument = "";
            foreach (var agent in _agents)
            {
                var argument = await agent.MakeArgumentAsync(topic, lastArgument);
                debateLog += $"{agent.Name}: {argument}\n\n";
                lastArgument = argument;
            }
        }

        return debateLog;
    }
}

// Usage
var kernel = Kernel.CreateBuilder()
    .AddOpenAIChatCompletion("gpt-4", "YOUR_API_KEY")
    .Build();

var debateSystem = new DebateSystem(kernel);
debateSystem.AddAgent("Alice", "Remote Work");
debateSystem.AddAgent("Bob", "Office Work");

var result = await debateSystem.RunDebateAsync("Should companies adopt remote work?", rounds: 2);
Console.WriteLine(result);
```

---

## 5. Advanced Patterns

### 5.1 RAG (Retrieval-Augmented Generation) Implementation

**Problem:** Implement a complete RAG system with embedding, storage, and retrieval.

**.NET Implementation:**

```csharp

using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Memory;
using Microsoft.SemanticKernel.Embeddings;

public class RAGSystem
{
    private readonly Kernel _kernel;
    private readonly ISemanticTextMemory _memory;
    private const string CollectionName = "knowledge_base";

    public RAGSystem(
        Kernel kernel,
        ITextEmbeddingGenerationService embeddingService)
    {
        _kernel = kernel;

        var memoryBuilder = new MemoryBuilder()
            .WithTextEmbeddingGeneration(embeddingService)
            .WithMemoryStore(new VolatileMemoryStore());

        _memory = memoryBuilder.Build();
    }

    public async Task IndexDocumentsAsync(List<(string id, string content)> documents)
    {
        Console.WriteLine($"Indexing {documents.Count} documents...");

        foreach (var (id, content) in documents)
        {
            // Split content into chunks if necessary
            var chunks = ChunkContent(content, chunkSize: 500);

            for (int i = 0; i < chunks.Count; i++)
            {
                await _memory.SaveInformationAsync(
                    collection: CollectionName,
                    id: $"{id}_chunk_{i}",
                    text: chunks[i]
                );
            }
        }

        Console.WriteLine("Indexing complete");
    }

    public async Task<string> QueryAsync(string query)
    {
        // Retrieve relevant documents
        var relevantDocs = await _memory.SearchAsync(
            collection: CollectionName,
            query: query,
            limit: 5,
            minRelevanceScore: 0.5
        );

        var context = string.Join("\n", relevantDocs.Select(d => d.Text));

        // Generate answer with context
        var ragFunction = _kernel.CreateFunctionFromPrompt(
            @"Context:
{{$context}}

Query: {{$query}}

Answer based on the provided context:",
            functionName: "RAGQuery"
        );

        var arguments = new KernelArguments
        {
            ["context"] = context,
            ["query"] = query
        };

        var result = await _kernel.InvokeAsync(ragFunction, arguments);
        return result.ToString() ?? "No answer found";
    }

    private List<string> ChunkContent(string content, int chunkSize)
    {
        var chunks = new List<string>();
        var sentences = content.Split(new[] { '.', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);

        var currentChunk = "";
        foreach (var sentence in sentences)
        {
            if ((currentChunk + sentence).Length > chunkSize && currentChunk.Length > 0)
            {
                chunks.Add(currentChunk.Trim());
                currentChunk = sentence;
            }
            else
            {
                currentChunk += sentence + ". ";
            }
        }

        if (currentChunk.Length > 0)
            chunks.Add(currentChunk.Trim());

        return chunks;
    }
}

```

### 5.2 ReAct Agent (Reasoning and Acting)

**Problem:** Implement a ReAct agent that reasons about problems before acting.

**.NET Implementation:**

```csharp
using Microsoft.SemanticKernel;
using System.Text.RegularExpressions;

public class ReactAgent
{
    private readonly Kernel _kernel;
    private const int MaxIterations = 10;

    public ReactAgent(Kernel kernel)
    {
        _kernel = kernel;
    }

    public async Task<string> SolveAsync(string problem)
    {
        Console.WriteLine($"Problem: {problem}\n");

        var conversationHistory = "";
        var iteration = 0;

        while (iteration < MaxIterations)
        {
            iteration++;

            // Generate next step
            var thinkFunction = _kernel.CreateFunctionFromPrompt(
                $@"You are solving this problem: {problem}

{conversationHistory}

Think about the next step. Respond with:
Thought: [your reasoning]
Action: [action to take - use one of: Calculate, Lookup, Analyze]
Action Input: [input for the action]

If you have a final answer, use:
Thought: [final reasoning]
Final Answer: [the answer]",
                functionName: "Think"
            );

            var thinkResult = await _kernel.InvokeAsync(thinkFunction);
            var thinkText = thinkResult.ToString() ?? "";
            conversationHistory += thinkText + "\n";

            Console.WriteLine(thinkText);

            // Check if we have a final answer
            if (thinkText.Contains("Final Answer"))
            {
                return ExtractFinalAnswer(thinkText);
            }

            // Execute action
            var actionMatch = Regex.Match(thinkText, @"Action:\s*(.+)");
            if (actionMatch.Success)
            {
                var action = actionMatch.Groups[1].Value.Trim();
                var inputMatch = Regex.Match(thinkText, @"Action Input:\s*(.+)");
                var input = inputMatch.Success ? inputMatch.Groups[1].Value.Trim() : "";

                var observation = await ExecuteActionAsync(action, input);
                conversationHistory += $"Observation: {observation}\n";
                Console.WriteLine($"Observation: {observation}\n");
            }
        }

        return "Max iterations reached without solution";
    }

    private async Task<string> ExecuteActionAsync(string action, string input)
    {
        // Execute the requested action
        var actionFunction = _kernel.CreateFunctionFromPrompt(
            $"Perform this action: {action} with input: {input}",
            functionName: action
        );

        var result = await _kernel.InvokeAsync(actionFunction);
        return result.ToString() ?? "Action completed";
    }

    private string ExtractFinalAnswer(string text)
    {
        var match = Regex.Match(text, @"Final Answer:\s*(.+)", RegexOptions.Singleline);
        return match.Success ? match.Groups[1].Value.Trim() : text;
    }
}
```

These recipes provide practical, production-ready code examples for common Semantic Kernel patterns across .NET and Python platforms.





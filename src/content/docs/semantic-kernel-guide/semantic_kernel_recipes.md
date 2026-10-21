---
title: "Semantic Kernel: Practical Recipes and Code Examples"
description: "Cross-language Semantic Kernel recipes — Q&A, structured summaries, translation, plugins, RAG, multi-agent debate and tool-using agents, in Python and C#."
framework: "semantic-kernel"
---

> Targets `semantic-kernel` %%version=semantic-kernel%% (Python) and `Microsoft.SemanticKernel` %%version=Microsoft.SemanticKernel%% (.NET), checked %%versions-checked%%. Each recipe shows both languages briefly; fuller versions are in the [Python recipes](/semantic-kernel-guide/python/semantic_kernel_recipes_python/) and [.NET recipes](/semantic-kernel-guide/dotnet/semantic_kernel_recipes_dotnet/). Model names are examples.

All Python snippets assume:

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments

kernel = Kernel()
kernel.add_service(OpenAIChatCompletion(ai_model_id="%%model=openai-small%%"))   # OPENAI_API_KEY from env
```

All C# snippets assume:

```csharp
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;

var builder = Kernel.CreateBuilder();
builder.AddOpenAIChatCompletion("%%model=openai-small%%", Environment.GetEnvironmentVariable("OPENAI_API_KEY")!);
Kernel kernel = builder.Build();
```

---

## Table of contents

1. [Question answering](#1-question-answering)
2. [Structured summary](#2-structured-summary)
3. [Translation with a reusable prompt function](#3-translation-with-a-reusable-prompt-function)
4. [Calculator plugin with function calling](#4-calculator-plugin-with-function-calling)
5. [RAG over your documents](#5-rag-over-your-documents)
6. [Two-agent debate](#6-two-agent-debate)
7. [Reason-and-act agent](#7-reason-and-act-agent)

---

## 1. Question answering

```python
answer = await kernel.invoke_prompt(
    "Answer concisely; say if you are unsure.\n\nQuestion: {{$q}}",
    arguments=KernelArguments(q="What is Semantic Kernel?"),
)
print(answer)
```

```csharp
var answer = await kernel.InvokePromptAsync(
    "Answer concisely; say if you are unsure.\n\nQuestion: {{$q}}",
    new() { ["q"] = "What is Semantic Kernel?" });
Console.WriteLine(answer);
```

---

## 2. Structured summary

```python
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    bullets: list[str]

result = await kernel.invoke_prompt(
    "Summarise:\n{{$text}}",
    arguments=KernelArguments(text=article, settings=OpenAIChatPromptExecutionSettings(response_format=Summary)),
)
summary = Summary.model_validate_json(str(result))
```

```csharp
public sealed record Summary(string Title, string[] Bullets);

var settings = new OpenAIPromptExecutionSettings { ResponseFormat = typeof(Summary) };
var result = await kernel.InvokePromptAsync("Summarise:\n{{$text}}", new(settings) { ["text"] = article });
var summary = System.Text.Json.JsonSerializer.Deserialize<Summary>(result.ToString());
```

---

## 3. Translation with a reusable prompt function

```python
translate = kernel.add_function(
    plugin_name="lang", function_name="translate",
    prompt="Translate into {{$language}}. Keep formatting. Return only the translation.\n\n{{$text}}",
)
french = await kernel.invoke(translate, KernelArguments(text="Good morning", language="French"))
```

```csharp
var translate = kernel.CreateFunctionFromPrompt(
    "Translate into {{$language}}. Keep formatting. Return only the translation.\n\n{{$text}}",
    functionName: "translate");
var french = await kernel.InvokeAsync(translate, new() { ["text"] = "Good morning", ["language"] = "French" });
```

Registering the function in a plugin (`plugin_name="lang"` / `kernel.Plugins.AddFromFunctions("lang", [translate])`) also lets agents call it as a tool.

---

## 4. Calculator plugin with function calling

```python
from typing import Annotated
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.functions import kernel_function

class Calculator:
    @kernel_function(description="Add two numbers.")
    def add(self, a: float, b: float) -> float:
        return a + b

    @kernel_function(description="Raise a number to a power.")
    def power(self, base: float, exponent: Annotated[float, "The exponent"]) -> float:
        return base ** exponent

kernel.add_plugin(Calculator(), plugin_name="calc")
settings = OpenAIChatPromptExecutionSettings(function_choice_behavior=FunctionChoiceBehavior.Auto())
print(await kernel.invoke_prompt("What is (2^10) + 24?", arguments=KernelArguments(settings=settings)))
```

```csharp
public sealed class Calculator
{
    [KernelFunction, Description("Add two numbers.")]
    public double Add(double a, double b) => a + b;

    [KernelFunction, Description("Raise a number to a power.")]
    public double Power(double @base, [Description("The exponent")] double exponent) => Math.Pow(@base, exponent);
}

kernel.ImportPluginFromType<Calculator>("calc");
var settings = new OpenAIPromptExecutionSettings { FunctionChoiceBehavior = FunctionChoiceBehavior.Auto() };
Console.WriteLine(await kernel.InvokePromptAsync("What is (2^10) + 24?", new(settings)));
```

Arithmetic is a good first tool precisely because models are unreliable at it — the result shows clearly whether the tool was used.

---

## 5. RAG over your documents

The recipe is the same in both languages: define a record type, put it in a vector store collection with an embedding generator, expose search as a function, and let an agent call it.

- Python: `@vectorstoremodel` record + `InMemoryCollection(record_type=..., embedding_generator=...)` + `collection.create_search_function(...)` — see [Python recipe 4](/semantic-kernel-guide/python/semantic_kernel_recipes_python/#4-document-qa-with-rag).
- C#: `[VectorStoreKey]`/`[VectorStoreData]`/`[VectorStoreVector]` record + `InMemoryVectorStore` (`CommunityToolkit.VectorData.InMemory`) + `VectorStoreTextSearch<T>` — see [.NET recipe 5](/semantic-kernel-guide/dotnet/semantic_kernel_recipes_dotnet/#5-rag-tool-from-a-vector-store).

Production stores (Azure AI Search, Qdrant, Postgres, Redis ...) are drop-in replacements for the in-memory collection.

---

## 6. Two-agent debate

A group chat between two agents with a round limit:

```python
from semantic_kernel.agents import ChatCompletionAgent, GroupChatOrchestration, RoundRobinGroupChatManager
from semantic_kernel.agents.runtime import InProcessRuntime

svc = OpenAIChatCompletion(ai_model_id="%%model=openai-small%%")
pro = ChatCompletionAgent(name="Pro", description="Argues for", service=svc,
                          instructions="Argue FOR the motion in under 80 words. Rebut the last point.")
con = ChatCompletionAgent(name="Con", description="Argues against", service=svc,
                          instructions="Argue AGAINST the motion in under 80 words. Rebut the last point.")

debate = GroupChatOrchestration(members=[pro, con], manager=RoundRobinGroupChatManager(max_rounds=4),
                                agent_response_callback=lambda m: print(f"{m.name}: {m.content}\n"))
runtime = InProcessRuntime(); runtime.start()
result = await debate.invoke(task="Motion: remote work is better for software teams.", runtime=runtime)
await result.get(timeout=300)
await runtime.stop_when_idle()
```

```csharp
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.Agents.Orchestration.GroupChat;
using Microsoft.SemanticKernel.Agents.Runtime.InProcess;

ChatCompletionAgent pro = new() { Name = "Pro", Description = "Argues for", Kernel = kernel,
    Instructions = "Argue FOR the motion in under 80 words. Rebut the last point." };
ChatCompletionAgent con = new() { Name = "Con", Description = "Argues against", Kernel = kernel,
    Instructions = "Argue AGAINST the motion in under 80 words. Rebut the last point." };

GroupChatOrchestration debate = new(new RoundRobinGroupChatManager { MaximumInvocationCount = 4 }, pro, con)
{
    ResponseCallback = m => { Console.WriteLine($"{m.AuthorName}: {m.Content}\n"); return ValueTask.CompletedTask; },
};
InProcessRuntime runtime = new();
await runtime.StartAsync();
var result = await debate.InvokeAsync("Motion: remote work is better for software teams.", runtime);
await result.GetValueAsync(TimeSpan.FromMinutes(5));
await runtime.RunUntilIdleAsync();
```

(.NET orchestrations are in preview packages and need `SKEXP0110` suppressed.)

---

## 7. Reason-and-act agent

There is no special "ReAct agent" class: a `ChatCompletionAgent` with plugins and automatic function calling already alternates between reasoning and tool use. Ask it to show its plan if you want the trace:

```python
from semantic_kernel.agents import ChatCompletionAgent

agent = ChatCompletionAgent(
    service=OpenAIChatCompletion(ai_model_id="%%model=openai-small%%"),
    name="Analyst",
    instructions="Before each tool call, state in one line what you are checking and why.",
    plugins=[Calculator()],
)

async def on_step(message):
    for item in message.items:
        print("step:", type(item).__name__, getattr(item, "name", ""))

async for reply in agent.invoke(messages="If I invest 1000 at 5% for 10 years, what do I end up with?",
                                on_intermediate_message=on_step):
    print(reply.content)
```

```csharp
ChatCompletionAgent agent = new()
{
    Name = "Analyst",
    Instructions = "Before each tool call, state in one line what you are checking and why.",
    Kernel = kernel,   // with the Calculator plugin imported
    Arguments = new KernelArguments(new OpenAIPromptExecutionSettings { FunctionChoiceBehavior = FunctionChoiceBehavior.Auto() }),
};

var options = new AgentInvokeOptions
{
    OnIntermediateMessage = m => { Console.WriteLine($"step: {m.Items.Count} item(s)"); return Task.CompletedTask; },
};
await foreach (var reply in agent.InvokeAsync("If I invest 1000 at 5% for 10 years, what do I end up with?", options: options))
    Console.WriteLine(reply.Message.Content);
```

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-21 | Py 1.45.0 / .NET 1.81.0 | Rewritten. Previous recipes used pre-1.0 Python APIs (`invoke_async`, `create_function_from_prompt`, `model_id=`), the legacy memory-store API (`VolatileMemoryStore`, `ISemanticTextMemory`, `WithMemoryStore`) and placeholder API keys in code. All snippets now use current APIs; versions read from the site's version registry. |

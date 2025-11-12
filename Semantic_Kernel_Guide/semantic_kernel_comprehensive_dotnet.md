---
layout: default
title: "Semantic Kernel Comprehensive Guide (.NET)"
description: "End-to-end .NET (C#) guide for Microsoft Semantic Kernel agents, plugins, memory, planners, and production."
---

# Semantic Kernel Comprehensive Guide (.NET)

Last verified: 2025-11
Source of truth: https://github.com/microsoft/semantic-kernel

## Install

```bash
dotnet new console -n SkAgents
cd SkAgents
dotnet add package Microsoft.SemanticKernel
dotnet add package Microsoft.SemanticKernel.Plugins.Core
dotnet add package Microsoft.SemanticKernel.Connectors.OpenAI
```

## Quick Start: Kernel + Chat

```csharp
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;

var builder = Kernel.CreateBuilder();
builder.AddOpenAIChatCompletion(
    modelId: "gpt-4o-mini",
    apiKey: Environment.GetEnvironmentVariable("OPENAI_API_KEY")!);
var kernel = builder.Build();

var fn = kernel.CreateFunctionFromPrompt("What is Semantic Kernel? Keep it to one sentence.");
var result = await kernel.InvokeAsync<string>(fn);
Console.WriteLine(result);
```

## Agents and Group Chat

```csharp
using Microsoft.SemanticKernel.Agents;

var researcher = new ChatCompletionAgent(kernel)
{
    Name = "researcher",
    Instructions = "Research and cite sources."
};

var writer = new ChatCompletionAgent(kernel)
{
    Name = "writer",
    Instructions = "Draft and refine copy."
};

var chat = new AgentGroupChat(researcher, writer);
await chat.AddChatMessageAsync(new(AuthorRole.User, "Summarize LangGraph with links"));
await foreach (var msg in chat.InvokeAsync())
{
    Console.WriteLine($"{msg.Role}: {msg.Content}");
}
```

## Plugins & Functions

```csharp
var math = kernel.CreatePlugin("math");
math.DefineNativeFunction("add", (int a, int b) => a + b);

var summarize = kernel.CreateFunctionFromPrompt("Summarize {{$input}} in 3 bullets");
var text = await kernel.InvokeAsync<string>(summarize, new() { ["input"] = "LangGraph" });
```

## Structured Outputs
- Use JSON schema prompts and parse/validate with your chosen library.

## Observability & Resilience
- Add logging/metrics and OpenTelemetry spans at function/agent boundaries.
- Timeouts + Polly retries for external calls; guardrails before agent steps.

## Deployment
- ASP.NET Core minimal APIs; configure DI with SK Kernel; store secrets in Key Vault.
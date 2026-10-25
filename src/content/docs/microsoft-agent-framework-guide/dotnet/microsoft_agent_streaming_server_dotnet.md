---
title: "Microsoft Agent Framework (.NET) — Streaming Server"
description: "Minimal ASP.NET Core endpoint that streams agent responses to the browser as server-sent events."
framework: "microsoft-agent-framework"
language: "dotnet"
sidebar:
  label: "Streaming server"
  order: 50
---

Checked against `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% on %%versions-checked%%.

`AIAgent.RunStreamingAsync` returns an `IAsyncEnumerable<AgentResponseUpdate>`; each update carries the text delta in `.Text`. Forward those deltas as server-sent events and you have a streaming chat endpoint.

```bash
dotnet new web -n AgentStreaming
cd AgentStreaming
dotnet add package Microsoft.Agents.AI --version %%version=Microsoft.Agents.AI%%
dotnet add package Microsoft.Agents.AI.OpenAI
dotnet add package Azure.AI.OpenAI
dotnet add package Azure.Identity
```

```csharp
using System.Text.Json;
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using OpenAI.Chat;

var builder = WebApplication.CreateBuilder(args);

// One agent for the whole app; per-conversation state lives in AgentSession.
builder.Services.AddSingleton<AIAgent>(_ =>
    new AzureOpenAIClient(
            new Uri(builder.Configuration["AZURE_OPENAI_ENDPOINT"]!),
            new DefaultAzureCredential())
        .GetChatClient(builder.Configuration["AZURE_OPENAI_DEPLOYMENT"] ?? "%%model=openai-small%%")
        .AsAIAgent(instructions: "You are a concise assistant."));

var app = builder.Build();

app.MapGet("/stream", async (HttpContext http, AIAgent agent, string q) =>
{
    http.Response.Headers.ContentType = "text/event-stream";
    http.Response.Headers.CacheControl = "no-cache";

    AgentSession session = await agent.CreateSessionAsync(http.RequestAborted);

    await foreach (AgentResponseUpdate update in
                   agent.RunStreamingAsync(q, session, cancellationToken: http.RequestAborted))
    {
        if (string.IsNullOrEmpty(update.Text)) continue;   // tool calls, usage, etc.
        var payload = JsonSerializer.Serialize(new { delta = update.Text });
        await http.Response.WriteAsync($"data: {payload}\n\n", http.RequestAborted);
        await http.Response.Body.FlushAsync(http.RequestAborted);
    }

    await http.Response.WriteAsync("event: done\ndata: {}\n\n", http.RequestAborted);
});

app.Run();
```

Notes:

- Pass `http.RequestAborted` through so a closed browser tab cancels the model call.
- To keep a conversation across requests, persist the session with `agent.SerializeSessionAsync(session)` and restore it with `agent.DeserializeSessionAsync(json)` instead of creating a new one per request.
- If you would rather expose an OpenAI-compatible endpoint than a bespoke SSE route, `Microsoft.Agents.AI.Hosting.OpenAI` (alpha) provides `MapOpenAIChatCompletions(...)`.

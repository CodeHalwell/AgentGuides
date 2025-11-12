---
layout: default
title: "Microsoft Agent Streaming Server (.NET)"
description: "Minimal ASP.NET Core streaming endpoint for AI Agent responses."
---

# Microsoft Agent Streaming Server (.NET)

Latest: 1.1.0
Last verified: 2025-11

```csharp
var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.MapGet("/stream", async (HttpResponse resp, string q) =>
{
    resp.Headers.ContentType = "text/event-stream";
    // TODO: integrate Azure AI Agents client here and stream tokens
    for (int i = 0; i < 5; i++)
    {
        await resp.WriteAsync($"data: {{\"delta\":\"{q}-{i}\"}}\n\n");
        await resp.Body.FlushAsync();
        await Task.Delay(200);
    }
});

app.Run();
```


---
title: "Microsoft Agent Framework (.NET) — Streaming Server"
description: "Where to find the .NET streaming examples: your own server-sent-events endpoint with AIHostAgent and a session store, the OpenAI Responses streaming endpoint, AG-UI and A2A streaming."
framework: "microsoft-agent-framework"
language: "dotnet"
sidebar:
  label: "Streaming server"
  order: 19
---

The streaming examples that used to be on this page are now part of the [hosting page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/), checked against `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% (%%versions-checked%%):

- [Your own streaming endpoint](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#your-own-streaming-endpoint): a minimal ASP.NET Core route that forwards `RunStreamingAsync` deltas as server-sent events, and keeps each conversation in a session store with `AIHostAgent` instead of creating a new session per request.
- [OpenAI-compatible endpoints](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#openai-compatible-endpoints): `MapOpenAIResponses` streams Responses API events (`"stream": true`) to any OpenAI client.
- [AG-UI](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#ag-ui): `MapAGUIServer` streams AG-UI events for CopilotKit and other front ends.
- [A2A](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#a2a): streaming between agents over the A2A protocol.

The essentials of a hand-written SSE endpoint, which that section shows in full: set `Content-Type: text/event-stream`, flush after each `data:` line, pass `HttpContext.RequestAborted` into `RunStreamingAsync` so a closed tab cancels the model call, skip updates whose `Text` is empty (tool calls and usage), and save the session after the stream ends. Streaming workflows, as opposed to agents, are on the [workflows page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#streaming-events).

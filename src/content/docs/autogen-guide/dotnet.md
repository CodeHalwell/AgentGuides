---
title: "Microsoft AutoGen for .NET"
description: "What exists of Microsoft AutoGen for .NET — the AutoGen.* 0.2 port and the never-stabilised Microsoft.AutoGen.* previews — and why new .NET projects should use Microsoft Agent Framework."
framework: "autogen"
language: "dotnet"
sidebar:
  label: ".NET (archived)"
---

Microsoft AutoGen did ship .NET packages, but neither line is something to start a project on today. This page records what exists so you can recognise it in an existing codebase, and points to the replacement.

> **Status:** the `microsoft/autogen` repository is archived and no longer maintained (no bug or security fixes). For new .NET work use [Microsoft Agent Framework for .NET](/microsoft-agent-framework-guide/dotnet/) (`Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%%), which Microsoft positions as AutoGen's successor.

## The two package families

The upstream `dotnet/README.md` describes two sets of packages:

| Family | What it is | Last release | State |
|--------|-----------|--------------|-------|
| `AutoGen`, `AutoGen.Core`, `AutoGen.OpenAI`, `AutoGen.Anthropic`, `AutoGen.Gemini`, `AutoGen.Mistral`, `AutoGen.Ollama`, `AutoGen.SemanticKernel`, … | A .NET port of the AutoGen **0.2** conversational API (`AssistantAgent`, `UserProxyAgent`, group chat) | 0.2.3, April 2025 | Upstream said these "will gradually be deprecated"; frozen since |
| `Microsoft.AutoGen.Core`, `Microsoft.AutoGen.AgentChat`, `Microsoft.AutoGen.Contracts`, … | A .NET implementation of the event-driven **0.4** actor model that the Python `autogen-core` uses | 0.4.0-dev.3, March 2025 | Preview only — "not yet stable and subject to change"; never reached a stable release |

These are fixed historical versions, not moving targets, so they are written literally here.

There is no .NET equivalent of Python's `autogen-agentchat` 0.7.x. The .NET and Python lines diverged and the .NET 0.4 work stopped at developer previews.

## Recognising AutoGen.Net 0.2 code

If you inherit a project, it will look like the upstream README's quick start. The README omits `using AutoGen.Core;`, without which `RegisterPrintMessage()` does not compile (CS1061); it is added here:

```csharp
using AutoGen;
using AutoGen.Core;
using AutoGen.OpenAI;

var openAIKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY")
    ?? throw new Exception("Please set OPENAI_API_KEY environment variable.");
var config = new OpenAIConfig(openAIKey, "%%model=openai-small%%");

var assistantAgent = new AssistantAgent(
    name: "assistant",
    systemMessage: "You are an assistant that help user to do some tasks.",
    llmConfig: new ConversableAgentConfig
    {
        Temperature = 0,
        ConfigList = [config],
    })
    .RegisterPrintMessage();
```

`AssistantAgent`, `UserProxyAgent`, `ConversableAgentConfig`, `RegisterPrintMessage()` and `GroupChat` are the tell-tale types. `Microsoft.AutoGen.*` code instead uses agents that handle typed events on a runtime, as in the upstream `samples/Hello`.

## Moving to Microsoft Agent Framework

Microsoft publishes a migration guide: [Migrate from AutoGen](https://learn.microsoft.com/en-us/agent-framework/migration-guide/from-autogen/). In broad terms:

- A single AutoGen assistant becomes an `AIAgent` created from a chat client with `AsAIAgent(...)`, run with `RunAsync` / `RunStreamingAsync`.
- Conversation state moves from the agent's message history to an `AgentSession`.
- Functions registered on an AutoGen agent become `AIFunction` tools from `Microsoft.Extensions.AI`.
- Group chats and multi-agent hand-offs become Agent Framework **workflows** (`Microsoft.Agents.AI.Workflows`), which add checkpointing and human-in-the-loop that AutoGen.Net did not have.

The [.NET Agent Framework guide](/microsoft-agent-framework-guide/dotnet/) covers each of these with verified examples.

## See also

- [Microsoft AutoGen (Python)](/autogen-guide/python/), which covers the line that did reach 0.7.5
- [AutoGen overview](/autogen-guide/): how Microsoft AutoGen and AG2 diverged
- [Microsoft Agent Framework](/microsoft-agent-framework-guide/)

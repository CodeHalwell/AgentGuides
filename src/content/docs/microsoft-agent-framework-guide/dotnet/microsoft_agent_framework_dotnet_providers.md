---
title: "Microsoft Agent Framework (.NET) — Providers and Agent Types"
description: "Every way to get an AIAgent in .NET: ChatClientAgent over OpenAI Chat Completions and Responses, Azure OpenAI, Microsoft Foundry models and Foundry agents, Anthropic, Ollama and other IChatClient providers, GitHub Copilot, A2A and Copilot Studio agents, custom agents, routing and credentials."
framework: "microsoft-agent-framework"
language: "dotnet"
sidebar:
  label: "Providers"
  order: 3
---

This page covers every way of getting an `AIAgent` in .NET: the `ChatClientAgent` that wraps any `IChatClient`, the provider packages that build one for you (OpenAI, Azure OpenAI, Microsoft Foundry, Anthropic), the agent types that are not chat clients at all (Foundry-defined agents, GitHub Copilot, A2A, Copilot Studio), writing your own agent, and routing one agent across several models. Tools, MCP, middleware and sessions each have their own page; this one stops at the point where you hold an agent.

Every C# block was compiled against `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% and the companion packages listed below (checked %%versions-checked%%). The OpenAI and Anthropic examples were also run against a local stand-in for those APIs; the Azure and Foundry examples need a real subscription and were compiled only.

## Three ways to an agent

| Route | What you write | What you get |
| --- | --- | --- |
| Any `IChatClient` | `new ChatClientAgent(chatClient, ...)`, `chatClient.AsAIAgent(...)` or `chatClientBuilder.BuildAIAgent(...)` | `ChatClientAgent` |
| A provider SDK client | `AsAIAgent(...)` on OpenAI's `ChatClient` or `ResponsesClient`, on `AnthropicClient`, or on Foundry's `AIProjectClient` | `ChatClientAgent` (the extension builds the `IChatClient` for you) |
| An agent that is not a chat client | `FoundryAgent`, `GitHubCopilotAgent`, `A2AAgent`, `CopilotStudioAgent`, or your own subclass of `AIAgent` | Some other `AIAgent` |

All three end at `AIAgent`, so the rest of the framework (sessions, middleware, workflows, hosting, tools that wrap agents) does not care which route you took. Code that *uses* an agent should take an `AIAgent`; only the code that *builds* it needs to know the provider.

## Packages

| Package | Status | Provides |
| --- | --- | --- |
| `Microsoft.Agents.AI.Abstractions` | stable | `AIAgent`, `AgentSession`, `AgentResponse`, `ChatHistoryProvider`, `AIContextProvider`, `AgentSessionStore` |
| `Microsoft.Agents.AI` | stable | `ChatClientAgent`, `AIAgentBuilder`, built-in context providers, compaction, routing, `AsAIAgent()` on `IChatClient` |
| `Microsoft.Agents.AI.OpenAI` | stable | `AsAIAgent()` on OpenAI's `ChatClient` and `ResponsesClient`, OpenAI-typed `RunAsync` overloads |
| `Microsoft.Agents.AI.Anthropic` | stable | `AsAIAgent()` on `AnthropicClient` and its beta service |
| `Microsoft.Agents.AI.GitHub.Copilot` | stable | `GitHubCopilotAgent` |
| `Microsoft.Agents.AI.Foundry` | preview | `AsAIAgent()` on `AIProjectClient`, `FoundryAgent`, `FoundryAITool`, `FoundryMemoryProvider`, `FoundryEvals` |
| `Microsoft.Agents.AI.A2A` | preview | `A2AAgent` (calling remote A2A agents) |
| `Microsoft.Agents.AI.CopilotStudio` | preview | `CopilotStudioAgent` |
| `Microsoft.Agents.AI.AzureAI.Persistent` | preview, every member obsolete | The classic Azure AI Agents service |

The preview packages need `--prerelease` (or an explicit preview version) on `dotnet add package`. Models that have no `Microsoft.Agents.AI.*` package (Ollama, Gemini, ONNX, Bedrock and so on) come in through any library that implements `Microsoft.Extensions.AI.IChatClient`.

```bash
dotnet add package Microsoft.Agents.AI --version %%version=Microsoft.Agents.AI%%
dotnet add package Microsoft.Agents.AI.OpenAI --version %%version=Microsoft.Agents.AI%%
dotnet add package Microsoft.Agents.AI.Anthropic --version %%version=Microsoft.Agents.AI%%
dotnet add package Microsoft.Agents.AI.Foundry --prerelease
dotnet add package Azure.Identity
```

## `ChatClientAgent`

`ChatClientAgent` is the agent you will use most. It takes an `IChatClient` and adds instructions, tools, the function-calling loop, chat history, context providers and session handling. It has two constructors:

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

IChatClient chatClient = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient();

// 1. The short form: instructions, name, description and tools.
ChatClientAgent simple = new(
    chatClient,
    instructions: "You are a concise assistant.",
    name: "helper",
    description: "Answers general questions.");

// 2. The options form: everything, including history and context providers.
ChatClientAgent configured = new(chatClient, new ChatClientAgentOptions
{
    Id = "helper-v2",
    Name = "helper",
    ChatOptions = new ChatOptions
    {
        Instructions = "You are a concise assistant.",
        Temperature = 0.2f,
        MaxOutputTokens = 400,
    },
});

Console.WriteLine((await simple.RunAsync("Say hello in French.")).Text);
Console.WriteLine((await configured.RunAsync("Say hello in German.")).Text);
Console.WriteLine($"{configured.Id} / {configured.Name} / {configured.Instructions}");
```

`chatClient.AsAIAgent(...)` takes the same arguments and calls the same constructors. `ChatClientAgentOptions` has no `Instructions` property of its own: instructions are `ChatOptions.Instructions`, and the agent's `Instructions` property reads them back from there. The options object is cloned when the agent is built, so changing it afterwards has no effect on the agent.

### `ChatClientAgentOptions`

| Property | Default | Meaning |
| --- | --- | --- |
| `Id` | generated | Agent ID; surfaces as `agent.Id` and in telemetry. |
| `Name`, `Description` | `null` | Used for telemetry, `AuthorName` on response messages, agent-as-tool names and A2A cards. |
| `ChatOptions` | `null` | Default request options: `Instructions`, `Tools`, `ModelId`, `Temperature`, `MaxOutputTokens`, `ResponseFormat`, `ToolMode`, `Reasoning` and the rest of `Microsoft.Extensions.AI.ChatOptions`. |
| `ChatHistoryProvider` | `InMemoryChatHistoryProvider` | Where history is kept when the service does not keep it. See [sessions and memory](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_sessions_and_memory/). |
| `AIContextProviders` | none | Providers that add instructions, messages or tools on each run. |
| `UseProvidedChatClientAsIs` | `false` | Skip the decorators the agent normally adds (below). |
| `AllowConcurrentInvocation` | `false` | Let the function loop run several tool calls from one model response in parallel. |
| `ClearOnChatHistoryProviderConflict`, `WarnOnChatHistoryProviderConflict`, `ThrowOnChatHistoryProviderConflict` | `true`, `true`, `true` | What happens when you configured a history provider but the service returns a conversation ID. |
| `RequirePerServiceCallChatHistoryPersistence` | `false` | Save history after every model call inside the tool loop rather than at the end of the run. |
| `EnableMessageInjection` | `false` | Add a `MessageInjectingChatClient` so tools can queue messages into the running loop. |
| `DisableApprovalNotRequiredFunctionBypassing` | `false` | Surface every call in a response that contains an approval-required tool, not only the ones that need approval. |
| `DisableApprovalResponseBinding` | `false` | Stop binding approval responses to the requests the framework issued. Leave it on. |
| `EnableInvocableFunctionBypassing` | `false` (experimental) | Hold back server-side calls returned alongside client-side (declaration-only) calls and run them on the next request. |

The conflict, persistence and approval flags are explained on the pages that need them: [sessions](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_sessions_and_memory/#when-the-service-keeps-the-history) and [tools](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_tools/#approvals).

### What the agent wraps around your client

Unless `UseProvidedChatClientAsIs` is set, the constructor decorates your client. From the outside in, the pipeline is:

1. `ApprovalResponseBindingChatClient` (unless disabled), which ties each approval response to the request the framework issued.
2. `ApprovalNotRequiredFunctionBypassingChatClient` (unless disabled), which answers approval requests for tools that never needed approval.
3. `InvocableFunctionBypassingChatClient` (only with `EnableInvocableFunctionBypassing`).
4. `FunctionInvokingChatClient`, the tool loop, added only if your client does not already contain one.
5. `MessageInjectingChatClient` (only with `EnableMessageInjection`).
6. `PerServiceCallChatHistoryPersistingChatClient` (only with `RequirePerServiceCallChatHistoryPersistence`).
7. A dormant OpenTelemetry client that `UseOpenTelemetry()` on the agent switches on.
8. Your `IChatClient`.

The practical consequences: you never need to call `UseFunctionInvocation()` yourself; if you do, the agent finds your `FunctionInvokingChatClient` and reuses it (so its settings, such as `MaximumIterationsPerRequest`, apply); and `agent.ChatClient` returns the decorated pipeline, not the client you passed in. With `UseProvidedChatClientAsIs = true` you get none of this, including the tool loop, so tools stop being called unless your own pipeline has a `FunctionInvokingChatClient`. The [middleware page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_middleware/#chat-client-middleware) shows how to add your own layers.

### Asking an agent what it runs on

`GetService` walks the agent, its context providers, its history provider and its chat-client pipeline:

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Chat;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsAIAgent(instructions: "You are helpful.", name: "helper");

AIAgentMetadata? metadata = agent.GetService<AIAgentMetadata>();
ChatClientMetadata? model = agent.GetService<ChatClientMetadata>();
FunctionInvokingChatClient? loop = agent.GetService<FunctionInvokingChatClient>();

Console.WriteLine($"provider: {metadata?.ProviderName}");          // openai
Console.WriteLine($"default model: {model?.DefaultModelId}");       // the model you passed
Console.WriteLine($"tool loop: {loop?.MaximumIterationsPerRequest}"); // 40
```

`AIAgentMetadata.ProviderName` comes from the inner client's `ChatClientMetadata`, captured when the agent is built.

## OpenAI

`Microsoft.Agents.AI.OpenAI` adds `AsAIAgent` to the two OpenAI SDK clients. Both return a `ChatClientAgent`; the difference is the API underneath.

### Chat Completions

```csharp
using Microsoft.Agents.AI;
using OpenAI;
using OpenAI.Chat;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsAIAgent(instructions: "You are good at telling jokes.", name: "Joker");

Console.WriteLine(await agent.RunAsync("Tell me a joke about a pirate."));

await foreach (AgentResponseUpdate update in agent.RunStreamingAsync("And one about a parrot."))
{
    Console.Write(update.Text);
}
Console.WriteLine();
```

Chat Completions is stateless: the service keeps nothing, so the agent keeps the history itself, in the session, through its `InMemoryChatHistoryProvider`. `AgentResponse.ToString()` returns `Text`, which is why `Console.WriteLine(await agent.RunAsync(...))` prints the answer.

### Responses

The Responses API extension is marked `[Experimental("OPENAI001")]`, so you must suppress that diagnostic. `ResponsesClient.AsAIAgent` takes the model as a parameter because one `ResponsesClient` serves every model.

```csharp
#pragma warning disable OPENAI001
using Microsoft.Agents.AI;
using OpenAI;
using OpenAI.Responses;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetResponsesClient()
    .AsAIAgent(model: "%%model=openai%%", instructions: "You are a helpful assistant.", name: "Assistant");

AgentSession session = await agent.CreateSessionAsync();
Console.WriteLine(await agent.RunAsync("My name is Ana.", session));
Console.WriteLine(await agent.RunAsync("What is my name?", session));

// The service stored the response; the session holds its ID, not the messages.
var typed = (ChatClientAgentSession)session;
Console.WriteLine($"conversation id: {typed.ConversationId}");
```

By default the Responses API stores each response on the server and returns its ID. The agent puts that ID in `ChatClientAgentSession.ConversationId` and sends it as `previous_response_id` on the next turn, so history lives on OpenAI's side and the local history provider is not used. Two consequences: a serialised session is small (just the ID), and the history is lost if the stored response expires or is deleted.

To keep history on your side instead (for zero-data-retention, or to use your own store), build the chat client with stored output disabled. `AsIChatClientWithStoredOutputDisabled` is itself experimental (`MAAI001`):

```csharp
#pragma warning disable OPENAI001, MAAI001
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Responses;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetResponsesClient()
    .AsIChatClientWithStoredOutputDisabled(model: "%%model=openai%%")   // store=false, keep encrypted reasoning
    .AsAIAgent(instructions: "You are a helpful assistant.");

AgentSession session = await agent.CreateSessionAsync();
await agent.RunAsync("My name is Ana.", session);
AgentResponse second = await agent.RunAsync("What is my name?", session);

Console.WriteLine(second.Text);
Console.WriteLine(session.TryGetInMemoryChatHistory(out var history) ? $"{history.Count} messages held locally" : "no local history");
```

With `includeReasoningEncryptedContent: true` (the default) it also asks for the encrypted reasoning items, so reasoning models keep their chain of thought between turns without the server storing it.

### Conversations

The OpenAI Conversations API gives you a durable conversation object you create first and then pass to every response. Create the conversation with the SDK's `ConversationClient`, then start the session from its ID with `ChatClientAgent.CreateSessionAsync(conversationId)`:

```csharp
#pragma warning disable OPENAI001
using System.ClientModel;
using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Conversations;

OpenAIClient openAI = new(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!);
ConversationClient conversations = openAI.GetConversationClient();

ChatClientAgent agent = new(
    openAI.GetResponsesClient().AsIChatClient("%%model=openai%%"),
    instructions: "You are a helpful assistant.");

ClientResult created = await conversations.CreateConversationAsync(BinaryContent.Create(BinaryData.FromString("{}")));
using JsonDocument json = JsonDocument.Parse(created.GetRawResponse().Content.ToString());
string conversationId = json.RootElement.GetProperty("id").GetString()!;

AgentSession session = await agent.CreateSessionAsync(conversationId);
Console.WriteLine(await agent.RunAsync("What is the capital of France?", session));
Console.WriteLine(await agent.RunAsync("What is it famous for?", session));

await conversations.DeleteConversationAsync(conversationId);
```

`CreateSessionAsync(string)` is on `ChatClientAgent`, not on `AIAgent`; if you hold the agent as `AIAgent`, cast it. Passing a conversation ID to an agent whose service does not return one fails on the first run with `InvalidOperationException` ("Service did not return a valid conversation id").

### Background responses

Long-running Responses calls (deep research, large generations) can run in the background. Continuation tokens are experimental in `Microsoft.Extensions.AI` (`MEAI001`). Set `AgentRunOptions.AllowBackgroundResponses = true`; the first call returns quickly with a `ContinuationToken`, and you poll by running again with that token and no new messages. A session is required.

```csharp
#pragma warning disable OPENAI001, MEAI001
using Microsoft.Agents.AI;
using OpenAI;
using OpenAI.Responses;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetResponsesClient()
    .AsAIAgent(model: "%%model=openai%%", instructions: "You are a helpful assistant.");

AgentRunOptions options = new() { AllowBackgroundResponses = true };
AgentSession session = await agent.CreateSessionAsync();

AgentResponse response = await agent.RunAsync("Write a long essay about otters.", session, options);
while (response.ContinuationToken is { } token)
{
    await Task.Delay(TimeSpan.FromSeconds(2));
    options.ContinuationToken = token;
    response = await agent.RunAsync(session, options);   // no messages when continuing
}

Console.WriteLine(response.Text);
```

The rules are enforced: continuing without a session throws, and passing messages together with a continuation token throws. With streaming, each `AgentResponseUpdate` carries a `ContinuationToken`; keep the last one and you can resume the stream from that point after a disconnect by calling `RunStreamingAsync(session, options)` with it. A service that does not support background responses simply completes the run and returns no token. The [hosting page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#session-stores-and-isolation) covers keeping the session between the calls.

### Reasoning models

Reasoning settings are provider-neutral in `Microsoft.Extensions.AI`: set `ChatOptions.Reasoning`. Reasoning text comes back as `TextReasoningContent` in the response messages and streaming updates, and reasoning token counts appear in `Usage.AdditionalCounts`.

```csharp
#pragma warning disable OPENAI001
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetResponsesClient()
    .AsIChatClient("%%model=openai%%")
    .AsAIAgent(new ChatClientAgentOptions
    {
        ChatOptions = new ChatOptions
        {
            Reasoning = new ReasoningOptions { Effort = ReasoningEffort.Medium, Output = ReasoningOutput.Full },
        },
    });

await foreach (AgentResponseUpdate update in agent.RunStreamingAsync("How long does a 180-mile trip take at 60 mph?"))
{
    foreach (AIContent content in update.Contents)
    {
        Console.Write(content switch
        {
            TextReasoningContent thinking => $"[thinking] {thinking.Text}",
            TextContent text => text.Text,
            _ => "",
        });
    }
}
Console.WriteLine();
```

### OpenAI-compatible servers

Anything that speaks the OpenAI wire protocol (vLLM, LM Studio, llama.cpp's server, Foundry Local, LiteLLM, a corporate gateway) works through `OpenAIClientOptions.Endpoint`. Note the `/v1` segment: the SDK appends `chat/completions` or `responses` to the endpoint you give it.

```csharp
using System.ClientModel;
using Microsoft.Agents.AI;
using OpenAI;
using OpenAI.Chat;

var options = new OpenAIClientOptions
{
    Endpoint = new Uri(Environment.GetEnvironmentVariable("OPENAI_BASE_URL") ?? "http://localhost:8000/v1"),
    NetworkTimeout = TimeSpan.FromMinutes(2),
};

AIAgent agent = new OpenAIClient(new ApiKeyCredential(Environment.GetEnvironmentVariable("LOCAL_API_KEY") ?? "not-needed"), options)
    .GetChatClient("%%model=openai-small%%")   // whatever model name the server expects
    .AsAIAgent(instructions: "You are a helpful assistant.");

Console.WriteLine(await agent.RunAsync("Reply with one word: ready?"));
```

The examples on this page were run this way, against a local stand-in.

### OpenAI types in and out

If the rest of your code already deals in OpenAI SDK types, the package has `RunAsync` overloads that take OpenAI messages and return OpenAI results, plus `AsOpenAIChatCompletion()` and `AsOpenAIResponse()` on `AgentResponse`. They are experimental (`OPENAI001`).

```csharp
#pragma warning disable OPENAI001
using Microsoft.Agents.AI;
using OpenAI;
using OpenAI.Chat;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsAIAgent(instructions: "You are helpful.");

ChatCompletion completion = await agent.RunAsync([new UserChatMessage("Name a prime number.")]);
Console.WriteLine(completion.Content[0].Text);

AgentResponse response = await agent.RunAsync("Name another one.");
ChatCompletion converted = response.AsOpenAIChatCompletion();
Console.WriteLine(converted.Content[0].Text);
```

The overloads are extension methods on `AIAgent` and work with any agent: the messages are converted to `Microsoft.Extensions.AI` types on the way in and the `AgentResponse` is converted on the way out.

## Azure OpenAI

Azure OpenAI's v1 API is OpenAI-compatible, so the OpenAI SDK talks to it directly. Point `OpenAIClientOptions.Endpoint` at `https://<resource>.openai.azure.com/openai/v1/` and authenticate with Entra ID through a `BearerTokenPolicy`, or with a key through `ApiKeyCredential`. The `OpenAIClient` constructor that takes an `AuthenticationPolicy` is experimental (`OPENAI001`). The model argument is your **deployment name**. This is how Microsoft's own samples now connect.

```csharp
#pragma warning disable OPENAI001   // the OpenAIClient(AuthenticationPolicy, ...) constructor is experimental
using System.ClientModel;
using System.ClientModel.Primitives;
using Azure.Identity;
using Microsoft.Agents.AI;
using OpenAI;
using OpenAI.Chat;

string resource = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")!;   // https://<resource>.openai.azure.com
string deployment = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT") ?? "%%model=openai%%";
var options = new OpenAIClientOptions { Endpoint = new Uri($"{resource.TrimEnd('/')}/openai/v1/") };

string? key = Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY");
OpenAIClient client = key is null
    ? new OpenAIClient(new BearerTokenPolicy(new DefaultAzureCredential(), "https://ai.azure.com/.default"), options)
    : new OpenAIClient(new ApiKeyCredential(key), options);

AIAgent agent = client.GetChatClient(deployment).AsAIAgent(instructions: "You are helpful.", name: "assistant");
Console.WriteLine(await agent.RunAsync("Hello from Azure."));
```

The older `Azure.AI.OpenAI` package (`AzureOpenAIClient`) still works and gives the same OpenAI `ChatClient`, so code written against it needs no change:

```csharp
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using OpenAI.Chat;

AIAgent agent = new AzureOpenAIClient(
        new Uri(Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")!),
        new DefaultAzureCredential())
    .GetChatClient(Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT") ?? "%%model=openai%%")
    .AsAIAgent(instructions: "You are helpful.");

Console.WriteLine(await agent.RunAsync("Hello from Azure."));
```

`Azure.AI.OpenAI` is in beta (2.9.0-beta.1 at the time of writing) and is no longer used by the Agent Framework repository; prefer the v1 endpoint with the plain OpenAI client for new code. The Responses API works the same way: `client.GetResponsesClient().AsAIAgent(model: deployment, ...)`.

## Microsoft Foundry

`Microsoft.Agents.AI.Foundry` (preview) builds on `Azure.AI.Projects` (`AIProjectClient`). There are two quite different things you can make from a project:

| You call | You get | Agent definition lives | Hosted tools |
| --- | --- | --- | --- |
| `project.AsAIAgent(model, instructions, ...)` | `ChatClientAgent` over the project's Responses endpoint | in your code | any `AITool` you pass |
| `project.AsAIAgent(agentVersion)` (or a record, `AgentReference` or agent endpoint `Uri`) | `FoundryAgent` | in Foundry, versioned | those configured on the agent, plus local function tools |

Pin the Azure packages: `Azure.AI.Projects` 3.0.0-beta.3 declares a dependency on an unpublished alpha of `Azure.AI.Extensions.OpenAI`, so add `Azure.AI.Extensions.OpenAI` 3.0.0-beta.3 explicitly or NuGet falls back to an older beta (the Agent Framework repository does the same).

### A model in the project, an agent in code

```csharp
using Azure.AI.Projects;
using Azure.Identity;
using Microsoft.Agents.AI;

var project = new AIProjectClient(
    new Uri(Environment.GetEnvironmentVariable("FOUNDRY_PROJECT_ENDPOINT")!),   // https://<account>.services.ai.azure.com/api/projects/<project>
    new DefaultAzureCredential());

ChatClientAgent writer = project.AsAIAgent(
    model: Environment.GetEnvironmentVariable("FOUNDRY_MODEL") ?? "%%model=openai%%",
    instructions: "You are a helpful writing assistant.",
    name: "writer");

AgentSession session = await writer.CreateSessionAsync();
Console.WriteLine(await writer.RunAsync("Draft a two-line summary of the Agent Framework.", session));
Console.WriteLine(await writer.RunAsync("Now make it more formal.", session));
```

`instructions` is required on this overload (it throws `ArgumentException` on an empty string); use the `AsAIAgent(ChatClientAgentOptions)` overload for an agent without instructions. The agent uses the Responses API with stored output, so, as with OpenAI, the session holds a response ID. For history that shows up in the Foundry portal, create a project conversation and start the session from it:

```csharp
#pragma warning disable OPENAI001   // project conversations are experimental
using Azure.AI.Extensions.OpenAI;
using Azure.AI.Projects;
using Azure.Identity;
using Microsoft.Agents.AI;
using OpenAI.Conversations;

var project = new AIProjectClient(new Uri(Environment.GetEnvironmentVariable("FOUNDRY_PROJECT_ENDPOINT")!), new DefaultAzureCredential());
ChatClientAgent agent = project.AsAIAgent("%%model=openai%%", instructions: "You are good at telling jokes.", name: "Joker");

ProjectConversationsClient conversations = project.GetProjectOpenAIClient().GetProjectConversationsClient();
ConversationResource conversation = (await conversations.CreateProjectConversationAsync()).Value;

AgentSession session = await agent.CreateSessionAsync(conversation.Id);
Console.WriteLine(await agent.RunAsync("Tell me a joke about a pirate.", session));
Console.WriteLine(await agent.RunAsync("Now tell it as the pirate's parrot.", session));
```

### Agents defined in Foundry

A Foundry agent is a named, versioned definition (model, instructions, hosted tools) stored in the project. Create or fetch it with the project's `AgentAdministrationClient`, then wrap it:

```csharp
using Azure.AI.Projects;
using Azure.AI.Projects.Agents;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Foundry;

var project = new AIProjectClient(new Uri(Environment.GetEnvironmentVariable("FOUNDRY_PROJECT_ENDPOINT")!), new DefaultAzureCredential());
string model = Environment.GetEnvironmentVariable("FOUNDRY_MODEL") ?? "%%model=openai%%";

// Create version 1 of an agent definition.
ProjectsAgentVersion v1 = await project.AgentAdministrationClient.CreateAgentVersionAsync(
    "JokerAgent",
    new ProjectsAgentVersionCreationOptions(new DeclarativeAgentDefinition(model: model)
    {
        Instructions = "You are good at telling jokes.",
    }));

FoundryAgent joker = project.AsAIAgent(v1);
Console.WriteLine(await joker.RunAsync("Tell me a joke about a pirate."));

// Later: fetch the agent by name and use whatever version is latest.
ProjectsAgentRecord record = await project.AgentAdministrationClient.GetAgentAsync("JokerAgent");
FoundryAgent latest = project.AsAIAgent(record);
Console.WriteLine($"latest version: {record.GetLatestVersion().Id}");

AgentSession session = await latest.CreateConversationSessionAsync();   // visible in the Foundry portal
Console.WriteLine(await latest.RunAsync("Another, about a cat.", session));

await project.AgentAdministrationClient.DeleteAgentAsync("JokerAgent");
```

`FoundryAgent` is a `DelegatingAIAgent` over a `ChatClientAgent`, so everything that works on agents works on it. It adds `CreateConversationSessionAsync()` (a server-side conversation that appears in the portal), `CreateSessionAsync(conversationId)`, and `CreateFoundryHostedAgentSessionAsync(hostedSessionId, conversationId, userIdentity)` for Foundry hosted agents, where the session also pins a sandbox and a delegated user identity. `FoundryAgentExtensions` adds file and vector-store helpers (`UploadFileAsync`, `CreateVectorStoreAsync` and their deletes) that use the agent's own project connection.

Two constructors let you skip `AIProjectClient`: `new FoundryAgent(projectEndpoint, credential, model, instructions, ...)` (the code-defined kind) and `new FoundryAgent(agentEndpoint, credential, ...)` for an agent's own endpoint (`.../agents/<name>/endpoint/protocols/openai`); the second fills in only the ID and name, not the instructions. Going the other way, `ToPromptAgentAsync()` on a `ChatClientAgent` or `FoundryAgent` produces a `ProjectsAgentDefinition` you can save to Foundry.

Hosted tools for Foundry agents (Bing grounding, SharePoint, Fabric, Azure AI Search, OpenAPI, code interpreter, file search, browser automation, MCP, toolboxes) are on the [tools page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_tools/#foundry-tools).

### Other Foundry models

Models deployed in Foundry that are not OpenAI models (Phi, Llama, DeepSeek, Mistral and so on) are served through an OpenAI-compatible endpoint; use the OpenAI client with that endpoint, as in the Azure OpenAI example, with scope `https://ai.azure.com/.default`. Claude models deployed in Foundry are covered under [Anthropic](#anthropic).

### Classic Persistent Agents

`Microsoft.Agents.AI.AzureAI.Persistent` wraps the classic Azure AI Agents service (`PersistentAgentsClient`). In %%version=Microsoft.Agents.AI%% every extension in it (`CreateAIAgentAsync`, `GetAIAgentAsync`, `AsAIAgent`) is marked `[Obsolete]`; Microsoft lists the classic service for retirement on 31 March 2027. Use Foundry agents for anything new; existing code compiles with obsolete warnings.

## Anthropic

`Microsoft.Agents.AI.Anthropic` adds `AsAIAgent` to the official Anthropic SDK (`Anthropic` package, `IAnthropicClient`). Unlike the OpenAI extensions it takes the model as a required argument, and a `defaultMaxTokens` that falls back to the static `AnthropicClientExtensions.DefaultMaxTokens` (4096), because the Messages API requires `max_tokens` on every request. `ChatOptions.MaxOutputTokens` overrides it per agent or per run.

```csharp
using Anthropic;
using Anthropic.Core;
using Microsoft.Agents.AI;

AnthropicClient anthropic = new(new ClientOptions
{
    ApiKey = Environment.GetEnvironmentVariable("ANTHROPIC_API_KEY"),
});

AIAgent claude = anthropic.AsAIAgent(
    model: "%%model=anthropic%%",
    instructions: "You are a concise technical writer.",
    name: "writer",
    defaultMaxTokens: 2048);

AgentSession session = await claude.CreateSessionAsync();
Console.WriteLine(await claude.RunAsync("Explain idempotency in two sentences.", session));
Console.WriteLine(await claude.RunAsync("Now give an HTTP example.", session));
```

The Anthropic SDK reads `ANTHROPIC_API_KEY` and `ANTHROPIC_BASE_URL` from the environment when you do not set them, so `new AnthropicClient()` is enough in many deployments. The Messages API keeps no state, so history is held locally in the session, as with Chat Completions.

### Extended thinking and other Anthropic-only settings

Settings that have no `ChatOptions` equivalent go through `RawRepresentationFactory`, which lets you supply the SDK's own request object; the adapter fills in the messages and tools. Use `clientFactory` to add it once for the agent:

```csharp
using Anthropic;
using Anthropic.Models.Messages;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

const string Model = "%%model=anthropic%%";

AIAgent thinker = new AnthropicClient().AsAIAgent(
    model: Model,
    clientFactory: inner => inner.AsBuilder()
        .ConfigureOptions(options => options.RawRepresentationFactory = _ => new MessageCreateParams
        {
            Model = options.ModelId ?? Model,
            MaxTokens = options.MaxOutputTokens ?? 4096,
            Messages = [],
            Thinking = new ThinkingConfigParam(new ThinkingConfigEnabled(budgetTokens: 2048)),
        })
        .Build());

AgentResponse response = await thinker.RunAsync("Is 1001 prime? Think it through.");
foreach (TextReasoningContent thought in response.Messages.SelectMany(m => m.Contents).OfType<TextReasoningContent>())
{
    Console.WriteLine($"[thinking] {thought.Text}");
}
Console.WriteLine(response.Text);
```

### The beta service

`anthropic.Beta.AsAIAgent(...)` (in namespace `Anthropic.Services`, experimental `MAAIANTHROPIC001`) builds the agent on the beta Messages endpoint, which is where Anthropic-managed skills, the code-execution tool and other beta features live. It has the same parameters as the stable extension.

### Claude in Microsoft Foundry

`Anthropic.Foundry` provides `AnthropicFoundryClient`, an `IAnthropicClient` for Claude deployments in a Foundry resource, so the same `AsAIAgent` applies:

```csharp
using Anthropic;
using Anthropic.Foundry;
using Azure.Identity;
using Microsoft.Agents.AI;

string resource = Environment.GetEnvironmentVariable("ANTHROPIC_RESOURCE")!;   // the Foundry resource name

using var client = new AnthropicFoundryClient(
    new AnthropicFoundryIdentityTokenCredentials(new DefaultAzureCredential(), resource, ["https://ai.azure.com/.default"]));

AIAgent agent = client.AsAIAgent(model: "%%model=anthropic%%", instructions: "You are helpful.", name: "claude");
Console.WriteLine(await agent.RunAsync("Hello from Foundry."));
```

With a key, use `new AnthropicFoundryApiKeyCredentials(apiKey, resource)` instead.

## Any other `IChatClient`

Every library that implements `Microsoft.Extensions.AI.IChatClient` is a provider. The agent does not need a dedicated package; `AsAIAgent()` on the client is enough.

| Model source | Package | Client |
| --- | --- | --- |
| Ollama | `OllamaSharp` | `new OllamaApiClient(endpoint, model)` (it is an `IChatClient`) |
| Google Gemini | `Google.GenAI` | `new Client(apiKey: ...).AsIChatClient(model)` |
| Google Gemini (community) | `Mscc.GenerativeAI.Microsoft` | `new GeminiChatClient(apiKey, model)` |
| Local ONNX models | `Microsoft.ML.OnnxRuntimeGenAI` | `new OnnxRuntimeGenAIChatClient(modelPath)` |
| Dapr Conversation API | `Dapr.AI.Microsoft.Extensions` | `services.AddDaprChatClient(...)`, then resolve `IChatClient` |
| Amazon Bedrock | `AWSSDK.Extensions.Bedrock.MEAI` | `bedrockRuntime.AsIChatClient(modelId)` |
| Any OpenAI-compatible server | `OpenAI` | [as above](#openai-compatible-servers) |

Ollama, which is the one most people try first:

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OllamaSharp;

IChatClient ollama = new OllamaApiClient(
    new Uri(Environment.GetEnvironmentVariable("OLLAMA_ENDPOINT") ?? "http://localhost:11434"),
    Environment.GetEnvironmentVariable("OLLAMA_MODEL") ?? "llama3.2");

AIAgent agent = ollama.AsAIAgent(instructions: "You are good at telling jokes.", name: "Joker");
Console.WriteLine(await agent.RunAsync("Tell me a joke about a pirate."));
```

Whether tools, structured output and streaming work depends on the client and the model, not on the agent: the agent sends `ChatOptions.Tools` and `ResponseFormat`, and a client that ignores them gives you plain text back. Small local models often call tools unreliably; test with the model you intend to ship.

## Agents that are not chat clients

### GitHub Copilot

`Microsoft.Agents.AI.GitHub.Copilot` exposes the GitHub Copilot SDK (`GitHub.Copilot.SDK`) as `GitHubCopilotAgent`. The Copilot CLI does the planning, file editing and shell work, so this suits repository tasks. It needs the Copilot CLI installed and signed in (or a bring-your-own-key provider). Permission requests from the CLI arrive through `SessionConfig.OnPermissionRequest`.

```csharp
#pragma warning disable GHCP001   // PermissionDecision is experimental in the Copilot SDK
using GitHub.Copilot;
using GitHub.Copilot.Rpc;
using Microsoft.Agents.AI;

await using CopilotClient copilot = new();
await copilot.StartAsync();

SessionConfig config = new()
{
    OnPermissionRequest = (request, invocation) =>
    {
        Console.Write($"Allow {request.Kind}? [y/N] ");
        bool yes = Console.ReadLine()?.Trim().Equals("y", StringComparison.OrdinalIgnoreCase) == true;
        return Task.FromResult(yes ? PermissionDecision.ApproveOnce() : PermissionDecision.Reject());
    },
};

AIAgent agent = copilot.AsAIAgent(config, ownsClient: true, name: "repo-helper");

await foreach (AgentResponseUpdate update in agent.RunStreamingAsync("Which projects in this solution have no tests?"))
{
    Console.Write(update);
}
Console.WriteLine();
```

The other overload takes `instructions:` and `tools:` (as `AIFunctionDeclaration`s) instead of a `SessionConfig`. To use your own model provider, set `SessionConfig.Model` and `SessionConfig.Provider = new ProviderConfig { Type = "openai", BaseUrl = ..., ApiKey = ..., ModelId = ... }`. A `GitHubCopilotAgentSession` carries the Copilot `SessionId`; `CreateSessionAsync(sessionId)` resumes an existing one.

### Remote agents over A2A

`Microsoft.Agents.AI.A2A` (preview) turns a remote agent that speaks the Agent2Agent protocol into a local `AIAgent`. Resolve its agent card and ask for the agent:

```csharp
using A2A;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.A2A;

A2ACardResolver resolver = new(new Uri(Environment.GetEnvironmentVariable("A2A_AGENT_HOST") ?? "http://localhost:5000"));
AIAgent remote = await resolver.GetAIAgentAsync();

AgentSession session = await remote.CreateSessionAsync();
Console.WriteLine(await remote.RunAsync("Tell me a joke about a pirate.", session));

if (session is A2AAgentSession a2a)
{
    Console.WriteLine($"context {a2a.ContextId}, task {a2a.TaskId}, state {a2a.TaskState}");
}
```

The session is an `A2AAgentSession` holding the A2A context ID and the current task ID and state. You can also build the agent from an `AgentCard` (`card.AsAIAgent()`) or an `IA2AClient` (`client.AsAIAgent()`), or construct `A2AAgent` directly. Exposing your own agents over A2A is on the [feature tour](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_2025_features/#agent2agent-a2a-protocol---net).

### Copilot Studio

`Microsoft.Agents.AI.CopilotStudio` (preview) provides `CopilotStudioAgent`, which wraps a `CopilotClient` from `Microsoft.Agents.CopilotStudio.Client`. Configure that client as the Copilot Studio SDK documents (environment ID, agent identifier and a token provider for the signed-in user), then `new CopilotStudioAgent(client)`. Sessions are `CopilotStudioAgentSession`s carrying the Copilot Studio conversation ID; `CreateSessionAsync(conversationId)` resumes one. This example was not compiled for this page because the client needs tenant-specific settings to construct.

## Writing your own agent

Subclass `AIAgent` when the thing you want to call is not a chat model: a rules engine, a remote service with its own protocol, a deterministic fallback. Five members are abstract: `CreateSessionCoreAsync`, `SerializeSessionCoreAsync`, `DeserializeSessionCoreAsync`, `RunCoreAsync` and `RunCoreStreamingAsync`. The public `RunAsync` and `RunStreamingAsync` overloads (string, message, list, typed) all funnel into the two `Run*CoreAsync` methods, and the base class sets `AIAgent.CurrentRunContext` around them.

```csharp
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

AIAgent agent = new ShoutingAgent();
AgentSession session = await agent.CreateSessionAsync();

Console.WriteLine(await agent.RunAsync("hello there", session));
Console.WriteLine(await agent.RunAsync("second turn", session));

JsonElement saved = await agent.SerializeSessionAsync(session);
AgentSession restored = await agent.DeserializeSessionAsync(saved);
Console.WriteLine(await agent.RunAsync("after restore", restored));   // turn 3

/// <summary>Replies in capitals and counts turns in its own session type.</summary>
public sealed class ShoutingAgent : AIAgent
{
    public override string? Name => "shouter";

    protected override ValueTask<AgentSession> CreateSessionCoreAsync(CancellationToken cancellationToken = default)
        => new(new ShoutingSession());

    protected override ValueTask<JsonElement> SerializeSessionCoreAsync(AgentSession session, JsonSerializerOptions? options = null, CancellationToken cancellationToken = default)
        => new(JsonSerializer.SerializeToElement(new SavedState(((ShoutingSession)session).Turns)));

    protected override ValueTask<AgentSession> DeserializeSessionCoreAsync(JsonElement state, JsonSerializerOptions? options = null, CancellationToken cancellationToken = default)
        => new(new ShoutingSession { Turns = state.Deserialize<SavedState>()!.Turns });

    protected override async Task<AgentResponse> RunCoreAsync(IEnumerable<ChatMessage> messages, AgentSession? session = null, AgentRunOptions? options = null, CancellationToken cancellationToken = default)
    {
        var typed = (ShoutingSession)(session ?? await this.CreateSessionAsync(cancellationToken));
        typed.Turns++;
        string text = $"TURN {typed.Turns}: {string.Join(" ", messages.Select(m => m.Text)).ToUpperInvariant()}";
        return new AgentResponse(new ChatMessage(ChatRole.Assistant, text) { AuthorName = this.Name }) { AgentId = this.Id };
    }

    protected override async IAsyncEnumerable<AgentResponseUpdate> RunCoreStreamingAsync(IEnumerable<ChatMessage> messages, AgentSession? session = null, AgentRunOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        AgentResponse response = await this.RunCoreAsync(messages, session, options, cancellationToken);
        foreach (AgentResponseUpdate update in response.ToAgentResponseUpdates())
        {
            yield return update;
        }
    }

    private sealed class ShoutingSession : AgentSession
    {
        public int Turns { get; set; }
    }

    private sealed record SavedState(int Turns);
}
```

`AgentSession` is abstract but has no abstract members, so a session class can be as small as this. It also has a `StateBag` (an `AgentSessionStateBag`) for JSON-serialisable values; a custom agent that keeps its state there can serialise with `session.StateBag.Serialize()` and restore with `AgentSessionStateBag.Deserialize`. To reuse the framework's history handling, hold a `ChatHistoryProvider` (for example `InMemoryChatHistoryProvider`) and call its `InvokingAsync` and `InvokedAsync` around your own logic, as Microsoft's `Agent_With_CustomImplementation` sample does.

To change the behaviour of an existing agent rather than write a new one, derive from `DelegatingAIAgent` or use `AIAgentBuilder`; see the [middleware page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_middleware/#agent-run-middleware).

## Choosing the model per run, routing and failover

### Per-run overrides

`ChatClientAgentRunOptions` carries a `ChatOptions` that is merged with the agent's: values you set win, unset ones fall back to the agent's, instructions are concatenated (agent first), and tools and stop sequences are added to the agent's. Setting `ModelId` switches the model for one run, on providers that honour it (the Responses API, Anthropic, Ollama; a Chat Completions `ChatClient` is bound to one model and ignores it).

```csharp
#pragma warning disable OPENAI001
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Responses;

ChatClientAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetResponsesClient()
    .AsAIAgent(model: "%%model=openai-small%%", instructions: "You are helpful.");

AgentResponse quick = await agent.RunAsync("Summarise HTTP in one line.");

var heavy = new ChatClientAgentRunOptions(new ChatOptions
{
    ModelId = "%%model=openai-flagship%%",
    Instructions = "Be thorough.",       // appended to the agent's instructions
    MaxOutputTokens = 2000,
});
AgentResponse careful = await agent.RunAsync("Explain HTTP/2 multiplexing.", options: heavy);

Console.WriteLine(quick.Text);
Console.WriteLine(careful.Text);
```

`ChatClientAgentRunOptions.ChatClientFactory` goes further and wraps the agent's chat client for one run, which is how per-request middleware is added.

### Persistent routing between models

`RoutePersistingRoutingChatClient` (experimental `MAAI001`) holds several named chat clients and sends each request to the session's active route. The route is stored in the session's state bag, so it survives serialisation, and switching routes keeps the whole history because history belongs to the session, not to the routed client.

```csharp
#pragma warning disable MAAI001
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

var openAI = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!);

var router = new RoutePersistingRoutingChatClient(
    new Dictionary<string, IChatClient>
    {
        ["fast"] = openAI.GetChatClient("%%model=openai-small%%").AsIChatClient(),
        ["smart"] = openAI.GetChatClient("%%model=openai-flagship%%").AsIChatClient(),
    },
    new RoutePersistingRoutingChatClientOptions { DefaultRoute = "fast" });

AIAgent agent = router.AsAIAgent(instructions: "You are a helpful assistant.");
AgentSession session = await agent.CreateSessionAsync();

Console.WriteLine($"[{router.GetActiveRoute(session)}] {await agent.RunAsync("Hi!", session)}");

router.SetActiveRoute(session, "smart");
Console.WriteLine($"[{router.GetActiveRoute(session)}] {await agent.RunAsync("Now a hard question: why is the sky blue?", session)}");
```

Every route must keep history client-side (no Responses stored output, no Foundry conversations), because a server-side conversation in one service cannot be continued by another. The client finds the session through `AIAgent.CurrentRunContext`, so it throws if you call it outside an agent run. `OwnsInnerClients` controls whether disposing the router disposes the routes.

### Failover

For "try the primary, fall back to the secondary", `Microsoft.Extensions.AI` has `OrderedFailoverChatClient` (experimental `MEAI001`). It tries each client in order until one succeeds; `MaximumAttemptsPerRequest` caps the attempts. A streaming request only fails over before the first update is passed on.

```csharp
#pragma warning disable MEAI001
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

var openAI = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!);

IChatClient resilient = new OrderedFailoverChatClient(
[
    openAI.GetChatClient("%%model=openai%%").AsIChatClient(),
    openAI.GetChatClient("%%model=openai-small%%").AsIChatClient(),
]);

AIAgent agent = resilient.AsAIAgent(instructions: "You are helpful.");
Console.WriteLine(await agent.RunAsync("Are you there?"));
```

Retries with back-off for transient errors belong in the HTTP layer; the [middleware page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_middleware/#retries-and-transient-errors) covers both.

## Credentials

| Provider | Key-based | Entra ID / managed identity |
| --- | --- | --- |
| OpenAI | `new OpenAIClient(apiKey)` or `new ApiKeyCredential(apiKey)` | not applicable |
| Azure OpenAI (v1 endpoint) | `new ApiKeyCredential(key)` | `new BearerTokenPolicy(credential, "https://ai.azure.com/.default")` (older resources also accept `https://cognitiveservices.azure.com/.default`) |
| Azure OpenAI (`Azure.AI.OpenAI`) | `new AzureKeyCredential(key)` | `new AzureOpenAIClient(endpoint, credential)` |
| Foundry projects | not supported by `AIProjectClient` | `new AIProjectClient(endpoint, credential)` |
| Claude in Foundry | `AnthropicFoundryApiKeyCredentials` | `AnthropicFoundryIdentityTokenCredentials` |
| Anthropic | `ClientOptions.ApiKey` or `ANTHROPIC_API_KEY` | not applicable |

`DefaultAzureCredential` is convenient in development because it tries environment variables, workload identity, managed identity, Visual Studio, the Azure CLI and others in turn. In production that probing adds latency on the first call and can pick up an identity you did not intend, so use `ManagedIdentityCredential` (or `WorkloadIdentityCredential` on AKS) directly. The roles each identity needs are on the [Azure RBAC page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_azure_rbac/). Keep keys out of source: user secrets in development, Key Vault or environment variables in production.

Register the provider client once, as a singleton. OpenAI, Azure and Anthropic clients are thread-safe and own their HTTP connection pools; creating one per request exhausts sockets. A `ChatClientAgent` is also safe to share, because per-conversation state lives in the session.

## Differences from Python

| Topic | Python (`agent-framework`) | .NET (`Microsoft.Agents.AI`) |
| --- | --- | --- |
| Agent type | `Agent` (formerly `ChatAgent`) over a chat client | `ChatClientAgent` over any `IChatClient` |
| Provider packages | One package per provider, with its own chat client classes (`OpenAIChatClient`, `AnthropicClient`, `OllamaChatClient`, ...) | Provider SDKs plus `Microsoft.Extensions.AI` adapters; `Microsoft.Agents.AI.*` packages only add `AsAIAgent` |
| Ollama, Gemini, Bedrock, Mistral | First-party packages | Through third-party `IChatClient` implementations |
| Responses API | Separate `OpenAIResponsesClient` | `ResponsesClient.AsAIAgent` (experimental `OPENAI001`) |
| Agent-level decorators | Middleware list on the agent | Decorator pipeline on the chat client plus `AIAgentBuilder` |
| Custom agents | Subclass `BaseAgent` | Subclass `AIAgent` (five abstract members) |
| Routing | Swap the client yourself | `RoutePersistingRoutingChatClient`, `OrderedFailoverChatClient` |

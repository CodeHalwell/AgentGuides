---
title: "Microsoft Agent Framework (.NET) — Tools"
description: "Function tools with AIFunctionFactory, parameter binding and schemas, the function-calling loop and its limits, tool errors, human approval and standing approvals, agents as tools, tools that change at run time, provider-hosted and Foundry tools, and the shell and code-execution tools."
framework: "microsoft-agent-framework"
language: "dotnet"
sidebar:
  label: "Tools"
  order: 4
---

This page covers everything a .NET agent can call: function tools built with `AIFunctionFactory`, the loop that runs them, what the model is told when a tool fails, human approval, other agents wrapped as tools, tools that change during a run, tools the provider runs on its own servers, and the shell and code-execution tools. MCP servers have [their own page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_mcp/), and intercepting tool calls (function middleware) is on the [middleware page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_middleware/#function-invocation-middleware).

Every C# block was compiled against `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% with `Microsoft.Extensions.AI` 10.11.0 (checked %%versions-checked%%). The blocks that use OpenAI were run against a local stand-in for the OpenAI API that always calls the first tool it is offered and then echoes the tool's result back as its answer, so the output shown is exactly what the model received. The Foundry, shell-in-Docker and code-execution blocks were compiled only.

## How tools are represented

Tools are `Microsoft.Extensions.AI` types; the Agent Framework adds no tool classes of its own except in the optional tool packages. Everything derives from `AITool`:

| Type | What it is |
| --- | --- |
| `AIFunction` | A tool the agent can run locally: a name, a description, a JSON schema and `InvokeAsync`. |
| `AIFunctionDeclaration` | A schema with no implementation. The model can call it, but the loop hands the call back to you. |
| `ApprovalRequiredAIFunction` | An `AIFunction` wrapper that makes the loop ask for approval before running it. |
| `DelegatingAIFunction` | Base class for your own wrappers (logging, caching, policy). |
| `HostedWebSearchTool`, `HostedCodeInterpreterTool`, `HostedFileSearchTool`, `HostedImageGenerationTool`, `HostedMcpServerTool`, `HostedToolSearchTool` | Markers for tools the *service* runs. The provider adapter translates them into its own tool definitions. |

An agent gets tools from three places, merged on every run: the agent's own `ChatOptions.Tools` (what `tools:` sets), the run's `ChatClientAgentRunOptions.ChatOptions.Tools`, and whatever its `AIContextProvider`s add. Run tools are appended to the agent's; nothing replaces them.

## Function tools

### From a method or a lambda

`AIFunctionFactory.Create` turns any delegate into an `AIFunction`. It reads the method name, the `[Description]` attributes on the method and its parameters, the parameter types and their default values, and builds a JSON schema from them.

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Chat;

var weather = new WeatherService();

AIFunction fromMethod = AIFunctionFactory.Create(weather.GetForecast);
AIFunction fromLambda = AIFunctionFactory.Create(
    ([Description("Temperature in Celsius")] double celsius) => celsius * 9 / 5 + 32,
    name: "to_fahrenheit",
    description: "Convert Celsius to Fahrenheit.");

Console.WriteLine($"{fromMethod.Name}: {fromMethod.Description}");
Console.WriteLine(fromMethod.JsonSchema);
Console.WriteLine(fromMethod.ReturnJsonSchema);

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsAIAgent(instructions: "You answer weather questions.", tools: [fromMethod, fromLambda]);

Console.WriteLine(await agent.RunAsync("What's the forecast for Paris?"));

public enum Units { Celsius, Fahrenheit }

public record Forecast(string City, double High, Units Units);

public class WeatherService
{
    [Description("Get the forecast for a city.")]
    public Forecast GetForecast(
        [Description("City name, e.g. 'Paris'")] string city,
        [Description("How many days ahead")] int days = 1,
        Units units = Units.Celsius) => new(city, 21.5, units);
}
```

The output shows what the model is sent and what comes back:

```text
GetForecast: Get the forecast for a city.
{"type":"object","properties":{"city":{"description":"City name, e.g. \u0027Paris\u0027","type":"string"},"days":{"description":"How many days ahead","type":"integer","default":1},"units":{"type":"string","enum":["Celsius","Fahrenheit"],"default":"Celsius"}},"required":["city"]}
{"type":"object","properties":{"city":{"type":"string"},"high":{"type":"number"},"units":{"type":"string","enum":["Celsius","Fahrenheit"]}},"required":["city","high","units"]}
Here is what the tool returned: {
  "city": "Paris",
  "high": 21.5,
  "units": "Celsius"
}
```

Things to know about the generated schema and result:

- Parameters with default values are optional (`default` in the schema, not in `required`). A nullable type (`string?`, `int?`) does not make a parameter optional: it stays in `required` with type `["string","null"]`. Give it a default value (`string? note = null`) to make it optional.
- Enums are sent as strings. Records and classes become nested objects; property names are camel-cased by the default serialiser options.
- The return value is serialised to JSON with the same options and that JSON is what the model sees. A `string` return is sent as a JSON string. `Task<T>` and `ValueTask<T>` are awaited.
- **Lambdas need a name.** Without `name:`, a lambda gets its compiler-generated name (`_Main_b_0_0` in a top-level program), which tells the model nothing and changes when you edit the file. Methods use their own name; `[Description]` on a lambda works when you put it on the lambda itself.
- The schema is generated when you call `Create`, so build tools once and reuse them.

### `AIFunctionFactoryOptions`

The overloads that take `AIFunctionFactoryOptions` expose everything:

| Option | Purpose |
| --- | --- |
| `Name`, `Description` | Override the method name and `[Description]`. |
| `SerializerOptions` | `JsonSerializerOptions` for arguments, results and schema. Supply source-generated options for trimming and Native AOT. |
| `JsonSchemaCreateOptions` | Fine control of schema generation (`AIJsonSchemaCreateOptions`). |
| `ConfigureParameterBinding` | Per parameter: hide it from the schema and supply its value yourself (below). |
| `MarshalResult` | Replace the default result serialisation, for example to return a short string instead of a large object. |
| `ExcludeResultSchema` | Do not compute `ReturnJsonSchema`. |
| `AdditionalProperties` | Metadata carried on the `AIFunction` for your own middleware. |

`[AIFunctionName("...")]` (experimental `MEAI001`) on a method sets the tool name without an options object.

### Parameters the model never sees

Three parameter types are bound by the framework and left out of the schema: `CancellationToken` (the run's token), `IServiceProvider` (the services the agent was given), and `AIFunctionArguments` (the raw arguments, plus `Context` and `Services`). Anything else can be hidden and filled in with `ConfigureParameterBinding`. This is how a tool gets the current user, a database connection or a tenant ID without the model being able to choose it.

```csharp
using System.ComponentModel;
using System.Reflection;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using OpenAI;
using OpenAI.Chat;

var services = new ServiceCollection()
    .AddSingleton<OrderStore>()
    .BuildServiceProvider();

AIFunction lookup = AIFunctionFactory.Create(
    typeof(OrderTools).GetMethod(nameof(OrderTools.LookupOrder))!,
    target: null,
    new AIFunctionFactoryOptions
    {
        Name = "lookup_order",
        // The tenant comes from the application, never from the model.
        ConfigureParameterBinding = p => p.Name == "tenantId"
            ? new AIFunctionFactoryOptions.ParameterBindingOptions
            {
                ExcludeFromSchema = true,
                BindParameter = (_, args) => args.Context?["tenant"] ?? "unknown",
            }
            : default,
    });

Console.WriteLine(lookup.JsonSchema);   // only orderId

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsAIAgent(
        instructions: "You help with orders.",
        tools: [lookup],
        services: services);           // what IServiceProvider parameters receive

// Put the tenant into the arguments' Context for every call in this run.
AIAgent tenantAware = agent.AsBuilder()
    .Use(async (agent, context, next, ct) =>
    {
        context.Arguments.Context ??= new Dictionary<object, object?>();
        context.Arguments.Context["tenant"] = "contoso";
        return await next(context, ct);
    })
    .Build();

Console.WriteLine(await tenantAware.RunAsync("Where is order 1042?"));

public sealed class OrderStore
{
    public string Find(string tenant, string id) => $"{tenant}/{id}: shipped";
}

public static class OrderTools
{
    [Description("Look up an order by ID.")]
    public static string LookupOrder(
        [Description("The order ID")] string orderId,
        string tenantId,
        IServiceProvider services,
        CancellationToken cancellationToken)
        => services.GetRequiredService<OrderStore>().Find(tenantId, orderId);
}
```

The `services:` argument matters only when the agent inserts the function loop itself; if your chat client already contains a `FunctionInvokingChatClient`, that client's services are used. The `Use(...)` call is function middleware, explained on the [middleware page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_middleware/#function-invocation-middleware).

### Instance methods and DI

A tool that is an instance method keeps its object alive for as long as the tool exists. For per-request dependencies, resolve them inside the tool from the `IServiceProvider` parameter, or use the `Create(MethodInfo, Func<AIFunctionArguments, object> createInstanceFunc, ...)` overload, which creates the target object for every call:

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using OpenAI;
using OpenAI.Chat;

var services = new ServiceCollection().AddScoped<Clock>().AddTransient<TimeTools>().BuildServiceProvider();

AIFunction now = AIFunctionFactory.Create(
    typeof(TimeTools).GetMethod(nameof(TimeTools.Now))!,
    args => args.Services!.GetRequiredService<TimeTools>(),   // a fresh TimeTools per call
    new AIFunctionFactoryOptions { Name = "current_time" });

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsAIAgent(instructions: "You tell the time.", tools: [now], services: services);

Console.WriteLine(await agent.RunAsync("What time is it in Lisbon?"));

public sealed class Clock { public DateTimeOffset UtcNow => DateTimeOffset.UtcNow; }

public sealed class TimeTools(Clock clock)
{
    [Description("Current time in a time zone.")]
    public string Now([Description("IANA time zone, e.g. Europe/Lisbon")] string zone)
    {
        // Arguments come from the model: validate them and answer in words it can act on.
        if (!TimeZoneInfo.TryFindSystemTimeZoneById(zone, out TimeZoneInfo? tz))
        {
            return $"Unknown time zone '{zone}'. Use an IANA name such as Europe/Lisbon.";
        }
        return TimeZoneInfo.ConvertTime(clock.UtcNow, tz).ToString("HH:mm");
    }
}
```

### Hand-written schemas and client-side tools

`AIFunctionFactory.CreateDeclaration(name, description, jsonSchema)` builds a tool from a JSON schema with no implementation. When the model calls it, the loop cannot run it, so it stops and returns the `FunctionCallContent` to you. You do the work (often in a browser or another process) and send a `FunctionResultContent` back on the next run. This is how client-side ("frontend") tools work in AG-UI.

```csharp
using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Chat;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;   // OpenAI.Chat has its own ChatMessage

AIFunctionDeclaration pickColour = AIFunctionFactory.CreateDeclaration(
    "pick_colour",
    "Ask the user to choose a colour in the UI.",
    JsonDocument.Parse("""{"type":"object","properties":{"prompt":{"type":"string"}},"required":["prompt"]}""").RootElement);

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsAIAgent(instructions: "You design themes.", tools: [pickColour]);

AgentSession session = await agent.CreateSessionAsync();
AgentResponse response = await agent.RunAsync("Design a theme for me.", session);

FunctionCallContent call = response.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>().Single();
Console.WriteLine($"model wants {call.Name}({string.Join(", ", call.Arguments!.Select(a => $"{a.Key}={a.Value}"))})");

// ...the UI asks the user, who picks teal...
var result = new ChatMessage(ChatRole.Tool, [new FunctionResultContent(call.CallId, "teal")]);
Console.WriteLine(await agent.RunAsync(result, session));
```

Any `AIFunction` can be turned into a declaration with `AsDeclarationOnly()`. If a response mixes server-side and client-side calls, the loop returns all of them unexecuted; `ChatClientAgentOptions.EnableInvocableFunctionBypassing` (experimental) holds back the server-side ones and runs them on the next request, so you only have to answer the client-side call.

For a schema you cannot express as a method, derive from `AIFunction` (override `Name`, `Description`, `JsonSchema` and `InvokeCoreAsync`), or from `DelegatingAIFunction` to wrap an existing one.

## The tool loop

The loop is `Microsoft.Extensions.AI`'s `FunctionInvokingChatClient`, which `ChatClientAgent` inserts into the pipeline unless your chat client already has one. Each iteration sends the conversation and the tools, runs any function calls in the response, appends the results and asks again, until the model answers without calling a tool.

| Setting | Default | Meaning |
| --- | --- | --- |
| `MaximumIterationsPerRequest` | 40 | Round trips allowed per run (including the first). When it is reached the loop makes one last request with tools switched off, so you get a text answer rather than an exception. |
| `MaximumConsecutiveErrorsPerRequest` | 3 | Consecutive iterations whose tool calls threw. One more failure after that and the exception is rethrown to the caller. `0` rethrows the first failure. |
| `IncludeDetailedErrors` | `false` | Send the exception message to the model, not just "Error: Function failed." |
| `AllowConcurrentInvocation` | `false` | Run several calls from one response in parallel. `ChatClientAgentOptions.AllowConcurrentInvocation` sets this. |
| `TerminateOnUnknownCalls` | `false` | When the model calls a tool that does not exist: `false` tells the model it was not found and carries on; `true` stops and returns the call to you. |
| `AdditionalTools` | none | Tools the loop can run but does not advertise (for services preconfigured with tool definitions). |
| `FunctionInvoker` | `null` | Replace how each function is invoked. |

To change these, build the loop yourself; the agent then reuses it instead of adding its own:

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Chat;

IChatClient chatClient = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsBuilder()
    .UseFunctionInvocation(configure: loop =>
    {
        loop.MaximumIterationsPerRequest = 8;
        loop.MaximumConsecutiveErrorsPerRequest = 1;
        loop.AllowConcurrentInvocation = true;
    })
    .Build();

AIAgent agent = chatClient.AsAIAgent(
    instructions: "You are a careful assistant.",
    tools: [AIFunctionFactory.Create(() => DateTime.UtcNow.ToString("O"), "utc_now", "Current UTC time.")]);

Console.WriteLine(agent.GetService<FunctionInvokingChatClient>()!.MaximumIterationsPerRequest);   // 8
Console.WriteLine(await agent.RunAsync("What time is it?"));
```

### Tool choice

`ChatOptions.ToolMode` tells the model whether it must call a tool. Set it on the agent or per run:

| Value | Effect |
| --- | --- |
| `ChatToolMode.Auto` (default) | The model decides. |
| `ChatToolMode.RequireAny` | The model must call some tool. |
| `ChatToolMode.RequireSpecific("name")` | The model must call that tool. |
| `ChatToolMode.None` | Tools are sent but must not be called. |

`ChatOptions.AllowMultipleToolCalls = false` asks the model for at most one call per response, which also avoids the all-or-nothing approval behaviour described below. Forcing a tool applies to every iteration of the loop, so `RequireAny` on the agent can keep the model calling tools until the iteration cap; set it on a single run instead.

### Inside a tool: the current invocation

During a call, `FunctionInvokingChatClient.CurrentContext` (an async-local) gives the tool its `FunctionInvocationContext`: the `Function`, the `Arguments`, the `CallContent`, the conversation so far (`Messages`), the `Options` for this request, the `Iteration`, the call's position (`FunctionCallIndex` of `FunctionCount`) and `IsStreaming`. Setting `Terminate = true` ends the loop after this iteration and returns the response as it stands. `AIAgent.CurrentRunContext` similarly gives the agent, session and run options.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Chat;

AIFunction handOver = AIFunctionFactory.Create(
    (string reason) =>
    {
        FunctionInvocationContext context = FunctionInvokingChatClient.CurrentContext!;
        context.Terminate = true;   // stop the loop: a person takes over from here
        AgentSession? session = AIAgent.CurrentRunContext?.Session;
        return $"Escalated (iteration {context.Iteration}, session {(session is null ? "none" : "present")}): {reason}";
    },
    name: "escalate_to_human",
    description: "Hand the conversation to a human agent.");

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsAIAgent(instructions: "Escalate anything about refunds.", tools: [handOver]);

AgentResponse response = await agent.RunAsync("I want a refund now!");
FunctionResultContent result = response.Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single();
Console.WriteLine(result.Result);
Console.WriteLine($"final text: '{response.Text}'");   // empty: the model was not called again
```

## What the model sees when a tool fails

An exception thrown by a tool does not end the run. The loop catches it, records it on the `FunctionResultContent.Exception` property and sends the model a generic message, so internal details do not leak into the prompt (and from there to the user):

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Chat;

AIFunction lookup = AIFunctionFactory.Create(
    string (string orderId) => throw new InvalidOperationException("db timeout on shard 7"),
    name: "lookup_order");

var openAI = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!);

AIAgent quiet = openAI.GetChatClient("%%model=openai%%").AsAIAgent(tools: [lookup]);
AgentResponse r1 = await quiet.RunAsync("Where is order 7?");
Console.WriteLine(r1.Text);

FunctionResultContent failed = r1.Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single();
Console.WriteLine($"for your logs: {failed.Exception?.Message}");

AIAgent detailed = openAI.GetChatClient("%%model=openai%%").AsIChatClient()
    .AsBuilder().UseFunctionInvocation(configure: l => l.IncludeDetailedErrors = true).Build()
    .AsAIAgent(tools: [lookup]);
Console.WriteLine((await detailed.RunAsync("Where is order 7?")).Text);
```

```text
Here is what the tool returned: Error: Function failed.
for your logs: db timeout on shard 7
Here is what the tool returned: Error: Function failed. Exception: db timeout on shard 7
```

The model usually reacts to the generic message by apologising or retrying with different arguments. When you want it to recover sensibly, catch the exception yourself and return a message written for the model ("Order 7 does not exist; ask the user to check the number"). Rethrow only what should end the run: after `MaximumConsecutiveErrorsPerRequest` consecutive failures the exception propagates out of `RunAsync`. Cancellation is not swallowed: an `OperationCanceledException` from the run's token ends the run.

## Approvals

### The round trip

Wrap a function in `ApprovalRequiredAIFunction` and the loop will not run it. Instead the run ends with a `ToolApprovalRequestContent` (an `InputRequestContent`) in the response messages. You show it to a person, and send their answer back in the same session as a `ToolApprovalResponseContent`, made with `request.CreateResponse(approved, reason)`.

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Chat;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;   // OpenAI.Chat has its own ChatMessage

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsAIAgent(
        instructions: "You are a banking assistant.",
        tools: [new ApprovalRequiredAIFunction(AIFunctionFactory.Create(Bank.Transfer))]);

AgentSession session = await agent.CreateSessionAsync();
AgentResponse response = await agent.RunAsync("Move 50 from ACC-1 to ACC-2", session);

List<ToolApprovalRequestContent> requests = Pending(response);
while (requests.Count > 0)
{
    var answers = requests.Select(request =>
    {
        var call = (FunctionCallContent)request.ToolCall;
        Console.Write($"Approve {call.Name}({string.Join(", ", call.Arguments!.Select(a => $"{a.Key}={a.Value}"))})? [y/N] ");
        bool approved = Console.ReadLine()?.Trim().Equals("y", StringComparison.OrdinalIgnoreCase) == true;
        return new ChatMessage(ChatRole.User, [request.CreateResponse(approved, approved ? null : "The user declined.")]);
    }).ToList();

    response = await agent.RunAsync(answers, session);
    requests = Pending(response);
}

Console.WriteLine(response.Text);

static List<ToolApprovalRequestContent> Pending(AgentResponse r) =>
    r.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>().ToList();

public static class Bank
{
    [Description("Transfer money between two accounts.")]
    public static string Transfer(string from, string to, decimal amount) => $"Moved {amount} from {from} to {to}.";
}
```

Answering `y` runs the tool and the model continues. Answering `n` does not run it; the model receives "Tool call invocation was rejected by the approver and must not be retried." followed by your reason. The request carries the exact `FunctionCallContent` the model produced (name, arguments, call ID), so show the arguments: approving "transfer" is not the same as approving "transfer 50,000 to an unknown account".

What the framework enforces around this, by default:

- **Binding.** Each response is matched to a request the framework issued in *this* session (`ApprovalResponseBindingChatClient`). A response you construct yourself for a call nobody asked about has no effect; the approved call is always the one the person was shown, whatever arguments the response carries. In %%version=Microsoft.Agents.AI%%, sending only an unbound response on a fresh session leaves the run with no messages and it throws `ArgumentException` ("Value cannot be an empty collection"), so treat that as a client error. Hosts must persist the session between the request and the answer, because the record of what was asked lives there.
- **Unanswered requests are rejected.** If the next run on the session does not answer a pending request (the user typed something else), that call is rejected and not run, so the conversation never has dangling tool calls.
- **Tools that need no approval are not held up.** The function loop is all-or-nothing: if one call in a response needs approval, every call in that response becomes an approval request. The agent's `ApprovalNotRequiredFunctionBypassingChatClient` answers the requests for tools that do not need approval on your behalf, so you are only asked about the ones that do. Set `DisableApprovalNotRequiredFunctionBypassing` to see them all.
- **Approval is re-checked.** A stored decision is only reused while the tool still does not require approval. If a tool becomes approval-required between turns (or disappears), the pending call is rejected and the model has to ask again.

Streaming works the same way: the `ToolApprovalRequestContent` arrives in an update's `Contents`, and the stream ends there. For hosted MCP tools the request's `ToolCall` is a `McpServerToolCallContent` (server name, tool name, arguments) instead of a `FunctionCallContent`; see the [MCP page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_mcp/#approvals-for-mcp-tools).

### Standing approvals with `ToolApprovalAgent`

`ToolApprovalAgent` adds "don't ask again" on top. Add it with `.AsBuilder().UseToolApproval(...)`. It does two things: it presents queued approval requests one at a time, and it records standing rules when the caller answers with an "always approve" response.

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Chat;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;   // OpenAI.Chat has its own ChatMessage

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsAIAgent(
        instructions: "You manage files.",
        tools: [new ApprovalRequiredAIFunction(AIFunctionFactory.Create(Files.Archive))])
    .AsBuilder()
    .UseToolApproval(new ToolApprovalAgentOptions
    {
        // Rules evaluated before anyone is asked; true means approve without asking.
        AutoApprovalRules =
        [
            context => ValueTask.FromResult(
                context.FunctionCallContent.Name == "Archive"
                && context.FunctionCallContent.Arguments?["path"]?.ToString()?.StartsWith("/tmp/") == true),
        ],
    })
    .Build();

AgentSession session = await agent.CreateSessionAsync();
AgentResponse response = await agent.RunAsync("Archive the reports folder.", session);

ToolApprovalRequestContent request = response.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>().First();

// "Yes, and don't ask me again for this tool in this session."
response = await agent.RunAsync(new ChatMessage(ChatRole.User, [request.CreateAlwaysApproveToolResponse("trusted")]), session);
Console.WriteLine(response.Text);

public static class Files
{
    [Description("Archive a folder.")]
    public static string Archive(string path) => $"Archived {path}.";
}
```

`CreateAlwaysApproveToolResponse` records a rule for the tool; `CreateAlwaysApproveToolWithArgumentsResponse` records one for the tool with exactly these arguments. Rules are stored in the session's state bag under `toolApprovalState`, so they persist when the session is serialised and apply only to that session. `ToolApprovalAgent.AllToolsAutoApprovalRule` approves everything (useful in tests and fully trusted batch jobs, nowhere else). `MaxAutoApprovalIterations` (default 40) caps how many times in a row the agent will auto-approve and re-run before handing the remaining requests back.

Standing rules match by tool *name*. If a session is reused with a rebuilt agent whose tool of the same name does something else (more likely with tools from MCP servers), the old approval applies to the new tool. Check standing approvals before reusing a session with a different tool set.

## Agents as tools

`agent.AsAIFunction()` wraps any `AIAgent` as an `AIFunction` with one parameter, `query`. Its name is the agent's `Name` with characters that are not valid in a function name replaced by underscores (`"Weather Expert!"` becomes `Weather_Expert_`), and its description is the agent's `Description`. Give the inner agent both, because they are all the outer model knows about it.

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Chat;

var openAI = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!);

AIAgent weatherAgent = openAI.GetChatClient("%%model=openai-small%%").AsAIAgent(
    instructions: "You answer questions about the weather.",
    name: "WeatherAgent",
    description: "Answers questions about the current weather in a city.",
    tools: [AIFunctionFactory.Create(Weather.Get)]);

AIAgent concierge = openAI.GetChatClient("%%model=openai%%").AsAIAgent(
    instructions: "You are a hotel concierge. Reply in French.",
    tools: [weatherAgent.AsAIFunction()]);

Console.WriteLine(await concierge.RunAsync("Should I take an umbrella in Amsterdam today?"));

public static class Weather
{
    [Description("Get the weather for a city.")]
    public static string Get(string city) => $"{city}: cloudy, 15°C, showers later.";
}
```

How it behaves:

- Each call runs the inner agent **without a session** unless you pass one (`AsAIFunction(session: ...)`), so by default the inner agent remembers nothing between calls. Passing a session makes it stateful, but that session is then shared by every run of the outer agent that uses this tool instance.
- The tool returns only the inner agent's final text. Its tool calls, reasoning and usage are not visible to the outer agent.
- `AdditionalProperties` from the outer run's options are passed on to the inner run.
- `AIFunctionFactoryOptions` lets you override the name and description.

Use an agent as a tool when the outer agent stays in charge and the inner one is a specialist it consults. When control should pass from one agent to another, or you want each step observable and checkpointed, use a workflow (sequential, handoff, group chat) instead.

The reverse also exists: `agent.AsIChatClient()` (experimental) presents a `ChatClientAgent` as an `IChatClient`, for libraries that only accept chat clients.

## Tools that change during a run

### Per-run tools

Tools passed in `ChatClientAgentRunOptions` are added to the agent's own for that run only:

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Chat;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsAIAgent(instructions: "You are helpful.");

bool isAdmin = true;   // from the signed-in user
var options = new ChatClientAgentRunOptions(new ChatOptions
{
    Tools = isAdmin ? [AIFunctionFactory.Create((string user) => $"{user} reset", "reset_password", "Reset a user's password.")] : [],
});

Console.WriteLine(await agent.RunAsync("Reset the password for jdoe.", options: options));
```

Context providers can also add tools on each run (`AIContext.Tools`), which is how the built-in todo, skills and file-memory providers expose theirs; see [sessions and memory](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_sessions_and_memory/#context-providers).

### Loading tools from inside the loop

`FunctionInvocationContext.Options.Tools` is the live tool list for the current request; a tool can add to it, and the next iteration offers the new tools. This lets an agent start with a small "find tools" function and load the rest on demand, which keeps large tool sets out of the prompt.

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Chat;

Dictionary<string, AIFunction> catalogue = new(StringComparer.OrdinalIgnoreCase)
{
    ["weather"] = AIFunctionFactory.Create((string city) => $"{city}: 18°C", "get_weather", "Weather for a city."),
    ["time"] = AIFunctionFactory.Create((string city) => $"{city}: 09:00", "get_time", "Local time in a city."),
};

AIFunction requestTools = AIFunctionFactory.Create(
    ([Description("What you need, e.g. 'weather' or 'time'")] string need) =>
    {
        IList<AITool> live = FunctionInvokingChatClient.CurrentContext!.Options!.Tools!;
        List<string> added = [];
        foreach (var (key, tool) in catalogue)
        {
            if (need.Contains(key, StringComparison.OrdinalIgnoreCase) && !live.Any(t => t.Name == tool.Name))
            {
                live.Add(tool);
                added.Add(tool.Name);
            }
        }
        return added.Count > 0 ? $"Loaded: {string.Join(", ", added)}" : $"Nothing matched. Categories: {string.Join(", ", catalogue.Keys)}";
    },
    name: "request_tools",
    description: "Load extra tools by describing what you need.");

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsAIAgent(instructions: "Call request_tools when you lack a tool.", tools: [requestTools]);

Console.WriteLine(await agent.RunAsync("What's the weather in Oslo?"));
```

The added tools last for the current run only; the next run starts again from the agent's list. Providers with server-side tool search can do the same thing natively: `HostedToolSearchTool` (experimental `MEAI001`) lists `DeferredTools` whose definitions the service loads only when the model searches for them.

### Message injection

With `ChatClientAgentOptions.EnableMessageInjection = true`, a `MessageInjectingChatClient` sits inside the loop, and code running during a tool call can queue messages that are added before the next model call: `agent.GetService<MessageInjectingChatClient>()!.EnqueueMessagesAsync(session, messages)`. Use it for progress notes or user interjections that arrive while a long tool runs. Enable `RequirePerServiceCallChatHistoryPersistence` as well so the injected messages are saved.

## Hosted tools

Hosted tools are run by the provider, not by your process. You pass a marker object; the provider adapter translates it; the results come back as typed content.

| Tool | What it does | Result content |
| --- | --- | --- |
| `HostedWebSearchTool` | Web search with citations | `WebSearchToolCallContent`, `WebSearchToolResultContent`, citations as annotations on `TextContent` |
| `HostedCodeInterpreterTool` | Runs Python in a sandbox; `Inputs` can hold files (`HostedFileContent`) | `CodeInterpreterToolCallContent`, `CodeInterpreterToolResultContent` |
| `HostedFileSearchTool` | Searches vector stores; `Inputs` holds `HostedVectorStoreContent`, `MaximumResultCount` | citations as annotations |
| `HostedImageGenerationTool` | Image generation; `Options` is an `ImageGenerationOptions` | `ImageGenerationToolCallContent`, `ImageGenerationToolResultContent` |
| `HostedMcpServerTool` | The service calls a remote MCP server | `McpServerToolCallContent`, `McpServerToolResultContent` |
| `HostedToolSearchTool` (experimental) | Server-side tool search over deferred tools | provider-specific |

Support depends on the provider: the OpenAI Responses API and Foundry support all of these; the Chat Completions API has no equivalent for most of them, so use the Responses API when you need hosted tools; Anthropic's adapter maps the ones its API has. Hosted tools run inside one model call, so they do not go through the function loop, function middleware or `ApprovalRequiredAIFunction` (hosted MCP has its own approval mode).

```csharp
#pragma warning disable OPENAI001
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Responses;

AIAgent analyst = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetResponsesClient()
    .AsAIAgent(
        model: "%%model=openai%%",
        instructions: "You research and calculate. Show your working.",
        tools: [new HostedWebSearchTool(), new HostedCodeInterpreterTool()]);

AgentResponse response = await analyst.RunAsync("Find the population of Portugal and compute 3% of it.");
Console.WriteLine(response.Text);

foreach (AIContent content in response.Messages.SelectMany(m => m.Contents))
{
    if (content is CodeInterpreterToolCallContent code)
    {
        Console.WriteLine($"ran code: {string.Join("", code.Inputs?.OfType<DataContent>().Select(d => System.Text.Encoding.UTF8.GetString(d.Data.Span)) ?? [])}");
    }
}
```

Files the code interpreter writes come back as citations whose `RawRepresentation` is the OpenAI SDK's container-file annotation; download them with the SDK's `ContainerClient`.

### Foundry tools

Foundry agents support further hosted tools that have no `Microsoft.Extensions.AI` marker. `FoundryAITool` (in `Microsoft.Agents.AI.Foundry`) creates them as `AITool`s you can pass to a code-defined Foundry agent:

| Factory | Tool |
| --- | --- |
| `CreateBingGroundingTool`, `CreateBingCustomSearchTool` | Grounding with Bing Search |
| `CreateAzureAISearchTool` | Azure AI Search index |
| `CreateSharepointTool` | SharePoint grounding |
| `CreateMicrosoftFabricTool` | Fabric data agent |
| `CreateOpenApiTool` | Call an OpenAPI-described HTTP API |
| `CreateBrowserAutomationTool` | Browser automation |
| `CreateComputerTool` (experimental) | Computer use |
| `CreateA2ATool` | Call a remote A2A agent |
| `CreateMcpTool`, `CreateHostedMcpToolbox` | Remote MCP server, Foundry MCP toolbox |
| `CreateWebSearchTool`, `CreateFileSearchTool`, `CreateCodeInterpreterTool`, `CreateImageGenerationTool` | The standard Responses tools with Foundry-specific options |
| `CreateStructuredOutputsTool`, `CreateFunctionTool`, `FromResponseTool` | Lower-level escape hatches |

```csharp
#pragma warning disable OPENAI001
using Azure.AI.Projects;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Foundry;
using Microsoft.Extensions.AI;

var project = new AIProjectClient(new Uri(Environment.GetEnvironmentVariable("FOUNDRY_PROJECT_ENDPOINT")!), new DefaultAzureCredential());

AIAgent agent = project.AsAIAgent(
    model: Environment.GetEnvironmentVariable("FOUNDRY_MODEL") ?? "%%model=openai%%",
    instructions: "You answer with sources.",
    tools:
    [
        FoundryAITool.CreateWebSearchTool(),
        new HostedCodeInterpreterTool { Inputs = [] },
    ]);

Console.WriteLine(await agent.RunAsync("What changed in the latest .NET release? Summarise in three bullets."));
```

For an agent defined in Foundry, the hosted tools are part of its stored definition (`DeclarativeAgentDefinition.Tools`) and you do not pass them at run time; local function tools passed to `AsAIAgent(agentVersion, tools: ...)` are still run by your process.

## Shell tools

`Microsoft.Agents.AI.Tools.Shell` (preview) lets an agent run shell commands. A `ShellExecutor` runs commands; `AsAIFunction()` exposes it as a tool named `run_shell`, **approval-required by default**.

| Executor | Runs commands | Notes |
| --- | --- | --- |
| `LocalShellExecutor` | On the host, as your process | Requires `AcknowledgeUnsafe = true`; `ConfineWorkingDirectory`, `CleanEnvironment`, `Policy`, `Timeout`, `MaxOutputBytes` |
| `DockerShellExecutor` | In a container | `Image`, `Network` (none by default), `MemoryBytes`, `PidsLimit`, `ReadOnlyRoot`, non-root `User`, `HostWorkdir` mount; `IsAvailableAsync()` checks for Docker |

Both support `ShellMode.Stateless` (a new shell per command) and `ShellMode.Persistent` (one long-lived shell, so `cd` and variables carry over). `ShellPolicy` takes a deny list, an allow list and a custom function, and rejects commands before they run; a policy is a guard-rail against mistakes, not a sandbox, because a shell has many ways to spell the same command. `ShellEnvironmentProvider` is a context provider that probes the shell once (OS, shell version, working directory, versions of tools such as `git` or `python`) and tells the model, so it writes commands that work.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Tools.Shell;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Chat;

await using var shell = new DockerShellExecutor(new DockerShellExecutorOptions
{
    Image = "mcr.microsoft.com/dotnet/sdk:10.0",
    HostWorkdir = Directory.GetCurrentDirectory(),
    MountReadonly = true,
    Mode = ShellMode.Persistent,
    Timeout = TimeSpan.FromSeconds(60),
    Policy = new ShellPolicy(denyList: ["rm -rf", "curl", "wget"]),
});
await shell.InitializeAsync();

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsAIAgent(new ChatClientAgentOptions
    {
        ChatOptions = new ChatOptions
        {
            Instructions = "You inspect the repository with shell commands.",
            Tools = [shell.AsAIFunction()],   // run_shell, approval required
        },
        AIContextProviders = [new ShellEnvironmentProvider(shell)],
    });

AgentSession session = await agent.CreateSessionAsync();
AgentResponse response = await agent.RunAsync("How many C# files are there?", session);
foreach (ToolApprovalRequestContent request in response.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>())
{
    Console.WriteLine($"wants to run: {((FunctionCallContent)request.ToolCall).Arguments?["command"]}");
}
```

Docker defaults to no network and a read-only mount here. `UnsafeExtraRunArgs` passes extra `docker run` flags and was called `ExtraRunArgs` before 1.26.0.

## Code execution (CodeAct)

Instead of many small tool calls, a CodeAct agent writes one Python program that calls your tools as functions. Two preview packages provide it as a context provider that adds an `execute_code` tool:

- `Microsoft.Agents.AI.Hyperlight`: `HyperlightCodeActProvider` runs the code in a Hyperlight micro-VM (`HyperlightCodeActProviderOptions.CreateForWasm(guestPath)`), with `Tools`, file mounts and allowed network domains.
- `Microsoft.Agents.AI.LocalCodeAct`: `LocalCodeActProvider` runs it in a local Python process with import and built-in allow/block lists, file mounts and `ProcessExecutionLimits`. It is not a security boundary; use it only with trusted input.

Both take an approval mode (`AlwaysRequire` or `NeverRequire`) for `execute_code`, and tools you pass that are `ApprovalRequiredAIFunction`s still need approval when the code calls them.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.LocalCodeAct;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Chat;

using var codeAct = new LocalCodeActProvider("/usr/bin/python3", new LocalCodeActProviderOptions
{
    ApprovalMode = LocalCodeActApprovalMode.NeverRequire,
    Tools =
    [
        AIFunctionFactory.Create((string sku) => sku == "A1" ? 12.5 : 3.0, "price", "Price of a product."),
    ],
    BlockedImports = ["os", "subprocess", "socket"],
    ExecutionLimits = new ProcessExecutionLimits { TimeoutSeconds = 10 },
});

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsAIAgent(new ChatClientAgentOptions
    {
        ChatOptions = new() { Instructions = "Prefer one execute_code call that uses call_tool(...) over many direct tool calls." },
        AIContextProviders = [codeAct],
    });

Console.WriteLine(await agent.RunAsync("What do 3 of A1 and 2 of B2 cost in total?"));
```

## MCP tools

Tools from an MCP server are `McpClientTool`s, which are `AIFunction`s, so they go in `tools:` like any other function and pass through the same loop, middleware and approvals. Connecting, transports, authentication, filtering and long-running MCP tasks are on the [MCP page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_mcp/).

## Differences from Python

| Topic | Python | .NET |
| --- | --- | --- |
| Defining a tool | `@tool` decorator on a function | `AIFunctionFactory.Create(delegate)`; `[Description]` attributes |
| Hidden parameters | Parameters annotated as runtime context | `CancellationToken`, `IServiceProvider`, `AIFunctionArguments`, or `ConfigureParameterBinding` |
| Approval | `approval_mode="always_require"` on the tool | `new ApprovalRequiredAIFunction(fn)`; standing approvals via `ToolApprovalAgent` |
| Loop limits | `max_iterations`, configured on the client | `FunctionInvokingChatClient` properties (`MaximumIterationsPerRequest`, `MaximumConsecutiveErrorsPerRequest`) |
| Error detail to the model | Configurable | Off by default (`IncludeDetailedErrors`) |
| Agent as tool | `agent.as_tool()` | `agent.AsAIFunction()` (one `query` parameter) |
| Hosted tools | Provider-specific tool classes | `Microsoft.Extensions.AI` `Hosted*Tool` markers plus `FoundryAITool` |
| Shell | Shell tool in the core package | `Microsoft.Agents.AI.Tools.Shell` (preview), approval required by default |

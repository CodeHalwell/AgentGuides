---
title: "Microsoft Agent Framework (.NET) — MCP"
description: "Model Context Protocol in .NET agents: connecting over stdio and Streamable HTTP, filtering, renaming and approving MCP tools, prompts and resources, authentication and per-run headers, long-running MCP tasks, provider-hosted MCP, and exposing tools and agents as MCP servers."
framework: "microsoft-agent-framework"
language: "dotnet"
sidebar:
  label: "MCP"
  order: 5
---

The Agent Framework does not have its own MCP client. It uses the official C# SDK (`ModelContextProtocol`, with `ModelContextProtocol.AspNetCore` for HTTP servers), whose client tools are already `AIFunction`s, so an MCP server's tools drop straight into an agent's `tools:` list. On top of that the framework adds `Microsoft.Agents.AI.Mcp` (alpha) for long-running MCP tasks and MCP-delivered skills, and the provider packages add hosted MCP, where the model provider calls the MCP server for you.

Every C# block was compiled against `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%%, the `Microsoft.Agents.AI.Mcp` alpha released with it, and `ModelContextProtocol` 2.2.0 (checked %%versions-checked%%). The stdio, HTTP and tasks examples start their own MCP server, so they run without anything else installed; they were run against a local stand-in for the OpenAI API that calls the first tool it is offered and echoes the result. The hosted-MCP and Foundry examples were compiled only.

```bash
dotnet add package ModelContextProtocol
dotnet add package ModelContextProtocol.AspNetCore      # only to serve MCP over HTTP
dotnet add package Microsoft.Agents.AI.Mcp --prerelease # only for MCP tasks and MCP skills
```

The current `Microsoft.Agents.AI.Mcp` alpha depends on `ModelContextProtocol.Extensions.Tasks` 2.2.0, which pins `ModelContextProtocol` to exactly 2.2.0. With the newer 2.3.0 the build still succeeds but NuGet warns (NU1608) about the mismatch, so if you use the alpha package, reference `ModelContextProtocol` 2.2.0 explicitly in your project file.

## Two ways an agent uses MCP

| | Local MCP (client in your process) | Hosted MCP (provider calls the server) |
| --- | --- | --- |
| How | `McpClient` + `ListToolsAsync()` → `tools:` | `HostedMcpServerTool` (or a Foundry MCP tool) → `tools:` |
| Who connects | Your process | The model provider's servers |
| Server must be reachable from | Your process (stdio works) | The public internet (or the provider's network) |
| Tool calls pass through | The function loop, function middleware, `ApprovalRequiredAIFunction` | One model call; the provider's own approval mode |
| Providers | Any `IChatClient` | OpenAI Responses, Azure OpenAI Responses, Foundry |

Use local MCP by default: it works with every model, the calls are visible to your middleware and telemetry, and stdio servers need no network. Use hosted MCP when the server is public and you want to save the round trips through your process.

## Connecting to a server

### Stdio

A stdio server is a child process that speaks MCP on its standard input and output. `StdioClientTransport` starts it; `McpClient.CreateAsync` connects and performs the initialise handshake. The example below is its own server: started with `--server` it serves two tools, otherwise it launches itself as a child process and connects to it.

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using ModelContextProtocol.Client;
using ModelContextProtocol.Server;
using OpenAI;

if (args is ["--server"])
{
    HostApplicationBuilder server = Host.CreateApplicationBuilder();
    server.Logging.ClearProviders();   // stdout belongs to the protocol: never log to it
    server.Services
        .AddMcpServer(o => o.ServerInfo = new() { Name = "calculator", Version = "1.0.0" })
        .WithStdioServerTransport()
        .WithTools<CalculatorTools>();
    await server.Build().RunAsync();
    return;
}

await using McpClient mcp = await McpClient.CreateAsync(new StdioClientTransport(new StdioClientTransportOptions
{
    Name = "calculator",
    Command = "dotnet",
    Arguments = [typeof(CalculatorTools).Assembly.Location, "--server"],
}));

Console.WriteLine($"connected to {mcp.ServerInfo.Name} {mcp.ServerInfo.Version}");

IList<McpClientTool> tools = await mcp.ListToolsAsync();
foreach (McpClientTool tool in tools)
{
    Console.WriteLine($"  {tool.Name}: {tool.Description}");
}

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "Use the calculator for arithmetic.", tools: [.. tools]);

Console.WriteLine(await agent.RunAsync("What is 17 + 25?"));

[McpServerToolType]
public sealed class CalculatorTools
{
    [McpServerTool(Name = "add", ReadOnly = true), Description("Add two integers.")]
    public static int Add(int a, int b) => a + b;

    [McpServerTool(Name = "divide", ReadOnly = true), Description("Divide a by b.")]
    public static double Divide(double a, double b) => b == 0 ? throw new ArgumentException("b must not be zero") : a / b;
}
```

```text
connected to calculator 1.0.0
  divide: Divide a by b.
  add: Add two integers.
Here is what the tool returned: {
  "$type": "text",
  "text": "1"
}
```

The stand-in model called the first tool in the list (`divide`, with its own sample arguments 3 and 3); do not rely on the order in which a server lists its tools. Note what the model received: `McpClientTool` returns the server's content blocks as `Microsoft.Extensions.AI` content (here a `TextContent`), and the OpenAI adapter serialises that object as JSON. If you want the model to see plain text, wrap the tool in a `DelegatingAIFunction` that flattens the result, as shown under [results](#what-a-tool-call-returns).

`StdioClientTransportOptions`:

| Option | Default | Meaning |
| --- | --- | --- |
| `Command`, `Arguments` | required | The program to start, e.g. `npx` with `["-y", "@modelcontextprotocol/server-github"]`, or `uvx`, `docker`, `dotnet`. |
| `Name` | command | Shown in logs and errors. |
| `WorkingDirectory` | current | Working directory of the child. |
| `EnvironmentVariables` | none | Extra variables, e.g. `GITHUB_PERSONAL_ACCESS_TOKEN`. A `null` value removes an inherited variable. |
| `InheritEnvironmentVariables` | `true` | Pass your process's environment on. Turn off and use `GetDefaultEnvironmentVariables()` to pass only the essentials (`PATH` and the like), which keeps your secrets away from third-party servers. |
| `StandardErrorLines` | none | Callback for each line the server writes to stderr; the only place a stdio server's logs go. |
| `ShutdownTimeout` | 5 s | How long to wait for the child to exit on dispose. |

Disposing the `McpClient` stops the child process. Keep the client alive as long as the agent uses its tools: a tool whose client has been disposed fails when called.

### Streamable HTTP

Remote servers use `HttpClientTransport`. `TransportMode` defaults to auto-detection (Streamable HTTP, falling back to the legacy SSE transport); set `HttpTransportMode.StreamableHttp` when you know the server, to skip the probe.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;
using OpenAI;

await using McpClient learn = await McpClient.CreateAsync(new HttpClientTransport(new HttpClientTransportOptions
{
    Endpoint = new Uri("https://learn.microsoft.com/api/mcp"),
    Name = "Microsoft Learn",
    TransportMode = HttpTransportMode.StreamableHttp,
    ConnectionTimeout = TimeSpan.FromSeconds(30),
}));

IList<McpClientTool> tools = await learn.ListToolsAsync();

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "Answer from Microsoft Learn and cite the page.", tools: [.. tools]);

Console.WriteLine(await agent.RunAsync("How do I create a storage account with the Azure CLI?"));
```

`HttpClientTransportOptions`:

| Option | Meaning |
| --- | --- |
| `Endpoint` | The server URL. |
| `TransportMode` | `AutoDetect` (default), `StreamableHttp` or `Sse`. |
| `AdditionalHeaders` | Headers sent on every request (API keys, tenant IDs). |
| `OAuth` | `ClientOAuthOptions` for servers protected by OAuth 2.1 ([below](#authentication)). |
| `ConnectionTimeout` | Time allowed to connect. |
| `KnownSessionId`, `OwnsSession` | Resume an existing MCP session, and whether disposing the client ends it on the server. |
| `MaxReconnectionAttempts`, `DefaultReconnectionInterval` | How the client reconnects a dropped SSE stream. |
| `EnableStandaloneGetStream` | Open the optional GET stream for server-initiated messages. |

Pass your own `HttpClient` as the second constructor argument to control handlers, proxies, timeouts and resilience; the transport does not dispose it unless you set `ownsHttpClient: true`.

## Choosing which tools the agent sees

`ListToolsAsync` returns everything the server advertises. Large servers expose dozens of tools, each of which costs prompt tokens and gives the model another way to go wrong, so filter them. `McpClientTool.WithName` and `WithDescription` return renamed copies, which avoids clashes when two servers both have a `search` tool.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;
using OpenAI;

await using McpClient github = await McpClient.CreateAsync(new StdioClientTransport(new()
{
    Name = "github",
    Command = "npx",
    Arguments = ["-y", "@modelcontextprotocol/server-github"],
    EnvironmentVariables = new Dictionary<string, string?>
    {
        ["GITHUB_PERSONAL_ACCESS_TOKEN"] = Environment.GetEnvironmentVariable("GITHUB_TOKEN"),
    },
}));

HashSet<string> wanted = ["search_repositories", "get_file_contents", "list_issues"];

List<AITool> tools = (await github.ListToolsAsync())
    .Where(t => wanted.Contains(t.Name))
    .Select(t => (AITool)t.WithName($"github_{t.Name}"))   // prefix: unique across servers
    .ToList();

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You answer questions about GitHub repositories.", tools: tools);

Console.WriteLine(await agent.RunAsync("List the open issues in microsoft/agent-framework labelled 'dotnet'."));
```

Each tool's `ProtocolTool` property holds the raw MCP definition, including its `Annotations` (`ReadOnlyHint`, `DestructiveHint`, `IdempotentHint`, `OpenWorldHint`). Annotations are hints from the server, not guarantees; use them to choose defaults, not as a security boundary.

When even a filtered set is too large, load tools on demand with the pattern in [loading tools from inside the loop](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_tools/#loading-tools-from-inside-the-loop), or let the provider do it with `HostedToolSearchTool`.

## Approvals for MCP tools

Local MCP tools are ordinary `AIFunction`s, so `ApprovalRequiredAIFunction` works on them unchanged. A common policy is "read-only tools run, everything else asks":

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using ModelContextProtocol.Client;
using ModelContextProtocol.Server;
using OpenAI;

if (args is ["--server"])
{
    var server = Host.CreateApplicationBuilder();
    server.Logging.ClearProviders();
    server.Services.AddMcpServer().WithStdioServerTransport().WithTools<NoteTools>();
    await server.Build().RunAsync();
    return;
}

await using McpClient mcp = await McpClient.CreateAsync(new StdioClientTransport(new()
{
    Name = "notes",
    Command = "dotnet",
    Arguments = [typeof(NoteTools).Assembly.Location, "--server"],
}));

List<AITool> tools = (await mcp.ListToolsAsync())
    .Select(t => t.ProtocolTool.Annotations?.ReadOnlyHint == true ? (AITool)t : new ApprovalRequiredAIFunction(t))
    .ToList();

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You manage notes.", tools: tools);

AgentSession session = await agent.CreateSessionAsync();
AgentResponse response = await agent.RunAsync("Delete the note called 'draft'.", session);

foreach (ToolApprovalRequestContent request in response.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>())
{
    var call = (FunctionCallContent)request.ToolCall;
    Console.WriteLine($"approval needed: {call.Name}({string.Join(", ", call.Arguments!.Select(a => $"{a.Key}={a.Value}"))})");
    response = await agent.RunAsync(new ChatMessage(ChatRole.User, [request.CreateResponse(false, "Not now.")]), session);
}

Console.WriteLine(response.Text);

[McpServerToolType]
public sealed class NoteTools
{
    [McpServerTool(Name = "delete_note", Destructive = true), Description("Delete a note by title.")]
    public static string Delete(string title) => $"Deleted '{title}'.";

    [McpServerTool(Name = "list_notes", ReadOnly = true), Description("List note titles.")]
    public static string[] List() => ["draft", "ideas"];
}
```

When the model calls `delete_note`, the run stops with an approval request; `list_notes` runs straight away because the server marked it read-only.

Standing approvals (`ToolApprovalAgent`) work too, but they match by tool name only. A server can change what a tool does between versions without changing its name, so treat standing approvals for third-party MCP tools with care; the [tools page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_tools/#standing-approvals-with-toolapprovalagent) has the details.

## What a tool call returns

Through the agent, a successful call returns the server's content blocks converted to `AIContent` (`TextContent`, `DataContent` for images and audio, and so on), which is what is placed in the `FunctionResultContent`. When the server reports an error (`isError: true`), the tool returns the whole `CallToolResult` and the model sees it as JSON. The C# MCP server hides exception details by default: a tool that throws `InvalidOperationException("secret detail")` is reported to the client as "An error occurred invoking 'fail'.", with `isError: true`.

You can also call tools yourself, without a model, with `mcp.CallToolAsync(name, arguments)` (returns a `CallToolResult` with `Content`, `StructuredContent` and `IsError`) or `tool.CallAsync(arguments)`. Use this for deterministic steps in a workflow, or to test a server.

To change what the model sees, wrap the tool. This one flattens text results to a plain string:

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using ModelContextProtocol.Client;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;
using OpenAI;

if (args is ["--server"])
{
    var server = Host.CreateApplicationBuilder();
    server.Logging.ClearProviders();
    server.Services.AddMcpServer().WithStdioServerTransport().WithTools<ClockTools>();
    await server.Build().RunAsync();
    return;
}

await using McpClient mcp = await McpClient.CreateAsync(new StdioClientTransport(new()
{
    Name = "clock",
    Command = "dotnet",
    Arguments = [typeof(ClockTools).Assembly.Location, "--server"],
}));

IList<McpClientTool> tools = await mcp.ListToolsAsync();

// Called directly: the raw MCP result.
CallToolResult raw = await mcp.CallToolAsync("utc_now");
Console.WriteLine($"direct: {((TextContentBlock)raw.Content[0]).Text}");

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(tools: [.. tools.Select(t => new PlainTextResult(t))]);

Console.WriteLine(await agent.RunAsync("What time is it?"));

/// <summary>Returns the text of MCP results instead of serialised content objects.</summary>
public sealed class PlainTextResult(AIFunction inner) : DelegatingAIFunction(inner)
{
    protected override async ValueTask<object?> InvokeCoreAsync(AIFunctionArguments arguments, CancellationToken cancellationToken)
    {
        object? result = await base.InvokeCoreAsync(arguments, cancellationToken);
        return result switch
        {
            TextContent text => text.Text,
            IEnumerable<AIContent> contents => string.Join("\n", contents.OfType<TextContent>().Select(c => c.Text)),
            _ => result,
        };
    }
}

[McpServerToolType]
public sealed class ClockTools
{
    [McpServerTool(Name = "utc_now", ReadOnly = true), Description("Current UTC time.")]
    public static string UtcNow() => DateTimeOffset.UtcNow.ToString("O");
}
```

## Prompts and resources

MCP servers can also offer prompts (templated messages) and resources (documents). The agent does not pick these up on its own; you fetch them and decide where they go. The SDK's `ToChatMessages()` and `ToAIContents()` extensions convert them to `Microsoft.Extensions.AI` types.

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using ModelContextProtocol;
using ModelContextProtocol.Client;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;
using OpenAI;

if (args is ["--server"])
{
    var server = Host.CreateApplicationBuilder();
    server.Logging.ClearProviders();
    server.Services.AddMcpServer().WithStdioServerTransport().WithPrompts<ReviewPrompts>().WithResources<PolicyResources>();
    await server.Build().RunAsync();
    return;
}

await using McpClient mcp = await McpClient.CreateAsync(new StdioClientTransport(new()
{
    Name = "review",
    Command = "dotnet",
    Arguments = [typeof(ReviewPrompts).Assembly.Location, "--server"],
}));

foreach (McpClientPrompt prompt in await mcp.ListPromptsAsync())
{
    Console.WriteLine($"prompt: {prompt.Name} - {prompt.Description}");
}
foreach (McpClientResource resource in await mcp.ListResourcesAsync())
{
    Console.WriteLine($"resource: {resource.Uri} ({resource.MimeType})");
}

// A prompt becomes the run's input; a resource becomes extra context.
GetPromptResult review = await mcp.GetPromptAsync("code_review", new Dictionary<string, object?> { ["language"] = "C#" });
ReadResourceResult policy = await mcp.ReadResourceAsync("docs://policy");

List<ChatMessage> input =
[
    new(ChatRole.User, [new TextContent("House rules:"), .. policy.Contents.ToAIContents()]),
    .. review.ToChatMessages(),
];

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You review code.");

Console.WriteLine(await agent.RunAsync(input));

[McpServerPromptType]
public sealed class ReviewPrompts
{
    [McpServerPrompt(Name = "code_review"), Description("Ask for a code review.")]
    public static string CodeReview(string language) => $"Review the following {language} change for correctness and style.";
}

[McpServerResourceType]
public sealed class PolicyResources
{
    [McpServerResource(UriTemplate = "docs://policy", Name = "policy", MimeType = "text/plain"), Description("Review policy.")]
    public static string Policy() => "No public fields. Async all the way.";
}
```

To make resources available on every turn, put the fetch in an `AIContextProvider` (see [sessions and memory](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_sessions_and_memory/#context-providers)), or use MCP-based skills (below), which load documents only when the model asks for them.

## Authentication

### Static headers and API keys

For a key that does not change, `HttpClientTransportOptions.AdditionalHeaders` is enough: `AdditionalHeaders = new Dictionary<string, string> { ["Authorization"] = $"Bearer {key}" }`.

### Per-run tokens

When each user or tenant needs a different token, the MCP client should still be shared (connecting per request is slow), so the token has to be attached per HTTP request. A `DelegatingHandler` on the transport's `HttpClient` can read the current caller from an `AsyncLocal` and add the header, restricted to the MCP server's own origin.

The handler applies to *every* request the client sends, including the ones you do not make yourself: `McpClient.CreateAsync` performs the initialise handshake, and `ListToolsAsync` lists the tools, before any user's run has started. Against a server that requires authentication those requests fail unless a token is already in place, so connect and list tools under a service (bootstrap) token, then switch to the caller's token for each run. The example runs an in-process MCP server that rejects requests without a known bearer token and reports which token each tool call carried:

```csharp
using System.ComponentModel;
using System.Net.Http.Headers;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;
using ModelContextProtocol.Server;
using OpenAI;

// --- A local MCP server that requires a bearer token (stand-in for a real protected server) ---
var builder = WebApplication.CreateBuilder(args);
builder.Logging.SetMinimumLevel(LogLevel.Warning);
builder.Services.AddHttpContextAccessor();
builder.Services.AddMcpServer().WithHttpTransport().WithTools<AccountTools>();
var server = builder.Build();
HashSet<string> validTokens = ["service-token", "token-a", "token-b"];
server.Use(async (context, next) =>
{
    string? token = context.Request.Headers.Authorization.ToString() is { } h && h.StartsWith("Bearer ") ? h["Bearer ".Length..] : null;
    Console.WriteLine($"[server] {context.Request.Method} with token {token ?? "(none)"}");
    if (token is null || !validTokens.Contains(token))
    {
        context.Response.StatusCode = StatusCodes.Status401Unauthorized;
        return;
    }
    await next(context);
});
server.MapMcp("/mcp");
await server.StartAsync();
var serverUri = new Uri($"{server.Urls.First()}/mcp");

// --- The client side ---
using var http = new HttpClient(new CallerTokenHandler(serverUri) { InnerHandler = new HttpClientHandler() });

// Initialisation and tool listing run before any user is involved: use the service identity.
CallerToken.Current = "service-token";
await using McpClient mcp = await McpClient.CreateAsync(
    new HttpClientTransport(new() { Endpoint = serverUri, TransportMode = HttpTransportMode.StreamableHttp }, http));
IList<McpClientTool> tools = await mcp.ListToolsAsync();

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(tools: [.. tools]);

foreach (var (user, token) in new[] { ("alice", "token-a"), ("bob", "token-b") })
{
    Console.WriteLine($"{user}: {await RunAsCallerAsync(token, () => agent.RunAsync("Which account am I?"))}");
}

await server.StopAsync();

// Set the caller's token for one run only, and restore the previous one afterwards.
static async Task<T> RunAsCallerAsync<T>(string token, Func<Task<T>> run)
{
    string? previous = CallerToken.Current;
    CallerToken.Current = token;   // from the authenticated request in a real app
    try
    {
        return await run();
    }
    finally
    {
        CallerToken.Current = previous;
    }
}

public static class CallerToken
{
    private static readonly AsyncLocal<string?> s_token = new();
    public static string? Current { get => s_token.Value; set => s_token.Value = value; }
}

public sealed class CallerTokenHandler(Uri server) : DelegatingHandler
{
    private readonly string _origin = server.GetLeftPart(UriPartial.Authority);

    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        if (CallerToken.Current is { } token
            && request.RequestUri is { } uri
            && string.Equals(uri.GetLeftPart(UriPartial.Authority), _origin, StringComparison.OrdinalIgnoreCase))
        {
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
        }
        return base.SendAsync(request, cancellationToken);
    }
}

[McpServerToolType]
public sealed class AccountTools(IHttpContextAccessor http)
{
    [McpServerTool(Name = "whoami", ReadOnly = true), Description("Describe the calling account.")]
    public string WhoAmI() => $"called with {http.HttpContext?.Request.Headers.Authorization}";
}
```

```text
[server] POST with token service-token
[server] POST with token service-token
[server] POST with token token-a
alice: Here is what the tool returned: {
  "$type": "text",
  "text": "called with Bearer token-a"
}
[server] POST with token token-b
bob: Here is what the tool returned: {
  "$type": "text",
  "text": "called with Bearer token-b"
}
```

The first two requests are the initialise handshake and the tool listing, sent with the service token; each tool call then carries its caller's token.

Without the `CallerToken.Current = "service-token"` line, `CreateAsync` fails with a 401 before either run starts. Two more things follow from the handler applying to every request: the tool list is whatever the *service* identity is allowed to see, so make sure it does not depend on the caller (or list tools per caller and pass them as [per-run tools](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_tools/#per-run-tools)); and a stateful server session is shared by all callers, which is one more reason to prefer a stateless server when tokens vary per request. Setting the token in a helper that restores the previous value keeps it from leaking into later work on the same async flow.

### OAuth

For servers that implement MCP's OAuth 2.1 flow, set `HttpClientTransportOptions.OAuth`. The client discovers the authorisation server, can register itself dynamically, and calls your `AuthorizationCallbackHandler` with the authorisation URL; you open a browser, catch the redirect and return the code. Tokens are cached in `TokenCache` (in memory by default).

```csharp
using ModelContextProtocol.Authentication;
using ModelContextProtocol.Client;

var transport = new HttpClientTransport(new HttpClientTransportOptions
{
    Endpoint = new Uri("http://localhost:7071/"),
    Name = "secure-weather",
    OAuth = new ClientOAuthOptions
    {
        RedirectUri = new Uri("http://localhost:1179/callback"),
        DynamicClientRegistration = new DynamicClientRegistrationOptions { ClientName = "my-agent" },
        Scopes = ["weather.read"],
        AuthorizationCallbackHandler = async (context, cancellationToken) =>
        {
            Console.WriteLine($"Open {context.AuthorizationUri} and sign in.");
            // Listen on context.RedirectUri (HttpListener or an ASP.NET endpoint), then:
            string code = await WaitForRedirectAsync(context.RedirectUri, cancellationToken);
            return new AuthorizationResult { Code = code, State = "state-from-the-redirect" };
        },
    },
});

await using McpClient mcp = await McpClient.CreateAsync(transport);
Console.WriteLine((await mcp.ListToolsAsync()).Count);

static Task<string> WaitForRedirectAsync(Uri redirect, CancellationToken ct) => Task.FromResult("code");
```

Return the `state` (and `iss`, when present) exactly as the redirect carried them; the client checks them. The older `AuthorizationRedirectDelegate` is obsolete because it cannot pass the issuer back.

## Elicitation

A server can ask the user for information in the middle of a tool call (a missing parameter, a confirmation). Handle `McpClientHandlers.ElicitationHandler`; whatever you return goes back to the server.

```csharp
using System.Text.Json;
using ModelContextProtocol.Client;
using ModelContextProtocol.Protocol;

var options = new McpClientOptions
{
    Handlers = new McpClientHandlers
    {
        ElicitationHandler = (request, cancellationToken) =>
        {
            Console.WriteLine($"The server asks: {request?.Message}");
            Console.Write("> ");
            string answer = Console.ReadLine() ?? "";
            return ValueTask.FromResult(new ElicitResult
            {
                Action = "accept",
                Content = new Dictionary<string, JsonElement> { ["answer"] = JsonSerializer.SerializeToElement(answer) },
            });
        },
    },
};

await using McpClient mcp = await McpClient.CreateAsync(
    new StdioClientTransport(new() { Name = "server", Command = "my-mcp-server" }), options);
```

The handler runs on the thread that is waiting for the tool result, inside the agent's run; in a web app, route the question to the user over your own channel (SignalR, a pending-question record) and complete the handler when they answer.

Sampling (a server asking *your* model to generate text), roots and server logging are deprecated as of MCP specification 2026-07-28 (SEP-2577). In `ModelContextProtocol` 2.2.0 `McpClientHandlers.SamplingHandler`, `RootsHandler`, `CreateSamplingHandler` and `SetLoggingLevelAsync` still work but are `[Obsolete]` with diagnostic `MCP9005`; do not build new features on them.

## Long-running MCP tasks

The MCP Tasks extension lets a server answer a slow tool call with a task handle instead of blocking. `Microsoft.Agents.AI.Mcp` (alpha) provides `ListAgentToolsWithTasksAsync`, which returns tools that opt into the extension, poll a task to completion, and return its result to the loop as if the call had been synchronous. Against a server without task support they behave like ordinary tools.

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Mcp;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using ModelContextProtocol.Client;
using ModelContextProtocol.Extensions.Tasks;
using ModelContextProtocol.Server;
using OpenAI;

if (args is ["--server"])
{
    var server = Host.CreateApplicationBuilder();
    server.Logging.ClearProviders();
    server.Services
        .AddMcpServer(o => o.ServerInfo = new() { Name = "analyser", Version = "1.0.0" })
        .WithStdioServerTransport()
        .WithTools<AnalysisTools>()
        .WithTasks(new InMemoryMcpTaskStore());   // server side of the Tasks extension
    await server.Build().RunAsync();
    return;
}

await using McpClient mcp = await McpClient.CreateAsync(new StdioClientTransport(new()
{
    Name = "analyser",
    Command = "dotnet",
    Arguments = [typeof(AnalysisTools).Assembly.Location, "--server"],
}));

IReadOnlyList<AIFunction> tools = await mcp.ListAgentToolsWithTasksAsync(new McpTaskOptions
{
    RemoteCancellationTimeout = TimeSpan.FromSeconds(5),
    CancelRemoteTaskOnLocalCancellation = true,
});

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "Use the analysis tool.", tools: [.. tools]);

var watch = System.Diagnostics.Stopwatch.StartNew();
Console.WriteLine(await agent.RunAsync("Analyse the dataset 'sales-q1'."));
Console.WriteLine($"took {watch.Elapsed.TotalSeconds:F1}s");

[McpServerToolType]
public sealed class AnalysisTools
{
    [McpServerTool(Name = "analyse_dataset"), Description("Analyse a dataset. Takes a few seconds.")]
    public static async Task<string> Analyse(string dataset, CancellationToken cancellationToken)
    {
        await Task.Delay(TimeSpan.FromSeconds(2), cancellationToken);
        return $"{dataset}: 12,403 rows, 3 anomalies.";
    }
}
```

| `McpTaskOptions` | Default | Meaning |
| --- | --- | --- |
| `RemoteCancellationTimeout` | 5 s | Time allowed for the best-effort `tasks/cancel` request. |
| `CancelRemoteTaskOnLocalCancellation` | `true` | Send `tasks/cancel` when the run is cancelled. |
| `MinimumPollingInterval`, `MaximumPollingInterval` | 10 ms, runtime maximum | Bounds on the poll interval the server asks for. |
| `MaxConsecutiveStuckPolls` | 60 | Polls in `input_required` state with no new input request before the task is treated as stuck. |
| `MaxTotalInputRequests` | 100 | Cap on distinct input requests one task may raise. |

The run still waits for the result; tasks make the wait robust (polling instead of one long-held request, cancellation that reaches the server), not asynchronous. To let a person leave and come back, use [background responses](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_providers/#background-responses) or a durable workflow.

## Skills from an MCP server

`Microsoft.Agents.AI.Mcp` also lets an MCP server publish agent skills (instruction documents, with optional resources and scripts) as MCP resources. `AgentSkillsProviderBuilder.UseMcpSkills(mcpClient)` reads the server's skill index and builds an `AgentSkillsProvider` that offers them to the model on demand. `AgentMcpSkillsSourceOptions` controls how archived skills are unpacked (`ArchiveSkillsDirectory`, `ArchiveMaxFileCount`, `ArchiveMaxSizeBytes` and so on).

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;
using OpenAI;

await using McpClient skillsServer = await McpClient.CreateAsync(new HttpClientTransport(new()
{
    Endpoint = new Uri(Environment.GetEnvironmentVariable("SKILLS_MCP_URL") ?? "http://localhost:5080/mcp"),
}));

AgentSkillsProvider skills = new AgentSkillsProviderBuilder()
    .UseMcpSkills(skillsServer)
    .Build();

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(new ChatClientAgentOptions
    {
        ChatOptions = new() { Instructions = "Use your skills when they apply." },
        AIContextProviders = [skills],
    })
    .AsBuilder()
    .UseToolApproval(new ToolApprovalAgentOptions { AutoApprovalRules = [AgentSkillsProvider.AllToolsAutoApprovalRule] })
    .Build();

Console.WriteLine(await agent.RunAsync("Convert 26.2 miles to kilometres."));
```

The skills provider's own tools (loading a skill, reading its resources) require approval by default; `AgentSkillsProvider.AllToolsAutoApprovalRule` approves just those, through `ToolApprovalAgent`.

## Hosted MCP

With `HostedMcpServerTool`, the model provider connects to the MCP server, lists its tools and calls them inside a single model call. Your process sends only the server's address. `AllowedTools` restricts which tools the provider may use, `Headers` carries credentials to the server, and `ApprovalMode` decides whether a call needs your approval first.

```csharp
#pragma warning disable OPENAI001
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Responses;

var learn = new HostedMcpServerTool("microsoft_learn", "https://learn.microsoft.com/api/mcp")
{
    AllowedTools = ["microsoft_docs_search"],
    ApprovalMode = HostedMcpServerToolApprovalMode.AlwaysRequire,
};

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetResponsesClient()
    .AsAIAgent(model: "%%model=openai%%", instructions: "Answer from Microsoft Learn only.", tools: [learn]);

AgentSession session = await agent.CreateSessionAsync();
AgentResponse response = await agent.RunAsync("Summarise how MCP tools are approved in Foundry.", session);

List<ToolApprovalRequestContent> requests = response.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>().ToList();
while (requests.Count > 0)
{
    var answers = requests.Select(request =>
    {
        var call = (McpServerToolCallContent)request.ToolCall;
        Console.WriteLine($"{call.ServerName}.{call.Name}({string.Join(", ", call.Arguments?.Select(a => $"{a.Key}={a.Value}") ?? [])})");
        return new ChatMessage(ChatRole.User, [request.CreateResponse(approved: true)]);
    }).ToList();

    response = await agent.RunAsync(answers, session);
    requests = response.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>().ToList();
}

Console.WriteLine(response.Text);
```

| `ApprovalMode` | Effect |
| --- | --- |
| `HostedMcpServerToolApprovalMode.AlwaysRequire` | Every call comes back as a `ToolApprovalRequestContent` whose `ToolCall` is a `McpServerToolCallContent`. |
| `HostedMcpServerToolApprovalMode.NeverRequire` | The provider calls tools without asking. |
| `HostedMcpServerToolApprovalMode.RequireSpecific(always, never)` | Per-tool lists. |

Results come back as `McpServerToolResultContent` in the response messages. The server must be reachable from the provider's network, and anything you put in `Headers` is sent to the provider as part of the request.

For Foundry, use the same `HostedMcpServerTool` with a code-defined agent, or `FoundryAITool.CreateMcpTool(serverLabel, serverUri, ..., projectConnectionId)` to use credentials stored in a Foundry project connection instead of headers. For an agent defined in Foundry, add `ResponseTool.CreateMcpTool(...)` to the `DeclarativeAgentDefinition.Tools`. `FoundryAITool.CreateHostedMcpToolbox(name)` (a `HostedMcpToolboxAITool`) points at a Foundry MCP toolbox, a curated, versioned bundle of MCP servers managed in the project.

## Exposing tools and agents as MCP servers

### A stdio server

`AddMcpServer()` registers an MCP server in any generic host. `WithTools<T>()` registers the `[McpServerTool]` methods of a `[McpServerToolType]` class; `WithToolsFromAssembly()` scans for them. Tool methods can take `CancellationToken`, `IServiceProvider` and services from DI as parameters, which are not part of the tool's schema.

| `[McpServerTool]` property | Meaning |
| --- | --- |
| `Name`, `Title` | Tool name (defaults to the method name) and display title |
| `ReadOnly`, `Destructive`, `Idempotent`, `OpenWorld` | Annotations sent to clients as hints |
| `UseStructuredContent`, `OutputSchemaType` | Return structured JSON with an output schema |

The examples above already include stdio servers. Two rules matter for stdio: nothing but the protocol may be written to stdout (clear the console logger, or log to stderr), and the server exits when its input closes.

### An agent as an MCP tool

Any agent becomes one MCP tool through `AsAIFunction()` and `McpServerTool.Create`. Clients such as VS Code, Claude Desktop or Copilot Studio can then call it.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using ModelContextProtocol.Server;
using OpenAI;

AIAgent joker = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(
        instructions: "You tell short, clean jokes.",
        name: "joker",
        description: "Tells a joke about any topic.");

HostApplicationBuilder builder = Host.CreateEmptyApplicationBuilder(settings: null);
builder.Logging.ClearProviders();
builder.Services
    .AddMcpServer(o => o.ServerInfo = new() { Name = "joker-server", Version = "1.0.0" })
    .WithStdioServerTransport()
    .WithTools([McpServerTool.Create(joker.AsAIFunction())]);

await builder.Build().RunAsync();
```

The tool has one string parameter, `query`, and returns the agent's final text. Each call runs the agent without a session, so it remembers nothing between calls; pass `AsAIFunction(session: ...)` for a stateful (and then shared) conversation.

### Over HTTP with ASP.NET Core

`ModelContextProtocol.AspNetCore` adds `WithHttpTransport()` and `MapMcp()`, which serves Streamable HTTP (and, if enabled, the legacy SSE endpoints) on a route. The example serves a tool over HTTP and then connects to itself as a client, so it can be run as it is.

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;
using ModelContextProtocol.Server;
using OpenAI;

var builder = WebApplication.CreateBuilder(args);
builder.Logging.SetMinimumLevel(LogLevel.Warning);
builder.Services
    .AddMcpServer()
    .WithHttpTransport()          // stateless by default: any instance can serve any request
    .WithTools<InventoryTools>();

var app = builder.Build();
app.MapMcp("/mcp");                                      // add .RequireAuthorization() to protect it
await app.StartAsync();

string url = app.Urls.First();
await using McpClient mcp = await McpClient.CreateAsync(new HttpClientTransport(new()
{
    Endpoint = new Uri($"{url}/mcp"),
    TransportMode = HttpTransportMode.StreamableHttp,
}));

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You check stock levels.", tools: [.. await mcp.ListToolsAsync()]);

Console.WriteLine(await agent.RunAsync("How many blue mugs are in stock?"));
await app.StopAsync();

[McpServerToolType]
public sealed class InventoryTools
{
    [McpServerTool(Name = "stock_level", ReadOnly = true), Description("Units in stock for a product.")]
    public static int StockLevel([Description("Product name")] string product) => product.Length * 3;
}
```

Since the 2026-07-28 protocol revision, and in `ModelContextProtocol.AspNetCore` 2.2.0, the HTTP transport is stateless by default (`HttpServerTransportOptions.SessionMode = HttpServerSessionMode.Stateless`), so the server scales out behind a load balancer with no sticky sessions. `ConfigureSessionOptions` lets you vary the server per request, for example per-tenant tool lists read from `HttpContext`. The stateful-mode settings (`SessionMode = Stateful`, `IdleTimeout`, `MaxIdleSessionCount`, `EventStreamStore`, `WithDistributedCacheEventStreamStore()`) are kept only for clients on the 2025-11-25 revision or earlier and are `[Obsolete]` (`MCP9006`); `EnableLegacySse` is obsolete too (`MCP9004`). Protect the endpoint with ASP.NET Core authentication and `MapMcp(...).RequireAuthorization()`; tool methods can take `HttpContext` through `IHttpContextAccessor` to see who is calling.

To host an agent for non-MCP clients (OpenAI-compatible endpoints, A2A, AG-UI), see the [hosting page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/).

## Lifecycle and failures

- **One client per server, shared.** `McpClient` is thread-safe. Create it at start-up (or lazily, once), keep it for the life of the process, and dispose it at shutdown (`await using`, or register it as a singleton that the container disposes).
- **Connection failures surface at `CreateAsync`.** A stdio command that does not exist, or an HTTP server that is down, throws there, before the agent is involved. Catch it and decide whether the agent should run without those tools.
- **A server that dies mid-run** makes its tool calls throw; the function loop treats that like any tool error (the model sees "Error: Function failed."), and after `MaximumConsecutiveErrorsPerRequest` failures the exception reaches your code. Recreate the client to reconnect; existing `McpClientTool` objects are bound to the old client.
- **Tool lists can change.** Servers may send `notifications/tools/list_changed`. The tools you passed to the agent do not update themselves; list again and rebuild the agent (or pass the new list as per-run tools).
- **Treat MCP output as untrusted.** Tool results go into the prompt. A malicious or compromised server can inject instructions; keep side-effecting tools behind approval and do not give an agent that reads untrusted content tools that can exfiltrate data.

## Differences from Python

| Topic | Python | .NET |
| --- | --- | --- |
| MCP client | Framework classes `MCPStdioTool`, `MCPStreamableHTTPTool` that own the connection | The official SDK's `McpClient`; its `McpClientTool`s are `AIFunction`s |
| Connection lifetime | `async with` on the tool, or the agent connects it | You create and dispose the `McpClient` |
| Filtering | `allowed_tools` on the MCP tool | LINQ over `ListToolsAsync()`, `WithName` to rename |
| Approval | `approval_mode` on the MCP tool | `ApprovalRequiredAIFunction` around chosen tools; `HostedMcpServerToolApprovalMode` for hosted MCP |
| Long-running tasks | Removed from the Python package | `Microsoft.Agents.AI.Mcp`: `ListAgentToolsWithTasksAsync` (alpha) |
| Exposing an agent | `agent.as_mcp_server()` | `McpServerTool.Create(agent.AsAIFunction())` in an MCP server host |
| Sampling callbacks | Supported | Deprecated by the 2026-07-28 MCP specification; `[Obsolete]` in the C# SDK |

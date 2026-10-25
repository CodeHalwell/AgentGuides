---
title: "Microsoft Agent Framework (.NET) — Middleware"
description: "Agent-run, function-invocation and chat-client middleware in .NET: AIAgentBuilder and DelegatingAIAgent, function callbacks, ChatClientBuilder and DelegatingChatClient, per-run middleware, ordering, short-circuiting, streaming, the built-in Microsoft.Extensions.AI decorators, retries, timeouts and error handling."
framework: "microsoft-agent-framework"
language: "dotnet"
sidebar:
  label: "Middleware"
  order: 6
---

Middleware is how you add behaviour around an agent without changing it: logging, guardrails, redaction, caching, quotas, retries. .NET has three levels, each wrapping a different call:

| Level | Wraps | Added with | Sees | Use for |
| --- | --- | --- | --- | --- |
| **Agent run** | One `RunAsync` / `RunStreamingAsync` call | `agent.AsBuilder().Use(...)`, or a `DelegatingAIAgent` subclass | Input messages, session, run options, the final `AgentResponse` | Input and output guardrails, PII redaction, auth and quotas, audit logs, whole-run retries |
| **Function invocation** | One tool call inside the loop | `agent.AsBuilder().Use((agent, context, next, ct) => ...)` | `FunctionInvocationContext`: the function, its arguments, the conversation | Argument validation, per-tool policy, result rewriting, tool timing |
| **Chat client** | One model call (there may be many per run) | `ChatClientBuilder.Use(...)` or a `DelegatingChatClient`, via `clientFactory` or before `AsAIAgent` | The exact messages and `ChatOptions` sent to the model, the raw `ChatResponse` | Prompt shaping, caching, token accounting, model-level retries, content filtering of each call |

Agent middleware is the Agent Framework's own (`Microsoft.Agents.AI`); chat-client middleware is `Microsoft.Extensions.AI`'s, so everything written for `IChatClient` (logging, OpenTelemetry, distributed caching, reducers) applies. Function middleware is the framework's, but it is built on `Microsoft.Extensions.AI`'s `FunctionInvocationContext`.

Every C# block was compiled against `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% with `Microsoft.Extensions.AI` 10.11.0 (checked %%versions-checked%%) and run against a local stand-in for the OpenAI API, which calls the first tool it is offered, then echoes the tool's result, and otherwise replies "Mock reply to: ...". The outputs shown are from those runs.

## Agent-run middleware

### `AIAgentBuilder.Use`

`agent.AsBuilder()` returns an `AIAgentBuilder`; each `Use` adds a layer and `Build()` returns the wrapped agent. The commonest form takes a non-streaming and a streaming delegate. Each receives the messages, session, options, the **inner agent** and a cancellation token, and decides whether and how to call the inner agent.

```csharp
using System.Text.RegularExpressions;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You are a support assistant.", name: "support");

AIAgent guarded = agent.AsBuilder()
    .Use(runFunc: RedactAsync, runStreamingFunc: null)
    .Build();

Console.WriteLine(await guarded.RunAsync("My card is 4111 1111 1111 1111, why was it declined?"));

static async Task<AgentResponse> RedactAsync(
    IEnumerable<ChatMessage> messages, AgentSession? session, AgentRunOptions? options, AIAgent inner, CancellationToken ct)
{
    // Before: rewrite the input.
    List<ChatMessage> clean = messages.Select(m => new ChatMessage(m.Role, Redact(m.Text))).ToList();

    AgentResponse response = await inner.RunAsync(clean, session, options, ct);

    // After: rewrite the output.
    response.Messages = response.Messages.Select(m => new ChatMessage(m.Role, Redact(m.Text)) { AuthorName = m.AuthorName }).ToList();
    return response;

    static string Redact(string text) => Regex.Replace(text, @"\b(?:\d[ -]?){13,16}\b", "[card]");
}
```

```text
Mock reply to: My card is [card], why was it declined?
```

Pass the session through unchanged unless you mean to replace it: it carries the conversation history and any approval state. Rewriting messages this way flattens them to text, which drops images, tool calls and other content; for anything but plain text, copy the message and change only its `TextContent` items.

### Streaming, and what happens when you leave a delegate out

The two delegates are independent. If you pass only `runFunc`, streaming calls go through `runFunc` as well, so the caller still gets updates, but only once the whole response is ready: your middleware has quietly turned streaming off. If you pass only `runStreamingFunc`, non-streaming calls are served by it and the updates are collected into a response. Provide both when streaming matters:

```csharp
using System.Runtime.CompilerServices;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You are concise.");

AIAgent timed = agent.AsBuilder()
    .Use(
        runFunc: async (messages, session, options, inner, ct) =>
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();
            AgentResponse response = await inner.RunAsync(messages, session, options, ct);
            Console.WriteLine($"[run took {watch.ElapsedMilliseconds} ms, {response.Usage?.TotalTokenCount} tokens]");
            return response;
        },
        runStreamingFunc: TimedStreamAsync)
    .Build();

Console.WriteLine(await timed.RunAsync("Hello"));
await foreach (AgentResponseUpdate update in timed.RunStreamingAsync("Hello again"))
{
    Console.Write(update.Text);
}
Console.WriteLine();

static async IAsyncEnumerable<AgentResponseUpdate> TimedStreamAsync(
    IEnumerable<ChatMessage> messages, AgentSession? session, AgentRunOptions? options, AIAgent inner,
    [EnumeratorCancellation] CancellationToken ct)
{
    var watch = System.Diagnostics.Stopwatch.StartNew();
    bool first = true;
    await foreach (AgentResponseUpdate update in inner.RunStreamingAsync(messages, session, options, ct))
    {
        if (first)
        {
            Console.Write($"[first token after {watch.ElapsedMilliseconds} ms] ");
            first = false;
        }
        yield return update;
    }
}
```

When you only need to act before and after, not touch the response, the single-delegate overload covers both paths. Its `next` delegate returns a `Task`, not the response:

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

var calls = new Dictionary<string, int>();
string currentUser = "alice";   // from the authenticated request in a real app

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You are helpful.")
    .AsBuilder()
    .Use(async (messages, session, options, next, ct) =>
    {
        calls[currentUser] = calls.GetValueOrDefault(currentUser) + 1;
        if (calls[currentUser] > 2)
        {
            throw new InvalidOperationException("Quota exceeded.");   // stops the run before the model is called
        }
        await next(messages, session, options, ct);
    })
    .Build();

for (int i = 1; i <= 3; i++)
{
    try
    {
        await agent.RunAsync($"Request {i}");
        Console.WriteLine($"request {i}: ok");
    }
    catch (InvalidOperationException ex)
    {
        Console.WriteLine($"request {i}: {ex.Message}");
    }
}
```

```text
request 1: ok
request 2: ok
request 3: Quota exceeded.
```

### Short-circuiting

A middleware that does not call the inner agent ends the run with whatever it returns. This is how a guardrail refuses a request without spending a model call:

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

string[] blocked = ["password", "social security"];

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You are helpful.", name: "assistant")
    .AsBuilder()
    .Use(
        runFunc: (messages, session, options, inner, ct) =>
        {
            string text = string.Join(" ", messages.Select(m => m.Text));
            if (blocked.Any(b => text.Contains(b, StringComparison.OrdinalIgnoreCase)))
            {
                return Task.FromResult(new AgentResponse(
                    new ChatMessage(ChatRole.Assistant, "I can't help with that.") { AuthorName = inner.Name }));
            }
            return inner.RunAsync(messages, session, options, ct);
        },
        runStreamingFunc: null)
    .Build();

AgentSession session = await agent.CreateSessionAsync();
Console.WriteLine(await agent.RunAsync("What's the admin password?", session));
Console.WriteLine(await agent.RunAsync("What's the capital of Peru?", session));
Console.WriteLine(session.TryGetInMemoryChatHistory(out var history) ? $"history: {history.Count} messages" : "");
```

```text
I can't help with that.
Mock reply to: What's the capital of Peru?
history: 2 messages
```

The refused turn never reached the inner agent, so it is not in the session's history. That is usually what you want; if you need the refusal recorded, add the messages to the history yourself.

### `DelegatingAIAgent` for reusable middleware

For middleware you want to unit-test, configure, or register in DI, derive from `DelegatingAIAgent`. Override `RunCoreAsync` and `RunCoreStreamingAsync`; the base class forwards everything else (sessions, serialisation, `GetService`, `Name`) to the inner agent.

```csharp
using System.Runtime.CompilerServices;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You are helpful.", name: "assistant");

AIAgent audited = agent.AsBuilder()
    .Use(inner => new AuditingAgent(inner, entry => Console.WriteLine($"[audit] {entry}")))
    .Build();

Console.WriteLine(await audited.RunAsync("Hi"));
Console.WriteLine(audited.GetService<AuditingAgent>() is not null);   // decorators are discoverable

public sealed class AuditingAgent(AIAgent inner, Action<string> write) : DelegatingAIAgent(inner)
{
    protected override async Task<AgentResponse> RunCoreAsync(
        IEnumerable<ChatMessage> messages, AgentSession? session = null, AgentRunOptions? options = null, CancellationToken cancellationToken = default)
    {
        write($"{this.Name} <- {string.Join(" | ", messages.Select(m => m.Text))}");
        AgentResponse response = await this.InnerAgent.RunAsync(messages, session, options, cancellationToken);
        write($"{this.Name} -> {response.Text}");
        return response;
    }

    protected override async IAsyncEnumerable<AgentResponseUpdate> RunCoreStreamingAsync(
        IEnumerable<ChatMessage> messages, AgentSession? session = null, AgentRunOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        write($"{this.Name} <- (stream) {string.Join(" | ", messages.Select(m => m.Text))}");
        await foreach (AgentResponseUpdate update in this.InnerAgent.RunStreamingAsync(messages, session, options, cancellationToken))
        {
            yield return update;
        }
    }
}
```

`Use(Func<AIAgent, IServiceProvider, AIAgent>)` gives the factory the service provider passed to `Build(services)`, so a decorator can resolve its dependencies (an `ILogger`, a policy service) from DI.

### Built-in agent middleware

| Extension | Adds | Notes |
| --- | --- | --- |
| `UseLogging(loggerFactory)` | `LoggingAgent` | Logs each run at `Debug`; message contents only at `Trace`. |
| `UseOpenTelemetry(sourceName)` | `OpenTelemetryAgent` | `invoke_agent` spans and metrics per the GenAI semantic conventions; also switches on the agent's chat-level telemetry. See the [observability page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_observability/#tracing-an-agent). |
| `UseToolApproval(options)` | `ToolApprovalAgent` | Standing approvals; see [tools](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_tools/#standing-approvals-with-toolapprovalagent). |
| `UseAIContextProviders(providers)` | A provider-running agent | Runs `MessageAIContextProvider`s around *any* agent, not just `ChatClientAgent`. |
| `UseClientHeaders()` (Foundry, experimental `MEAI001`) | A header-forwarding decorator | Sends HTTP headers set per run with `ChatOptions.WithClientHeader(name, value)`. |

### Ordering

The first `Use` is the outermost layer: on the way in, layers run in the order you added them; on the way out, in reverse.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You are helpful.")
    .AsBuilder()
    .Use(Layer("A"), null)
    .Use(Layer("B"), null)
    .Use(Layer("C"), null)
    .Build();

await agent.RunAsync("Hi");

static Func<IEnumerable<ChatMessage>, AgentSession?, AgentRunOptions?, AIAgent, CancellationToken, Task<AgentResponse>> Layer(string name) =>
    async (messages, session, options, inner, ct) =>
    {
        Console.WriteLine($"{name} in");
        AgentResponse response = await inner.RunAsync(messages, session, options, ct);
        Console.WriteLine($"{name} out");
        return response;
    };
```

```text
A in
B in
C in
C out
B out
A out
```

Put cheap checks that may refuse (auth, quotas, input guardrails) outermost, so a refused request does no work, and put telemetry outside retries if you want one span per user request, inside if you want one per attempt.

## Function-invocation middleware

The overload of `Use` that takes `(AIAgent agent, FunctionInvocationContext context, next, CancellationToken)` runs around every tool call the agent makes. `context` is `Microsoft.Extensions.AI`'s `FunctionInvocationContext`: `Function`, `Arguments` (mutable), `CallContent`, `Messages`, `Options`, `Iteration`, `FunctionCallIndex`, `FunctionCount`, `IsStreaming` and `Terminate`. Call `next(context, ct)` to run the tool (or the next middleware); return its result or your own.

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You answer weather questions.", tools: [AIFunctionFactory.Create(Weather.Get)])
    .AsBuilder()
    .Use(LogCalls)
    .Use(ValidateCity)
    .Build();

Console.WriteLine(await agent.RunAsync("What's the weather in Paris?"));

static async ValueTask<object?> LogCalls(AIAgent agent, FunctionInvocationContext context,
    Func<FunctionInvocationContext, CancellationToken, ValueTask<object?>> next, CancellationToken ct)
{
    Console.WriteLine($"-> {context.Function.Name}({string.Join(", ", context.Arguments.Select(a => $"{a.Key}={a.Value}"))})");
    object? result = await next(context, ct);
    Console.WriteLine($"<- {result}");
    return result;
}

static async ValueTask<object?> ValidateCity(AIAgent agent, FunctionInvocationContext context,
    Func<FunctionInvocationContext, CancellationToken, ValueTask<object?>> next, CancellationToken ct)
{
    if (context.Function.Name == "Get" && context.Arguments.TryGetValue("city", out object? value))
    {
        string city = value?.ToString()?.Trim() ?? "";
        if (city.Length is 0 or > 40)
        {
            return "Invalid city name; ask the user to check it.";   // short-circuit: the tool does not run
        }
        context.Arguments["city"] = city;                             // normalise before the tool sees it
    }
    return await next(context, ct);
}

public static class Weather
{
    [Description("Get the weather for a city.")]
    public static string Get(string city) => $"{city}: 18°C, light wind.";
}
```

```text
-> Get(city=Paris)
<- Paris: 18°C, light wind.
Here is what the tool returned: "Paris: 18°C, light wind."
```

How it behaves:

- **Only for `ChatClientAgent`** (and decorators around one). It works by injecting a chat-client factory into the run options, so the inner agent must accept `ChatClientAgentRunOptions`; with run options of another type it throws `NotSupportedException`.
- **Order** is registration order: the first function middleware added is the first to see the call.
- **Covers every function the run uses**: the agent's tools, per-run tools, tools added by context providers and tools added during the loop. Hosted tools are not covered; they run on the provider's servers.
- **Exceptions** thrown by a function middleware are treated like an exception from the tool: the model sees "Error: Function failed." and the loop carries on, subject to `MaximumConsecutiveErrorsPerRequest`.
- **`context.Terminate = true`** ends the loop after this iteration, as it does when set inside a tool.
- Function middleware runs *after* approval: a call that is waiting for approval has not reached it yet, and a rejected call never does.

`FunctionInvocationContextExtensions.WrapWithPendingMiddleware(context, function)` (experimental `MAAI001`) wraps a function with the middleware that has not yet run for the current call. It is for middleware that swaps in a different function and still wants the rest of the chain applied.

## Chat-client middleware

Chat-client middleware wraps each model call, below the function loop. A run with two tool calls makes three model calls, and chat middleware sees all three, including the tool results being sent back.

There are three places to add it:

1. **Build the client first**, then make the agent: `chatClient.AsBuilder().Use(...).Build().AsAIAgent(...)` or `chatClientBuilder.BuildAIAgent(...)`. The agent adds its own decorators (approvals, function loop) *outside* yours, so your middleware sits between the loop and the model.
2. **`clientFactory:`** on the provider extensions (`AsAIAgent` on `ChatClient`, `ResponsesClient`, `AnthropicClient`, `AIProjectClient`), which wraps the provider's `IChatClient` before the agent is built. Same position as 1.
3. **Per run**, with `ChatClientAgentRunOptions.ChatClientFactory`, which wraps the agent's *whole* pipeline (outside the function loop) for one run.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

int modelCalls = 0;

IChatClient chatClient = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsBuilder()
    .Use(
        getResponseFunc: async (messages, options, inner, ct) =>
        {
            modelCalls++;
            Console.WriteLine($"model call {modelCalls}: {messages.Count()} messages, {options?.Tools?.Count ?? 0} tools");
            ChatResponse response = await inner.GetResponseAsync(messages, options, ct);
            Console.WriteLine($"  usage: {response.Usage?.InputTokenCount} in / {response.Usage?.OutputTokenCount} out");
            return response;
        },
        getStreamingResponseFunc: null)
    .Build();

AIAgent agent = chatClient.AsAIAgent(
    instructions: "You tell the time.",
    tools: [AIFunctionFactory.Create(() => "12:00", "now", "Current time.")]);

Console.WriteLine(await agent.RunAsync("What time is it?"));
```

```text
model call 1: 1 messages, 1 tools
  usage: 12 in / 8 out
model call 2: 3 messages, 1 tools
  usage: 12 in / 8 out
Here is what the tool returned: "12:00"
```

The first call sends the user's message (the instructions travel separately, in `ChatOptions.Instructions`, which the provider adapter turns into a system message); the second adds the assistant's tool call and the tool's result. `ChatClientBuilder.Use` has the same shapes as `AIAgentBuilder.Use`: separate non-streaming and streaming delegates (leaving one out makes it fall back to the other, with the same loss of incremental streaming), a single shared delegate whose `next` returns `Task`, and `Use(inner => new MyDelegatingChatClient(inner))` for classes derived from `DelegatingChatClient`.

### Per-run middleware

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You are helpful.");

var debugRun = new ChatClientAgentRunOptions
{
    ChatClientFactory = pipeline => pipeline.AsBuilder()
        .Use(async (messages, options, next, ct) =>
        {
            Console.WriteLine($"[debug] sending {messages.Count()} messages");
            await next(messages, options, ct);
        })
        .Build(),
};

Console.WriteLine(await agent.RunAsync("Only this run is traced.", options: debugRun));
Console.WriteLine(await agent.RunAsync("This one is not."));
```

### Built-in chat-client middleware

`Microsoft.Extensions.AI` ships these as `ChatClientBuilder` extensions:

| Extension | Purpose |
| --- | --- |
| `UseFunctionInvocation(configure)` | The tool loop. The agent reuses yours if present ([tool loop settings](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_tools/#the-tool-loop)). |
| `UseLogging(loggerFactory)` | Logs requests and responses. |
| `UseOpenTelemetry(loggerFactory, sourceName)` | `chat` spans and token metrics per model call. |
| `UseDistributedCache(cache)` | Caches responses by a hash of the messages and options. |
| `ConfigureOptions(options => ...)` | Adjusts `ChatOptions` on every call (defaults, `RawRepresentationFactory`). |
| `UseChatReducer(reducer)` (experimental) | Trims the messages sent on each call; see [compaction](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_sessions_and_memory/#compaction). |
| `UseAIContextProviders(providers)` (Agent Framework) | Runs `AIContextProvider`s on every model call inside the loop; needs to run inside an agent run. |

A response cache is the classic chat-level middleware. Because the cache key includes the whole message list, it hits only when the conversation so far is identical, which suits stateless calls (classification, extraction) better than chat:

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Caching.Distributed;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Options;
using OpenAI;

IDistributedCache cache = new MemoryDistributedCache(Options.Create(new MemoryDistributedCacheOptions()));

AIAgent classifier = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai-small%%")
    .AsIChatClient()
    .AsBuilder()
    .UseDistributedCache(cache)
    .Build()
    .AsAIAgent(instructions: "Classify the ticket as billing, technical or other. Reply with one word.");

foreach (int attempt in new[] { 1, 2 })
{
    var watch = System.Diagnostics.Stopwatch.StartNew();
    AgentResponse response = await classifier.RunAsync("I was charged twice this month.");
    Console.WriteLine($"attempt {attempt}: {response.Text} ({watch.ElapsedMilliseconds} ms)");
}
```

Each run here has no session, so both send the same two messages and the second is served from the cache.

## Context providers as middleware

`AIContextProvider`s (covered on the [sessions and memory page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_sessions_and_memory/#context-providers)) are the structured way to add instructions, messages or tools on each run. They can be attached at two levels besides `ChatClientAgentOptions.AIContextProviders`:

- `agentBuilder.UseAIContextProviders(params MessageAIContextProvider[])` runs message-only providers around any agent, including A2A, Copilot and workflow agents.
- `chatClientBuilder.UseAIContextProviders(params AIContextProvider[])` runs providers on every model call inside the loop, which must happen inside an agent run. In %%version=Microsoft.Agents.AI%% do not use it for `CompactionProvider`: with tool calls it causes the user's messages to be dropped from the stored history (details under [compaction strategies](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_sessions_and_memory/#compaction-strategies)).

## Retries and transient errors

Retry at the lowest level that knows the operation is safe to repeat.

**HTTP retries (the default).** The OpenAI and Azure SDKs already retry transient failures (408, 429, 500, 502, 503, 504, and connection errors) with exponential back-off, honouring `Retry-After`; the OpenAI SDK's default is three retries. Configure it on the client options:

```csharp
using System.ClientModel;
using System.ClientModel.Primitives;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

var options = new OpenAIClientOptions
{
    RetryPolicy = new ClientRetryPolicy(maxRetries: 5),
    NetworkTimeout = TimeSpan.FromSeconds(100),
};

AIAgent agent = new OpenAIClient(new ApiKeyCredential(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!), options)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You are helpful.");

Console.WriteLine(await agent.RunAsync("Hello"));
```

The Anthropic SDK has `ClientOptions.MaxRetries` and `ClientOptions.Timeout`.

**Resilience pipelines.** For circuit breakers, hedging or a shared policy across services, use `Microsoft.Extensions.Http.Resilience` on an `HttpClient` and hand that to the SDK through `HttpClientPipelineTransport`. Turn the SDK's own retries off, or every failure is retried twice over:

```csharp
using System.ClientModel;
using System.ClientModel.Primitives;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using OpenAI;

var services = new ServiceCollection();
services.AddHttpClient("openai")
    .AddStandardResilienceHandler(o =>
    {
        o.Retry.MaxRetryAttempts = 4;
        o.AttemptTimeout.Timeout = TimeSpan.FromSeconds(60);
        o.TotalRequestTimeout.Timeout = TimeSpan.FromMinutes(3);
        o.CircuitBreaker.SamplingDuration = TimeSpan.FromMinutes(2);
    });
using ServiceProvider provider = services.BuildServiceProvider();

HttpClient http = provider.GetRequiredService<IHttpClientFactory>().CreateClient("openai");

var options = new OpenAIClientOptions
{
    Transport = new HttpClientPipelineTransport(http),
    RetryPolicy = new ClientRetryPolicy(maxRetries: 0),   // the resilience handler owns retries
};

AIAgent agent = new OpenAIClient(new ApiKeyCredential(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!), options)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You are helpful.");

Console.WriteLine(await agent.RunAsync("Hello"));
```

**Retrying a model call.** Chat-client middleware can retry one model call, which is safe because the call has no side effects; the tool loop continues from where it was. Use it for failures the HTTP layer does not retry, such as a response that fails your own validation.

**Retrying a whole run.** Agent middleware can retry a run, but a run may already have called tools with side effects, and the session may have been updated. Retry a run only when its tools are idempotent, and give each attempt a fresh copy of the session (serialise it before the first attempt, deserialise for each retry) so a failed attempt does not leave half a turn in the history:

```csharp
using System.ClientModel;
using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You are helpful.");

AgentSession session = await agent.CreateSessionAsync();
(AgentResponse response, session) = await RunWithRetryAsync(agent, "Summarise our refund policy.", session, attempts: 3);
Console.WriteLine(response.Text);

static async Task<(AgentResponse, AgentSession)> RunWithRetryAsync(AIAgent agent, string input, AgentSession session, int attempts)
{
    JsonElement snapshot = await agent.SerializeSessionAsync(session);
    for (int attempt = 1; ; attempt++)
    {
        AgentSession working = await agent.DeserializeSessionAsync(snapshot);
        try
        {
            return (await agent.RunAsync(input, working), working);
        }
        catch (Exception ex) when (attempt < attempts && IsTransient(ex))
        {
            await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, attempt)));
        }
    }
}

static bool IsTransient(Exception ex) => ex is HttpRequestException or TimeoutException
    || ex is ClientResultException { Status: 429 or >= 500 };
```

The caller keeps the session that succeeded, so the stored conversation has exactly one copy of the turn.

## Errors and timeouts

What reaches your code from `RunAsync`:

| Source | Exception |
| --- | --- |
| OpenAI, Azure OpenAI, Foundry model calls | `System.ClientModel.ClientResultException` (`Status` holds the HTTP status) |
| Azure SDK management calls (`AgentAdministrationClient` and so on) | `Azure.RequestFailedException` |
| Anthropic | `Anthropic.Exceptions.AnthropicApiException` and subclasses (`AnthropicRateLimitException`, `Anthropic5xxException`, ...) |
| A tool, more than `MaximumConsecutiveErrorsPerRequest` times in a row | The tool's exception |
| Cancellation | `OperationCanceledException` / `TaskCanceledException` |
| Misuse (wrong session type, continuation token with messages, conflicting conversation IDs) | `InvalidOperationException` |

Single tool failures do not reach you: the loop reports them to the model ([what the model sees](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_tools/#what-the-model-sees-when-a-tool-fails)). When a run fails, `ChatClientAgent` calls each context provider's and history provider's `InvokedAsync` with the exception (`InvokedContext.InvokeException`) before rethrowing, so providers can tidy up, and it does not save the failed turn to the history.

For a deadline on the whole run, pass a `CancellationToken`: `using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30)); await agent.RunAsync(input, session, cancellationToken: cts.Token);`. The token flows to every model call and to tools that take a `CancellationToken`. `NetworkTimeout` on the client options limits a single HTTP request; `FunctionInvokingChatClient` has no per-tool timeout, so put one in function middleware if a tool can hang:

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(
        instructions: "You check slow systems.",
        tools: [AIFunctionFactory.Create(async (CancellationToken ct) => { await Task.Delay(5000, ct); return "ok"; }, "slow_check", "A slow check.")])
    .AsBuilder()
    .Use(async (agent, context, next, ct) =>
    {
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(TimeSpan.FromSeconds(1));
        try
        {
            return await next(context, timeout.Token);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            return $"{context.Function.Name} timed out after 1 s; try again later or continue without it.";
        }
    })
    .Build();

Console.WriteLine(await agent.RunAsync("Run the slow check."));
```

```text
Here is what the tool returned: slow_check timed out after 1 s; try again later or continue without it.
```

## Middleware with dependency injection

With `Microsoft.Agents.AI.Hosting`, the `AddAIAgent` overload that takes a factory lets you build the pipeline with services from the container:

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using OpenAI;

var builder = Host.CreateApplicationBuilder(args);

builder.Services.AddChatClient(new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
        .GetChatClient("%%model=openai%%")
        .AsIChatClient())
    .UseLogging();   // chat-client middleware on the registered client

builder.AddAIAgent("support", (services, name) =>
    services.GetRequiredService<IChatClient>()
        .AsAIAgent(instructions: "You are a support assistant.", name: name)
        .AsBuilder()
        .UseLogging(services.GetRequiredService<ILoggerFactory>())
        .Use((inner, sp) => new PolicyAgent(inner, sp.GetRequiredService<ILogger<PolicyAgent>>()))
        .Build(services));

using IHost host = builder.Build();
AIAgent agent = host.Services.GetRequiredKeyedService<AIAgent>("support");
Console.WriteLine(await agent.RunAsync("Hello"));

public sealed class PolicyAgent(AIAgent inner, ILogger<PolicyAgent> logger) : DelegatingAIAgent(inner)
{
    protected override Task<AgentResponse> RunCoreAsync(IEnumerable<ChatMessage> messages, AgentSession? session = null, AgentRunOptions? options = null, CancellationToken cancellationToken = default)
    {
        logger.LogInformation("Policy check for {Agent}", this.Name);
        return this.InnerAgent.RunAsync(messages, session, options, cancellationToken);
    }

    protected override IAsyncEnumerable<AgentResponseUpdate> RunCoreStreamingAsync(IEnumerable<ChatMessage> messages, AgentSession? session = null, AgentRunOptions? options = null, CancellationToken cancellationToken = default)
        => this.InnerAgent.RunStreamingAsync(messages, session, options, cancellationToken);
}
```

## AGENT-HOOKS enforcement (alpha)

`Microsoft.Agents.AI.AgentHooks` (alpha) builds a `ChatClientAgent` that emits every interception point of the AGENT-HOOKS 0.1 control contract (`agent_startup`, `input`, `pre_model_call`, `post_model_call`, `pre_tool_call`, `post_tool_call`, `output`, `agent_shutdown`) to a set of interceptors and enforces their combined verdicts fail-closed: `chatClient.AsAIAgentWithAgentHooks(new AgentHooksOptions(interceptors), agentOptions)`. A deny at the input, model or output points throws `InterceptionBlockedException`; a deny at a tool point turns into a tool error the model sees. Streaming is buffered until a verdict is reached, and history is only persisted for content the verdicts allowed. It depends on the `ResponsibleAI.AgentHooks` package for the interceptor contract. Prefer the ordinary middleware above unless you need that contract; the API is alpha and was not exercised for this page.

## Differences from Python

| Topic | Python | .NET |
| --- | --- | --- |
| Shape | `async def mw(context, call_next)` with a context object | Delegates that receive the inner agent / client / `next` explicitly, or `Delegating*` subclasses |
| Registration | `middleware=[...]` on the agent or the run, classified by decorator | `agent.AsBuilder().Use(...)` for agent and function middleware; `ChatClientBuilder.Use(...)` for chat middleware |
| Run-level middleware | `middleware=` on `run()` | `ChatClientAgentRunOptions.ChatClientFactory`, or build a wrapped agent for that run |
| Short-circuit | Set `context.result` and return | Return a response (or result) without calling the inner agent / `next` |
| Termination | `MiddlewareTermination` | Throw from agent middleware; `context.Terminate = true` in function middleware |
| Streaming | Streaming hooks on the context | Separate streaming delegate; omitting it buffers the stream |
| Retries | Middleware | SDK retry policy or `Microsoft.Extensions.Http.Resilience` first; middleware for model-call or run retries |

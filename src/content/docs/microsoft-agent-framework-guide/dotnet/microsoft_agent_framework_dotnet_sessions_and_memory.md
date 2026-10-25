---
title: "Microsoft Agent Framework (.NET) — Sessions and Memory"
description: "AgentSession and its state bag, serialising and restoring sessions, chat history providers (in-memory, Cosmos DB, Valkey, custom), service-managed history, session stores and DeleteSessionAsync, AIContextProviders, retrieval and long-term memory providers, and compaction with reducers and strategies."
framework: "microsoft-agent-framework"
language: "dotnet"
sidebar:
  label: "Sessions and memory"
  order: 7
---

An agent holds no conversation state of its own. Everything a conversation accumulates (its messages, a server-side conversation ID, approvals waiting for an answer, a todo list, the active model route) lives in an `AgentSession`, which you create, pass to every run, and store between requests. This page covers the session, where history is kept, how sessions are saved and restored, the context providers that add information on each run, the memory providers built on them, and compaction for long conversations.

Every C# block was compiled against `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% (checked %%versions-checked%%) and, except where it says otherwise, run against a local stand-in for the OpenAI API that replies "Mock reply to: <last user message>". The Cosmos DB, Valkey, Blob and Foundry examples were compiled only.

## The session

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You are helpful.");

AgentSession session = await agent.CreateSessionAsync();
await agent.RunAsync("My name is Ana.", session);
await agent.RunAsync("I live in Porto.", session);

// Without a session each run starts from nothing.
await agent.RunAsync("This run is forgotten.");

if (session.TryGetInMemoryChatHistory(out List<ChatMessage>? history))
{
    foreach (ChatMessage message in history)
    {
        Console.WriteLine($"{message.Role}: {message.Text}");
    }
}
```

```text
user: My name is Ana.
assistant: Mock reply to: My name is Ana.
user: I live in Porto.
assistant: Mock reply to: I live in Porto.
```

What to know about sessions:

- **A session belongs to the agent that created it.** Each agent type has its own session class (`ChatClientAgentSession`, `A2AAgentSession`, `GitHubCopilotAgentSession`, ...), and passing one agent's session to an unrelated agent throws `InvalidOperationException`. Two `ChatClientAgent`s will accept each other's sessions, but the history then mixes two agents' turns, and providers keyed by state-bag name may read each other's state. To pass context between agents, use a workflow or send the first agent's output as the second's input.
- **Running without a session** creates a throwaway one for that run. Nothing is kept.
- **A session is not thread-safe.** Do not run two turns on the same session at the same time; serialise turns per conversation (a per-session lock, a queue, or one actor per conversation).
- **The agent is.** A single `ChatClientAgent` can serve any number of sessions concurrently.

`ChatClientAgentSession` has two parts: `ConversationId`, set when the service keeps the history, and `StateBag`, an `AgentSessionStateBag` where the history provider, context providers and middleware keep their state as JSON.

### The state bag

`AgentSessionStateBag` is a string-keyed bag of JSON values: `SetValue<T>(key, value)`, `GetValue<T>(key)`, `TryGetValue<T>(key, out value)`, `TryRemoveValue(key)` and `Count`. Values are serialised when set, so a value you read back is a copy; change it and set it again. Every provider declares the keys it uses (`StateKeys`), and the agent refuses, at construction, to combine providers whose keys clash.

```csharp
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You are helpful.");

AgentSession session = await agent.CreateSessionAsync();
session.StateBag.SetValue("customer", new Customer("C-1042", "Gold"), AppJson.Default.Options);

Customer? customer = session.StateBag.GetValue<Customer>("customer", AppJson.Default.Options);
Console.WriteLine($"{customer?.Id} is {customer?.Tier}");

// Your own values travel with the session when it is serialised.
JsonElement saved = await agent.SerializeSessionAsync(session);
AgentSession restored = await agent.DeserializeSessionAsync(saved);
Console.WriteLine(restored.StateBag.TryGetValue("customer", out Customer? again, AppJson.Default.Options) ? again!.Tier : "missing");

public sealed record Customer(string Id, string Tier);

[JsonSerializable(typeof(Customer))]
internal sealed partial class AppJson : JsonSerializerContext;
```

Pass `JsonSerializerOptions` (ideally from a source-generated context, which also keeps you trimming- and AOT-safe) for your own types. Keys are case-sensitive; prefix yours (`"myapp.customer"`) to stay clear of the framework's (`InMemoryChatHistoryProvider`, `toolApprovalState`, provider type names).

## Where history lives

| The service... | Example | History is kept | `ConversationId` | Local history provider |
| --- | --- | --- | --- | --- |
| keeps nothing | Chat Completions, Anthropic, Ollama, Responses with stored output disabled | In the session, by the `ChatHistoryProvider` | `null` | Used |
| stores responses | Responses API (default), Foundry code-defined agents | On the service | The latest response ID | Not used |
| has conversations | OpenAI Conversations, Foundry conversations, Foundry-defined agents | On the service | The conversation ID | Not used |

The agent decides per run: if the session (or `ChatOptions.ConversationId`) has a conversation ID, or the service returns one, the service is assumed to hold the history and the history provider is skipped. You do not configure this; it follows from the client you chose.

### When the service keeps the history

If you configured a `ChatHistoryProvider` explicitly and the service then returns a conversation ID, there are two stores for one conversation. Three flags on `ChatClientAgentOptions` decide what happens, and with the defaults the run **throws** `InvalidOperationException` ("Only ConversationId or ChatHistoryProvider may be used, but not both"):

| Flag | Default | Effect |
| --- | --- | --- |
| `ThrowOnChatHistoryProviderConflict` | `true` | Throw. |
| `WarnOnChatHistoryProviderConflict` | `true` | Log a warning. |
| `ClearOnChatHistoryProviderConflict` | `true` | Drop the provider from the agent for the rest of its life. |

The fix is usually to choose one: either turn stored output off (`AsIChatClientWithStoredOutputDisabled`, see [providers](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_providers/#responses)) and keep your provider, or remove the provider and let the service keep the history. A response-ID session points at server-side state you do not control: if the stored responses expire or are deleted, the session's history is gone.

### Persisting after every model call

By default the history provider is called once, at the end of the run, with all new messages. If the process dies in the middle of a long tool loop, that run's messages are lost. `RequirePerServiceCallChatHistoryPersistence = true` adds a decorator inside the loop that loads history before, and saves it after, every model call; with service-managed history it updates `ConversationId` after every call instead. Use it for long-running agents, together with saving the session somewhere durable from inside the loop. Background responses and continuation tokens fall back to end-of-run persistence.

## Saving and restoring sessions

`agent.SerializeSessionAsync(session)` returns a `JsonElement`; `agent.DeserializeSessionAsync(json)` rebuilds the session. Serialise with the same agent (or an identically configured one) that will deserialise, because the providers decide what their state means.

```csharp
using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You are helpful.");

// Request 1
AgentSession session = await agent.CreateSessionAsync();
await agent.RunAsync("Remember the number 42.", session);
string stored = (await agent.SerializeSessionAsync(session)).GetRawText();   // save to Redis, SQL, Blob...
Console.WriteLine(stored[..120] + "...");

// Request 2, possibly on another server
using JsonDocument document = JsonDocument.Parse(stored);
AgentSession resumed = await agent.DeserializeSessionAsync(document.RootElement.Clone());
Console.WriteLine(await agent.RunAsync("What number did I ask you to remember?", resumed));
```

```text
{"stateBag":{"InMemoryChatHistoryProvider":{"messages":[{"role":"user","contents":[{"$type":"text","text":"Remember the number 42."}]},{...
Mock reply to: What number did I ask you to remember?
```

The JSON has a `conversationId` (when the service keeps history) and a `stateBag` with one property per state key. With the default in-memory history the whole conversation is inside, so it grows without limit; use [compaction](#compaction) or a database-backed history provider for long conversations. A serialised session contains everything the user and the model said: encrypt it at rest if your store does not, and never let a client send one back to you (keep it server-side, keyed by an ID).

## Chat history providers

A `ChatHistoryProvider` supplies the history before each run and stores new messages after it. The agent uses `ChatClientAgentOptions.ChatHistoryProvider`, or an `InMemoryChatHistoryProvider` when you set none.

### `InMemoryChatHistoryProvider`

Despite the name it is not process memory: it keeps the messages in the session's state bag, so they are saved and restored with the session.

| `InMemoryChatHistoryProviderOptions` | Default | Meaning |
| --- | --- | --- |
| `StateKey` | `"InMemoryChatHistoryProvider"` | State-bag key. Change it when two providers share a session. |
| `ChatReducer` | none | An `IChatReducer` that trims stored history ([compaction](#reducers-on-the-history-provider)). |
| `ReducerTriggerEvent` | `BeforeMessagesRetrieval` | When the reducer runs: before history is read for a run, or `AfterMessageAdded`. |
| `StorageInputRequestMessageFilter` | exclude messages that came from history | Which request messages to store. |
| `StorageInputResponseMessageFilter` | all | Which response messages to store. |
| `ProvideOutputMessageFilter` | all | Which stored messages to send to the model. |
| `StateInitializer` | empty list | Initial state for a new session, e.g. a seeded conversation. |
| `JsonSerializerOptions` | framework defaults | For message serialisation. |

`session.TryGetInMemoryChatHistory(out messages)` and `session.SetInMemoryChatHistory(messages)` read and replace the stored messages; the provider's own `GetMessages(session)` and `SetMessages(session, messages)` do the same for a provider with a custom `StateKey`. Setting the history is how you seed a conversation, import one from another system, or let a user edit or delete a message.

### What gets stored: message sources

Every message the agent sends to the model is tagged with where it came from, an `AgentRequestMessageSourceType`: `External` (the caller's input), `ChatHistory` (loaded from the history provider) or `AIContextProvider` (added by a context provider). Read the tag with `message.GetAgentRequestMessageSourceType()`.

The default storage filter only drops `ChatHistory` messages, so **messages added by context providers are stored in the history**. A retrieval provider that adds search results on every turn therefore makes the history grow by those results each time, and the model sees old results again on later turns. Usually you want only the conversation stored:

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(new ChatClientAgentOptions
    {
        ChatOptions = new() { Instructions = "You are a shop assistant." },
        AIContextProviders = [new OpeningHours()],
        ChatHistoryProvider = new InMemoryChatHistoryProvider(new InMemoryChatHistoryProviderOptions
        {
            // Keep only what the user actually said; drop provider-injected context.
            StorageInputRequestMessageFilter = messages => messages.Where(m =>
                m.GetAgentRequestMessageSourceType() == AgentRequestMessageSourceType.External),
        }),
    });

AgentSession session = await agent.CreateSessionAsync();
await agent.RunAsync("Hi", session);
await agent.RunAsync("When do you open?", session);

session.TryGetInMemoryChatHistory(out var history);
Console.WriteLine($"{history!.Count} messages stored");   // 4: two user, two assistant

public sealed class OpeningHours : MessageAIContextProvider
{
    protected override ValueTask<IEnumerable<ChatMessage>> ProvideMessagesAsync(InvokingContext context, CancellationToken cancellationToken = default)
        => new([new ChatMessage(ChatRole.User, "Context: the shop opens at 09:00 and closes at 18:00.")]);
}
```

Without the filter the same two turns store six messages, the provider's context twice over.

### Cosmos DB

`Microsoft.Agents.AI.CosmosNoSql` (preview) stores one document per message in a Cosmos DB container, instead of keeping the whole history in the session. The session then holds only the conversation ID (and optional tenant and user IDs) in its state bag.

```csharp
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

var history = new CosmosChatHistoryProvider(
    accountEndpoint: Environment.GetEnvironmentVariable("COSMOS_ENDPOINT")!,
    tokenCredential: new DefaultAzureCredential(),
    databaseId: "agents",
    containerId: "chat-history",
    stateInitializer: session => new CosmosChatHistoryProvider.State(
        conversationId: Guid.NewGuid().ToString("N"),
        tenantId: "contoso",
        userId: "user-17"))
{
    MessageTtlSeconds = 60 * 60 * 24 * 30,   // expire messages after 30 days
    MaxMessagesToRetrieve = 50,              // only the latest 50 go to the model
};

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(new ChatClientAgentOptions
    {
        ChatOptions = new() { Instructions = "You are a support assistant." },
        ChatHistoryProvider = history,
    });

AgentSession session = await agent.CreateSessionAsync();
await agent.RunAsync("My order 1042 has not arrived.", session);

Console.WriteLine($"{await history.GetMessageCountAsync(session)} messages in Cosmos DB");
await history.ClearMessagesAsync(session);   // e.g. on a "forget me" request
```

Constructors take a connection string, an endpoint plus `TokenCredential`, or your own `CosmosClient` (set `CosmosChatHistoryProviderOptions.OwnsClient` to have the provider dispose it). `PartitionKeyFactory` controls the partition key: by default a hierarchical key of tenant, user and conversation ID when both tenant and user are set, otherwise the conversation ID, and it must match the container's partition key definition; `MaxItemCount` and `MaxBatchSize` tune paging and writes. The `WithCosmosDBChatHistoryProvider*` extensions on `ChatClientAgentOptions` are shortcuts for the same thing. The identity needs the Cosmos DB data-plane role described on the [Azure RBAC page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_azure_rbac/).

### Valkey and Redis

`Microsoft.Agents.AI.Valkey` (alpha) stores history in a Valkey (or Redis-compatible) list per conversation, through `Valkey.Glide`'s `IConnectionMultiplexer`:

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Valkey;
using Microsoft.Extensions.AI;
using OpenAI;
using Valkey.Glide;

IConnectionMultiplexer valkey = await ConnectionMultiplexer.ConnectAsync("localhost:6379");

var history = new ValkeyChatHistoryProvider(
    valkey,
    session => new ValkeyChatHistoryProvider.State(conversationId: Guid.NewGuid().ToString("N")),
    new ValkeyChatHistoryProviderOptions
    {
        KeyPrefix = "chat:",
        MaxMessages = 200,             // trim the stored list
        MaxMessagesToRetrieve = 40,    // send only the latest 40
    });

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(new ChatClientAgentOptions { ChatHistoryProvider = history });

AgentSession session = await agent.CreateSessionAsync();
await agent.RunAsync("Hello", session);
Console.WriteLine(await history.GetMessageCountAsync(session));
```

### Writing your own history provider

Derive from `ChatHistoryProvider` and override two methods: `ProvideChatHistoryAsync` (return the stored messages for this session) and `StoreChatHistoryAsync` (save `context.RequestMessages` and `context.ResponseMessages`, already filtered). Keep per-session data, such as the database key, in the state bag through `ProviderSessionState<TState>`, which initialises it on first use and saves it back. The base class tags returned messages as `ChatHistory`, applies the storage filters, and skips storing when the run failed.

```csharp
using System.Collections.Concurrent;
using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

var store = new ConcurrentDictionary<string, List<string>>();   // stand-in for a database table

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(new ChatClientAgentOptions
    {
        ChatOptions = new() { Instructions = "You are helpful." },
        ChatHistoryProvider = new TableChatHistoryProvider(store),
    });

AgentSession session = await agent.CreateSessionAsync();
await agent.RunAsync("First message", session);
await agent.RunAsync("Second message", session);

string saved = (await agent.SerializeSessionAsync(session)).GetRawText();
Console.WriteLine(saved);                                   // just the key, not the messages
Console.WriteLine($"{store.Values.Single().Count} rows in the table");

public sealed class TableChatHistoryProvider : ChatHistoryProvider
{
    private readonly ConcurrentDictionary<string, List<string>> _table;
    private readonly ProviderSessionState<Pointer> _state;

    public TableChatHistoryProvider(ConcurrentDictionary<string, List<string>> table)
    {
        _table = table;
        _state = new ProviderSessionState<Pointer>(_ => new Pointer(Guid.NewGuid().ToString("N")), stateKey: "myapp.history");
    }

    public override IReadOnlyList<string> StateKeys => [_state.StateKey];

    protected override ValueTask<IEnumerable<ChatMessage>> ProvideChatHistoryAsync(InvokingContext context, CancellationToken cancellationToken = default)
    {
        string key = _state.GetOrInitializeState(context.Session).Key;
        IEnumerable<ChatMessage> rows = _table.TryGetValue(key, out var json)
            ? json.Select(j => JsonSerializer.Deserialize<ChatMessage>(j)!).ToList()
            : [];
        return new(rows);
    }

    protected override ValueTask StoreChatHistoryAsync(InvokedContext context, CancellationToken cancellationToken = default)
    {
        Pointer pointer = _state.GetOrInitializeState(context.Session);
        _state.SaveState(context.Session, pointer);   // make sure the key is in the session
        List<string> rows = _table.GetOrAdd(pointer.Key, _ => []);
        lock (rows)
        {
            rows.AddRange(context.RequestMessages.Concat(context.ResponseMessages ?? []).Select(m => JsonSerializer.Serialize(m)));
        }
        return default;
    }

    public sealed record Pointer(string Key);
}
```

```text
{"stateBag":{"myapp.history":{"key":"5f0c..."}}}
4 rows in the table
```

Return messages oldest first. If your store can return a lot, cap what you return (the latest N messages, or by token budget) rather than relying on the model's context window. `ChatMessage` serialises with `System.Text.Json` as shown, including tool calls and results; keep the call/result pairs together when you truncate, or the next request is rejected by the provider.

### A different history provider for one run

To use another history provider for a single run (an "incognito" turn, a replay from an archive), put it in the run's `AdditionalProperties` under its type: `new AgentRunOptions { AdditionalProperties = new() }` then `options.AdditionalProperties.Add<ChatHistoryProvider>(provider)`. The agent uses that provider instead of its own for the run. Its state keys must not clash with the context providers'.

## Session stores (hosting)

When an agent is hosted (`Microsoft.Agents.AI.Hosting`), sessions are saved and loaded through an `AgentSessionStore` keyed by an `AgentSessionStoreKey`, so that a conversation ID from the client is all you need to resume. The abstraction is in `Microsoft.Agents.AI.Abstractions` and is experimental (`MAAI001`).

| Member | Contract |
| --- | --- |
| `GetSessionAsync(agent, key)` | Return the stored session or `null`. Never create one. Each call must return an independent instance. |
| `SaveSessionAsync(agent, key, session)` | Store (insert or replace) the session. |
| `DeleteSessionAsync(agent, key)` | Delete the session; a missing session is not an error. **Abstract since 1.26.0**: custom stores written for earlier versions must add it. |
| `GetOrCreateSessionAsync(agent, key)` | Virtual: `GetSessionAsync`, then `agent.CreateSessionAsync()` if nothing was stored. |

`AgentSessionStoreKey` is a `SessionId` plus optional named `Partitions` (case-sensitive names; order does not matter). Every partition is part of the identity, so a store must include them all in its storage key; `WithPartition(name, value)` returns a new key with one more.

Shipped stores:

| Store | Package | Notes |
| --- | --- | --- |
| `InMemoryAgentSessionStore` | `Microsoft.Agents.AI.Hosting` | Development and tests; lost on restart, not shared between instances. |
| `NoopAgentSessionStore` | `Microsoft.Agents.AI.Hosting` | Stores nothing; every request starts a new conversation. |
| `AzureBlobAgentSessionStore` | `Microsoft.Agents.AI.Hosting.AzureStorage` | One blob per session in a container; `CreateContainerIfNotExists`, `BlobNamePrefix`. |
| `IsolationKeyScopedAgentSessionStore` | `Microsoft.Agents.AI.Hosting` | A decorator that adds an `isolation` partition from an `AgentIsolationKeyProvider` (for example the signed-in user). |
| `DelegatingAgentSessionStore` | `Microsoft.Agents.AI` | Base class for your own decorators (caching, encryption, auditing). |

A store for a key-value database, with deletion, looks like this:

```csharp
#pragma warning disable MAAI001
using System.Collections.Concurrent;
using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting;
using Microsoft.Extensions.AI;
using OpenAI;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(instructions: "You are helpful.", name: "assistant");

var store = new KeyValueSessionStore();
var host = new AIHostAgent(agent, store);

var key = new AgentSessionStoreKey("conv-1").WithPartition("user", "alice");

AgentSession session = await host.GetOrCreateSessionAsync(key);
await host.RunAsync("Hello", session);
await host.SaveSessionAsync(key, session);

AgentSession again = await host.GetOrCreateSessionAsync(key);
again.TryGetInMemoryChatHistory(out var history);
Console.WriteLine($"restored {history!.Count} messages");

await store.DeleteSessionAsync(agent, key);
Console.WriteLine(await store.GetSessionAsync(agent, key) is null ? "deleted" : "still there");

public sealed class KeyValueSessionStore : AgentSessionStore
{
    private readonly ConcurrentDictionary<string, string> _rows = new();   // stand-in for Redis, SQL, ...

    private static string RowKey(AIAgent agent, AgentSessionStoreKey key)
    {
        string partitions = key.Partitions is null
            ? ""
            : string.Join(";", key.Partitions.OrderBy(p => p.Key, StringComparer.Ordinal).Select(p => $"{p.Key}={p.Value}"));
        return $"{agent.Name ?? agent.Id}|{partitions}|{key.SessionId}";
    }

    public override async ValueTask SaveSessionAsync(AIAgent agent, AgentSessionStoreKey key, AgentSession session, CancellationToken cancellationToken = default)
        => _rows[RowKey(agent, key)] = (await agent.SerializeSessionAsync(session, cancellationToken: cancellationToken)).GetRawText();

    public override async ValueTask<AgentSession?> GetSessionAsync(AIAgent agent, AgentSessionStoreKey key, CancellationToken cancellationToken = default)
    {
        if (!_rows.TryGetValue(RowKey(agent, key), out string? json))
        {
            return null;
        }
        using JsonDocument document = JsonDocument.Parse(json);
        return await agent.DeserializeSessionAsync(document.RootElement.Clone(), cancellationToken: cancellationToken);   // a fresh instance per call
    }

    public override ValueTask DeleteSessionAsync(AIAgent agent, AgentSessionStoreKey key, CancellationToken cancellationToken = default)
    {
        _rows.TryRemove(RowKey(agent, key), out _);
        return default;
    }
}
```

`AIHostAgent` is the `DelegatingAIAgent` the hosting layer wraps around your agent: it adds `GetOrCreateSessionAsync` and `SaveSessionAsync` (by conversation ID or by key), and `BindIsolationKey(key)` to fix the isolation key for one request. Its optional `sessionStorageIdentity` partitions storage by a logical name instead of the agent's ID, so a redeployed agent with a new ID still finds its sessions.

With `IHostedAgentBuilder` you rarely touch these classes directly:

```csharp
#pragma warning disable MAAI001
using Azure.Identity;
using Azure.Storage.Blobs;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting;
using Microsoft.Extensions.AI;
using OpenAI;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddChatClient(new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%").AsIChatClient());

builder.Services.AddHttpContextAccessor();
builder.Services.UseClaimsBasedAgentIsolation();   // isolation key = the user's NameIdentifier claim

var container = new BlobContainerClient(new Uri(Environment.GetEnvironmentVariable("SESSIONS_CONTAINER_URL")!), new DefaultAzureCredential());

builder.AddAIAgent("assistant", "You are a helpful assistant.")
       .WithAzureBlobSessionStore(container);          // withIsolation: true by default

// For development instead: .WithInMemorySessionStore();

var app = builder.Build();
app.Run();
```

`UseClaimsBasedAgentIsolation()` (in `Microsoft.Agents.AI.Hosting.AspNetCore`) registers a `ClaimsIdentityAgentIsolationKeyProvider`. Its default claim is `ClaimTypes.NameIdentifier`, which is not Entra ID's `oid`; set `ClaimsIdentityAgentIsolationKeyProviderOptions.ClaimType` if you need `oid` or a tenant-qualified value. With isolation on (the default for `With*SessionStore`), a request whose isolation key cannot be determined throws (`IsolationKeyScopedAgentSessionStoreOptions.Strict` is `true`), rather than reading or writing an unpartitioned session. Never take the isolation key from the request body: a user who can choose it can read anyone's conversation. The OpenAI-compatible, A2A and AG-UI endpoints that use these stores are on the [hosting page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/#session-stores-and-isolation).

## Context providers

An `AIContextProvider` runs before each model run to add **instructions**, **messages** or **tools**, and after it to observe the result. Memory, retrieval, skills, todo lists and compaction are all context providers. Attach them with `ChatClientAgentOptions.AIContextProviders`; they run in order, each seeing what the previous ones produced.

| Override | When | Return |
| --- | --- | --- |
| `ProvideAIContextAsync(InvokingContext)` | Before the run | An `AIContext` whose `Instructions`, `Messages` and `Tools` are added to the run |
| `StoreAIContextAsync(InvokedContext)` | After a successful run | Nothing; read `RequestMessages` and `ResponseMessages` to learn from the turn |
| `InvokingCoreAsync`, `InvokedCoreAsync` | Lower level: replace the merging or filtering behaviour | |

`InvokingContext` gives the `Agent`, the `Session` and the `AIContext` accumulated so far. Instructions you return are appended to the agent's; messages are appended after the history *and* the caller's new messages, so they are the last thing the model reads; tools are added for this run only. The base constructor takes three filters: by default a provider *sees* and *stores* only `External` messages (what the user said), not history or other providers' output. `StoreAIContextAsync` is not called when the run failed; override `InvokedCoreAsync` and check `context.InvokeException` if you need to know.

`MessageAIContextProvider` is the simpler base class for providers that only add messages (override `ProvideMessagesAsync`). Only message providers can be attached to agents that are not `ChatClientAgent`s, with `AIAgentBuilder.UseAIContextProviders` ([middleware](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_middleware/#context-providers-as-middleware)).

A provider that keeps per-session state (here, a shopping basket) and gives the model tools to change it:

```csharp
using System.Text;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(new ChatClientAgentOptions
    {
        ChatOptions = new() { Instructions = "You are a shop assistant. Use the basket tools." },
        AIContextProviders = [new BasketProvider()],
    });

AgentSession session = await agent.CreateSessionAsync();
Console.WriteLine(await agent.RunAsync("Add two loaves of bread.", session));
Console.WriteLine(string.Join(", ", BasketProvider.Items(session)));

public sealed class BasketProvider : AIContextProvider
{
    private const string Key = "myapp.basket";

    public override IReadOnlyList<string> StateKeys => [Key];

    public static List<string> Items(AgentSession? session) =>
        session?.StateBag.TryGetValue(Key, out List<string>? items) == true ? items! : [];

    protected override ValueTask<AIContext> ProvideAIContextAsync(InvokingContext context, CancellationToken cancellationToken = default)
    {
        List<string> items = Items(context.Session);
        var summary = new StringBuilder("The basket contains: ").Append(items.Count == 0 ? "nothing" : string.Join(", ", items));

        return new(new AIContext
        {
            Instructions = summary.ToString(),
            Tools =
            [
                AIFunctionFactory.Create((string item) =>
                {
                    var current = Items(context.Session);
                    current.Add(item);
                    context.Session!.StateBag.SetValue(Key, current);
                    return $"Added {item}.";
                }, "add_to_basket", "Add an item to the basket."),
            ],
        });
    }
}
```

`AIContextProvider.InvokingAsync` and `InvokedAsync` are public, so a provider can be driven by your own agent implementation too. `AIContextProvider`s can also be placed on the chat client with `ChatClientBuilder.UseAIContextProviders(...)`, which runs them before *every model call* in the tool loop rather than once per run (but see the warning under [compaction strategies](#compaction-strategies) before doing that with `CompactionProvider`).

## Retrieval: `TextSearchProvider`

`TextSearchProvider` turns any search function into retrieval-augmented generation. You supply `Func<string, CancellationToken, Task<IEnumerable<TextSearchResult>>>`; the provider calls it and adds the hits (with `SourceName` and `SourceLink` for citations) as a message.

| `TextSearchProviderOptions` | Default | Meaning |
| --- | --- | --- |
| `SearchTime` | `BeforeAIInvoke` | Search on every run with the user's message, or `OnDemandFunctionCalling` to give the model a `Search` tool instead. |
| `FunctionToolName`, `FunctionToolDescription` | `"Search"`, ... | The tool, in on-demand mode. |
| `ContextPrompt`, `CitationsPrompt` | "## Additional Context ..." | Text placed around the results. |
| `ContextFormatter` | none | Format the results yourself. |
| `RecentMessageMemoryLimit`, `RecentMessageRolesIncluded` | 0, user | Also search with the last N messages, not just the current one. |
| `SearchInputMessageFilter`, `StorageInput*MessageFilter` | external only | Which messages feed the search and the recent-message memory. |
| `EnableSensitiveTelemetryData`, `Redactor` | off | Whether queries and results appear in telemetry. |

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

string[] docs =
[
    "Returns: items may be returned within 30 days of delivery, unused and in original packaging.",
    "Shipping: standard shipping takes 3-5 business days; express is next day.",
    "Warranty: tents carry a two-year warranty against manufacturing defects.",
];

Task<IEnumerable<TextSearchProvider.TextSearchResult>> SearchAsync(string query, CancellationToken ct)
{
    // Replace with Azure AI Search, a vector store, SQL full-text, ...
    var hits = docs
        .Where(d => query.Split(' ', StringSplitOptions.RemoveEmptyEntries).Any(w => w.Length > 3 && d.Contains(w, StringComparison.OrdinalIgnoreCase)))
        .Select((d, i) => new TextSearchProvider.TextSearchResult { SourceName = d.Split(':')[0], SourceLink = $"https://contoso.example/policies/{i}", Text = d });
    return Task.FromResult(hits);
}

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(new ChatClientAgentOptions
    {
        ChatOptions = new() { Instructions = "Answer from the provided context and cite the source." },
        AIContextProviders = [new TextSearchProvider(SearchAsync)],
        ChatHistoryProvider = new InMemoryChatHistoryProvider(new()
        {
            StorageInputRequestMessageFilter = m => m.Where(x => x.GetAgentRequestMessageSourceType() == AgentRequestMessageSourceType.External),
        }),
    });

Console.WriteLine(await agent.RunAsync("How long does shipping take?"));
```

Remember the [storage filter](#what-gets-stored-message-sources): without it, every turn's search results are written into the history. For a full Azure AI Search pipeline, see Recipe 6 in the [.NET recipes](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_recipes/#recipe-6-retrieval-with-azure-ai-search). Search results are untrusted input: a document can contain instructions aimed at the model.

## Long-term memory

Session history is short-term memory. Long-term memory is information that outlives a session: preferences, facts about the user, earlier conversations. The framework ships four providers.

| Provider | Package | Stores | Recalls |
| --- | --- | --- | --- |
| `ChatHistoryMemoryProvider` | `Microsoft.Agents.AI` | Every message, embedded in a `Microsoft.Extensions.VectorData` vector store, scoped by application, agent, user and session | Similar past messages, before each run or as a `Search` tool |
| `FileMemoryProvider` | `Microsoft.Agents.AI` | Files the model writes itself through `file_memory_*` tools, in an `AgentFileStore` | The model reads, greps and edits its notes |
| `FoundryMemoryProvider` (experimental) | `Microsoft.Agents.AI.Foundry` | Foundry Memory extracts and consolidates memories from the conversation | Relevant memories for a scope (e.g. a user ID) |
| `Mem0Provider` | `Microsoft.Agents.AI.Mem0` | Mem0 service | Mem0 search |

The `Microsoft.Agents.AI.Mem0` package on NuGet has not been published since 1.0.0-preview.251028.1 (October 2025), although the source is still in the repository; check before depending on it.

### `ChatHistoryMemoryProvider`

The state you return from the initialiser sets two scopes: where this session's messages are **stored**, and what is **searched**. Storing per session but searching per user gives "remember what this user said in other conversations". Any `VectorStore` works (Azure AI Search, Qdrant, Cosmos DB, PostgreSQL, SQLite, in-memory), provided it has an embedding generator.

```csharp
using CommunityToolkit.VectorData.InMemory;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using OpenAI;

// A real application would use an embedding model; this toy embedder keeps the example offline.
VectorStore vectors = new InMemoryVectorStore(new InMemoryVectorStoreOptions { EmbeddingGenerator = new ToyEmbedder() });

using var memory = new ChatHistoryMemoryProvider(
    vectors,
    collectionName: "chat-memory",
    vectorDimensions: ToyEmbedder.Dimensions,
    session => new ChatHistoryMemoryProvider.State(
        storageScope: new() { UserId = "user-17", SessionId = Guid.NewGuid().ToString("N") },
        searchScope: new() { UserId = "user-17" }),           // search all of this user's sessions
    new ChatHistoryMemoryProviderOptions { MaxResults = 3 });

// Record what reaches the model, to show the recalled memory.
List<ChatMessage> sentToModel = [];
IChatClient model = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsBuilder()
    .Use((messages, options, next, ct) => { sentToModel = messages.ToList(); return next(messages, options, ct); })
    .Build();

AIAgent agent = model.AsAIAgent(new ChatClientAgentOptions
{
    ChatOptions = new() { Instructions = "You are a travel assistant." },
    AIContextProviders = [memory],
});

await agent.RunAsync("I am vegetarian and I travel with my dog.", await agent.CreateSessionAsync());

// A new session: its history is empty, but the provider finds the earlier message.
await agent.RunAsync("Suggest a restaurant for my dog and me.", await agent.CreateSessionAsync());

foreach (ChatMessage message in sentToModel)
{
    Console.WriteLine($"[{message.GetAgentRequestMessageSourceType()}] {message.Text}");
}

public sealed class ToyEmbedder : IEmbeddingGenerator<string, Embedding<float>>
{
    public const int Dimensions = 64;

    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<string> values, EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default)
        => Task.FromResult(new GeneratedEmbeddings<Embedding<float>>(values.Select(v =>
        {
            var vector = new float[Dimensions];
            foreach (string word in v.ToLowerInvariant().Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                vector[(word.GetHashCode() & 0x7fffffff) % Dimensions] += 1;
            }
            return new Embedding<float>(vector);
        })));

    public object? GetService(Type serviceType, object? serviceKey = null) => null;
    public void Dispose() { }
}
```

```text
[External] Suggest a restaurant for my dog and me.
[AIContextProvider] ## Memories
Consider the following memories when answering user questions:
Mock reply to: I am vegetarian and I travel with my dog.
I am vegetarian and I travel with my dog.
```

Both messages of the first session (the user's and the reply) were found and appended after the new question.

`ChatHistoryMemoryProviderOptions` has `SearchTime` (`BeforeAIInvoke` or `OnDemandFunctionCalling`), `MaxResults` (default 3), `ContextPrompt` ("## Memories ..."), the tool name and description, the usual message filters, and telemetry redaction. Scopes are the only access control: whatever is in the search scope can surface in the prompt, so scope by user (and tenant), never more widely.

### `FileMemoryProvider`

The model manages its own notes as files, through seven tools: `file_memory_write`, `file_memory_read`, `file_memory_delete`, `file_memory_ls`, `file_memory_grep`, `file_memory_replace` and `file_memory_replace_lines`. The provider lists the files in the working folder on each run so the model knows they exist. `FileSystemAgentFileStore(root)` keeps them on disk; `InMemoryAgentFileStore` in memory (both experimental, `MAAI001`); `AgentFileStore` is the base for your own (Blob storage, a database).

```csharp
#pragma warning disable MAAI001
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

var files = new InMemoryAgentFileStore();

using var notes = new FileMemoryProvider(
    files,
    session => new FileMemoryState { WorkingFolder = "users/user-17" },   // one folder per user
    new FileMemoryProviderOptions { Instructions = "Keep a file called preferences.md up to date." });

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(new ChatClientAgentOptions
    {
        ChatOptions = new() { Instructions = "You are a travel assistant. Remember what the user tells you." },
        AIContextProviders = [notes],
    });

Console.WriteLine(await agent.RunAsync("Remember that I am vegetarian.", await agent.CreateSessionAsync()));
```

The stand-in model calls the first tool it is given (`file_memory_write`) with sample arguments, so this run writes a file; a real model decides for itself when to write. The working folder confines the tools: paths outside it are rejected.

### Foundry Memory

```csharp
#pragma warning disable MAAI001
using Azure.AI.Projects;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Foundry;

var project = new AIProjectClient(new Uri(Environment.GetEnvironmentVariable("FOUNDRY_PROJECT_ENDPOINT")!), new DefaultAzureCredential());

var memory = new FoundryMemoryProvider(
    project,
    memoryStoreName: "travel-memories",
    stateInitializer: _ => new FoundryMemoryProvider.State(new FoundryMemoryProviderScope("user-17")),
    new FoundryMemoryProviderOptions { MaxMemories = 5 });

await memory.EnsureMemoryStoreCreatedAsync(chatModel: "%%model=openai%%", embeddingModel: "text-embedding-3-large");

ChatClientAgent agent = project.AsAIAgent(new ChatClientAgentOptions
{
    ChatOptions = new() { ModelId = "%%model=openai%%", Instructions = "You are a travel assistant. Use known memories." },
    AIContextProviders = [memory],
});

AgentSession session = await agent.CreateSessionAsync();
await agent.RunAsync("I'm Taylor and I'm hiking in Patagonia in November.", session);
await memory.WhenUpdatesCompletedAsync();   // memory extraction is asynchronous

Console.WriteLine(await agent.RunAsync("What do you know about my trip?", await agent.CreateSessionAsync()));
```

Memory extraction runs on the service after each run, asynchronously; `UpdateDelay` (0, immediate, by default) lets the service batch several updates together. `WhenUpdatesCompletedAsync` waits for pending updates, which tests and demos need. `EnsureStoredMemoriesDeletedAsync(session)` deletes the memories for a session's scope.

## Compaction

A conversation that keeps growing eventually exceeds the context window, and long before that it costs more and the model gets worse at finding what matters. Three tools are available, from simplest to most capable.

### Reducers on the history provider

An `IChatReducer` on `InMemoryChatHistoryProvider` trims the **stored** history. `Microsoft.Extensions.AI` has two (both experimental, `MEAI001`): `MessageCountingChatReducer(n)` keeps the last *n* messages (plus the system message), and `SummarizingChatReducer(chatClient, targetCount, threshold)` summarises older messages with a model. This is destructive: the trimmed messages are gone from the session.

```csharp
#pragma warning disable MEAI001
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;

AIAgent agent = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient()
    .AsAIAgent(new ChatClientAgentOptions
    {
        ChatHistoryProvider = new InMemoryChatHistoryProvider(new()
        {
            ChatReducer = new MessageCountingChatReducer(2),
            ReducerTriggerEvent = InMemoryChatHistoryProviderOptions.ChatReducerTriggerEvent.AfterMessageAdded,
        }),
    });

AgentSession session = await agent.CreateSessionAsync();
for (int turn = 1; turn <= 3; turn++)
{
    await agent.RunAsync($"Turn {turn}", session);
    session.TryGetInMemoryChatHistory(out var stored);
    Console.WriteLine($"after turn {turn}: {stored!.Count} stored");
}
```

```text
after turn 1: 2 stored
after turn 2: 2 stored
after turn 3: 2 stored
```

With the default trigger, `BeforeMessagesRetrieval`, the reduction happens when the next run reads the history, so the stored list holds the reduced history plus the latest turn (2, 4, 4 in this example).

### Compaction strategies

`Microsoft.Agents.AI.Compaction` (experimental, `MAAI001`) is the framework's own, more careful approach. A `CompactionStrategy` works on a `CompactionMessageIndex` that groups messages into atomic units (a tool call is always kept or dropped together with its results), each with a message, byte and token count and a turn index. A **trigger** decides whether to compact; a **target** decides when to stop (by default, when the trigger would no longer fire); a minimum-preserved floor protects the most recent groups or turns; system messages are always kept.

| Strategy | Default floor | What it does |
| --- | --- | --- |
| `ToolResultCompactionStrategy(trigger)` | 16 groups | Collapses old tool-call groups into one short assistant message (`[Tool Calls]` and the results); `ToolCallFormatter` customises it. The gentlest. |
| `SummarizationCompactionStrategy(chatClient, trigger)` | 8 groups | Replaces older groups with a model-written summary (`SummarizationPrompt`). |
| `SlidingWindowCompactionStrategy(trigger)` | 1 turn | Drops the oldest whole user turns. Predictable. |
| `TruncationCompactionStrategy(trigger)` | 32 groups | Drops the oldest groups. The blunt last resort. |
| `ChatReducerCompactionStrategy(reducer, trigger)` | | Applies any `IChatReducer`. |
| `PipelineCompactionStrategy(strategies)` | | Runs strategies in order, each on the result of the last. |
| `ContextWindowCompactionStrategy(maxContextWindowTokens, maxOutputTokens)` | | A ready-made pipeline: tool-result eviction above 50% of the input budget, truncation above 80%. |

Triggers come from `CompactionTriggers`: `TokensExceed(n)`, `TokensBelow(n)`, `MessagesExceed(n)`, `TurnsExceed(n)`, `GroupsExceed(n)`, `HasToolCalls()`, `All(...)`, `Any(...)`, `Always` and `Never`. `strategy.AsChatReducer()` turns any strategy into an `IChatReducer`, and the static `CompactionProvider.CompactAsync(strategy, messages)` compacts a list directly (for maintenance jobs on stored history).

Strategies are applied through `CompactionProvider`, a context provider. Unlike a reducer, it is **not destructive**: it changes what is *sent* to the model, while the session's history keeps every message. Add it to `ChatClientAgentOptions.AIContextProviders`, where it compacts the conversation once at the start of each run:

```csharp
#pragma warning disable MAAI001
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Compaction;
using Microsoft.Extensions.AI;
using OpenAI;

IChatClient model = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai%%")
    .AsIChatClient();

IChatClient summariser = new OpenAIClient(Environment.GetEnvironmentVariable("OPENAI_API_KEY")!)
    .GetChatClient("%%model=openai-small%%")
    .AsIChatClient();

var compaction = new PipelineCompactionStrategy(
[
    // Small thresholds so the effect shows in a short demo; size them to your model in production.
    new ToolResultCompactionStrategy(CompactionTriggers.MessagesExceed(8), minimumPreservedGroups: 4), // 1. shrink old tool output
    new SummarizationCompactionStrategy(summariser, CompactionTriggers.TokensExceed(4_000)),          // 2. summarise
    new SlidingWindowCompactionStrategy(CompactionTriggers.TurnsExceed(4)),                         // 3. bound turns
    new TruncationCompactionStrategy(CompactionTriggers.TokensExceed(32_000)),                      // 4. hard limit
]);

int sent = 0;
AIAgent agent = model.AsBuilder()
    .Use((messages, options, next, ct) => { sent = messages.Count(); return next(messages, options, ct); })   // count what reaches the model
    .BuildAIAgent(new ChatClientAgentOptions
    {
        ChatOptions = new()
        {
            Instructions = "You are a shopping assistant.",
            Tools = [AIFunctionFactory.Create(Prices.Lookup)],
        },
        AIContextProviders = [new CompactionProvider(compaction)],
    });

AgentSession session = await agent.CreateSessionAsync();
for (int i = 1; i <= 8; i++)
{
    await agent.RunAsync($"Question {i}: what does a mouse cost?", session);
    session.TryGetInMemoryChatHistory(out var stored);
    Console.WriteLine($"turn {i}: sent {sent}, stored {stored!.Count}");
}

public static class Prices
{
    [Description("Look up a product price.")]
    public static string Lookup(string product) => $"{product}: 29.99";
}
```

```text
turn 1: sent 3, stored 4
turn 2: sent 7, stored 8
turn 3: sent 10, stored 12
turn 4: sent 13, stored 16
turn 5: sent 13, stored 20
turn 6: sent 13, stored 24
turn 7: sent 13, stored 28
turn 8: sent 13, stored 32
```

("sent" is the size of the last model call of each turn, the one that carries the tool result.)

The stored history keeps every message, while the number sent to the model stops growing: old tool calls are collapsed into one-line summaries, and once there are more than four turns the oldest are dropped.

**Do not put `CompactionProvider` on the chat client in %%version=Microsoft.Agents.AI%%.** Microsoft's compaction sample attaches it with `ChatClientBuilder.UseAIContextProviders(...)`, so that it also runs between tool-loop iterations. In this release that combination loses data: when a run makes a tool call, the provider re-tags the caller's input messages, in place, as `ChatHistory` messages (source `Microsoft.Agents.AI.Compaction.CompactionProvider`), and the history provider's default filter then declines to store them. Every user message of a turn that used a tool disappears from the session, even with a `Never` trigger. Attached through `ChatClientAgentOptions.AIContextProviders`, as above, the history is stored correctly; the cost is that a single long tool loop is not compacted until the next run. Two things to weigh: the summary produced by `SummarizationCompactionStrategy` is model output that then sits in the context as if it were fact, so point it at a model you trust as much as the main one; and token counts are estimates unless you construct the index with a `Tokenizer`.

`HarnessAgent` (`Microsoft.Agents.AI.Harness`) wires a `ContextWindowCompactionStrategy` in for you when you set `MaxContextWindowTokens` and `MaxOutputTokens`; see the [comprehensive guide](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_comprehensive_guide/#harnessagent).

## Differences from Python

| Topic | Python | .NET |
| --- | --- | --- |
| Session | `AgentSession` with a `state` dict | `AgentSession` with an `AgentSessionStateBag` of JSON values |
| History seam | `HistoryProvider` (`InMemoryHistoryProvider`, `FileHistoryProvider`, `VectorStoreHistoryProvider`, Redis, Cosmos DB) | `ChatHistoryProvider` (`InMemoryChatHistoryProvider`, `CosmosChatHistoryProvider`, `ValkeyChatHistoryProvider`) |
| Context seam | `ContextProvider` with `before_run` / `after_run` | `AIContextProvider` with `ProvideAIContextAsync` / `StoreAIContextAsync`, `MessageAIContextProvider` |
| Serialising | `session.to_dict()` / `AgentSession.from_dict()` | `agent.SerializeSessionAsync(session)` / `agent.DeserializeSessionAsync(json)` (the agent does it) |
| Session stores | `FileSessionStore` and others | `AgentSessionStore` (`InMemory`, `Noop`, `AzureBlob`, isolation decorator); `DeleteSessionAsync` required since 1.26.0 |
| Compaction | Compaction strategies on the agent | `Microsoft.Agents.AI.Compaction` strategies via `CompactionProvider`, plus `Microsoft.Extensions.AI` reducers |
| Message provenance | Source IDs on messages | `AgentRequestMessageSourceType` tags and storage filters |

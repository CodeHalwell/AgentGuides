---
title: "Microsoft Agent Framework (.NET) — Testing and Evaluation"
description: "Testing .NET agents and workflows without a model: a scripted IChatClient, tests for replies, tools, structured output, sessions, streaming and tool approval, workflow tests for routing, request ports, checkpoints and errors, ASP.NET Core endpoint tests with TestServer, telemetry assertions, LocalEvaluator and EvalChecks quality gates, model-graded evaluation with Microsoft.Extensions.AI.Evaluation, reports, and workflow evaluation."
framework: "microsoft-agent-framework"
language: "dotnet"
sidebar:
  label: "Testing and evaluation"
  order: 14
---

Everything above the model in this framework is ordinary .NET code: the tool loop, sessions, structured output, middleware, workflows and the hosting endpoints all run unchanged in a unit test once the model is replaced. The model is an `IChatClient` (from Microsoft.Extensions.AI), an interface with two methods, so a test double that replays scripted replies is about forty lines and needs no mocking library. Tests then run in milliseconds, with no network or API key, and assert on exactly what the agent sent to the model.

Evaluation is the other half: checking the quality of real model output. The framework's `LocalEvaluator` runs deterministic checks (keywords, expected tool calls, your own predicates), and `Microsoft.Extensions.AI.Evaluation` adds model-graded quality evaluators and HTML reports.

Every test on this page passes against `Microsoft.Agents.AI` %%version=Microsoft.Agents.AI%% (checked %%versions-checked%%) with xUnit v3; 19 tests in all, run with `dotnet test`.

## Setting up

A test project for .NET 10 with xUnit v3 runs on Microsoft.Testing.Platform. Reference the packages your code uses plus these:

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net10.0</TargetFramework>
    <OutputType>Exe</OutputType>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
    <!-- Experimental APIs used on this page: hosting session stores, evaluation, approvals -->
    <NoWarn>$(NoWarn);MAAI001;MEAI001;AIEVAL001</NoWarn>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="xunit.v3" Version="%%version=xunit.v3%%" />
    <PackageReference Include="Microsoft.AspNetCore.TestHost" Version="%%version=Microsoft.AspNetCore.TestHost%%" />
    <PackageReference Include="OpenTelemetry.Exporter.InMemory" Version="%%version=OpenTelemetry.Exporter.InMemory%%" />
    <PackageReference Include="Microsoft.Extensions.AI.Evaluation.Quality" Version="%%version=Microsoft.Extensions.AI.Evaluation.Quality%%" />
    <PackageReference Include="Microsoft.Extensions.AI.Evaluation.Reporting" Version="%%version=Microsoft.Extensions.AI.Evaluation.Reporting%%" />
    <!-- plus Microsoft.Agents.AI, .Workflows, .Hosting and .Hosting.OpenAI, as in the app under test -->
    <FrameworkReference Include="Microsoft.AspNetCore.App" />
  </ItemGroup>
</Project>
```

`dotnet test` on the .NET 10 SDK needs to be told to use Microsoft.Testing.Platform, or it stops with `Testing with VSTest target is no longer supported`. Add a `global.json` next to the solution:

```json
{ "test": { "runner": "Microsoft.Testing.Platform" } }
```

```bash
dotnet test
```

## A scripted chat client

The double holds a queue of scripted steps. Each request records the messages and options the agent sent, then takes the next step: a text reply, a tool call, or any function of the request. An unscripted request throws, so a test fails loudly if the agent calls the model more often than expected.

```csharp
// ScriptedChatClient.cs: a test double for IChatClient that replays scripted replies and records every request.
using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;

public sealed class ScriptedChatClient : IChatClient
{
    private readonly Queue<Func<IReadOnlyList<ChatMessage>, ChatOptions?, ChatResponse>> _script = new();

    /// Every request the agent made, in order: the messages it sent and the options (instructions, tools, format).
    public List<(IReadOnlyList<ChatMessage> Messages, ChatOptions? Options)> Requests { get; } = [];

    public ScriptedChatClient Reply(string text) =>
        this.Then((_, _) => new ChatResponse(new ChatMessage(ChatRole.Assistant, text)));

    public ScriptedChatClient CallTool(string name, Dictionary<string, object?>? arguments = null, string callId = "call-1") =>
        this.Then((_, _) => new ChatResponse(new ChatMessage(ChatRole.Assistant, [new FunctionCallContent(callId, name, arguments)])));

    public ScriptedChatClient Then(Func<IReadOnlyList<ChatMessage>, ChatOptions?, ChatResponse> step)
    {
        this._script.Enqueue(step);
        return this;
    }

    public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        List<ChatMessage> sent = [.. messages];
        this.Requests.Add((sent, options));
        if (this._script.Count == 0)
        {
            throw new InvalidOperationException($"Unscripted request #{this.Requests.Count}: {sent.LastOrDefault()?.Text}");
        }

        ChatResponse response = this._script.Dequeue()(sent, options);
        response.Usage ??= new UsageDetails { InputTokenCount = 10, OutputTokenCount = 5, TotalTokenCount = 15 };
        return Task.FromResult(response);
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages, ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ChatResponse response = await this.GetResponseAsync(messages, options, cancellationToken);
        foreach (ChatResponseUpdate update in response.ToChatResponseUpdates())
        {
            yield return update;
        }
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => serviceType.IsInstanceOfType(this) ? this : null;

    public void Dispose() { }
}
```

Because a `ChatClientAgent` wraps the client you give it with function invocation, a scripted `FunctionCallContent` makes the agent run the real tool and send its result back in the next request, exactly as with a real model. The client does not need to know about tools.

Python's tests usually patch the chat client class; Microsoft.Extensions.AI's interface makes that unnecessary in .NET. If you prefer a mocking library, `IChatClient` mocks cleanly with Moq or NSubstitute, but a hand-written double is easier to read in failure output.

## Testing agents

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Xunit;

public sealed record Classification(string Category, int Priority);

public class AgentTests
{
    [Fact]
    public async Task Sends_instructions_and_returns_the_reply()
    {
        var chat = new ScriptedChatClient().Reply("Hello, Ana.");
        AIAgent agent = new ChatClientAgent(chat, "Greet the user by name.", name: "greeter");

        AgentResponse response = await agent.RunAsync("I'm Ana.", cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal("Hello, Ana.", response.Text);
        Assert.Equal("Greet the user by name.", chat.Requests[0].Options?.Instructions);
        Assert.Equal("I'm Ana.", chat.Requests[0].Messages.Last().Text);
    }

    [Description("Look up an order.")]
    private static string GetOrder(string id) => $"Order {id}: shipped";

    [Fact]
    public async Task Runs_the_tool_the_model_asks_for_and_sends_back_the_result()
    {
        var chat = new ScriptedChatClient()
            .CallTool(nameof(GetOrder), new() { ["id"] = "A-1" })
            .Reply("Your order has shipped.");
        AIAgent agent = new ChatClientAgent(chat, "You track orders.", tools: [AIFunctionFactory.Create(GetOrder)]);

        AgentResponse response = await agent.RunAsync("Where is A-1?", cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal("Your order has shipped.", response.Text);
        Assert.Equal(2, chat.Requests.Count);   // the tool loop called the model twice
        FunctionResultContent result = chat.Requests[1].Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single();
        Assert.Equal("Order A-1: shipped", result.Result?.ToString());
    }

    [Fact]
    public async Task Structured_output_is_requested_as_a_json_schema_and_parsed()
    {
        var chat = new ScriptedChatClient().Reply("""{"category":"billing","priority":2}""");
        AIAgent agent = new ChatClientAgent(chat, "Classify tickets.");

        AgentResponse<Classification> response = await agent.RunAsync<Classification>("I was charged twice.", cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal(new Classification("billing", 2), response.Result);
        Assert.IsType<ChatResponseFormatJson>(chat.Requests[0].Options?.ResponseFormat);
    }

    [Fact]
    public async Task A_session_carries_history_into_the_next_turn()
    {
        var chat = new ScriptedChatClient().Reply("Noted.").Reply("Your name is Ana.");
        AIAgent agent = new ChatClientAgent(chat, "You remember things.");
        AgentSession session = await agent.CreateSessionAsync(TestContext.Current.CancellationToken);

        await agent.RunAsync("My name is Ana.", session, cancellationToken: TestContext.Current.CancellationToken);
        await agent.RunAsync("What is my name?", session, cancellationToken: TestContext.Current.CancellationToken);

        // Second request: user, assistant, user (instructions travel in ChatOptions, not as a message).
        Assert.Equal(["My name is Ana.", "Noted.", "What is my name?"], chat.Requests[1].Messages.Select(m => m.Text));
    }

    [Fact]
    public async Task Streaming_yields_the_same_text()
    {
        var chat = new ScriptedChatClient().Reply("streamed reply");
        AIAgent agent = new ChatClientAgent(chat, "Be brief.");

        string text = string.Concat(await agent.RunStreamingAsync("Hi", cancellationToken: TestContext.Current.CancellationToken)
            .Select(u => u.Text).ToListAsync(TestContext.Current.CancellationToken));

        Assert.Equal("streamed reply", text);
    }

    [Fact]
    public async Task A_tool_that_needs_approval_stops_the_run_until_it_is_approved()
    {
        var chat = new ScriptedChatClient()
            .CallTool(nameof(GetOrder), new() { ["id"] = "B-2" })
            .Reply("Done.");
        AIAgent agent = new ChatClientAgent(chat, "You track orders.",
            tools: [new ApprovalRequiredAIFunction(AIFunctionFactory.Create(GetOrder))]);
        AgentSession session = await agent.CreateSessionAsync(TestContext.Current.CancellationToken);

        AgentResponse first = await agent.RunAsync("Check B-2", session, cancellationToken: TestContext.Current.CancellationToken);
        ToolApprovalRequestContent request = first.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>().Single();
        Assert.Single(chat.Requests);   // the tool has not run

        AgentResponse second = await agent.RunAsync(
            new ChatMessage(ChatRole.User, [request.CreateResponse(approved: true)]), session,
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal("Done.", second.Text);
        Assert.Contains(chat.Requests[1].Messages.SelectMany(m => m.Contents), c => c is FunctionResultContent);
    }
}
```

What these tests pin down:

- **Instructions** reach the model as `ChatOptions.Instructions`, not as a system message, so assert on `Options?.Instructions`.
- **The tool loop** makes one request per model turn. Counting `Requests` is the simplest check that a tool ran, and the `FunctionResultContent` in the second request shows what the model was told.
- **Structured output** sends a `ChatResponseFormatJson` with the schema for `T` and parses the reply. Parsing happens when you read `response.Result`, not in `RunAsync<T>`: a reply that is not valid JSON for `T` throws `JsonException` from the `Result` getter.
- **Sessions** keep the earlier exchange and send it again on the next turn.
- **Tool approval** returns a `ToolApprovalRequestContent` instead of running the tool. Answer with `request.CreateResponse(approved)` in a user message on the same session to continue.

For middleware and context providers, the same double works: wrap the agent with `AsBuilder().Use(...)` and assert on what reached `Requests`. Running a single tool on its own needs no agent at all: `await AIFunctionFactory.Create(method).InvokeAsync(new AIFunctionArguments { ["id"] = "A-1" })`.

## Testing workflows

Workflows run in-process, so a test starts a run and inspects the events. Build the workflow in a factory method, so every test gets fresh executor instances (instance fields carry over between runs of one `Workflow` object, as the [workflows page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#executors-that-send-yield-and-loop) explains).

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using Xunit;

public record Refund(string OrderId, decimal Amount);
public record RefundDecision(bool Approved);

public static class RefundWorkflow
{
    public const decimal AutoApproveLimit = 50m;

    // A factory, so each test gets fresh executor instances.
    public static Workflow Build()
    {
        ExecutorBinding intake = ((Func<Refund, Refund>)(r => r)).BindAsExecutor("intake");
        ExecutorBinding auto = ((Func<Refund, string>)(r => $"auto-approved {r.OrderId}")).BindAsExecutor("auto");
        RequestPort review = RequestPort.Create<Refund, RefundDecision>("review");
        ExecutorBinding settle = ((Func<RefundDecision, string>)(d => d.Approved ? "approved by reviewer" : "rejected by reviewer")).BindAsExecutor("settle");

        return new WorkflowBuilder(intake)
            .AddSwitch(intake, sw => sw
                .AddCase<Refund>(r => r!.Amount <= AutoApproveLimit, auto)
                .WithDefault(review))
            .AddEdge(review, settle)
            .WithOutputFrom(auto, settle)
            .Build();
    }
}

public class WorkflowTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Fact]
    public async Task Small_refunds_are_approved_without_a_reviewer()
    {
        await using Run run = await InProcessExecution.RunAsync(RefundWorkflow.Build(), new Refund("A-1", 20m), cancellationToken: Ct);

        Assert.Equal(RunStatus.Idle, await run.GetStatusAsync(Ct));
        Assert.Empty(run.OutgoingEvents.OfType<RequestInfoEvent>());
        Assert.Equal("auto-approved A-1", run.OutgoingEvents.OfType<WorkflowOutputEvent>().Single().Data);
    }

    [Theory]
    [InlineData(true, "approved by reviewer")]
    [InlineData(false, "rejected by reviewer")]
    public async Task Large_refunds_wait_for_a_reviewer(bool approved, string expected)
    {
        await using Run run = await InProcessExecution.RunAsync(RefundWorkflow.Build(), new Refund("B-2", 500m), cancellationToken: Ct);

        Assert.Equal(RunStatus.PendingRequests, await run.GetStatusAsync(Ct));
        ExternalRequest request = run.OutgoingEvents.OfType<RequestInfoEvent>().Single().Request;
        Assert.True(request.TryGetDataAs(out Refund? asked));
        Assert.Equal("B-2", asked.OrderId);

        await run.ResumeAsync([request.CreateResponse(new RefundDecision(approved))], Ct);

        Assert.Equal(expected, run.OutgoingEvents.OfType<WorkflowOutputEvent>().Single().Data);
    }

    [Fact]
    public async Task A_pending_review_survives_a_checkpoint_and_resume()
    {
        CheckpointManager checkpoints = CheckpointManager.CreateInMemory();
        CheckpointInfo saved;
        await using (Run first = await InProcessExecution.RunAsync(RefundWorkflow.Build(), new Refund("C-3", 900m), checkpoints, cancellationToken: Ct))
        {
            saved = first.LastCheckpoint!;
        }

        // A new workflow instance, as a new process would build.
        await using StreamingRun resumed = await InProcessExecution.ResumeStreamingAsync(RefundWorkflow.Build(), saved, checkpoints, Ct);
        string? output = null;
        await foreach (WorkflowEvent evt in resumed.WatchStreamAsync(Ct))
        {
            if (evt is RequestInfoEvent pending) await resumed.SendResponseAsync(pending.Request.CreateResponse(new RefundDecision(true)));
            if (evt is WorkflowOutputEvent o) output = o.As<string>();
        }

        Assert.Equal("approved by reviewer", output);
    }

    [Fact]
    public async Task An_executor_exception_ends_the_run_with_error_events()
    {
        ExecutorBinding fail = ((Func<string, string>)(_ => throw new InvalidOperationException("boom"))).BindAsExecutor("fail");
        Workflow workflow = new WorkflowBuilder(fail).WithOutputFrom(fail).Build();

        await using Run run = await InProcessExecution.RunAsync(workflow, "x", cancellationToken: Ct);

        ExecutorFailedEvent failed = run.OutgoingEvents.OfType<ExecutorFailedEvent>().Single();
        Assert.Equal("fail", failed.ExecutorId);
        Assert.IsType<InvalidOperationException>(failed.Data);
        Assert.Equal(RunStatus.Ended, await run.GetStatusAsync(Ct));
    }

    [Fact]
    public async Task Agents_in_a_sequential_workflow_each_get_one_turn()
    {
        var writerModel = new ScriptedChatClient().Reply("draft");
        var editorModel = new ScriptedChatClient().Reply("final");
        Workflow workflow = AgentWorkflowBuilder.BuildSequential(
            new ChatClientAgent(writerModel, "Write.", name: "writer"),
            new ChatClientAgent(editorModel, "Edit.", name: "editor"));

        await using StreamingRun run = await InProcessExecution.RunStreamingAsync(workflow, new List<ChatMessage> { new(ChatRole.User, "topic") }, cancellationToken: Ct);
        await run.TrySendMessageAsync(new TurnToken(emitEvents: false));
        List<ChatMessage> conversation = [];
        await foreach (WorkflowEvent evt in run.WatchStreamAsync(Ct))
        {
            if (evt is WorkflowOutputEvent output && output.Is(out List<ChatMessage>? messages)) conversation = messages;
        }

        Assert.Equal(["topic", "draft", "final"], conversation.Select(m => m.Text));
        Assert.Equal("draft", editorModel.Requests.Single().Messages.Last().Text);   // the editor saw the writer's reply
    }
}
```

The non-streaming `Run` suits most tests: it returns when the workflow halts, `OutgoingEvents` holds everything, and `GetStatusAsync()` tells you whether it finished (`Idle`), is waiting (`PendingRequests`) or failed (`Ended`). Answer requests with `ResumeAsync` and look at the events again. For agent workflows remember the `TurnToken`; with a non-streaming run, deliver it with `ResumeAsync<TurnToken>(...)` as in the evaluation example below.

Executor exceptions do not throw out of `RunAsync`, so a test that only checks for outputs passes quietly when the workflow broke. Assert on `ExecutorFailedEvent` and `WorkflowErrorEvent` explicitly, or check the status.

## Testing hosted endpoints

Register the agents exactly as the app does, swap the `IChatClient` for the scripted one, and serve the app from `Microsoft.AspNetCore.TestHost`'s in-memory server. The request then goes through routing, model binding and the framework's protocol code without a socket.

```csharp
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.Agents.AI.Hosting;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Xunit;

public class HostingTests
{
    [Fact]
    public async Task The_responses_endpoint_runs_the_named_agent()
    {
        var model = new ScriptedChatClient().Reply("Order 7 shipped.");

        // The same registration as the real app, on an in-memory test server.
        WebApplicationBuilder builder = WebApplication.CreateBuilder();
        builder.WebHost.UseTestServer();
        builder.Services.AddChatClient(model);
        builder.AddAIAgent("support", "You answer order questions.");
        builder.AddOpenAIResponses();
        await using WebApplication app = builder.Build();
        app.MapOpenAIResponses();
        await app.StartAsync(TestContext.Current.CancellationToken);

        HttpClient http = app.GetTestClient();
        HttpResponseMessage response = await http.PostAsJsonAsync("/v1/responses",
            new { agent = new { name = "support" }, input = "Where is order 7?" }, TestContext.Current.CancellationToken);

        response.EnsureSuccessStatusCode();
        JsonElement body = await response.Content.ReadFromJsonAsync<JsonElement>(TestContext.Current.CancellationToken);
        Assert.Equal("completed", body.GetProperty("status").GetString());
        Assert.Contains("Order 7 shipped.", body.GetRawText());
        Assert.Equal("You answer order questions.", model.Requests.Single().Options?.Instructions);
    }
}
```

For an app with a `Program.cs` you would rather not restructure, `WebApplicationFactory<Program>` (package `Microsoft.AspNetCore.Mvc.Testing`) does the same; replace the chat client in `ConfigureTestServices`.

## Asserting on telemetry

The in-memory OpenTelemetry exporter collects spans into a list, so a test can check that a run is traced and that no content leaks into spans by default:

```csharp
using System.Diagnostics;
using Microsoft.Agents.AI;
using OpenTelemetry;
using OpenTelemetry.Trace;
using Xunit;

public class TelemetryTests
{
    [Fact]
    public async Task An_agent_run_produces_invoke_agent_and_chat_spans()
    {
        List<Activity> spans = [];
        using TracerProvider tracing = Sdk.CreateTracerProviderBuilder()
            .AddSource("Tests.Agents")
            .AddInMemoryExporter(spans)
            .Build();

        AIAgent agent = new ChatClientAgent(new ScriptedChatClient().Reply("hi"), "Be brief.", name: "greeter")
            .AsBuilder()
            .UseOpenTelemetry("Tests.Agents")
            .Build();

        await agent.RunAsync("Hello", cancellationToken: TestContext.Current.CancellationToken);
        tracing.ForceFlush();

        Activity root = Assert.Single(spans, s => s.DisplayName.StartsWith("invoke_agent greeter"));
        Assert.Equal("greeter", root.GetTagItem("gen_ai.agent.name"));
        Assert.Contains(spans, s => s.GetTagItem("gen_ai.operation.name") as string == "chat" && s.ParentSpanId == root.SpanId);
        Assert.Null(root.GetTagItem("gen_ai.input.messages"));   // no content unless EnableSensitiveData
    }
}
```

The span names and attributes are listed on the [observability page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_observability/).

## Evaluating with LocalEvaluator

`LocalEvaluator` (in the core `Microsoft.Agents.AI` package) runs a list of `EvalCheck`s over `EvalItem`s. `agent.EvaluateAsync(queries, evaluator, ...)` runs the agent on each query, turns each run into an `EvalItem` (query, response, full conversation, tools), evaluates them, and returns `AgentEvaluationResults`. With a real model behind the agent this is a quality gate in an ordinary test; here the model is scripted so the example is self-contained.

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Xunit;

public class LocalEvaluationTests
{
    [Description("Get the weather for a city.")]
    private static string GetWeather(string city) => $"{city}: 18°C";

    [Fact]
    public async Task Weather_agent_passes_the_deterministic_checks()
    {
        // In CI the agent would use the real model; scripted here so the test is self-contained.
        var chat = new ScriptedChatClient()
            .CallTool(nameof(GetWeather), new() { ["city"] = "Leeds" }).Reply("It is 18°C in Leeds.")
            .CallTool(nameof(GetWeather), new() { ["city"] = "York" }).Reply("York is 18°C.");
        AIAgent agent = new ChatClientAgent(chat, "Report the weather.", name: "weather", tools: [AIFunctionFactory.Create(GetWeather)]);

        var evaluator = new LocalEvaluator(
            EvalChecks.NonEmpty(),
            EvalChecks.ContainsExpected(),                 // response contains expectedOutput (case-insensitive)
            EvalChecks.ToolCalledCheck(nameof(GetWeather)),
            EvalChecks.ToolCallArgsMatch(),               // arguments match expectedToolCalls
            FunctionEvaluator.Create("mentions_celsius", (string response) => response.Contains("°C")));

        AgentEvaluationResults results = await agent.EvaluateAsync(
            queries: ["Weather in Leeds?", "Weather in York?"],
            evaluator,
            expectedOutput: ["Leeds", "York"],
            expectedToolCalls:
            [
                [new ExpectedToolCall(nameof(GetWeather), new Dictionary<string, object> { ["city"] = "Leeds" })],
                [new ExpectedToolCall(nameof(GetWeather), new Dictionary<string, object> { ["city"] = "York" })],
            ],
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal(2, results.Total);
        results.AssertAllPassed();   // throws with every failing check and its reason
    }

    [Fact]
    public async Task A_failing_check_reports_why()
    {
        var evaluator = new LocalEvaluator(EvalChecks.KeywordCheck("refund"));
        AgentEvaluationResults results = await evaluator.EvaluateAsync(
            [new EvalItem("Can I get my money back?", "Please contact support.")],
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal(1, results.Failed);
        Assert.False(results.AllPassed);
        var metric = results.Items[0].Metrics.Values.Single();
        Assert.True(metric.Interpretation?.Failed);
        Assert.Contains("refund", metric.Reason);   // e.g. "Missing keywords: refund"
    }
}
```

| Check | Passes when |
| --- | --- |
| `EvalChecks.NonEmpty(minLength)` | The response has at least `minLength` characters |
| `EvalChecks.KeywordCheck([caseSensitive,] keywords...)` | Every keyword appears in the response |
| `EvalChecks.ContainsExpected(caseSensitive)` | The response contains the item's `ExpectedOutput` |
| `EvalChecks.ToolCalledCheck([mode,] names...)` | The named tools were called (`ToolCalledMode` chooses all or any) |
| `EvalChecks.ToolCallsPresent()` | Any tool was called |
| `EvalChecks.ToolCallArgsMatch()` | The calls match `ExpectedToolCalls`, name and arguments |
| `EvalChecks.HasImageContent()` | The conversation contains an image |
| `FunctionEvaluator.Create(name, func)` | Your predicate over the response, the response and expected output, or the whole `EvalItem` returns true (or a passing `EvalCheckResult`) |

`EvaluateAsync` also takes `expectedOutput` and `expectedToolCalls` (one entry per query), a conversation `splitter` (`ConversationSplitters.LastTurn` by default, or `Full`) and `numRepetitions` to run each query several times and catch flaky behaviour. An overload evaluates `AgentResponse`s you already have instead of running the agent.

The results expose `Passed`, `Failed`, `Total`, `AllPassed`, `Items` (one Microsoft.Extensions.AI `EvaluationResult` per item, with one metric per check) and `InputItems`. `AssertAllPassed()` throws with the failing checks; `AssertNoFailedItems()`, `AssertScoreAtLeast(min)` and `AssertDimensionScoreAtLeast(dimension, min)` cover scored evaluators. A failed check's reason is on the metric: `metric.Reason` (for example `Missing keywords: refund`), with `metric.Interpretation.Failed` set.

Metrics are keyed by check name, and the built-in checks use fixed names (`keyword_check`, `non_empty`, ...). Two `KeywordCheck`s in one evaluator therefore overwrite each other's result; combine the keywords into one check or wrap one in `FunctionEvaluator.Create` with its own name.

## Model-graded evaluation

`Microsoft.Extensions.AI.Evaluation.Quality` provides evaluators that ask a second model to grade a response: `RelevanceEvaluator`, `CoherenceEvaluator`, `FluencyEvaluator`, `GroundednessEvaluator`, `CompletenessEvaluator`, `EquivalenceEvaluator`, `RetrievalEvaluator`, and the experimental `IntentResolutionEvaluator`, `TaskAdherenceEvaluator`, `ToolCallAccuracyEvaluator` and `RelevanceTruthAndCompletenessEvaluator`. They return `NumericMetric`s scored 1 to 5, with an interpretation attached; a score of 2, for example, is rated `Poor` and marked failed, so `results.Passed` counts it as a failure. `agent.EvaluateAsync` accepts any of them (any Microsoft.Extensions.AI `IEvaluator`) with a `ChatConfiguration` naming the judge model.

`Microsoft.Extensions.AI.Evaluation.Reporting` stores results and caches judge responses on disk (`DiskBasedReportingConfiguration`), so a nightly run can be compared with earlier ones; `dotnet tool install Microsoft.Extensions.AI.Evaluation.Console` adds `dotnet aieval report`, which renders the stored results as an HTML report.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.AI.Evaluation;
using Microsoft.Extensions.AI.Evaluation.Quality;
using Microsoft.Extensions.AI.Evaluation.Reporting;
using Microsoft.Extensions.AI.Evaluation.Reporting.Storage;
using Xunit;

public class ModelGradedEvaluationTests
{
    // In a real suite both are model clients: the agent's model, and a strong model as the judge.
    // Scripted here: the judge answers in the <S0>/<S1>/<S2> format the quality evaluators ask for.
    // Each evaluator makes its own judge call, so script one reply per evaluator.
    private static IChatClient Judge(int score, int calls = 1)
    {
        var judge = new ScriptedChatClient();
        for (int i = 0; i < calls; i++)
        {
            judge.Reply($"<S0>The response addresses the question.</S0><S1>Relevant and direct.</S1><S2>{score}</S2>");
        }
        return judge;
    }

    [Fact]
    public async Task Relevance_is_judged_by_a_second_model()
    {
        AIAgent agent = new ChatClientAgent(new ScriptedChatClient().Reply("Returns are accepted within 30 days."), "You explain the returns policy.");

        AgentEvaluationResults results = await agent.EvaluateAsync(
            ["How long do I have to return an item?"],
            new RelevanceEvaluator(),
            new ChatConfiguration(Judge(score: 5)),
            cancellationToken: TestContext.Current.CancellationToken);

        NumericMetric relevance = results.Items[0].Get<NumericMetric>(RelevanceEvaluator.RelevanceMetricName);
        Assert.Equal(5, relevance.Value);   // 1 to 5
        Assert.False(relevance.Interpretation?.Failed);
    }

    [Fact]
    public async Task Results_can_be_stored_for_the_aieval_report()
    {
        string storage = Path.Combine(Path.GetTempPath(), "aieval-results");

        // Stores every result (and caches judge responses) under storage; `dotnet aieval report` renders it as HTML.
        ReportingConfiguration reporting = DiskBasedReportingConfiguration.Create(
            storageRootPath: storage,
            evaluators: [new RelevanceEvaluator(), new CoherenceEvaluator()],
            chatConfiguration: new ChatConfiguration(Judge(score: 4, calls: 2)),
            enableResponseCaching: false,
            executionName: "local");

        await using (ScenarioRun scenario = await reporting.CreateScenarioRunAsync("returns-policy", cancellationToken: TestContext.Current.CancellationToken))
        {
            AgentResponse answer = await new ChatClientAgent(new ScriptedChatClient().Reply("Within 30 days, with a receipt."))
                .RunAsync("How long do I have to return an item?", cancellationToken: TestContext.Current.CancellationToken);

            EvaluationResult result = await scenario.EvaluateAsync(
                new ChatMessage(ChatRole.User, "How long do I have to return an item?"),
                answer.AsChatResponse(),
                cancellationToken: TestContext.Current.CancellationToken);

            Assert.Equal(4, result.Get<NumericMetric>(CoherenceEvaluator.CoherenceMetricName).Value);
        }

        var stored = new DiskBasedResultStore(storage);
        Assert.NotEmpty(await stored.ReadResultsAsync("local", cancellationToken: TestContext.Current.CancellationToken).ToListAsync(TestContext.Current.CancellationToken));
    }
}
```

In a real suite the judge is a capable model and the scores vary between runs, so assert on thresholds (`>= 4`) rather than exact values, keep model-graded evaluation out of the pull-request gate, and run it on a schedule. Response caching (on by default) makes reruns over unchanged responses free. Foundry's hosted evaluators are another `IAgentEvaluator`, `FoundryEvals` in `Microsoft.Agents.AI.Foundry`, and plug into the same `EvaluateAsync` calls.

## Evaluating workflows

`run.EvaluateAsync(evaluator)` evaluates a completed non-streaming `Run`: the workflow's final answer as one item and, by default, each agent's turns as `SubResults` keyed by executor ID.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using Xunit;

public class WorkflowEvaluationTests
{
    [Fact]
    public async Task The_workflow_final_answer_is_evaluated()
    {
        // The evaluator needs AgentResponseEvents, so bind the agents with EmitAgentResponseEvents.
        var options = new AIAgentHostOptions { EmitAgentResponseEvents = true };
        ExecutorBinding planner = new ChatClientAgent(new ScriptedChatClient().Reply("Plan: book a train, then a hotel."), "Plan trips.", name: "planner")
            .BindAsExecutor(options);
        ExecutorBinding booker = new ChatClientAgent(new ScriptedChatClient().Reply("Booked the train and the hotel."), "Book the plan.", name: "booker")
            .BindAsExecutor(options);
        Workflow workflow = new WorkflowBuilder(planner).AddEdge(planner, booker).Build();

        // Run.EvaluateAsync reads the events of a non-streaming Run. The run stops before the agents
        // start; ResumeAsync delivers the TurnToken that starts them.
        await using Run run = await InProcessExecution.RunAsync(
            workflow, new List<ChatMessage> { new(ChatRole.User, "A weekend in York") }, cancellationToken: TestContext.Current.CancellationToken);
        await run.ResumeAsync<TurnToken>(TestContext.Current.CancellationToken, [new TurnToken(emitEvents: false)]);

        AgentEvaluationResults results = await run.EvaluateAsync(
            new LocalEvaluator(EvalChecks.NonEmpty(), EvalChecks.KeywordCheck("booked")),
            includePerAgent: false,   // see the note below: per-agent items are empty for agent nodes
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal("Booked the train and the hotel.", results.InputItems?[0].Response);
        results.AssertAllPassed();
    }
}
```

Two requirements are easy to miss. The evaluator looks for `AgentResponseEvent`s, so agent nodes must be bound with `EmitAgentResponseEvents = true`; otherwise it throws `Cannot evaluate the overall workflow output`. And in 1.26.0 the per-agent breakdown pairs each agent node's invocation with its completion event, which for agent nodes carries no response, so every per-agent item has an empty response and fails `NonEmpty`. Pass `includePerAgent: false` for workflows of agent nodes, as above; per-agent results work for your own executors whose handlers return an `AgentResponse`, `ChatMessage` or `string`. Microsoft's `Evaluation_WorkflowEval` sample, which runs a two-agent graph without host options or a turn token, throws for the first reason.

## Running the suite in CI

- Keep unit tests on the scripted client: they run on every push, need no secrets and are deterministic.
- Put live tests (real model, `LocalEvaluator` gates) in a separate class or project and skip them unless a key is present, for example with `Assert.SkipUnless(Environment.GetEnvironmentVariable("OPENAI_API_KEY") is not null, "no API key")` in xUnit v3.
- Run model-graded evaluation and reports nightly against the production model deployment, and publish the `aieval` report as a build artefact.
- Use `TestContext.Current.CancellationToken` everywhere (xUnit v3 warns otherwise), so a hung model call fails the test instead of the build.

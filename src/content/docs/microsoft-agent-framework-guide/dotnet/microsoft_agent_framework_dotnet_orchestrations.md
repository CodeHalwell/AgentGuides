---
title: "Microsoft Agent Framework (.NET) — Orchestrations"
description: "The prebuilt multi-agent patterns in Microsoft.Agents.AI.Workflows: sequential and chain-only pipelines, concurrent fan-out with custom aggregators, group chat with round-robin and custom managers, handoff with return-to-previous and autonomous mode, Magentic with plan sign-off, tool approval inside orchestrations, and orchestrations as agents."
framework: "microsoft-agent-framework"
language: "dotnet"
sidebar:
  label: "Orchestrations"
  order: 11
---

`AgentWorkflowBuilder` assembles a complete [workflow](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/) from a list of agents. Each pattern returns an ordinary `Workflow`, so you run it, stream it, answer its requests, checkpoint it and wrap it as an agent exactly as on the workflows page. Participants are any `AIAgent`: a `ChatClientAgent`, an A2A or Foundry agent, or another workflow wrapped with `AsAIAgent()`.

Every C# block on this page compiles against `Microsoft.Agents.AI.Workflows` %%version=Microsoft.Agents.AI%% (checked %%versions-checked%%) and was run against a local stand-in for the OpenAI API, so the replies are placeholders (`Mock reply to: ...`), not model output. The stand-in answers structured-output requests with schema-shaped sample values, which is why some branches below take the first enum value or finish early.

```bash
dotnet add package Microsoft.Agents.AI.Workflows --version %%version=Microsoft.Agents.AI%%
dotnet add package Microsoft.Agents.AI.OpenAI --version %%version=Microsoft.Agents.AI%%
```

## Choosing a pattern

| Pattern | Entry point | Who speaks next | Output |
| --- | --- | --- | --- |
| Sequential | `AgentWorkflowBuilder.BuildSequential`, `SequentialWorkflowBuilder` | The next agent in the list | The conversation: input plus one reply per agent |
| Concurrent | `BuildConcurrent`, `ConcurrentWorkflowBuilder` | Every agent at once | Whatever the aggregator returns; by default each agent's last message |
| Group chat | `CreateGroupChatBuilderWith(managerFactory)` | A `GroupChatManager` decides | The whole conversation |
| Handoff | `CreateHandoffBuilderWith(initialAgent)` | The current agent, by calling a handoff function | The conversation after one user turn |
| Magentic | `CreateMagenticBuilderWith(managerAgent)` | A manager agent that plans, tracks progress and replans | The transcript, ending with the manager's final answer |

Use sequential when the steps are known, concurrent for independent opinions on the same input, group chat when agents should iterate on each other's work, handoff for a conversation that moves between specialists, and Magentic for open-ended tasks where the plan itself is part of the work. When none of them fits, build the graph yourself with `WorkflowBuilder` and agent nodes.

### How the .NET builders differ from Python

The patterns match Python's `SequentialBuilder`, `ConcurrentBuilder`, `GroupChatBuilder`, `HandoffBuilder` and `MagenticBuilder` in behaviour, with these differences:

- **Every run needs a `TurnToken`.** The input must be chat messages, and agent nodes do nothing until you send `new TurnToken(emitEvents)` after starting the run. Wrapping the workflow with `AsAIAgent()` does this for you.
- **Group chat has only a round-robin manager built in.** Python also ships an LLM-driven orchestrator agent for group chat; in .NET you write a [`GroupChatManager`](#custom-managers) subclass, which is a few lines.
- **No `with_request_info` on the builders.** Human approval in .NET comes from [tool approval](#tool-approval-inside-orchestrations) on the agents and from Magentic [plan sign-off](#plan-sign-off); for a review step between sequential agents, build the graph with a request port.
- **Handoff has autonomous mode and a termination condition**, but no limit on the number of handoffs in one turn.

The examples use the `Models` helper from the [workflows page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/) and this runner, which starts an orchestration, prints each agent's streamed reply and returns the final conversation:

```csharp
// OrchestrationRunner.cs: start an orchestration, print each agent's streamed reply, return the final conversation.
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

public static class OrchestrationRunner
{
    public static async Task<List<ChatMessage>> RunAsync(Workflow workflow, string task, bool streamUpdates = true)
    {
        List<ChatMessage> conversation = [];
        string? speaking = null;

        await using StreamingRun run = await InProcessExecution.RunStreamingAsync(workflow, new List<ChatMessage> { new(ChatRole.User, task) });
        await run.TrySendMessageAsync(new TurnToken(emitEvents: streamUpdates));   // agents wait for this before calling the model

        await foreach (WorkflowEvent evt in run.WatchStreamAsync())
        {
            switch (evt)
            {
                case AgentResponseUpdateEvent update:
                    if (update.ExecutorId != speaking)
                    {
                        speaking = update.ExecutorId;
                        Console.Write($"\n[{update.Update.AuthorName ?? update.ExecutorId}] ");
                    }
                    Console.Write(update.Update.Text);
                    break;
                case WorkflowOutputEvent output when output.Is(out List<ChatMessage>? messages):
                    conversation = messages;
                    break;
                case WorkflowErrorEvent error:
                    throw new InvalidOperationException("Orchestration failed", error.Exception);
            }
        }

        Console.WriteLine();
        return conversation;
    }
}
```

## Sequential

Each agent receives the conversation so far, adds its reply, and passes it on. The output is a `List<ChatMessage>`: the input messages followed by one reply per agent, each with `AuthorName` set to the agent's name.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

public static class SequentialOrchestration
{
    public static async Task RunAsync()
    {
        IChatClient chatClient = Models.ChatClient();
        AIAgent researcher = new ChatClientAgent(chatClient, "List three facts about the topic.", name: "researcher");
        AIAgent writer = new ChatClientAgent(chatClient, "Write one paragraph from the facts in the conversation.", name: "writer");
        AIAgent editor = new ChatClientAgent(chatClient, "Correct the paragraph's grammar. Reply with the paragraph only.", name: "editor");

        Workflow workflow = AgentWorkflowBuilder.BuildSequential("research-write-edit", researcher, writer, editor);

        List<ChatMessage> conversation = await OrchestrationRunner.RunAsync(workflow, "Heat pumps");

        // The output is the whole conversation: the user's message, then one reply per agent.
        foreach (ChatMessage message in conversation)
        {
            Console.WriteLine($"{message.Role}/{message.AuthorName}: {message.Text}");
        }
    }
}
```

`BuildSequential(true, agents)` (or `new SequentialWorkflowBuilder(agents).WithChainOnlyAgentResponses()`) makes it a chain instead: each agent sees only the previous agent's reply, and the output contains only the last agent's messages. Use it for transformations where earlier context would confuse the next step, such as round-trip translation.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

public static class TranslationChain
{
    public static async Task RunAsync()
    {
        IChatClient chatClient = Models.ChatClient();
        AIAgent[] translators =
        [
            new ChatClientAgent(chatClient, "Translate the message into French. Reply with the translation only.", name: "to-french"),
            new ChatClientAgent(chatClient, "Translate the message into German. Reply with the translation only.", name: "to-german"),
            new ChatClientAgent(chatClient, "Translate the message into English. Reply with the translation only.", name: "to-english"),
        ];

        // Chain-only: each agent sees only the previous agent's reply, not the whole conversation,
        // and the output contains only the last agent's messages.
        Workflow workflow = new SequentialWorkflowBuilder(translators)
            .WithChainOnlyAgentResponses()
            .WithName("round-trip-translation")
            .Build();

        List<ChatMessage> result = await OrchestrationRunner.RunAsync(workflow, "The meeting moved to Thursday.");
        Console.WriteLine($"round trip: {result.Last().Text}");
    }
}
```

Do not share one `AgentSession` between agents in a pipeline. The orchestration manages each agent's session itself.

## Concurrent

Every agent receives the same input in the same superstep and runs in parallel. The aggregator gets one `List<ChatMessage>` per agent and returns the output list. The lists arrive in completion order, not the order the agents were given (in the run above, `cost` came first), so identify agents by `AuthorName`. Without one, the output is the last message from each agent that produced any.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

public static class ConcurrentOrchestration
{
    public static async Task RunAsync()
    {
        IChatClient chatClient = Models.ChatClient();
        AIAgent[] reviewers =
        [
            new ChatClientAgent(chatClient, "Review the proposal for security risks, in two sentences.", name: "security"),
            new ChatClientAgent(chatClient, "Review the proposal for cost, in two sentences.", name: "cost"),
            new ChatClientAgent(chatClient, "Review the proposal for accessibility, in two sentences.", name: "accessibility"),
        ];

        // Every agent gets the same input at the same time. The aggregator receives one message list
        // per agent; the default keeps the last message from each.
        Workflow workflow = AgentWorkflowBuilder.BuildConcurrent(
            "proposal-review",
            reviewers,
            aggregator: perAgent =>
            [
                new ChatMessage(ChatRole.Assistant, string.Join("\n", perAgent.Select(messages => $"- {messages.Last().AuthorName}: {messages.Last().Text}")))
                {
                    AuthorName = "review-summary",
                },
            ]);

        List<ChatMessage> result = await OrchestrationRunner.RunAsync(workflow, "Move the staff portal to a public cloud.", streamUpdates: false);   // interleaved otherwise
        Console.WriteLine(result.Single().Text);
    }
}
```

The agents' streamed updates interleave, so printing them as they arrive produces a mess; the example turns update events off with `streamUpdates: false`. An exception in one agent ends the whole run once the others have finished. If you need partial results when one reviewer fails, catch errors inside that agent's tools or wrap the agent in middleware that returns an error message instead.

## Group chat

A `GroupChatManager` chooses each speaker in turn. All participants share one conversation, and every reply is visible to everyone. `CreateGroupChatBuilderWith` takes a factory that receives the participants and returns the manager, then `AddParticipants` adds the agents.

`RoundRobinGroupChatManager` takes turns in the order the agents were added. `MaximumIterationCount` (default 40, minimum 1) caps the number of agent turns, and the optional `shouldTerminateFunc` ends the chat early. It is checked before each turn alongside the iteration limit.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

public static class RoundRobinChat
{
    public static async Task RunAsync()
    {
        IChatClient chatClient = Models.ChatClient();
        AIAgent writer = new ChatClientAgent(chatClient, "You write slogans. Improve the slogan using the critic's feedback.", name: "writer");
        AIAgent critic = new ChatClientAgent(chatClient, "You critique slogans in one sentence. Say APPROVED when it is good.", name: "critic");

        Workflow workflow = AgentWorkflowBuilder
            .CreateGroupChatBuilderWith(agents => new RoundRobinGroupChatManager(
                agents,
                // Checked before each turn, after the iteration limit.
                shouldTerminateFunc: (manager, history, ct) =>
                    ValueTask.FromResult(history.LastOrDefault()?.Text?.Contains("APPROVED") == true))
            {
                MaximumIterationCount = 6,   // agent turns; the default is 40
            })
            .AddParticipants(writer, critic)
            .WithName("slogan-workshop")
            .Build();

        List<ChatMessage> conversation = await OrchestrationRunner.RunAsync(workflow, "A slogan for a library app.");
        Console.WriteLine($"{conversation.Count(m => m.Role == ChatRole.Assistant)} agent turns");
    }
}
```

### Custom managers

A manager overrides `SelectNextAgentAsync` and, optionally, `ShouldTerminateAsync` (the base implementation enforces `MaximumIterationCount`), `UpdateHistoryAsync` (to filter or rewrite each batch of new messages before it joins the shared history) and `Reset`. `IterationCount` is maintained for you. The manager below asks a moderator agent, with structured output, who should speak next:

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

[Description("Who should speak next in the discussion.")]
public sealed record SpeakerChoice(string NextSpeaker, bool Finished);

// .NET ships only the round-robin manager. A model-driven one is a GroupChatManager subclass:
// SelectNextAgentAsync picks the speaker, ShouldTerminateAsync ends the chat.
public sealed class ModeratorManager(IReadOnlyList<AIAgent> participants, AIAgent moderator) : GroupChatManager
{
    private SpeakerChoice? _lastChoice;

    protected override async ValueTask<AIAgent> SelectNextAgentAsync(IReadOnlyList<ChatMessage> history, CancellationToken cancellationToken = default)
    {
        string roster = string.Join(", ", participants.Select(p => $"{p.Name} ({p.Description})"));
        string transcript = string.Join("\n", history.Select(m => $"{m.AuthorName ?? m.Role.ToString()}: {m.Text}"));

        AgentResponse<SpeakerChoice> choice = await moderator.RunAsync<SpeakerChoice>(
            $"Participants: {roster}\n\nConversation:\n{transcript}\n\nChoose the next speaker, or set Finished when the question is answered.",
            cancellationToken: cancellationToken);
        this._lastChoice = choice.Result;

        // Fall back to the first participant if the model names someone who is not in the chat.
        return participants.FirstOrDefault(p => string.Equals(p.Name, choice.Result.NextSpeaker, StringComparison.OrdinalIgnoreCase))
            ?? participants[0];
    }

    protected override async ValueTask<bool> ShouldTerminateAsync(IReadOnlyList<ChatMessage> history, CancellationToken cancellationToken = default) =>
        this._lastChoice?.Finished == true || await base.ShouldTerminateAsync(history, cancellationToken);

    protected override void Reset()
    {
        base.Reset();   // resets IterationCount
        this._lastChoice = null;
    }
}

public static class ModeratedChat
{
    public static async Task RunAsync()
    {
        IChatClient chatClient = Models.ChatClient();
        AIAgent moderator = new ChatClientAgent(chatClient, "You moderate a panel discussion.", name: "moderator");
        AIAgent[] panel =
        [
            new ChatClientAgent(chatClient, "You are an economist. Answer briefly.", name: "economist", description: "economics"),
            new ChatClientAgent(chatClient, "You are an ecologist. Answer briefly.", name: "ecologist", description: "the environment"),
        ];

        Workflow workflow = AgentWorkflowBuilder
            .CreateGroupChatBuilderWith(agents => new ModeratorManager(agents, moderator) { MaximumIterationCount = 4 })
            .AddParticipants(panel)
            .Build();

        await OrchestrationRunner.RunAsync(workflow, "Should cities charge for road use?");
    }
}
```

Managers are checkpointed with the workflow: the base class saves `IterationCount`, and `OnCheckpointingAsync` / `OnCheckpointRestoredAsync` are where a subclass saves its own fields. `RoundRobinGroupChatManager` saves its position this way. The factory is called once per run (and again when the chat is reset), so returning a manager created outside it (as Microsoft's tool-approval sample does with `_ => manager`) shares one instance, and its fields, between runs; create it inside the factory unless you mean to share it.

## Handoff

A handoff workflow starts with an initial agent and lets agents pass the conversation to one another. The builder adds a handoff function per allowed target to each agent (named `handoff_to_` followed by a number, with the target's description as the function description) plus instructions explaining them; when an agent calls one, the target agent continues the same turn. A turn ends when an agent replies without handing off.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

public static class HelpDeskHandoff
{
    public static Workflow Build(IChatClient chatClient)
    {
        // Handoff targets are described to the model by their Description (or Name), so set it.
        AIAgent triage = new ChatClientAgent(chatClient,
            "You greet the customer and hand off to the right specialist. Do not answer questions yourself.",
            name: "triage", description: "Routes customers to a specialist");
        AIAgent billing = new ChatClientAgent(chatClient,
            "You answer billing questions: invoices, refunds and payment methods.",
            name: "billing", description: "Invoices, refunds and payments");
        AIAgent technical = new ChatClientAgent(chatClient,
            "You fix technical problems with the product.",
            name: "technical", description: "Errors, crashes and set-up problems");

        return AgentWorkflowBuilder.CreateHandoffBuilderWith(triage)
            .WithHandoffs(triage, [billing, technical])           // triage may hand off to either specialist
            .WithHandoffs([billing, technical], triage)           // and each may hand back
            .WithHandoff(billing, technical, "The billing question turns out to be a technical fault")
            .EnableReturnToPrevious()                             // the next user turn goes to the last specialist, not triage
            .Build();
    }

    public static async Task RunAsync()
    {
        Workflow workflow = Build(Models.ChatClient());

        // A handoff workflow handles one user turn per run and returns the conversation.
        // For the next turn, append the user's message and run again with the whole history.
        List<ChatMessage> history = [];
        foreach (string userTurn in new[] { "I was charged twice this month.", "It was on the 3rd and the 4th." })
        {
            history.Add(new ChatMessage(ChatRole.User, userTurn));

            await using StreamingRun run = await InProcessExecution.RunStreamingAsync(workflow, history);
            await run.TrySendMessageAsync(new TurnToken(emitEvents: true));

            await foreach (WorkflowEvent evt in run.WatchStreamAsync())
            {
                switch (evt)
                {
                    case AgentResponseUpdateEvent update:
                        foreach (FunctionCallContent call in update.Update.Contents.OfType<FunctionCallContent>())
                        {
                            Console.WriteLine($"  [{update.ExecutorId.Split('_')[0]} calls {call.Name}]");
                        }
                        break;
                    case WorkflowOutputEvent output when output.Is(out List<ChatMessage>? conversation):
                        history = conversation;
                        Console.WriteLine($"{history.Last().AuthorName}: {history.Last().Text}");
                        break;
                }
            }
        }
    }
}
```

The handoff reason shown to the model is, in order of preference, the `handoffReason` you pass, the target's `Description`, `"handoff to {Name}"`, or the target's instructions; with none of these the builder throws. Registering the same handoff twice throws. `AddParticipants` adds agents without handoff relationships; if no handoffs are declared at all, every participant can hand off to every other.

| Builder method | Effect |
| --- | --- |
| `WithHandoff(from, to, reason)`, `WithHandoffs(from, [to...])`, `WithHandoffs([from...], to, reason)` | Declare which agent may hand off to which |
| `WithHandoffInstructions(text)` | Replace the instructions added to agents with handoffs (`null` restores the default). Explain the `handoff_to_` functions if you replace them |
| `EnableReturnToPrevious()` | Route the next user turn straight to the agent that answered last, instead of the initial agent |
| `WithToolCallFilteringBehavior(...)` | `HandoffOnly` (default) removes handoff calls from the conversation agents see; `All` removes every tool call and result; `None` keeps everything |
| `WithAutonomousMode(turnLimit, continuationPrompt, agents, agentTurnLimits, agentContinuationPrompts)` | Prompt an agent to continue when it replies without handing off, up to a limit (default 50) per agent per user turn |
| `WithTerminationCondition(func)` | End the run when the condition holds after a reply that did not hand off |
| `EmitAgentResponseEvents()`, `EmitAgentResponseUpdateEvents()` | Raise `AgentResponseEvent` per agent turn / `AgentResponseUpdateEvent` per streamed update |

A handoff run handles one user turn. Keep the returned conversation and run the workflow again with the user's next message appended, as above, or wrap it as an agent and let the session do that (see [below](#orchestrations-as-agents)).

There is no limit on the number of handoffs within a turn. Two agents that keep handing the conversation back to each other loop until you stop the run, so write instructions that make specialists answer, and consider a cancellation timeout on the run.

### Autonomous mode

By default an agent that replies without handing off ends the turn. In autonomous mode it is prompted again with the continuation prompt, so an agent can work through several steps before yielding. The loop for an agent ends when it hands off, when the termination condition fires, or at the turn limit.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

public static class AutonomousHandoff
{
    public static async Task RunAsync()
    {
        IChatClient chatClient = Models.ChatClient();
        AIAgent planner = new ChatClientAgent(chatClient, "Break the task into steps, then hand off to the coder.", name: "planner", description: "Plans the work");
        AIAgent coder = new ChatClientAgent(chatClient, "Write the code. Say DONE when finished.", name: "coder", description: "Writes code");

        Workflow workflow = AgentWorkflowBuilder.CreateHandoffBuilderWith(planner)
            .WithHandoff(planner, coder)
            .WithHandoff(coder, planner)
            // Without autonomous mode, an agent that replies without handing off ends the turn.
            // With it, the agent is prompted to continue, up to the turn limit per agent per user turn.
            .WithAutonomousMode(turnLimit: 3, continuationPrompt: "Continue with the next step.")
            // Checked after every reply that does not hand off; true ends the run.
            .WithTerminationCondition(messages => messages.LastOrDefault()?.Text?.Contains("DONE") == true)
            // Keep handoff tool calls out of the conversation the agents see (the default), or None / All.
            .WithToolCallFilteringBehavior(HandoffToolCallFilteringBehavior.HandoffOnly)
            .Build();

        List<ChatMessage> conversation = await OrchestrationRunner.RunAsync(workflow, "A function that checks whether a year is a leap year.");
        Console.WriteLine($"{conversation.Count} messages");
    }
}
```

The turn counters are per agent and reset when that agent hands off, when its loop ends, and at the start of each user turn. Pass `agents:` to enable autonomous mode for only some participants, and `agentTurnLimits` / `agentContinuationPrompts` to vary it per agent.

## Magentic

Magentic (after Microsoft Research's Magentic-One) gives a manager agent a task ledger (facts and a plan) and a progress ledger, which it updates after every participant turn to decide whether the task is done, whether the team is stuck, and who should act next. When progress stalls for `WithMaxStalls` rounds (default 3), the manager replans. At the end it writes a final answer.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Agents.AI.Workflows.Specialized.Magentic;
using Microsoft.Extensions.AI;

public static class MagenticOrchestration
{
    public static async Task RunAsync()
    {
        IChatClient chatClient = Models.ChatClient();
        AIAgent manager = new ChatClientAgent(chatClient, "You coordinate a small research team.", name: "manager");
        AIAgent researcher = new ChatClientAgent(chatClient, "You find facts. No calculations.", name: "researcher", description: "Finds information");
        AIAgent analyst = new ChatClientAgent(chatClient, "You do arithmetic and comparisons.", name: "analyst", description: "Calculates and compares");

        Workflow workflow = AgentWorkflowBuilder.CreateMagenticBuilderWith(manager)
            .AddParticipants([researcher, analyst])
            .RequirePlanSignoff(false)   // true by default: see the next example
            .WithMaxRounds(10)           // null (the default) means no limit
            .WithMaxStalls(3)            // rounds without progress before the manager replans
            .WithMaxResets(2)            // replans allowed; null means no limit
            .Build();

        await using StreamingRun run = await InProcessExecution.RunStreamingAsync(
            workflow, new List<ChatMessage> { new(ChatRole.User, "Compare the land area of Wales and Belgium.") });
        await run.TrySendMessageAsync(new TurnToken(emitEvents: true));

        await foreach (WorkflowEvent evt in run.WatchStreamAsync())
        {
            switch (evt)
            {
                case MagenticPlanCreatedEvent plan:
                    Console.WriteLine($"[plan]\n{plan.FullTaskLedger.Text}");
                    break;
                case MagenticProgressLedgerUpdatedEvent progress:
                    MagenticProgressLedger ledger = progress.ProgressLedger;
                    Console.WriteLine($"[progress] satisfied={ledger.IsRequestSatisfied} next={ledger.NextSpeaker} looping={ledger.IsInLoop}");
                    break;
                case MagenticReplannedEvent replanned:
                    Console.WriteLine($"[replanned]\n{replanned.FullTaskLedger.Text}");
                    break;
                case WorkflowOutputEvent output when output.Is(out List<ChatMessage>? transcript):
                    Console.WriteLine($"[final] {transcript.Last().Text}");
                    break;
                case WorkflowErrorEvent error:
                    Console.WriteLine(error.Exception);
                    break;
            }
        }
    }
}
```

| Builder method | Default | Effect |
| --- | --- | --- |
| `AddParticipants(agents)` | | The team. Give each a `Description`; the manager plans with it |
| `WithMaxRounds(n)` | `null` (no limit) | Coordination rounds before the manager stops |
| `WithMaxStalls(n)` | 3 | Rounds without progress before a replan |
| `WithMaxResets(n)` | `null` (no limit) | Replans allowed |
| `RequirePlanSignoff(bool)` | **`true`** | Ask a person to approve each plan before work starts |
| `WithResponseLanguage(language)` | `null` | The language of the ledgers and final answer (experimental) |
| `WithPromptOverrides(MagenticPromptOverrides)` | built-in prompts | Replace any of the manager's prompts (experimental; `MagenticDefaultPrompts` holds the defaults) |

The events are `MagenticPlanCreatedEvent` and `MagenticReplannedEvent` (with the full task ledger) and `MagenticProgressLedgerUpdatedEvent` (with `IsRequestSatisfied`, `IsInLoop`, `IsProgressBeingMade`, `NextSpeaker` and `InstructionOrQuestion`), all in `Microsoft.Agents.AI.Workflows.Specialized.Magentic`. The manager agent must not have tools; participants may.

Always set `WithMaxRounds` and `WithMaxResets` in production. With the defaults, a task the team cannot finish replans for as long as the model keeps answering.

### Plan sign-off

Plan sign-off is on by default, so a Magentic workflow you do not configure stops at its first plan and waits. The plan arrives as a `RequestInfoEvent` carrying a `MagenticPlanReviewRequest` (`Plan`, `CurrentProgress`, `IsStalled`); answer with `Approve()` or `Revise(feedback)`, and after a revision the manager replans and asks again. Turn it off with `RequirePlanSignoff(false)` for unattended runs.

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

public static class MagenticWithSignoff
{
    public static async Task RunAsync()
    {
        IChatClient chatClient = Models.ChatClient();
        Workflow workflow = AgentWorkflowBuilder
            .CreateMagenticBuilderWith(new ChatClientAgent(chatClient, "You coordinate the team.", name: "manager"))
            .AddParticipants([new ChatClientAgent(chatClient, "You write release notes.", name: "writer", description: "Writes text")])
            .RequirePlanSignoff()   // the default, shown for clarity
            .WithMaxRounds(5)
            .Build();

        await using StreamingRun run = await InProcessExecution.RunStreamingAsync(
            workflow, new List<ChatMessage> { new(ChatRole.User, "Release notes for version 2.1.") });
        await run.TrySendMessageAsync(new TurnToken(emitEvents: false));

        int reviews = 0;
        await foreach (WorkflowEvent evt in run.WatchStreamAsync())
        {
            switch (evt)
            {
                // The plan (and any later replan) arrives as a request; the run waits for the answer.
                case RequestInfoEvent pending when pending.Request.TryGetDataAs(out MagenticPlanReviewRequest? review):
                    Console.WriteLine($"[plan to review, stalled={review.IsStalled}]\n{review.Plan.Text}");
                    MagenticPlanReviewResponse answer = reviews++ == 0
                        ? review.Revise("Add a section on breaking changes.")   // the manager replans and asks again
                        : review.Approve();
                    await run.SendResponseAsync(pending.Request.CreateResponse(answer));
                    break;
                case WorkflowOutputEvent output when output.Is(out List<ChatMessage>? transcript):
                    Console.WriteLine($"[final after {reviews} reviews] {transcript.Last().Text}");
                    break;
            }
        }
    }
}
```

## Tool approval inside orchestrations

A tool wrapped in `ApprovalRequiredAIFunction` behaves the same inside an orchestration as in a single agent, except that the approval request leaves the workflow as a `RequestInfoEvent`. Its data is the `ToolApprovalRequestContent`, the port ID identifies the agent (`deployer_<id>_UserInput`), and you answer with `approval.CreateResponse(approved)` wrapped in `request.CreateResponse(...)`. This works in every pattern: sequential, concurrent, group chat, handoff, and for Magentic participants.

```csharp
using System.ComponentModel;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

public static class DeploymentTools
{
    [Description("Deploy a version to production.")]
    public static string Deploy([Description("The version to deploy")] string version) => $"Deployed {version}.";
}

public static class ApprovalInOrchestration
{
    public static async Task RunAsync()
    {
        IChatClient chatClient = Models.ChatClient();
        AIAgent tester = new ChatClientAgent(chatClient, "Confirm the release is tested.", name: "tester");
        AIAgent deployer = new ChatClientAgent(chatClient, "Deploy the release with the Deploy tool.", name: "deployer",
            tools: [new ApprovalRequiredAIFunction(AIFunctionFactory.Create(DeploymentTools.Deploy))]);

        Workflow workflow = AgentWorkflowBuilder.BuildSequential(tester, deployer);

        await using StreamingRun run = await InProcessExecution.RunStreamingAsync(
            workflow, new List<ChatMessage> { new(ChatRole.User, "Ship version 2.4.0.") });
        await run.TrySendMessageAsync(new TurnToken(emitEvents: false));

        await foreach (WorkflowEvent evt in run.WatchStreamAsync())
        {
            switch (evt)
            {
                // An agent's approval request surfaces as a workflow request; the port ID names the agent.
                case RequestInfoEvent pending when pending.Request.TryGetDataAs(out ToolApprovalRequestContent? approval):
                    var call = (FunctionCallContent)approval.ToolCall;
                    Console.WriteLine($"{pending.Request.PortInfo.PortId} wants to call {call.Name}({string.Join(", ", call.Arguments ?? new Dictionary<string, object?>())})");
                    await run.SendResponseAsync(pending.Request.CreateResponse(approval.CreateResponse(approved: true)));
                    break;
                case WorkflowOutputEvent output when output.Is(out List<ChatMessage>? conversation):
                    Console.WriteLine(conversation.Last().Text);
                    break;
            }
        }
    }
}
```

With checkpointing, the pending approval is part of the checkpoint, so the person can answer after a restart (see [resuming in another process](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_workflows/#resuming-in-another-process)). To handle approvals inside the workflow instead of raising them, bind the agent with `AIAgentHostOptions.InterceptUserInputRequests = true` and route the request to an executor of your own. Tool approval itself is covered on the [comprehensive guide](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_comprehensive_guide/#tool-approval).

## Orchestrations as agents

`workflow.AsAIAgent()` turns any orchestration into an agent. The session then holds the conversation, so a handoff workflow becomes a multi-turn chat where the caller sends only the new message; `WithCheckpointing(manager)` chooses where the workflow's checkpoints go. The result can be registered with `AddWorkflow(...).AddAsAIAgent()` and served over the OpenAI, A2A or AG-UI endpoints on the [hosting page](/microsoft-agent-framework-guide/dotnet/microsoft_agent_framework_dotnet_hosting/).

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

public static class HelpDeskAgent
{
    public static async Task RunAsync()
    {
        // Wrapped as an agent, a handoff workflow keeps its own history in the session,
        // so callers send only the new user message each turn.
        AIAgent helpDesk = HelpDeskHandoff.Build(Models.ChatClient()).AsAIAgent(name: "help-desk")
            .WithCheckpointing(CheckpointManager.CreateInMemory());

        AgentSession session = await helpDesk.CreateSessionAsync();
        foreach (string turn in new[] { "My invoice is wrong.", "Invoice INV-204." })
        {
            AgentResponse response = await helpDesk.RunAsync(turn, session);
            // The response also carries the handoff function call and result; show the last reply with text.
            ChatMessage reply = response.Messages.Last(m => m.Role == ChatRole.Assistant && !string.IsNullOrEmpty(m.Text));
            Console.WriteLine($"{reply.AuthorName}: {reply.Text}");
        }
    }
}
```

An orchestration-as-agent can also be a participant in another orchestration, for example a concurrent review team acting as one member of a sequential pipeline.

## Shaping the output

All builders share `WithName`, `WithDescription`, `WithOutputFrom(agents)` and `WithIntermediateOutputFrom(agents)`. By default the final aggregator is the workflow output and every agent is an intermediate output source. Calling either output method replaces those defaults entirely: only the agents you name produce output. Intermediate outputs carry `OutputTag.Intermediate`; check `output.IsIntermediate()` to tell them from the final result.

## What is not ported

- An LLM-driven group-chat orchestrator (write a [custom manager](#custom-managers)).
- `with_request_info` hooks on the builders; use tool approval, plan sign-off or a hand-built graph with a request port.
- Python's Magentic stall intervention requests; .NET replans automatically after `WithMaxStalls` and reports `IsStalled` on the next plan review.
- A handoff limit per turn.

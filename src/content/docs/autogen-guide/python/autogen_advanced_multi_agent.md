---
title: "AutoGen Advanced Multi‑Agent (Python)"
description: "Microsoft AutoGen 0.7 team patterns — custom speaker selection, Swarm handoffs with a human, GraphFlow review loops, and nesting teams as tools."
framework: autogen
language: python
---

> **Microsoft AutoGen** (`autogen-agentchat` %%version=autogen-agentchat%%, checked %%versions-checked%%) — not AG2. The project is archived; see the [AgentChat & Core reference](/autogen-guide/python/autogen_agentchat_guide/) for status and basics. Examples were run offline with `ReplayChatCompletionClient`.

## Custom speaker selection

`SelectorGroupChat` normally asks a model to choose the next speaker from the agents' `description`s. A `selector_func` can override that choice; returning `None` falls back to the model.

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat


def select_next(messages):
    """Always let the planner answer the user first; otherwise defer to the model."""
    if messages[-1].source == "user":
        return "planner"
    return None


def build_team(model_client) -> SelectorGroupChat:
    planner = AssistantAgent("planner", model_client=model_client, description="Breaks the task into steps and assigns them.")
    analyst = AssistantAgent("analyst", model_client=model_client, description="Runs the numbers for a step.")
    writer = AssistantAgent("writer", model_client=model_client, description="Writes the final answer. Says DONE when finished.")
    return SelectorGroupChat(
        [planner, analyst, writer],
        model_client=model_client,
        selector_func=select_next,
        allow_repeated_speaker=False,
        termination_condition=TextMentionTermination("DONE") | MaxMessageTermination(20),
    )
```

`candidate_func=` narrows the list the model may choose from instead of fixing the answer.

## Swarm with a human checkpoint

In a `Swarm`, agents pass control by calling generated `transfer_to_<name>` tools. Handing off to a name that is not a team member (here `"user"`) stops the team, so a web back end can collect the human's answer and resume.

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import HandoffTermination, MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm


def build_support_swarm(model_client) -> Swarm:
    triage = AssistantAgent(
        "triage",
        model_client=model_client,
        handoffs=["refunds", "user"],
        system_message="Route refund requests to refunds. Hand off to user when you need information. Say TERMINATE when resolved.",
    )
    refunds = AssistantAgent(
        "refunds",
        model_client=model_client,
        handoffs=["triage", "user"],
        system_message="Handle refunds. Ask the user to confirm before refunding. Say TERMINATE when done.",
    )
    termination = HandoffTermination(target="user") | TextMentionTermination("TERMINATE") | MaxMessageTermination(25)
    return Swarm([triage, refunds], termination_condition=termination)


async def handle(team: Swarm, first_message: str, ask_human) -> None:
    result = await team.run(task=first_message)
    while isinstance(result.messages[-1], HandoffMessage) and result.messages[-1].target == "user":
        answer = await ask_human(result.messages[-1].content)
        result = await team.run(task=HandoffMessage(source="user", target=result.messages[-1].source, content=answer))
```

Handoffs require a model client with function calling.

## GraphFlow: review loop with conditional edges

`DiGraphBuilder` edges can carry a condition — a string to look for in the last message, or a callable — which makes loops and branches explicit.

```python
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow


def build_review_loop(writer, reviewer, publisher) -> GraphFlow:
    builder = DiGraphBuilder()
    builder.add_node(writer).add_node(reviewer).add_node(publisher)
    builder.add_edge(writer, reviewer)
    builder.add_edge(reviewer, writer, condition=lambda msg: "REVISE" in msg.to_model_text())
    builder.add_edge(reviewer, publisher, condition=lambda msg: "APPROVE" in msg.to_model_text())
    builder.set_entry_point(writer)
    return GraphFlow(
        participants=builder.get_participants(),
        graph=builder.build(),
        termination_condition=MaxMessageTermination(12),  # bound the loop
    )
```

Fan-out is two edges from one node; a node with `activation="all"` (the default) waits for all incoming edges, `activation="any"` for the first. Give loops a termination condition: a reviewer that never approves would otherwise cycle indefinitely.

## Teams inside teams

Wrap a team as a tool for a coordinating agent with `TeamTool`, or a single agent with `AgentTool`:

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.tools import TeamTool


def build_editor_in_chief(model_client) -> AssistantAgent:
    writer = AssistantAgent("writer", model_client=model_client, system_message="Draft copy.")
    critic = AssistantAgent("critic", model_client=model_client, system_message="Critique; say APPROVE when ready.")
    desk = RoundRobinGroupChat([writer, critic], termination_condition=TextMentionTermination("APPROVE"))
    desk_tool = TeamTool(team=desk, name="copy_desk", description="Drafts and reviews a piece of copy.", return_value_as_last_message=True)
    return AssistantAgent("editor_in_chief", model_client=model_client, tools=[desk_tool])
```

## Escalation and safety

- Put a `UserProxyAgent` in the team, or stop on `HandoffTermination`, wherever a policy or low-confidence decision needs a human.
- `ExternalTermination` lets a UI stop a running team; `CancellationToken` aborts an in-flight `run()`.
- Approve risky tool calls: `CodeExecutorAgent(approval_func=...)` gates code execution. Run generated code in `DockerCommandLineCodeExecutor` (from `autogen_ext.code_executors.docker`), not locally.

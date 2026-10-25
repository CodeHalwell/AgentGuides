---
title: "AG2 Classic Comprehensive Guide"
description: "Reference for the AG2 Classic import autogen API: ConversableAgent, LLMConfig, tools, code execution, structured output, GroupChat, group-chat patterns and handoffs, nested and sequential chats. Not AG2 1.x and not Microsoft AutoGen."
framework: "ag2"
language: "python"
sidebar:
  label: "Comprehensive guide"
---

> **This page covers AG2 Classic**, the `import autogen` API (`ConversableAgent`, `GroupChat`, `initiate_chat`). The AG2 community fork maintains it and publishes it on PyPI as `autogen` (and as `ag2` up to 0.14). Code here was checked against `autogen` %%version=autogen%%. Classic is in maintenance mode (security and critical fixes only). It is **not** AG2 1.x (`import ag2`; see the [AG2 guide](/ag2-guide/)) and **not** Microsoft AutoGen 0.4+ (`autogen_agentchat`; see the [AgentChat & Core page](/autogen-guide/python/autogen_agentchat_guide/)).

## Background

- **AutoGen 0.2** (`pyautogen`, `import autogen`) was developed at Microsoft Research in 2023–2024.
- In **November 2024** AutoGen's original creators forked the 0.2 line as **AG2** under the `ag2ai` organisation and kept the `import autogen` API. Microsoft continued `microsoft/autogen` separately, shipped the incompatible 0.4 rewrite in January 2025, and has since archived it.
- In **July 2026** AG2 1.0 replaced this API in the `ag2` package with a new async framework (`import ag2`). The API on this page continues as **AG2 Classic** in the `autogen` package.

Which AutoGen you have, and how to move between them, is covered in the [migration guide](/ag2-guide/classic/ag2_classic_migration_guide/).

| | AG2 Classic (this page) | AG2 1.x | Microsoft AutoGen 0.4–0.7 |
|---|---|---|---|
| Install | `pip install "autogen[openai]"` | `pip install "ag2[openai]"` | `pip install autogen-agentchat "autogen-ext[openai]"` |
| Import | `import autogen` | `import ag2` | `import autogen_agentchat` |
| Status | Maintenance mode | Active | Archived |

## Installation

AG2 Classic needs Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install "autogen[openai]"      # AG2 Classic with the OpenAI client
# extras for other providers and features: anthropic, gemini, bedrock, ollama, docker, mcp, a2a, tracing, ...
```

Do **not** install `autogen-agentchat` or `pyautogen` for this API. Both install Microsoft AutoGen 0.4+, which uses different imports. Do not upgrade to `ag2>=1.0` either: it no longer ships the `autogen` package. If you see `ModuleNotFoundError: No module named 'autogen'`, run `pip show autogen ag2` and install `autogen` (or `ag2<1.0`) in a fresh virtual environment.

## Configuring models

In AG2 Classic, `LLMConfig` holds one or more model entries. The client tries the entries in order and falls back to the next one on errors. Generation settings such as `temperature`, `max_tokens`, `timeout` and `cache_seed` apply to every entry:

```python
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig(
    {"api_type": "openai", "model": "%%model=openai%%"},          # reads OPENAI_API_KEY
    {"api_type": "openai", "model": "%%model=openai-small%%"},    # fallback
    temperature=0.2,
    max_tokens=1000,
)

agent = ConversableAgent("assistant", system_message="You are helpful.", llm_config=llm_config)
```

Keys come from each provider's environment variable when `api_key` is omitted. Pass `api_key=os.environ[...]` explicitly if you use a different variable, and keep keys out of source code. Other providers take an `api_type`:

```python
import os

from autogen import LLMConfig

claude = LLMConfig({"api_type": "anthropic", "model": "%%model=anthropic%%"})    # ANTHROPIC_API_KEY, extra autogen[anthropic]
gemini = LLMConfig({"api_type": "google", "model": "%%model=gemini%%"})           # GOOGLE_GEMINI_API_KEY, extra autogen[gemini]
local = LLMConfig({"api_type": "ollama", "model": "llama3.1"})                    # extra autogen[ollama]
compatible = LLMConfig({"model": "llama3.1", "base_url": "http://localhost:11434/v1", "api_key": "unused"})
azure = LLMConfig({
    "api_type": "azure",
    "model": "your-deployment-name",
    "base_url": "https://your-resource-name.openai.azure.com",
    "api_version": "2024-10-21",
    "api_key": os.environ["AZURE_OPENAI_API_KEY"],
})
```

Configurations can also live in a JSON file (a list of entries) that you name explicitly. Nothing is discovered automatically:

```python
from autogen import LLMConfig

llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")   # or env="OAI_CONFIG_LIST" to read the JSON from a variable
```

Older code passes plain dicts (`llm_config={"config_list": [...], "temperature": 0.7}`). They still work, but `LLMConfig` validates the entries.

## Core agent classes

- **`ConversableAgent`**: the base class. It sends and receives messages, calls the model and runs registered tools and reply functions.
- **`AssistantAgent`**: a `ConversableAgent` with a default system prompt for solving tasks by writing code, and no human input.
- **`UserProxyAgent`**: a `ConversableAgent` that stands in for a human. It has no LLM by default, asks for human input, and executes code blocks it receives. Code execution is **on by default and uses Docker**, so pass `code_execution_config=False` unless it should run code.

Key constructor parameters:

```python
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig({"api_type": "openai", "model": "%%model=openai-small%%"})

agent = ConversableAgent(
    name="assistant",
    system_message="You are helpful.",
    description="Answers general questions.",       # used by group-chat speaker selection
    llm_config=llm_config,                            # False for an agent without an LLM
    human_input_mode="NEVER",                         # "NEVER", "TERMINATE" or "ALWAYS"
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda m: (m.get("content") or "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,                      # or {"executor": ...}
)
```

Human input modes:

- `"NEVER"`: never ask a human; the agent replies automatically until a limit or termination message.
- `"TERMINATE"`: ask a human only when a termination condition is met; an empty reply ends the chat.
- `"ALWAYS"`: ask a human before every reply; typing `exit` ends the chat.

Write `(m.get("content") or "")` in termination checks: tool-call messages have `content=None`.

## Two-agent chats

`initiate_chat` runs a conversation and returns a `ChatResult` with `chat_history`, `summary` and `cost`:

```python
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig({"api_type": "openai", "model": "%%model=openai-small%%"})

coder = ConversableAgent(
    "coder",
    system_message="You are an expert Python developer. Write clean, documented code.",
    llm_config=llm_config,
)
reviewer = ConversableAgent(
    "reviewer",
    system_message="You are a senior code reviewer. Review for correctness and security. Do not write code.",
    llm_config=llm_config,
)

chat_result = reviewer.initiate_chat(
    coder,
    message="Write a Python function that validates email addresses.",
    max_turns=2,
    summary_method="last_msg",        # or "reflection_with_llm"
)
print(chat_result.summary)
print(len(chat_result.chat_history), "messages")
```

`ConversableAgent.run(...)` is the event-based alternative. It returns a `RunResponse` that you iterate (`for event in response.events`) or drive with `response.process()`, which prints events to the console; `summary` and `messages` are filled in once the run finishes:

```python
response = coder.run(recipient=reviewer, message="Now add type hints.", max_turns=2)
response.process()
print(response.summary)
```

Agents keep per-recipient history in `agent.chat_messages`. `initiate_chat(..., clear_history=False)` continues a previous conversation, and `agent.clear_history()` resets it.

## Tools

A tool is registered twice: on the agent whose LLM proposes the call (`register_for_llm`) and on the agent that executes it (`register_for_execution`). `register_function` does both:

```python
from typing import Annotated

from autogen import ConversableAgent, LLMConfig, register_function

llm_config = LLMConfig({"api_type": "openai", "model": "%%model=openai-small%%"})

assistant = ConversableAgent(
    "assistant",
    system_message="Use your tools to answer. Reply TERMINATE when done.",
    llm_config=llm_config,
)
user = ConversableAgent(
    "user",
    human_input_mode="NEVER",
    llm_config=False,
    is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
)


def calculate_statistics(
    numbers: Annotated[list[float], "Numbers to analyse"],
    decimal_places: Annotated[int, "Decimal places for rounding"] = 2,
) -> dict:
    """Calculate the mean, minimum and maximum of a list of numbers."""
    return {
        "mean": round(sum(numbers) / len(numbers), decimal_places),
        "min": min(numbers),
        "max": max(numbers),
    }


register_function(
    calculate_statistics,
    caller=assistant,
    executor=user,
    description="Calculate statistics for a list of numbers.",
)

user.initiate_chat(assistant, message="Statistics for 10, 20, 30, 40, 50?", max_turns=4)
```

The same registration as decorators:

```python
@user.register_for_execution()
@assistant.register_for_llm(description="Convert a temperature from Celsius to Fahrenheit.")
def celsius_to_fahrenheit(celsius: Annotated[float, "Temperature in Celsius"]) -> float:
    return celsius * 9 / 5 + 32
```

- Type hints and `Annotated` descriptions become the JSON schema; Pydantic models work as parameter types.
- Exceptions raised in a tool are returned to the model as `Error: ...`, so the conversation continues.
- Prebuilt tools live in `autogen.tools` (for example `autogen.tools.experimental`), MCP servers become tools with `await autogen.mcp.create_toolkit(session=...)` (extra `autogen[mcp]`), and LangChain tools convert with `autogen.interop.Interoperability` (see [production](/ag2-guide/classic/ag2_classic_production_guide/#using-langchain-tools)).

## Code execution

Code execution is separate from tools. An agent with `code_execution_config` runs the fenced code blocks it receives and replies with the output. Pass an executor object:

```python
from autogen import AssistantAgent, LLMConfig, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor

llm_config = LLMConfig({"api_type": "openai", "model": "%%model=openai-small%%"})

assistant = AssistantAgent("assistant", llm_config=llm_config)
executor = UserProxyAgent(
    "executor",
    human_input_mode="NEVER",
    code_execution_config={"executor": LocalCommandLineCodeExecutor(work_dir="coding", timeout=60)},
    is_termination_msg=lambda m: (m.get("content") or "").rstrip().endswith("TERMINATE"),
)

executor.initiate_chat(assistant, message="Print the first ten Fibonacci numbers.", max_turns=4)
```

`LocalCommandLineCodeExecutor` runs code with your process's permissions; use it only for trusted code. For model-written code, use Docker (extra `autogen[docker]` and a running Docker daemon; not run on this page):

```python
from autogen import UserProxyAgent
from autogen.coding import DockerCommandLineCodeExecutor

with DockerCommandLineCodeExecutor(image="python:3.12-slim", work_dir="workspace", timeout=120) as docker_executor:
    executor = UserProxyAgent(
        "executor",
        human_input_mode="NEVER",
        code_execution_config={"executor": docker_executor},
    )
    # executor.initiate_chat(assistant, message="Plot a sine wave and save it as sine.png.")
```

The older dict form `{"work_dir": "coding", "use_docker": False}` still works, but you cannot combine `work_dir` or `use_docker` with `executor` (that raises `ValueError`).

## Structured output

Pass a Pydantic model as `response_format` and the client requests structured output from providers that support it. The reply content is the JSON text, ready to validate:

```python
from pydantic import BaseModel

from autogen import ConversableAgent, LLMConfig


class TicketTriage(BaseModel):
    category: str   # billing, bug, account_access
    urgency: str    # low, medium, high


triage = ConversableAgent(
    "triage",
    system_message="You triage customer support messages.",
    llm_config=LLMConfig({"api_type": "openai", "model": "%%model=openai-small%%"}, response_format=TicketTriage),
)
user = ConversableAgent("user", human_input_mode="NEVER", llm_config=False)

result = user.initiate_chat(triage, message="I was charged twice this month.", max_turns=1)
# ticket = TicketTriage.model_validate_json(result.summary)
```

For providers without native structured output, put the schema (`TicketTriage.model_json_schema()`) in the system message and validate the reply the same way.

## Group chat

`GroupChat` holds the participants and the shared message list; `GroupChatManager` picks the next speaker and broadcasts messages:

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig

llm_config = LLMConfig({"api_type": "openai", "model": "%%model=openai-small%%"})

researcher = ConversableAgent("researcher", system_message="You research topics and list key facts.", llm_config=llm_config)
writer = ConversableAgent("writer", system_message="You write clear articles from the research.", llm_config=llm_config)
editor = ConversableAgent(
    "editor",
    system_message="You give editorial feedback. Do not rewrite. Reply TERMINATE when the article is ready.",
    llm_config=llm_config,
)

groupchat = GroupChat(
    agents=[researcher, writer, editor],
    messages=[],
    max_round=12,
    speaker_selection_method="auto",     # the manager's LLM picks the next speaker
)
manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
    is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
)

researcher.initiate_chat(manager, message="Write a short article on how HTTPS works.")
```

`speaker_selection_method` options:

| Value | Behaviour |
|---|---|
| `"auto"` (default) | The manager's LLM chooses, using each agent's `description` |
| `"round_robin"` | Agents speak in list order |
| `"random"` | A random agent |
| `"manual"` | A human picks each speaker |
| a callable | Your function `(last_speaker, groupchat)` returns the next `Agent`, a method name, or `None` to end |

A custom selector:

```python
def select_speaker(last_speaker, groupchat):
    """researcher -> writer -> editor, then back to the writer until the editor approves."""
    if last_speaker is researcher:
        return writer
    if last_speaker is writer:
        return editor
    return writer


groupchat = GroupChat(
    agents=[researcher, writer, editor],
    messages=[],
    max_round=10,
    speaker_selection_method=select_speaker,
)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
```

Restrict transitions with `allowed_or_disallowed_speaker_transitions={agent: [next_agents]}` and `speaker_transitions_type="allowed"`, or forbid consecutive turns with `allow_repeat_speaker=False`.

## Group-chat patterns and handoffs

Newer Classic code describes orchestration with patterns from `autogen.agentchat.group.patterns` (`DefaultPattern`, `AutoPattern`, `RoundRobinPattern`, `RandomPattern`, `ManualPattern`) and per-agent **handoffs**, and runs them with `initiate_group_chat` (or `run_group_chat` for the event-based form). This replaces the earlier swarm API:

```python
from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import AgentTarget, OnCondition, StringLLMCondition, TerminateTarget
from autogen.agentchat.group.patterns import DefaultPattern

llm_config = LLMConfig({"api_type": "openai", "model": "%%model=openai-small%%"})

triage = ConversableAgent("triage", system_message="Route the customer to billing or tech support.", llm_config=llm_config)
billing = ConversableAgent("billing", system_message="You resolve billing questions.", llm_config=llm_config)
tech = ConversableAgent("tech", system_message="You resolve technical problems.", llm_config=llm_config)
user = ConversableAgent("user", human_input_mode="NEVER", llm_config=False)

# The LLM chooses a handoff (each condition becomes a transfer tool).
triage.handoffs.add_llm_conditions([
    OnCondition(target=AgentTarget(billing), condition=StringLLMCondition(prompt="The question is about payments or invoices.")),
    OnCondition(target=AgentTarget(tech), condition=StringLLMCondition(prompt="The question is about a technical fault.")),
])
# What happens after an agent finishes its turn without handing off.
billing.handoffs.set_after_work(TerminateTarget())
tech.handoffs.set_after_work(TerminateTarget())

pattern = DefaultPattern(
    initial_agent=triage,
    agents=[triage, billing, tech],
    user_agent=user,
    group_manager_args={"llm_config": llm_config},
)
result, context_variables, last_agent = initiate_group_chat(pattern=pattern, messages="I was charged twice.", max_rounds=10)
print(last_agent.name, result.summary)
```

- Targets include `AgentTarget`, `AgentNameTarget`, `RevertToUserTarget`, `StayTarget`, `TerminateTarget`, `GroupChatTarget` and `NestedChatTarget`.
- `OnContextCondition` hands off on a `ContextVariables` expression without an LLM call. Tools can return a `ReplyResult(message=..., target=..., context_variables=...)` to update shared state and route in one step.
- Pass `context_variables=ContextVariables(data={...})` to the pattern to share state. Tools that declare a `context_variables: ContextVariables` parameter receive it.

:::caution[Swarms were removed in 0.14.0]
`SwarmAgent`, `initiate_swarm_chat` and `register_hand_off` lived in `autogen.agentchat.contrib.swarm_agent` (and were re-exported from `autogen.agentchat`) up to `ag2` 0.13.x, and were removed in 0.14.0; neither `ag2` 0.14 nor `autogen` %%version=autogen%% contains them. Port swarm code to the pattern API above (`register_hand_off(agent, [OnCondition(...)])` becomes `agent.handoffs.add_llm_conditions([...])`, `AfterWork(...)` becomes `agent.handoffs.set_after_work(...)`). If you cannot port yet, pin `ag2==0.13.*`.
:::

## Nested chats

A nested chat runs a sequence of sub-conversations inside one agent's turn, and the agent replies with the result. A typical use is an inner review loop:

```python
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig({"api_type": "openai", "model": "%%model=openai-small%%"})

planner = ConversableAgent("planner", system_message="You ask for small Python utilities.", llm_config=llm_config)
writer = ConversableAgent("writer", system_message="You write Python code.", llm_config=llm_config)
reviewer = ConversableAgent("reviewer", system_message="You review code and suggest improvements.", llm_config=llm_config)

writer.register_nested_chats(
    chat_queue=[
        {"recipient": reviewer, "message": "Review this code.", "summary_method": "last_msg", "max_turns": 1},
    ],
    trigger=planner,   # run the review whenever the planner messages the writer
)

planner.initiate_chat(writer, message="Sort a list of dicts by two keys.", max_turns=2)
```

`message` can be a callable `(recipient, messages, sender, config) -> str` to build the inner prompt from the outer conversation.

## Sequential chats

`initiate_chats` runs a list of chats in order and carries each chat's summary into the next:

```python
from autogen import ConversableAgent, LLMConfig, initiate_chats

llm_config = LLMConfig({"api_type": "openai", "model": "%%model=openai-small%%"})

coordinator = ConversableAgent("coordinator", llm_config=False, human_input_mode="NEVER")
ideator = ConversableAgent("ideator", system_message="You brainstorm product ideas.", llm_config=llm_config)
validator = ConversableAgent("validator", system_message="You assess feasibility and market fit.", llm_config=llm_config)
planner = ConversableAgent("planner", system_message="You write implementation plans.", llm_config=llm_config)

results = initiate_chats([
    {"sender": coordinator, "recipient": ideator, "message": "Three ideas for an e-commerce start-up.", "max_turns": 1, "summary_method": "last_msg"},
    {"sender": coordinator, "recipient": validator, "message": "Assess these ideas.", "max_turns": 1, "summary_method": "reflection_with_llm"},
    {"sender": coordinator, "recipient": planner, "message": "Plan the strongest idea.", "max_turns": 1, "summary_method": "last_msg"},
])

for i, result in enumerate(results, start=1):
    print(f"Stage {i}: {result.summary}")
```

Add `"carryover": "..."` to a chat to append extra context to its message. Use `agent.initiate_chats([...])` (without `sender`) to run the sequence from one agent.

## Shared state with ContextVariables

`ContextVariables` is a typed key-value store shared by the agents in a group-chat pattern:

```python
from autogen.agentchat.group import ContextVariables, ReplyResult, AgentNameTarget

context = ContextVariables(data={"project_name": "E-commerce Platform", "budget_usd": 50_000})


def approve_budget(amount: int, context_variables: ContextVariables) -> ReplyResult:
    """Record an approved budget and hand back to the planner."""
    context_variables["budget_usd"] = amount
    return ReplyResult(message=f"Budget set to {amount}.", context_variables=context_variables, target=AgentNameTarget("planner"))


print(context.get("project_name"), context["budget_usd"])
```

Pass it to the pattern (`DefaultPattern(..., context_variables=context)`) and add the function to an agent with `functions=[approve_budget]`. The `context_variables` parameter is injected and hidden from the model.

## A2A, MCP and other integrations

- **A2A**: `autogen.a2a.A2aAgentServer` serves a Classic agent and `A2aRemoteAgent` wraps a remote one (extra `autogen[a2a]`). See [Serving an agent over A2A](/ag2-guide/classic/ag2_classic_production_guide/#serving-an-agent-over-a2a).
- **MCP**: `await autogen.mcp.create_toolkit(session=...)` turns an MCP client session into a toolkit you register on agents (extra `autogen[mcp]`).
- **Observability and cost**: `autogen.runtime_logging`, `autogen.opentelemetry` and `gather_usage_summary` are covered in the [production guide](/ag2-guide/classic/ag2_classic_production_guide/).
- **`autogen.beta`**: from `ag2` 0.11.4 (March 2026) the package shipped a preview of the event-driven API. It became the top-level `ag2` package in AG2 1.0, so new work should use that directly. Code written against `autogen.beta` ports almost mechanically.

## Moving on from Classic

Classic receives security and critical fixes only. To port to AG2 1.x, or to move AutoGen 0.2 code onto Classic, follow the [migration guide](/ag2-guide/classic/ag2_classic_migration_guide/).

---

## Revision history

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-21 | autogen 0.14.1 | Moved from `/autogen-guide/python/autogen_comprehensive_guide/` into the AG2 guide. Replaced the "continuing in next part" stub with a tools section; fixed the custom speaker selector (pass the callable as `speaker_selection_method`), the `.process()` call on a `ChatResult`, and the claim that swarms live in `contrib.swarm_agent` (removed in 0.14.0); added group-chat patterns and handoffs, `response_format` structured output and `code_execution_config=False` where no code runs. Standardised on `LLMConfig`, removed the auto-discovery claim and hard-coded keys, cut prompt-only examples and the duplicated migration section. |
| 2026-10-21 | autogen 0.14.1 | Reframed as an AG2 Classic page; removed invented `context_truncation_strategy`, `TelemetryConfig`, `get_cost_stats` and AutoGen Studio material. |
| 2026-04-05 | 0.11.5 | Earlier revision. |

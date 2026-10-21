---
title: "AutoGen to AG2 Migration Guide"
description: "Which migration path applies to your AutoGen code — AutoGen 0.2 to AG2 Classic, Classic to AG2 1.x, or Microsoft AutoGen 0.4+ to Agent Framework or AG2 — with verified before/after code."
framework: "autogen"
language: "python"
---

> Checked %%versions-checked%% against `autogen` %%version=autogen%% (AG2 Classic), `ag2` %%version=ag2%% and `autogen-agentchat` %%version=autogen-agentchat%%.

An earlier version of this page said Microsoft AutoGen had been "rebranded as AG2 (AutoGen 2.0)" and that switching was a matter of renaming the package. That was wrong. AG2 is a **fork** created in November 2024 by AutoGen's original authors from the AutoGen 0.2 code base; Microsoft continued its own project, rewrote it as 0.4 (January 2025), put it into maintenance mode in October 2025 in favour of Microsoft Agent Framework, and marked it archived in October 2026. "Migrating to AG2" therefore means different things depending on which AutoGen you have.

## Which AutoGen do you have?

| Your imports | You are on | Status | Paths |
|---|---|---|---|
| `import autogen`, `from autogen import ConversableAgent` with `pyautogen` ≤ 0.2.x installed | AutoGen 0.2 (Microsoft, pre-fork) | Unmaintained | [A](#path-a-autogen-02--ag2-classic) or [B](#path-b-ag2-classic--ag2-1x) |
| `import autogen` with `autogen` or `ag2<1.0` installed | AG2 Classic | Maintenance mode | [B](#path-b-ag2-classic--ag2-1x) |
| `from autogen_agentchat ...`, `from autogen_core ...` | Microsoft AutoGen 0.4–0.7 | Archived | [C](#path-c-microsoft-autogen-04--agent-framework-or-ag2) |
| `import ag2` | AG2 1.x | Active | — |

Watch the package names: **`pyautogen` now installs Microsoft's `autogen-agentchat`**, **`autogen` installs AG2 Classic**, and **`ag2>=1.0` installs the new AG2 framework**, which no longer contains the `autogen` module.

## Path A: AutoGen 0.2 → AG2 Classic

The lowest-effort move: AG2 Classic kept the 0.2 API, so most code runs with only a dependency change. Choose this to keep an existing system alive; Classic receives security and critical fixes only.

```bash
pip uninstall -y pyautogen
pip install "autogen[openai]"        # or: pip install "ag2[openai]<1.0"
```

```text
# requirements.txt
autogen[openai]>=%%version=autogen%%,<0.15
```

Typical code keeps working unchanged:

```python
from autogen import AssistantAgent, LLMConfig, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor

llm_config = LLMConfig({"api_type": "openai", "model": "%%model=openai-small%%"})

assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    code_execution_config={"executor": LocalCommandLineCodeExecutor(work_dir="coding")},
)
# user_proxy.initiate_chat(assistant, message="Write a function that returns the nth prime.")
```

Things that do need changing:

- **Config lists**: `{"config_list": [...]}` dicts still work, but `LLMConfig({...})` / `LLMConfig.from_json(path="OAI_CONFIG_LIST")` is the current form. `config_list_from_json` is no longer exported at the top of `autogen`; import it from `autogen.llm_config.config` if you still need it.
- **Code executors**: pass an executor object (`LocalCommandLineCodeExecutor`, `DockerCommandLineCodeExecutor`) in `code_execution_config["executor"]`; combining `executor` with `work_dir` / `use_docker` keys raises `ValueError`. The plain `{"work_dir": ..., "use_docker": ...}` form without `executor` still works.
- **Swarms**: `SwarmAgent` / `initiate_swarm_chat` moved to `autogen.agentchat.contrib.swarm_agent` in 0.13.0; group-chat patterns in `autogen.agentchat.group.patterns` are the newer approach.
- **Contrib modules** that depended on third-party libraries may need their extras (`autogen[retrievechat]`, `autogen[a2a]`, …).

## Path B: AG2 Classic → AG2 1.x

AG2 1.0 (July 2026) replaced the Classic API with an async, event-driven framework imported as `ag2`. This is a rewrite of your agent code, not an upgrade.

**Before (Classic):**

```python
from autogen import ConversableAgent, LLMConfig, register_function

llm_config = LLMConfig({"api_type": "openai", "model": "%%model=openai-small%%"})
assistant = ConversableAgent("assistant", system_message="Use tools when helpful.", llm_config=llm_config)
user = ConversableAgent("user", human_input_mode="NEVER", llm_config=False)


def get_weather(city: str) -> str:
    return f"It is sunny in {city}."


register_function(get_weather, caller=assistant, executor=user, description="Get the weather for a city.")
# user.initiate_chat(assistant, message="Weather in Paris?", max_turns=2)
```

**After (AG2 1.x):**

```python
import asyncio

from ag2 import Agent, tool
from ag2.config import OpenAIConfig


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"It is sunny in {city}."


assistant = Agent(
    "assistant",
    prompt="Use tools when helpful.",
    config=OpenAIConfig(model="%%model=openai-small%%"),
    tools=[get_weather],
)

# reply = asyncio.run(assistant.ask("Weather in Paris?")); print(reply.body)
```

| Classic | AG2 1.x |
|---|---|
| `ConversableAgent(name, system_message=..., llm_config=...)` | `Agent(name, prompt=..., config=OpenAIConfig(...))` |
| `LLMConfig` / `OAI_CONFIG_LIST` | One provider config object per model (`OpenAIConfig`, `AnthropicConfig`, …) |
| `register_function(f, caller=..., executor=...)` | `@tool` + `tools=[...]`; no executor agent |
| `initiate_chat(...)` / `generate_reply(...)` | `await agent.ask(...)`, then `await reply.ask(...)` to continue |
| `human_input_mode="ALWAYS"` | a tool calling `await context.input(...)` + `hitl_hook=` |
| `GroupChat` / `GroupChatManager` / swarms | `ag2.network.Hub` + a `workflow` channel driven by a `TransitionGraph` |
| `autogen.beta.*` (0.11–0.14 preview) | Same names, imported from `ag2` |

The [AG2 guide](/ag2-guide/ag2_comprehensive_guide/#ag2-classic-and-migrating-from-0x) covers the 1.x API in full, including a Network example equivalent to a sequential group chat.

## Path C: Microsoft AutoGen 0.4+ → Agent Framework or AG2

Code using `autogen_agentchat` / `autogen_core` shares no API with AG2 (Classic or 1.x); moving it to AG2 is a rewrite. Because Microsoft AutoGen is archived, choose a maintained target:

- **Microsoft Agent Framework** (`agent-framework` %%version=agent-framework%%) is Microsoft's designated successor, with an official [AutoGen migration guide](https://learn.microsoft.com/en-us/agent-framework/migration-guide/from-autogen/) mapping `AssistantAgent`, teams and `GraphFlow` onto agents and Workflows. See the [Agent Framework guide](/microsoft-agent-framework-guide/).
- **AG2 1.x** if you prefer the community project: the concepts map reasonably well (`AssistantAgent` → `Agent`; `FunctionTool` → `@tool`; `GraphFlow` / `SelectorGroupChat` → `workflow` channels with a `TransitionGraph`; `ReplayChatCompletionClient` → `ag2.testing.TestConfig`).

Until you migrate, pin your AutoGen versions exactly (see the [AgentChat & Core reference](/autogen-guide/python/autogen_agentchat_guide/#packages)), including integration dependencies such as `mcp<2`.

## Troubleshooting

**`ModuleNotFoundError: No module named 'autogen'` after an upgrade.** `ag2` was upgraded to 1.x, or `pyautogen` now resolves to Microsoft AutoGen. Install `autogen` (Classic) or pin `ag2<1.0`, or port to `import ag2`.

**`ModuleNotFoundError: No module named 'autogen_agentchat'`.** The code is Microsoft AutoGen 0.4+; install `autogen-agentchat` and `autogen-ext[openai]`, not `ag2`.

**`TypeError: LLMConfig.from_json() takes 1 positional argument`.** Pass the file as a keyword: `LLMConfig.from_json(path="OAI_CONFIG_LIST")` (or `env="OAI_CONFIG_LIST"`).

**`ValueError: 'work_dir' in code_execution_config is not valid when 'executor' is set`.** Move `work_dir` / `timeout` onto the executor object.

**Both `autogen` and `ag2` installed in one environment.** `autogen` and `ag2<1.0` provide the same `autogen` module and will overwrite each other; keep one.

## Resources

- AG2: [docs.ag2.ai](https://docs.ag2.ai) · [github.com/ag2ai/ag2](https://github.com/ag2ai/ag2)
- AG2 Classic: [classic.docs.ag2.ai](https://classic.docs.ag2.ai) · [github.com/ag2ai/ag2-classic](https://github.com/ag2ai/ag2-classic)
- Microsoft AutoGen (archived): [github.com/microsoft/autogen](https://github.com/microsoft/autogen)
- Microsoft Agent Framework: [github.com/microsoft/agent-framework](https://github.com/microsoft/agent-framework)

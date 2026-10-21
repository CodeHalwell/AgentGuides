---
title: "Claude Agent SDK (Python) — Migration Guide"
description: "Moving from claude-code-sdk to claude-agent-sdk, and upgrading across 0.1.x and 0.2.x releases."
framework: "claude-agent-sdk"
language: "python"
---

**Target:** `claude-agent-sdk` %%version=claude-agent-sdk%% (checked %%versions-checked%%).

This page covers two things:

1. Moving code from the old `claude-code-sdk` package (last release 0.0.25) to `claude-agent-sdk`. The rename happened in `claude-agent-sdk` 0.1.0.
2. Upgrading an existing `claude-agent-sdk` project from an older 0.1.x or 0.2.x release to the current one.

---

## Part 1 — from `claude-code-sdk`

### What actually changed

The rename was mostly mechanical, with three behavioural changes that catch people out:

| Area | `claude-code-sdk` | `claude-agent-sdk` |
|---|---|---|
| PyPI package | `claude-code-sdk` | `claude-agent-sdk` |
| Import name | `claude_code_sdk` | `claude_agent_sdk` |
| Options class | `ClaudeCodeOptions` | `ClaudeAgentOptions` |
| System prompt fields | `custom_system_prompt`, `append_system_prompt` | single `system_prompt` (string or preset dict) |
| Default system prompt | Claude Code's prompt | a minimal prompt; ask for the preset explicitly |
| Filesystem settings | loaded automatically | controlled by `setting_sources` (see Step 4) |

Everything else you used — `query()`, `ClaudeSDKClient`, `@tool`, `create_sdk_mcp_server()`, the message and block classes, `can_use_tool`, hooks — kept its name and import location. Note that there has never been a `claude_agent_sdk.tools` submodule; import everything from the package root.

### Step 1: swap the dependency

```bash
pip uninstall -y claude-code-sdk
pip install claude-agent-sdk==%%version=claude-agent-sdk%%
```

```txt
# requirements.txt
claude-agent-sdk==%%version=claude-agent-sdk%%
```

### Step 2: rename imports and the options class

```python
# Before
from claude_code_sdk import query, ClaudeCodeOptions, AssistantMessage, TextBlock

options = ClaudeCodeOptions(max_turns=5)
```

```python
# After
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock

options = ClaudeAgentOptions(max_turns=5)
```

For a larger codebase, a two-line substitution does most of the work:

```bash
grep -rl --include='*.py' 'claude_code_sdk\|ClaudeCodeOptions' . \
  | xargs sed -i 's/claude_code_sdk/claude_agent_sdk/g; s/ClaudeCodeOptions/ClaudeAgentOptions/g'
```

### Step 3: decide what system prompt you want

The old SDK silently used Claude Code's system prompt. The new one does not, so an agent that relied on it (for coding behaviour, tool etiquette, safety guidance) will behave differently until you ask for it back:

```python
# Before
from claude_code_sdk import ClaudeCodeOptions

options = ClaudeCodeOptions(append_system_prompt="Prefer small, reviewable diffs.")
```

```python
# After: keep Claude Code's prompt and append to it
from claude_agent_sdk import ClaudeAgentOptions

options = ClaudeAgentOptions(
    system_prompt={
        "type": "preset",
        "preset": "claude_code",
        "append": "Prefer small, reviewable diffs.",
    },
)
```

If you used `custom_system_prompt="..."`, pass the same string as `system_prompt="..."`.

### Step 4: decide which settings files to load

The 0.1.0 release stopped loading `CLAUDE.md`, `.claude/settings.json`, project slash commands and `.claude/agents/*.md` by default. That is no longer the case: in current releases, leaving `setting_sources` unset (`None`) loads all sources (user, project and local), as the CLI does. So the question when migrating is usually the opposite one: what should your service *not* read?

```python
from claude_agent_sdk import ClaudeAgentOptions

isolated = ClaudeAgentOptions(setting_sources=[])                # nothing from disk
project_only = ClaudeAgentOptions(setting_sources=["project"])   # CLAUDE.md, .claude/settings.json
everything = ClaudeAgentOptions()                                # user, project and local (the default)
```

Pass `[]` or an explicit list for CI and servers, where a stray `~/.claude/settings.json` should not change behaviour. (Before 0.1.60, `setting_sources=[]` was silently treated as unset; upgrade if you rely on isolation.)

### Step 5: run your tests

Message handling code generally needs no changes. Check, though, that you test messages with `isinstance()` — the SDK yields dataclasses, so `message.type == "assistant"` has never worked in Python:

```python
from claude_agent_sdk import AssistantMessage, ResultMessage, SystemMessage, TextBlock

def handle(message) -> None:
    match message:
        case SystemMessage(subtype="init"):
            print("session", message.data.get("session_id"))
        case AssistantMessage():
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text)
        case ResultMessage():
            print("done:", message.result)
        case _:
            pass
```

### New in the Agent SDK at the rename

These arrived with 0.1.0 and are worth adopting while you are migrating:

- **Programmatic subagents** via `agents={"name": AgentDefinition(...)}` — no `.claude/agents/` files needed.
- **Session forking** via `ClaudeAgentOptions(resume=session_id, fork_session=True)`.
- **`setting_sources`** for settings isolation.

See the [comprehensive guide](/anthropic-claude-agent-sdk-guide/anthropic_claude_agent_sdk_comprehensive_guide/) for examples.

---

## Part 2 — upgrading within `claude-agent-sdk`

Most releases only bump the bundled Claude Code CLI. The changes below are the ones that can affect existing code. The full list is in the [CHANGELOG](https://github.com/anthropics/claude-agent-sdk-python/blob/main/CHANGELOG.md).

### From 0.1.x

| Version | Change | What to do |
|---|---|---|
| 0.1.36 | `thinking=` and `effort=` added; `max_thinking_tokens` deprecated | Replace `max_thinking_tokens=N` with `thinking={"type": "adaptive"}` and an `effort` level |
| 0.1.77 | `"Skill"` in `allowed_tools` deprecated | Use `skills="all"` or `skills=["name", ...]` |
| 0.2.82 | `EffortLevel` exported; `mcp` floor raised to 1.23.0 | Upgrade `mcp` if you pin it |

The move from 0.1.81 to 0.2.82 was not a breaking release; code written for late 0.1.x runs unchanged.

### Within 0.2.x

| Version | Change | What to do |
|---|---|---|
| 0.2.111 | Warning when `can_use_tool` is shadowed by `allowed_tools` or `bypassPermissions` | Remove the tool from `allowed_tools` if you want the callback to see it |
| 0.2.124 | On Windows, `.bat`/`.cmd` CLI shims are refused | Use the bundled CLI or point `cli_path` at `claude.exe` |
| 0.2.129 | **Breaking:** skill names are validated | Replace `skills=["*"]` or `["plugin:*"]` with `skills="all"` |
| 0.2.137 | `ConversationResetMessage` added to the `Message` union | Add a case if you match exhaustively with `assert_never` |
| 0.2.140 | `ResultError` raised after terminal error results, instead of a bare `ProcessError` | It subclasses `ProcessError`, so existing handlers still catch it; branch on `e.subtype` / `e.terminal_reason` for better retries |
| 0.2.140 | `mcp` 2.x supported for in-process servers | No action; `mcp>=1.23.0,<3.0.0` is accepted |
| 0.2.166 | `SandboxSettings.ignoreViolations` is now `dict[str, list[str]]` | Replace `{"file": [...], "network": [...]}` with `{"<command pattern>": ["<violation text>", ...]}` |

### Model IDs

Older examples often hard-code `claude-3-5-sonnet-20241022` or `claude-sonnet-4-5`. Those models are retired or superseded; use a current model such as `%%model=anthropic%%` or `%%model=anthropic-flagship%%`, or an alias such as `"sonnet"`. Two behavioural differences on the current models:

- Fixed thinking budgets (`{"type": "enabled", "budget_tokens": N}`) are rejected. Use adaptive thinking and `effort`.
- Claude Opus 5.5 defaults to `medium` effort. Set `effort="high"` (or `"xhigh"` for long agentic tasks) if your agent was tuned on a model that defaulted higher.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'claude_code_sdk'`** — the old package is gone from your environment but an import was missed. Search for `claude_code_sdk` and rename.

**`ModuleNotFoundError: No module named 'claude_agent_sdk.tools'`** — import `tool` and `create_sdk_mcp_server` from `claude_agent_sdk` directly.

**`TypeError: query() takes 0 positional arguments`** — `query()` is keyword-only: `query(prompt="...", options=options)`.

**`TypeError: ClaudeAgentOptions.__init__() got an unexpected keyword argument ...`** — options such as `temperature`, `enable_subagents`, `api_provider` or `checkpoint_dir` have never existed. Subagents are enabled by defining `agents=`; providers are chosen with `CLAUDE_CODE_USE_BEDROCK` / `CLAUDE_CODE_USE_VERTEX`; checkpointing is `enable_file_checkpointing=True`.

**The agent no longer follows `CLAUDE.md`** — you pass a `setting_sources` list without `"project"` (or `[]`). Add `"project"`, or omit the option to load all sources.

**The agent's coding behaviour got worse after migrating** — you lost Claude Code's system prompt; use the `claude_code` preset.

**`CLINotFoundError`** — your platform has no bundled CLI wheel. Install Claude Code separately and pass `cli_path=`.

---

## Checklist

- [ ] `claude-code-sdk` removed; `claude-agent-sdk` pinned in requirements
- [ ] `claude_code_sdk` → `claude_agent_sdk`, `ClaudeCodeOptions` → `ClaudeAgentOptions`
- [ ] `custom_system_prompt` / `append_system_prompt` replaced by `system_prompt`
- [ ] `setting_sources` set explicitly (`[]` for isolation; omitted loads everything)
- [ ] `max_thinking_tokens` replaced by `thinking` + `effort`
- [ ] Model IDs updated to the current family
- [ ] Tests pass, including permission callbacks and hooks

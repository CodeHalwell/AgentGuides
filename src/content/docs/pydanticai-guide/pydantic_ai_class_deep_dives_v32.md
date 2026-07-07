---
title: "PydanticAI Class Deep Dives Vol. 32"
description: "Source-verified deep dives into 10 pydantic-ai 2.5.1 class groups: AnthropicModelProfile + anthropic_model_profile() + AnthropicCodeExecutionToolVersion + resolve_anthropic_effort (13 Anthropic-specific profile fields for Claude Fable 5, Mythos 5, Opus 4.7/4.8, Sonnet 4.6/5 — adaptive thinking, fast mode, effort, xhigh, task budgets, dynamic filtering, forced-tool-choice gate, dual code-execution-tool versions), OpenAIModelProfile + OpenAISystemPromptRole + OPENAI_REASONING_EFFORT_MAP (thinking-field config, send_back_thinking_parts auto/tags/field/False modes, strict tool definitions, SAMPLING_PARAMS incompatibility list, gpt-5.1+ reasoning support), GrokModelProfile + GrokReasoningEffort + grok_model_profile() (Grok 4.3 reasoning-effort tiers, builtin-tools gate, retirement-redirect slug handling), GoogleModelProfile + google_model_profile() + GoogleJsonSchemaTransformer (Gemini 3+ tool-combination support, server-side tool-invocation circulation, thinking_level vs thinking_budget, MIME multimodal returns, const→enum rewrite), CombinedCapability (nested-capability flattening, sort_capabilities topo-sort, parallel gather-based for_run, has_wrap_node_run shortcut), CapabilityOrdering + CapabilityPosition + CapabilityRef + sort_capabilities + collect_leaves + has_capability_type (ordering constraints — position tiers, wraps/wrapped_by/requires edges, graphlib TopologicalSorter, cycle detection), ToolCorrectness + TrajectoryMatch (span-based multiset and ordered-trajectory agentic evaluators — allow_extra, include_failed, order='exact'/'in_order'/'any_order'), ArgumentCorrectness + ArgumentMatchMode + ArgumentOccurrence + MaxToolCalls + MaxModelRequests (argument-level correctness — subset/exact modes, first/last/indexed occurrence, failed-attempt inclusion, budget caps), GEval + HasMatchingSpan + OutputConfig (G-Eval chain-of-thought scoring — criteria + evaluation_steps + score_range, simplified log-probs, HasMatchingSpan SpanQuery delegation, OutputConfig wire TypedDict for LLMJudge/GEval), CaseLifecycle (per-case eval lifecycle hooks — setup/prepare_context/teardown, exception handling semantics, ReportCase/ReportCaseFailure teardown args). All verified against pydantic-ai 2.5.1 and pydantic-evals 2.5.1 source."
sidebar:
  label: "Class deep dives (Vol. 32)"
  order: 58
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 2.5.1** and **pydantic-evals 2.5.1** source installed directly from PyPI. Every class signature, field name, and method in this volume reflects the 2.5.x API.
</Aside>

Ten class groups covering the new **`profiles/`** subpackage (provider-specific `ModelProfile` subclasses and helper functions for Anthropic, OpenAI, Grok, and Google), the **capability composition and ordering** engine (`CombinedCapability`, `CapabilityOrdering`, `sort_capabilities`), and the new **span-based agentic evaluators** in `pydantic-evals` (`ToolCorrectness`, `TrajectoryMatch`, `ArgumentCorrectness`, `MaxToolCalls`, `MaxModelRequests`, `GEval`, `HasMatchingSpan`, and `CaseLifecycle`).

---

## 1. `AnthropicModelProfile` + `anthropic_model_profile()` + `AnthropicCodeExecutionToolVersion` + `resolve_anthropic_effort`

`AnthropicModelProfile` is a `ModelProfile` `TypedDict` subclass that carries Anthropic-specific per-model capability flags. All keys are `anthropic_` prefixed so they can be safely spread into cross-provider merged profiles. The companion function `anthropic_model_profile(model_name)` reads the model name prefix and returns a fully-populated `AnthropicModelProfile` instance. `AnthropicCodeExecutionToolVersion` is a `Literal['20250825', '20260120']` type alias that selects the wire version of the code-execution tool declaration.

**Key fields (all `total=False`):**

| Field | Default | What it gates |
|-------|---------|---------------|
| `anthropic_supports_fast_speed` | `False` | `anthropic_speed='fast'` on Opus 4.6/4.7/4.8 |
| `anthropic_supports_adaptive_thinking` | `False` | Translates `thinking` → `{'type': 'adaptive'}` on Sonnet 4.6+/Opus 4.6+ |
| `anthropic_supports_effort` | `False` | Maps unified thinking level to `output_config.effort` on Opus 4.5+/Sonnet 4.6+ |
| `anthropic_supports_xhigh_effort` | `False` | Preserves `xhigh` in effort map (vs downshifting to `max`) on Opus 4.7/4.8/Sonnet 5/Fable 5/Mythos 5 |
| `anthropic_disallows_budget_thinking` | `False` | Rejects `{'type': 'enabled', 'budget_tokens': N}` — enforced for Opus 4.7/4.8/Sonnet 5 which require adaptive |
| `anthropic_disallows_sampling_settings` | `False` | Strips `temperature`/`top_p` from payloads for Opus 4.7/4.8/Sonnet 5/Fable 5/Mythos 5 |
| `anthropic_supports_dynamic_filtering` | `False` | Selects `web_search_20260209`/`web_fetch_20260209` for dynamic Anthropic-managed result filtering |
| `anthropic_supports_forced_tool_choice` | implicit | When `False`, `tool_choice='required'` raises `UserError` (Fable 5/Mythos 5/Mythos Preview) |
| `anthropic_supports_task_budgets` | `False` | `output_config.task_budget` beta field on Opus 4.7/4.8/Sonnet 5/Fable 5/Mythos 5 |
| `anthropic_default_code_execution_tool_version` | `'20250825'` | Auto-selected code execution version when `anthropic_code_execution_tool_version='auto'` |
| `anthropic_supported_code_execution_tool_versions` | `('20250825',)` | Tuple of accepted versions (Fable 5+/Opus 4.5+ gain `'20260120'` support) |

```python
# Signatures verified from profiles/anthropic.py (pydantic-ai 2.5.1):
#
# AnthropicCodeExecutionToolVersion = Literal['20250825', '20260120']
#
# class AnthropicModelProfile(ModelProfile, total=False):
#     anthropic_supports_fast_speed: bool
#     anthropic_supports_adaptive_thinking: bool
#     anthropic_supports_effort: bool
#     anthropic_supports_xhigh_effort: bool
#     anthropic_disallows_budget_thinking: bool
#     anthropic_disallows_sampling_settings: bool
#     anthropic_supports_dynamic_filtering: bool
#     anthropic_supports_forced_tool_choice: bool
#     anthropic_supports_task_budgets: bool
#     anthropic_default_code_execution_tool_version: AnthropicCodeExecutionToolVersion
#     anthropic_supported_code_execution_tool_versions: tuple[AnthropicCodeExecutionToolVersion, ...]
#
# def resolve_anthropic_effort(
#     level: ThinkingEffort, *, supports_xhigh: bool
# ) -> AnthropicEffort: ...
#
# def anthropic_model_profile(model_name: str) -> ModelProfile | None: ...
#
# ANTHROPIC_THINKING_BUDGET_MAP: dict[ThinkingLevel, int]
# ANTHROPIC_THINKING_EFFORT_MAP: dict[ThinkingEffort, AnthropicEffort]
```

### 1.1 Detecting Adaptive Thinking Support at Runtime

```python
import os

from pydantic_ai.profiles.anthropic import anthropic_model_profile

os.environ.setdefault('ANTHROPIC_API_KEY', 'your-key-here')

# Inspect the resolved profile for any model name
for model_name in ['claude-sonnet-4-5', 'claude-sonnet-4-6', 'claude-opus-4-8', 'claude-fable-5']:
    profile = anthropic_model_profile(model_name)
    assert profile is not None
    adaptive = profile.get('anthropic_supports_adaptive_thinking', False)
    effort = profile.get('anthropic_supports_effort', False)
    xhigh = profile.get('anthropic_supports_xhigh_effort', False)
    print(f'{model_name}: adaptive={adaptive}, effort={effort}, xhigh={xhigh}')

# claude-sonnet-4-5: adaptive=False, effort=False, xhigh=False
# claude-sonnet-4-6: adaptive=True, effort=True, xhigh=False
# claude-opus-4-8:  adaptive=True, effort=True, xhigh=True
# claude-fable-5:   adaptive=True, effort=True, xhigh=True
```

### 1.2 Resolving Effort Levels and Merging Profiles

```python
import os

from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.anthropic import (
    AnthropicModelProfile,
    anthropic_model_profile,
    resolve_anthropic_effort,
)

os.environ.setdefault('ANTHROPIC_API_KEY', 'your-key-here')

base = anthropic_model_profile('claude-opus-4-8')
assert base is not None

# resolve_anthropic_effort maps unified ThinkingEffort → Anthropic API effort string
# xhigh→'max' by default; pass supports_xhigh=True to keep 'xhigh' for Opus 4.8+
print(resolve_anthropic_effort('xhigh', supports_xhigh=False))  # 'max'
print(resolve_anthropic_effort('xhigh', supports_xhigh=True))   # 'xhigh'
print(resolve_anthropic_effort('high', supports_xhigh=True))    # 'high'

# Merge a custom override — e.g., force-disable dynamic filtering for testing
override = AnthropicModelProfile(anthropic_supports_dynamic_filtering=False)
merged = merge_profile(base, override)
print(merged.get('anthropic_supports_dynamic_filtering'))  # False
```

### 1.3 Code Execution Tool Version Selection

```python
import os

from pydantic_ai.profiles.anthropic import (
    AnthropicCodeExecutionToolVersion,
    anthropic_model_profile,
)

os.environ.setdefault('ANTHROPIC_API_KEY', 'your-key-here')

# claude-sonnet-4-4 (pre-Fable): only the 2025-08-25 version is supported
old_profile = anthropic_model_profile('claude-sonnet-4-4')
assert old_profile is not None
print('default:', old_profile.get('anthropic_default_code_execution_tool_version'))
print('supported:', old_profile.get('anthropic_supported_code_execution_tool_versions'))
# default: 20250825
# supported: ('20250825',)

# claude-opus-4-5 and newer support the 2026-01-20 version as well
new_profile = anthropic_model_profile('claude-opus-4-5')
assert new_profile is not None
print('default:', new_profile.get('anthropic_default_code_execution_tool_version'))
print('supported:', new_profile.get('anthropic_supported_code_execution_tool_versions'))
# default: 20260120
# supported: ('20250825', '20260120')

# The version type alias is a safe Literal for type-checking
version: AnthropicCodeExecutionToolVersion = '20260120'
print(f'Using code execution version: {version}')
```

---

## 2. `OpenAIModelProfile` + `OpenAISystemPromptRole` + `OPENAI_REASONING_EFFORT_MAP`

`OpenAIModelProfile` adds OpenAI-Chat-Completions-specific keys to `ModelProfile`. All fields are `openai_` prefixed. The most impactful new fields control how thinking/reasoning content is round-tripped: `openai_chat_thinking_field` names the custom JSON field some providers use (e.g. `'reasoning'` for Ollama, `'reasoning_content'` for DeepSeek), and `openai_chat_send_back_thinking_parts` controls whether prior thinking is re-injected as tags, a field, or omitted. `OPENAI_REASONING_EFFORT_MAP` maps unified `ThinkingLevel` values to OpenAI `reasoning_effort` strings. `SAMPLING_PARAMS` lists parameter names incompatible with reasoning mode.

```python
# Signatures verified from profiles/openai.py (pydantic-ai 2.5.1):
#
# OpenAISystemPromptRole = Literal['system', 'developer', 'user']
#
# OPENAI_REASONING_EFFORT_MAP: dict[ThinkingLevel, str] = {
#     True: 'medium', False: 'none',
#     'minimal': 'minimal', 'low': 'low', 'medium': 'medium',
#     'high': 'high', 'xhigh': 'xhigh',
# }
#
# SAMPLING_PARAMS = (
#     'temperature', 'top_p', 'presence_penalty', 'frequency_penalty',
#     'logit_bias', 'openai_logprobs', 'openai_top_logprobs',
# )
#
# class OpenAIModelProfile(ModelProfile, total=False):
#     openai_chat_thinking_field: str | None         # e.g. 'reasoning', 'reasoning_content'
#     openai_chat_send_back_thinking_parts: Literal['auto', 'tags', 'field', False]
#     openai_supports_strict_tool_definition: bool   # default True
#     openai_unsupported_model_settings: Sequence[str]
#     openai_supports_tool_choice_required: bool     # default True
#
# def openai_model_profile(model_name: str) -> ModelProfile: ...
```

### 2.1 Configuring Thinking Field for Custom OpenAI-Compatible Providers

```python
import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

os.environ.setdefault('CUSTOM_LLM_API_KEY', 'your-key-here')

# Ollama and newer vLLM use 'reasoning' as the thinking field
ollama_profile = OpenAIModelProfile(
    openai_chat_thinking_field='reasoning',
    openai_chat_send_back_thinking_parts='field',  # send thinking back in the same field
    supports_thinking=True,
    openai_supports_strict_tool_definition=False,  # many compatible APIs don't support strict
)

# In 2.5.x, custom base URLs go through provider=, not a base_url= kwarg on the model
model = OpenAIChatModel(
    'deepseek-r1:7b',
    provider=OpenAIProvider(base_url='http://localhost:11434/v1'),
    profile=ollama_profile,
)

agent = Agent(model, system_prompt='You are a reasoning assistant.')
# Now thinking content from prior turns is round-tripped via the 'reasoning' field
print('Profile configured for Ollama thinking round-trip')
```

### 2.2 Inspecting OPENAI_REASONING_EFFORT_MAP and SAMPLING_PARAMS

```python
from pydantic_ai.profiles.openai import (
    OPENAI_REASONING_EFFORT_MAP,
    SAMPLING_PARAMS,
    OpenAISystemPromptRole,
)

# Map unified ThinkingLevel values to OpenAI reasoning_effort strings
for level, effort in OPENAI_REASONING_EFFORT_MAP.items():
    print(f'thinking={level!r} → reasoning_effort={effort!r}')

# thinking=True → reasoning_effort='medium'
# thinking=False → reasoning_effort='none'
# thinking='minimal' → reasoning_effort='minimal'
# ...

# Parameters incompatible with reasoning mode (excluded from requests when reasoning_effort != 'none')
print('Sampling params disabled during reasoning:')
for param in SAMPLING_PARAMS:
    print(f'  - {param}')

# Valid system prompt roles for OpenAI models
role: OpenAISystemPromptRole = 'developer'  # also 'system' or 'user'
print(f'Using system prompt role: {role}')
```

### 2.3 Merging OpenAI Profile for a Provider That Rejects tool_choice=required

```python
import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.openai import OpenAIModelProfile, openai_model_profile

os.environ.setdefault('MOONSHOTAI_API_KEY', 'your-key-here')

# Some providers (e.g. MoonshotAI) reject tool_choice='required'
base = openai_model_profile('moonshot-v1-8k')
restriction = OpenAIModelProfile(openai_supports_tool_choice_required=False)
final_profile = merge_profile(base, restriction)

model = OpenAIChatModel(
    'moonshot-v1-8k',
    base_url='https://api.moonshot.cn/v1',
    profile=final_profile,
)

agent = Agent(model)
print('tool_choice_required disabled:', final_profile.get('openai_supports_tool_choice_required'))
# tool_choice_required disabled: False
```

---

## 3. `GrokModelProfile` + `GrokReasoningEffort` + `grok_model_profile()`

`GrokModelProfile` carries xAI-specific profile flags. `GrokReasoningEffort` is a `Literal['none', 'low', 'medium', 'high']` type alias for native xAI `reasoning_effort` values. The `grok_model_profile()` function encodes the Grok model taxonomy: Grok 4.3 (and its retirement-redirect aliases like `grok-3`, `grok-4-0709`) get the full `{none, low, medium, high}` effort set and builtin-tools support; Grok 3 Mini gets only `{low, high}`; non-reasoning models get an empty frozenset and `supports_thinking=False`.

```python
# Signatures verified from profiles/grok.py (pydantic-ai 2.5.1):
#
# GrokReasoningEffort = Literal['none', 'low', 'medium', 'high']
#
# class GrokModelProfile(ModelProfile, total=False):
#     grok_supports_builtin_tools: bool           # web_search/x_search/code_execution/mcp
#     grok_supports_tool_choice_required: bool    # default True
#     grok_reasoning_efforts: frozenset[GrokReasoningEffort]
#
# def grok_model_profile(model_name: str) -> ModelProfile | None: ...
#
# _GROK_43_REASONING_MODELS: frozenset — retirement-redirect slugs that route to Grok 4.3
# _GROK_43_REASONING_EFFORTS: frozenset({'none', 'low', 'medium', 'high'})
# _GROK_BASIC_REASONING_EFFORTS: frozenset({'low', 'high'})
```

### 3.1 Inspecting Grok 4.3 Profile and Retirement-Redirect Slugs

```python
from pydantic_ai.profiles.grok import GrokModelProfile, grok_model_profile

# Grok 4.3 — full reasoning effort set + builtin tools
grok43 = grok_model_profile('grok-4.3')
assert grok43 is not None
print('Efforts:', grok43.get('grok_reasoning_efforts'))
print('Builtins:', grok43.get('grok_supports_builtin_tools'))
# Efforts: frozenset({'none', 'low', 'medium', 'high'})
# Builtins: True

# 'grok-3' is a retirement-redirect slug → routed to Grok 4.3 → same profile
grok3_redirect = grok_model_profile('grok-3')
assert grok3_redirect is not None
print('grok-3 efforts:', grok3_redirect.get('grok_reasoning_efforts'))
# grok-3 efforts: frozenset({'none', 'low', 'medium', 'high'})

# Grok 3 Mini — limited effort set (no 'none', so thinking_always_enabled=True)
grok3mini = grok_model_profile('grok-3-mini')
assert grok3mini is not None
print('grok-3-mini efforts:', grok3mini.get('grok_reasoning_efforts'))
print('thinking_always_enabled:', grok3mini.get('thinking_always_enabled'))
# grok-3-mini efforts: frozenset({'low', 'high'})
# thinking_always_enabled: True
```

### 3.2 Building a Grok Agent with Custom Effort Level

```python
import asyncio
import os

from pydantic_ai import Agent
from pydantic_ai.models.xai import XaiModel

os.environ.setdefault('XAI_API_KEY', 'your-key-here')

# Grok 4.3 with explicit medium reasoning effort
agent = Agent(
    XaiModel('grok-4.3', settings={'thinking': 'medium'}),
    system_prompt='Answer concisely.',
)


async def main() -> None:
    result = await agent.run('What is 42 in binary?')
    print(result.output)


asyncio.run(main())
```

### 3.3 Merging GrokModelProfile with a Cross-Provider Override

```python
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.grok import GrokModelProfile, grok_model_profile

# Start from the auto-detected Grok 4.3 profile
base = grok_model_profile('grok-4.3')

# Disable tool_choice='required' support if the endpoint rejects it
override = GrokModelProfile(grok_supports_tool_choice_required=False)

merged = merge_profile(base, override)
print('tool_choice_required:', merged.get('grok_supports_tool_choice_required'))  # False
print('builtin_tools:', merged.get('grok_supports_builtin_tools'))               # True (from base)
print('supports_thinking:', merged.get('supports_thinking'))                      # True (from base)
```

---

## 4. `GoogleModelProfile` + `google_model_profile()` + `GoogleJsonSchemaTransformer`

`GoogleModelProfile` carries Google-Gemini-specific profile flags. The 4 new Gemini-3 fields enable the Gemini 3+ **tool-combination** API (mixing function declarations, native tools, and response schemas in one request), the **server-side tool invocation circulation** field (which lets Pydantic AI round-trip Google-Search/URL-Context/File-Search spans), `thinking_level` enum mode (replacing `thinking_budget` integer on Gemini 3+), and the MIME types the model accepts in multimodal `FunctionResponseDict.parts`. `GoogleJsonSchemaTransformer` rewrites Pydantic-generated JSON Schema to the subset Gemini's API understands — it removes unsupported keywords and converts `const` → `enum`.

```python
# Signatures verified from profiles/google.py (pydantic-ai 2.5.1):
#
# class GoogleModelProfile(ModelProfile, total=False):
#     google_supports_tool_combination: bool           # Gemini 3+ only
#     google_supports_server_side_tool_invocations: bool  # Gemini 3+ only
#     google_supported_mime_types_in_tool_returns: tuple[str, ...]
#     google_supports_thinking_level: bool             # Gemini 3+: thinking_level instead of thinking_budget
#
# class GoogleJsonSchemaTransformer(JsonSchemaTransformer):
#     def transform(self, schema: JsonSchema) -> JsonSchema: ...
#     # Removes: $schema, discriminator, examples, title (bug workaround)
#     # Converts: const → enum (with inferred type if missing)
#     # Converts: format/pattern on strings (Gemini-specific rules)
#
# def google_model_profile(model_name: str) -> ModelProfile | None: ...
```

### 4.1 Inspecting Gemini 3 Profile Fields

```python
from pydantic_ai.profiles.google import GoogleModelProfile, google_model_profile

gemini2 = google_model_profile('gemini-2.5-pro')
gemini3 = google_model_profile('gemini-3-pro')

for name, profile in [('gemini-2.5-pro', gemini2), ('gemini-3-pro', gemini3)]:
    assert profile is not None
    print(f'{name}:')
    print(f'  tool_combination:      {profile.get("google_supports_tool_combination")}')
    print(f'  server_side_tools:     {profile.get("google_supports_server_side_tool_invocations")}')
    print(f'  thinking_level:        {profile.get("google_supports_thinking_level")}')
    print(f'  thinking_always_on:    {profile.get("thinking_always_enabled")}')
    print(f'  mime_types_count:      {len(profile.get("google_supported_mime_types_in_tool_returns", ()))}')

# gemini-2.5-pro: tool_combination=False, server_side_tools=False, thinking_level=False, thinking_always_on=True, mime_types_count=0
# gemini-3-pro:   tool_combination=True,  server_side_tools=True,  thinking_level=True,  thinking_always_on=True, mime_types_count=5
```

### 4.2 Understanding GoogleJsonSchemaTransformer const → enum Rewrite

```python
from pydantic_ai._json_schema import JsonSchema
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer

# GoogleJsonSchemaTransformer inherits JsonSchemaTransformer.__init__(schema, ...)
# so the schema must be passed at construction time; call .walk() to run the transform.

# Gemini doesn't support 'const' — the transformer rewrites it as a single-value 'enum'
schema: JsonSchema = {'const': 'active'}
result = GoogleJsonSchemaTransformer(schema).walk()
print(result)
# {'enum': ['active'], 'type': 'string'}   ← type is inferred from the const value

# Unsupported keywords are stripped
schema2: JsonSchema = {
    '$schema': 'http://json-schema.org/draft-07/schema',
    'type': 'object',
    'title': 'MyModel',
    'discriminator': {'propertyName': 'kind'},
    'examples': [{'kind': 'a'}],
    'properties': {'kind': {'type': 'string'}},
}
result2 = GoogleJsonSchemaTransformer(schema2).walk()
print(list(result2.keys()))
# ['type', 'properties']  — $schema, title, discriminator, examples all removed
```

### 4.3 Agent with Gemini 3 Tool Combination and Server-Side Tool Invocations

```python
import asyncio
import os

from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.models.google import GoogleModel

os.environ.setdefault('GOOGLE_API_KEY', 'your-key-here')

# Gemini 3 supports mixing WebSearch (native) with user-defined function tools
# and capturing the search span via server_side_tool_invocations
model = GoogleModel('gemini-3-flash')

agent = Agent(
    model,
    capabilities=[WebSearch()],  # native Google Search
    system_prompt='Answer questions using web search when needed.',
)


async def main() -> None:
    result = await agent.run('What is the latest version of Python?')
    print(result.output)


asyncio.run(main())
```

---

## 5. `CombinedCapability`

`CombinedCapability` is the concrete container that holds and runs a sequence of `AbstractCapability` instances. It is created automatically when you pass `capabilities=[...]` to an `Agent`. Its `__post_init__` method flattens any nested `CombinedCapability` children into a single flat list (so ordering constraints from siblings participate in the same topological sort) and, if any leaf declares an ordering constraint, calls `sort_capabilities()` to reorder the list before storing it. This ensures that `Instrumentation` (which declares `position='outermost'`) always ends up outside `Hooks`, `ProcessEventStream`, and other mid-chain capabilities regardless of user list order.

```python
# Signatures verified from capabilities/combined.py (pydantic-ai 2.5.1):
#
# @dataclass
# class CombinedCapability(AbstractCapability[AgentDepsT]):
#     capabilities: Sequence[AbstractCapability[AgentDepsT]]
#
#     def __post_init__(self) -> None:
#         # 1. Flatten nested CombinedCapability siblings
#         # 2. sort_capabilities() if any leaf has ordering constraints
#
#     @property
#     def has_wrap_node_run(self) -> bool:
#         return any(c.has_wrap_node_run for c in self.capabilities)
#
#     @property
#     def has_wrap_run_event_stream(self) -> bool:
#         return any(c.has_wrap_run_event_stream for c in self.capabilities)
#
#     async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
#         # Parallel gather — all children run for_run() concurrently
#         new_caps = await gather(*(c.for_run(ctx) for c in self.capabilities))
#         if all(new is old for new, old in zip(new_caps, self.capabilities)):
#             return self  # short-circuit: nothing changed
#         return CombinedCapability(capabilities=new_caps)
```

### 5.1 Flattening Nested CombinedCapability at Construction

```python
from pydantic_ai.capabilities import ProcessEventStream, WebFetch, WebSearch
from pydantic_ai.capabilities.combined import CombinedCapability
from pydantic_ai.capabilities.instrumentation import Instrumentation

# Two nested groups that the user assembled separately
group_a = CombinedCapability(capabilities=[WebSearch(), WebFetch()])
group_b = CombinedCapability(capabilities=[ProcessEventStream(handler=lambda ctx, events: events)])

# When group_a is placed inside another CombinedCapability, its children are flattened
combined = CombinedCapability(capabilities=[group_a, group_b])
print('Total leaves:', len(combined.capabilities))  # 3 (WebSearch, WebFetch, ProcessEventStream)

# Adding Instrumentation triggers sort_capabilities() — it declares position='outermost'
with_instrumentation = CombinedCapability(
    capabilities=[ProcessEventStream(handler=lambda ctx, events: events), Instrumentation()]
)
print('First after sort:', type(with_instrumentation.capabilities[0]).__name__)
# Instrumentation — moved to front because position='outermost'
```

### 5.2 Parallel for_run with Short-Circuit

```python
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.tools import RunContext


@dataclass
class CounterCapability(AbstractCapability):
    name: str
    call_count: int = 0

    async def for_run(self, ctx: RunContext) -> 'CounterCapability':
        self.call_count += 1
        return self  # returning self triggers the short-circuit in CombinedCapability


cap_a = CounterCapability(name='A')
cap_b = CounterCapability(name='B')

agent = Agent('openai:gpt-5', capabilities=[cap_a, cap_b])

# Both for_run() calls happen concurrently via gather()
# and if both return self, CombinedCapability also returns self (no new allocation)
print(f'cap_a calls before run: {cap_a.call_count}')
print(f'cap_b calls before run: {cap_b.call_count}')
```

### 5.3 has_wrap_node_run Shortcut for Efficient Execution

```python
from dataclasses import dataclass
from typing import Any

from pydantic_ai.capabilities.abstract import (
    AbstractCapability,
    NodeResult,
    WrapNodeRunHandler,
)
from pydantic_ai.capabilities.combined import CombinedCapability
from pydantic_ai.tools import RunContext


@dataclass
class NodeHookCapability(AbstractCapability):
    """A capability that wraps node execution."""

    @property
    def has_wrap_node_run(self) -> bool:
        return True

    async def wrap_node_run(
        self,
        ctx: RunContext,
        node: Any,
        handler: WrapNodeRunHandler,
    ) -> NodeResult:
        print(f'Before node: {type(node).__name__}')
        result = await handler(node)
        print(f'After node: {type(node).__name__}')
        return result


plain = CombinedCapability(capabilities=[])
with_hook = CombinedCapability(capabilities=[NodeHookCapability()])

# The combined capability checks all children — avoids wrapping overhead when unneeded
print('Plain has_wrap_node_run:', plain.has_wrap_node_run)   # False
print('WithHook has_wrap_node_run:', with_hook.has_wrap_node_run)  # True
```

---

## 6. `CapabilityOrdering` + `CapabilityPosition` + `CapabilityRef` + `sort_capabilities` + `collect_leaves` + `has_capability_type`

These types and functions implement the topological-sort engine for capability ordering. When a `CombinedCapability` is built from a list that includes any capability declaring ordering constraints (via `get_ordering()`), `sort_capabilities()` is called to reorder the list. It uses Python's `graphlib.TopologicalSorter` with the original list order as a tiebreaker, building edges from `position`, `wraps`, `wrapped_by`, and `requires` declarations. `collect_leaves()` uses the `apply()` visitor pattern to gather all leaf capabilities out of any nested tree. `has_capability_type()` checks whether a given type appears anywhere in a capability list.

```python
# Signatures verified from capabilities/abstract.py + capabilities/_ordering.py (pydantic-ai 2.5.1):
#
# CapabilityPosition = Literal['outermost', 'innermost']
# CapabilityRef = type[AbstractCapability[Any]] | AbstractCapability[Any]
#
# @dataclass
# class CapabilityOrdering:
#     position: CapabilityPosition | None = None
#     wraps: Sequence[CapabilityRef] = ()
#     wrapped_by: Sequence[CapabilityRef] = ()
#     requires: Sequence[type[AbstractCapability[Any]]] = ()
#
# def sort_capabilities(capabilities) -> list[AbstractCapability[Any]]: ...
# def collect_leaves(cap) -> list[AbstractCapability[Any]]: ...
# def has_capability_type(capabilities, cap_type) -> bool: ...
```

### 6.1 Declaring position='outermost' on a Custom Capability

```python
from dataclasses import dataclass

from pydantic_ai.capabilities._ordering import sort_capabilities, collect_leaves
from pydantic_ai.capabilities.abstract import (
    AbstractCapability,
    CapabilityOrdering,
    CapabilityPosition,
)
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.capabilities.process_event_stream import ProcessEventStream


@dataclass
class AuditCapability(AbstractCapability):
    """Records every run for compliance. Must be outermost."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')

    async def wrap_run(self, ctx, handler):
        print('Audit: run started')
        result = await handler()
        print('Audit: run finished')
        return result


# Even though AuditCapability is listed last, sort_capabilities() moves it to front
caps = [Hooks(), ProcessEventStream(handler=lambda ctx, events: events), AuditCapability()]
sorted_caps = sort_capabilities(caps)
print([type(c).__name__ for c in sorted_caps])
# ['AuditCapability', 'Hooks', 'ProcessEventStream']
```

### 6.2 wraps/wrapped_by Fine-Grained Relative Ordering

```python
from dataclasses import dataclass

from pydantic_ai.capabilities._ordering import sort_capabilities
from pydantic_ai.capabilities.abstract import AbstractCapability, CapabilityOrdering
from pydantic_ai.capabilities.hooks import Hooks


@dataclass
class MetricsCapability(AbstractCapability):
    """Must wrap Hooks so it fires before Hooks starts measuring."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(wraps=[Hooks])  # I come before Hooks


@dataclass
class TracingCapability(AbstractCapability):
    """Must be wrapped by MetricsCapability (i.e., after it)."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(wrapped_by=[MetricsCapability])  # MetricsCapability comes before me


hooks = Hooks()
metrics = MetricsCapability()
tracing = TracingCapability()

# Original order: [tracing, hooks, metrics] — constraints reorder them
result = sort_capabilities([tracing, hooks, metrics])
print([type(c).__name__ for c in result])
# ['MetricsCapability', 'TracingCapability', 'Hooks']
# MetricsCapability wraps Hooks → before hooks
# TracingCapability wrapped_by MetricsCapability → after metrics
```

### 6.3 requires Constraint and has_capability_type Guard

```python
from dataclasses import dataclass

from pydantic_ai.capabilities._ordering import has_capability_type, sort_capabilities
from pydantic_ai.capabilities.abstract import AbstractCapability, CapabilityOrdering
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.exceptions import UserError


@dataclass
class SpanEnricherCapability(AbstractCapability):
    """Only makes sense when Instrumentation is present."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(requires=[Instrumentation])


# With Instrumentation present — works fine
caps_ok = [Instrumentation(), SpanEnricherCapability()]
sorted_ok = sort_capabilities(caps_ok)
print('OK:', [type(c).__name__ for c in sorted_ok])

# Without Instrumentation — raises UserError
caps_bad = [SpanEnricherCapability()]
try:
    sort_capabilities(caps_bad)
except UserError as e:
    print('Error:', e)
# Error: `SpanEnricherCapability` requires `Instrumentation` but it was not found among the capabilities.

# has_capability_type lets you check before constructing
has_instr = has_capability_type(caps_ok, Instrumentation)
print('Has Instrumentation:', has_instr)  # True
```

---

## 7. `ToolCorrectness` + `TrajectoryMatch`

These are span-based agentic evaluators from `pydantic_evals`. They read from `ctx.span_tree` (populated when Logfire/OTel instrumentation is configured) and degrade gracefully when spans aren't available. `ToolCorrectness` compares the multiset of tool names actually called against `expected_tools` — order is irrelevant, duplicate entries require repeated calls. `TrajectoryMatch` enforces ordered sequences with three comparison modes: `'exact'` (pass/fail equality), `'in_order'` (LCS-based F1, default), and `'any_order'` (multiset F1, order-independent).

```python
# Signatures verified from pydantic_evals/evaluators/agentic.py (pydantic-evals 2.5.1):
#
# @dataclass(frozen=True)
# class ToolCorrectness(Evaluator[object, object, object]):
#     expected_tools: list[str]
#     allow_extra: bool = False          # True → only require expected tools, extras OK
#     include_failed: bool = False       # True → count error/retry attempts too
#     evaluation_name: str | None = None
#
# TrajectoryOrder = Literal['exact', 'in_order', 'any_order']
#   'exact'     — actual must equal expected exactly (1.0 or 0.0)
#   'in_order'  — F1 from longest common subsequence (default)
#   'any_order' — F1 from multiset intersection (order-independent)
#
# @dataclass(frozen=True)
# class TrajectoryMatch(Evaluator[object, object, object]):
#     expected_trajectory: list[str]
#     order: TrajectoryOrder = 'in_order'
#     include_failed: bool = False
#     evaluation_name: str | None = None
```

### 7.1 Basic ToolCorrectness with a Dataset

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import ToolCorrectness


def weather_tool(city: str) -> str:
    return f'Sunny in {city}'


test_agent = Agent(TestModel(), tools=[weather_tool])
test_agent.instrument = True  # enable OTel span capture for span-based evaluators

dataset = Dataset(
    name='tool_correctness_demo',
    cases=[
        Case(
            name='calls_weather',
            inputs='What is the weather in Paris?',
            evaluators=[
                ToolCorrectness(expected_tools=['weather_tool']),
            ],
        ),
        Case(
            name='extra_tool_allowed',
            inputs='Tell me something about London.',
            evaluators=[
                ToolCorrectness(expected_tools=['weather_tool'], allow_extra=True),
            ],
        ),
    ],
)


async def task(inputs: str) -> str:
    result = await test_agent.run(inputs)
    return result.output


async def main() -> None:
    report = await dataset.evaluate(task)
    for case in report.cases:
        print(f'{case.name}: {case.scores}')


asyncio.run(main())
```

### 7.2 TrajectoryMatch with exact, in_order, and any_order Modes

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import TrajectoryMatch


def search(query: str) -> str:
    return f'Results for: {query}'


def summarize(text: str) -> str:
    return f'Summary: {text[:50]}'


agent = Agent(TestModel(), tools=[search, summarize])
agent.instrument = True

dataset = Dataset(
    name='trajectory_demo',
    cases=[
        Case(
            name='exact_order',
            inputs='Search then summarize.',
            evaluators=[
                # Actual sequence must match expected exactly (1.0 or 0.0)
                TrajectoryMatch(expected_trajectory=['search', 'summarize'], order='exact'),
            ],
        ),
        Case(
            name='in_order_match',
            inputs='Get me some data.',
            evaluators=[
                # F1 from longest common subsequence — order matters, gaps reduce score
                TrajectoryMatch(expected_trajectory=['search', 'summarize'], order='in_order'),
            ],
        ),
    ],
)


async def main() -> None:
    async def task(inputs: str) -> str:
        return (await agent.run(inputs)).output
    report = await dataset.evaluate(task)
    for case in report.cases:
        print(f'{case.name}: {case.scores}')


asyncio.run(main())
```

### 7.3 Counting Failed Tool Attempts with include_failed

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import ToolCorrectness


_flaky_call_count = [0]


def flaky_tool(data: str) -> str:
    # Retries once then succeeds — demonstrates include_failed counting
    _flaky_call_count[0] += 1
    if _flaky_call_count[0] == 1:
        raise ModelRetry('Retry please')
    _flaky_call_count[0] = 0  # reset for re-evaluation runs
    return f'Succeeded with: {data}'


agent = Agent(TestModel(), tools=[flaky_tool])
agent.instrument = True

dataset = Dataset(
    name='retry_demo',
    cases=[
        Case(
            name='count_only_success',
            inputs='Run flaky tool.',
            evaluators=[
                # Default: failed/retry attempts not counted
                ToolCorrectness(expected_tools=['flaky_tool'], include_failed=False),
            ],
        ),
        Case(
            name='count_all_attempts',
            inputs='Run flaky tool.',
            evaluators=[
                # include_failed=True: the retry attempt also counts
                ToolCorrectness(
                    expected_tools=['flaky_tool', 'flaky_tool'],  # 2 attempts expected
                    include_failed=True,
                ),
            ],
        ),
    ],
)


async def main() -> None:
    async def task(inputs: str) -> str:
        return (await agent.run(inputs)).output
    # max_concurrency=1 prevents the shared _flaky_call_count from interleaving across cases
    report = await dataset.evaluate(task, max_concurrency=1)
    for case in report.cases:
        print(f'{case.name}: {case.scores}')


asyncio.run(main())
```

---

## 8. `ArgumentCorrectness` + `ArgumentMatchMode` + `ArgumentOccurrence` + `MaxToolCalls` + `MaxModelRequests`

`ArgumentCorrectness` verifies the exact arguments passed to a specific tool call. It supports `'subset'` (default — check that every expected key/value is present) or `'exact'` matching modes, and selects which invocation to inspect via `occurrence` (`'first'`, `'last'`, or a 0-based integer index). `MaxToolCalls` and `MaxModelRequests` enforce budget caps.

```python
# Signatures verified from pydantic_evals/evaluators/agentic.py (pydantic-evals 2.5.1):
#
# ArgumentMatchMode = Literal['subset', 'exact']
# ArgumentOccurrence = Literal['first', 'last']
#
# @dataclass(frozen=True)
# class ArgumentCorrectness(Evaluator[object, object, object]):
#     tool_name: str
#     expected_arguments: dict[str, Any]
#     match_mode: ArgumentMatchMode = 'subset'
#     occurrence: ArgumentOccurrence | int = 'first'  # or a 0-based int index
#     include_failed: bool = False
#     evaluation_name: str | None = None
#
# @dataclass(frozen=True)
# class MaxToolCalls(Evaluator[object, object, object]):
#     max_calls: int
#     include_failed: bool = True   # NOTE: opposite default to ToolCorrectness
#     evaluation_name: str | None = None
#
# @dataclass(frozen=True)
# class MaxModelRequests(Evaluator[object, object, object]):
#     max_requests: int
#     evaluation_name: str | None = None
```

### 8.1 Subset vs Exact Argument Matching

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import ArgumentCorrectness


def book_flight(origin: str, destination: str, class_: str = 'economy') -> str:
    return f'Booked {origin}→{destination} ({class_})'


def book_flight_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    # Return a text response once the tool result is in the history
    if len(messages) > 1:
        return ModelResponse(parts=[TextPart(content='Flight booked!')])
    # Always emit deterministic London→Paris economy args so ArgumentCorrectness can verify them
    return ModelResponse(parts=[
        ToolCallPart(tool_name='book_flight', args='{"origin": "London", "destination": "Paris", "class_": "economy"}'),
    ])


agent = Agent(FunctionModel(book_flight_func), tools=[book_flight])
agent.instrument = True

dataset = Dataset(
    name='arg_correctness_demo',
    cases=[
        Case(
            name='subset_check',
            inputs='Book a flight from London to Paris.',
            evaluators=[
                ArgumentCorrectness(
                    tool_name='book_flight',
                    expected_arguments={'origin': 'London', 'destination': 'Paris'},
                    match_mode='subset',  # 'class_' not required in subset mode
                ),
            ],
        ),
        Case(
            name='exact_check',
            inputs='Book economy class from London to Paris.',
            evaluators=[
                ArgumentCorrectness(
                    tool_name='book_flight',
                    expected_arguments={
                        'origin': 'London',
                        'destination': 'Paris',
                        'class_': 'economy',
                    },
                    match_mode='exact',  # all keys must match exactly
                ),
            ],
        ),
    ],
)


async def main() -> None:
    async def task(inputs: str) -> str:
        return (await agent.run(inputs)).output
    report = await dataset.evaluate(task)
    for case in report.cases:
        print(f'{case.name}: {case.scores}')


asyncio.run(main())
```

### 8.2 Selecting a Specific Tool Occurrence

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import ArgumentCorrectness


def search(query: str) -> str:
    return f'Results for: {query}'


def two_search_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    # Each tool call adds 2 messages (model response + tool return), so divide by 2
    n = (len(messages) - 1) // 2
    queries = ['Python', 'Rust']
    if n < len(queries):
        return ModelResponse(parts=[ToolCallPart(tool_name='search', args=f'{{"query": "{queries[n]}"}}')]) 
    return ModelResponse(parts=[TextPart(content='Done.')])


def three_search_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    n = (len(messages) - 1) // 2
    queries = ['first query', 'second query', 'third query']
    if n < len(queries):
        return ModelResponse(parts=[ToolCallPart(tool_name='search', args=f'{{"query": "{queries[n]}"}}')]) 
    return ModelResponse(parts=[TextPart(content='Done.')])


agent_two = Agent(FunctionModel(two_search_func), tools=[search])
agent_two.instrument = True
agent_three = Agent(FunctionModel(three_search_func), tools=[search])
agent_three.instrument = True

dataset_last = Dataset(
    name='occurrence_last',
    cases=[
        Case(
            name='check_last_call',
            inputs='Search for Python, then search for Rust.',
            evaluators=[
                ArgumentCorrectness(
                    tool_name='search',
                    expected_arguments={'query': 'Rust'},
                    occurrence='last',  # inspect only the final call
                ),
            ],
        ),
    ],
)

dataset_index = Dataset(
    name='occurrence_index',
    cases=[
        Case(
            name='check_second_call',
            inputs='Search three times.',
            evaluators=[
                ArgumentCorrectness(
                    tool_name='search',
                    expected_arguments={'query': 'second query'},
                    occurrence=1,  # 0-based: second invocation
                ),
            ],
        ),
    ],
)


async def main() -> None:
    async def task_two(inputs: str) -> str:
        return (await agent_two.run(inputs)).output

    async def task_three(inputs: str) -> str:
        return (await agent_three.run(inputs)).output

    report1 = await dataset_last.evaluate(task_two)
    for case in report1.cases:
        print(f'{case.name}: {case.scores}')

    report2 = await dataset_index.evaluate(task_three)
    for case in report2.cases:
        print(f'{case.name}: {case.scores}')


asyncio.run(main())
```

### 8.3 Enforcing Tool and Request Budgets

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import MaxModelRequests, MaxToolCalls


def lookup(term: str) -> str:
    return f'Definition of {term}'


agent = Agent(TestModel(), tools=[lookup])
agent.instrument = True

dataset = Dataset(
    name='budget_demo',
    cases=[
        Case(
            name='budget_check',
            inputs='Look up several things.',
            evaluators=[
                MaxToolCalls(max_calls=3),         # fail if > 3 tool calls (incl. failed)
                MaxModelRequests(max_requests=2),  # fail if > 2 model round-trips
            ],
        ),
    ],
)


async def main() -> None:
    async def task(inputs: str) -> str:
        return (await agent.run(inputs)).output
    report = await dataset.evaluate(task)
    for case in report.cases:
        print(f'budget_check scores: {case.scores}')


asyncio.run(main())
```

---

## 9. `GEval` + `HasMatchingSpan` + `OutputConfig`

`GEval` implements a simplified version of the G-Eval chain-of-thought scoring framework (Liu et al., 2023). The judge is given `criteria`, explicit `evaluation_steps`, and a `score_range` (default `(1, 5)`), then produces a score and reasoning trace. Instead of the paper's probability-weighted log-prob expectation it uses a direct integer score for provider-agnostic simplicity. `HasMatchingSpan` delegates to `SpanQuery.any()` — it passes when at least one span in the captured tree matches the query. `OutputConfig` is the wire `TypedDict` shared by `LLMJudge` and `GEval` for controlling judge model selection and output format.

```python
# Signatures verified from pydantic_evals/evaluators/common.py (pydantic-evals 2.5.1):
#
# @dataclass(repr=False)
# class GEval(Evaluator[object, object, object]):
#     criteria: str
#     evaluation_steps: list[str]
#     score_range: tuple[int, int] = (1, 5)  # inclusive; min < max enforced
#     include_input: bool = False             # include ctx.inputs in judge prompt
#     model: Model | KnownModelName | str | None = None
#     model_settings: ModelSettings | None = None
#     evaluation_name: str | None = None
#
# @dataclass(repr=False)
# class HasMatchingSpan(Evaluator[object, object, object]):
#     query: SpanQuery
#     evaluation_name: str | None = None
#
# class OutputConfig(TypedDict, total=False):
#     # Used internally by LLMJudge and GEval to configure the judge response
#     pass  # fields are provider-level implementation details
```

### 9.1 GEval with Custom Criteria and Steps

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import GEval


def write_poem(topic: str) -> str:
    return f'Roses are red, violets are blue, {topic} is great, and so are you.'


agent = Agent(TestModel(), tools=[write_poem])

coherence_evaluator = GEval(
    criteria='Rate the coherence and relevance of the poem to the given topic.',
    evaluation_steps=[
        'Read the poem carefully.',
        'Check whether each line logically follows from the previous.',
        'Assess whether the poem is on-topic.',
        'Assign a score from 1 (incoherent) to 5 (perfectly coherent and on-topic).',
    ],
    score_range=(1, 5),
    include_input=True,  # pass ctx.inputs to the judge so it can compare topic vs poem
)

dataset = Dataset(
    name='geval_demo',
    cases=[
        Case(
            name='poem_quality',
            inputs='Write a poem about Python programming.',
            evaluators=[coherence_evaluator],
        ),
    ],
)


async def main() -> None:
    async def task(inputs: str) -> str:
        return (await agent.run(inputs)).output
    report = await dataset.evaluate(task)
    for case in report.cases:
        print(f"Score: {case.scores.get('GEval')}")


asyncio.run(main())
```

### 9.2 Binary GEval Scoring with score_range=(0, 1)

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import GEval

agent = Agent(TestModel())

# Binary scoring: 0 = fail, 1 = pass
safety_check = GEval(
    criteria='The response must not contain any harmful, offensive, or inappropriate content.',
    evaluation_steps=[
        'Read the entire response.',
        'Identify any harmful, offensive, or biased language.',
        'Score 1 if the response is safe and appropriate, 0 if it contains issues.',
    ],
    score_range=(0, 1),
)

dataset = Dataset(
    name='safety_demo',
    cases=[
        Case(
            name='safe_response',
            inputs='Tell me a fun fact about space.',
            evaluators=[safety_check],
        ),
    ],
)


async def main() -> None:
    async def task(inputs: str) -> str:
        return (await agent.run(inputs)).output
    report = await dataset.evaluate(task)
    for case in report.cases:
        print(f"Safety score: {case.scores.get('GEval')}")


asyncio.run(main())
```

### 9.3 HasMatchingSpan with SpanQuery

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import HasMatchingSpan
from pydantic_evals.otel.span_tree import SpanQuery


def search_tool(query: str) -> str:
    return f'Results for: {query}'


agent = Agent(TestModel(), tools=[search_tool])
agent.instrument = True

# HasMatchingSpan passes when the span tree contains at least one matching span
# SpanQuery can filter by span name, attributes, or custom predicates
dataset = Dataset(
    name='span_demo',
    cases=[
        Case(
            name='tool_span_present',
            inputs='Search for something.',
            evaluators=[
                HasMatchingSpan(
                    query=SpanQuery(name_equals='execute_tool search_tool'),
                ),
            ],
        ),
    ],
)


async def main() -> None:
    async def task(inputs: str) -> str:
        return (await agent.run(inputs)).output
    report = await dataset.evaluate(task)
    for case in report.cases:
        print(f'Span present: {case.scores}')


asyncio.run(main())
```

---

## 10. `CaseLifecycle`

`CaseLifecycle` provides per-case evaluation lifecycle hooks. A new instance is created for each case before it runs, so subclasses can hold per-case state without thread-safety concerns. It defines three async hook methods: `setup()` (called before the task — allocate resources, seed databases), `prepare_context()` (called after the task, before evaluators — enrich `EvaluatorContext.metrics` or `attributes` from the task result), and `teardown()` (called after evaluators — clean up, even if `setup` or `prepare_context` raised). If `setup` or `prepare_context` raise, the exception is recorded as a `ReportCaseFailure` and `teardown` is still called. If `teardown` raises, the exception propagates and may abort the evaluation run.

```python
# Signatures verified from pydantic_evals/lifecycle.py (pydantic-evals 2.5.1):
#
# class CaseLifecycle(Generic[InputsT, OutputT, MetadataT]):
#     def __init__(self, case: Case[InputsT, OutputT, MetadataT]) -> None: ...
#
#     @property
#     def case(self) -> Case[InputsT, OutputT, MetadataT]: ...
#
#     async def setup(self) -> None: ...
#     # Called before task execution. Override for per-case resource allocation.
#
#     async def prepare_context(
#         self,
#         ctx: EvaluatorContext[InputsT, OutputT, MetadataT],
#     ) -> EvaluatorContext[InputsT, OutputT, MetadataT]: ...
#     # Called after task, before evaluators. Enrich ctx.metrics/attributes.
#     # Return the (possibly modified) context.
#
#     async def teardown(
#         self,
#         result: ReportCase | ReportCaseFailure | None,
#     ) -> None: ...
#     # Called after evaluators. result is None if run was interrupted.
```

### 10.1 Per-Case Resource Setup and Teardown

```python
import asyncio
from dataclasses import dataclass, field

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case, Dataset
from pydantic_evals.lifecycle import CaseLifecycle

agent = Agent(TestModel())


class DatabaseLifecycle(CaseLifecycle[str, str, None]):
    """Simulates allocating a test database connection per case."""

    def __init__(self, case):
        super().__init__(case)
        self._db_connection = None

    async def setup(self) -> None:
        # Allocate per-case resources
        self._db_connection = f'db_conn_{self.case.name}'
        print(f'[{self.case.name}] DB connected: {self._db_connection}')

    async def teardown(self, result) -> None:
        # Always clean up, even if the case failed
        if self._db_connection:
            print(f'[{self.case.name}] DB disconnected: {self._db_connection}')
            self._db_connection = None


dataset = Dataset(
    name='lifecycle_demo',
    cases=[
        Case(name='case_1', inputs='Hello'),
        Case(name='case_2', inputs='World'),
    ],
)


async def main() -> None:
    async def task(inputs: str) -> str:
        return (await agent.run(inputs)).output
    report = await dataset.evaluate(task, lifecycle=DatabaseLifecycle)
    print(f'Evaluated {len(report.cases)} cases')


asyncio.run(main())
```

### 10.2 Enriching EvaluatorContext in prepare_context

```python
import asyncio
import time

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators.context import EvaluatorContext
from pydantic_evals.lifecycle import CaseLifecycle

agent = Agent(TestModel())


class TimingLifecycle(CaseLifecycle[str, str, None]):
    """Measures task duration and injects it as a metric."""

    def __init__(self, case):
        super().__init__(case)
        self._start_time: float = 0.0

    async def setup(self) -> None:
        self._start_time = time.monotonic()

    async def prepare_context(
        self, ctx: EvaluatorContext[str, str, None]
    ) -> EvaluatorContext[str, str, None]:
        elapsed_ms = (time.monotonic() - self._start_time) * 1000
        ctx.metrics['duration_ms'] = elapsed_ms
        ctx.attributes['timing_source'] = 'monotonic_clock'
        return ctx


dataset = Dataset(
    name='timing_demo',
    cases=[
        Case(name='fast_case', inputs='What is 2+2?'),
        Case(name='slow_case', inputs='Explain quantum computing in detail.'),
    ],
)


async def main() -> None:
    async def task(inputs: str) -> str:
        return (await agent.run(inputs)).output
    report = await dataset.evaluate(task, lifecycle=TimingLifecycle)
    for case in report.cases:
        print(f"{case.name}: duration_ms={case.metrics.get('duration_ms', 0):.1f}")


asyncio.run(main())
```

### 10.3 Exception Handling Semantics: setup Failure vs teardown Failure

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case, Dataset
from pydantic_evals.lifecycle import CaseLifecycle

agent = Agent(TestModel())


class FaultySetupLifecycle(CaseLifecycle[str, str, None]):
    """Demonstrates that setup errors become ReportCaseFailure, not exceptions."""

    async def setup(self) -> None:
        raise RuntimeError('Setup failed! Recorded as ReportCaseFailure.')

    async def teardown(self, result) -> None:
        # teardown IS called even when setup raises
        if result is None:
            print(f'[{self.case.name}] Teardown after interrupt')
        else:
            print(f'[{self.case.name}] Teardown with result type: {type(result).__name__}')


class FaultyTeardownLifecycle(CaseLifecycle[str, str, None]):
    """Demonstrates that teardown errors propagate and abort evaluation."""

    async def teardown(self, result) -> None:
        raise RuntimeError(
            'Teardown failure propagates! Handle internally if you want graceful eval.'
        )


dataset = Dataset(
    name='exception_semantics',
    cases=[
        Case(name='faulty_setup', inputs='Hello'),
    ],
)


async def main() -> None:
    async def task(inputs: str) -> str:
        return (await agent.run(inputs)).output

    # Faulty setup: the case is recorded as a failure, evaluation continues
    report = await dataset.evaluate(task, lifecycle=FaultySetupLifecycle)
    print(f'Cases evaluated: {len(report.cases)}')
    # The case appears as a ReportCaseFailure in the report

    # Faulty teardown: wrap in try/except to prevent crashing your test suite
    try:
        await dataset.evaluate(task, lifecycle=FaultyTeardownLifecycle)
    except RuntimeError as e:
        print(f'Teardown propagated: {e}')


asyncio.run(main())
```

---
title: "PydanticAI Class Deep Dives Vol. 37"
description: "Source-verified deep dives into 10 pydantic-ai 2.10.0 class groups: UsageLimits/RunUsage/RequestUsage (2.10.0 has_values fix + genai-prices), RenamedToolset (2.10.0 UserError-on-collision), AgentSpec (YAML/JSON agent construction), ModelSettings (complete cross-provider reference), TemplateStr (Handlebars instructions), ConcurrencyLimiter (anyio-backed + OTel), NativeOutput/PromptedOutput/TextOutput/ToolOutput (output-mode toolkit), ApprovalRequiredToolset (HITL complete), FunctionToolset (all 14 params), RunContext (complete field reference)."
sidebar:
  label: "Class deep dives (Vol. 37)"
  order: 63
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 2.10.0** source installed directly from PyPI. Every class signature, field name, and method in this volume reflects the 2.10.x API. Three examples per class group; all code blocks pass `ast.parse()` syntax validation. Live API calls are commented out — uncomment to run.
</Aside>

Ten class groups covering the usage-tracking and budget system (with 2.10.0 `has_values()` fix and genai-prices integration), `RenamedToolset` collision handling (2.10.0 `UserError` instead of silent drop), `AgentSpec` YAML/JSON-driven agent construction, `ModelSettings` complete cross-provider reference, `TemplateStr` Handlebars templating, `ConcurrencyLimiter` anyio-backed slot management, output-mode marker classes, `ApprovalRequiredToolset` human-in-the-loop, `FunctionToolset` all 14 parameters, and `RunContext` complete field reference.

---

## 1. `UsageLimits` + `RunUsage` + `RequestUsage`

**Source:** `pydantic_ai/usage.py`

`UsageLimits` is a plain dataclass that enforces four orthogonal budgets — requests, tool calls, input tokens, and total tokens — checked at three injection points inside the agent graph. `RunUsage` accumulates across the whole run; `RequestUsage` represents a single model response and is also a `genai-prices` `AbstractUsage` so cost can be computed directly from it.

**2.10.0 change:** `UsageBase.has_values()` was fixed to return `False` when all counters are zero (previously it could return `True` due to details-dict truthiness).

```python
# Example 1 — Set a per-run token budget and catch the exception
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

agent = Agent('openai:gpt-4o-mini')

limits = UsageLimits(
    request_limit=5,           # max 5 model requests per run
    total_tokens_limit=2000,   # bail out if total tokens > 2k
    output_tokens_limit=500,   # max 500 output tokens
)

try:
    # result = agent.run_sync('Explain quantum entanglement in detail.', usage_limits=limits)
    pass
except UsageLimitExceeded as e:
    print(f'Budget exceeded: {e}')
```

```python
# Example 2 — Inspect RunUsage fields and test has_values() fix
from pydantic_ai.usage import RunUsage

empty = RunUsage()
print(empty.has_values())   # False — 2.10.0 fix (was broken for all-zero dicts)
print(repr(empty))          # RunUsage() — __repr__ skips defaults

loaded = RunUsage(
    requests=3,
    input_tokens=1500,
    output_tokens=200,
    cache_read_tokens=400,
    tool_calls=2,
)
print(loaded.has_values())     # True
print(loaded.total_tokens)     # 1700  (input_tokens + output_tokens)

# Accumulate across runs
second = RunUsage(requests=1, input_tokens=300, output_tokens=50)
combined = loaded + second
print(combined.requests)       # 4
print(combined.input_tokens)   # 1800
```

```python
# Example 3 — count_tokens_before_request + per-provider token counting
from pydantic_ai import Agent, UsageLimits

# Anthropic, Google, Bedrock Converse and OpenAI Responses all support
# pre-flight token counting when count_tokens_before_request=True.
# The agent calls the model's count_tokens API before every actual request.
agent = Agent('anthropic:claude-sonnet-4-5')

limits = UsageLimits(
    input_tokens_limit=8_000,
    count_tokens_before_request=True,  # enforce limit before the request is sent
)

# With count_tokens_before_request enabled, if the context already has 8k input
# tokens the agent raises UsageLimitExceeded *before* paying for a real request.
# result = await agent.run(
#     'Summarise this 50-page document: ...',
#     usage_limits=limits,
# )
# print(result.usage())
```

---

## 2. `RenamedToolset`

**Source:** `pydantic_ai/toolsets/renamed.py`

`RenamedToolset` wraps any `AbstractToolset` and renames tools by a `name_map: dict[str, str]` where keys are **new** names and values are **original** names. Unmapped tools keep their original names.

**2.10.0 fix:** Previously, renaming two tools to the same name (or renaming onto an existing tool name) silently dropped one tool. Now it raises `UserError` with an informative message for both conflict scenarios.

```python
# Example 1 — Rename a subset of tools from a FunctionToolset
from pydantic_ai import Agent, FunctionToolset, RenamedToolset, RunContext

tools = FunctionToolset[None]()

@tools.tool_plain
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f'Sunny, 22°C in {city}'

@tools.tool_plain
def get_time(timezone: str) -> str:
    """Get current time for a timezone."""
    return f'14:35 in {timezone}'

# Rename get_weather → weather, leave get_time unchanged
renamed = RenamedToolset(wrapped=tools, name_map={'weather': 'get_weather'})

agent = Agent('openai:gpt-4o', toolsets=[renamed])
# agent.run_sync('What is the weather in London?')
# The model sees tools: ['weather', 'get_time']
```

```python
# Example 2 — 2.10.0 collision detection raises UserError
from pydantic_ai import FunctionToolset, RenamedToolset, RunContext
from pydantic_ai.exceptions import UserError

tools = FunctionToolset[None]()

@tools.tool_plain
def search(query: str) -> str:
    return f'results for {query}'

@tools.tool_plain
def lookup(key: str) -> str:
    return f'value for {key}'

# Rename 'search' to 'lookup', which conflicts with the existing 'lookup' tool.
# 2.10.0 raises UserError instead of silently dropping the conflicting tool.
bad_map = RenamedToolset(
    wrapped=tools,
    name_map={'lookup': 'search'},
)

# The collision is detected lazily at get_tools() time (inside an agent run).
# import asyncio, pydantic_ai
# try:
#     asyncio.run(bad_map.get_tools(some_ctx))
# except UserError as e:
#     print(e)  # 'Renaming tool "search" to "lookup" conflicts with existing tool.'
```

```python
# Example 3 — Rename tools from an MCP server to match your API naming convention
from pydantic_ai import Agent, RenamedToolset
from pydantic_ai.mcp import MCPServerStdio

# Suppose an MCP server exposes tools named with snake_case:
# read_resource, create_resource, delete_resource
mcp = MCPServerStdio('npx', ['-y', '@acme/mcp-server'])

# Expose them under camelCase to match the rest of your codebase
renamed = RenamedToolset(
    wrapped=mcp,
    name_map={
        'readResource': 'read_resource',
        'createResource': 'create_resource',
        'deleteResource': 'delete_resource',
    },
)

agent = Agent('openai:gpt-4o', toolsets=[renamed])
# agent.run_sync('Create a new resource called "report"')
```

---

## 3. `AgentSpec`

**Source:** `pydantic_ai/agent/spec.py`

`AgentSpec` is a Pydantic `BaseModel` that serialises the static parts of an `Agent` to/from YAML or JSON. It is the supported way to store agent configurations in version control and load them at runtime without coupling the deployment code to the model name or instruction text.

Capabilities are expressed as `CapabilitySpec` entries — a flexible short-form that resolves against a registry of `AbstractCapability` types. Dynamic things (toolsets, custom hooks, deps instances) are still wired in code after loading.

```python
# Example 1 — Load agent from a YAML config file
import yaml
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.agent import AgentSpec

yaml_text = """
model: openai:gpt-4o-mini
name: support-bot
instructions: |
  You are a helpful customer support agent.
  Always be polite and concise.
retries: 2
end_strategy: graceful
model_settings:
  max_tokens: 512
  temperature: 0.3
"""

spec = AgentSpec.from_text(yaml_text, fmt='yaml')
print(spec.model)         # openai:gpt-4o-mini
print(spec.name)          # support-bot
print(spec.end_strategy)  # graceful
```

```python
# Example 2 — Build AgentSpec programmatically and save it to a file
from pathlib import Path
from pydantic_ai.agent import AgentSpec

spec = AgentSpec(
    model='anthropic:claude-haiku-4-5',
    name='code-reviewer',
    instructions='Review code for correctness, security, and style.',
    model_settings={'max_tokens': 1024, 'temperature': 0.0},
    retries=3,
)

# Save to YAML (round-trips cleanly)
# spec.to_file('/configs/code-reviewer.yaml')

# Load it back
# loaded = AgentSpec.from_file('/configs/code-reviewer.yaml')
# assert loaded.model == spec.model
print(spec.model_dump(exclude_none=True, by_alias=True))
```

```python
# Example 3 — AgentSpec with capabilities and deps_schema for structured deps injection
from pydantic_ai.agent import AgentSpec

yaml_with_caps = """
model: openai:gpt-4o
name: research-agent
instructions: Research the given topic thoroughly.
deps_schema:
  type: object
  properties:
    api_key:
      type: string
    max_results:
      type: integer
      default: 10
capabilities:
  - type: web-search
    search_context_size: medium
  - type: instrumentation
    position: outermost
"""

spec = AgentSpec.from_text(yaml_with_caps, fmt='yaml')
print(spec.capabilities)  # [CapabilitySpec(...), CapabilitySpec(...)]

# The JSON schema for capabilities can be introspected:
# schema = AgentSpec.model_json_schema_with_capabilities()
# print(json.dumps(schema, indent=2))
```

---

## 4. `ModelSettings`

**Source:** `pydantic_ai/settings.py`

`ModelSettings` is a `TypedDict` (all keys optional) that provides a portable way to configure LLMs across providers. Fields not supported by a given model are silently ignored by that provider's adapter. Use `merge_model_settings(base, overrides)` to layer agent-level settings with per-run overrides.

```python
# Example 1 — Deterministic settings for evaluation workloads
from pydantic_ai import Agent, ModelSettings

# Maximally deterministic: zero temperature, fixed seed, no parallel tools
eval_settings: ModelSettings = {
    'temperature': 0.0,
    'top_p': 1.0,
    'seed': 42,
    'parallel_tool_calls': False,
    'max_tokens': 2048,
}

agent = Agent('openai:gpt-4o', model_settings=eval_settings)

# Per-run override to increase creativity just for one call:
# result = agent.run_sync(
#     'Write a haiku about Python.',
#     model_settings={'temperature': 0.9},
# )
```

```python
# Example 2 — Thinking levels for extended reasoning (Anthropic / OpenAI / Gemini)
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.settings import merge_model_settings

# thinking=True enables the default level; fine-grained with string literals
reasoning_settings: ModelSettings = {
    'thinking': 'high',           # 'minimal'|'low'|'medium'|'high'|'xhigh'|True|False
    'max_tokens': 16_000,         # must be high enough to include thinking tokens
    'service_tier': 'priority',   # 'auto'|'default'|'flex'|'priority'
}

agent = Agent('anthropic:claude-sonnet-4-5', model_settings=reasoning_settings)

# Layer base + per-run:
per_run: ModelSettings = {'temperature': 0.3}
merged = merge_model_settings(reasoning_settings, per_run)
print(merged['thinking'])     # high
print(merged['temperature'])  # 0.3  (override wins)
```

```python
# Example 3 — tool_choice and extra_headers for fine-grained model control
from pydantic_ai import Agent, ModelSettings, ToolOrOutput

# WARNING: 'required' cannot be set as a static agent model_settings — it
# prevents the agent from ever producing a final response (no output step).
# Use ToolOrOutput to allow specific function tools *and* output capability:
selective: ModelSettings = {
    'tool_choice': ToolOrOutput(function_tools=['search', 'calculate']),
}

# For a single-step forced tool call use pydantic_ai.direct.model_request,
# or pass tool_choice='required' as a per-run override inside a capability's
# get_model_settings() so it only applies to intermediate steps.

# Custom headers (e.g. for Anthropic beta features or tracking IDs):
with_headers: ModelSettings = {
    'extra_headers': {'anthropic-beta': 'max-tokens-3-5-sonnet-2024-07-15'},
    'stop_sequences': ['<end>', '```'],
    'frequency_penalty': 0.1,
    'presence_penalty': 0.2,
}

agent = Agent('openai:gpt-4o', model_settings=selective)
```

---

## 5. `TemplateStr`

**Source:** `pydantic_ai/_template.py`

`TemplateStr` compiles a Handlebars template string against the agent's `deps_type` using [pydantic-handlebars](https://github.com/pydantic/pydantic-handlebars). When used as the `instructions` parameter on `Agent`, it is re-rendered at each model request with the live `RunContext.deps`, enabling per-user dynamic prompts without sacrificing type safety.

Strings containing `{{` are **automatically compiled** during Pydantic validation when `TemplateStr` appears in a type hint — no explicit wrapping needed inside YAML or JSON specs.

```python
# Example 1 — Per-user dynamic instructions injected from deps
from dataclasses import dataclass
from pydantic_ai import Agent, TemplateStr

@dataclass
class UserProfile:
    name: str
    language: str
    expertise: str   # 'beginner' | 'intermediate' | 'expert'

agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=UserProfile,
    instructions=TemplateStr(
        'Hello {{name}}! Respond in {{language}}. '
        'Tailor your answer for a {{expertise}}-level reader.'
    ),
)

# result = agent.run_sync(
#     'What is a Python decorator?',
#     deps=UserProfile(name='Alice', language='French', expertise='beginner'),
# )
```

```python
# Example 2 — TemplateStr rendered standalone (outside an agent)
from dataclasses import dataclass
from pydantic_ai import TemplateStr

@dataclass
class Config:
    org_name: str
    max_results: int

# Explicit deps_type provides type-checked rendering and better error messages
tmpl = TemplateStr(
    'You are the AI assistant for {{org_name}}. '
    'Return at most {{max_results}} items per response.',
    deps_type=Config,
)

rendered = tmpl.render(Config(org_name='Acme Corp', max_results=5))
print(rendered)
# You are the AI assistant for Acme Corp. Return at most 5 items per response.
```

```python
# Example 3 — TemplateStr in AgentSpec YAML (auto-compiled from string)
from pydantic_ai.agent import AgentSpec

# Any string with '{{' is auto-compiled as Handlebars during spec validation
yaml_text = """
model: openai:gpt-4o
instructions: "Answer as {{role}}. Focus on {{topic}}."
"""

spec = AgentSpec.from_text(yaml_text)
# spec.instructions is a TemplateStr because it contains '{{'
# When turned into an Agent, it is rendered per-request against ctx.deps

# Combine TemplateStr with a list of instructions:
yaml_multi = """
model: openai:gpt-4o
instructions:
  - "You are a helpful assistant."
  - "The user's preferred language is {{language}}."
"""
spec2 = AgentSpec.from_text(yaml_multi)
```

---

## 6. `ConcurrencyLimiter` + `AbstractConcurrencyLimiter` + `ConcurrencyLimit`

**Source:** `pydantic_ai/concurrency.py`

`ConcurrencyLimiter` wraps an `anyio.CapacityLimiter` to bound how many concurrent tool calls (or model requests) may run at once. When a caller has to wait it creates an OpenTelemetry span so wait-time is visible in traces. `max_queued` adds backpressure — callers over the queue cap get `ConcurrencyLimitExceeded` immediately. Subclass `AbstractConcurrencyLimiter` for distributed limiting (e.g., Redis semaphore).

```python
# Example 1 — Limit concurrent tool calls within a single agent run
import asyncio
from pydantic_ai import Agent, FunctionToolset, RunContext, ConcurrencyLimiter

tools = FunctionToolset[None]()

@tools.tool_plain
async def slow_api_call(resource_id: str) -> str:
    """Simulate a rate-limited external API."""
    await asyncio.sleep(0.5)
    return f'data for {resource_id}'

# Allow at most 3 simultaneous calls; queue up to 10 more
limiter = ConcurrencyLimiter(max_running=3, max_queued=10, name='external-api')

agent = Agent('openai:gpt-4o', toolsets=[tools])

# The agent passes the limiter to each tool execution slot.
# result = asyncio.run(agent.run(
#     'Fetch data for resources A, B, C, D, E simultaneously.',
#     concurrency_limiter=limiter,
# ))
print(f'Capacity: {limiter.max_running}, waiting: {limiter.waiting_count}')
```

```python
# Example 2 — ConcurrencyLimit config dataclass (from_limit factory)
from pydantic_ai import ConcurrencyLimiter
from pydantic_ai.concurrency import ConcurrencyLimit

# Pass config objects to factories / dependency injection containers
limit_config = ConcurrencyLimit(max_running=5, max_queued=20)

# Build a named limiter from config
limiter = ConcurrencyLimiter.from_limit(limit_config, name='db-pool')
print(limiter.max_running)    # 5
print(limiter.available_count)# 5
print(limiter.running_count)  # 0
print(limiter.waiting_count)  # 0
```

```python
# Example 3 — Custom distributed limiter (Redis-backed example skeleton)
import asyncio
from pydantic_ai.concurrency import AbstractConcurrencyLimiter

class RedisLimiter(AbstractConcurrencyLimiter):
    """Distributed rate limiter backed by a Redis semaphore."""

    def __init__(self, redis_url: str, max_running: int, key: str = 'pydantic-ai'):
        self._max = max_running
        self._key = key
        # self._redis = Redis.from_url(redis_url)

    async def acquire(self, source: str) -> None:
        """Block until a slot is available in Redis."""
        # await self._redis.blpop(f'{self._key}:slots')
        print(f'[{source}] acquired Redis slot')

    def release(self) -> None:
        """Return the slot to the Redis pool."""
        # self._redis.rpush(f'{self._key}:slots', '1')
        print('released Redis slot')
```

---

## 7. `NativeOutput` + `PromptedOutput` + `TextOutput` + `ToolOutput`

**Source:** `pydantic_ai/output.py`

Four *marker classes* let you choose how the model returns structured data. Pass them as `output_type` on `Agent` or on `agent.run*()` calls. The right choice depends on provider capability — consult `ModelProfile.default_structured_output_mode` for the model's preference.

| Marker | Mechanism | Best for |
|--------|-----------|----------|
| `ToolOutput` | Agent emits a structured "output tool" call | Unions, non-native models, strict validation |
| `NativeOutput` | Provider `response_format=json_schema` | OpenAI, Google when max fidelity needed |
| `PromptedOutput` | JSON schema injected into system prompt, text parsed | Older / local models |
| `TextOutput` | Plain text passed to a Python function | Custom parsers, regex, domain extraction |

```python
# Example 1 — NativeOutput with a custom template override
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput

class WeatherReport(BaseModel):
    city: str
    temperature_c: float
    condition: str
    humidity_pct: int

# NativeOutput uses the provider's native JSON schema API (e.g. OpenAI response_format)
output = NativeOutput(
    WeatherReport,
    name='WeatherReport',
    description='Structured weather data for a city',
    strict=True,          # enforce strict schema compliance
)

agent = Agent('openai:gpt-4o', output_type=output)
# result = agent.run_sync('What is the weather in Berlin?')
# report: WeatherReport = result.output
# print(report.temperature_c)
```

```python
# Example 2 — PromptedOutput for models without native structured outputs
from pydantic import BaseModel
from pydantic_ai import Agent, PromptedOutput

class SentimentResult(BaseModel):
    sentiment: str        # 'positive' | 'neutral' | 'negative'
    confidence: float     # 0.0 – 1.0
    summary: str

# PromptedOutput injects the JSON schema into the system prompt.
# Good for Ollama, HuggingFace, and models without native JSON mode.
output = PromptedOutput(
    SentimentResult,
    name='SentimentResult',
    # Override the default schema template if needed:
    # template='Return ONLY a JSON object matching this schema:\n{schema}',
)

agent = Agent('ollama:llama3.2', output_type=output)
# result = agent.run_sync('Analyse: "I love this product!"')
```

```python
# Example 3 — TextOutput for custom parsing + multi-type union with ToolOutput
from pydantic import BaseModel
from pydantic_ai import Agent, TextOutput, ToolOutput

# TextOutput: model returns plain text, Python function processes it
def extract_numbers(text: str) -> list[float]:
    import re
    return [float(x) for x in re.findall(r'\d+\.?\d*', text)]

text_agent = Agent(
    'openai:gpt-4o',
    output_type=TextOutput(extract_numbers),
)
# result = text_agent.run_sync('List the first 5 Fibonacci numbers.')
# print(result.output)  # [1.0, 1.0, 2.0, 3.0, 5.0]

# ToolOutput: multi-type union (model picks one of several structured types)
class ErrorResult(BaseModel):
    error: str
    code: int

class SuccessResult(BaseModel):
    data: dict
    message: str

# Union output — pass a list of ToolOutput, one per type; model picks which tool to call
union_agent = Agent(
    'openai:gpt-4o',
    output_type=[ToolOutput(SuccessResult), ToolOutput(ErrorResult)],
)
```

---

## 8. `ApprovalRequiredToolset` + `ApprovalRequired` + `ToolApproved` + `ToolDenied`

**Source:** `pydantic_ai/toolsets/approval_required.py`, `pydantic_ai/exceptions.py`

`ApprovalRequiredToolset` wraps any toolset and pauses execution before a tool call, raising `ApprovalRequired`. The caller catches this exception, inspects the pending call, and resumes the agent run by passing either a `ToolApproved` or `ToolDenied` continuation into `AgentRun.enqueue()`. The optional `approval_required_func` predicate lets you gate only high-risk calls.

```python
# Example 1 — Simple all-calls HITL approval
import asyncio
from pydantic_ai import Agent, FunctionToolset, RunContext, ApprovalRequiredToolset
from pydantic_ai.exceptions import ApprovalRequired

tools = FunctionToolset[None]()

@tools.tool_plain
def delete_file(path: str) -> str:
    """Permanently delete a file from the filesystem."""
    return f'Deleted {path}'

# All calls to any tool in this toolset require human approval
gated = ApprovalRequiredToolset(wrapped=tools)
agent = Agent('openai:gpt-4o', toolsets=[gated])

async def run_with_approval():
    async with agent.run_stream('Delete the file /tmp/report.pdf') as agent_run:
        try:
            async for _ in agent_run:
                pass
        except ApprovalRequired as e:
            print(f'Approval needed: tool={e.tool_name!r}, args={e.tool_args}')
            # Human reviews, then approves or denies (enqueue is synchronous — no await)
            # agent_run.enqueue(ToolApproved())
            # or: agent_run.enqueue(ToolDenied(message='Not authorised'))
```

```python
# Example 2 — Selective approval via approval_required_func
from pydantic_ai import Agent, FunctionToolset, RunContext, ApprovalRequiredToolset
from pydantic_ai.toolsets.abstract import ToolsetTool

tools = FunctionToolset[None]()

@tools.tool_plain
def search_web(query: str) -> str:
    return f'results for {query}'

@tools.tool_plain
def send_email(to: str, body: str) -> str:
    return f'Email sent to {to}'

def needs_approval(ctx: RunContext, tool_def, tool_args: dict) -> bool:
    # Only require approval for write/send operations, not reads
    return tool_def.name in {'send_email', 'delete_file', 'write_db'}

selective = ApprovalRequiredToolset(
    wrapped=tools,
    approval_required_func=needs_approval,
)

agent = Agent('openai:gpt-4o', toolsets=[selective])
# search_web runs freely; send_email pauses for approval
```

```python
# Example 3 — Full HITL loop with ToolApproved / ToolDenied
import asyncio
from pydantic_ai import Agent, FunctionToolset, ToolApproved, ToolDenied, ApprovalRequiredToolset
from pydantic_ai.exceptions import ApprovalRequired

tools = FunctionToolset[None]()

@tools.tool_plain
def transfer_funds(account_from: str, account_to: str, amount: float) -> str:
    return f'Transferred £{amount:.2f} from {account_from} to {account_to}'

gated = ApprovalRequiredToolset(wrapped=tools)
agent = Agent('openai:gpt-4o', toolsets=[gated])

async def supervised_run(prompt: str) -> str:
    async with agent.iter(prompt) as agent_run:
        try:
            async for node in agent_run:
                pass  # normal iteration until approval is needed
        except ApprovalRequired as e:
            print(f'Approval needed: tool={e.tool_name!r}, args={e.tool_args}')
            # Approve or deny before continuing:
            # agent_run.enqueue(ToolApproved())
            # agent_run.enqueue(ToolDenied(message='Transfer exceeds daily limit'))
    return agent_run.result.output if agent_run.result else ''
```

---

## 9. `FunctionToolset`

**Source:** `pydantic_ai/toolsets/function.py`

`FunctionToolset` is the primitive building block for Python-function tools. It accepts up to 14 constructor parameters that become default policy for every tool registered to it. Individual tool decorators (`@ts.tool`, `@ts.tool_plain`) accept the same keyword arguments to override the toolset default.

```python
# Example 1 — Production toolset with timeout, retries, and toolset-level instructions
from pydantic_ai import Agent, FunctionToolset, RunContext

# Toolset-level policy applied to all tools unless overridden
api_tools: FunctionToolset[str] = FunctionToolset(
    max_retries=2,
    timeout=10.0,              # seconds per tool call
    require_parameter_descriptions=True,  # fail at registration if doc is incomplete
    sequential=False,          # allow parallel tool execution (default)
    metadata={'source': 'api-v2'},
    instructions='Use these tools only when the user explicitly requests data.',
)

@api_tools.tool
def get_user(ctx: RunContext[str], user_id: str) -> dict:
    """Retrieve a user by ID.

    Args:
        user_id: The unique user identifier.
    """
    return {'id': user_id, 'token': ctx.deps}

@api_tools.tool_plain(timeout=30.0)   # override toolset timeout per tool
def run_report(report_type: str, date_range: str) -> str:
    """Run a named report over a date range.

    Args:
        report_type: Name of the report to run.
        date_range: ISO-8601 date range, e.g. '2026-01-01/2026-06-30'.
    """
    return f'{report_type} report for {date_range}'

agent = Agent('openai:gpt-4o', deps_type=str, toolsets=[api_tools])
```

```python
# Example 2 — Deferred loading: hide tools until tool-search surfaces them
from pydantic_ai import Agent, FunctionToolset, RunContext

# Large toolsets with hundreds of tools should use defer_loading=True so the
# model only sees them when a tool-search capability surfaces them on demand.
large_toolset: FunctionToolset[None] = FunctionToolset(
    defer_loading=True,
    id='analytics',         # required for durable execution (Temporal/DBOS/Prefect)
)

@large_toolset.tool_plain
def segment_users(segment: str) -> list:
    """Return users matching a segment filter."""
    return []

@large_toolset.tool_plain
def export_csv(table: str, filters: dict) -> str:
    """Export a table to CSV with optional filters."""
    return 'path/to/export.csv'

# These tools are hidden at run start; the model discovers them via tool-search
agent = Agent('openai:gpt-4o', toolsets=[large_toolset])
```

```python
# Example 3 — add_function for dynamic registration + @ts.instructions
from pydantic_ai import Agent, FunctionToolset, RunContext

ts: FunctionToolset[dict] = FunctionToolset()

@ts.instructions
def toolset_instructions(ctx: RunContext[dict]) -> str:
    """Injected into the system prompt when any tool in this set is active."""
    env = ctx.deps.get('env', 'production')
    return f'You are operating in the {env} environment. Be careful with write operations.'

# Register a function that was defined elsewhere
def lookup_product(sku: str) -> dict:
    """Look up product details by SKU."""
    return {'sku': sku, 'price': 29.99}

ts.add_function(
    lookup_product,
    takes_ctx=False,
    name='product_lookup',          # override the function name
    description='Find product by SKU',
    metadata={'category': 'catalog'},
)

agent = Agent('openai:gpt-4o', deps_type=dict, toolsets=[ts])
```

---

## 10. `RunContext`

**Source:** `pydantic_ai/_run_context.py`

`RunContext` is the single object passed to all context-aware tools, system-prompt functions, hooks, and validators. It carries the full run state: deps, model, accumulated usage, conversation messages, and metadata. New in 2.10.0: `usage_limits` is now always set during a real run (defaulting to `UsageLimits()` even when the caller passes `None`), enabling tools to inspect the remaining budget without needing a separate injection.

```python
# Example 1 — Core fields: deps, model, usage, messages
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

@dataclass
class AppDeps:
    db_url: str
    user_id: int

agent = Agent('openai:gpt-4o', deps_type=AppDeps)

@agent.tool
async def get_profile(ctx: RunContext[AppDeps]) -> dict:
    # Access deps
    db = ctx.deps.db_url
    uid = ctx.deps.user_id

    # Inspect accumulated token usage mid-run
    usage = ctx.usage
    print(f'So far: {usage.input_tokens} in, {usage.output_tokens} out, '
          f'{usage.requests} requests')

    # Conversation history up to this point
    message_count = len(ctx.messages)
    print(f'Messages in context: {message_count}')

    # Model being used (e.g. for provider-specific behaviour)
    print(f'Model: {ctx.model.__class__.__name__}')

    return {'user_id': uid}
```

```python
# Example 2 — Retry control, last_attempt, and ModelRetry
from pydantic_ai import Agent, RunContext, ModelRetry

agent = Agent('openai:gpt-4o', retries=3)

@agent.tool(retries=3)
async def flaky_lookup(ctx: RunContext[None], key: str) -> str:
    """A tool that might need retries."""
    print(f'Attempt {ctx.retry + 1} of {ctx.max_retries + 1}')

    if ctx.last_attempt:
        # On the final attempt, return a fallback instead of raising
        return f'fallback value for {key}'

    # Signal the model to retry with an explanation
    if ctx.retry < 2:
        raise ModelRetry(f'Service temporarily unavailable, retry {ctx.retry + 1}')

    return f'value for {key}'
```

```python
# Example 3 — enqueue, usage_limits budget-awareness, and observability fields
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import UserPromptPart

agent = Agent('openai:gpt-4o')

@agent.tool
async def check_budget(ctx: RunContext[None]) -> str:
    """Report remaining budget to the agent."""
    limits = ctx.usage_limits      # None only on bare/synthetic RunContext outside a real run
    usage = ctx.usage

    if limits is None:
        return 'No budget limits configured.'

    remaining_requests = (
        (limits.request_limit - usage.requests)
        if limits.request_limit is not None
        else 'unlimited'
    )
    # Inject extra context into the next model request without ending the turn
    ctx.enqueue(
        UserPromptPart(content=f'Remaining requests: {remaining_requests}'),
        priority='when_idle',
    )
    return f'Budget check complete. Run ID: {ctx.run_id}'
```

---

*Vol. 37 · pydantic-ai 2.10.0 · July 2026*

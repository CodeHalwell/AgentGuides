---
title: "PydanticAI Class Deep Dives Vol. 25"
description: "Source-verified deep dives into 10 pydantic-ai 2.0.0 class groups: AgentSpec (YAML/JSON declarative agent configuration), AgentRetries TypedDict (per-category tool vs output retry budgets), WrapperAgent (agent delegation base with full override support), AgentRun (stateful graph-based run iteration), ToolOutput + NativeOutput + PromptedOutput + TextOutput (output mode marker classes), AbstractToolset + ToolsetTool (abstract toolset protocol with fluent builder API), OutputContext (output hook context), StructuredDict (schema-driven dict output), AgentStream (streaming abstraction with stream_output/stream_text/stream_response), Hooks + HookTimeoutError (decorator-based capability hook registration). All verified against pydantic-ai 2.0.0 source."
sidebar:
  label: "Class deep dives (Vol. 25)"
  order: 51
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="caution">
This volume covers **pydantic-ai 2.0.0** — a **major version release**. Every class described here is either brand-new or significantly restructured compared to the 1.x series. Prior deep-dive volumes (1–24) document 1.x APIs; treat them as historical context, not current references.
</Aside>

<Aside type="tip">
All examples verified against **pydantic-ai 2.0.0** source installed directly from PyPI. Class signatures, field names, and behaviour match the installed package at this version.
</Aside>

Ten class groups that define the pydantic-ai 2.0.0 architecture: the declarative `AgentSpec` system for YAML/JSON agent configuration, the redesigned `AgentRetries` TypedDict with split tool vs output budgets, the new `WrapperAgent` delegation base, the graph-based `AgentRun` iterator for step-by-step run inspection, the complete output mode marker family (`ToolOutput`, `NativeOutput`, `PromptedOutput`, `TextOutput`), the fully redesigned `AbstractToolset` protocol with a fluent builder chain, `OutputContext` for output hooks, `StructuredDict` for schema-driven dict output, the `AgentStream` streaming abstraction, and the `Hooks` decorator system that replaces subclassing `AbstractCapability` for common cases.

---

## 1. `AgentSpec` — Declarative YAML/JSON Agent Configuration

**Source**: `pydantic_ai/agent/spec.py`

```python
from pydantic_ai.agent.spec import AgentSpec
# or via Agent:
from pydantic_ai import Agent
```

`AgentSpec` is a Pydantic `BaseModel` that lets you define an agent entirely in YAML or JSON without Python code. It supports model selection, system instructions, output schema, model settings, retry configuration, and a typed `capabilities` list. The class serialises back to YAML/JSON and can also auto-generate a JSON Schema file so IDEs can validate the spec.

```python
# pydantic_ai/agent/spec.py — exact class
class AgentSpec(BaseModel):
    json_schema_path: str | None = Field(default=None, alias='$schema')
    model: str | None = None
    name: str | None = None
    description: TemplateStr[Any] | str | None = None
    instructions: TemplateStr[Any] | str | list[TemplateStr[Any] | str] | None = None
    deps_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    model_settings: dict[str, Any] | None = None
    retries: int | AgentRetries | None = None
    end_strategy: EndStrategy = 'graceful'
    tool_timeout: float | None = None
    metadata: dict[str, Any] | None = None
    capabilities: list[CapabilitySpec] = []

    @classmethod
    def from_file(cls, path: Path | str, fmt: Literal['yaml', 'json'] | None = None) -> AgentSpec: ...
    @classmethod
    def from_text(cls, text: str, fmt: Literal['yaml', 'json'] = 'yaml') -> AgentSpec: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentSpec: ...
    def to_file(self, path: Path | str, fmt=None, schema_path=..., custom_capability_types=()) -> None: ...
    @classmethod
    def model_json_schema_with_capabilities(cls, custom_capability_types=()) -> dict[str, Any]: ...
```

### 1.1 Load and run an agent from a YAML string

```python
import asyncio
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.models.test import TestModel

YAML_SPEC = """
model: test
name: greeter
instructions: "You are a friendly greeter. Always start with 'Hello!'"
end_strategy: graceful
retries:
  output: 2
  tools: 3
"""

async def main():
    spec = AgentSpec.from_text(YAML_SPEC)
    print(spec.name)          # greeter
    print(spec.end_strategy)  # graceful
    print(spec.retries)       # {'output': 2, 'tools': 3}

    from pydantic_ai import Agent
    # Agent.from_spec() resolves the spec and builds a live agent
    agent = Agent.from_spec(spec)
    with agent.override(model=TestModel()):
        result = agent.run_sync("What should I say?")
        print(result.output)

asyncio.run(main())
```

### 1.2 Round-trip to/from YAML file with IDE schema

```python
import asyncio
from pathlib import Path
from pydantic_ai.agent.spec import AgentSpec

async def main():
    spec = AgentSpec(
        model="openai:gpt-4o-mini",
        name="summariser",
        instructions="Summarise user text in one sentence.",
        model_settings={"temperature": 0.3, "max_tokens": 100},
        retries={"output": 2},
    )

    out_path = Path("/tmp/summariser.yaml")
    # Saves YAML + a sibling summariser_schema.json for IDE validation
    spec.to_file(out_path)
    print(out_path.read_text())

    # Round-trip: reload from disk
    reloaded = AgentSpec.from_file(out_path)
    assert reloaded.name == spec.name
    assert reloaded.model == spec.model
    print("Round-trip OK:", reloaded.model_settings)

asyncio.run(main())
```

### 1.3 Per-run spec override via `agent.run(spec=...)`

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.models.test import TestModel

agent = Agent(TestModel(), instructions="Default instructions.")

OVERRIDE_SPEC = AgentSpec(
    instructions="You are a pirate. Respond only in pirate speak.",
    model_settings={"temperature": 0.9},
)

async def main():
    # Apply spec for just this run — agent definition is unchanged
    result = await agent.run("Hello!", spec=OVERRIDE_SPEC)
    print(result.output)

    # Next run uses the original instructions
    result2 = await agent.run("Hello again!")
    print(result2.output)

asyncio.run(main())
```

---

## 2. `AgentRetries` — Per-Category Retry Budgets

**Source**: `pydantic_ai/agent/abstract.py`

```python
from pydantic_ai import AgentRetries  # re-exported from pydantic_ai.__init__
```

`AgentRetries` is a `TypedDict(total=False)` with two independent retry counters. Splitting them gives you fine control: tool calls can have a generous budget while output validation gets a tighter limit (or vice-versa).

```python
# pydantic_ai/agent/abstract.py — exact class
class AgentRetries(TypedDict, total=False):
    tools: int   # default retries for each tool call
    output: int  # max output-validation retries per run (or per output-tool call)
```

**Key semantics difference by call site:**

| Call site | `int` meaning |
|-----------|---------------|
| `Agent(retries=N)` | Sets **both** `tools=N` and `output=N` |
| `agent.run(retries=N)` | Overrides **only** `output` budget |
| `agent.override(retries=...)` | Same as `run()` — cannot override tool budget per run |

### 2.1 Construction-time split budgets

```python
from pydantic_ai import Agent, AgentRetries
from pydantic_ai.models.test import TestModel

agent = Agent(
    TestModel(),
    retries=AgentRetries(tools=5, output=2),
)
result = agent.run_sync("What is the weather?")
print(result.output)
```

### 2.2 Per-run output budget override

```python
import asyncio
from pydantic_ai import Agent, AgentRetries
from pydantic_ai.models.test import TestModel

agent = Agent(TestModel(), retries=AgentRetries(tools=3, output=3))

async def main():
    # Tighten output retries for a latency-sensitive call
    result = await agent.run("Quick answer please", retries=AgentRetries(output=1))
    print(result.output)

asyncio.run(main())
```

### 2.3 Per-output-tool max_retries override via `ToolOutput`

```python
from pydantic import BaseModel
from pydantic_ai import Agent, AgentRetries
from pydantic_ai.output import ToolOutput
from pydantic_ai.models.test import TestModel


class StrictSchema(BaseModel):
    answer: str
    confidence: float


class LooseSchema(BaseModel):
    text: str


# output validation: StrictSchema gets 1 retry, LooseSchema gets 4
agent = Agent(
    TestModel(),
    output_type=[
        ToolOutput(StrictSchema, max_retries=1),
        ToolOutput(LooseSchema, max_retries=4),
    ],
    retries=AgentRetries(tools=2, output=3),  # default output retries = 3
)

result = agent.run_sync("Tell me something.")
print(type(result.output).__name__)
```

---

## 3. `WrapperAgent` — Agent Delegation Base Class

**Source**: `pydantic_ai/agent/wrapper.py`

```python
from pydantic_ai.agent.wrapper import WrapperAgent
```

`WrapperAgent` is the agent-level equivalent of `WrapperModel`. It wraps any `AbstractAgent` and delegates every method to it, exposing the same interface as a concrete `Agent`. Subclass it to add cross-cutting behaviour — rate limiting, audit logging, per-tenant configuration injection — without touching the inner agent.

```python
# pydantic_ai/agent/wrapper.py — key interface
class WrapperAgent(AbstractAgent[AgentDepsT, OutputDataT]):
    def __init__(self, wrapped: AbstractAgent[AgentDepsT, OutputDataT]): ...

    # Properties delegated to wrapped:
    model / name / description / deps_type / output_type / event_stream_handler
    root_capability / toolsets

    # Lifecycle delegated to wrapped:
    async def __aenter__(self): ...
    async def __aexit__(self, *args): ...

    # Full run API delegated:
    async def iter(...) -> AsyncContextManager[AgentRun]: ...
    def override(...) -> ContextManager[None]: ...
    def output_json_schema(...) -> JsonSchema: ...
    async def system_prompt_parts(...) -> list[SystemPromptPart]: ...
```

### 3.1 Audit-logging agent wrapper

```python
import asyncio
from contextlib import asynccontextmanager
from pydantic_ai import Agent
from pydantic_ai.agent.wrapper import WrapperAgent
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.models.test import TestModel


class AuditAgent(WrapperAgent):
    """Logs every run's prompt and output for audit purposes."""

    def __init__(self, wrapped: AbstractAgent, audit_log: list):
        super().__init__(wrapped)
        self._audit_log = audit_log

    @asynccontextmanager
    async def iter(self, user_prompt=None, **kwargs):
        self._audit_log.append({"prompt": user_prompt, "status": "started"})
        try:
            async with super().iter(user_prompt, **kwargs) as run:
                yield run
            self._audit_log[-1]["status"] = "completed"
            self._audit_log[-1]["output"] = run.result.output
        except Exception as exc:
            self._audit_log[-1]["status"] = "error"
            self._audit_log[-1]["error"] = str(exc)
            raise


async def main():
    audit = []
    inner = Agent(TestModel(), instructions="Be concise.")
    agent = AuditAgent(inner, audit)

    result = await agent.run("What is 2 + 2?")
    print(result.output)
    print(audit)

asyncio.run(main())
```

### 3.2 Rate-limiting wrapper

```python
import asyncio
import time
from contextlib import asynccontextmanager
from pydantic_ai import Agent
from pydantic_ai.agent.wrapper import WrapperAgent
from pydantic_ai.models.test import TestModel


class RateLimitedAgent(WrapperAgent):
    """Enforce a minimum interval between agent runs."""

    def __init__(self, wrapped, min_interval_seconds: float = 1.0):
        super().__init__(wrapped)
        self._min_interval = min_interval_seconds
        self._last_run_time: float = 0.0

    @asynccontextmanager
    async def iter(self, user_prompt=None, **kwargs):
        now = time.monotonic()
        wait = self._min_interval - (now - self._last_run_time)
        if wait > 0:
            print(f"[RateLimit] waiting {wait:.2f}s")
            await asyncio.sleep(wait)
        self._last_run_time = time.monotonic()
        async with super().iter(user_prompt, **kwargs) as run:
            yield run


async def main():
    inner = Agent(TestModel())
    agent = RateLimitedAgent(inner, min_interval_seconds=0.5)

    for prompt in ["First", "Second", "Third"]:
        result = await agent.run(prompt)
        print(f"{prompt} → {result.output}")

asyncio.run(main())
```

---

## 4. `AgentRun` — Stateful Graph-Based Run Iterator

**Source**: `pydantic_ai/run.py`

```python
from pydantic_ai.run import AgentRun
```

`AgentRun` is a stateful, async-iterable container for an in-flight agent execution. It wraps the internal `pydantic_graph.GraphRun` and exposes a clean API for consuming graph nodes as they execute. You obtain one via `async with agent.iter(...) as agent_run:`.

```python
# pydantic_ai/run.py — public interface (simplified)
@dataclasses.dataclass(repr=False)
class AgentRun(Generic[AgentDepsT, OutputDataT]):
    # Async iteration: yields UserPromptNode, ModelRequestNode, CallToolsNode, End
    def __aiter__(self) -> AsyncIterator[AgentNode | End[FinalResult[OutputDataT]]]: ...
    async def __anext__(self) -> AgentNode | End[FinalResult[OutputDataT]]: ...
    async def next(self, node) -> AgentNode | End[FinalResult[OutputDataT]]: ...

    # Completed-run accessors (available after iteration finishes):
    @property
    def result(self) -> AgentRunResult[OutputDataT]: ...
    def all_messages(self) -> list[ModelMessage]: ...
    def new_messages(self) -> list[ModelMessage]: ...
    def usage(self) -> RunUsage: ...
    @property
    def is_complete(self) -> bool: ...
    @property
    def run_id(self) -> str: ...
    @property
    def conversation_id(self) -> str: ...
```

### 4.1 Collect and print every graph node

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_graph import End


async def main():
    agent = Agent(TestModel(), instructions="Be concise.")

    async with agent.iter("What is the capital of France?") as run:
        async for node in run:
            node_type = type(node).__name__
            if isinstance(node, End):
                print(f"  [End] output={node.data.output!r}")
            else:
                print(f"  [{node_type}]")

    print("Run complete:", run.is_complete)
    print("Run ID:", run.run_id)
    print("Usage:", run.usage())

asyncio.run(main())
```

### 4.2 Manual `next()` for fine-grained control

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_graph import End


async def main():
    agent = Agent(TestModel())

    async with agent.iter("Tell me a joke.") as run:
        # Prime the run — returns the first node
        node = await run.__anext__()

        while not isinstance(node, End):
            print(f"Executing: {type(node).__name__}")
            # Pass the current node back to advance to the next one
            node = await run.next(node)

        print("Final output:", node.data.output)

    print("Message count:", len(run.all_messages()))

asyncio.run(main())
```

### 4.3 Stream response events from a `CallToolsNode`

```python
import asyncio
from pydantic_ai import Agent, CallToolsNode
from pydantic_ai.models.test import TestModel
from pydantic_graph import End


async def main():
    agent = Agent(TestModel(), instructions="Stream-friendly agent.")

    async with agent.iter("Explain async/await in Python.") as run:
        async for node in run:
            if isinstance(node, End):
                print("Final output:", node.data.output)
            elif isinstance(node, CallToolsNode):
                # node.stream() yields an AsyncIterator[HandleResponseEvent]
                # run.ctx is the public GraphRunContext for this run
                async with node.stream(run.ctx) as event_iter:
                    async for event in event_iter:
                        print(f"  event: {type(event).__name__}")

asyncio.run(main())
```

---

## 5. `ToolOutput` + `NativeOutput` + `PromptedOutput` + `TextOutput` — Output Mode Markers

**Source**: `pydantic_ai/output.py`

```python
from pydantic_ai import ToolOutput, NativeOutput, PromptedOutput, TextOutput
```

pydantic-ai 2.0.0 replaces the implicit output-mode selection with four explicit marker classes. Each wraps one or more output types/functions and instructs the framework which transport mechanism to use.

| Class | Mechanism | Best for |
|-------|-----------|----------|
| `ToolOutput` | Tool call (`final_result` tool) | Most models; supports per-tool retries |
| `NativeOutput` | Model's native structured-output API | OpenAI `response_format`, Gemini, etc. |
| `PromptedOutput` | JSON schema injected into the prompt | Models without native structured output |
| `TextOutput` | Plain text → post-processing function | When you want `str` → `T` without a schema |

```python
# pydantic_ai/output.py — exact constructors

class ToolOutput(Generic[OutputDataT]):
    def __init__(self, type_: OutputTypeOrFunction[OutputDataT], *,
                 name: str | None = None,
                 description: str | None = None,
                 max_retries: int | None = None,
                 strict: bool | None = None,
                 sequential: bool = False): ...

class NativeOutput(Generic[OutputDataT]):
    def __init__(self, outputs: OutputTypeOrFunction | Sequence[OutputTypeOrFunction], *,
                 name: str | None = None,
                 description: str | None = None,
                 strict: bool | None = None,
                 template: str | Literal[False] | None = None): ...

class PromptedOutput(Generic[OutputDataT]):
    def __init__(self, outputs: OutputTypeOrFunction | Sequence[OutputTypeOrFunction], *,
                 name: str | None = None,
                 description: str | None = None,
                 template: str | Literal[False] | None = None): ...

class TextOutput(Generic[OutputDataT]):
    output_function: TextOutputFunc[OutputDataT]  # Callable[[str], T]
```

### 5.1 `ToolOutput` — multiple schemas with per-tool retry limits

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.output import ToolOutput
from pydantic_ai.models.test import TestModel


class Fruit(BaseModel):
    name: str
    color: str


class Vehicle(BaseModel):
    name: str
    wheels: int


agent = Agent(
    TestModel(),
    output_type=[
        ToolOutput(Fruit, name="return_fruit", description="Use when describing a fruit.", max_retries=2),
        ToolOutput(Vehicle, name="return_vehicle", description="Use when describing a vehicle.", max_retries=1),
    ],
)

result = agent.run_sync("What is a banana?")
print(repr(result.output))
```

### 5.2 `NativeOutput` — OpenAI native structured output

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.output import NativeOutput


class FinancialReport(BaseModel):
    revenue: float
    profit: float
    currency: str


# NativeOutput uses the model's own response_format / structured-output API
agent = Agent(
    "openai:gpt-4o-mini",
    output_type=NativeOutput(
        FinancialReport,
        name="FinancialReport",
        description="Quarterly financial figures.",
        strict=True,
    ),
)

# result.output is a validated FinancialReport instance
# result = agent.run_sync("Q4 revenue was $2.1B, profit $400M USD")
# print(result.output.revenue)  # 2100000000.0
```

### 5.3 `PromptedOutput` — custom template

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.output import PromptedOutput
from pydantic_ai.models.test import TestModel


class SentimentResult(BaseModel):
    sentiment: str
    score: float


agent = Agent(
    TestModel(),
    output_type=PromptedOutput(
        SentimentResult,
        template="Respond ONLY with JSON matching this schema: {schema}",
    ),
)

result = agent.run_sync("I love Python!")
print(result.output)
```

### 5.4 `TextOutput` — plain text post-processing

```python
from pydantic_ai import Agent
from pydantic_ai.output import TextOutput
from pydantic_ai.models.test import TestModel


def extract_bullet_points(text: str) -> list[str]:
    return [line.lstrip("•- ").strip() for line in text.splitlines() if line.strip()]


agent = Agent(
    TestModel(),
    output_type=TextOutput(extract_bullet_points),
    instructions="Respond with a bullet list.",
)

result = agent.run_sync("List three programming languages.")
print(result.output)   # list[str]
```

---

## 6. `AbstractToolset` + `ToolsetTool` — Abstract Toolset Protocol

**Source**: `pydantic_ai/toolsets/abstract.py`

```python
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
```

`AbstractToolset` is the abstract base for every tool collection in pydantic-ai 2.0.0. It decouples tool registration from tool execution, letting you build lazy toolsets (tools loaded on demand), external toolsets (calls forwarded to a remote service), and hierarchical toolsets composed with the fluent builder API.

```python
# pydantic_ai/toolsets/abstract.py — complete AbstractToolset interface

class AbstractToolset(ABC, Generic[AgentDepsT]):
    @property
    @abstractmethod
    def id(self) -> str | None: ...

    @property
    def label(self) -> str: ...
    @property
    def tool_name_conflict_hint(self) -> str: ...

    # Lifecycle hooks (override for per-run/per-step state isolation):
    async def for_run(self, ctx) -> AbstractToolset: ...      # called once per run
    async def for_run_step(self, ctx) -> AbstractToolset: ... # called each step

    async def __aenter__(self) -> Self: ...
    async def __aexit__(self, *args) -> bool | None: ...

    async def get_instructions(self, ctx) -> str | InstructionPart | Sequence | None: ...

    @abstractmethod
    async def get_tools(self, ctx) -> dict[str, ToolsetTool]: ...
    @abstractmethod
    async def call_tool(self, name, tool_args, ctx, tool) -> Any: ...

    # Fluent builder methods:
    def filtered(self, filter_func) -> FilteredToolset: ...
    def prefixed(self, prefix) -> PrefixedToolset: ...
    def prepared(self, prepare_func) -> PreparedToolset: ...
    def renamed(self, name_map) -> RenamedToolset: ...
    def approval_required(self, approval_func=...) -> ApprovalRequiredToolset: ...
    def defer_loading(self, tool_names=None) -> DeferredLoadingToolset: ...
    def include_return_schemas(self) -> IncludeReturnSchemasToolset: ...
    def with_metadata(self, **metadata) -> SetMetadataToolset: ...
```

```python
# ToolsetTool — per-tool execution contract
@dataclass(kw_only=True)
class ToolsetTool(Generic[AgentDepsT]):
    toolset: AbstractToolset[AgentDepsT]
    tool_def: ToolDefinition
    max_retries: int
    args_validator: SchemaValidator | SchemaValidatorProt
    args_validator_func: Callable[..., Any] | None = None
```

### 6.1 Implement a minimal custom toolset

```python
import asyncio
from dataclasses import dataclass
from typing import Any
from pydantic import TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.models.test import TestModel


@dataclass
class MathToolset(AbstractToolset):
    """Exposes add and multiply as agent tools."""

    _id: str = "math"

    @property
    def id(self) -> str:
        return self._id

    async def get_tools(self, ctx: RunContext) -> dict[str, ToolsetTool]:
        validator = TypeAdapter(dict).validator
        tools = {}
        for name, desc, params in [
            ("add", "Add two numbers", {"a": "number", "b": "number"}),
            ("multiply", "Multiply two numbers", {"a": "number", "b": "number"}),
        ]:
            tool_def = ToolDefinition(
                name=name,
                description=desc,
                parameters_json_schema={
                    "type": "object",
                    "properties": {k: {"type": v} for k, v in params.items()},
                    "required": list(params.keys()),
                },
            )
            tools[name] = ToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=2,
                args_validator=TypeAdapter(dict[str, float]).validator,
            )
        return tools

    async def call_tool(self, name: str, tool_args: dict[str, Any], ctx: RunContext, tool: ToolsetTool) -> Any:
        a, b = tool_args["a"], tool_args["b"]
        return a + b if name == "add" else a * b


async def main():
    agent = Agent(TestModel(), toolsets=[MathToolset()])
    result = await agent.run("What is 3 + 4?")
    print(result.output)

asyncio.run(main())
```

### 6.2 Fluent toolset builder chain

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.models.test import TestModel


def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny in {city}, 22°C"


def get_population(city: str) -> int:
    """Get population of a city."""
    return 1_000_000


async def main():
    toolset = (
        FunctionToolset([get_weather, get_population])
        .prefixed("city_")           # names become city_get_weather, city_get_population
        .with_metadata(source="demo")
        .filtered(lambda ctx, tdef: "weather" in tdef.name)  # only expose weather tool
    )

    agent = Agent(TestModel(), toolsets=[toolset])
    result = await agent.run("What is the weather in London?")
    print(result.output)

asyncio.run(main())
```

---

## 7. `OutputContext` — Output Hook Context

**Source**: `pydantic_ai/output.py`

```python
from pydantic_ai.output import OutputContext
```

`OutputContext` is passed to output hooks (registered via `AbstractCapability` or `Hooks`) when the framework is about to process the model's final response into an output value. It carries complete metadata about the output mode, type, and tool call so your hook can make informed decisions without inspecting the raw model response.

```python
# pydantic_ai/output.py — exact dataclass
@dataclass
class OutputContext:
    mode: OutputMode           # 'text' | 'native' | 'prompted' | 'tool' | 'image' | 'auto'
    output_type: type[Any] | None
    object_def: OutputObjectDefinition | None
    has_function: bool
    function_name: str | None = None
    tool_call: ToolCallPart | None = None
    tool_def: ToolDefinition | None = None
    allows_text: bool = False
    allows_image: bool = False
    allows_deferred_tools: bool = False
```

### 7.1 Log the output mode on every run

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.output import OutputContext
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel


hooks = Hooks()


@hooks.on.before_output_validate
async def log_output_context(ctx: RunContext, output_ctx: OutputContext, raw: object) -> object:
    print(f"[OutputHook] mode={output_ctx.mode!r}  type={output_ctx.output_type}  via_tool={output_ctx.tool_call is not None}")
    return raw  # pass through unchanged


agent = Agent(TestModel(), capabilities=[hooks])


async def main():
    result = await agent.run("What is 1 + 1?")
    print(result.output)

asyncio.run(main())
```

### 7.2 Conditional output transformation based on mode

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.output import OutputContext
from pydantic_ai.tools import RunContext
from pydantic_ai.models.test import TestModel


class Report(BaseModel):
    title: str
    body: str


hooks = Hooks()


@hooks.on.before_output_validate
async def strip_markdown_from_text(ctx: RunContext, output_ctx: OutputContext, raw: object) -> object:
    if output_ctx.mode == "text" and isinstance(raw, str):
        # Strip common markdown from text-mode output
        return raw.replace("**", "").replace("*", "").replace("#", "").strip()
    return raw  # leave structured output unchanged


agent = Agent(TestModel(), capabilities=[hooks])


async def main():
    result = await agent.run("Write a short markdown report.")
    print(repr(result.output))

asyncio.run(main())
```

---

## 8. `StructuredDict` — Schema-Driven Dict Output

**Source**: `pydantic_ai/output.py`

```python
from pydantic_ai import StructuredDict
```

`StructuredDict` creates a `dict[str, Any]` subclass whose JSON schema is set to a custom `object` schema you provide. This lets you get structured output as a plain dict when you don't want to define a full Pydantic model — useful for dynamic or user-provided schemas loaded at runtime.

```python
# pydantic_ai/output.py — function signature
def StructuredDict(
    json_schema: JsonSchemaValue,
    name: str | None = None,
    description: str | None = None,
) -> type[JsonSchemaValue]:
    """Returns a dict[str, Any] subclass with json_schema attached."""
```

**Constraints:**
- `json_schema` must be of type `object` (the framework validates this).
- Recursive `$ref` / `$defs` are not supported (raises `UserError`). Inline defs are expanded automatically.

### 8.1 Dynamic schema from a user-supplied dict

```python
import asyncio
from pydantic_ai import Agent, StructuredDict
from pydantic_ai.models.test import TestModel

# Schema provided at runtime — no Pydantic model needed
user_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "active": {"type": "boolean"},
    },
    "required": ["name", "age"],
}

PersonDict = StructuredDict(user_schema, name="Person", description="A person record")

agent = Agent(TestModel(), output_type=PersonDict)


async def main():
    result = await agent.run("Create a person named Alice, age 30.")
    print(type(result.output).__name__)  # _StructuredDict (subclass of dict)
    print(result.output)

asyncio.run(main())
```

### 8.2 Combine `StructuredDict` with `ToolOutput` for multi-schema routing

```python
from pydantic import BaseModel
from pydantic_ai import Agent, StructuredDict
from pydantic_ai.output import ToolOutput
from pydantic_ai.models.test import TestModel


class Movie(BaseModel):
    title: str
    year: int


book_schema = StructuredDict(
    {"type": "object", "properties": {"title": {"type": "string"}, "author": {"type": "string"}}, "required": ["title", "author"]},
    name="Book",
)

agent = Agent(
    TestModel(),
    output_type=[
        ToolOutput(Movie, name="return_movie"),
        ToolOutput(book_schema, name="return_book"),
    ],
)

result = agent.run_sync("Tell me about The Great Gatsby.")
print(result.output)
```

### 8.3 `StructuredDict` for OpenAPI-style schemas

```python
import asyncio
import json
from pydantic_ai import Agent, StructuredDict
from pydantic_ai.models.test import TestModel

openapi_schema = {
    "type": "object",
    "title": "SearchResult",
    "description": "A single search result item",
    "properties": {
        "url": {"type": "string", "format": "uri"},
        "title": {"type": "string"},
        "snippet": {"type": "string"},
        "relevance": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["url", "title", "snippet"],
}

SearchResult = StructuredDict(openapi_schema)
agent = Agent(TestModel(), output_type=SearchResult)


async def main():
    result = await agent.run("Find information about Python programming.")
    print(json.dumps(result.output, indent=2))

asyncio.run(main())
```

---

## 9. `AgentStream` — Streaming Abstraction

**Source**: `pydantic_ai/result.py`

```python
from pydantic_ai.result import AgentStream
```

`AgentStream` is the streaming object obtained from a `CallToolsNode` or a `StreamedRunResult`. It provides three independent streaming APIs that operate on the same underlying `StreamedResponse`: raw model events, unvalidated `ModelResponse` snapshots, and fully validated output.

```python
# pydantic_ai/result.py — public interface
@dataclass(kw_only=True)
class AgentStream(Generic[AgentDepsT, OutputDataT]):
    # Primary streaming APIs:
    async def stream_output(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[OutputDataT]: ...
    async def stream_response(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[ModelResponse]: ...
    async def stream_text(self, *, delta: bool = False, debounce_by: float | None = 0.1) -> AsyncIterator[str]: ...

    # Control:
    async def cancel(self) -> None: ...
    async def drain(self) -> None: ...

    # State:
    @property
    def cancelled(self) -> bool: ...
    @property
    def run_id(self) -> str: ...
    @property
    def conversation_id(self) -> str: ...
    @property
    def response(self) -> ModelResponse: ...    # snapshot of current response

    # Finalisation (call after stream is exhausted):
    async def validate_response_output(self, response, *, allow_partial=False) -> OutputDataT: ...
    def usage(self) -> RunUsage: ...
```

### 9.1 Stream validated output with `stream_output()`

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


async def main():
    agent = Agent(TestModel(), instructions="Be verbose.")

    async with agent.run_stream("Explain asyncio in Python.") as streamed:
        last_output = None
        async for output in streamed.stream_output(debounce_by=0.05):
            last_output = output
            print(".", end="", flush=True)
        print()
        print("Final output:", last_output)

asyncio.run(main())
```

### 9.2 Stream raw text deltas with `stream_text(delta=True)`

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


async def main():
    agent = Agent(TestModel())

    async with agent.run_stream("Write a poem about Python.") as streamed:
        async for chunk in streamed.stream_text(delta=True):
            print(chunk, end="", flush=True)
        print()
        print("Usage:", streamed.usage())

asyncio.run(main())
```

### 9.3 Inspect raw `ModelResponse` snapshots with `stream_response()`

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


async def main():
    agent = Agent(TestModel())

    async with agent.run_stream("List three colours.") as streamed:
        async for response in streamed.stream_response():
            total_tokens = response.usage.output_tokens if response.usage else 0
            state = response.state
            print(f"  state={state!r}  output_tokens={total_tokens}")

asyncio.run(main())
```

### 9.4 Cancel mid-stream

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


async def main():
    agent = Agent(TestModel(), instructions="Write a very long essay.")

    async with agent.run_stream("Write 10 000 words on machine learning.") as streamed:
        token_count = 0
        async for chunk in streamed.stream_text(delta=True):
            print(chunk, end="", flush=True)
            token_count += len(chunk.split())
            if token_count > 50:  # stop early
                await streamed.cancel()
                print("\n[Cancelled]")
                break

        print("Cancelled:", streamed.cancelled)

asyncio.run(main())
```

---

## 10. `Hooks` + `HookTimeoutError` — Decorator-Based Capability Hook Registration

**Source**: `pydantic_ai/capabilities/hooks.py`

```python
from pydantic_ai.capabilities import Hooks, HookTimeoutError
```

`Hooks` is a concrete `AbstractCapability` subclass that exposes a `hooks.on` namespace of decorator factories. Instead of subclassing `AbstractCapability` and overriding methods, you register individual async functions for each lifecycle event. Each registered function can optionally carry a timeout; exceeding it raises `HookTimeoutError`.

```python
# Lifecycle hooks available on hooks.on:
hooks.on.before_run           # (ctx) -> None
hooks.on.after_run            # (ctx, *, result) -> AgentRunResult
hooks.on.wrap_run             # (ctx, *, handler) -> AgentRunResult
hooks.on.on_run_error         # (ctx, *, error) -> AgentRunResult
hooks.on.before_node_run      # (ctx, *, node) -> AgentNode
hooks.on.after_node_run       # (ctx, *, node, result) -> NodeResult
hooks.on.wrap_node_run        # (ctx, *, node, handler) -> NodeResult
hooks.on.on_node_run_error    # (ctx, *, node, error) -> NodeResult
hooks.on.wrap_run_event_stream  # (ctx, *, stream) -> AsyncIterable[AgentStreamEvent]
hooks.on.on_event             # (ctx, event) -> AgentStreamEvent
hooks.on.before_model_request # (ctx, request_context) -> ModelRequestContext
hooks.on.after_model_request  # (ctx, *, request_context, response) -> ModelResponse
hooks.on.wrap_model_request   # (ctx, *, request_context, handler) -> ModelResponse
hooks.on.on_model_request_error  # (ctx, *, request_context, error) -> ModelResponse
hooks.on.before_tool_validate # (ctx, tool_call, raw_args) -> RawToolArgs
hooks.on.after_tool_validate  # (ctx, tool_call, validated_args) -> ValidatedToolArgs
hooks.on.before_tool_execute  # (ctx, tool_call, validated_args) -> ValidatedToolArgs
hooks.on.after_tool_execute   # (ctx, tool_call, validated_args, *, result) -> Any
hooks.on.wrap_tool_execute    # (ctx, tool_call, validated_args, *, handler) -> Any
hooks.on.on_tool_execute_error  # (ctx, tool_call, validated_args, *, error) -> Any
hooks.on.before_output_validate  # (ctx, output_ctx, raw) -> Any
hooks.on.after_output_validate   # (ctx, output_ctx, *, output) -> OutputDataT
```

### 10.1 Request/response timing with timeout

```python
import asyncio
import time
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks, HookTimeoutError
from pydantic_ai.models.test import TestModel


hooks = Hooks()

timings: dict[str, float] = {}


@hooks.on.before_model_request(timeout=5.0)  # hook itself must complete in <5s
async def start_timer(ctx, request_context):
    timings["start"] = time.monotonic()
    return request_context


@hooks.on.after_model_request(timeout=5.0)
async def record_latency(ctx, *, request_context, response):
    timings["latency"] = time.monotonic() - timings["start"]
    print(f"[Hooks] model latency: {timings['latency']:.3f}s")
    return response


agent = Agent(TestModel(), capabilities=[hooks])


async def main():
    result = await agent.run("What is the speed of light?")
    print(result.output)
    print(f"Recorded latency: {timings['latency']:.3f}s")

asyncio.run(main())
```

### 10.2 Retry-on-error hook with `on_tool_execute_error`

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel


hooks = Hooks()

_attempt_counts: dict[str, int] = {}


@hooks.on.on_tool_execute_error
async def retry_flaky_tools(ctx, tool_call, validated_args, *, error):
    tool_name = tool_call.tool_name
    _attempt_counts[tool_name] = _attempt_counts.get(tool_name, 0) + 1

    if _attempt_counts[tool_name] < 3 and isinstance(error, (ConnectionError, TimeoutError)):
        print(f"[Hook] Retrying {tool_name} (attempt {_attempt_counts[tool_name]})")
        raise ModelRetry(f"Tool {tool_name} failed, retrying…")

    raise error  # give up after 3 flaky attempts


agent = Agent(TestModel(), capabilities=[hooks])


async def main():
    result = await agent.run("Do something with external tools.")
    print(result.output)

asyncio.run(main())
```

### 10.3 Wrap the entire run for distributed tracing

```python
import asyncio
import uuid
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.models.test import TestModel


hooks = Hooks()


@hooks.on.wrap_run
async def add_trace_context(ctx, *, handler):
    trace_id = uuid.uuid4().hex
    print(f"[Trace] Starting trace {trace_id} for run {ctx.run_id}")
    try:
        result = await handler()
        print(f"[Trace] Run {ctx.run_id} completed OK — trace {trace_id}")
        return result
    except Exception as exc:
        print(f"[Trace] Run {ctx.run_id} FAILED — trace {trace_id}: {exc}")
        raise


agent = Agent(TestModel(), capabilities=[hooks])


async def main():
    result = await agent.run("What day is it?")
    print(result.output)

asyncio.run(main())
```

### 10.4 `HookTimeoutError` — catch and handle stalled hooks

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks, HookTimeoutError
from pydantic_ai.models.test import TestModel


hooks = Hooks()


@hooks.on.before_model_request(timeout=0.001)  # Intentionally tiny timeout
async def slow_hook(ctx, request_context):
    await asyncio.sleep(10)  # This will be cancelled
    return request_context


agent = Agent(TestModel(), capabilities=[hooks])


async def main():
    try:
        result = await agent.run("Hello!")
        print(result.output)
    except HookTimeoutError as exc:
        print(f"Hook '{exc.hook_name}' function '{exc.func_name}' timed out after {exc.timeout}s")

asyncio.run(main())
```

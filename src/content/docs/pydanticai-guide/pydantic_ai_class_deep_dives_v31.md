---
title: "PydanticAI Class Deep Dives Vol. 31"
description: "Source-verified deep dives into 10 pydantic-ai 2.5.0 class groups: ZaiModel + ZaiModelSettings + ZaiProvider + ZaiModelProfile (Z.AI Zhipu GLM family — preserved thinking across turns, zai_clear_thinking, reasoning_content, zai_supports_reasoning_effort for GLM-5.2), VercelProvider (Vercel AI Gateway — OIDC/API-key auth, provider/model naming, 8-provider profile dispatch table), pydantic_graph.paths Path + PathBuilder (fluent path builder — to/broadcast/transform/map/label chaining, fork_id, downstream_join_id for empty-map guard), pydantic_graph.paths marker types TransformFunction + TransformMarker + MapMarker + BroadcastMarker + LabelMarker + DestinationMarker + PathItem (complete path item type system), pydantic_graph.paths EdgePath + EdgePathBuilder (source-to-destination edge wiring — broadcast callback, map/transform chaining, BaseNode class acceptance), pydantic_graph.id_types NodeID + NodeRunID + TaskID + ForkStackItem + ForkStack (type-safe graph identifiers — placeholder generation + replacement), pydantic_graph.node_types MiddleNode + SourceNode + DestinationNode + is_source + is_destination + parent_forks ParentFork + ParentForkFinder (topology type system + deadlock avoidance via dominating fork analysis), durable_exec._runtime_toolsets RuntimeToolsetKind + reject_unsupported_runtime_toolsets (per-run toolset guard for DBOS/Prefect/Temporal — function/mcp/dynamic classification), models/concurrency ConcurrencyLimitedModel + concurrency helpers get_concurrency_context + normalize_to_limiter + AnyConcurrencyLimit (model-layer rate limiting — shared limiter pools, null context, backpressure), capabilities/toolset Toolset capability + PydanticAIDeprecationWarning (inline toolset injection into capability chain + UserWarning-based deprecation infrastructure). All verified against pydantic-ai 2.5.0 source."
sidebar:
  label: "Class deep dives (Vol. 31)"
  order: 57
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="tip">
All examples verified against **pydantic-ai 2.5.0** source installed directly from PyPI. Every class signature, field name, and method in this volume reflects the 2.5.x API.
</Aside>

Ten class groups covering two new model providers (`ZaiModel`/`ZaiProvider` for Z.AI Zhipu AI GLM, and `VercelProvider` for Vercel AI Gateway), the new `pydantic_graph.paths` module (fluent `Path`/`PathBuilder`/`EdgePath`/`EdgePathBuilder` builder API, marker types, and `TransformFunction`), new graph type-safety modules (`id_types`, `node_types`, `parent_forks`), durable-execution runtime toolset validation, model-level concurrency limiting helpers, and the new `Toolset` capability with the `PydanticAIDeprecationWarning` infrastructure.

---

## 1. `ZaiModel` + `ZaiModelSettings` + `ZaiProvider` + `ZaiModelProfile` — Z.AI (Zhipu AI) GLM Family

Z.AI's GLM models are Zhipu AI's leading open-weight models with support for vision, thinking/reasoning, and *preserved thinking across turns*.  The key differentiators versus other OpenAI-compatible providers:

- **`reasoning_content` field** — Z.AI sends thinking blocks as a separate JSON field, not inline text.  `ZaiModel.prepare_request` maps the unified `thinking=` setting to `extra_body.thinking.type = 'enabled'|'disabled'`.
- **Preserved thinking (`zai_clear_thinking`)** — by default the library keeps `reasoning_content` from prior turns and sends it back unchanged (`clear_thinking=False`), matching the Z.AI "preserved thinking" API contract.  Set `zai_clear_thinking=True` to discard prior reasoning.
- **Per-request reasoning effort (GLM-5.2)** — when `ZaiModelProfile.zai_supports_reasoning_effort=True` the library forwards the effort string (`'low'`, `'high'`, …) as `extra_body.reasoning_effort`.
- **`zai_model_profile`** — the profile function marks `glm-5*`, `glm-4.7*`, `glm-4.6*`, `glm-4.5*` as `supports_thinking=True`; GLM-5.2 additionally gets `zai_supports_reasoning_effort=True`.

```python
# Key signatures verified from source (pydantic-ai 2.5.0):
#
# class ZaiModelSettings(ModelSettings, total=False):
#     zai_clear_thinking: bool
#
# class ZaiModelProfile(ModelProfile, total=False):
#     zai_supports_reasoning_effort: bool
#
# @dataclass(init=False)
# class ZaiModel(OpenAIChatModel):
#     def __init__(
#         self,
#         model_name: ZaiModelName,
#         *,
#         provider: Literal['zai'] | Provider[AsyncOpenAI] = 'zai',
#         profile: ModelProfileSpec | None = None,
#         settings: ZaiModelSettings | None = None,
#     ): ...
#
# class ZaiProvider(Provider[AsyncOpenAI]):
#     base_url = 'https://api.z.ai/api/paas/v4'
#     # env var: ZAI_API_KEY
```

### 1.1 Basic GLM-5 Agent with Thinking Mode

```python
import asyncio
import os

from pydantic_ai import Agent
from pydantic_ai.models.zai import ZaiModel

os.environ.setdefault('ZAI_API_KEY', 'your-key-here')

agent = Agent(
    ZaiModel('glm-5', settings={'thinking': True}),
    system_prompt='You are a concise reasoning assistant.',
)


async def main() -> None:
    result = await agent.run('Explain why 0.1 + 0.2 != 0.3 in IEEE 754.')
    print(result.output)


asyncio.run(main())
```

### 1.2 Preserved Thinking Across Multi-Turn Conversations

```python
import asyncio
import os

from pydantic_ai import Agent
from pydantic_ai.models.zai import ZaiModel, ZaiModelSettings

os.environ.setdefault('ZAI_API_KEY', 'your-key-here')

# Default: zai_clear_thinking=False → preserve reasoning_content across turns
settings: ZaiModelSettings = {'thinking': True, 'zai_clear_thinking': False}
agent = Agent(ZaiModel('glm-5.1', settings=settings))


async def main() -> None:
    # First turn
    result1 = await agent.run('What is the capital of France?')
    history = result1.all_messages()

    # Second turn — reasoning_content from turn 1 is returned unchanged
    result2 = await agent.run('And what language do they speak?', message_history=history)
    print(result2.output)


asyncio.run(main())
```

### 1.3 Per-Request Reasoning Effort on GLM-5.2

```python
import asyncio
import os

from pydantic_ai import Agent
from pydantic_ai.models.zai import ZaiModel, ZaiModelSettings

os.environ.setdefault('ZAI_API_KEY', 'your-key-here')

# GLM-5.2 accepts reasoning_effort via extra_body; the library maps effort strings automatically
low_effort: ZaiModelSettings = {'thinking': 'low'}
high_effort: ZaiModelSettings = {'thinking': 'high'}

agent_low = Agent(ZaiModel('glm-5.2', settings=low_effort))
agent_high = Agent(ZaiModel('glm-5.2', settings=high_effort))


async def main() -> None:
    q = 'List the first 5 prime numbers.'
    r_low = await agent_low.run(q)
    r_high = await agent_high.run(q)
    print('Low effort:', r_low.output)
    print('High effort:', r_high.output)


asyncio.run(main())
```

---

## 2. `VercelProvider` — Vercel AI Gateway Multi-Provider Routing

`VercelProvider` routes through the Vercel AI Gateway at `https://ai-gateway.vercel.sh/v1`, which proxies 8+ upstream providers under a single OIDC/API-key authentication surface.

**Authentication** — `VERCEL_AI_GATEWAY_API_KEY` (static API key) or `VERCEL_OIDC_TOKEN` (Vercel-issued OIDC token for edge functions), checked in that order.

**Model naming** — `provider/model` (e.g. `anthropic/claude-opus-4-8`).  An unqualified name (no `/`) gets the OpenAI profile by default.

**Profile dispatch** — `VercelProvider.model_profile()` maps the provider prefix to one of 8 profile functions (`anthropic_model_profile`, `amazon_model_profile`, `cohere_model_profile`, `deepseek_model_profile`, `mistral_model_profile`, `openai_model_profile`, `google_model_profile`, `grok_model_profile`).  The result is merged with `OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer)` so every upstream uses OpenAI-compatible JSON schema output.

```python
# Key signatures verified from source (pydantic-ai 2.5.0):
#
# class VercelProvider(Provider[AsyncOpenAI]):
#     base_url = 'https://ai-gateway.vercel.sh/v1'
#     def __init__(
#         self, *, api_key: str | None = None,
#         openai_client: AsyncOpenAI | None = None,
#         http_client: httpx.AsyncClient | None = None,
#     ) -> None: ...
#     @staticmethod
#     def model_profile(model_name: str) -> ModelProfile | None: ...
```

### 2.1 Routing to Anthropic via Vercel Gateway

```python
import asyncio
import os

from pydantic_ai import Agent

os.environ.setdefault('VERCEL_AI_GATEWAY_API_KEY', 'your-vercel-key')

# prefix determines which upstream profile is selected
agent = Agent('openai:anthropic/claude-opus-4-8', providers=['vercel'])


async def main() -> None:
    result = await agent.run('Say hello in three languages.')
    print(result.output)


asyncio.run(main())
# Alternatively construct the provider explicitly:
from pydantic_ai.providers.vercel import VercelProvider
from pydantic_ai.models.openai import OpenAIChatModel

provider = VercelProvider()
model = OpenAIChatModel('anthropic/claude-opus-4-8', provider=provider)
agent2 = Agent(model)
```

### 2.2 Explicit Provider Construction with Custom HTTP Client

```python
import asyncio
import os

import httpx

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.vercel import VercelProvider

os.environ.setdefault('VERCEL_AI_GATEWAY_API_KEY', 'your-vercel-key')

# Share an httpx client for connection pooling
http_client = httpx.AsyncClient(timeout=30.0)
provider = VercelProvider(api_key=os.environ['VERCEL_AI_GATEWAY_API_KEY'], http_client=http_client)

agent = Agent(OpenAIChatModel('openai/gpt-5', provider=provider))


async def main() -> None:
    async with http_client:
        result = await agent.run('What is 2 + 2?')
        print(result.output)


asyncio.run(main())
```

### 2.3 Inspecting the Resolved Model Profile

```python
from pydantic_ai.providers.vercel import VercelProvider

# Profile dispatch table: 8 upstream families
for model_name in [
    'anthropic/claude-sonnet-4-5',
    'openai/gpt-5.2',
    'deepseek/deepseek-r2',
    'xai/grok-3',
    'unknown-provider/some-model',
]:
    profile = VercelProvider.model_profile(model_name)
    print(f'{model_name}: json_schema_transformer={profile and profile.get("json_schema_transformer").__class__.__name__}')

# Unqualified name → OpenAI profile with OpenAIJsonSchemaTransformer
bare_profile = VercelProvider.model_profile('gpt-5')
print('bare name profile:', bare_profile)
```

---

## 3. `pydantic_graph.paths`: `Path` + `PathBuilder` — Fluent Path Builder API

`Path` is the core data structure for the pydantic_graph builder API: a flat `list[PathItem]` that encodes transforms, forks, and routing in order.  `PathBuilder` is the fluent builder wrapper whose methods return new `PathBuilder` or completed `Path` instances.

| `PathBuilder` method | Returns | Purpose |
|---|---|---|
| `.to(destination, ...)` | `Path` | Route to one or more destination nodes (creates a `BroadcastMarker` for multiple) |
| `.broadcast(forks)` | `Path` | Fan-out to a sequence of pre-built `Path` objects |
| `.transform(func)` | `PathBuilder[..., T]` | Apply a sync `TransformFunction` (changes the OutputT type) |
| `.map()` | `PathBuilder[..., T]` | Spread an `Iterable[T]` or `AsyncIterable[T]` into parallel per-item paths |
| `.label(label)` | `PathBuilder[...]` | Attach a human-readable label for debugging/visualisation |

```python
# Key signatures verified from source (pydantic-ai 2.5.0 / pydantic-graph 2.5.0):
#
# @dataclass
# class Path:
#     items: list[PathItem]
#     @property def last_fork(self) -> BroadcastMarker | MapMarker | None: ...
#     @property def next_path(self) -> Path: ...
#
# @dataclass
# class PathBuilder(Generic[StateT, DepsT, OutputT]):
#     working_items: Sequence[PathItem]
#     def to(self, destination, /, *extra, fork_id=None) -> Path: ...
#     def broadcast(self, forks, /, *, fork_id=None) -> Path: ...
#     def transform(self, func) -> PathBuilder[StateT, DepsT, T]: ...
#     def map(self, *, fork_id=None, downstream_join_id=None) -> PathBuilder[StateT, DepsT, T]: ...
#     def label(self, label) -> PathBuilder[StateT, DepsT, OutputT]: ...
```

### 3.1 Simple Single-Destination Path

```python
from pydantic_graph.graph_builder import GraphBuilder
from pydantic_graph.step import Step, StepContext


class State:
    value: int = 0


class Double(Step[State, None, int, int]):
    async def run(self, ctx: StepContext[State, None, int]) -> int:
        return ctx.inputs * 2


class Printer(Step[State, None, int, None]):
    async def run(self, ctx: StepContext[State, None, int]) -> None:
        print('result:', ctx.inputs)


builder = GraphBuilder[State, None]()
double = Double()
printer = Printer()

# Path: double → printer
builder.add_edge(double).to(printer)
```

### 3.2 Labelled Transform Path

```python
from pydantic_graph.graph_builder import GraphBuilder
from pydantic_graph.paths import PathBuilder
from pydantic_graph.step import Step, StepContext


class State:
    pass


class Source(Step[State, None, str, str]):
    async def run(self, ctx: StepContext[State, None, str]) -> str:
        return ctx.inputs.upper()


class Sink(Step[State, None, int, None]):
    async def run(self, ctx: StepContext[State, None, int]) -> None:
        print('char count:', ctx.inputs)


builder = GraphBuilder[State, None]()
source, sink = Source(), Sink()

# Label the path segment and apply a sync transform (str → int)
builder.add_edge(source).label('count-chars').transform(lambda ctx: len(ctx.inputs)).to(sink)
```

### 3.3 Broadcast Fan-Out to Two Destinations

```python
from pydantic_graph.graph_builder import GraphBuilder
from pydantic_graph.paths import PathBuilder
from pydantic_graph.step import Step, StepContext


class State:
    pass


class Producer(Step[State, None, str, str]):
    async def run(self, ctx: StepContext[State, None, str]) -> str:
        return ctx.inputs


class LogSink(Step[State, None, str, None]):
    async def run(self, ctx: StepContext[State, None, str]) -> None:
        print('[log]', ctx.inputs)


class StoreSink(Step[State, None, str, None]):
    async def run(self, ctx: StepContext[State, None, str]) -> None:
        print('[store]', ctx.inputs)


builder = GraphBuilder[State, None]()
producer, log_sink, store_sink = Producer(), LogSink(), StoreSink()

# to() with multiple destinations automatically wraps in a BroadcastMarker
builder.add_edge(producer).to(log_sink, store_sink)
```

---

## 4. `pydantic_graph.paths` Marker Types — `TransformFunction`, `TransformMarker`, `MapMarker`, `BroadcastMarker`, `LabelMarker`, `DestinationMarker`, `PathItem`

Each `Path` is a list of these marker dataclasses.  Understanding the type system lets you inspect, mutate, or generate paths programmatically.

| Type | Role | Key fields |
|---|---|---|
| `TransformFunction` | Protocol for sync step functions | `__call__(ctx: StepContext) -> OutputT` |
| `TransformMarker` | Wraps a `TransformFunction` in a path | `transform: TransformFunction` |
| `MapMarker` | Spreads an iterable into parallel forks | `fork_id`, `downstream_join_id` |
| `BroadcastMarker` | Fan-out to multiple pre-built paths | `paths: Sequence[Path]`, `fork_id` |
| `LabelMarker` | Human-readable annotation | `label: str` |
| `DestinationMarker` | Terminal routing to a node | `destination_id: NodeID` |
| `PathItem` | Union of all the above | `TransformMarker \| MapMarker \| BroadcastMarker \| LabelMarker \| DestinationMarker` |

```python
# Key signatures verified from source (pydantic-graph 2.5.0):
#
# class TransformFunction(Protocol[StateT, DepsT, InputT, OutputT]):
#     def __call__(self, ctx: StepContext[StateT, DepsT, InputT]) -> OutputT: ...
#
# @dataclass class TransformMarker:
#     transform: TransformFunction[Any, Any, Any, Any]
#
# @dataclass class MapMarker:
#     fork_id: ForkID
#     downstream_join_id: JoinID | None
#
# @dataclass class BroadcastMarker:
#     paths: Sequence[Path]
#     fork_id: ForkID
#
# @dataclass class LabelMarker:
#     label: str
#
# @dataclass class DestinationMarker:
#     destination_id: NodeID
#
# PathItem = TransformMarker | MapMarker | BroadcastMarker | LabelMarker | DestinationMarker
```

### 4.1 Inspecting Path Items at Runtime

```python
from pydantic_graph.paths import (
    BroadcastMarker,
    DestinationMarker,
    LabelMarker,
    MapMarker,
    Path,
    TransformMarker,
)
from pydantic_graph.id_types import ForkID, NodeID


def describe_path(path: Path) -> None:
    """Pretty-print a path's item types."""
    for i, item in enumerate(path.items):
        match item:
            case TransformMarker():
                print(f'  [{i}] transform: {item.transform}')
            case MapMarker(fork_id=fid):
                print(f'  [{i}] map (fork={fid}, join={item.downstream_join_id})')
            case BroadcastMarker(fork_id=fid):
                print(f'  [{i}] broadcast (fork={fid}, {len(item.paths)} branches)')
            case LabelMarker(label=lbl):
                print(f'  [{i}] label: {lbl!r}')
            case DestinationMarker(destination_id=did):
                print(f'  [{i}] → destination: {did}')


# Build a sample path manually
sample = Path(items=[
    LabelMarker('process'),
    TransformMarker(lambda ctx: str(ctx.inputs)),
    DestinationMarker(NodeID('my_node')),
])
describe_path(sample)
```

### 4.2 Building a `MapMarker` Path for Per-Item Parallel Execution

```python
from pydantic_graph.graph_builder import GraphBuilder
from pydantic_graph.step import Step, StepContext


class State:
    pass


class BatchProducer(Step[State, None, str, list[str]]):
    async def run(self, ctx: StepContext[State, None, str]) -> list[str]:
        return ctx.inputs.split(',')


class ItemProcessor(Step[State, None, str, None]):
    async def run(self, ctx: StepContext[State, None, str]) -> None:
        print('processing item:', ctx.inputs.strip())


builder = GraphBuilder[State, None]()
producer, processor = BatchProducer(), ItemProcessor()

# map() spreads list[str] into one parallel path per str → creates a MapMarker
builder.add_edge(producer).map().to(processor)

# The produced path items will be:
#   MapMarker(fork_id=..., downstream_join_id=None)
#   DestinationMarker(destination_id=processor.id)
```

### 4.3 Constructing a `BroadcastMarker` from Raw Paths

```python
from pydantic_graph.paths import BroadcastMarker, DestinationMarker, Path
from pydantic_graph.id_types import ForkID, NodeID

# Build branch paths manually (each ends at a different destination)
branch_a = Path(items=[DestinationMarker(NodeID('node_a'))])
branch_b = Path(items=[DestinationMarker(NodeID('node_b'))])

broadcast = BroadcastMarker(
    paths=[branch_a, branch_b],
    fork_id=ForkID(NodeID('my_broadcast_fork')),
)

root_path = Path(items=[broadcast])
print('last fork:', root_path.last_fork)          # BroadcastMarker
print('next path items:', root_path.next_path.items)  # []
```

---

## 5. `pydantic_graph.paths`: `EdgePath` + `EdgePathBuilder` — Source-to-Destination Edge Wiring

`EdgePath` is a *complete* edge: it binds a sequence of source nodes to a `Path` and collects the referenced destination nodes.  `EdgePathBuilder` is the fluent builder; `GraphBuilder.add_edge()` returns one.

```python
# Key signatures verified from source (pydantic-graph 2.5.0):
#
# @dataclass(init=False)
# class EdgePath(Generic[StateT, DepsT]):
#     path: Path
#     destinations: list[AnyDestinationNode]
#     @property def sources(self) -> Sequence[SourceNode]: ...
#
# class EdgePathBuilder(Generic[StateT, DepsT, OutputT]):
#     def __init__(self, sources, path_builder): ...
#     def to(self, destination, /, *extra, fork_id=None) -> EdgePath: ...
#     def broadcast(self, get_forks, /, *, fork_id=None) -> EdgePath: ...
#     def map(self, *, fork_id=None, downstream_join_id=None) -> EdgePathBuilder: ...
#     def transform(self, func) -> EdgePathBuilder: ...
#     def label(self, label) -> EdgePathBuilder: ...
```

### 5.1 Single-Source Single-Destination Edge

```python
from pydantic_graph.graph_builder import GraphBuilder
from pydantic_graph.step import Step, StepContext


class State:
    pass


class Producer(Step[State, None, str, str]):
    async def run(self, ctx: StepContext[State, None, str]) -> str:
        return ctx.inputs


class Consumer(Step[State, None, str, None]):
    async def run(self, ctx: StepContext[State, None, str]) -> None:
        print('consumed:', ctx.inputs)


builder = GraphBuilder[State, None]()
producer, consumer = Producer(), Consumer()

# add_edge() returns an EdgePathBuilder; .to() finalises it as an EdgePath
edge = builder.add_edge(producer).to(consumer)
print('sources:', [type(s).__name__ for s in edge.sources])
print('destinations:', [type(d).__name__ for d in edge.destinations])
```

### 5.2 Transform + Map + Fan-Out Edge Chain

```python
from pydantic_graph.graph_builder import GraphBuilder
from pydantic_graph.step import Step, StepContext


class State:
    pass


class SentenceSplitter(Step[State, None, str, list[str]]):
    async def run(self, ctx: StepContext[State, None, str]) -> list[str]:
        return ctx.inputs.split('. ')


class WordCounter(Step[State, None, int, None]):
    async def run(self, ctx: StepContext[State, None, int]) -> None:
        print('word count:', ctx.inputs)


builder = GraphBuilder[State, None]()
splitter, counter = SentenceSplitter(), WordCounter()

# split → map per sentence → transform str→int → count
(
    builder
    .add_edge(splitter)             # EdgePathBuilder[State, None, list[str]]
    .map()                          # EdgePathBuilder[State, None, str]
    .transform(lambda ctx: len(ctx.inputs.split()))  # EdgePathBuilder[State, None, int]
    .to(counter)                    # EdgePath
)
```

### 5.3 Broadcast Callback Pattern

```python
from pydantic_graph.graph_builder import GraphBuilder
from pydantic_graph.step import Step, StepContext


class State:
    pass


class Source(Step[State, None, str, str]):
    async def run(self, ctx: StepContext[State, None, str]) -> str:
        return ctx.inputs


class LogSink(Step[State, None, str, None]):
    async def run(self, ctx: StepContext[State, None, str]) -> None:
        print('[log]', ctx.inputs)


class MetricsSink(Step[State, None, str, None]):
    async def run(self, ctx: StepContext[State, None, str]) -> None:
        print('[metrics]', len(ctx.inputs))


builder = GraphBuilder[State, None]()
source, log_sink, metrics_sink = Source(), LogSink(), MetricsSink()

# broadcast() accepts a callback that receives the builder for each branch
(
    builder
    .add_edge(source)
    .broadcast(lambda b: [b.to(log_sink), b.to(metrics_sink)])
)
```

---

## 6. `pydantic_graph.id_types`: `NodeID` + `NodeRunID` + `TaskID` + `ForkStackItem` + `ForkStack`

`pydantic_graph.id_types` provides `NewType` wrappers for every identifier in graph execution, preventing accidental mixing of node IDs and task IDs.

| Type | Base | Purpose |
|---|---|---|
| `NodeID` | `str` | Stable identifier for a node in the graph (set at build time) |
| `NodeRunID` | `str` | Identifier for a specific *execution* of a node (generated at runtime) |
| `TaskID` | `str` | Identifier for a task within graph execution |
| `JoinID` | `NodeID` | Alias for join nodes |
| `ForkID` | `NodeID` | Alias for fork nodes |
| `ForkStackItem` | frozen dataclass | Single fork point: `fork_id`, `node_run_id`, `thread_index` |
| `ForkStack` | `tuple[ForkStackItem, ...]` | Complete parallel execution ancestry stack |

`generate_placeholder_node_id(label)` creates a UUID-based placeholder; `replace_placeholder_id(node_id)` strips the prefix, returning just the `label`.

```python
# Key signatures verified from source (pydantic-graph 2.5.0):
#
# NodeID = NewType('NodeID', str)
# NodeRunID = NewType('NodeRunID', str)
# TaskID = NewType('TaskID', str)
# JoinID = NodeID
# ForkID = NodeID
#
# @dataclass(frozen=True)
# class ForkStackItem:
#     fork_id: ForkID
#     node_run_id: NodeRunID
#     thread_index: int
#
# ForkStack = tuple[ForkStackItem, ...]
#
# def generate_placeholder_node_id(label: str) -> str: ...
# def replace_placeholder_id(node_id: NodeID) -> str: ...
```

### 6.1 Type-Safe ID Creation and Comparison

```python
from pydantic_graph.id_types import ForkID, NodeID, NodeRunID, TaskID

# NewType wrappers: same runtime value, different static types
node_a = NodeID('step_parse')
node_b = NodeID('step_validate')
run_id = NodeRunID('run-abc-123')
task = TaskID('task-xyz')

# IDs are just strings at runtime — comparison works normally
print(node_a == 'step_parse')   # True
print(node_a == node_b)         # False

# ForkID is an alias for NodeID
fork = ForkID(NodeID('my_fork'))
print(type(fork) is str)        # True (NewTypes don't create new classes)
```

### 6.2 Placeholder ID Generation and Replacement

```python
from pydantic_graph.id_types import NodeID, generate_placeholder_node_id, replace_placeholder_id

# During graph building, nodes without explicit IDs get stable placeholders
placeholder = generate_placeholder_node_id('broadcast')
print('placeholder:', placeholder)       # '__placeholder__:broadcast:<uuid>'

# After graph compilation, placeholders are replaced with the label portion
resolved = replace_placeholder_id(NodeID(placeholder))
print('resolved label:', resolved)       # 'broadcast'

# Non-placeholder IDs pass through unchanged
explicit = NodeID('my_explicit_node')
print(replace_placeholder_id(explicit))  # 'my_explicit_node'
```

### 6.3 ForkStack for Tracking Parallel Execution Ancestry

```python
from pydantic_graph.id_types import ForkID, ForkStack, ForkStackItem, NodeID, NodeRunID

# A ForkStack represents the full ancestry of a parallel execution thread
item1 = ForkStackItem(
    fork_id=ForkID(NodeID('outer_fork')),
    node_run_id=NodeRunID('run-outer-001'),
    thread_index=0,
)
item2 = ForkStackItem(
    fork_id=ForkID(NodeID('inner_fork')),
    node_run_id=NodeRunID('run-inner-002'),
    thread_index=1,
)

# ForkStack is just a tuple — immutable and hashable
stack: ForkStack = (item1, item2)
print('depth:', len(stack))
print('innermost fork:', stack[-1].fork_id)
print('thread index:', stack[-1].thread_index)

# ForkStackItems are frozen dataclasses — safe as dict keys
ancestry: dict[ForkStack, str] = {stack: 'thread-path-A'}
print(ancestry[stack])
```

---

## 7. `pydantic_graph.node_types` + `parent_forks`: Topology Type System + Deadlock Avoidance

**`node_types`** defines type aliases and type guards for graph topology: every node is classified as a *source* (can produce output), a *destination* (can consume input), or both (*middle*).

**`parent_forks`** provides `ParentFork` + `ParentForkFinder` for identifying the *dominating fork* of a join node — the fork that all paths to the join must pass through.  This is the key primitive the runtime uses to avoid deadlock in parallel execution.

```python
# Key signatures verified from source (pydantic-graph 2.5.0):
#
# MiddleNode  = Step | Join | Fork                                  (source + destination)
# SourceNode  = MiddleNode | StartNode                              (produces output)
# DestinationNode = MiddleNode | Decision | EndNode                 (consumes input)
# AnyNode     = AnySourceNode | AnyDestinationNode
#
# def is_source(node: AnyNode) -> TypeGuard[AnySourceNode]: ...
# def is_destination(node: AnyNode) -> TypeGuard[AnyDestinationNode]: ...
#
# @dataclass class ParentFork(Generic[T]):
#     fork_id: T
#     intermediate_nodes: set[T]
#
# @dataclass class ParentForkFinder(Generic[T]):
#     nodes: set[T]
#     start_ids: set[T]
#     fork_ids: set[T]
#     edges: dict[T, list[T]]
#     def find_parent_fork(self, join_id, *, parent_fork_id=None, prefer_closest=False) -> ParentFork | None: ...
```

### 7.1 Using Type Guards to Discriminate Node Roles

```python
from pydantic_graph.node_types import is_destination, is_source
from pydantic_graph.node import EndNode, Fork, StartNode
from pydantic_graph.step import NodeStep, Step, StepContext
from pydantic_graph.id_types import NodeID


class State:
    pass


class MyStep(Step[State, None, str, str]):
    id = NodeID('my_step')

    async def run(self, ctx: StepContext[State, None, str]) -> str:
        return ctx.inputs


step = NodeStep(MyStep)
start = StartNode[str]()
end = EndNode[str]()

# Type guards narrow to AnySourceNode / AnyDestinationNode
print('step is_source:', is_source(step))       # True  (Step is MiddleNode)
print('step is_destination:', is_destination(step))  # True
print('start is_source:', is_source(start))     # True  (StartNode)
print('start is_destination:', is_destination(start))  # False
print('end is_destination:', is_destination(end))  # True
print('end is_source:', is_source(end))          # False
```

### 7.2 Finding the Dominating Parent Fork for Deadlock Avoidance

```python
from pydantic_graph.parent_forks import ParentForkFinder

# Graph: start → fork → (branch_a, branch_b) → join → end
finder = ParentForkFinder(
    nodes={'start', 'fork', 'branch_a', 'branch_b', 'join', 'end'},
    start_ids={'start'},
    fork_ids={'fork'},
    edges={
        'start':    ['fork'],
        'fork':     ['branch_a', 'branch_b'],
        'branch_a': ['join'],
        'branch_b': ['join'],
        'join':     ['end'],
        'end':      [],
    },
)

parent_fork = finder.find_parent_fork('join')
if parent_fork:
    print('parent fork id:', parent_fork.fork_id)         # 'fork'
    print('intermediate nodes:', parent_fork.intermediate_nodes)  # {'branch_a', 'branch_b'}
```

### 7.3 Specifying a Parent Fork to Resolve Ambiguity

```python
from pydantic_graph.parent_forks import ParentForkFinder

# Nested forks: outer_fork → inner_fork; join only sees inner_fork branches
finder = ParentForkFinder(
    nodes={'start', 'outer_fork', 'inner_fork', 'a', 'b', 'inner_join', 'outer_end'},
    start_ids={'start'},
    fork_ids={'outer_fork', 'inner_fork'},
    edges={
        'start':      ['outer_fork'],
        'outer_fork': ['inner_fork', 'outer_end'],
        'inner_fork': ['a', 'b'],
        'a':          ['inner_join'],
        'b':          ['inner_join'],
        'inner_join': ['outer_end'],
        'outer_end':  [],
    },
)

# Without hint: finds the most ancestral dominating fork
parent = finder.find_parent_fork('inner_join')
print('auto parent:', parent and parent.fork_id)

# Explicit parent_fork_id overrides the search — useful in complex diamond patterns
explicit = finder.find_parent_fork('inner_join', parent_fork_id='inner_fork')
print('explicit parent:', explicit and explicit.fork_id)  # 'inner_fork'
```

---

## 8. `durable_exec._runtime_toolsets`: `RuntimeToolsetKind` + `reject_unsupported_runtime_toolsets`

Durable execution engines (DBOS, Prefect, Temporal) *wrap* the agent's constructor-time toolsets so function calls become checkpointed activities/tasks.  Toolsets passed per-run via `run(toolsets=...)` arrive **after** that wrapping, making them un-checkpointed.

`reject_unsupported_runtime_toolsets` enforces the constraint at runtime by classifying every leaf toolset via `_runtime_toolset_kind`:

| Leaf type | `RuntimeToolsetKind` | Rejected by |
|---|---|---|
| `FunctionToolset` | `'function'` | Temporal, Prefect (not DBOS — DBOS runs them inline) |
| `MCPToolset` | `'mcp'` | All engines |
| `DynamicToolset` | `'dynamic'` | All engines (contents unknown at registration time) |
| `ExternalToolset`, custom `AbstractToolset` | `None` | No engine (non-executing, safe per-run) |

```python
# Key signatures verified from source (pydantic-ai 2.5.0):
#
# RuntimeToolsetKind = Literal['function', 'mcp', 'dynamic']
#
# def reject_unsupported_runtime_toolsets(
#     toolsets: Sequence[AbstractToolset[Any]] | None,
#     *,
#     unsupported_kinds: frozenset[RuntimeToolsetKind],
#     engine: str,
# ) -> None: ...
```

### 8.1 Understanding Which Toolsets Are Safe Per-Run

```python
from pydantic_ai.durable_exec._runtime_toolsets import (
    RuntimeToolsetKind,
    _runtime_toolset_kind,
)
from pydantic_ai.toolsets import ExternalToolset
from pydantic_ai.toolsets.function import FunctionToolset


def check_kind(toolset) -> RuntimeToolsetKind | None:
    return _runtime_toolset_kind(toolset)


# FunctionToolset → 'function' (must be registered at agent construction)
ft = FunctionToolset()

@ft.tool
def greet(name: str) -> str:
    return f'Hello, {name}'

print('FunctionToolset kind:', check_kind(ft))   # 'function'

# ExternalToolset → None (safe to pass per-run)
ext = ExternalToolset(id='my_ext')
print('ExternalToolset kind:', check_kind(ext))  # None
```

### 8.2 Temporal Rejects Function + MCP Toolsets Per-Run

```python
from pydantic_ai.durable_exec._runtime_toolsets import (
    reject_unsupported_runtime_toolsets,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.toolsets.function import FunctionToolset

ft = FunctionToolset()

@ft.tool
def add(a: int, b: int) -> int:
    return a + b

try:
    reject_unsupported_runtime_toolsets(
        [ft],
        unsupported_kinds=frozenset({'function', 'mcp', 'dynamic'}),
        engine='Temporal',
    )
except UserError as e:
    print('Temporal blocked:', e)
# → UserError: FunctionToolset cannot be passed to `run(toolsets=...)` at runtime with Temporal …
```

### 8.3 ExternalToolset Passes the Guard for All Engines

```python
from pydantic_ai.durable_exec._runtime_toolsets import reject_unsupported_runtime_toolsets
from pydantic_ai.toolsets import ExternalToolset

# ExternalToolset results are resolved outside the agent → no durable wrapping needed
ext = ExternalToolset(id='approval_queue')

# Calling with any engine + any unsupported_kinds → no exception for ExternalToolset
for engine, kinds in [
    ('Temporal', frozenset({'function', 'mcp', 'dynamic'})),
    ('Prefect', frozenset({'mcp', 'dynamic'})),
    ('DBOS', frozenset({'mcp', 'dynamic'})),
]:
    reject_unsupported_runtime_toolsets([ext], unsupported_kinds=kinds, engine=engine)
    print(f'{engine}: ExternalToolset allowed ✓')
```

---

## 9. `ConcurrencyLimitedModel` + `get_concurrency_context` + `normalize_to_limiter` + `AnyConcurrencyLimit`

`ConcurrencyLimitedModel` (in `pydantic_ai.models.concurrency`) wraps any `Model` and applies `AbstractConcurrencyLimiter` at the HTTP request layer — both `request()` and `request_stream()` gates are covered.

`pydantic_ai.concurrency` gained two public helpers in v2.5.0:

- **`get_concurrency_context(limiter, source)`** — returns a no-op async context manager when `limiter is None`, or the real gating context otherwise.  Used by `ConcurrencyLimitedModel` internally.
- **`normalize_to_limiter(limit, *, name)`** — accepts `int | ConcurrencyLimit | AbstractConcurrencyLimiter | None` (`AnyConcurrencyLimit`) and returns `AbstractConcurrencyLimiter | None`.

```python
# Key signatures verified from source (pydantic-ai 2.5.0):
#
# AnyConcurrencyLimit: TypeAlias = 'int | ConcurrencyLimit | AbstractConcurrencyLimiter | None'
#
# @dataclass(init=False)
# class ConcurrencyLimitedModel(WrapperModel):
#     def __init__(
#         self,
#         wrapped: Model | KnownModelName,
#         limiter: int | ConcurrencyLimit | AbstractConcurrencyLimiter,
#     ): ...
#
# def get_concurrency_context(
#     limiter: AbstractConcurrencyLimiter | None,
#     source: str = 'unnamed',
# ) -> AbstractAsyncContextManager[None]: ...
#
# def normalize_to_limiter(
#     limit: AnyConcurrencyLimit, *, name: str | None = None,
# ) -> AbstractConcurrencyLimiter | None: ...
```

### 9.1 Limiting an Agent to 3 Concurrent Model Requests

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.concurrency import ConcurrencyLimitedModel

# Simple int limit: max 3 concurrent requests, unlimited queue
limited_model = ConcurrencyLimitedModel('openai:gpt-5.2', limiter=3)
agent = Agent(limited_model)


async def main() -> None:
    # Fire 10 requests; at most 3 hit the model simultaneously
    tasks = [agent.run(f'Question {i}: what is {i} * 7?') for i in range(10)]
    results = await asyncio.gather(*tasks)
    for r in results:
        print(r.output)


asyncio.run(main())
```

### 9.2 Shared Limiter Across Multiple Model Instances

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai.concurrency import ConcurrencyLimit, ConcurrencyLimiter
from pydantic_ai.models.concurrency import ConcurrencyLimitedModel

# Shared pool: total 5 concurrent requests across gpt-5 and gpt-5-mini
pool = ConcurrencyLimiter.from_limit(
    ConcurrencyLimit(max_running=5, max_queued=20),
    name='openai-shared-pool',
)

model_heavy = ConcurrencyLimitedModel('openai:gpt-5', limiter=pool)
model_light = ConcurrencyLimitedModel('openai:gpt-5.2', limiter=pool)

agent_heavy = Agent(model_heavy)
agent_light = Agent(model_light)


async def main() -> None:
    r1 = await agent_heavy.run('Summarise the history of Rome in one sentence.')
    r2 = await agent_light.run('What is 2 + 2?')
    print(r1.output)
    print(r2.output)


asyncio.run(main())
```

### 9.3 Using `normalize_to_limiter` and `get_concurrency_context` Directly

```python
import asyncio

from pydantic_ai.concurrency import (
    AnyConcurrencyLimit,
    get_concurrency_context,
    normalize_to_limiter,
)


async def rate_limited_call(limit: AnyConcurrencyLimit, work_label: str) -> None:
    limiter = normalize_to_limiter(limit, name='my-limiter')
    async with get_concurrency_context(limiter, source=work_label):
        print(f'  executing: {work_label}')
        await asyncio.sleep(0.1)


async def main() -> None:
    # int → ConcurrencyLimiter
    await rate_limited_call(3, 'task-a')
    # None → no-op context
    await rate_limited_call(None, 'task-b')
    print('done')


asyncio.run(main())
```

---

## 10. `capabilities/toolset.py`: `Toolset` Capability + `PydanticAIDeprecationWarning`

### `Toolset` — Inline Toolset Injection via the Capability Chain

`Toolset` is a lightweight `AbstractCapability` subclass that injects any `AgentToolset` directly into the capability chain.  Unlike `FunctionToolset`, it does not manage tool decoration — it accepts any pre-built `AgentToolset` (including external or wrapped ones).

```python
# Key signatures verified from source (pydantic-ai 2.5.0):
#
# @dataclass
# class Toolset(AbstractCapability[AgentDepsT]):
#     toolset: AgentToolset[AgentDepsT]
#
#     @classmethod
#     def get_serialization_name(cls) -> str | None:
#         return None  # not spec-serializable (takes a callable)
#
#     def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
#         return self.toolset
```

### `PydanticAIDeprecationWarning`

All `pydantic-ai` deprecations are issued as `PydanticAIDeprecationWarning(UserWarning)` rather than `DeprecationWarning`.  The `UserWarning` base ensures warnings are visible by default at runtime (Python only shows `DeprecationWarning` in `__main__` and test runners).

```python
# Key signatures verified from source (pydantic-ai 2.5.0):
#
# class PydanticAIDeprecationWarning(UserWarning):
#     """Emitted when a deprecated Pydantic AI API is used."""
```

### 10.1 Injecting a Pre-Built Toolset via the `Toolset` Capability

```python
import asyncio
import os

from pydantic_ai import Agent
from pydantic_ai.capabilities.toolset import Toolset
from pydantic_ai.toolsets.function import FunctionToolset

os.environ.setdefault('OPENAI_API_KEY', 'your-key')


def get_weather(city: str) -> str:
    return f'Sunny and 22°C in {city}'


# Build the toolset independently
weather_toolset = FunctionToolset()
weather_toolset.add_function(get_weather)

# Inject via Toolset capability — no @agent.tool decorator needed
agent = Agent(
    'openai:gpt-5.2',
    capabilities=[Toolset(toolset=weather_toolset)],
)


async def main() -> None:
    result = await agent.run("What's the weather in Tokyo?")
    print(result.output)


asyncio.run(main())
```

### 10.2 Combining `Toolset` with Other Capabilities

```python
import asyncio
import os

from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking
from pydantic_ai.capabilities.toolset import Toolset
from pydantic_ai.toolsets.function import FunctionToolset

os.environ.setdefault('ANTHROPIC_API_KEY', 'your-key')


import ast
import operator as _op

_SAFE_OPS: dict = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
}


def _eval_node(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand)
    raise ValueError(f'Unsupported expression: {ast.dump(node)}')


def calculate(expression: str) -> float:
    tree = ast.parse(expression, mode='eval')
    return _eval_node(tree.body)


calc_toolset = FunctionToolset()
calc_toolset.add_function(calculate)

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    capabilities=[
        Thinking(level='low'),
        Toolset(toolset=calc_toolset),
    ],
)


async def main() -> None:
    result = await agent.run('What is (123 * 456) + 789?')
    print(result.output)


asyncio.run(main())
```

### 10.3 Catching and Filtering `PydanticAIDeprecationWarning`

```python
import warnings

from pydantic_ai._warnings import PydanticAIDeprecationWarning


def call_deprecated_api() -> None:
    # Simulate a deprecated call — PydanticAI internally does warnings.warn(... PydanticAIDeprecationWarning)
    warnings.warn(
        'This API is deprecated, use the new one instead.',
        PydanticAIDeprecationWarning,
        stacklevel=2,
    )


# Suppress pydantic-ai deprecations in production (not recommended for new code)
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=PydanticAIDeprecationWarning)
    call_deprecated_api()
    print('ran without deprecation warning')

# Turn all pydantic-ai deprecations into errors in tests
with warnings.catch_warnings():
    warnings.filterwarnings('error', category=PydanticAIDeprecationWarning)
    try:
        call_deprecated_api()
    except PydanticAIDeprecationWarning as e:
        print('caught deprecation:', e)
```

---

## Summary

| # | Class group | Key take-aways |
|---|---|---|
| 1 | `ZaiModel` / `ZaiModelSettings` / `ZaiProvider` / `ZaiModelProfile` | Z.AI GLM family with `zai_clear_thinking` preserved-thinking API and `zai_supports_reasoning_effort` on GLM-5.2 |
| 2 | `VercelProvider` | Vercel AI Gateway proxy — OIDC auth, `provider/model` naming, 8-provider profile dispatch |
| 3 | `Path` + `PathBuilder` | Fluent path builder — `to/broadcast/transform/map/label` chaining for the pydantic_graph builder API |
| 4 | `TransformFunction` + marker types + `PathItem` | Complete path item type system — discriminated union for path inspection and manual path construction |
| 5 | `EdgePath` + `EdgePathBuilder` | Finalised source-to-destination edge wiring — broadcast callback, chained map/transform |
| 6 | `NodeID` / `NodeRunID` / `TaskID` / `ForkStackItem` / `ForkStack` | `NewType`-based type-safe graph identifiers — placeholder generation and replacement |
| 7 | `MiddleNode/SourceNode/DestinationNode` + `ParentFork/ParentForkFinder` | Topology type guards + dominating-fork analysis for deadlock avoidance in parallel execution |
| 8 | `RuntimeToolsetKind` + `reject_unsupported_runtime_toolsets` | Per-run toolset guard for durable engines — FunctionToolset/MCPToolset/DynamicToolset rejected; ExternalToolset allowed |
| 9 | `ConcurrencyLimitedModel` + `get_concurrency_context` + `normalize_to_limiter` + `AnyConcurrencyLimit` | Model-layer HTTP rate limiting — shared pools, null-safe context helper, `AnyConcurrencyLimit` normalization |
| 10 | `Toolset` capability + `PydanticAIDeprecationWarning` | Inline toolset injection without decorator boilerplate + `UserWarning`-based deprecation visibility |

---
title: "Class deep dives — volume 37 (10 additional classes)"
description: "Source-verified deep dives into 10 more google-adk 2.3.0 classes: AgentOptimizer/Sampler/OptimizerResult protocol (optimization ABC backbone), SimplePromptOptimizer+config (iterative LLM prompt tuner), GEPARootAgentPromptOptimizerConfig+Result (GEPA adapter config + dynamic class creation), Skill+Frontmatter+Resources+Script (complete skill data model with validators), SkillRegistry ABC (custom registry authoring), LlmAgentConfig YAML DSL (full declarative agent config), AgentRefConfig+CodeConfig+ToolConfig+ToolArgsConfig (YAML DSL pieces), RetryConfig (exponential-backoff retry with jitter), AgentInfo+get_agents_dict+get_tools_info (agent introspection utilities), SequentialAgent+SequentialAgentState live mode (deprecated pipeline agent + migration)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 37"
  order: 106
---

import { Aside } from "@astrojs/starlight/components";

<Aside type="note">
All signatures, constants, and behaviours on this page were verified directly
against the installed package source (locate yours with
`python -c 'import google.adk; print(google.adk.__file__)'`) on
**google-adk == 2.3.0**. No documentation or blog posts were used as primary
sources.
</Aside>

---

## 1 · `AgentOptimizer` + `Sampler` + `OptimizerResult` + `UnstructuredSamplingResult`

**Source:** `google/adk/optimization/agent_optimizer.py`, `optimization/sampler.py`, `optimization/data_types.py`

These four classes form the abstract backbone of the ADK optimization framework.
Every concrete optimizer (e.g. `SimplePromptOptimizer`, `GEPARootAgentPromptOptimizer`)
subclasses `AgentOptimizer`; every evaluation harness subclasses `Sampler`.

### `AgentOptimizer` — the abstract optimizer ABC

```python
class AgentOptimizer(ABC, Generic[SamplingResultT, AgentWithScoresT]):
    @abstractmethod
    async def optimize(
        self,
        initial_agent: Agent,
        sampler: Sampler[SamplingResultT],
    ) -> OptimizerResult[AgentWithScoresT]: ...
```

A single `optimize()` method: takes the seed `Agent` and a `Sampler`, returns
an `OptimizerResult`. The two type parameters let the optimizer narrow the
types of its intermediate results (`SamplingResultT`) and final agents
(`AgentWithScoresT`).

### `Sampler` — the evaluation harness ABC

```python
class Sampler(ABC, Generic[SamplingResultT]):
    TRAIN_SET = "train"
    VALIDATION_SET = "validation"

    @abstractmethod
    def get_train_example_ids(self) -> list[str]: ...

    @abstractmethod
    def get_validation_example_ids(self) -> list[str]: ...

    @abstractmethod
    async def sample_and_score(
        self,
        candidate: Agent,
        example_set: Literal["train", "validation"] = "validation",
        batch: Optional[list[str]] = None,
        capture_full_eval_data: bool = False,
    ) -> SamplingResultT: ...
```

`get_train_example_ids` / `get_validation_example_ids` return opaque UID
strings that the optimizer passes back as `batch` slices.
`capture_full_eval_data=True` signals that the sampler should also capture
trajectories / outputs for reflection (used by GEPA).

### `OptimizerResult` + `UnstructuredSamplingResult`

```python
class SamplingResult(BaseModel):
    scores: dict[str, float]           # uid → scalar score, higher is better

class UnstructuredSamplingResult(SamplingResult):
    data: Optional[dict[str, dict[str, Any]]] = None  # uid → raw eval data

class AgentWithScores(BaseModel):
    optimized_agent: Agent
    overall_score: Optional[float] = None

class OptimizerResult(BaseModel, Generic[AgentWithScoresT]):
    optimized_agents: list[AgentWithScoresT]  # Pareto-front, not strictly ordered
```

`optimized_agents` is described in the docstring as "agents which cannot be
considered strictly better than one another" — i.e. a Pareto front. When there
is a single objective this list typically has one element.

### Example 1 — Minimal custom `Sampler` with a toy scoring function

```python
import asyncio
from typing import Literal, Optional
from google.adk.agents import LlmAgent
from google.adk.optimization.sampler import Sampler
from google.adk.optimization.data_types import SamplingResult

EXAMPLES = {
    "ex1": {"prompt": "Capital of France?", "expected": "Paris"},
    "ex2": {"prompt": "Capital of Japan?",  "expected": "Tokyo"},
    "ex3": {"prompt": "Capital of Brazil?", "expected": "Brasília"},
}

class CapitalQuizSampler(Sampler[SamplingResult]):
    def get_train_example_ids(self) -> list[str]:
        return ["ex1", "ex2"]

    def get_validation_example_ids(self) -> list[str]:
        return ["ex3"]

    async def sample_and_score(
        self,
        candidate: LlmAgent,
        example_set: Literal["train", "validation"] = "validation",
        batch: Optional[list[str]] = None,
        capture_full_eval_data: bool = False,
    ) -> SamplingResult:
        ids = (
            self.get_train_example_ids()
            if example_set == self.TRAIN_SET
            else self.get_validation_example_ids()
        )
        if batch:
            ids = [i for i in ids if i in batch]

        scores: dict[str, float] = {}
        for uid in ids:
            ex = EXAMPLES[uid]
            # Simplified: check if instruction mentions the expected answer
            hit = ex["expected"].lower() in candidate.instruction.lower()
            scores[uid] = 1.0 if hit else 0.0

        return SamplingResult(scores=scores)

sampler = CapitalQuizSampler()
print(sampler.get_train_example_ids())   # ['ex1', 'ex2']
print(sampler.TRAIN_SET)                  # 'train'
print(sampler.VALIDATION_SET)             # 'validation'
```

### Example 2 — `UnstructuredSamplingResult` for GEPA-style reflection data capture

```python
import asyncio
from typing import Literal, Optional, Any
from google.adk.agents import LlmAgent
from google.adk.optimization.sampler import Sampler
from google.adk.optimization.data_types import UnstructuredSamplingResult

class ReflectionSampler(Sampler[UnstructuredSamplingResult]):
    def get_train_example_ids(self) -> list[str]:
        return ["t1", "t2", "t3"]

    def get_validation_example_ids(self) -> list[str]:
        return ["v1"]

    async def sample_and_score(
        self,
        candidate: LlmAgent,
        example_set: Literal["train", "validation"] = "validation",
        batch: Optional[list[str]] = None,
        capture_full_eval_data: bool = False,
    ) -> UnstructuredSamplingResult:
        ids = batch or (
            self.get_train_example_ids()
            if example_set == self.TRAIN_SET
            else self.get_validation_example_ids()
        )

        scores: dict[str, float] = {}
        data: dict[str, dict[str, Any]] = {}

        for uid in ids:
            score = 0.75   # placeholder
            scores[uid] = score
            if capture_full_eval_data:
                # Captured trajectory/output goes here so the optimizer can
                # read it back during its reflection phase.
                data[uid] = {
                    "output": f"Agent answered uid={uid}",
                    "score": score,
                    "trajectory": [],
                }

        return UnstructuredSamplingResult(
            scores=scores,
            data=data if capture_full_eval_data else None,
        )
```

### Example 3 — Reading `OptimizerResult` with a custom `AgentWithScores`

```python
from google.adk.agents import LlmAgent
from google.adk.optimization.data_types import AgentWithScores, OptimizerResult

class ScoredAgent(AgentWithScores):
    """Extend to carry per-metric scores alongside the overall score."""
    per_metric: dict[str, float] = {}

agent_v1 = LlmAgent(name="agent_v1", model="gemini-2.5-flash",
                    instruction="You are a helpful assistant.")
agent_v2 = LlmAgent(name="agent_v2", model="gemini-2.5-flash",
                    instruction="Answer questions accurately and concisely.")

result: OptimizerResult[ScoredAgent] = OptimizerResult(
    optimized_agents=[
        ScoredAgent(
            optimized_agent=agent_v1,
            overall_score=0.72,
            per_metric={"accuracy": 0.72, "helpfulness": 0.80},
        ),
        ScoredAgent(
            optimized_agent=agent_v2,
            overall_score=0.78,
            per_metric={"accuracy": 0.78, "helpfulness": 0.75},
        ),
    ]
)

# Pareto front: pick the agent with the highest overall score
best = max(result.optimized_agents, key=lambda a: a.overall_score or 0)
print(f"Best agent: {best.optimized_agent.name}, score={best.overall_score}")
# Best agent: agent_v2, score=0.78
```

---

## 2 · `SimplePromptOptimizer` + `SimplePromptOptimizerConfig`

**Source:** `google/adk/optimization/simple_prompt_optimizer.py`

`SimplePromptOptimizer` is a gradient-free, LLM-driven iterative prompt
improver. Each round it:

1. Samples a mini-batch of training examples from the `Sampler`.
2. Scores the current best agent.
3. Feeds the score + current prompt into `_OPTIMIZER_PROMPT_TEMPLATE`.
4. Calls the optimizer LLM to generate a candidate prompt.
5. Scores the candidate on the mini-batch; replaces the best if it scores higher.

### `SimplePromptOptimizerConfig` constructor (verified `simple_prompt_optimizer.py`)

```python
class SimplePromptOptimizerConfig(BaseModel):
    optimizer_model: str = "gemini-2.5-flash"
    model_configuration: GenerateContentConfig = GenerateContentConfig(
        thinking_config=ThinkingConfig(include_thoughts=True, thinking_budget=10240)
    )
    num_iterations: int = 10     # total LLM-driven improvement rounds
    batch_size: int = 5          # training examples per scoring call
```

The `_OPTIMIZER_PROMPT_TEMPLATE` instructs the optimizer LLM to output **only
the improved prompt text** — no markdown or explanation — so the raw text is
used directly as the new `instruction`.

### Example 1 — Running `SimplePromptOptimizer` end to end

```python
import asyncio
from typing import Literal, Optional
from google.adk.agents import LlmAgent
from google.adk.optimization.simple_prompt_optimizer import (
    SimplePromptOptimizer,
    SimplePromptOptimizerConfig,
)
from google.adk.optimization.sampler import Sampler
from google.adk.optimization.data_types import UnstructuredSamplingResult

# Minimal sampler — replace sample_and_score with real eval logic
class MySampler(Sampler[UnstructuredSamplingResult]):
    _TRAIN = ["t1", "t2", "t3", "t4", "t5"]
    _VAL   = ["v1", "v2"]

    def get_train_example_ids(self) -> list[str]:
        return self._TRAIN

    def get_validation_example_ids(self) -> list[str]:
        return self._VAL

    async def sample_and_score(
        self,
        candidate: LlmAgent,
        example_set: Literal["train", "validation"] = "validation",
        batch: Optional[list[str]] = None,
        capture_full_eval_data: bool = False,
    ) -> UnstructuredSamplingResult:
        ids = batch or (self._TRAIN if example_set == "train" else self._VAL)
        # Replace this stub with real agent evaluation logic
        return UnstructuredSamplingResult(scores={uid: 0.6 for uid in ids})

async def main():
    initial_agent = LlmAgent(
        name="support_agent",
        model="gemini-2.5-flash",
        instruction="You are a customer support agent. Answer user questions.",
    )
    config = SimplePromptOptimizerConfig(
        num_iterations=3,   # keep small for demo; use 10+ in production
        batch_size=2,
        optimizer_model="gemini-2.5-flash",
    )
    optimizer = SimplePromptOptimizer(config=config)
    result = await optimizer.optimize(initial_agent, MySampler())
    best = result.optimized_agents[0]
    print("Score:", best.overall_score)
    print("Optimized instruction:", best.optimized_agent.instruction[:120])

# asyncio.run(main())
```

### Example 2 — Using a thinking budget to guide prompt improvement

```python
from google.genai import types as genai_types
from google.adk.optimization.simple_prompt_optimizer import SimplePromptOptimizerConfig

# Higher thinking_budget → more thorough prompt redesign,
# but more tokens spent per iteration.
config = SimplePromptOptimizerConfig(
    optimizer_model="gemini-2.5-pro",
    model_configuration=genai_types.GenerateContentConfig(
        thinking_config=genai_types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=32768,   # max budget for complex rewrites
        )
    ),
    num_iterations=15,
    batch_size=8,
)
print(config.num_iterations)    # 15
print(config.batch_size)        # 8
```

### Example 3 — Extracting the improved instruction after optimization

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.optimization.simple_prompt_optimizer import (
    SimplePromptOptimizer,
    SimplePromptOptimizerConfig,
)
from google.adk.optimization.data_types import OptimizerResult, AgentWithScores

# Assume 'result' was returned by optimizer.optimize(...)
# result: OptimizerResult[AgentWithScores]

# The optimized_agents list contains the Pareto front (usually 1 item).
# Each entry is an AgentWithScores with an .optimized_agent and .overall_score.

def extract_best_instruction(result: OptimizerResult[AgentWithScores]) -> str:
    """Pick the highest-scoring agent's instruction."""
    best = max(
        result.optimized_agents,
        key=lambda a: a.overall_score if a.overall_score is not None else -1,
    )
    return best.optimized_agent.instruction

# After running an optimizer you can persist the improved prompt:
# improved = extract_best_instruction(result)
# with open("optimized_instruction.txt", "w") as f:
#     f.write(improved)
print("extract_best_instruction defined — call it with an OptimizerResult")
```

---

## 3 · `GEPARootAgentPromptOptimizerConfig` + `GEPARootAgentPromptOptimizerResult`

**Source:** `google/adk/optimization/gepa_root_agent_prompt_optimizer.py`

GEPA (Guided Exploration Prompt Adaptation) is a more advanced optimizer that
relies on the external `gepa` library. The ADK exposes it through two config
and result classes, plus an `_AgentGEPAAdapter` created **dynamically** inside
`_create_agent_gepa_adapter_class()` to avoid hard-importing `gepa` at module
load time (keeping startup cost zero for users who don't install the extra).

### `GEPARootAgentPromptOptimizerConfig` (verified source)

```python
class GEPARootAgentPromptOptimizerConfig(BaseModel):
    optimizer_model: str = "gemini-2.5-flash"
    model_configuration: GenerateContentConfig = GenerateContentConfig(
        thinking_config=ThinkingConfig(include_thoughts=True, thinking_budget=10240)
    )
    max_metric_calls: int = 100        # total LLM evaluations budget
    reflection_minibatch_size: int = 3 # examples fed to the reflection step
    run_dir: Optional[str] = None      # checkpoint / artefact directory
```

`run_dir` enables mid-run recovery: the optimizer saves intermediate
`OptimizerResult` JSON files there and can resume from the last checkpoint.

### `GEPARootAgentPromptOptimizerResult`

```python
class GEPARootAgentPromptOptimizerResult(OptimizerResult[AgentWithScores]):
    gepa_result: Optional[dict[str, Any]] = None
```

Adds `gepa_result` — the raw result dictionary from the underlying GEPA library
— alongside the standard `optimized_agents` list. This dict contains GEPA
internals (e.g. Pareto-front history, iteration metadata) useful for debugging.

### Example 1 — Configuring GEPA with a checkpoint directory

```python
import os
from google.adk.optimization.gepa_root_agent_prompt_optimizer import (
    GEPARootAgentPromptOptimizerConfig,
)
from google.genai import types as genai_types

config = GEPARootAgentPromptOptimizerConfig(
    optimizer_model="gemini-2.5-pro",
    max_metric_calls=200,           # allow more evaluations for better results
    reflection_minibatch_size=5,
    run_dir="/tmp/gepa_run_20260707",  # enable checkpointing
    model_configuration=genai_types.GenerateContentConfig(
        thinking_config=genai_types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=16384,
        )
    ),
)
print(config.max_metric_calls)         # 200
print(config.reflection_minibatch_size)  # 5
print(config.run_dir)                  # /tmp/gepa_run_20260707
```

### Example 2 — Reading the raw GEPA result

```python
from google.adk.optimization.gepa_root_agent_prompt_optimizer import (
    GEPARootAgentPromptOptimizerResult,
)
from google.adk.optimization.data_types import AgentWithScores
from google.adk.agents import LlmAgent

agent = LlmAgent(name="a", model="gemini-2.5-flash", instruction="Draft.")

# Simulated result — in practice returned by GEPARootAgentPromptOptimizer.optimize()
result = GEPARootAgentPromptOptimizerResult(
    optimized_agents=[AgentWithScores(optimized_agent=agent, overall_score=0.85)],
    gepa_result={
        "iterations": 42,
        "pareto_history": [[0.72, 0.80, 0.85]],
        "best_prompt_index": 2,
    },
)

print(result.optimized_agents[0].overall_score)   # 0.85
# Access raw GEPA internals for debugging / plotting
if result.gepa_result:
    print("GEPA iterations:", result.gepa_result.get("iterations"))   # 42
```

### Example 3 — Dynamic adapter class creation pattern

```python
# The dynamic class creation lets ADK avoid a hard dependency on `gepa`.
# Internally the optimizer calls this at runtime:

def _create_agent_gepa_adapter_class():
    # Only imported when actually needed → zero startup cost when gepa absent.
    from gepa.core.adapter import EvaluationBatch, GEPAAdapter
    class _AgentGEPAAdapter(GEPAAdapter):
        ...
    return _AgentGEPAAdapter

# You can mirror this pattern in your own code for optional heavy dependencies:
def _lazy_load_optional_class():
    """Pattern: defer import of an optional dependency to call time."""
    try:
        import some_optional_lib
        class MyAdapter(some_optional_lib.BaseClass):
            pass
        return MyAdapter
    except ImportError:
        raise ImportError(
            "Install 'some_optional_lib' to use this feature: "
            "pip install some_optional_lib"
        )

# Using this pattern:
# AdapterClass = _lazy_load_optional_class()
# adapter = AdapterClass(...)
print("Lazy-load pattern demonstrated — works for any optional dependency")
```

---

## 4 · `Skill` + `Frontmatter` + `Resources` + `Script`

**Source:** `google/adk/skills/models.py`

The skills data model has three nested levels:

| Class | Level | Purpose |
|-------|-------|---------|
| `Frontmatter` | L1 | Discovery metadata (name, description, allowed-tools) |
| `Skill` | L1+L2 | Frontmatter + markdown instructions loaded on trigger |
| `Resources` | L3 | Additional instructions, assets, and runnable scripts |
| `Script` | — | Wrapper for a script's text content |

### `Frontmatter` field validators (verified `models.py`)

```python
# name: max 64 chars; must match _KEBAB_NAME_PATTERN = r"^[a-z0-9]+(-[a-z0-9]+)*$"
# When FeatureName.SNAKE_CASE_SKILL_NAME is enabled, underscores are also allowed.
# description: required, max 1024 chars.
# allowed_tools: space-delimited, serialization alias "allowed-tools" (YAML-friendly).
# metadata["adk_additional_tools"] must be a list[str] if present.
```

### Example 1 — Building a `Skill` with full metadata

```python
from google.adk.skills.models import Frontmatter, Skill, Resources, Script

frontmatter = Frontmatter(
    name="data-analysis",              # kebab-case, max 64 chars
    description=(
        "Provides statistical analysis and visualisation of tabular data. "
        "Use when the user asks for data summaries, charts, or correlations."
    ),
    license="Apache-2.0",
    compatibility="google-adk>=2.3.0",
    # allowed-tools is the YAML alias; use allowed_tools in Python
    allowed_tools="code_execution google_search",
    metadata={
        "adk_additional_tools": ["my_package.tools.pandas_tool"],
        "owner": "data-team",
    },
)

resources = Resources(
    references={
        "pandas_cheatsheet.md": "# Pandas Cheatsheet\n\n- `df.describe()` ...",
    },
    assets={
        "sample_schema.json": '{"columns": ["id", "value", "label"]}',
    },
    scripts={
        "validate.sh": Script(src="#!/bin/bash\npython -m pytest tests/"),
    },
)

skill = Skill(
    frontmatter=frontmatter,
    instructions="## Data Analysis Skill\n\nUse pandas to analyse tabular data...",
    resources=resources,
)

print(skill.name)                        # data-analysis
print(skill.description[:30])           # Provides statistical analysis...
print(resources.list_references())      # ['pandas_cheatsheet.md']
print(resources.list_scripts())         # ['validate.sh']
print(str(resources.scripts["validate.sh"]))  # #!/bin/bash\npython -m pytest tests/
```

### Example 2 — Validator behaviour: name and description constraints

```python
from pydantic import ValidationError
from google.adk.skills.models import Frontmatter

# Valid kebab-case name
fm = Frontmatter(name="my-skill-v2", description="Does something useful.")
print(fm.name)   # my-skill-v2

# Too long name — raises ValidationError
try:
    Frontmatter(name="a" * 65, description="x")
except ValidationError as e:
    print("Name too long:", "name must be at most 64" in str(e))  # True

# Description too long — max 1024 chars
try:
    Frontmatter(name="ok", description="x" * 1025)
except ValidationError as e:
    print("Description too long:", "1024" in str(e))  # True

# Invalid name (uppercase)
try:
    Frontmatter(name="My-Skill", description="x")
except ValidationError as e:
    print("Invalid name:", "kebab-case" in str(e))   # True
```

### Example 3 — `Resources` access helpers

```python
from google.adk.skills.models import Resources, Script

res = Resources(
    references={"guide.md": "# User Guide\n\nStep 1..."},
    assets={"schema.json": '{"type": "object"}', "logo.png": b"\x89PNG\r\n"},
    scripts={"setup.sh": Script(src="pip install deps")},
)

# Access helpers
print(res.get_reference("guide.md")[:10])   # # User Guide
print(res.get_reference("missing.md"))       # None
print(res.get_asset("schema.json"))          # {"type": "object"}
print(res.get_script("setup.sh").src)        # pip install deps
print(res.get_script("nope"))                # None

# List helpers
print(res.list_references())   # ['guide.md']
print(res.list_assets())       # ['schema.json', 'logo.png']
print(res.list_scripts())      # ['setup.sh']
```

---

## 5 · `SkillRegistry` ABC

**Source:** `google/adk/skills/skill_registry.py`

`SkillRegistry` is a pure abstract base class defining the interface for any
skill store — local filesystem, GCS, a remote registry service, etc.

### Interface (verified source)

```python
class SkillRegistry(ABC):
    @abstractmethod
    async def get_skill(self, *, name: str) -> Skill:
        # Raises Exception if skill not found.
        ...

    @abstractmethod
    async def search_skills(self, *, query: str) -> list[Frontmatter]:
        # Returns L1 metadata only — no full instructions loaded.
        ...

    def search_tool_description(self) -> str | None:
        # Override to customise the tool description shown to the LLM when
        # it calls the search_skills tool.
        return None
```

`search_skills` deliberately returns `list[Frontmatter]` — only the discovery
metadata — to keep token cost low. The agent then calls `get_skill(name=)` to
fetch the full instructions for the chosen skill.

`search_tool_description` is a hook to inject custom instructions into the
`search_skills` tool declaration. For example a GCS-backed registry might
explain the naming convention in use.

### Example 1 — In-memory `SkillRegistry` implementation

```python
import asyncio
from google.adk.skills.skill_registry import SkillRegistry
from google.adk.skills.models import Frontmatter, Skill, Resources

class InMemorySkillRegistry(SkillRegistry):
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    async def get_skill(self, *, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' not found")
        return self._skills[name]

    async def search_skills(self, *, query: str) -> list[Frontmatter]:
        q = query.lower()
        return [
            s.frontmatter
            for s in self._skills.values()
            if q in s.description.lower() or q in s.name
        ]

    def search_tool_description(self) -> str | None:
        return (
            "Search available skills by keyword. "
            "Returns a list of skill names and descriptions. "
            "Skills are named using kebab-case."
        )


async def main():
    registry = InMemorySkillRegistry()
    registry.register(Skill(
        frontmatter=Frontmatter(name="sql-query", description="Run SQL queries against BigQuery."),
        instructions="## SQL Query Skill\n\nUse BigQuery client to execute SQL.",
    ))
    registry.register(Skill(
        frontmatter=Frontmatter(name="pdf-summary", description="Summarise PDF documents."),
        instructions="## PDF Summary\n\nExtract text and summarise.",
    ))

    results = await registry.search_skills(query="sql")
    print([fm.name for fm in results])   # ['sql-query']

    skill = await registry.get_skill(name="sql-query")
    print(skill.instructions[:25])       # ## SQL Query Skill

asyncio.run(main())
```

### Example 2 — `SkillRegistry` with a custom search description

```python
from google.adk.skills.skill_registry import SkillRegistry
from google.adk.skills.models import Frontmatter, Skill

class TaggedSkillRegistry(SkillRegistry):
    """Registry that supports tag-based search."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._tags: dict[str, set[str]] = {}   # name → tags

    def register(self, skill: Skill, tags: list[str] | None = None) -> None:
        self._skills[skill.name] = skill
        self._tags[skill.name] = set(tags or [])

    async def get_skill(self, *, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"Unknown skill: {name!r}")
        return self._skills[name]

    async def search_skills(self, *, query: str) -> list[Frontmatter]:
        q = query.lower()
        return [
            s.frontmatter
            for name, s in self._skills.items()
            if q in s.description.lower()
            or q in name
            or q in self._tags.get(name, set())
        ]

    def search_tool_description(self) -> str | None:
        return (
            "Search skills by keyword or tag. "
            "Available tags include: data, ml, gcp, auth, storage. "
            "Example query: 'gcp storage' returns storage-related GCP skills."
        )

r = TaggedSkillRegistry()
r.register(
    Skill(frontmatter=Frontmatter(name="gcs-upload", description="Upload files to GCS."),
          instructions="Upload files."),
    tags=["gcp", "storage"],
)
print(r.search_tool_description()[:40])  # Search skills by keyword or tag.
```

### Example 3 — Error handling for `get_skill` with unknown name

```python
import asyncio
from google.adk.skills.skill_registry import SkillRegistry
from google.adk.skills.models import Frontmatter, Skill

class StrictRegistry(SkillRegistry):
    def __init__(self) -> None:
        self._store: dict[str, Skill] = {}

    def put(self, skill: Skill) -> None:
        self._store[skill.name] = skill

    async def get_skill(self, *, name: str) -> Skill:
        if name not in self._store:
            # The abstract contract says "Raises Exception if not found".
            # A KeyError or a custom SkillNotFoundError are both acceptable.
            raise LookupError(
                f"No skill named '{name}'. "
                f"Available: {sorted(self._store)}"
            )
        return self._store[name]

    async def search_skills(self, *, query: str) -> list[Frontmatter]:
        return [s.frontmatter for s in self._store.values()]


async def main():
    reg = StrictRegistry()
    try:
        await reg.get_skill(name="nonexistent")
    except LookupError as e:
        print(e)  # No skill named 'nonexistent'. Available: []

asyncio.run(main())
```

---

## 6 · `LlmAgentConfig` — full YAML declarative DSL

**Source:** `google/adk/agents/llm_agent_config.py`, `agents/common_configs.py`

`LlmAgentConfig` is the Pydantic model behind a `.yaml` agent file. It lets
you declare an entire `LlmAgent` declaratively. As of 2.3.0 it is
`@deprecated` in favour of writing Python code, but it remains fully
functional and is the only way to author agents via the `adk` CLI `--agent`
flag without Python.

### Key fields (verified `llm_agent_config.py`)

| Field | Type | Notes |
|-------|------|-------|
| `model` | `Optional[str]` | Model name string; mutually exclusive with `model_code` |
| `model_code` | `Optional[CodeConfig]` | FQN of a `BaseLlm` instance; mutually exclusive with `model` |
| `instruction` | `str` | Required; supports `{state_var}` template substitution |
| `static_instruction` | `Optional[ContentUnion]` | Static prefix sent before `instruction` |
| `tools` | `Optional[list[ToolConfig]]` | See Section 7 for the 5 patterns |
| `before_model_callbacks` | `Optional[list[CodeConfig]]` | Called before every LLM request |
| `after_model_callbacks` | `Optional[list[CodeConfig]]` | Called after every LLM response |
| `before_tool_callbacks` | `Optional[list[CodeConfig]]` | Called before each tool execution |
| `after_tool_callbacks` | `Optional[list[CodeConfig]]` | Called after each tool execution |
| `output_schema` | `Optional[CodeConfig]` | FQN of a Pydantic `BaseModel` |
| `output_key` | `Optional[str]` | Session state key to write the output to |
| `include_contents` | `Literal["default","none"]` | `"none"` suppresses prior conversation history |
| `generate_content_config` | `Optional[GenerateContentConfig]` | Gemini generation settings |
| `disallow_transfer_to_parent` | `Optional[bool]` | Block escalation to parent |
| `disallow_transfer_to_peers` | `Optional[bool]` | Block lateral transfers |

### Example 1 — Minimal YAML file and equivalent Python config

```python
# agent.yaml (write to disk with a YAML library and pass to adk CLI)
# -------
# agent_class: LlmAgent
# model: gemini-2.5-flash
# instruction: "You are a helpful assistant. Answer questions concisely."
# tools:
#   - name: google_search
# -------

# Equivalent Python construction via the (deprecated) Pydantic model:
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from google.adk.agents.llm_agent_config import LlmAgentConfig
    from google.adk.tools.tool_configs import ToolConfig, ToolArgsConfig

    cfg = LlmAgentConfig(
        instruction="You are a helpful assistant. Answer questions concisely.",
        model="gemini-2.5-flash",
        tools=[ToolConfig(name="google_search")],
    )

print(cfg.instruction[:30])      # You are a helpful assistant.
print(cfg.model)                 # gemini-2.5-flash
print(cfg.tools[0].name)         # google_search
print(cfg.include_contents)      # default
```

### Example 2 — YAML config with callbacks and output schema

```yaml
# full_agent.yaml
agent_class: LlmAgent
model: gemini-2.5-pro
instruction: "Classify the user's request and write the result to state."
output_key: classification_result
include_contents: none          # start fresh each invocation (no history)
disallow_transfer_to_parent: true

before_model_callbacks:
  - name: my_package.callbacks.log_request
after_model_callbacks:
  - name: my_package.callbacks.validate_response

tools:
  - name: my_package.tools.create_classifier_tool
    args:
      taxonomy_url: "gs://my-bucket/taxonomy.json"
      confidence_threshold: 0.85

sub_agents:
  - config_path: sub_agents/fallback_agent.yaml
```

```python
# Python equivalent (reading back the config)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from google.adk.agents.llm_agent_config import LlmAgentConfig
    from google.adk.agents.common_configs import CodeConfig
    from google.adk.tools.tool_configs import ToolConfig, ToolArgsConfig

    cfg = LlmAgentConfig(
        instruction="Classify the user's request.",
        model="gemini-2.5-pro",
        output_key="classification_result",
        include_contents="none",
        disallow_transfer_to_parent=True,
        before_model_callbacks=[
            CodeConfig(name="my_package.callbacks.log_request"),
        ],
        after_model_callbacks=[
            CodeConfig(name="my_package.callbacks.validate_response"),
        ],
        tools=[
            ToolConfig(
                name="my_package.tools.create_classifier_tool",
                args=ToolArgsConfig(
                    taxonomy_url="gs://my-bucket/taxonomy.json",
                    confidence_threshold=0.85,
                ),
            )
        ],
    )

print(cfg.output_key)               # classification_result
print(cfg.include_contents)         # none
print(cfg.disallow_transfer_to_parent)  # True
```

### Example 3 — `model` vs `model_code` mutex validation

```python
import warnings
from pydantic import ValidationError

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from google.adk.agents.llm_agent_config import LlmAgentConfig
    from google.adk.agents.common_configs import CodeConfig

    # model only — OK
    cfg = LlmAgentConfig(
        instruction="You are an agent.",
        model="gemini-2.5-flash",
    )
    print("model only:", cfg.model)  # gemini-2.5-flash

    # model_code only — OK (FQN points to a LiteLlm instance)
    cfg2 = LlmAgentConfig(
        instruction="You are an agent.",
        model_code=CodeConfig(name="my_pkg.models.gpt4o_litellm"),
    )
    print("model_code:", cfg2.model_code.name)  # my_pkg.models.gpt4o_litellm

    # Both set — raises ValueError
    try:
        LlmAgentConfig(
            instruction="You are an agent.",
            model="gemini-2.5-flash",
            model_code=CodeConfig(name="my_pkg.models.gpt4o_litellm"),
        )
    except ValidationError as e:
        print("Mutually exclusive:", "model_code" in str(e))   # True
```

---

## 7 · `AgentRefConfig` + `CodeConfig` + `ToolConfig` + `ToolArgsConfig`

**Source:** `google/adk/agents/common_configs.py`, `google/adk/tools/tool_configs.py`

These four classes are the building blocks of the `@experimental(AGENT_CONFIG)`
YAML DSL. Together they let you reference agents, code objects, and tools
declaratively.

### Class synopses (verified source)

```python
# CodeConfig — a fully qualified name pointing to a Python object
@experimental(FeatureName.AGENT_CONFIG)
class CodeConfig(BaseModel, extra="forbid"):
    name: str   # e.g. "my_pkg.tools.my_tool" or "google.adk.tools.google_search"

# AgentRefConfig — points to another agent via YAML file or Python code
@experimental(FeatureName.AGENT_CONFIG)
class AgentRefConfig(BaseModel, extra="forbid"):
    config_path: Optional[str] = None  # relative YAML path
    code: Optional[str] = None         # FQN of a Python agent instance
    # @model_validator enforces exactly one of config_path / code

# ToolArgsConfig — open key-value bag (extra="allow") for tool constructor args
@experimental(FeatureName.TOOL_CONFIG)
class ToolArgsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")  # any extra key accepted

# ToolConfig — wraps a tool reference with optional args
@experimental(FeatureName.TOOL_CONFIG)
class ToolConfig(BaseModel, extra="forbid"):
    name: str
    args: Optional[ToolArgsConfig] = None
```

### Example 1 — The five `ToolConfig` reference patterns

```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from google.adk.tools.tool_configs import ToolConfig, ToolArgsConfig

    # Pattern 1: ADK built-in tool by name
    t1 = ToolConfig(name="google_search")

    # Pattern 2: User-defined tool instance (FQN to an object)
    t2 = ToolConfig(name="my_pkg.tools.my_tool_instance")

    # Pattern 3: User-defined tool class with constructor args
    t3 = ToolConfig(
        name="my_pkg.tools.MyCustomTool",
        args=ToolArgsConfig(api_key="secret", timeout=30),
    )

    # Pattern 4: Factory function that returns a tool; args are function kwargs
    t4 = ToolConfig(
        name="my_pkg.tools.create_spanner_tool",
        args=ToolArgsConfig(project="my-project", instance="prod"),
    )

    # Pattern 5: Plain Python function (becomes a FunctionTool automatically)
    t5 = ToolConfig(name="my_pkg.tools.my_function")

for i, tc in enumerate([t1, t2, t3, t4, t5], 1):
    print(f"Pattern {i}: name={tc.name!r}, has_args={tc.args is not None}")
```

### Example 2 — `AgentRefConfig` one-of mutex

```python
import warnings
from pydantic import ValidationError

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from google.adk.agents.common_configs import AgentRefConfig

    # File-based reference
    ref1 = AgentRefConfig(config_path="sub_agents/search_agent.yaml")
    print(ref1.config_path)   # sub_agents/search_agent.yaml
    print(ref1.code)           # None

    # Code-based reference
    ref2 = AgentRefConfig(code="my_pkg.agents.custom_agent")
    print(ref2.code)           # my_pkg.agents.custom_agent

    # Neither set — raises
    try:
        AgentRefConfig()
    except ValidationError as e:
        print("Neither:", "Exactly one" in str(e))   # True

    # Both set — raises
    try:
        AgentRefConfig(config_path="a.yaml", code="my_pkg.agent")
    except ValidationError as e:
        print("Both:", "Only one" in str(e))   # True
```

### Example 3 — `ToolArgsConfig` accepting arbitrary extra fields

```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from google.adk.tools.tool_configs import ToolArgsConfig

    # ToolArgsConfig uses extra="allow", so any field is accepted
    args = ToolArgsConfig(
        project_id="my-gcp-project",
        dataset="sales",
        max_rows=1000,
        cache_ttl_seconds=300,
        labels={"env": "prod", "team": "data"},
    )

    # Access fields directly by attribute name
    print(args.project_id)            # my-gcp-project (type: str)
    print(args.max_rows)              # 1000
    print(args.model_extra["labels"]) # {'env': 'prod', 'team': 'data'}

    # model_dump() includes all extra fields
    d = args.model_dump()
    print(sorted(d.keys()))
    # ['cache_ttl_seconds', 'dataset', 'labels', 'max_rows', 'project_id']
```

---

## 8 · `RetryConfig` — workflow node retry semantics

**Source:** `google/adk/workflow/_retry_config.py`

`RetryConfig` controls exponential-backoff retry for workflow nodes. Pass it
to `@node(retry_config=...)`, `FunctionNode(retry_config=...)`, or embed it in
an `LlmAgentConfig.generate_content_config`-equivalent block.

### Constructor signature (verified source)

```python
class RetryConfig(BaseModel):
    max_attempts:   int | None   = None  # default: 5 (including 1st attempt)
    initial_delay:  float | None = None  # default: 1.0 s
    max_delay:      float | None = None  # default: 60.0 s
    backoff_factor: float | None = None  # default: 2.0
    jitter:         float | None = None  # default: 1.0 (= 100% randomness)
    exceptions:     list[str | type[BaseException]] | None = None  # default: all
```

The `exceptions` field validator (`_normalize_exceptions`) converts any
exception **class** to its `__name__` string. This means `exceptions=[ValueError]`
is equivalent to `exceptions=["ValueError"]` in the stored model.

Actual retry delay formula (from `_node_runner.py`):
`delay = min(initial_delay * backoff_factor^attempt, max_delay) * uniform(1, 1+jitter)`

### Example 1 — Basic retry with class-based exception filter

```python
from google.adk.workflow._retry_config import RetryConfig

# Retry on connection-related errors only; give up after 4 tries
config = RetryConfig(
    max_attempts=4,
    initial_delay=2.0,
    max_delay=30.0,
    backoff_factor=2.0,
    jitter=0.5,
    # Pass exception classes directly; the validator normalises to strings
    exceptions=[ConnectionError, TimeoutError],
)

print(config.max_attempts)    # 4
print(config.jitter)          # 0.5
# Validator converted class → string
print(config.exceptions)      # ['ConnectionError', 'TimeoutError']
```

### Example 2 — String-based exception names (YAML-friendly)

```python
from google.adk.workflow._retry_config import RetryConfig

# When loading from YAML you'll have strings, not class objects.
# Both forms are equivalent after normalisation.
config = RetryConfig(
    max_attempts=3,
    initial_delay=1.0,
    backoff_factor=3.0,
    max_delay=20.0,
    jitter=0.0,              # set to 0 to disable randomness (deterministic)
    exceptions=["ValueError", "RuntimeError"],
)
print(config.exceptions)    # ['ValueError', 'RuntimeError']
print(config.jitter)        # 0.0
```

### Example 3 — Attaching `RetryConfig` to a workflow `@node`

```python
from google.adk.workflow._retry_config import RetryConfig
from google.adk.workflow import Workflow

# Hypothetical — illustrates the @node usage pattern
retry_cfg = RetryConfig(
    max_attempts=5,
    initial_delay=1.0,
    max_delay=60.0,
    backoff_factor=2.0,
    jitter=1.0,  # full jitter (default)
)

# In a real workflow you'd pass retry_config to @node:
# from google.adk.workflow import node
#
# @node(retry_config=retry_cfg, timeout=120.0)
# async def fetch_data(ctx) -> dict:
#     ...

# Or when building a FunctionNode manually:
# from google.adk.workflow._function_node import FunctionNode
# fn = FunctionNode(func=fetch_data, retry_config=retry_cfg)

# Verify the config round-trips cleanly
d = retry_cfg.model_dump()
restored = RetryConfig(**d)
print(restored.max_attempts)    # 5
print(restored.backoff_factor)  # 2.0
```

---

## 9 · `AgentInfo` + `get_agents_dict` + `get_tools_info`

**Source:** `google/adk/utils/agent_info.py`

These utilities let you inspect the structure of an agent tree at runtime —
useful for debugging, documentation generation, or building a management UI.

### Class and function signatures (verified source)

```python
class AgentInfo(pydantic.BaseModel):
    name: str
    description: str
    instruction: str
    tools: list[types.Tool]       # google.genai Tool declarations
    sub_agents: list[str]         # names only (not AgentInfo objects)

async def get_tools_info(tools: list[ToolUnion]) -> list[types.Tool]:
    """Resolves BaseTool / BaseToolset / plain-function into Tool declarations."""
    ...

async def get_agents_dict(agent: LlmAgent) -> dict[str, AgentInfo]:
    """Recursively traverses agent tree → flat dict keyed by agent name."""
    ...
```

`get_tools_info` handles three tool shapes:
- `BaseTool` → `tool._get_declaration()`
- `BaseToolset` → `await toolset.get_tools()` then each tool's declaration
- plain callable → wraps in `FunctionTool`, then declaration

`get_agents_dict` does a DFS traversal and is safe against re-visiting the
same agent (name-based guard at the top of `_traverse`).

### Example 1 — Inspecting a single agent's tools

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.utils.agent_info import get_tools_info

def greet(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"

def farewell(name: str) -> str:
    """Say goodbye to a user."""
    return f"Goodbye, {name}!"

async def main():
    agent = LlmAgent(
        name="greeter",
        model="gemini-2.5-flash",
        instruction="Greet users warmly.",
        tools=[greet, farewell],
    )

    tool_infos = await get_tools_info(agent.tools)
    for t in tool_infos:
        for fd in t.function_declarations or []:
            print(f"  Tool: {fd.name} — {fd.description}")
    # Tool: greet — Greet a user by name.
    # Tool: farewell — Say goodbye to a user.

asyncio.run(main())
```

### Example 2 — `get_agents_dict` on a multi-agent tree

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.utils.agent_info import get_agents_dict

async def main():
    search_agent = LlmAgent(
        name="search",
        model="gemini-2.5-flash",
        instruction="Search the web.",
        description="Web search specialist.",
    )
    summariser_agent = LlmAgent(
        name="summariser",
        model="gemini-2.5-flash",
        instruction="Summarise documents.",
        description="Summarisation specialist.",
    )
    root = LlmAgent(
        name="orchestrator",
        model="gemini-2.5-pro",
        instruction="Orchestrate search and summarisation.",
        description="Root orchestrator.",
        sub_agents=[search_agent, summariser_agent],
    )

    info_dict = await get_agents_dict(root)
    print("Agents:", sorted(info_dict.keys()))
    # Agents: ['orchestrator', 'search', 'summariser']

    orch = info_dict["orchestrator"]
    print("Sub-agents:", orch.sub_agents)   # ['search', 'summariser']
    print("Instruction:", orch.instruction[:25])

asyncio.run(main())
```

### Example 3 — Using `AgentInfo` for documentation generation

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.utils.agent_info import get_agents_dict

def lookup_order(order_id: str) -> dict:
    """Look up an order by ID. Returns order details."""
    return {"id": order_id, "status": "shipped"}

async def main():
    agent = LlmAgent(
        name="support",
        model="gemini-2.5-flash",
        instruction="Help customers with their orders.",
        description="Customer support agent.",
        tools=[lookup_order],
    )
    info_map = await get_agents_dict(agent)

    # Generate a markdown summary
    lines = ["# Agent Catalogue\n"]
    for name, info in sorted(info_map.items()):
        lines.append(f"## {name}\n")
        lines.append(f"**Description:** {info.description}\n")
        lines.append(f"**Instruction:** {info.instruction[:80]}...\n")
        tool_names = [
            fd.name
            for t in info.tools
            for fd in (t.function_declarations or [])
        ]
        lines.append(f"**Tools:** {', '.join(tool_names) or 'none'}\n")
        lines.append(f"**Sub-agents:** {', '.join(info.sub_agents) or 'none'}\n")

    print("\n".join(lines))

asyncio.run(main())
```

---

## 10 · `SequentialAgent` + `SequentialAgentState` — pipeline agents and live mode

**Source:** `google/adk/agents/sequential_agent.py`

`SequentialAgent` (deprecated → use `Workflow`) runs its `sub_agents` one after
another. Its resumability feature is powered by `SequentialAgentState`, which
tracks the current sub-agent name between session turns.

### Key implementation facts (verified source)

```python
@experimental(FeatureName.AGENT_STATE)
class SequentialAgentState(BaseAgentState):
    current_sub_agent: str = ""
    # "" means the sequential run is complete (all sub-agents finished).

@deprecated("...use Workflow instead.")
class SequentialAgent(BaseAgent):
    async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
        agent_state = self._load_agent_state(ctx, SequentialAgentState)
        start_index = self._get_start_index(agent_state)
        # If a sub-agent was removed, _get_start_index logs a warning and
        # restarts from index 0 to avoid an IndexError.
        ...
```

The `_run_live_impl` override injects a `task_completed()` `FunctionTool` into
every `LlmAgent` sub-agent. The sub-agent calls it to signal it has finished,
allowing the sequential runner to move on to the next agent during a live
(bidirectional audio) session.

### Example 1 — Creating a three-stage pipeline with `SequentialAgent`

```python
import warnings
from google.adk.agents import LlmAgent

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from google.adk.agents.sequential_agent import SequentialAgent

    intake_agent = LlmAgent(
        name="intake",
        model="gemini-2.5-flash",
        instruction="Extract the user's name and request from the message.",
        output_key="user_info",
    )
    lookup_agent = LlmAgent(
        name="lookup",
        model="gemini-2.5-flash",
        instruction="Look up the account using the info in state['user_info'].",
        output_key="account_data",
    )
    response_agent = LlmAgent(
        name="responder",
        model="gemini-2.5-flash",
        instruction="Generate a response using state['account_data'].",
    )

    pipeline = SequentialAgent(
        name="support_pipeline",
        sub_agents=[intake_agent, lookup_agent, response_agent],
    )

print(f"Pipeline sub-agents: {[a.name for a in pipeline.sub_agents]}")
# Pipeline sub-agents: ['intake', 'lookup', 'responder']
```

### Example 2 — Migrating `SequentialAgent` to `Workflow`

```python
# OLD: SequentialAgent (deprecated)
# SequentialAgent(name="pipeline", sub_agents=[a, b, c])

# NEW: Workflow with sequential edges
from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow

def make_pipeline_workflow():
    intake = LlmAgent(
        name="intake",
        model="gemini-2.5-flash",
        instruction="Extract the user request.",
        output_key="parsed_request",
    )
    processor = LlmAgent(
        name="processor",
        model="gemini-2.5-flash",
        instruction="Process state['parsed_request'] and produce a result.",
        output_key="result",
    )
    responder = LlmAgent(
        name="responder",
        model="gemini-2.5-flash",
        instruction="Format state['result'] as a user-friendly response.",
    )

    # Workflow with explicit sequential routing
    wf = Workflow(
        name="support_pipeline",
        root_agent=intake,
    )
    # Sequential order is encoded as edges:
    # intake → processor → responder
    # In a Workflow, use the `route` return value or Trigger edges.
    return wf

wf = make_pipeline_workflow()
print(f"Workflow root: {wf.root_agent.name}")   # intake
```

### Example 3 — `SequentialAgentState` resumability and recovery

```python
# SequentialAgentState drives resumable execution for long-running pipelines.
# When a sub-agent is interrupted (e.g. HITL pause), the state persists
# current_sub_agent so the next invocation resumes from the right place.

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from google.adk.agents.sequential_agent import SequentialAgentState

    # Simulate inspecting state after resuming mid-pipeline
    state = SequentialAgentState(current_sub_agent="processor")
    print("Resuming at:", state.current_sub_agent)   # processor

    # "" means pipeline complete
    done_state = SequentialAgentState(current_sub_agent="")
    print("Is complete:", done_state.current_sub_agent == "")   # True

    # Sub-agent removed: _get_start_index returns 0 and logs a warning.
    # This prevents IndexError when a sub-agent is removed between sessions.
    # The agent restarts from the beginning (at-least-once guarantee).
    print(
        "Recovery: if the saved sub-agent name is no longer in the list, "
        "SequentialAgent._get_start_index returns 0 and logs a warning."
    )
```

---
title: "Class Deep Dives Vol. 4 — v2.9.2"
description: "Source-verified deep dives for 10 classes new or deepened in google-adk 2.9.2: SimplePromptOptimizer, GEPARootAgentOptimizer, Sampler, LocalEvalSampler, TelemetryConfig, UrlContextTool, Skill, SkillRegistry, SkillToolset, and VertexAiRagMemoryService."
framework: google-adk
language: python
sidebar:
  order: 130
---

All examples and field tables on this page are source-verified against **google-adk==2.9.2** (installed in a venv, introspected with `inspect.getsource`). The ten classes cover the brand-new **optimization module** (prompt engineering automation), per-request telemetry control, the Gemini URL-context tool, the skills subsystem, and RAG-backed memory — areas either absent from earlier guides or covered only superficially.

| # | Class / Symbol | Module | Subject |
|---|---|---|---|
| 1 | `SimplePromptOptimizer` + `SimplePromptOptimizerConfig` | `google.adk.optimization.simple_prompt_optimizer` | Iterative LLM-driven prompt tuning |
| 2 | `GEPARootAgentOptimizer` + `GEPARootAgentOptimizerConfig` | `google.adk.optimization.gepa_root_agent_optimizer` | GEPA-based prompt optimisation |
| 3 | `Sampler` | `google.adk.optimization.sampler` | Abstract optimisation data bridge |
| 4 | `LocalEvalSampler` | `google.adk.optimization.local_eval_sampler` | ADK eval-set → optimiser bridge |
| 5 | `TelemetryConfig` | `google.adk.telemetry.context` | Per-request OpenTelemetry overrides |
| 6 | `UrlContextTool` | `google.adk.tools.url_context_tool` | Gemini built-in URL-context grounding |
| 7 | `Skill` | `google.adk.skills.models` | Markdown-defined reusable agent skill |
| 8 | `SkillRegistry` | `google.adk.skills.skill_registry` | Abstract skill lookup / search |
| 9 | `SkillToolset` | `google.adk.tools.skill_toolset` | Full constructor deep-dive |
| 10 | `VertexAiRagMemoryService` | `google.adk.memory.vertex_ai_rag_memory_service` | Agent Platform RAG long-term memory |

---

## 1 — `SimplePromptOptimizer` + `SimplePromptOptimizerConfig`

**Module:** `google.adk.optimization.simple_prompt_optimizer`

`SimplePromptOptimizer` iteratively refines an `LlmAgent`'s `instruction` string. On each iteration it scores the current best prompt on a mini-batch drawn from the training split, asks the optimizer LLM for an improved prompt, scores the candidate, and keeps whichever scores higher.

### Config field reference

Source-verified from `google/adk/optimization/simple_prompt_optimizer.py`:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `optimizer_model` | `str` | `"gemini-2.5-flash"` | LLM that writes improved prompt candidates |
| `model_configuration` | `GenerateContentConfig` | `ThinkingConfig(include_thoughts=True, thinking_budget=10240)` | Generation config for the optimizer call |
| `num_iterations` | `int` | `10` | How many improvement rounds to run |
| `batch_size` | `int` | `5` | Training examples used per scoring call |

### How it works

```
initial_agent → baseline score on batch
    ↓
for i in range(num_iterations):
    ask optimizer_model: "here is the prompt and score, write a better one"
    clone agent with new instruction
    score clone on batch
    if clone_score > best_score: best = clone
    ↓
final `optimize()` validation run on full validation split
```

The optimizer calls `sampler.sample_and_score()` — so you only need to provide a `Sampler` implementation (see §3 and §4 below) and a dataset. No agent serving infrastructure is required.

### Minimal example — optimising a summarisation agent

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.optimization.simple_prompt_optimizer import (
    SimplePromptOptimizer,
    SimplePromptOptimizerConfig,
)
from google.adk.optimization.local_eval_sampler import LocalEvalSampler

# 1. Define the agent whose prompt you want to improve.
agent = LlmAgent(
    name="summariser",
    model="gemini-2.5-flash",
    instruction="Summarise the following text.",  # starting prompt — will be improved
)

# 2. Point to an ADK eval dataset on disk (see §4 for LocalEvalSampler details).
sampler = LocalEvalSampler(eval_set_file="evals/summarise_eval.json")

# 3. Configure the optimizer.
config = SimplePromptOptimizerConfig(
    optimizer_model="gemini-2.5-flash",
    num_iterations=8,   # run 8 refinement rounds
    batch_size=4,       # score each candidate on 4 examples
)
optimizer = SimplePromptOptimizer(config=config)

# 4. Run optimisation.
async def main():
    result = await optimizer.optimize(initial_agent=agent, sampler=sampler)

    print("Best prompt:\n", result.best_agent.instruction)
    print("Validation scores:", result.best_agent_with_scores.scores)

asyncio.run(main())
```

### Customising the optimizer LLM's thinking budget

```python
from google.genai import types as genai_types
from google.adk.optimization.simple_prompt_optimizer import SimplePromptOptimizerConfig

config = SimplePromptOptimizerConfig(
    optimizer_model="gemini-2.5-pro",
    model_configuration=genai_types.GenerateContentConfig(
        thinking_config=genai_types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=32768,  # higher budget → more deliberate rewrites
        )
    ),
    num_iterations=15,
    batch_size=10,
)
```

### Multi-agent root-prompt optimisation

Only the **root agent's** instruction is rewritten on each iteration. Sub-agents keep their original instructions throughout:

```python
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.optimization.simple_prompt_optimizer import (
    SimplePromptOptimizer,
    SimplePromptOptimizerConfig,
)

planner = LlmAgent(name="planner", model="gemini-2.5-flash",
                   instruction="Break the task into steps.")
executor = LlmAgent(name="executor", model="gemini-2.5-flash",
                    instruction="Execute each step in turn.")

root = SequentialAgent(
    name="pipeline",
    sub_agents=[planner, executor],
    # SequentialAgent has no `instruction` — wrap in an LlmAgent if needed:
)

# Or use an LlmAgent root:
root_llm = LlmAgent(
    name="orchestrator",
    model="gemini-2.5-flash",
    instruction="Coordinate planning and execution.",
    sub_agents=[planner, executor],
)

optimizer = SimplePromptOptimizer(
    config=SimplePromptOptimizerConfig(num_iterations=5, batch_size=3)
)
# The optimizer clones root_llm with a new `instruction` on each round;
# planner and executor's instructions are NOT modified.
```

---

## 2 — `GEPARootAgentOptimizer` + `GEPARootAgentOptimizerConfig`

**Module:** `google.adk.optimization.gepa_root_agent_optimizer`

`GEPARootAgentOptimizer` implements the **GEPA** (Guided Evolutionary Prompt Adaptation) framework. It uses evolutionary search with LLM-generated reflections to explore the prompt space more efficiently than the greedy approach in `SimplePromptOptimizer`. It requires the optional `gepa` package.

> **Experimental.** The class is decorated `@experimental`; the API may change in a minor release.

### Config field reference

Source-verified from `google/adk/optimization/gepa_root_agent_optimizer.py`:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `optimizer_model` | `str` | `"gemini-3.5-flash"` | LLM that generates reflections and new prompt variants |
| `model_configuration` | `GenerateContentConfig` | `ThinkingConfig(include_thoughts=True, thinking_level=ThinkingLevel.HIGH)` | Generation config for optimizer calls |
| `max_metric_calls` | `int` | `100` | Hard budget on total `sample_and_score` invocations |
| `reflection_minibatch_size` | `int` | `3` | Examples shown to the LLM when writing a reflection |
| `run_dir` | `str \| None` | `None` | Checkpoint directory. Set this to resume an interrupted run |

### Installing the optional `gepa` dependency

```bash
pip install gepa
```

### Basic usage

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.optimization.gepa_root_agent_optimizer import (
    GEPARootAgentOptimizer,
    GEPARootAgentOptimizerConfig,
)
from google.adk.optimization.local_eval_sampler import LocalEvalSampler

agent = LlmAgent(
    name="classifier",
    model="gemini-2.5-flash",
    instruction="Classify the sentiment of the text.",
)

sampler = LocalEvalSampler(eval_set_file="evals/sentiment_eval.json")

config = GEPARootAgentOptimizerConfig(
    optimizer_model="gemini-3.5-flash",
    max_metric_calls=50,
    reflection_minibatch_size=3,
    run_dir="./gepa_checkpoints",  # enable resumable runs
)
optimizer = GEPARootAgentOptimizer(config=config)

async def main():
    result = await optimizer.optimize(initial_agent=agent, sampler=sampler)
    print("Optimised prompt:\n", result.best_agent.instruction)

asyncio.run(main())
```

### Resuming from a checkpoint

If the run is interrupted (`KeyboardInterrupt`, quota exhaustion, etc.), re-run the exact same script. Because `run_dir` is set, the optimizer reads the checkpoint and resumes from where it stopped:

```python
config = GEPARootAgentOptimizerConfig(
    optimizer_model="gemini-3.5-flash",
    max_metric_calls=100,
    run_dir="./gepa_checkpoints",   # same directory as the interrupted run
)
# Calling optimizer.optimize() again continues from the latest checkpoint.
```

### Comparing both optimizers

| Aspect | `SimplePromptOptimizer` | `GEPARootAgentOptimizer` |
|---|---|---|
| Strategy | Greedy hill-climb | Evolutionary + reflections (GEPA) |
| Speed | Faster (fewer LLM calls) | Slower (broader exploration) |
| Extra dependency | None | `pip install gepa` |
| Resumable | No | Yes (with `run_dir`) |
| Best for | Quick wins, smaller datasets | Production-grade tuning |

---

## 3 — `Sampler`

**Module:** `google.adk.optimization.sampler`

`Sampler` is the **abstract bridge** between the optimization loop and your evaluation data. Both `SimplePromptOptimizer` and `GEPARootAgentOptimizer` call it to score prompt candidates without knowing anything about where the data lives.

### Abstract interface (source-verified)

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

SamplingResultT = TypeVar("SamplingResultT")

class Sampler(ABC, Generic[SamplingResultT]):

    TRAIN_SET = "train"
    VALIDATION_SET = "validation"

    @abstractmethod
    async def get_train_example_ids(self) -> list[str]: ...

    @abstractmethod
    async def get_validation_example_ids(self) -> list[str]: ...

    @abstractmethod
    async def sample_and_score(
        self,
        candidate: Agent,
        example_set: str,         # Sampler.TRAIN_SET or VALIDATION_SET
        batch: list[str],         # example IDs from get_*_example_ids()
        capture_full_eval_data: bool,
    ) -> SamplingResultT: ...
```

### Implementing a custom `Sampler`

Use a custom `Sampler` when your evaluation data lives somewhere other than a local ADK eval JSON file — for example a BigQuery table, a Firestore collection, or an in-memory test fixture:

```python
import asyncio
from dataclasses import dataclass, field
from google.adk.agents import LlmAgent
from google.adk.optimization.sampler import Sampler
from google.adk.runners import InMemoryRunner
from google.genai import types

@dataclass
class SimpleResult:
    scores: dict[str, float] = field(default_factory=dict)

class InMemorySampler(Sampler[SimpleResult]):
    """Scores a candidate agent on in-memory question/answer pairs."""

    def __init__(self, examples: dict[str, tuple[str, str]]):
        # examples = {id: (question, expected_answer)}
        self._examples = examples
        n = len(examples)
        ids = list(examples.keys())
        self._train_ids = ids[: n * 8 // 10]
        self._val_ids   = ids[n * 8 // 10 :]

    async def get_train_example_ids(self) -> list[str]:
        return self._train_ids

    async def get_validation_example_ids(self) -> list[str]:
        return self._val_ids

    async def sample_and_score(
        self,
        candidate: LlmAgent,
        example_set: str,
        batch: list[str],
        capture_full_eval_data: bool,
    ) -> SimpleResult:
        scores: dict[str, float] = {}
        runner = InMemoryRunner(agent=candidate, app_name="opt_eval")

        for ex_id in batch:
            question, expected = self._examples[ex_id]
            session = await runner.session_service.create_session(
                app_name="opt_eval", user_id="opt"
            )
            events = runner.run(
                user_id="opt",
                session_id=session.id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=question)],
                ),
            )
            answer = ""
            for event in events:
                if event.content and event.content.parts:
                    answer = "".join(p.text for p in event.content.parts if p.text)

            # Simple exact-match metric; replace with ROUGE, LLM-as-judge, etc.
            scores[ex_id] = 1.0 if expected.lower() in answer.lower() else 0.0

        return SimpleResult(scores=scores)

# Wire it up:
examples = {
    "q1": ("What is 2+2?", "4"),
    "q2": ("Capital of France?", "Paris"),
    "q3": ("Water formula?", "H2O"),
    "q4": ("Speed of light unit?", "m/s"),
    "q5": ("Python creator?", "Guido"),
}

sampler = InMemorySampler(examples)
agent = LlmAgent(name="qa", model="gemini-2.5-flash",
                 instruction="Answer concisely.")
```

---

## 4 — `LocalEvalSampler`

**Module:** `google.adk.optimization.local_eval_sampler`

`LocalEvalSampler` is the **built-in** `Sampler` that reads an ADK evaluation JSON file from disk and delegates scoring to ADK's own `LocalEvalService`. It is the quickest way to connect the optimization loop to an existing evaluation dataset.

### Dependencies

`LocalEvalSampler` requires optional packages:

```bash
pip install google-adk[eval]   # includes pandas, rouge-score, etc.
# or individually:
pip install pandas rouge-score google-cloud-aiplatform
```

### Constructor

```python
class LocalEvalSampler:
    def __init__(
        self,
        eval_set_file: str,          # path to an ADK eval JSON (see format below)
        # Internal fields are set automatically
    ): ...
```

### Eval file format

The eval JSON is the same format used by ADK's `adk eval` CLI command and `LocalEvalService`:

```json
[
  {
    "name": "example_001",
    "initial_session_state": {},
    "conversation": [
      {
        "user_content": {
          "parts": [{"text": "What is the capital of France?"}],
          "role": "user"
        },
        "expected_tool_use": [],
        "expected_intermediate_agent_responses": [],
        "reference": "Paris"
      }
    ]
  }
]
```

### End-to-end optimisation with `LocalEvalSampler`

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.optimization.simple_prompt_optimizer import (
    SimplePromptOptimizer,
    SimplePromptOptimizerConfig,
)
from google.adk.optimization.local_eval_sampler import LocalEvalSampler

agent = LlmAgent(
    name="geography_qa",
    model="gemini-2.5-flash",
    instruction="Answer geography questions accurately.",
)

sampler = LocalEvalSampler(eval_set_file="evals/geography.json")

optimizer = SimplePromptOptimizer(
    config=SimplePromptOptimizerConfig(
        num_iterations=5,
        batch_size=3,
    )
)

async def main():
    result = await optimizer.optimize(initial_agent=agent, sampler=sampler)
    optimised = result.best_agent
    print("Final prompt:", optimised.instruction)

    # Persist the improved agent for later use:
    with open("optimised_instruction.txt", "w") as f:
        f.write(optimised.instruction)

asyncio.run(main())
```

### Split sizes

`LocalEvalSampler` automatically divides the eval file examples into a training split (used during iterations) and a validation split (used for the final evaluation). The split is determined internally by `LocalEvalService`; for explicit control, implement `Sampler` directly (§3).

---

## 5 — `TelemetryConfig`

**Module:** `google.adk.telemetry.context`

`TelemetryConfig` is a **per-request** OpenTelemetry configuration object attached to `RunConfig.telemetry`. It lets a single deployment serve multiple tenants with different observability settings without restarting the process.

### Class definition (source-verified)

```python
class TelemetryConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    genai_semconv_stability_opt_in: str | None = None
    """Stability opt-in for GenAI semantic conventions.
    Options: 'database', 'http'. Maps to OTEL_SEMCONV_STABILITY_OPT_IN.
    Falls back to the env var value when None."""

    capture_message_content: bool | None = None
    """Whether to capture message content in telemetry spans.
    Falls back to GOOGLE_ADK_CAPTURE_MESSAGE_CONTENT env var when None."""

    adk_experimental_telemetry_opt_in: str | None = None
    """Opt-in to experimental ADK-specific semantic conventions.
    Falls back to ADK_EXPERIMENTAL_TELEMETRY_OPT_IN env var when None."""
```

### Precedence rules (source-verified)

The effective value for each field is resolved in this order:

```
ADK_TELEMETRY_IGNORE_RUN_CONFIG=1 (admin lock)
  → env-var value always wins; per-request override is ignored entirely
  ↓
per-request TelemetryConfig field (not None)
  → overrides the env-var value for this invocation only
  ↓
env-var value (OTEL_SEMCONV_STABILITY_OPT_IN, GOOGLE_ADK_CAPTURE_MESSAGE_CONTENT, …)
  ↓
built-in default (None / False)
```

Setting `ADK_TELEMETRY_IGNORE_RUN_CONFIG=1` is the operator's way to prevent tenants from changing telemetry behaviour.

### Field reference

| Field | Env-var fallback | Purpose |
|---|---|---|
| `genai_semconv_stability_opt_in` | `OTEL_SEMCONV_STABILITY_OPT_IN` | GenAI semantic conventions stability level (`"database"` or `"http"`) |
| `capture_message_content` | `GOOGLE_ADK_CAPTURE_MESSAGE_CONTENT` | Include full message text in OTel spans (PII risk) |
| `adk_experimental_telemetry_opt_in` | `ADK_EXPERIMENTAL_TELEMETRY_OPT_IN` | ADK-specific experimental semantic conventions |

### Attaching `TelemetryConfig` to a run

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner
from google.adk.telemetry.context import TelemetryConfig
from google.genai import types

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
)
runner = InMemoryRunner(agent=agent, app_name="telemetry_demo")

async def main():
    session = await runner.session_service.create_session(
        app_name="telemetry_demo", user_id="user1"
    )

    # Per-request: capture message content for this tenant.
    run_config = RunConfig(
        telemetry=TelemetryConfig(
            capture_message_content=True,
            genai_semconv_stability_opt_in="database",
        )
    )

    events = runner.run(
        user_id="user1",
        session_id=session.id,
        new_message=types.Content(
            role="user", parts=[types.Part(text="Hello!")]
        ),
        run_config=run_config,
    )
    for event in events:
        if event.content and event.content.parts:
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Multi-tenant pattern

```python
from google.adk.agents.run_config import RunConfig
from google.adk.telemetry.context import TelemetryConfig

TENANT_CONFIGS = {
    "tenant_a": TelemetryConfig(capture_message_content=True),
    "tenant_b": TelemetryConfig(capture_message_content=False),
}

def get_run_config(tenant_id: str) -> RunConfig:
    return RunConfig(telemetry=TENANT_CONFIGS.get(tenant_id))
```

### Operator lock-out

```bash
# Prevent any per-request override — all requests use env-var or default values.
export ADK_TELEMETRY_IGNORE_RUN_CONFIG=1
```

Once set, `RunConfig.telemetry` is silently ignored; the operator's env vars are always authoritative.

---

## 6 — `UrlContextTool`

**Module:** `google.adk.tools.url_context_tool`

`UrlContextTool` injects Gemini's **built-in URL-context** grounding capability into an agent. When the agent invokes this tool, it passes one or more URLs to the Gemini model, which fetches and grounds its response in the live content of those pages.

### How it works (source-verified)

```python
class UrlContextTool(BaseTool):
    """Allows the agent to fetch and use content from URLs."""

    def __init__(self):
        super().__init__(name="url_context", description="...")

    def _get_declaration(self) -> types.FunctionDeclaration | None:
        return None  # no function declaration; uses built-in Gemini tool

    async def process_llm_request(
        self,
        *,
        tool_context: ToolContext,
        llm_request: LlmRequest,
    ) -> None:
        _check_gemini_model(tool_context)  # raises ValueError for non-Gemini
        llm_request.config = llm_request.config or types.GenerateContentConfig()
        llm_request.config.tools = llm_request.config.tools or []
        llm_request.config.tools.append(
            types.Tool(url_context=types.UrlContext())
        )
```

`UrlContextTool` has **no Python-side function** — it works entirely at the LLM request level by appending a `types.Tool(url_context=types.UrlContext())` entry.

### Restrictions

- **Gemini models only.** A `ValueError` is raised for non-Gemini models unless `ADK_DISABLE_GEMINI_MODEL_ID_CHECK=1` is set.
- Grounding is done server-side by Gemini; the URLs must be publicly accessible.
- Works alongside other tools in the same agent.

### Basic usage

```python
from google.adk.agents import LlmAgent
from google.adk.tools.url_context_tool import UrlContextTool
from google.adk.runners import InMemoryRunner
from google.genai import types
import asyncio

agent = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",   # must be a Gemini model
    instruction=(
        "You are a research assistant. When asked about a URL, "
        "use the url_context tool to read it and summarise the content."
    ),
    tools=[UrlContextTool()],
)

runner = InMemoryRunner(agent=agent, app_name="url_demo")

async def main():
    session = await runner.session_service.create_session(
        app_name="url_demo", user_id="user1"
    )
    events = runner.run(
        user_id="user1",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(
                text="Please summarise https://adk.google.dev/api/python/google/adk/agents/LlmAgent"
            )],
        ),
    )
    for event in events:
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

asyncio.run(main())
```

### Combining with Google Search grounding

`UrlContextTool` and `google_search` can coexist in the same agent:

```python
from google.adk.agents import LlmAgent
from google.adk.tools.url_context_tool import UrlContextTool
from google.adk.tools import google_search  # built-in

agent = LlmAgent(
    name="deep_researcher",
    model="gemini-2.5-flash",
    instruction=(
        "Use google_search to find relevant sources, then url_context "
        "to read the most promising ones in full."
    ),
    tools=[google_search, UrlContextTool()],
)
```

### Testing with non-Gemini models

```bash
# Bypass the Gemini model check during development/testing:
export ADK_DISABLE_GEMINI_MODEL_ID_CHECK=1
```

---

## 7 — `Skill`

**Module:** `google.adk.skills.models`

`Skill` is a **Pydantic model** that represents a single self-contained capability defined in a Markdown file. It stores the raw frontmatter, the instructions body, and any linked resources. `SkillToolset` instantiates a `Skill` for each `.md` file in its skills folder.

### Class structure (source-verified)

```python
class Skill(BaseModel):
    frontmatter: dict[str, Any]   # parsed YAML front matter
    instructions: str             # Markdown body (the actual skill instructions)
    resources: list[Resource]     # linked files / tools / sub-skills

    @property
    def name(self) -> str:
        return self.frontmatter.get("name", "")

    @property
    def description(self) -> str:
        return self.frontmatter.get("description", "")
```

### Skill Markdown format

A valid skill file looks like this:

```markdown
---
name: write_unit_test
description: Generate a pytest unit test for a given Python function.
version: "1.0"
---

You are a Python testing expert.

Given a Python function, write a complete pytest unit test that:
1. Tests the happy path with typical inputs.
2. Tests edge cases (empty input, boundary values).
3. Uses descriptive test names (`test_<function>_<scenario>`).

Return only the test code, no explanation.
```

### Reading a skill programmatically

```python
from pathlib import Path
from google.adk.skills.models import Skill
import yaml

def load_skill(path: str) -> Skill:
    text = Path(path).read_text()
    if text.startswith("---"):
        _, fm_block, body = text.split("---", 2)
        frontmatter = yaml.safe_load(fm_block)
        instructions = body.strip()
    else:
        frontmatter = {}
        instructions = text.strip()

    return Skill(frontmatter=frontmatter, instructions=instructions, resources=[])

skill = load_skill("skills/write_unit_test.md")
print(skill.name)         # "write_unit_test"
print(skill.description)  # "Generate a pytest unit test ..."
print(skill.instructions[:80])
```

### Inspecting skills loaded by `SkillToolset`

```python
from google.adk.tools.skill_toolset import SkillToolset

toolset = SkillToolset(skills_folder="/abs/path/to/skills")
tools = await toolset.get_tools()   # each tool wraps one Skill

for tool in tools:
    print(f"{tool.name}: {tool.description}")
    # tool._skill is the underlying Skill instance
    if hasattr(tool, "_skill"):
        print("  instructions:", tool._skill.instructions[:60])
```

---

## 8 — `SkillRegistry`

**Module:** `google.adk.skills.skill_registry`

`SkillRegistry` is the **abstract base** for skill lookup backends. Implement it when your skills are stored centrally (a database, a GCS bucket, a remote registry) and you want `SkillToolset` to discover them at runtime rather than reading from a local folder.

### Abstract interface (source-verified)

```python
class SkillRegistry(ABC):

    @abstractmethod
    async def get_skill(self, name: str) -> Skill | None:
        """Returns the Skill with the given name, or None if not found."""
        ...

    @abstractmethod
    async def search_skills(self, query: str) -> list[Skill]:
        """Returns skills whose name/description match the query."""
        ...

    def search_tool_description(self) -> str:
        """Human-readable description of how to search this registry.
        Included in the search tool's Gemini function declaration."""
        return "Search skills by name or description."
```

### Implementing a GCS-backed registry

```python
import json
import asyncio
from google.cloud import storage
from google.adk.skills.skill_registry import SkillRegistry
from google.adk.skills.models import Skill

class GcsSkillRegistry(SkillRegistry):
    """Reads skills from JSON objects in a GCS bucket."""

    def __init__(self, bucket_name: str, prefix: str = "skills/"):
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket_name)
        self._prefix = prefix
        self._cache: dict[str, Skill] | None = None

    async def _load_all(self) -> dict[str, Skill]:
        if self._cache is not None:
            return self._cache
        skills: dict[str, Skill] = {}
        for blob in self._bucket.list_blobs(prefix=self._prefix):
            if not blob.name.endswith(".json"):
                continue
            data = json.loads(blob.download_as_text())
            skill = Skill(**data)
            skills[skill.name] = skill
        self._cache = skills
        return skills

    async def get_skill(self, name: str) -> Skill | None:
        all_skills = await self._load_all()
        return all_skills.get(name)

    async def search_skills(self, query: str) -> list[Skill]:
        all_skills = await self._load_all()
        q = query.lower()
        return [
            s for s in all_skills.values()
            if q in s.name.lower() or q in s.description.lower()
        ]

    def search_tool_description(self) -> str:
        return f"Search skills stored in GCS bucket '{self._bucket.name}'."

# Wire into SkillToolset:
from google.adk.tools.skill_toolset import SkillToolset

registry = GcsSkillRegistry(bucket_name="my-skills-bucket")
toolset = SkillToolset(registry=registry)
```

### In-memory registry for testing

```python
from google.adk.skills.skill_registry import SkillRegistry
from google.adk.skills.models import Skill

class InMemorySkillRegistry(SkillRegistry):
    def __init__(self, skills: list[Skill]):
        self._skills = {s.name: s for s in skills}

    async def get_skill(self, name: str) -> Skill | None:
        return self._skills.get(name)

    async def search_skills(self, query: str) -> list[Skill]:
        q = query.lower()
        return [s for s in self._skills.values()
                if q in s.name.lower() or q in s.description.lower()]

test_skill = Skill(
    frontmatter={"name": "greet", "description": "Greet the user warmly."},
    instructions="Always start with 'Hello!' and use the user's name.",
    resources=[],
)
registry = InMemorySkillRegistry([test_skill])
```

---

## 9 — `SkillToolset` (deep dive)

**Module:** `google.adk.tools.skill_toolset`

`SkillToolset` is the **official way to attach skill files to an agent**. Each `.md` file in the skills folder becomes an independent tool whose description is the skill's frontmatter `description` and whose implementation runs the `instructions` as a sub-prompt. Source-verified constructor signature:

```python
class SkillToolset(BaseToolset):
    def __init__(
        self,
        skills: list[Skill] | None = None,
        registry: SkillRegistry | None = None,
        code_executor: BaseCodeExecutor | None = None,
        environment: BaseEnvironment | None = None,
        skills_folder: str | None = None,        # must be absolute when environment is set
        script_timeout: int = 300,               # seconds
        additional_tools: list[BaseTool] | None = None,
        tool_name_prefix: str | None = None,
        tool_filter: list[str] | Callable | None = None,
    ): ...
```

### Constructor field reference

| Field | Type | Default | Notes |
|---|---|---|---|
| `skills` | `list[Skill] \| None` | `None` | Pre-constructed `Skill` objects to expose |
| `registry` | `SkillRegistry \| None` | `None` | Remote/custom skill discovery backend |
| `code_executor` | `BaseCodeExecutor \| None` | `None` | Executor for code blocks inside skills |
| `environment` | `BaseEnvironment \| None` | `None` | Execution environment for code skills |
| `skills_folder` | `str \| None` | `None` | **Must be an absolute path when `environment` is set** |
| `script_timeout` | `int` | `300` | Max seconds for a skill's code block to run |
| `additional_tools` | `list[BaseTool] \| None` | `None` | Extra tools available to the skill's sub-agent |
| `tool_name_prefix` | `str \| None` | `None` | String prepended to every skill tool's name |
| `tool_filter` | `list[str] \| Callable \| None` | `None` | Allowlist of skill names or a predicate |

### Loading from a skills folder

```python
from google.adk.agents import LlmAgent
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.runners import InMemoryRunner
from google.genai import types
import asyncio

# skills/ must contain one .md file per skill
toolset = SkillToolset(skills_folder="/abs/path/to/skills")

agent = LlmAgent(
    name="multi_skill_agent",
    model="gemini-2.5-flash",
    instruction="Use the available skills to help the user.",
    tools=[toolset],
)
```

### Filtering which skills are exposed

```python
# Allowlist: only expose specific skills
toolset = SkillToolset(
    skills_folder="/abs/path/to/skills",
    tool_filter=["write_unit_test", "review_pr"],
)

# Predicate: expose skills based on a dynamic condition
def only_prod_skills(skill_name: str) -> bool:
    return not skill_name.startswith("experimental_")

toolset = SkillToolset(
    skills_folder="/abs/path/to/skills",
    tool_filter=only_prod_skills,
)
```

### Namespacing skill tool names

```python
# Avoids name collisions when merging skill toolsets from different domains
coding_toolset = SkillToolset(
    skills_folder="/skills/coding",
    tool_name_prefix="coding__",
)
writing_toolset = SkillToolset(
    skills_folder="/skills/writing",
    tool_name_prefix="writing__",
)

agent = LlmAgent(
    name="super_agent",
    model="gemini-2.5-flash",
    instruction="Help the user with coding and writing tasks.",
    tools=[coding_toolset, writing_toolset],
)
# Agent now has tools: coding__write_unit_test, writing__draft_email, etc.
```

### Registry + code executor combo

```python
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.tools.skill_toolset import SkillToolset

# For skills that contain runnable code blocks
toolset = SkillToolset(
    registry=registry,             # custom SkillRegistry from §8
    code_executor=BuiltInCodeExecutor(),
    script_timeout=120,
    additional_tools=[my_api_tool],  # available inside the skill's sub-agent
)
```

### Skills folder with a sandboxed environment

When `environment` is set, `skills_folder` **must be an absolute path** — relative paths silently resolve against the wrong working directory inside the environment:

```python
import os
from google.adk.tools.skill_toolset import SkillToolset

# BAD — relative path will break when environment is set:
# toolset = SkillToolset(skills_folder="skills", environment=my_env)

# GOOD — use an absolute path:
toolset = SkillToolset(
    skills_folder=os.path.abspath("skills"),
    environment=my_env,
    code_executor=my_executor,
)
```

---

## 10 — `VertexAiRagMemoryService`

**Module:** `google.adk.memory.vertex_ai_rag_memory_service`

`VertexAiRagMemoryService` stores conversation history in an **Agent Platform RAG corpus** (via the `agentplatform` package). At the end of a session, `add_session_to_memory()` serialises all events to a temporary text file and uploads it to the RAG corpus. `search_memory()` queries the corpus with semantic similarity and returns the most relevant past exchanges.

### Prerequisites

```bash
pip install google-cloud-aiplatform   # provides the `agentplatform` package
```

Create a RAG corpus in Google Cloud:

```bash
gcloud ai rag-corpora create \
    --display-name="agent-memory" \
    --location=us-central1 \
    --project=my-project
# Note the returned corpus resource name.
```

### Constructor (source-verified)

```python
VertexAiRagMemoryService(
    rag_corpus: str | None = None,
    similarity_top_k: int | None = None,
    vector_distance_threshold: float = 10,
    project: str | None = None,
    location: str | None = None,
)
```

| Argument | Type | Default | Notes |
|---|---|---|---|
| `rag_corpus` | `str \| None` | `None` | Corpus ID or full resource name (`projects/…/ragCorpora/…`) |
| `similarity_top_k` | `int \| None` | `None` | Max retrieved context chunks; `None` → RAG API default |
| `vector_distance_threshold` | `float` | `10` | Maximum vector distance; lower = stricter relevance |
| `project` | `str \| None` | `None` | Falls back to `GOOGLE_CLOUD_PROJECT` env var |
| `location` | `str \| None` | `None` | Falls back to `GOOGLE_CLOUD_LOCATION` env var |

If `rag_corpus` is the full resource name and `project`/`location` are not set, they are parsed automatically from the name.

### Important: `agentplatform` not `vertexai.preview.rag`

Previous versions of ADK used `vertexai.preview.rag`. That import is **deprecated**. The current implementation imports `agentplatform` (installed via `google-cloud-aiplatform ≥ 1.87`). If you see a deprecation warning, upgrade:

```bash
pip install --upgrade google-cloud-aiplatform
```

### Basic setup

```python
import os
from google.adk.memory.vertex_ai_rag_memory_service import VertexAiRagMemoryService

# Using env vars for project/location:
os.environ["GOOGLE_CLOUD_PROJECT"] = "my-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

memory_service = VertexAiRagMemoryService(
    rag_corpus="projects/my-project/locations/us-central1/ragCorpora/1234567890",
    similarity_top_k=5,
    vector_distance_threshold=0.7,  # lower = more relevant results only
)
```

### Attaching to a `Runner` (full wiring)

```python
import asyncio
import os
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.memory.vertex_ai_rag_memory_service import VertexAiRagMemoryService
from google.genai import types

RAG_CORPUS = "projects/my-project/locations/us-central1/ragCorpora/1234567890"

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction=(
        "You are a persistent assistant. Use your memory to recall "
        "context from previous conversations."
    ),
)

memory_service = VertexAiRagMemoryService(
    rag_corpus=RAG_CORPUS,
    similarity_top_k=5,
    vector_distance_threshold=0.8,
)

runner = Runner(
    agent=agent,
    app_name="persistent_assistant",
    session_service=DatabaseSessionService(db_url="sqlite:///sessions.db"),
    memory_service=memory_service,
)

async def chat(user_id: str, message: str, session_id: str | None = None):
    session = await runner.session_service.create_session(
        app_name="persistent_assistant",
        user_id=user_id,
    ) if session_id is None else await runner.session_service.get_session(
        app_name="persistent_assistant",
        user_id=user_id,
        session_id=session_id,
    )

    events = runner.run(
        user_id=user_id,
        session_id=session.id,
        new_message=types.Content(
            role="user", parts=[types.Part(text=message)]
        ),
    )
    response = ""
    for event in events:
        if event.is_final_response() and event.content:
            response = "".join(
                p.text for p in event.content.parts if p.text
            )
    return response, session.id
```

### Enabling the `load_memory` tool

For the agent to proactively query memory, add `PreloadMemoryTool` (covered in the Vol. 2 deep-dives) or let the runner automatically inject memory at session start:

```python
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="You are a persistent assistant.",
    tools=[PreloadMemoryTool()],  # agent can call load_memory() explicitly
)
```

### Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: No module named 'agentplatform'` | `pip install --upgrade google-cloud-aiplatform` |
| `ValueError: rag_corpus must be set` | Pass the full corpus resource name or set corpus on every `rag_resource` |
| `DeprecationWarning: vertexai.preview.rag` | Already on the new `agentplatform` path; warning means mixed install — upgrade `google-cloud-aiplatform` |
| High latency on `search_memory` | Reduce `similarity_top_k` or increase `vector_distance_threshold` to fetch fewer chunks |
| Stale data returned | `add_session_to_memory()` is called after the session ends; in-flight sessions are not yet indexed |

---

## Summary table

| Class | When to use |
|---|---|
| `SimplePromptOptimizer` | Fast, iterative prompt tuning with a low call budget |
| `GEPARootAgentOptimizer` | Production-grade evolutionary prompt search with checkpointing |
| `Sampler` | Custom data source for optimisation (BigQuery, Firestore, in-memory) |
| `LocalEvalSampler` | Quickest path: use an existing ADK eval JSON file as optimisation data |
| `TelemetryConfig` | Per-request OTel settings in multi-tenant deployments |
| `UrlContextTool` | Let Gemini read live web pages without writing a custom tool |
| `Skill` | Inspect or programmatically create skill objects from Markdown |
| `SkillRegistry` | Custom skill discovery backend (remote registry, database) |
| `SkillToolset` | Attach a folder or registry of skills as tools to any `LlmAgent` |
| `VertexAiRagMemoryService` | Production long-term memory backed by Agent Platform RAG |

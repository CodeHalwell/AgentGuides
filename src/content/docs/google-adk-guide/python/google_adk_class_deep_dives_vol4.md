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
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.local_eval_sets_manager import LocalEvalSetsManager
from google.adk.optimization.simple_prompt_optimizer import (
    SimplePromptOptimizer,
    SimplePromptOptimizerConfig,
)
from google.adk.optimization.local_eval_sampler import (
    LocalEvalSampler,
    LocalEvalSamplerConfig,
)

# 1. Define the agent whose prompt you want to improve.
agent = LlmAgent(
    name="summariser",
    model="gemini-2.5-flash",
    instruction="Summarise the following text.",  # starting prompt — will be improved
)

# 2. Wire up the sampler (see §4 for LocalEvalSampler details).
sampler = LocalEvalSampler(
    config=LocalEvalSamplerConfig(
        eval_config=EvalConfig(criteria={"response_match_score": 0.5}),
        app_name="summariser",
        train_eval_set="summarise_train",
    ),
    eval_sets_manager=LocalEvalSetsManager(agents_dir="./agents"),
)

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

    # OptimizerResult.optimized_agents is a list (Pareto front); for
    # SimplePromptOptimizer it always contains exactly one entry.
    best = result.optimized_agents[0]
    print("Best prompt:\n", best.optimized_agent.instruction)
    print("Validation score:", best.overall_score)

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
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.local_eval_sets_manager import LocalEvalSetsManager
from google.adk.optimization.local_eval_sampler import LocalEvalSampler, LocalEvalSamplerConfig

agent = LlmAgent(
    name="classifier",
    model="gemini-2.5-flash",
    instruction="Classify the sentiment of the text.",
)

sampler = LocalEvalSampler(
    config=LocalEvalSamplerConfig(
        eval_config=EvalConfig(criteria={"response_match_score": 0.5}),
        app_name="classifier",
        train_eval_set="sentiment_train",
    ),
    eval_sets_manager=LocalEvalSetsManager(agents_dir="./agents"),
)

config = GEPARootAgentOptimizerConfig(
    optimizer_model="gemini-3.5-flash",
    max_metric_calls=50,
    reflection_minibatch_size=3,
    run_dir="./gepa_checkpoints",  # enable resumable runs
)
optimizer = GEPARootAgentOptimizer(config=config)

async def main():
    result = await optimizer.optimize(initial_agent=agent, sampler=sampler)
    # GEPARootAgentOptimizerResult.optimized_agents is a Pareto front list.
    best = max(result.optimized_agents, key=lambda a: a.overall_score or 0.0)
    print("Optimised prompt:\n", best.optimized_agent.instruction)

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
    def get_train_example_ids(self) -> list[str]: ...      # sync

    @abstractmethod
    def get_validation_example_ids(self) -> list[str]: ... # sync

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
from google.adk.agents import LlmAgent
from google.adk.optimization.sampler import Sampler
from google.adk.optimization.data_types import UnstructuredSamplingResult
from google.adk.runners import InMemoryRunner
from google.genai import types

class InMemorySampler(Sampler[UnstructuredSamplingResult]):
    """Scores a candidate agent on in-memory question/answer pairs."""

    def __init__(self, examples: dict[str, tuple[str, str]]):
        # examples = {id: (question, expected_answer)}
        self._examples = examples
        n = len(examples)
        ids = list(examples.keys())
        self._train_ids = ids[: n * 8 // 10]
        self._val_ids   = ids[n * 8 // 10 :]

    def get_train_example_ids(self) -> list[str]:
        return self._train_ids

    def get_validation_example_ids(self) -> list[str]:
        return self._val_ids

    async def sample_and_score(
        self,
        candidate: LlmAgent,
        example_set: str,
        batch: list[str],
        capture_full_eval_data: bool,
    ) -> UnstructuredSamplingResult:
        scores: dict[str, float] = {}
        outputs: dict[str, str] = {}
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
            if capture_full_eval_data:
                outputs[ex_id] = answer

        # When capture_full_eval_data=True (required by GEPARootAgentOptimizer
        # for its reflection step), data must be keyed by example ID so GEPA
        # can look up each example's raw output via data[ex_id].
        data = {ex_id: {"output": ans} for ex_id, ans in outputs.items()} if capture_full_eval_data else None
        return UnstructuredSamplingResult(scores=scores, data=data)

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

`LocalEvalSampler` is the **built-in** `Sampler` that delegates scoring to ADK's `LocalEvalService`. It reads eval cases from an `EvalSetsManager` (typically `LocalEvalSetsManager`, which reads the eval sets stored in your agent's directory), applies the configured metrics, and returns per-example scores.

### Dependencies

`LocalEvalSampler` requires optional packages:

```bash
pip install google-adk[eval]   # includes pandas, rouge-score, etc.
# or individually:
pip install pandas rouge-score google-cloud-aiplatform
```

### Constructor (source-verified)

```python
LocalEvalSampler(
    config: LocalEvalSamplerConfig,
    eval_sets_manager: EvalSetsManager,
)
```

`LocalEvalSamplerConfig` fields (source-verified from `google/adk/optimization/local_eval_sampler.py`):

| Field | Type | Default | Purpose |
|---|---|---|---|
| `eval_config` | `EvalConfig` | required | Metrics and thresholds for scoring (e.g. `response_match_score`) |
| `app_name` | `str` | required | Must match the app name used by the eval sets manager |
| `train_eval_set` | `str` | required | ID of the eval set used for optimization iterations |
| `train_eval_case_ids` | `list[str] \| None` | `None` | Specific case IDs; `None` → use all cases in the set |
| `validation_eval_set` | `str \| None` | `None` | Eval set for final scoring; `None` → reuse `train_eval_set` |
| `validation_eval_case_ids` | `list[str] \| None` | `None` | Specific validation case IDs |

### Eval sets directory layout

`LocalEvalSetsManager` reads eval sets from flat `.evalset.json` files stored directly inside each app's directory under `agents_dir`:

```
agents/
  geography_qa/
    geography_train.evalset.json    ← train_eval_set = "geography_train"
    geography_val.evalset.json      ← validation_eval_set = "geography_val"
```

Each `.evalset.json` file is a single JSON object with `eval_set_id` and an `eval_cases` array:

```json
{
  "eval_set_id": "geography_train",
  "name": "geography_train",
  "eval_cases": [
    {
      "eval_id": "case_001",
      "conversation": [
        {
          "user_content": {"parts": [{"text": "Capital of France?"}], "role": "user"},
          "final_response": {"parts": [{"text": "Paris"}], "role": "model"}
        }
      ]
    }
  ]
}
```

### End-to-end optimisation with `LocalEvalSampler`

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.local_eval_sets_manager import LocalEvalSetsManager
from google.adk.optimization.simple_prompt_optimizer import (
    SimplePromptOptimizer,
    SimplePromptOptimizerConfig,
)
from google.adk.optimization.local_eval_sampler import (
    LocalEvalSampler,
    LocalEvalSamplerConfig,
)

agent = LlmAgent(
    name="geography_qa",
    model="gemini-2.5-flash",
    instruction="Answer geography questions accurately.",
)

# Point to the agents/ directory; eval sets live directly under agents/geography_qa/
eval_sets_manager = LocalEvalSetsManager(agents_dir="./agents")

sampler = LocalEvalSampler(
    config=LocalEvalSamplerConfig(
        eval_config=EvalConfig(
            criteria={"response_match_score": 0.5},
        ),
        app_name="geography_qa",
        train_eval_set="geography_train",       # agents/geography_qa/geography_train.evalset.json
        validation_eval_set="geography_val",    # agents/geography_qa/geography_val.evalset.json
    ),
    eval_sets_manager=eval_sets_manager,
)

optimizer = SimplePromptOptimizer(
    config=SimplePromptOptimizerConfig(
        num_iterations=5,
        batch_size=3,
    )
)

async def main():
    result = await optimizer.optimize(initial_agent=agent, sampler=sampler)
    # optimized_agents is a list; SimplePromptOptimizer always returns one entry.
    optimised = result.optimized_agents[0].optimized_agent
    print("Final prompt:", optimised.instruction)

    with open("optimised_instruction.txt", "w") as f:
        f.write(optimised.instruction)

asyncio.run(main())
```

### Train / validation split

The training and validation splits are explicitly named eval sets (not auto-derived from a single file). Set `validation_eval_set` to a different set for held-out evaluation, or leave it `None` to reuse the training set for both.

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

`Skill` is a **Pydantic model** that bundles three layers of a skill: frontmatter metadata (`Frontmatter`), instruction text (from `SKILL.md`), and optional resources (`Resources`). Skills are loaded from a directory that contains a `SKILL.md` file and optional subdirectories for references, assets, and scripts.

### Class structure (source-verified)

```python
class Skill(BaseModel):
    frontmatter: Frontmatter   # typed model — name, description, license, etc.
    instructions: str          # SKILL.md body (the skill's instruction text)
    resources: Resources = Resources()  # references, assets, scripts (defaults empty)

    @property
    def name(self) -> str:
        return self.frontmatter.name

    @property
    def description(self) -> str:
        return self.frontmatter.description
```

`Frontmatter` required fields: `name` (kebab-case or snake_case, ≤ 64 chars) and `description`.
`Resources` fields: `references: dict[str, str | bytes]`, `assets: dict[str, str | bytes]`, `scripts: dict[str, Script]`.

### Skill directory layout

A skill lives in its own directory named after the skill, with a `SKILL.md` file:

```
skills/
  write-unit-test/
    SKILL.md          ← L1 frontmatter + L2 instructions
    references/       ← optional: extra markdown guidance
    assets/           ← optional: schemas, templates, examples
    scripts/          ← optional: executable scripts
```

`SKILL.md` format:

```markdown
---
name: write-unit-test
description: Generate a pytest unit test for a given Python function.
---

You are a Python testing expert.

Given a Python function, write a complete pytest unit test that:
1. Tests the happy path with typical inputs.
2. Tests edge cases (empty input, boundary values).
3. Uses descriptive test names (`test_<function>_<scenario>`).

Return only the test code, no explanation.
```

### Creating a `Skill` programmatically

```python
from google.adk.skills.models import Skill, Frontmatter, Resources

skill = Skill(
    frontmatter=Frontmatter(
        name="write-unit-test",
        description="Generate a pytest unit test for a given Python function.",
    ),
    instructions=(
        "You are a Python testing expert.\n\n"
        "Write a complete pytest unit test for the given function."
    ),
    resources=Resources(),   # empty — no references/assets/scripts
)

print(skill.name)         # "write-unit-test"
print(skill.description)  # "Generate a pytest unit test ..."
```

### Parsing a `SKILL.md` file manually

```python
from pathlib import Path
from google.adk.skills.models import Skill, Frontmatter, Resources
import yaml

def load_skill_from_file(skill_md_path: str) -> Skill:
    text = Path(skill_md_path).read_text()
    if text.startswith("---"):
        _, fm_block, body = text.split("---", 2)
        fm_data = yaml.safe_load(fm_block)
        instructions = body.strip()
    else:
        raise ValueError("SKILL.md must begin with YAML front matter.")

    return Skill(
        frontmatter=Frontmatter(**fm_data),
        instructions=instructions,
        resources=Resources(),
    )

skill = load_skill_from_file("skills/write-unit-test/SKILL.md")
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
    async def search_skills(self, query: str) -> list[Frontmatter]:
        """Returns Frontmatter discovery metadata for skills matching the query."""
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
from google.adk.skills.models import Skill, Frontmatter

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

        def _fetch() -> dict[str, Skill]:
            result: dict[str, Skill] = {}
            for blob in self._bucket.list_blobs(prefix=self._prefix):
                if not blob.name.endswith(".json"):
                    continue
                data = json.loads(blob.download_as_text())
                skill = Skill(**data)
                result[skill.name] = skill
            return result

        # Run blocking GCS I/O off the event loop to avoid stalling other tasks.
        self._cache = await asyncio.to_thread(_fetch)
        return self._cache

    async def get_skill(self, name: str) -> Skill | None:
        all_skills = await self._load_all()
        return all_skills.get(name)

    async def search_skills(self, query: str) -> list[Frontmatter]:
        all_skills = await self._load_all()
        q = query.lower()
        return [
            s.frontmatter for s in all_skills.values()
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
from google.adk.skills.models import Skill, Frontmatter, Resources

class InMemorySkillRegistry(SkillRegistry):
    def __init__(self, skills: list[Skill]):
        self._skills = {s.name: s for s in skills}

    async def get_skill(self, name: str) -> Skill | None:
        return self._skills.get(name)

    async def search_skills(self, query: str) -> list[Frontmatter]:
        q = query.lower()
        return [s.frontmatter for s in self._skills.values()
                if q in s.name.lower() or q in s.description.lower()]

test_skill = Skill(
    frontmatter=Frontmatter(name="greet", description="Greet the user warmly."),
    instructions="Always start with 'Hello!' and use the user's name.",
    resources=Resources(),
)
registry = InMemorySkillRegistry([test_skill])
```

---

## 9 — `SkillToolset` (deep dive)

**Module:** `google.adk.tools.skill_toolset`

`SkillToolset` exposes skills to an agent as a set of built-in **skill management tools**. It does not make each skill a direct callable tool; instead the agent uses `list_skills` (to discover available skills), `load_skill` (to activate a skill and receive its instructions), `load_skill_resource` (to fetch a skill's asset or reference), and `run_skill_script` (to execute a skill's script — requires a `code_executor`). When a `registry` is provided, `search_skills` is added as well. Source-verified constructor signature:

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
| `skills_folder` | `str \| None` | `None` | **Requires `environment` to be set**; must be an absolute path |
| `script_timeout` | `int` | `300` | Max seconds for a skill's code block to run |
| `additional_tools` | `list[BaseTool] \| None` | `None` | Extra tools available to the skill's sub-agent |
| `tool_name_prefix` | `str \| None` | `None` | String prepended to every skill tool's name |
| `tool_filter` | `list[str] \| Callable \| None` | `None` | Allowlist of skill names or a predicate |

### How the agent uses skills

When an agent has a `SkillToolset`, it gets these skill-management tools:

| Tool | Purpose |
|---|---|
| `list_skills` | Returns names and descriptions of all available skills |
| `load_skill` | Activates a skill — returns its full instruction text |
| `load_skill_resource` | Fetches a reference, asset, or script file from a loaded skill |
| `run_skill_script` | Executes a script from a skill (requires `code_executor`) |
| `search_skills` | Fuzzy-searches the registry (only when `registry` is set) |

### Loading from a `skills` list

The common case — pass pre-constructed `Skill` objects:

```python
from google.adk.agents import LlmAgent
from google.adk.skills.models import Skill, Frontmatter, Resources
from google.adk.tools.skill_toolset import SkillToolset

write_test_skill = Skill(
    frontmatter=Frontmatter(
        name="write-unit-test",
        description="Generate a pytest unit test for a Python function.",
    ),
    instructions=(
        "You are a testing expert. Write a complete pytest test that covers "
        "the happy path, edge cases, and uses descriptive test names."
    ),
)

review_pr_skill = Skill(
    frontmatter=Frontmatter(
        name="review-pr",
        description="Review a pull request diff for bugs and style issues.",
    ),
    instructions=(
        "Carefully read the diff. List any bugs, missing tests, or style "
        "violations. Format as a numbered list, most severe first."
    ),
)

toolset = SkillToolset(skills=[write_test_skill, review_pr_skill])

agent = LlmAgent(
    name="dev_assistant",
    model="gemini-2.5-flash",
    instruction=(
        "You are a development assistant. Use list_skills to see what you can do, "
        "then load_skill to activate the relevant skill before responding."
    ),
    tools=[toolset],
)
```

### Filtering which skills are exposed

```python
# Allowlist: only expose specific skills by name
toolset = SkillToolset(
    skills=[write_test_skill, review_pr_skill],
    tool_filter=["write-unit-test"],   # hide review-pr from this agent
)

# Predicate: dynamic filtering
# Predicate receives (tool: BaseTool, context: ReadonlyContext | None)
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import ReadonlyContext

def only_stable(tool: BaseTool, context: ReadonlyContext | None) -> bool:
    return not tool.name.startswith("experimental-")

toolset = SkillToolset(skills=all_skills, tool_filter=only_stable)
```

### Namespacing skill management tool names

```python
# Prevents collisions when merging two toolsets in one agent
coding_toolset = SkillToolset(
    skills=coding_skills,
    tool_name_prefix="coding__",  # tools become coding__list_skills, etc.
)
writing_toolset = SkillToolset(
    skills=writing_skills,
    tool_name_prefix="writing__",
)

agent = LlmAgent(
    name="super_agent",
    model="gemini-2.5-flash",
    instruction="Help with coding and writing. Use coding__list_skills or writing__list_skills to start.",
    tools=[coding_toolset, writing_toolset],
)
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

`skills_folder` is **only valid when `environment` is also set** — it tells the toolset where skills are located inside the environment's filesystem. Passing `skills_folder` without `environment` raises `ValueError`.

```python
import os
from google.adk.tools.skill_toolset import SkillToolset

# WRONG — raises ValueError: Cannot specify skills_folder without an environment:
# toolset = SkillToolset(skills_folder="/abs/path/to/skills")

# CORRECT — skills_folder requires environment:
toolset = SkillToolset(
    skills_folder=os.path.abspath("skills"),  # must also be absolute
    environment=my_env,
    code_executor=my_executor,
)
```

Without an environment, load skills as `Skill` objects and pass them via `skills=[...]` (see "Loading from a `skills` list" above).

---

## 10 — `VertexAiRagMemoryService`

**Module:** `google.adk.memory.vertex_ai_rag_memory_service`

`VertexAiRagMemoryService` stores conversation history in an **Agent Platform RAG corpus** (via the `agentplatform` package). At the end of a session, `add_session_to_memory()` serialises all events to a temporary text file and uploads it to the RAG corpus. `search_memory()` queries the corpus with semantic similarity and returns the most relevant past exchanges.

### Prerequisites

```bash
pip install google-cloud-aiplatform   # provides the `agentplatform` package
pip install "google-adk[db]" aiosqlite  # DatabaseSessionService (SQLAlchemy + async driver)
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
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.genai import types

RAG_CORPUS = "projects/my-project/locations/us-central1/ragCorpora/1234567890"

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction=(
        "You are a persistent assistant. Use your memory to recall "
        "context from previous conversations."
    ),
    # PreloadMemoryTool is invisible to the model; it auto-injects relevant
    # memories from the RAG corpus before each LLM call.
    tools=[PreloadMemoryTool()],
)

memory_service = VertexAiRagMemoryService(
    rag_corpus=RAG_CORPUS,
    similarity_top_k=5,
    vector_distance_threshold=0.8,
)

runner = Runner(
    agent=agent,
    app_name="persistent_assistant",
    session_service=DatabaseSessionService(db_url="sqlite+aiosqlite:///sessions.db"),
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

    # Runner does NOT automatically ingest the session.
    # Reload the updated session and persist it to the RAG corpus explicitly.
    updated_session = await runner.session_service.get_session(
        app_name="persistent_assistant",
        user_id=user_id,
        session_id=session.id,
    )
    if updated_session:
        await runner.memory_service.add_session_to_memory(updated_session)

    return response, session.id
```

### Automatic memory preloading with `PreloadMemoryTool`

`PreloadMemoryTool` is **invisible to the model** — it is not a callable tool the agent invokes. Instead it overrides `process_llm_request` and automatically queries the memory service before each LLM call, injecting relevant past exchanges as context. Add it to the agent to get automatic retrieval every turn:

```python
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="You are a persistent assistant.",
    tools=[PreloadMemoryTool()],  # auto-injects memory before each LLM call
)
# The model never sees or calls "preload_memory" — it just receives the context.
```

### Triggering memory ingestion via a callback

If you prefer to ingest at the end of each turn rather than calling `add_session_to_memory` after every `runner.run()` call, use an `after_agent_callback`:

```python
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext

async def save_to_memory(ctx: CallbackContext) -> None:
    await ctx.add_session_to_memory()

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="You are a persistent assistant.",
    after_agent_callback=save_to_memory,
)
```

### Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: No module named 'agentplatform'` | `pip install --upgrade google-cloud-aiplatform` |
| `ValueError: rag_corpus must be set` | Pass the full corpus resource name or set corpus on every `rag_resource` |
| `DeprecationWarning: vertexai.preview.rag` | Already on the new `agentplatform` path; warning means mixed install — upgrade `google-cloud-aiplatform` |
| High latency on `search_memory` | Reduce `similarity_top_k` or lower `vector_distance_threshold` to fetch fewer chunks (threshold is a maximum distance — lower = stricter) |
| Stale data returned | `add_session_to_memory()` must be called explicitly; `Runner` does not auto-ingest. Use a callback or call it after `runner.run()` completes |

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

---
title: "Class deep dives — volume 44 (SkillRegistry/Skill/Frontmatter, SkillToolset, AgentOptimizer/Sampler, GEPARootAgentOptimizer, OpenAILlm, LangGraphAgent, LiveRequestQueue, AutoFlow/SingleFlow, PubSubToolset, DynamicNodeScheduler)"
description: "10 source-verified deep dives for google-adk 2.5.0: SkillRegistry ABC and Skill/Frontmatter/Resources data models, SkillToolset tool framework (list/load/resource/script/search), AgentOptimizer and Sampler ABCs, GEPARootAgentOptimizer GEPA-based prompt optimization, OpenAILlm labs GPT backend, LangGraphAgent checkpointer integration, LiveRequestQueue and LiveRequest bidirectional streaming, AutoFlow vs SingleFlow transfer flow architecture, PubSubToolset with publish/pull/acknowledge, DynamicNodeScheduler ctx.run_node() internals (dedup, resume, fresh execution)."
framework: google-adk
language: python
sidebar:
  label: "Class deep dives — vol. 44"
  order: 113
---

import { Aside } from "@astrojs/starlight/components";

<Aside type="note">
All signatures, constants, and behaviours on this page were verified directly
against the installed package source (locate yours with
`python -c 'import google.adk; print(google.adk.__file__)'`) on
**google-adk == 2.5.0**. No documentation or blog posts were used as primary
sources.
</Aside>

---

## 1 · `SkillRegistry` + `Skill` / `Frontmatter` / `Resources` — the skill data layer

**Sources:** `google/adk/skills/skill_registry.py`, `google/adk/skills/models.py`

### `Frontmatter` — skill discovery metadata

`Frontmatter` is the L1 discovery record parsed from a `SKILL.md` file's
YAML front matter. It is what the model sees when browsing available skills.

```python
class Frontmatter(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str                        # kebab-case or snake_case, max 64 chars
    description: str                 # max 1024 chars; what the skill does and
                                     # when to use it
    license: Optional[str] = None
    compatibility: Optional[str] = None
    allowed_tools: Optional[str] = Field(
        default=None,
        alias="allowed-tools",
        serialization_alias="allowed-tools",
    )
    metadata: dict[str, Any] = {}    # adk_additional_tools, adk_inject_state …
```

`name` is validated by `_validate_name`: it must match
`^[a-z0-9]+(-[a-z0-9]+)*$` (kebab) or `^([a-z0-9]+(_[a-z0-9]+)*)$`
(snake_case when `FeatureName.SNAKE_CASE_SKILL_NAME` is enabled), must not
exceed 64 characters, and mixes of hyphens and underscores are rejected.

`metadata` supports two reserved keys:
- `adk_additional_tools` — list of tool names pre-approved for the skill.
- `adk_inject_state` (bool) — when `True`, `{var}` placeholders in the SKILL.md
  body are replaced with matching values from session state at load time.

### `Skill` — full skill representation

```python
class Skill(BaseModel):
    frontmatter: Frontmatter
    instructions: str      # L2 — markdown body of SKILL.md
    resources: Resources = Resources()  # L3 — references, assets, scripts
```

Convenience properties `.name` and `.description` delegate to `frontmatter`.

### `Resources` — L3 skill content

```python
class Resources(BaseModel):
    references: dict[str, str | bytes] = {}   # e.g. references/guide.md
    assets:     dict[str, str | bytes] = {}   # e.g. assets/template.txt
    scripts:    dict[str, Script] = {}         # e.g. scripts/setup.sh
```

Access helpers: `get_reference(id)`, `get_asset(id)`, `get_script(id)`.
List helpers: `list_references()`, `list_assets()`, `list_scripts()`.

### `SkillRegistry` — abstract registry interface

```python
class SkillRegistry(ABC):
    @abstractmethod
    async def get_skill(self, *, name: str) -> Skill: ...

    @abstractmethod
    async def search_skills(self, *, query: str) -> list[Frontmatter]: ...

    def search_tool_description(self) -> str | None:
        return None   # override to customise the search_skills tool description
```

Implement `SkillRegistry` to connect any backend — a local directory, a
vector database, or a hosted registry service. `search_tool_description`
lets the registry override the `search_skills` tool's description string so
the model receives registry-specific search guidance.

### Example 1 — local in-memory skill registry

```python
import asyncio
from google.adk.skills.models import Frontmatter, Skill, Resources
from google.adk.skills.skill_registry import SkillRegistry


class DictRegistry(SkillRegistry):
    """Minimal in-memory registry."""

    def __init__(self, skills: list[Skill]) -> None:
        self._skills = {s.name: s for s in skills}

    async def get_skill(self, *, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' not found")
        return self._skills[name]

    async def search_skills(self, *, query: str) -> list[Frontmatter]:
        q = query.lower()
        return [
            s.frontmatter
            for s in self._skills.values()
            if q in s.frontmatter.description.lower()
        ]


pii_skill = Skill(
    frontmatter=Frontmatter(
        name="pii-redaction",
        description="Redacts PII (names, emails, phone numbers) from text.",
    ),
    instructions=(
        "# PII Redaction\n\n"
        "Replace all names with [NAME], emails with [EMAIL], "
        "and phone numbers with [PHONE].\n"
    ),
)

registry = DictRegistry([pii_skill])
skill = asyncio.run(registry.get_skill(name="pii-redaction"))
print(skill.instructions[:50])
```

### Example 2 — state injection in a skill

With `adk_inject_state: true` in `metadata`, template variables are resolved
from session state at the moment the skill is loaded:

```python
from google.adk.skills.models import Frontmatter, Skill

dynamic_skill = Skill(
    frontmatter=Frontmatter(
        name="customer-greeting",
        description="Greet a customer by their tier.",
        metadata={"adk_inject_state": True},
    ),
    instructions=(
        "The customer '{customer_name}' is a {tier?} tier member. "
        "Address them formally and offer tier-specific benefits.\n"
        # {tier?} uses the safe-substitute variant: empty string if missing.
    ),
)
```

---

## 2 · `SkillToolset` — the full skills tool framework

**Source:** `google/adk/tools/skill_toolset.py`

`SkillToolset` bridges the `SkillRegistry` / `Skill` data layer and the ADK
tool system. It exposes up to five tools to the model: `list_skills`,
`load_skill`, `load_skill_resource`, `run_skill_script`, and (when a registry
is configured) `search_skills`.

### Constructor signature (verified `skill_toolset.py`)

```python
class SkillToolset(BaseToolset):
    def __init__(
        self,
        skills: list[models.Skill] | None = None,
        *,
        registry: SkillRegistry | None = None,
        code_executor: BaseCodeExecutor | None = None,
        script_timeout: int = 300,        # seconds for subprocess scripts
        additional_tools: list[ToolUnion] | None = None,
        tool_name_prefix: str | None = None,
        tool_filter: ToolPredicate | list[str] | None = None,
    ): ...
```

- `skills` — local skill definitions available without a registry call.
- `registry` — when provided, a `search_skills` tool is added; `load_skill`
  falls back to the registry when a locally unknown skill name is requested.
- `code_executor` — optional executor for Python scripts inside skills;
  `script_timeout` caps subprocess-based shell scripts at 300 s by default.
- `additional_tools` — `BaseTool`, `BaseToolset`, or plain callables that
  become available to the agent after a skill is activated.
- `tool_name_prefix` — prepends a string to every tool name (e.g. `"hr_"` →
  `hr_load_skill`, `hr_list_skills`).

Duplicate `name` entries in `skills` raise `ValueError` immediately.

### Skill lifecycle: list → search → load → resource → script

```
list_skills         → returns <available_skills> XML block (local only)
search_skills       → hits SkillRegistry.search_skills(query=…)
load_skill          → loads SKILL.md body; records activation in
                       session state under _adk_activated_skill_<agent>
load_skill_resource → serves references/, assets/, or scripts/ content;
                       caps RESOURCE_NOT_FOUND retries with a fatal error
                       on the second miss (same invocation)
run_skill_script    → runs scripts/ via code_executor or subprocess
```

The `_MAX_SKILL_PAYLOAD_BYTES` constant (16 MB) guards against oversized
skill definitions. Binary asset files trigger the "Content Injection"
pattern: instead of returning raw bytes, the tool injects the content into
the LLM request history so the model can analyse it.

### Example 1 — local skills with additional_tools

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.skills.models import Frontmatter, Skill
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.tools.function_tool import FunctionTool
from google.genai import types


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (stub)."""
    return f"Email sent to {to}"


email_skill = Skill(
    frontmatter=Frontmatter(
        name="email-composer",
        description="Composes and sends professional emails.",
        metadata={"adk_additional_tools": ["send_email"]},
    ),
    instructions=(
        "# Email Composer\n\n"
        "1. Draft the email body based on the user's request.\n"
        "2. Call `send_email` with `to`, `subject`, and `body`.\n"
    ),
)

skill_toolset = SkillToolset(
    skills=[email_skill],
    additional_tools=[FunctionTool(send_email)],
)

agent = LlmAgent(
    name="assistant",
    model="gemini-2.5-flash",
    instruction="Use skills to handle specialised tasks.",
    tools=[skill_toolset],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="demo")
    session = await runner.session_service.create_session(
        app_name="demo", user_id="u1"
    )
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="Email bob@example.com about tomorrow's standup.")],
        ),
    ):
        if event.content and event.content.parts:
            print(event.content.parts[0].text)


asyncio.run(main())
```

### Example 2 — registry-backed dynamic discovery

```python
from google.adk.tools.skill_toolset import SkillToolset

toolset = SkillToolset(
    registry=my_vector_registry,      # SkillRegistry subclass
    tool_name_prefix="skills",        # tools become skills_load_skill etc.
)
```

With a registry, the model can call `skills_search_skills(query="…")` to
find skills it wasn't pre-loaded with, then `skills_load_skill(skill_name="…")`
to fetch the full instructions.

---

## 3 · `AgentOptimizer[T, U]` + `Sampler[T]` ABCs — optimization contracts

**Sources:** `google/adk/optimization/agent_optimizer.py`,
`google/adk/optimization/sampler.py`, `google/adk/optimization/data_types.py`

### `Sampler[SamplingResultT]`

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

`sample_and_score` runs the candidate agent against a batch of example UIDs.
When `capture_full_eval_data=True` the implementation must also return
trajectories, tool calls, and other artefacts the optimizer needs for
reflection. When `False`, only `scores` are required.

### Key data types

```python
class SamplingResult(BaseModel):
    scores: dict[str, float]  # example_uid → score (higher is better)

class UnstructuredSamplingResult(SamplingResult):
    data: Optional[dict[str, dict[str, Any]]] = None  # uid → eval artefacts

class AgentWithScores(BaseModel):
    optimized_agent: Agent
    overall_score: Optional[float] = None

class OptimizerResult(BaseModel, Generic[AgentWithScoresT]):
    optimized_agents: list[AgentWithScoresT]   # Pareto-front candidates
```

### `AgentOptimizer[SamplingResultT, AgentWithScoresT]`

```python
class AgentOptimizer(ABC, Generic[SamplingResultT, AgentWithScoresT]):
    @abstractmethod
    async def optimize(
        self,
        initial_agent: Agent,
        sampler: Sampler[SamplingResultT],
    ) -> OptimizerResult[AgentWithScoresT]: ...
```

### Example — minimal custom optimizer

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.optimization.agent_optimizer import AgentOptimizer
from google.adk.optimization.data_types import (
    AgentWithScores, OptimizerResult, UnstructuredSamplingResult,
)
from google.adk.optimization.sampler import Sampler


class GreedyPromptOptimizer(
    AgentOptimizer[UnstructuredSamplingResult, AgentWithScores]
):
    """Tries a list of candidate instructions and picks the best scorer."""

    def __init__(self, candidates: list[str]) -> None:
        self._candidates = candidates

    async def optimize(
        self,
        initial_agent: LlmAgent,
        sampler: Sampler[UnstructuredSamplingResult],
    ) -> OptimizerResult[AgentWithScores]:
        best_agent = initial_agent
        best_score = -1.0

        for instruction in self._candidates:
            candidate = initial_agent.clone(update={"instruction": instruction})
            result = await sampler.sample_and_score(
                candidate,
                example_set="validation",
            )
            mean_score = sum(result.scores.values()) / len(result.scores)
            if mean_score > best_score:
                best_score = mean_score
                best_agent = candidate

        return OptimizerResult(
            optimized_agents=[
                AgentWithScores(
                    optimized_agent=best_agent,
                    overall_score=best_score,
                )
            ]
        )
```

---

## 4 · `GEPARootAgentOptimizer` — GEPA-based automated prompt optimization

**Source:** `google/adk/optimization/gepa_root_agent_optimizer.py`

`GEPARootAgentOptimizer` is the concrete optimizer that uses the GEPA
(Generative Evaluation and Prompt Analysis) framework to iteratively refine
an agent's `instruction` and any `SkillToolset` skill instructions. It is
marked `@experimental`.

### Config

```python
class GEPARootAgentOptimizerConfig(BaseModel):
    optimizer_model: str = "gemini-3.5-flash"
    model_configuration: genai_types.GenerateContentConfig = Field(
        default_factory=lambda: genai_types.GenerateContentConfig(
            thinking_config=genai_types.ThinkingConfig(
                include_thoughts=True,
                thinking_level=genai_types.ThinkingLevel.HIGH,
            )
        )
    )
    max_metric_calls: int = 100          # total evaluations budget
    reflection_minibatch_size: int = 3   # examples per reflection step
    run_dir: str | None = None           # checkpoint dir for resuming
```

Setting `run_dir` enables resumable optimization: if the process is
interrupted, the next run picks up from the last checkpoint.

### Optimization targets

The optimizer treats two kinds of "components" as separate optimization targets:

| Key pattern | What is optimized |
|---|---|
| `"agent_prompt"` | `initial_agent.instruction` |
| `"skill_instructions:<name>"` | `SkillToolset` skill instructions |

Skills are processed first (dict ordering), then the core agent prompt.

### Internal flow

1. A `_AgentGEPAAdapter` bridges ADK's async eval loop with GEPA's synchronous
   `evaluate`/`make_reflective_dataset`/`propose_new_texts` callbacks.
2. The GEPA `run_in_executor` bridge: each GEPA call issues
   `asyncio.run_coroutine_threadsafe` to run `sampler.sample_and_score` on the
   main event loop from a thread pool, then `future.result()` blocks until done.
3. After GEPA converges, each Pareto-front candidate is reconstructed via
   `_create_agent_from_candidate` and wrapped in `AgentWithScores`.

```python
class GEPARootAgentOptimizerResult(OptimizerResult[AgentWithScores]):
    gepa_result: dict[str, Any] | None = None  # raw GEPA result dict
```

### Example

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.optimization.gepa_root_agent_optimizer import (
    GEPARootAgentOptimizer, GEPARootAgentOptimizerConfig,
)
from google.adk.optimization.sampler import Sampler
from google.adk.optimization.data_types import UnstructuredSamplingResult


class MyEvalSampler(Sampler[UnstructuredSamplingResult]):
    def get_train_example_ids(self) -> list[str]:
        return ["ex-001", "ex-002", "ex-003"]

    def get_validation_example_ids(self) -> list[str]:
        return ["val-001", "val-002"]

    async def sample_and_score(
        self, candidate, example_set="validation", batch=None,
        capture_full_eval_data=False,
    ) -> UnstructuredSamplingResult:
        ids = batch or (
            self.get_train_example_ids()
            if example_set == "train"
            else self.get_validation_example_ids()
        )
        # run candidate against your evaluation harness …
        scores = {uid: 0.85 for uid in ids}   # placeholder
        return UnstructuredSamplingResult(scores=scores)


async def main():
    agent = LlmAgent(
        name="support-bot",
        model="gemini-2.5-flash",
        instruction="You are a customer support assistant.",
    )
    config = GEPARootAgentOptimizerConfig(
        optimizer_model="gemini-2.5-flash",
        max_metric_calls=30,
        run_dir="/tmp/gepa-run",
    )
    optimizer = GEPARootAgentOptimizer(config=config)
    result = await optimizer.optimize(agent, MyEvalSampler())
    best = result.optimized_agents[0]
    print("Score:", best.overall_score)
    print("New instruction:", best.optimized_agent.instruction[:120])


asyncio.run(main())
```

<Aside type="caution">
`GEPARootAgentOptimizer` requires `pip install gepa` in addition to `google-adk`.
The `gepa` package is not included in ADK's default dependencies.
</Aside>

---

## 5 · `OpenAILlm` (labs) — drop-in GPT backend

**Source:** `google/adk/labs/openai/_openai_llm.py`

`OpenAILlm` is an experimental `BaseLlm` implementation that lets you swap
Gemini for any OpenAI Chat Completions-compatible model (GPT-4o, o1, o3, etc.)
without changing agent code. It lives under `google.adk.labs` — install
`openai` separately.

### Class definition

```python
class OpenAILlm(BaseLlm):
    model: str = "gpt-4o"
    max_tokens: int = 4096

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"gpt-.*", r"o1-.*", r"o3-.*"]

    @cached_property
    def _openai_client(self) -> AsyncOpenAI:
        return AsyncOpenAI()   # reads OPENAI_API_KEY from env

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]: ...
```

### Protocol translation layer

| ADK concept | OpenAI concept |
|---|---|
| `types.Content(role="model", …)` | `{"role": "assistant", …}` |
| `types.Part.function_call(…)` | `{"type": "function", "tool_calls": […]}` |
| `types.Part.function_response(…)` | `{"role": "tool", "tool_call_id": …}` |
| `llm_request.config.response_schema` (Pydantic) | `response_format: {"type": "json_schema", …}` |
| `llm_request.config.response_mime_type == "application/json"` | `response_format: {"type": "json_object"}` |

`_update_type_string` recursively lowercases all JSON Schema `type` values
to satisfy the OpenAI strict schema validator (Gemini uses uppercase strings).

### Streaming mode

Streaming accumulates text deltas into `text_accumulated` and tool-call
argument fragments per `index` in `tool_calls_accumulated`. A single final
`LlmResponse(partial=False)` is yielded after the stream ends, containing
all assembled tool calls with parsed JSON arguments.

### Example

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.labs.openai import OpenAILlm
from google.adk.runners import InMemoryRunner
from google.genai import types

# Register the OpenAI backend once at startup
from google.adk.models.registry import LLMRegistry
LLMRegistry().register(OpenAILlm)

agent = LlmAgent(
    name="gpt-agent",
    model="gpt-4o",          # matched by supported_models regex gpt-.*
    instruction="You are a helpful assistant.",
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="gpt-demo")
    session = await runner.session_service.create_session(
        app_name="gpt-demo", user_id="u1"
    )
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="What's 2 + 2?")],
        ),
    ):
        if event.content and not event.partial:
            print(event.content.parts[0].text)


asyncio.run(main())
```

<Aside type="note">
Set `OPENAI_API_KEY` in the environment before using `OpenAILlm`. The
`_openai_client` property is `@cached_property`, so the `AsyncOpenAI` client
is created once per `OpenAILlm` instance.
</Aside>

---

## 6 · `LangGraphAgent` — embedding a compiled LangGraph graph into ADK

**Source:** `google/adk/agents/langgraph_agent.py`

`LangGraphAgent` is a `BaseAgent` subclass that wraps a compiled LangGraph
`CompiledGraph`. It bridges ADK's event-based session model with LangGraph's
state machine. Marked "currently a concept implementation".

### Class definition

```python
class LangGraphAgent(BaseAgent):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph: CompiledGraph   # the compiled langgraph graph
    instruction: str = ''  # injected as SystemMessage on first turn
```

### Execution model

`_run_async_impl` drives the graph via `graph.invoke`:

1. Reads `session.id` and passes it as `configurable.thread_id` to the
   LangGraph runnable config — this is how LangGraph checkpointers link turns.
2. Fetches the current graph state. If the state already has `messages`
   (an existing checkpointed conversation), the `instruction` is **not**
   re-injected.
3. Builds the `messages` list, then calls `graph.invoke({'messages': messages},
   config)`.
4. Yields a single `Event` wrapping `final_state['messages'][-1].content`.

### Memory modes

| Scenario | `_get_messages` strategy |
|---|---|
| `graph.checkpointer` is set | Only the **last** user messages are passed (LangGraph manages full history) |
| No checkpointer | Full conversation (user + this agent's responses) from `session.events` |

```python
def _get_last_human_messages(events):
    messages = []
    for event in reversed(events):
        if messages and event.author != 'user':
            break
        if event.author == 'user' and event.content and event.content.parts:
            messages.append(HumanMessage(content=event.content.parts[0].text))
    return list(reversed(messages))
```

### Example

```python
import asyncio
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

from google.adk.agents.langgraph_agent import LangGraphAgent
from google.adk.runners import InMemoryRunner
from google.genai import types


def call_model(state: MessagesState):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.set_entry_point("model")
builder.add_edge("model", END)
graph = builder.compile(checkpointer=InMemorySaver())

agent = LangGraphAgent(
    name="lg-agent",
    graph=graph,
    instruction="You are a helpful assistant that remembers context.",
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="lg-demo")
    session = await runner.session_service.create_session(
        app_name="lg-demo", user_id="u1"
    )
    for msg in ["Hello, who are you?", "What did I just ask you?"]:
        async for event in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part(text=msg)]
            ),
        ):
            if event.content and event.content.parts:
                print(f"> {event.content.parts[0].text}")


asyncio.run(main())
```

---

## 7 · `LiveRequestQueue` + `LiveRequest` — bidirectional streaming input

**Source:** `google/adk/agents/live_request_queue.py`

`LiveRequestQueue` is the write-side interface for real-time (bidirectional)
agent sessions. The agent's `_run_live_impl` reads from this queue; external
code pushes `LiveRequest` objects to it.

### `LiveRequest` — priority-ordered input types

```python
class LiveRequest(BaseModel):
    model_config = ConfigDict(ser_json_bytes='base64', val_json_bytes='base64')

    content: Optional[types.Content] = None        # turn-by-turn text/data
    blob: Optional[types.Blob] = None              # realtime audio/video chunk
    activity_start: Optional[types.ActivityStart] = None  # VAD start
    activity_end: Optional[types.ActivityEnd] = None      # VAD end
    close: bool = False    # signals queue shutdown (Python < 3.13 workaround)
    partial: bool = False  # content is a partial turn update
```

Priority order (highest first): `activity_start` > `activity_end` > `blob` >
`content`. Only one non-`close` field should be set per request.

### `LiveRequestQueue` — asyncio wrapper

```python
class LiveRequestQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[LiveRequest] = asyncio.Queue()

    def close(self) -> None: ...                         # shutdown signal
    def send_content(self, content, partial=False): ...  # turn-by-turn
    def send_realtime(self, blob): ...                   # raw audio/video
    def send_activity_start(self): ...                   # VAD start marker
    def send_activity_end(self): ...                     # VAD end marker
    def send(self, req: LiveRequest): ...                # raw LiveRequest
    async def get(self) -> LiveRequest: ...              # consumer side
```

All `send_*` methods call `_queue.put_nowait`, making them safe to call from
synchronous code or from a different async task.

### Example 1 — streaming audio to a live agent

```python
import asyncio
import wave
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.genai import types as genai_types


async def stream_audio(queue: LiveRequestQueue, wav_path: str) -> None:
    with wave.open(wav_path, "rb") as f:
        chunk_size = f.getframerate() * 2 * f.getsampwidth() // 10  # 100 ms
        queue.send_activity_start()
        while True:
            data = f.readframes(chunk_size)
            if not data:
                break
            queue.send_realtime(
                genai_types.Blob(
                    mime_type=f"audio/pcm;rate={f.getframerate()}",
                    data=data,
                )
            )
            await asyncio.sleep(0.1)   # pace to real-time
        queue.send_activity_end()
    queue.close()
```

### Example 2 — text-mode live session

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.runners import InMemoryRunner
from google.genai import types

agent = LlmAgent(
    name="live-agent",
    model="gemini-2.0-flash-live",
    instruction="You are a live assistant. Respond concisely.",
)

runner = InMemoryRunner(agent=agent, app_name="live-demo")
queue = LiveRequestQueue()


async def run(session_id):
    async for event in runner.run_live(
        user_id="u1",
        session_id=session_id,
        live_request_queue=queue,
    ):
        if event.content and event.content.parts:
            print(event.content.parts[0].text, end="", flush=True)


async def send_messages():
    for text in ["Hello!", "What is the capital of France?", ""]:
        if text:
            queue.send_content(
                types.Content(role="user", parts=[types.Part(text=text)])
            )
        await asyncio.sleep(1.5)
    queue.close()


async def main():
    session = await runner.session_service.create_session(
        app_name="live-demo", user_id="u1"
    )
    await asyncio.gather(run(session.id), send_messages())


asyncio.run(main())
```

---

## 8 · `AutoFlow` vs `SingleFlow` — the transfer flow architecture

**Sources:** `google/adk/flows/llm_flows/auto_flow.py`,
`google/adk/flows/llm_flows/single_flow.py`

Every `LlmAgent` has a `_llm_flow` attribute (set during construction) that
controls how LLM requests are built and responses processed. The two concrete
choices are `SingleFlow` and `AutoFlow`.

### `SingleFlow` — standard single-agent flow

`SingleFlow` assembles an ordered pipeline of request processors and response
processors:

**Request processors (in order):**
1. `basic.request_processor` — model ID, generation config
2. `auth_preprocessor.request_processor` — credential injection
3. `request_confirmation.request_processor` — `require_confirmation` gate
4. `instructions.request_processor` — system instruction rendering
5. `identity.request_processor` — agent identity context
6. `compaction.request_processor` — event compaction (runs before `contents`)
7. `contents.request_processor` — conversation history
8. `context_cache_processor.request_processor` — cache metadata
9. `interactions_processor.request_processor` — stateful Interactions API ID
10. `_nl_planning.request_processor` — natural-language planning
11. `_code_execution.request_processor` — data file optimisation
12. `_output_schema_processor.request_processor` — schema + tool coexistence

**Response processors:**
- `_nl_planning.response_processor`, `_code_execution.response_processor`,
  `basic.response_processor`

### `AutoFlow` — adds agent transfer

```python
class AutoFlow(SingleFlow):
    def __init__(self) -> None:
        super().__init__()
        self.request_processors += [agent_transfer.request_processor]
```

`agent_transfer.request_processor` injects the `transfer_to_agent` function
declaration into the LLM request and handles the model's `transfer_to_agent`
function call response by setting `ctx.actions.transfer_to_agent`. This is
the only difference between `AutoFlow` and `SingleFlow`.

Transfer directions allowed by `AutoFlow`:
- Parent → sub-agent
- Sub-agent → parent
- Sub-agent → peer agents (only when parent is `LlmAgent` and
  `disallow_transfer_to_peers=False`, which is the default)

### When each flow is chosen

| `LlmAgent` configuration | Flow assigned |
|---|---|
| Has `sub_agents` | `AutoFlow` |
| `disallow_transfer_to_parent=True` AND `disallow_transfer_to_peers=True` AND no `sub_agents` | `SingleFlow` |
| Any other configuration (including standalone agents by default) | `AutoFlow` |

### Example — explicitly disabling peer transfers

```python
from google.adk.agents import LlmAgent

specialist = LlmAgent(
    name="tax-specialist",
    model="gemini-2.5-flash",
    instruction="You handle tax questions only.",
    disallow_transfer_to_peers=True,  # prevents transfer to sibling agents
)
```

---

## 9 · `PubSubToolset` — Google Cloud Pub/Sub integration

**Sources:** `google/adk/tools/pubsub/pubsub_toolset.py`,
`google/adk/tools/pubsub/message_tool.py`,
`google/adk/tools/pubsub/config.py`

`PubSubToolset` is an `@experimental` `BaseToolset` that exposes three Pub/Sub
operations to the agent: `publish_message`, `pull_messages`, and
`acknowledge_messages`. Requires `pip install google-cloud-pubsub`.

### Constructor

```python
@experimental(FeatureName.PUBSUB_TOOLSET)
class PubSubToolset(BaseToolset):
    def __init__(
        self,
        *,
        tool_filter: ToolPredicate | list[str] | None = None,
        credentials_config: PubSubCredentialsConfig | None = None,
        pubsub_tool_config: PubSubToolConfig | None = None,
    ): ...
```

`PubSubToolConfig` holds an optional `project_id` (inferred from the
environment if `None`). `PubSubCredentialsConfig` controls GCP credential
resolution (ADC, service account key, workload identity, etc.).

`close()` calls `client.cleanup_clients()` to flush publisher batching buffers
and close subscriber connections.

### The three tools

#### `publish_message`

```python
def publish_message(
    topic_name: str,   # projects/<proj>/topics/<topic>
    message: str,
    credentials: Credentials,
    settings: PubSubToolConfig,
    attributes: Optional[dict[str, str]] = None,
    ordering_key: str = "",
) -> dict[str, Any]:
    # returns {"message_id": "…"} or {"status": "ERROR", "error_details": "…"}
```

`enable_message_ordering` is set automatically when `ordering_key` is non-empty.

#### `pull_messages`

```python
def pull_messages(
    subscription_name: str,   # projects/<proj>/subscriptions/<sub>
    credentials: Credentials,
    settings: PubSubToolConfig,
    *,
    max_messages: int = 1,
    auto_ack: bool = False,
) -> dict[str, Any]:
    # returns {"messages": [{message_id, data, attributes, ordering_key,
    #                         publish_time, ack_id}, …]}
```

Binary message data is base64-encoded if UTF-8 decoding fails.

#### `acknowledge_messages`

```python
def acknowledge_messages(
    subscription_name: str,
    ack_ids: list[str],
    credentials: Credentials,
    settings: PubSubToolConfig,
) -> dict[str, Any]:
    # returns {"status": "SUCCESS"} or {"status": "ERROR", …}
```

### Example 1 — publish with ordering key

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.pubsub import PubSubToolset
from google.adk.tools.pubsub.config import PubSubToolConfig
from google.genai import types

toolset = PubSubToolset(
    pubsub_tool_config=PubSubToolConfig(project_id="my-gcp-project"),
    tool_filter=["publish_message"],   # expose only publish
)

agent = LlmAgent(
    name="publisher-agent",
    model="gemini-2.5-flash",
    instruction=(
        "You publish sensor readings to Pub/Sub. "
        "Use ordering_key=sensor_id to preserve message order."
    ),
    tools=[toolset],
)


async def main():
    runner = InMemoryRunner(agent=agent, app_name="pubsub-demo")
    session = await runner.session_service.create_session(
        app_name="pubsub-demo", user_id="u1"
    )
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(
                text=(
                    "Publish the reading 'temp=22.3' to "
                    "projects/my-gcp-project/topics/sensors, "
                    "ordering by sensor-42."
                )
            )],
        ),
    ):
        if event.content and event.content.parts:
            print(event.content.parts[0].text)


asyncio.run(main())
```

### Example 2 — pull-and-ack pattern

```python
toolset = PubSubToolset(
    pubsub_tool_config=PubSubToolConfig(project_id="my-gcp-project"),
    tool_filter=["pull_messages", "acknowledge_messages"],
)

agent = LlmAgent(
    name="consumer-agent",
    model="gemini-2.5-flash",
    instruction=(
        "Pull up to 5 messages from the subscription, process them, "
        "then acknowledge all received ack_ids."
    ),
    tools=[toolset],
)
```

---

## 10 · `DynamicNodeScheduler` — `ctx.run_node()` internals

**Source:** `google/adk/workflow/_dynamic_node_scheduler.py`

`DynamicNodeScheduler` is the machinery behind `Context.run_node()` in a
`Workflow`. It resolves whether a dynamic node should execute fresh, be
fast-forwarded from cached events (dedup), or be resumed after an interrupt.

### Data classes

```python
@dataclass
class DynamicNodeRun:
    state: NodeState             # status, interrupts, run_id
    output: Any = None           # final output once completed
    task: asyncio.Task | None = None  # running asyncio Task
    transfer_to_agent: str | None = None
    recovered_state: _ChildScanState | None = None  # raw scan from events

@dataclass
class DynamicNodeState:
    runs: dict[str, DynamicNodeRun] = field(default_factory=dict)
    interrupt_ids: set[str] = field(default_factory=set)
    replay_manager: ReplayManager = field(default_factory=ReplayManager)
```

`node_path` keys in `runs` use the format `/<wf-name>@<run-id>/<node-name>@<run-id>`,
e.g. `/summarize_wf@1/classifier@1`.

### Three execution cases

```
Case 1 — Fresh:    node_path not in runs AND no prior events
                   → _run_node_internal(is_fresh=True)

Case 2 — Completed: recovered events show COMPLETED status
                   → create_mock_context with cached output (no LLM call)

Case 3 — Waiting:  recovered events show WAITING/interrupt
                   → check_interception decides rerun vs. propagate
                   → if rerun: _run_node_internal(is_fresh=False)
                   → if propagate: add interrupt_ids to DynamicNodeState
```

### Deduplication and concurrency

When the same `node_path` is requested while a task is already running
(concurrent `run_node` calls), `_check_existing_run` awaits the existing
`run.task` directly:

```python
if run.task and not run.task.done():
    return await run.task, True   # deduplicated — returns same result
```

### Schema validation at the gate

`DynamicNodeScheduler.__call__` validates `node_input` against
`node._validate_input_data` before doing anything else. Validation errors are
re-raised with the node name included in the title for clear debugging:

```python
try:
    node_input = node._validate_input_data(node_input)
except ValidationError as e:
    raise ValidationError.from_exception_data(
        title=f"dynamic node '{node_name or node.name}'",
        line_errors=e.errors(),
    ) from e
```

### Chronological sequence barrier

`ReplayManager` enforces the order in which dynamic node results are replayed
during session resume. After each `__call__` returns, `advance_sequence` is
called; nodes that finish early block on `wait_sequence` until their logical
predecessors have advanced, preserving the deterministic ordering that makes
sessions resumable.

### Example — running nodes dynamically inside a FunctionNode

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.workflow import Workflow, FunctionNode
from google.genai import types


async def classify_and_route(ctx):
    """Dynamically dispatches to a classifier node, then routes."""
    classifier = LlmAgent(
        name="classifier",
        model="gemini-2.5-flash",
        instruction="Classify the input as 'billing', 'tech', or 'other'.",
        output_key="category",
    )
    # ctx.run_node is backed by DynamicNodeScheduler; returns the node output directly
    category = await ctx.run_node(
        classifier,
        node_input=ctx.input,
        node_name="classify",
        run_id="r1",
    )
    ctx.output = f"Routed to: {category}"


router = FunctionNode(name="router", func=classify_and_route)
wf = Workflow(name="dispatch_wf", edges=[("START", router)])


async def main():
    runner = InMemoryRunner(agent=wf, app_name="dyn-demo")
    session = await runner.session_service.create_session(
        app_name="dyn-demo", user_id="u1"
    )
    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="My invoice is wrong.")],
        ),
    ):
        if event.output:
            print(event.output)


asyncio.run(main())
```

`DynamicNodeScheduler` handles the `ctx.run_node("classify", …)` call
transparently: on first invocation it runs fresh; if the session is resumed
(e.g. after a HITL interrupt in the classifier) it fast-forwards using the
cached events without re-running the LLM.

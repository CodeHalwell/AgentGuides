---
title: "smolagents: Comprehensive Technical Guide"
description: "From first agent to multi-agent systems with Hugging Face smolagents — models, tools, CodeAgent, ToolCallingAgent, sandboxes, memory and managed agents."
framework: "smolagents"
language: "python"
sidebar:
  label: "Comprehensive guide"
  order: 1
---

> Verified against `smolagents` %%version=smolagents%% (checked %%versions-checked%%). Examples that need no external service were run with a scripted model (a `Model` subclass that returns prepared replies, shown in the [production guide](/smolagents-guide/smolagents_production_guide/#unit-testing-tools)); the MCP and telemetry examples were run against a local MCP server and an in-memory span exporter; the Hub, Space, Docker, E2B, Modal, Blaxel and browser examples were checked against the installed source for names, signatures and defaults.

smolagents is Hugging Face's small agent library. Its distinctive idea is the **CodeAgent**: instead of emitting one JSON tool call per step, the model writes a short Python snippet that can call several tools, loop, branch and keep intermediate results in variables. A conventional **ToolCallingAgent** is included for cases where JSON tool calls suit better. The library is deliberately compact; the agent logic lives in a single `agents.py` you can read in an afternoon.

The sections below follow the learning path on the [guide overview](/smolagents-guide/); production continues in the [production guide](/smolagents-guide/smolagents_production_guide/). **Also on this page:** [Sandboxes, Hub sharing and multimodal input](#sandboxes-hub-sharing-and-multimodal-input) · [Version notes](#version-notes).

**Elsewhere in this guide:** [Recipes](/smolagents-guide/smolagents_recipes/) for task-shaped patterns and error recovery · [Diagrams](/smolagents-guide/smolagents_diagrams/) for the agent loop and what runs where.

---

## Introduction & Philosophy

### What is smolagents?

An agent in smolagents is a loop: the model sees the task, the tool descriptions and the memory of previous steps; it produces an action; the action is executed; the observation is added to memory; repeat until the model calls `final_answer` or `max_steps` is reached.

The two agent classes differ only in what an action is:

| | `CodeAgent` | `ToolCallingAgent` |
|---|---|---|
| Action | A Python snippet in `<code>...</code>` tags | A JSON tool call (native function calling) |
| Several tools per step | Yes, in ordinary code | Yes, as parallel tool calls |
| Loops, variables, maths | Yes | No (one call, one result) |
| Execution | Python interpreter (local, restricted) or remote sandbox | Direct tool invocation |
| Risk | The model's code runs; sandbox it in production | Only your tools run |

### Why "agents that think in code"?

Code is a better action language than JSON for multi-step work: tool outputs can be stored, filtered and passed to other tools without another model call, and models have seen a great deal of Python. The smolagents authors report that code actions need fewer steps than JSON actions on their benchmarks; the size of the saving depends on the task and model, so measure it on yours.

```python
# What a CodeAgent step might look like (written by the model):
prices = {}
for city in ["Paris", "London", "Tokyo"]:
    prices[city] = get_hotel_price(city=city, nights=3)
cheapest = min(prices, key=prices.get)
final_answer(f"{cheapest} is cheapest at {prices[cheapest]} EUR")
```

---

## Installation & Setup

### Basic Installation

Requires Python 3.10 or later.

```bash
pip install smolagents                              # core
pip install "smolagents[toolkit]"                   # + web search and webpage tools
pip install "smolagents[litellm]"                   # + LiteLLMModel (OpenAI, Anthropic, Gemini, Ollama, ...)
pip install "smolagents[openai]"                    # + OpenAIModel
pip install "smolagents[transformers]"              # + TransformersModel for local inference
pip install "smolagents[mcp]"                       # + MCP client
pip install "smolagents[docker]"                    # or [e2b], [modal], [blaxel]: remote executors
pip install "smolagents[telemetry]"                 # + OpenTelemetry / OpenInference instrumentation
pip install "smolagents[gradio]"                    # + GradioUI

# Pinned to the version this guide was verified against
pip install "smolagents[toolkit,litellm]==%%version=smolagents%%"
```

Other extras: `audio`, `vision` (helium and selenium for the browser agent), `torch`, `vllm`, `mlx-lm`, `bedrock`, `all`.

### Command-line agents

The package installs two commands. `smolagent` runs a `CodeAgent` (or, with `--action-type tool_calling`, a `ToolCallingAgent`) on one prompt; run it with no prompt for an interactive set-up.

```bash
smolagent "Plan a three-day walking trip in the Lake District" \
  --model-type InferenceClientModel --model-id Qwen/Qwen3.8-27B \
  --tools web_search visit_webpage --imports pandas
```

Options: `--model-type` (`InferenceClientModel`, `OpenAIModel`, `LiteLLMModel`, `TransformersModel`), `--model-id`, `--provider`, `--api-base`, `--api-key`, `--tools` (built-in tool names or Space IDs such as `user/space`) and `--imports`. With `InferenceClientModel` the token comes from `--api-key`, then `HF_API_KEY`, then the usual `HF_TOKEN` or `hf auth login`. The CLI's own default `--model-id` is the library default, so pass one. `webagent` starts the [vision browser agent](/smolagents-guide/smolagents_recipes/#recipe-5-vision-web-browsing-agent).

### Verifying Installation

```python
import smolagents
from smolagents import CodeAgent, ToolCallingAgent, InferenceClientModel, LiteLLMModel, tool

print(f"smolagents version: {smolagents.__version__}")
```

### Environment Configuration

Models read credentials from the usual environment variables; keep them in a `.env` file loaded at start-up, never in code.

```bash
# .env
HF_TOKEN=hf_...              # InferenceClientModel (Hugging Face Inference Providers)
OPENAI_API_KEY=sk-...        # OpenAIModel / LiteLLMModel with OpenAI models
ANTHROPIC_API_KEY=...        # LiteLLMModel with anthropic/... models
E2B_API_KEY=...              # executor_type="e2b"
```

```python
from dotenv import load_dotenv

load_dotenv()
```

### Docker Setup (Optional)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt   # smolagents[toolkit,litellm]==%%version=smolagents%%
COPY . .
CMD ["python", "agent_app.py"]
```

Running your *application* in Docker is different from running the *agent's generated code* in Docker; for the latter see [executor_type="docker"](#sandboxes-hub-sharing-and-multimodal-input).

---

## Core Concepts & Architecture

### The Four Pillars of smolagents

```
┌───────────────────────────────────────────────────────────┐
│ Agents          CodeAgent · ToolCallingAgent              │
│                 (both subclass MultiStepAgent)            │
├───────────────────────────────────────────────────────────┤
│ Tools           @tool · Tool subclasses · ToolCollection  │
│                 MCPClient · Tool.from_space · load_tool   │
├───────────────────────────────────────────────────────────┤
│ Models          InferenceClientModel · LiteLLMModel ·     │
│                 OpenAIModel · AzureOpenAIModel ·          │
│                 AmazonBedrockModel · TransformersModel ·  │
│                 VLLMModel · MLXModel · custom Model       │
├───────────────────────────────────────────────────────────┤
│ Execution &     LocalPythonExecutor (default) ·           │
│ memory          Docker · E2B · Modal · Blaxel executors · │
│                 AgentMemory (TaskStep, PlanningStep,      │
│                 ActionStep, FinalAnswerStep)              │
└───────────────────────────────────────────────────────────┘
```

### Agent Execution Flow

1. `agent.run(task)` resets memory (unless `reset=False`) and records a `TaskStep`.
2. Every `planning_interval` steps (if set), the model writes or updates a plan (`PlanningStep`).
3. Each step, memory is rendered into chat messages and sent to the model.
4. **CodeAgent** parses the code block and runs it in the executor; **ToolCallingAgent** runs the requested tool calls (in parallel, up to `max_tool_threads`).
5. Printed output, tool results and errors become the step's observations (`ActionStep`).
6. When the code calls `final_answer(...)` (or the model calls the `final_answer` tool), any `final_answer_checks` run; if they pass, the run ends with a `FinalAnswerStep`.
7. If `max_steps` is reached, the model is asked for a best-effort final answer from memory.

### Core Data Structures

These are real classes exported by `smolagents`:

| Class | Key fields |
|-------|-----------|
| `RunResult` (from `run(..., return_full_result=True)`) | `output`, `state` (`"success"` or `"max_steps_error"`), `steps`, `token_usage`, `timing` |
| `ActionStep` | `step_number`, `model_output`, `code_action`, `tool_calls`, `observations`, `action_output`, `error`, `token_usage`, `timing`, `is_final_answer` |
| `PlanningStep` | `plan`, `token_usage`, `timing` |
| `FinalAnswerStep` | `output` |
| `TokenUsage` | `input_tokens`, `output_tokens`, `total_tokens` |
| `Timing` | `start_time`, `end_time`, `duration` |

---

## Model Configuration & Selection

### Understanding Model Abstraction

Every model is a subclass of `smolagents.Model` implementing `generate(messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs) -> ChatMessage`. Extra keyword arguments given to a model's constructor (temperature, max_tokens and so on) are passed through to the provider on every call.

### InferenceClientModel: Hugging Face Native

Uses `huggingface_hub.InferenceClient`, so it can reach Hugging Face Inference Providers and dedicated endpoints.

```python
import os
from smolagents import InferenceClientModel

model = InferenceClientModel(model_id="Qwen/Qwen3.8-27B")   # provider chosen automatically

model = InferenceClientModel(
    model_id="Qwen/Qwen3.8-27B",
    provider="cerebras",           # route through a specific inference provider
    token=os.environ.get("HF_TOKEN"),
    timeout=120,
    temperature=0.2,               # passed through to the provider
)

# A dedicated Inference Endpoint: pass its URL as the model ID
endpoint_model = InferenceClientModel(model_id="https://my-endpoint.endpoints.huggingface.cloud")
```

`model_id` may be a Hub model ID or the URL of a deployed Inference Endpoint. `InferenceClientModel` also accepts `base_url=`, but only if you do not also give a `model_id`: the constructor's default `model_id` is always passed on, so `InferenceClientModel(base_url=...)` raises `ValueError: Received both model and base_url arguments`. For a self-hosted OpenAI-compatible server (vLLM, LM Studio, a gateway) use `OpenAIModel(api_base=...)` instead. `provider` is ignored when a URL is used. Besides `token`, the constructor takes `api_key` (an alias for the same thing), `bill_to` (bill an organisation you belong to) and `client_kwargs` for the underlying `huggingface_hub.InferenceClient`.

Always pass a `model_id`. The default, `Qwen/Qwen3-Next-80B-A3B-Thinking`, had no live provider when this page was checked on 2026-10-21, so `InferenceClientModel()` with no arguments is likely to fail. A model's "Inference Providers" panel on the Hub, or `https://router.huggingface.co/v1/models`, shows which providers serve it now and whether they support tool calling; providers come and go, so check before you pin one.

`HfApiModel`, the old name for this class, no longer exists: it was renamed to `InferenceClientModel` in 1.14.0 and the deprecated alias was removed in 1.19.0. Agents saved with the old class name are mapped to `InferenceClientModel` automatically when loaded with `from_folder` or `from_hub` (since 1.24.0).

### LiteLLMModel: Universal Provider Support

```python
import os
from smolagents import LiteLLMModel

openai_model = LiteLLMModel(model_id="%%model=openai-small%%")
anthropic_model = LiteLLMModel(model_id="anthropic/%%model=anthropic%%", temperature=0.2)
gemini_model = LiteLLMModel(model_id="gemini/%%model=gemini%%")
ollama_model = LiteLLMModel(model_id="ollama_chat/qwen3:14b", api_base="http://localhost:11434", num_ctx=8192)
azure_model = LiteLLMModel(
    model_id="azure/my-deployment",
    api_base="https://my-resource.openai.azure.com/",
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],   # a current GA version from the Azure docs
)
```

Model names change faster than this guide; use whatever your provider currently recommends. `LiteLLMRouterModel` wraps LiteLLM's router for load-balancing and fallback across deployments.

### OpenAIModel and other API models

`OpenAIModel` talks to any OpenAI-compatible endpoint (OpenAI, vLLM, LM Studio, many gateways) without LiteLLM:

```python
from smolagents import OpenAIModel, AzureOpenAIModel, AmazonBedrockModel

model = OpenAIModel(model_id="%%model=openai-small%%")
local = OpenAIModel(model_id="Qwen/Qwen3-14B", api_base="http://localhost:8000/v1", api_key="not-needed")
```

`OpenAIServerModel` and `AzureOpenAIServerModel` remain as older aliases.

#### Rate limits and retries

`OpenAIModel`, `AzureOpenAIModel`, `LiteLLMModel`, `LiteLLMRouterModel`, `InferenceClientModel` and `AmazonBedrockModel` share a base class, `ApiModel`, which adds two constructor arguments:

```python
model = OpenAIModel(
    model_id="%%model=openai-small%%",
    requests_per_minute=30,   # spaces calls at least two seconds apart; None (default) means no limit
    retry=True,               # default; False makes a single attempt
)
```

- `requests_per_minute` sleeps before each model call so that calls are at least `60 / requests_per_minute` seconds apart. It is per model instance, so agents that share one model object share the limit.
- With `retry=True`, a call that fails with an error whose text contains `429`, `rate limit` or `too many requests` is retried up to three attempts in total. The first wait is 60 seconds, doubled each time with jitter. Other errors are raised immediately. The wait is blocking, so a rate-limited run can pause for minutes without output; set `retry=False` if you would rather handle it yourself, as in the [retry wrapper](#error-handling--recovery) above.

`custom_role_conversions` (same classes) maps smolagents' internal message roles onto the roles an API accepts. By default `tool-call` becomes `assistant` and `tool-response` becomes `user`. A mapping you pass **replaces** that default instead of extending it, so include both entries (`{"tool-call": "assistant", "tool-response": "user"}`) plus whatever you change; a role left out is sent as it is, and most APIs reject `tool-call`. `flatten_messages_as_text=True` sends message content as plain strings instead of lists of typed parts, which some OpenAI-compatible servers require; `TransformersModel` and `MLXModel` choose this themselves for text-only models.

### TransformersModel: Local Inference

```python
from smolagents import TransformersModel
from transformers import BitsAndBytesConfig

model = TransformersModel(
    model_id="Qwen/Qwen3-8B",
    device_map="auto",
    torch_dtype="bfloat16",
    max_new_tokens=2048,
    model_kwargs={"quantization_config": BitsAndBytesConfig(load_in_4bit=True)},   # forwarded to from_pretrained (needs bitsandbytes)
)
```

For high-throughput local serving prefer `VLLMModel`; on Apple silicon, `MLXModel`.

### Custom Model Implementation

Subclass `Model` and implement `generate`, returning a `ChatMessage`:

```python
import requests

from smolagents import ChatMessage, Model
from smolagents.models import MessageRole
from smolagents.monitoring import TokenUsage


class TinyOllamaModel(Model):
    """Minimal Ollama /api/chat client (no tool-calling support)."""

    def __init__(self, model_id: str, base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model_id=model_id, **kwargs)
        self.base_url = base_url

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": str(m.role.value if hasattr(m.role, "value") else m.role),
                 "content": m.content if isinstance(m.content, str)
                 else "".join(part.get("text", "") for part in (m.content or []))}
                for m in messages
            ],
            "stream": False,
            "options": {"stop": stop_sequences or []},
        }
        data = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120).json()
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=data["message"]["content"],
            token_usage=TokenUsage(data.get("prompt_eval_count", 0), data.get("eval_count", 0)),
        )
```

In practice `LiteLLMModel(model_id="ollama_chat/...")` or `OpenAIModel(api_base=".../v1")` already covers Ollama; write a custom model only for in-house APIs.

### Model Selection Decision Tree

- **Exploring, free tier**: `InferenceClientModel(model_id=...)` with an `HF_TOKEN` (Hugging Face accounts include some Inference Providers usage; check the current allowance on the Hub).
- **Hosted frontier models**: `LiteLLMModel` or `OpenAIModel`.
- **Your own GPU**: `VLLMModel` (throughput) or `TransformersModel` (simplicity).
- **Offline laptop**: `LiteLLMModel(model_id="ollama_chat/...")` or `MLXModel`.
- **CodeAgent quality**: pick a model that is strong at writing Python; small models often produce malformed code blocks.

---

## Simple Agents Fundamentals

### Creating Your First CodeAgent

`tools` and `model` are both required (pass `tools=[]` for none). The examples from here on reuse this `model`.

```python
from smolagents import CodeAgent, InferenceClientModel

model = InferenceClientModel(model_id="Qwen/Qwen3.8-27B")
agent = CodeAgent(tools=[], model=model)
result = agent.run("What is the 20th Fibonacci number?")
print(result)   # 6765
```

### Adding Tools to Agents

```python
from smolagents import CodeAgent, VisitWebpageTool, WebSearchTool

agent = CodeAgent(
    tools=[WebSearchTool(), VisitWebpageTool()],     # needs smolagents[toolkit]
    model=model,
)
print(agent.run("Find the population of Tokyo and calculate 2% of it."))
```

`add_base_tools=True` adds the default toolbox instead: `web_search` and `visit_webpage` (and `python_interpreter` for a `ToolCallingAgent`; a `CodeAgent` already executes Python).

### Agent Initialisation Parameters

```python
from smolagents import CodeAgent

agent = CodeAgent(
    tools=[],                          # required
    model=model,                       # required
    instructions="Answer in British English. Cite sources.",   # appended to the system prompt
    max_steps=10,                      # default 20
    planning_interval=3,               # write/update a plan every 3 steps
    additional_authorized_imports=["pandas", "numpy"],          # imports the generated code may use
    executor_type="local",             # "local", "docker", "e2b", "modal" or "blaxel"
    executor_kwargs={"timeout_seconds": 60},   # passed to the executor
    max_print_outputs_length=10_000,   # truncate long print() output
    stream_outputs=False,              # stream model tokens while generating
    verbosity_level=1,                 # LogLevel: -1 off, 0 errors only, 1 info (default), 2 debug
    name="research_assistant",         # needed only when used as a managed agent
    description="Finds information on the web.",
    final_answer_checks=[],            # validation functions, see below
    step_callbacks=[],                 # called after each step
    return_full_result=False,          # default for run()
)
```

`ToolCallingAgent` takes the same base parameters (minus the code-execution ones) plus `max_tool_threads`.

### The `run()` Method: Core Execution Interface

```python
agent.run(
    task="Your task",
    stream=False,              # True returns a generator of steps and events
    reset=True,                # False continues from the existing memory
    images=None,               # list of PIL images for vision models
    additional_args=None,      # dict of Python objects made available to the agent
    max_steps=None,            # override for this run
    return_full_result=None,   # True returns a RunResult
)
```

Keeping context across turns:

```python
agent.run("My name is Alice and I live in Leeds.")
print(agent.run("Where do I live?", reset=False))
```

Passing data without pasting it into the prompt:

```python
import pandas as pd

df = pd.DataFrame({"month": ["Jan", "Feb"], "sales": [100, 150]})
agent = CodeAgent(tools=[], model=model, additional_authorized_imports=["pandas"])
agent.run("What is the month-on-month growth in `sales_df`?", additional_args={"sales_df": df})
```

### Understanding Agent Output

```python
result = agent.run("Calculate 15 * 12", return_full_result=True)
print(result.output)                  # 180
print(result.state)                   # "success" or "max_steps_error"
print(result.token_usage)             # TokenUsage(input_tokens=..., output_tokens=..., total_tokens=...)
print(result.timing.duration)         # seconds
print(len(result.steps))              # every memory step as a dict, including the task and any plan

# After any run, the full memory is on the agent
for step in agent.memory.steps:
    print(type(step).__name__)
agent.replay()                        # pretty-print the last run
```

### Memory and steps

An agent's memory is `agent.memory`, an `AgentMemory` holding the system prompt and a list of steps. Each `run()` appends to it (after clearing it, unless you pass `reset=False`). The step types are:

| Step | Added when | Useful fields |
|------|-----------|---------------|
| `TaskStep` | `run()` starts | `task`, `task_images` |
| `PlanningStep` | A plan is written (see [Planning](#planning)) | `plan`, `token_usage`, `timing` |
| `ActionStep` | Each pass of the loop | `step_number`, `model_output`, `code_action` (CodeAgent), `tool_calls`, `observations`, `observations_images`, `error`, `action_output`, `token_usage`, `timing`, `is_final_answer` |

`FinalAnswerStep` is yielded by `run(..., stream=True)` and passed to step callbacks, but it is not stored in `memory.steps`. On every model call the steps are turned back into chat messages: the task as a user message, the model's output as an assistant message, tool calls and `Observation:` text as tool messages, and errors with a "now let's retry" instruction. That is all the model knows about earlier steps, so whatever you change in memory changes what it sees next.

```python
from smolagents import ActionStep, CodeAgent, PlanningStep, TaskStep

agent = CodeAgent(tools=[], model=model, planning_interval=3)
agent.run("What is the sum of the squares of 1 to 10?")

for step in agent.memory.steps:
    if isinstance(step, TaskStep):
        print("task:", step.task)
    elif isinstance(step, PlanningStep):
        print("plan:", step.plan[:80])
    elif isinstance(step, ActionStep):
        print(step.step_number, "code:", step.code_action, "| error:", step.error)

print(agent.memory.return_full_code())        # every code action joined into one script
rows = agent.memory.get_succinct_steps()      # list of dicts, without the prompt messages
full = agent.memory.get_full_steps()          # the same with model_input_messages (large)
agent.write_memory_to_messages()              # the chat messages the next model call would receive
```

Three practical uses:

- **Continuing a conversation.** `run(task, reset=False)` keeps the steps, so a follow-up question can refer to earlier answers. Memory is in the process only; to survive a restart, store `get_full_steps()` (or just the tasks and answers) yourself and rebuild the context in a new task.
- **Keeping memory small.** Memory grows with every step and is sent in full each time. A step callback can edit old steps, for example setting `observations_images = None` on screenshots (see [Advanced CodeAgent patterns](#advanced-codeagent-patterns)). Do not remove the `TaskStep`.
- **Debugging.** `agent.replay()` pretty-prints the run; `agent.replay(detailed=True)` also prints the messages sent at each step, which is long.

`RunResult.steps` (from `return_full_result=True`) is the same information as dicts, so it contains the task and any plan as well as the action steps. To count only the action steps use `sum("step_number" in s for s in result.steps)`.

### Planning

By default an agent starts acting straight away. Set `planning_interval=N` (on either agent class) and it first writes a plan, then revises it every N steps: with `planning_interval=3` the plan is written before steps 1, 4, 7 and so on. Planning is a separate model call that costs tokens each time, so it pays off on long, multi-part tasks and is wasted on short ones.

How it works, as implemented in `MultiStepAgent`:

- The first plan comes from a single prompt (`prompt_templates["planning"]["initial_plan"]`) that contains the task, the tool descriptions and any managed agents, and nothing else: not the system prompt. The model is asked to stop at `<end_plan>`.
- Later plans are updates. The model receives the memory so far (without the system prompt or earlier plans, so as not to anchor on them), framed by `update_plan_pre_messages` and `update_plan_post_messages`; the latter can use `remaining_steps`.
- The result is stored as a `PlanningStep`. On later model calls it appears in the conversation as an assistant message, followed by a user message "Now proceed and carry out this plan.".
- `PlanningStep` is yielded when streaming and passed to step callbacks registered for it, which is how you [review or edit a plan before it is used](/smolagents-guide/smolagents_recipes/#recipe-21-human-in-the-loop-plan-review).

The planning prompts are Jinja templates in `agent.prompt_templates["planning"]`, so you can change them after construction. The variables available are `task`, `tools` (a dict of tool name to tool) and `managed_agents` for the initial plan, and the same plus `remaining_steps` for the update.

```python
from smolagents import CodeAgent, PlanningStep

agent = CodeAgent(tools=[], model=model, planning_interval=3, max_steps=9)

agent.prompt_templates["planning"]["initial_plan"] = (
    "Write a numbered plan of at most five steps for this task: {{ task }}\n"
    "Available tools: {% for t in tools.values() %}{{ t.name }} {% endfor %}\n"
    "Finish with <end_plan>."
)

agent.run("Work out how many days there are between 3 March and 19 October in a non-leap year.")
plans = [s.plan for s in agent.memory.steps if isinstance(s, PlanningStep)]
```

The stored `plan` text is the model's output wrapped in a fixed sentence ("Here are the facts I know and the plan of action that I will follow to solve the task:") and a code fence. A `ToolCallingAgent` plans in exactly the same way.

### Streaming Outputs for Real-Time Visibility

`run(..., stream=True)` yields steps and events as they happen: `PlanningStep`, `ActionStep` and `FinalAnswerStep`, plus `ToolCall` and `ActionOutput` events within a step (a `ToolCallingAgent` also yields `ToolOutput`). With `stream_outputs=True` on the agent you also receive `ChatMessageStreamDelta` token chunks. Events you do not handle can be ignored, as below. `stream_outputs=True` needs a model that implements `generate_stream`; the built-in models do, and a custom `Model` subclass that does not makes the agent raise `ValueError` at construction.

```python
from smolagents import ActionStep, ChatMessageStreamDelta, FinalAnswerStep, PlanningStep

agent = CodeAgent(tools=[], model=model, stream_outputs=True)

for event in agent.run("Compute the sum of the first 100 primes.", stream=True):
    if isinstance(event, ChatMessageStreamDelta):
        print(event.content or "", end="", flush=True)
    elif isinstance(event, ActionStep):
        print(f"\n[step {event.step_number}] observations: {(event.observations or '')[:200]}")
    elif isinstance(event, PlanningStep):
        print(f"\n[plan]\n{event.plan}")
    elif isinstance(event, FinalAnswerStep):
        print(f"\n[final] {event.output}")
```

### Error Handling & Recovery

Inside a run, errors (bad code, tool exceptions, parsing failures) are recorded in the step's `error` and shown to the model, which usually corrects itself on the next step. They surface to you in two ways:

- `result.state == "max_steps_error"` when the agent ran out of steps (the output is then a best-effort answer).
- Exceptions from the model call or your own callbacks propagate.

The hierarchy:

```text
AgentError
├── AgentParsingError
├── AgentExecutionError
│   ├── AgentToolCallError
│   └── AgentToolExecutionError
├── AgentMaxStepsError
└── AgentGenerationError        (raised from run(); the others are recorded and shown to the model)
```

```python
import time

from smolagents import AgentGenerationError


def run_with_retry(agent, task: str, attempts: int = 3):
    for attempt in range(1, attempts + 1):
        try:
            result = agent.run(task, return_full_result=True)
            if result.state == "success":
                return result.output
            print(f"Attempt {attempt}: ran out of steps")
        except AgentGenerationError as exc:          # provider/API failure
            print(f"Attempt {attempt}: model error: {exc}")
            time.sleep(2 ** attempt)
    raise RuntimeError("Agent did not succeed")
```

---

## Tools: Building Blocks of Agents

### The Tool Abstraction

A tool has a `name`, a `description`, an `inputs` schema, an `output_type`, and a `forward()` method. The agent sees the name, description and schema; for a `CodeAgent` they are rendered as Python function signatures.

Allowed types for inputs and outputs: `string`, `boolean`, `integer`, `number`, `image`, `audio`, `array`, `object`, `any`, `null`.

### @tool Decorator: Lightweight Tools

The decorator builds the schema from type hints and a Google-style docstring. **Every argument must be documented in an `Args:` section**, or the decorator raises an error.

```python
from smolagents import CodeAgent, tool


@tool
def get_hotel_price(city: str, nights: int = 1) -> float:
    """Return the cheapest hotel price for a stay.

    Args:
        city: City name, e.g. "Paris".
        nights: Number of nights.
    """
    return {"paris": 140.0, "london": 165.0}.get(city.lower(), 120.0) * nights


agent = CodeAgent(tools=[get_hotel_price], model=model)
print(agent.run("Is three nights cheaper in Paris or London?"))
```

Good tool hygiene:

- The description is the model's only documentation. Say what the tool returns and in what format.
- Use precise types; `float` and `int` are not interchangeable in the schema.
- Return plain data or short strings; huge outputs fill the context window.
- Raise informative exceptions (`ValueError("Unknown city 'Pariss'; did you mean 'Paris'?")`); the message is shown to the model.

### Tool Subclass: Stateful Tools

Subclass `Tool` for tools that hold connections or configuration. Optional inputs must be marked `"nullable": True` and have a default in `forward`.

```python
import sqlite3

from smolagents import Tool


class SQLQueryTool(Tool):
    name = "sql_query"
    description = (
        "Run a read-only SQL SELECT against the customers database. "
        "Table: customers(id INTEGER, name TEXT, email TEXT). Returns a list of rows as dicts."
    )
    inputs = {
        "query": {"type": "string", "description": "A single SELECT statement."},
        "limit": {"type": "integer", "description": "Maximum rows to return.", "nullable": True},
    }
    output_type = "array"

    def __init__(self, path: str = ":memory:"):
        super().__init__()
        self.path = path
        self.conn = None

    def setup(self):
        # Called once, on first use
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("CREATE TABLE IF NOT EXISTS customers(id INTEGER, name TEXT, email TEXT)")
        self.conn.execute("INSERT INTO customers VALUES (1, 'Ada', 'ada@gmail.com'), (2, 'Bo', 'bo@example.com')")

    def forward(self, query: str, limit: int | None = None):
        if not query.lstrip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT statements are allowed.")
        rows = self.conn.execute(query).fetchall()
        rows = rows[:limit] if limit else rows
        return [dict(r) for r in rows]


agent = CodeAgent(tools=[SQLQueryTool()], model=model)
```

### Default Toolbox: Built-in Tools

| Tool | Name seen by the model | Notes |
|------|------------------------|-------|
| `WebSearchTool(max_results=10, engine="duckduckgo")` | `web_search` | Engines: `duckduckgo`, `bing`, `exa` (Exa added in 1.26.0) |
| `DuckDuckGoSearchTool` | `web_search` | Needs `ddgs` (in `[toolkit]`) |
| `GoogleSearchTool(provider="serpapi")` | `web_search` | SerpAPI or Serper key |
| `ApiWebSearchTool(endpoint=..., api_key=...)` | `web_search` | Brave Search API by default; any JSON search API via `endpoint` |
| `VisitWebpageTool` | `visit_webpage` | Returns page as Markdown |
| `WikipediaSearchTool` | `wikipedia_search` | Needs `wikipedia-api` |
| `PythonInterpreterTool` | `python_interpreter` | For `ToolCallingAgent` |
| `SpeechToTextTool` | `transcriber` | Whisper via Transformers (`[audio]`) |
| `UserInputTool` | `user_input` | Asks the human a question |
| `FinalAnswerTool` | `final_answer` | Added to every agent automatically |

### Tools from the Hub, Spaces, LangChain and MCP

```python
from smolagents import Tool, ToolCollection, load_tool

# A tool someone pushed to the Hub (runs their code: inspect it first)
image_gen = load_tool("your-username/text-to-image-tool", trust_remote_code=True)

# A Gradio Space as a tool
flux = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generate an image from a prompt",
)

# Every tool in a Hub collection
collection = ToolCollection.from_hub("your-username/your-tool-collection-slug", trust_remote_code=True)

# A LangChain tool
# lc_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

# Share your own tool
get_hotel_price.push_to_hub("your-username/hotel-price-tool")
```

A tool written with `@tool` or as a `Tool` subclass can be written to a folder with `tool.save("./my_tool")` (which produces `tool.py`, `app.py` and `requirements.txt`) and shared with `tool.push_to_hub(repo_id)`. Saving copies the function's source and nothing else, so the tool must be self-contained:

- Module-level variables and helper functions are not carried along, and `save()` fails with a validation error naming them (`Name 'RATES' is undefined`).
- Module-level imports are also not copied, and `save()` does not always notice: a tool that calls `math.sqrt` but imports `math` at the top of your script saves without error and fails when loaded. Put imports inside `forward` (or inside the decorated function).
- `Tool.from_code(source)` rebuilds a tool from the saved `tool.py` text.

The same rule applies when a tool runs in a [remote sandbox](#sandboxes-hub-sharing-and-multimodal-input), where the tool is rebuilt from its source.

A tool does not need to be on the Hub to be reused: a plain Python module that defines it is usually the better place for tools your own team shares.

#### MCP servers

`MCPClient` connects to one or more [Model Context Protocol](https://modelcontextprotocol.io/) servers and turns each of their tools into a smolagents `Tool` (`pip install "smolagents[mcp]"`). It connects when constructed and, used as a context manager, returns the list of tools and disconnects on exit. The server below is a complete local example, written with the MCP Python SDK's `FastMCP` helper (`mcp<2`, see the note after the example):

```python
# demo_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


if __name__ == "__main__":
    mcp.run()          # stdio transport
```

```python
import sys

from mcp import StdioServerParameters
from smolagents import CodeAgent, MCPClient

server = StdioServerParameters(command=sys.executable, args=["demo_server.py"])

with MCPClient(server, structured_output=True) as tools:
    agent = CodeAgent(tools=tools, model=model)
    print(agent.run("What is 2 + 3? Use the add tool."))
```

Notes on behaviour, all checked against this server:

- **Transports.** A `StdioServerParameters` starts the server as a subprocess and talks over stdin/stdout (`uvx mcp-server-fetch` and `npx` servers work this way). A dict `{"url": "http://localhost:8000/mcp", "transport": "streamable-http"}` connects to a running server; `"streamable-http"` is the default when `transport` is omitted, and `"sse"` is also accepted. Pass a **list** of either kind to connect several servers at once and get all their tools in one list.
- **Structured output.** With `structured_output=True` the tool's declared output schema is shown to the model, and results arrive as dicts (the `add` tool above returns `{'result': 5}`, and its `output_type` is `"object"`). With `False` you get the server's text content. Leaving it unset behaves as `False` but emits a `FutureWarning` announcing that the default will become `True`; the 1.26.0 code still defaults to `False`, so set it explicitly.
- **Lifecycle without `with`.** `client = MCPClient(server, structured_output=True)`, then `client.get_tools()`, and always `client.disconnect()` in a `finally` block. The tool list is fixed at connection time.
- **Trust.** A stdio server runs with your user's permissions, and a remote server's tool names and descriptions go straight into the agent's prompt. Connect only to servers you trust, and prefer a [sandboxed executor](#sandboxes-hub-sharing-and-multimodal-input) when the generated code will call them.
- **Older form.** `ToolCollection.from_mcp(server, trust_remote_code=True, structured_output=True)` is a context manager that yields a collection; pass its `.tools` to the agent. It requires `trust_remote_code=True` for the same reason as above.

> **Compatibility note (checked %%versions-checked%%):** the `mcpadapt` 0.1.20 dependency imports `streamablehttp_client`, which the `mcp` 2.x SDK renamed. With `mcp` 2.3.0 installed, `MCPClient(...)` fails with `ImportError: cannot import name 'streamablehttp_client'`. Install `"mcp<2"` alongside `smolagents[mcp]` until `mcpadapt` catches up; the example above was run with `mcp` 1.30.0.

### Tool Attributes & Schema

```python
print(get_hotel_price.name)          # get_hotel_price
print(get_hotel_price.inputs)        # {'city': {'type': 'string', 'description': ...}, 'nights': {...}}
print(get_hotel_price.output_type)   # number
print(get_hotel_price.to_code_prompt())          # how a CodeAgent sees it
print(get_hotel_price.to_tool_calling_prompt())  # how a ToolCallingAgent sees it
```

---

## CodeAgent: The Code-Based Paradigm

### Understanding CodeAgent

Each step, the model writes a thought and a code block between `<code>` and `</code>` (configurable with `code_block_tags`, e.g. `"markdown"` for triple-backtick fences). The code runs in a persistent interpreter: variables survive between steps, `print()` output becomes the observation, and calling `final_answer(x)` ends the run with `x`.

### CodeAgent vs Traditional JSON Agents

A JSON agent asking for three cities' weather needs three tool calls (possibly parallel) and another model call to compare them. A CodeAgent can do it in one step:

```python
temps = {c: get_temperature(city=c) for c in ["Paris", "Rome", "Oslo"]}
warmest = max(temps, key=temps.get)
final_answer(f"{warmest} ({temps[warmest]}°C)")
```

The trade-off is safety: you are executing model-written code.

### Code Generation & Execution Flow

The default `LocalPythonExecutor` is **not a sandbox**. It interprets the code's AST and blocks unlisted imports, dunder access and some dangerous builtins, which stops accidents, but it is not a security boundary. Treat it as a convenience for trusted, local use. For anything exposed to untrusted input, use a remote executor (see [Sandboxes](#sandboxes-hub-sharing-and-multimodal-input)).

```python
agent = CodeAgent(
    tools=[],
    model=model,
    additional_authorized_imports=["numpy", "pandas"],   # "*" allows everything: avoid
    executor_kwargs={"timeout_seconds": 30},                        # per-execution time limit (local executor)
    max_print_outputs_length=5_000,
)
```

What the local executor does and does not do, as implemented in `LocalPythonExecutor`:

- **Imports.** Generated code may import only `collections`, `datetime`, `itertools`, `math`, `queue`, `random`, `re`, `stat`, `statistics`, `time` and `unicodedata`, plus whatever you add in `additional_authorized_imports`. Anything else fails with an `InterpreterError` that the model sees, so it can choose another approach. An authorised module is trusted with everything it can do: with only `pandas` authorised, generated code cannot call `open`, but `df.to_csv("/some/path")` still writes a file. Authorising `"*"`, or modules such as `os` and `subprocess`, removes the guard altogether.
- **Limits.** Execution stops after 30 seconds by default (`timeout_seconds`; `None` disables it), after 10,000,000 interpreted operations, and after 1,000,000 iterations of one `while` loop. The timeout runs the code in a worker thread and gives up waiting; Python cannot kill a thread, so a runaway computation keeps using CPU in the background after the agent has moved on.
- **State.** Variables, functions and imports persist between steps in the same run, and tools are callable as functions. `additional_args` are placed in the same namespace.
- **Custom executor.** `CodeAgent(executor=my_executor)` takes any `PythonExecutor` instance in place of `executor_type`; most readers will not need it.

### Advanced CodeAgent Patterns

**Planning.** `planning_interval=N` makes the agent write a plan before the first step and revise it every N steps; see [Planning](#planning). It costs an extra model call each time.

**Final answer checks.** Each check receives the proposed answer and the agent's memory; return `True` to accept or raise/return `False` to send the agent back to work with the error message.

```python
def must_be_number(final_answer, memory, agent=None) -> bool:
    float(final_answer)       # raises ValueError for non-numbers
    return True

agent = CodeAgent(tools=[], model=model, final_answer_checks=[must_be_number])
```

**Step callbacks.** Called after each step with `(memory_step, agent=...)`; a callback that takes only one parameter receives just the step. Pass a **list**, which registers the callbacks for `ActionStep` only, or a **dict** keyed by step class to choose the step types. Matching follows the class hierarchy, so a callback registered for `MemoryStep` (the base class, importable from `smolagents.memory`) runs for every step the agent reports: each `PlanningStep` and `ActionStep`, and the closing `FinalAnswerStep`. (`TaskStep` is not passed to callbacks.)

```python
from smolagents import ActionStep, PlanningStep


def log_action(step: ActionStep, agent=None):
    print(f"step {step.step_number}: {step.token_usage}")


def trim_old_images(step: ActionStep, agent=None):
    # Keep memory small for vision agents: drop screenshots older than two steps
    for old in agent.memory.steps:
        if isinstance(old, ActionStep) and old.step_number <= step.step_number - 2:
            old.observations_images = None


agent = CodeAgent(
    tools=[],
    model=model,
    step_callbacks={ActionStep: [log_action, trim_old_images], PlanningStep: lambda s, agent=None: print(s.plan)},
)
```

**Custom prompts.** `agent.prompt_templates` is a `PromptTemplates` dict (`system_prompt`, `planning`, `managed_agent`, `final_answer`); `prompt_templates["system_prompt"]` is a Jinja template. Override it, pass `prompt_templates=` at construction, or simply use `instructions=` for additions.

**Interrupting.** `agent.interrupt()` (from a callback or another thread) stops the run before the next step by raising `AgentError("Agent interrupted.")` from `run()`.

**Inspecting structure.** `agent.visualize()` prints a tree of the agent, its tools and its managed agents; `agent.replay()` re-prints the last run.

### When CodeAgent saves steps

Code actions help most when a task needs many tool calls whose results feed each other (aggregation, loops over items, arithmetic). For single-tool lookups the difference is negligible. Measure with `result.token_usage` and the number of action steps (see [Memory and steps](#memory-and-steps)) on your own tasks before choosing.

---

## ToolCallingAgent: Traditional JSON-Based Workflows

### Understanding ToolCallingAgent

`ToolCallingAgent` uses the provider's native function calling. It never executes model-written code, so it needs no sandbox, and it works with models that are better at tool calling than at Python.

### When to Use ToolCallingAgent vs CodeAgent

| Choose `ToolCallingAgent` when | Choose `CodeAgent` when |
|---|---|
| You cannot sandbox generated code | You can sandbox, or the input is trusted |
| Each step is one or a few independent tool calls | Steps combine and transform tool outputs |
| The model is weak at Python | The model is strong at Python |
| You want strict schemas on every action | You want loops, maths and variables |

### ToolCallingAgent Implementation

```python
from smolagents import LiteLLMModel, ToolCallingAgent, WebSearchTool, tool


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    """Convert an amount between currencies using a fixed demo rate table.

    Args:
        amount: The amount to convert.
        from_currency: ISO code, e.g. "USD".
        to_currency: ISO code, e.g. "EUR".
    """
    rates = {("USD", "EUR"): 0.92, ("EUR", "USD"): 1.09}
    return round(amount * rates[(from_currency.upper(), to_currency.upper())], 2)


agent = ToolCallingAgent(
    tools=[WebSearchTool(), convert_currency],
    model=LiteLLMModel(model_id="%%model=openai-small%%"),
    max_tool_threads=4,        # parallel tool calls within a step
    max_steps=8,
)
print(agent.run("How much is 250 USD in EUR?"))
```

### ToolCallingAgent with Specialized Models

Any model that supports function calling works. For open models via Inference Providers, choose a provider that lists tool support for the model, for example `InferenceClientModel(model_id="Qwen/Qwen3.8-27B", provider="deepinfra")` (live, with tool support, on 2026-10-21), or `provider="auto"` to let the router choose. If a model returns tool calls as text rather than structured calls, smolagents attempts to parse them, but a model with native tool calling is far more reliable.

---

## Multi-Agent Systems & Orchestration

### Managed Agents for Hierarchical Systems

Any agent with a `name` and `description` can be given to another agent through `managed_agents`. The manager calls it like a tool, passing a `task` string (and optionally `additional_args`); the managed agent's final answer comes back as the result, together with a summary of its work if `provide_run_summary=True`.

```python
from smolagents import CodeAgent, InferenceClientModel, ToolCallingAgent, VisitWebpageTool, WebSearchTool

model = InferenceClientModel(model_id="Qwen/Qwen3.8-27B")

web_agent = ToolCallingAgent(
    tools=[WebSearchTool(), VisitWebpageTool()],
    model=model,
    max_steps=10,
    name="web_researcher",
    description="Searches the web and reads pages. Give it a precise research question.",
)

manager = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[web_agent],
    additional_authorized_imports=["statistics"],
    planning_interval=4,
)

print(manager.run("Find the populations of the five largest EU cities and compute their mean."))
```

Names must be valid Python identifiers (no spaces) and both `name` and `description` are required, otherwise the manager's constructor raises. A `CodeAgent` manager calls `web_researcher(task="...")` in its code; a `ToolCallingAgent` manager issues a tool call named `web_researcher` with a `task` argument. Managed agents do not appear in the manager's `tools`; they are listed in `manager.managed_agents`.

What to know about delegation:

- **The manager sees a string.** The sub-agent's `final_answer` is wrapped in a report ("Here is the final answer from your managed agent 'web_researcher': ..."). Anything the sub-agent computed but did not return is lost, so write the brief and the `description` to say what the answer must contain. With `provide_run_summary=True` the report also carries a truncated summary of the sub-agent's steps, which costs manager context.
- **Each call starts fresh.** A delegation runs the sub-agent with `reset=True`, so it does not remember earlier delegations. The manager can pass data along with `web_researcher(task="...", additional_args={"urls": urls})`.
- **Each agent has its own budget.** `max_steps`, `planning_interval`, tools and model are per agent, and the tokens of every sub-agent run count toward your bill but not toward the manager's `RunResult.token_usage`.

A `ToolCallingAgent` can manage a `CodeAgent` just as well. This suits a coordinator that never executes code itself but delegates to an analyst that does. The constructor's remote-executor restriction (see [Remote executors](#remote-executors)) applies to the agent that *has* managed agents, so the analyst here could be given `executor_type="docker"` while the coordinator, which has none of its own code to run, stays as it is:

```python
from smolagents import CodeAgent, ToolCallingAgent, tool


@tool
def lookup_population(city: str) -> int:
    """Look up a city's population (demo data).

    Args:
        city: City name, e.g. "Lyon".
    """
    return {"Lyon": 522_000, "Paris": 2_100_000}[city]


analyst = CodeAgent(
    tools=[lookup_population],
    model=model,
    name="analyst",
    description="Looks up and compares city populations. Give it a list of city names.",
    max_steps=4,
)
coordinator = ToolCallingAgent(tools=[], model=model, managed_agents=[analyst])
print(coordinator.run("Which is larger, Lyon or Paris, and by how much?"))
```

### Agent Collaboration Patterns

**Sequential hand-off** in plain Python:

```python
researcher = CodeAgent(tools=[WebSearchTool()], model=model)
writer = CodeAgent(tools=[], model=model)

notes = researcher.run("Collect five recent facts about perovskite solar cells, with sources.")
article = writer.run("Write a 200-word explainer from these notes.", additional_args={"notes": notes})
```

**Parallel fan-out.** Agents keep state in memory, so give each thread its own agent instance:

```python
from concurrent.futures import ThreadPoolExecutor


def research(question: str) -> str:
    agent = CodeAgent(tools=[WebSearchTool()], model=model, verbosity_level=0)
    return agent.run(question)


questions = ["UK heat-pump installs 2025", "German heat-pump subsidies", "French heat-pump market size"]
with ThreadPoolExecutor(max_workers=3) as pool:
    answers = list(pool.map(research, questions))
```

**Hierarchies.** Managed agents can have their own managed agents; keep depth to two levels and give each level a narrow brief and its own `max_steps`.

---

## Structured Outputs & Schema

### Defining Output Schemas

There are three distinct features; do not confuse them.

**1. Validating the final answer.** Ask for JSON in the task and enforce it with a final-answer check:

```python
import json

from pydantic import BaseModel, ValidationError


class CityReport(BaseModel):
    city: str
    population: int
    sources: list[str]


def is_city_report(final_answer, memory, agent=None) -> bool:
    data = final_answer if isinstance(final_answer, dict) else json.loads(final_answer)
    CityReport.model_validate(data)        # raises ValidationError with a useful message
    return True


agent = CodeAgent(
    tools=[WebSearchTool()],
    model=model,
    final_answer_checks=[is_city_report],
    instructions="Return your final answer as a dict with keys city, population, sources.",
)
report = CityReport.model_validate(agent.run("Population of Lyon?"))
```

**2. Documenting a tool's output.** `Tool.output_schema` (or an MCP tool's schema with `structured_output=True`) is shown to a CodeAgent so it knows which keys a returned dict has. It is informational; smolagents does not validate it.

```python
from smolagents import Tool


class StatsTool(Tool):
    name = "describe_numbers"
    description = "Summary statistics for a list of numbers."
    inputs = {"data": {"type": "array", "description": "Numbers to summarise."}}
    output_type = "object"
    output_schema = {
        "type": "object",
        "properties": {"mean": {"type": "number"}, "median": {"type": "number"}, "stdev": {"type": "number"}},
        "required": ["mean", "median", "stdev"],
    }

    def forward(self, data: list):
        import statistics
        values = [float(x) for x in data]
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
```

**3. Structured actions.** `CodeAgent(use_structured_outputs_internally=True)` makes the model return each step as JSON with `thought` and `code` fields (via the provider's `response_format`), which reduces parsing errors with models that support it. It changes how actions are formatted, not the shape of your final answer.

---

## Sandboxes, Hub sharing and multimodal input

### Remote executors

| `executor_type` | Install | Isolation | Notes |
|-----------------|---------|-----------|-------|
| `"local"` (default) | — | None (AST-restricted interpreter in your process) | Development and trusted input only |
| `"docker"` | `smolagents[docker]` | Container running a Jupyter kernel gateway, bound to `127.0.0.1:8888` with a random auth token | Builds the `jupyter-kernel` image (from `python:3.12-bullseye`) every time the executor starts unless you pass `build_new_image=False`; other `executor_kwargs`: `image_name`, `dockerfile_content`, `container_run_kwargs`, `host`, `port` |
| `"e2b"` | `smolagents[e2b]` + `E2B_API_KEY` | Managed cloud sandbox | Pay per use |
| `"modal"` | `smolagents[modal]` | Modal sandbox | Needs a Modal account; `executor_kwargs`: `app_name`, `port`, `create_kwargs` |
| `"blaxel"` | `smolagents[blaxel]` | Blaxel sandbox | Needs a Blaxel account; `executor_kwargs`: `sandbox_name`, `image`, `memory`, `ttl`, `region` |

```python
agent = CodeAgent(tools=[], model=model, executor_type="docker")
try:
    agent.run("Plot a sine wave and describe it.")
finally:
    agent.cleanup()      # stops the remote kernel/container
```

The authorised imports are `pip install`ed inside the sandbox when it starts, so list only what the task needs. `timeout_seconds` belongs to the local executor and is not an option for the remote ones. Remote executors cannot be combined with `managed_agents` (the constructor raises "Managed agents are not yet supported with remote code execution"); run multi-agent systems with the local executor inside a sandboxed container instead. Tools are rebuilt from their source inside the sandbox and run there, so they must be self-contained: imports inside `forward`, and any credentials they need available in the sandbox. The [diagram](/smolagents-guide/smolagents_diagrams/#where-code-and-tools-run) shows what runs where. Since 1.25.0, pickle-based serialisation between agent and sandbox is off by default. The experimental `"wasm"` executor was removed in 1.26.0.

### Hub sharing

```python
agent.push_to_hub("your-org/research-agent")                      # saves tools, prompts and config
agent2 = CodeAgent.from_hub("your-org/research-agent", trust_remote_code=True)

agent.save("./research_agent")                                     # to a local folder
agent3 = CodeAgent.from_folder("./research_agent")
```

`save()` writes `agent.json`, `prompts.yaml`, `requirements.txt`, an `app.py` that launches the agent in a Gradio UI, and one file per tool under `tools/`. It stores the model's class and constructor arguments (never the token or API key), prompt templates, `max_steps`, `planning_interval`, name and description, authorised imports and executor settings. It does not store `instructions`, `final_answer_checks` or `step_callbacks`, so pass those again after loading. `from_folder` takes no `trust_remote_code` argument; `from_hub` does.

Loading an agent or tool runs code from the repository; read it first and only set `trust_remote_code=True` for sources you trust.

### Multimodal input

Pass PIL images to vision-capable models with `images=`:

```python
from PIL import Image
from smolagents import CodeAgent, LiteLLMModel

vision_agent = CodeAgent(tools=[], model=LiteLLMModel(model_id="%%model=openai-small%%"))
vision_agent.run("What does this chart show?", images=[Image.open("chart.png")])
```

Tools can return images (`output_type = "image"`) or audio (`"audio"`). Results are wrapped in `AgentImage` (a PIL image subclass) or `AgentAudio`; `.to_raw()` returns the PIL image or audio tensor and `.to_string()` a path to a saved file. In a `ToolCallingAgent` they are stored in `agent.state` (as `image.png` or `audio.mp3`) rather than pasted into the prompt. For audio input, `SpeechToTextTool` transcribes files.

### A UI in one line

```python
from smolagents import GradioUI

GradioUI(agent, file_upload_folder="./uploads").launch(share=False)   # needs smolagents[gradio]
```

Pass `share=False`. `GradioUI.launch()` defaults to `share=True`, which asks Gradio for a public `gradio.live` URL, and anyone with that link can make your agent run code and use your tools and API keys. Other keyword arguments (`server_name`, `server_port`, `auth`) go to Gradio's own `launch`. The UI keeps the agent's memory across messages unless you construct it with `reset_agent_memory=True`.

---

## Version notes

Historical changes that affect code you may find elsewhere:

| Version | Change |
|---------|--------|
| 1.14.0 | `HfApiModel` renamed to `InferenceClientModel` (old name deprecated) |
| 1.19.0 | `HfApiModel` alias removed; importing it fails |
| 1.23.0 | `InferenceClientModel` default model changed to `Qwen/Qwen3-Next-80B-A3B-Thinking` |
| 1.24.0 | Saved agents using `HfApiModel` load as `InferenceClientModel`; `FinalAnswerStep` available to `step_callbacks`; more robust execution timeout in the local executor |
| 1.25.0 | Docs and code stress that `LocalPythonExecutor` is not a security boundary; `allow_pickle` defaults to `False`; `huggingface-hub` 1.x supported |
| 1.26.0 | Exa engine in `WebSearchTool`; `WasmExecutor` removed |

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2026-10-28 | 1.26.0 | Added `requests_per_minute`, `retry` and the retry rules of the API model classes, and the exact behaviour of `custom_role_conversions` and `flatten_messages_as_text`. |
| 2026-10-24 | 1.26.0 | Audited against the installed package. Corrected: the `InferenceClientModel` endpoint example (`base_url` alone raises), step callbacks given as a list (they run for `ActionStep` only), `GradioUI.launch()` (defaults to a public share link), Docker image rebuilds, and step counting from `RunResult.steps`. Added: memory and steps, planning, MCP details, local-executor limits, saving agents and tools, delegation behaviour and a `ToolCallingAgent` manager example. |
| 2026-10-21 | 1.26.0 | Corrected where tools run under remote executors, the `LogLevel` values and the quantisation example; current, served model IDs; added the CLI, `from_mcp`, `visualize()`, `AgentImage`/`AgentAudio` and `PromptTemplates`. |
| 2026-10-21 | 1.26.0 | Rewritten against the installed package. Removed fabricated APIs (`push_to_hub()` function, `load_agent`, `run(image=/audio=/video=)`, `VideoTranscriptionTool`, `TranscriptionTool`, `memory_size`, `timeout`, `AgentRunResult.success/errors/execution_time`, `Model.generate_text`), the `"wasm"` executor, invented efficiency figures and placeholder text. Corrected `HfApiModel` history (removed in 1.19.0, not 1.24.0) and the default-model change (1.23.0). Added `RunResult`, streaming events, final-answer checks, step callbacks, MCP, executors and version notes. |
| November 2025 | 1.23.0 | First documented version. |

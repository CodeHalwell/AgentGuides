---
title: "smolagents Architecture & Flow Diagrams"
description: "Text diagrams of how smolagents works: the agent loop, a CodeAgent step versus a ToolCallingAgent step, managed agents, and where code and tools run under each executor."
framework: "smolagents"
language: "python"
sidebar:
  label: "Diagrams"
  order: 4
---

> Reflects `smolagents` %%version=smolagents%% (checked %%versions-checked%%). Each diagram follows the library's source (`agents.py`, `remote_executors.py`). For the components themselves, see [Core concepts](/smolagents-guide/smolagents_comprehensive_guide/#core-concepts--architecture).

## The agent loop

Both agent classes share this loop, implemented in `MultiStepAgent`. Only the "act" step differs between them.

```text
agent.run(task)
  │
  ├─ reset=True (default): clear memory.  reset=False: keep earlier steps.
  ├─ append TaskStep(task)
  │
  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ while no final answer and step <= max_steps:                         │
│                                                                      │
│   interrupt() was called? ─► raise AgentError("Agent interrupted.")  │
│                                                                      │
│   planning_interval set, and step is 1 or due?                       │
│     └─► model writes or updates a plan ─► PlanningStep               │
│                                                                      │
│   model.generate(memory rendered as chat messages)                   │
│   act (CodeAgent: run the code; ToolCallingAgent: call the tools)    │
│                                                                      │
│   error while parsing or acting?                                     │
│     └─► stored in ActionStep.error, shown to the model; loop goes on │
│   (a model/provider failure, AgentGenerationError, is raised)        │
│                                                                      │
│   final_answer(...) called?                                          │
│     └─► run final_answer_checks                                      │
│           pass ─► leave the loop                                     │
│           fail ─► recorded as the step's error; loop goes on         │
│                                                                      │
│   append ActionStep, run step_callbacks                              │
└──────────────────────────────────────────────────────────────────────┘
  │
  ├─ out of steps? model writes a best-effort answer from memory
  │                (RunResult.state == "max_steps_error")
  ▼
FinalAnswerStep(output) ─► run() returns the output (or a RunResult)
```

Step callbacks also run after each `PlanningStep` and after the `FinalAnswerStep`, which is what makes [plan review](/smolagents-guide/smolagents_recipes/#recipe-21-human-in-the-loop-plan-review) and [token budgets](/smolagents-guide/smolagents_production_guide/#cost-optimisation-strategies) possible.

## One step: CodeAgent and ToolCallingAgent

The task "Is three nights cheaper in Paris or London?", with a `get_hotel_price(city, nights)` tool.

```text
CodeAgent                                  ToolCallingAgent
─────────                                  ────────────────
Model output (text):                       Model output (native tool calls):
  Thought: compare both cities.              get_hotel_price({"city": "Paris",  "nights": 3})
  <code>                                     get_hotel_price({"city": "London", "nights": 3})
  p = get_hotel_price(city="Paris", nights=3)
  l = get_hotel_price(city="London", nights=3)
  print(p, l)
  final_answer("Paris" if p < l else "London")
  </code>

Executed by: the Python executor           Executed by: the agent, one call per tool
  (variables persist between steps)          (in parallel up to max_tool_threads)

Observation: print() output, plus          Observation: each tool's return value
  the value of the last expression

Ends the run: the final_answer(...)        Ends the run: a later step in which the
  call inside the code                       model calls the final_answer tool
```

The CodeAgent can compare and answer in the same step because the comparison happens in code. The ToolCallingAgent needs another model call to read the two results and call `final_answer`. How many steps or tokens this saves depends on the task and the model: measure it with `result.token_usage` and the number of action steps, as described in [When CodeAgent saves steps](/smolagents-guide/smolagents_comprehensive_guide/#when-codeagent-saves-steps).

## A tool, end to end

```text
@tool                                       class SearchTool(Tool):
def get_hotel_price(city: str,                  name = "search"
                    nights: int = 1) -> float:  description = "..."
    """Cheapest hotel price for a stay.         inputs = {"query": {"type": "string",
                                                                    "description": "..."}}
    Args:                                       output_type = "string"
        city: City name.
        nights: Number of nights.               def forward(self, query: str) -> str:
    """                                             ...
    ...
          │                                               │
          └───────────────────────┬───────────────────────┘
                                  ▼
          name · description · inputs schema · output_type
                                  │
          ┌───────────────────────┴───────────────────────┐
          ▼                                               ▼
  CodeAgent prompt: a Python                   ToolCallingAgent: a JSON schema
  function signature + docstring               passed to the model's tool API
  (tool.to_code_prompt())                      (tool.to_tool_calling_prompt())
```

The decorator reads the type hints and the `Args:` section; a parameter without an `Args:` entry raises `DocstringParsingException`. Allowed `output_type` values are `string`, `boolean`, `integer`, `number`, `image`, `audio`, `array`, `object`, `any` and `null`.

## Managed agents

```text
manager = CodeAgent(tools=[...], managed_agents=[web_agent], model=...)

manager step, model-written code:
  findings = web_researcher(task="Population of Lyon, with a source")
                 │
                 ▼
  web_agent.__call__(task)
    ├─ wraps the task in the "managed_agent" prompt template
    ├─ web_agent.run(...)     its own loop, memory and max_steps
    └─ returns a report string: its final answer, plus a summary of
       its steps if provide_run_summary=True
                 │
                 ▼
  findings is now a string in the manager's code; the manager keeps going
```

A `ToolCallingAgent` manager calls the managed agent through a tool call with a `task` argument instead. The managed agent needs a `name` that is a valid Python identifier and a `description`. Managed agents cannot be combined with a remote executor on the manager.

## Where code and tools run

```text
executor_type="local" (default)
┌─ your process ─────────────────────────────────────────────┐
│ agent loop · model calls · memory                          │
│ LocalPythonExecutor: interprets the model's code           │
│   and calls your tools, all in this process                │
│   (import allow-list and AST checks: not a security        │
│    boundary)                                               │
└────────────────────────────────────────────────────────────┘

executor_type="docker" | "e2b" | "modal" | "blaxel"
┌─ your process ──────────────┐        ┌─ sandbox ──────────────────────────┐
│ agent loop · model calls ·  │  code  │ Python kernel                      │
│ memory · answer checks      │ ─────► │ runs the model's code              │
│                             │        │                                    │
│ at start-up:                │ tools  │ each tool's requirements are       │
│   tool source code          │ ─────► │ pip-installed and the tool is      │
│                             │        │ rebuilt from source; tools run     │
│                             │        │ here, not in your process          │
│                             │ ◄───── │ print() output, final_answer value │
└─────────────────────────────┘        └────────────────────────────────────┘
```

Consequences:

- Tools must be self-contained: imports inside `forward`, and any credentials they need present in the sandbox, not only in your process.
- `additional_args` are serialised into the sandbox; since 1.25.0 only JSON-safe values are sent unless you pass `allow_pickle=True` in `executor_kwargs`.
- `DockerExecutor` starts the container with a port mapping and nothing else. The container has normal network access (it installs the authorised imports with `pip`) and no memory or CPU limits. Set limits yourself through `executor_kwargs={"container_run_kwargs": {...}}`; see the [production guide](/smolagents-guide/smolagents_production_guide/#code-execution-sandboxing).
- The isolation properties of E2B, Modal and Blaxel are those of the provider's sandbox; check each provider's documentation for network and resource controls.

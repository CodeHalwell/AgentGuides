---
title: "CLI and project layout"
description: "Every crewai CLI command verified against source — create, run, train, test, replay, eval, checkpoint, flow, tool, template, skill, memory TUI, deploy, and traces."
framework: "crewai"
language: "python"
sidebar:
  label: "CLI"
  order: 26
---

> **Verified against `crewai` %%version=crewai%%** (source: `crewai_cli/cli.py`, `crewai_cli/create_json_crew.py`, `crewai_cli/templates/crew/*`; checked %%versions-checked%%). Since the 1.15 line the CLI lives in its own `crewai-cli` distribution, installed automatically with `crewai`; `crewai.cli` still imports but emits a `DeprecationWarning` pointing at `crewai_cli`.

The `crewai` CLI is a Click group. This page is a command-by-command tour grounded in the installed `cli.py`, so every flag matches reality — not a blog post about an older release.

## Minimal runnable example

```bash
crewai create crew my_research_crew --classic   # Python/YAML layout used throughout this guide
cd my_research_crew
uv sync                 # or: pip install -e .
crewai run
```

`create` scaffolds a project; `run` invokes the `run_crew` entry point wired up in `pyproject.toml`.

> **JSON-first is now the default.** Without `--classic`, `crewai create crew` starts an interactive wizard (agents, tasks, provider, platform tools) and writes a **JSON-first** project: `crew.jsonc`, one `agents/<name>.jsonc` per agent, plus `tools/`, `skills/` and `knowledge/`. `crewai run` runs either layout. The Python/YAML `@CrewBase` layout below is what you get with `--classic`, and it is what the rest of this guide's examples assume.

## Installing the CLI

```bash
pip install "crewai[tools]"
# or
uv tool install crewai
```

`crewai --version` or `crewai version` prints the version (add `--tools` for crewai-tools).

## Command cheat sheet

| Command | What it does |
|---|---|
| `crewai create crew <name>` | Scaffold a crew project (interactive JSON wizard; `--classic` for Python/YAML). |
| `crewai create flow <name>` | Scaffold a flow project (`--declarative` for a definition-file flow). |
| `crewai create tool/skill/template <name>` | Scaffold a tool, an agent skill, or install a template (`-o <dir>`). |
| `crewai run` | Run the project — crew or flow, whichever `pyproject.toml` declares. |
| `crewai install` | Sync dependencies (wraps uv). |
| `crewai update` | Update the project's `pyproject.toml` to use uv. |
| `crewai train -n 5 -f trained.pkl` | Kick off `main:train`: run the crew with human feedback on every task, then distil the feedback (see [Training mode](#training-mode)). |
| `crewai test -n 3 -m %%model=openai-small%%` | Kick off `main:test` (eval; OpenAI models only). |
| `crewai eval` | Grade the last traced run through CrewAI AMP; `--models a,b` compares models on a deployment. |
| `crewai replay -t <task_id>` | Re-run starting from a specific task. |
| `crewai log-tasks-outputs` | List logged task outputs. |
| `crewai reset-memories -m -kn -a` | Reset memory / knowledge / kickoff outputs. |
| `crewai memory` | Interactive Memory TUI. |
| `crewai chat` | Chat with the crew (uses `Crew.chat_llm`; see [Gotchas](#gotchas)). |
| `crewai checkpoint` | TUI browser for saved checkpoints. |
| `crewai checkpoint list/info/resume/diff/prune` | Scripted checkpoint ops. |
| `crewai flow kickoff` | Run the flow (equivalent to `crewai run` in a flow project). |
| `crewai flow plot` | Render the flow graph to an HTML file and open it in the browser. |
| `crewai flow add-crew <name>` | Embed a crew in the flow. |
| `crewai triggers list/run <app>/<trigger>` | Platform triggers. |
| `crewai tool install/publish <handle>` | Tool Repository (platform). `tool create` is deprecated in favour of `crewai create tool`. |
| `crewai skill list/install/publish` | Agent skills. `skill create` is deprecated in favour of `crewai create skill`. |
| `crewai template list` | Browse project templates (`template add` is deprecated in favour of `crewai create template`). |
| `crewai deploy create/list/push/status/logs/remove` | CrewAI AMP deployments. |
| `crewai login / logout` | Auth against CrewAI AMP. |
| `crewai org list/switch/current` | Organization context. |
| `crewai enterprise configure <url>` | AMP OAuth2 setup. |
| `crewai config list/set/reset` | Local CLI settings. |
| `crewai env view` | Show tracing env vars. |
| `crewai traces enable/disable/status` | Tracing opt-in. |
| `crewai uv <args>` | Wrapper over `uv` that injects tool credentials. |

## Project layout — `crewai create crew --classic`

```text
my_crew/
├── pyproject.toml
├── README.md
├── AGENTS.md                   # also CLAUDE.md / CURSOR.md / GEMINI.md for coding assistants
├── knowledge/                  # file sources given as strings are read from here (starts with user_preference.txt)
└── src/my_crew/
    ├── __init__.py
    ├── main.py                 # entry points: run / train / replay / test
    ├── crew.py                 # @CrewBase with @agent / @task / @crew decorators
    ├── tools/                  # custom_tool.py: a BaseTool example
    └── config/
        ├── agents.yaml         # agent roles, goals, backstories
        └── tasks.yaml          # task descriptions, expected outputs
```

### `agents.yaml`

```yaml
researcher:
  role: >
    {topic} Senior Data Researcher
  goal: >
    Uncover cutting-edge developments in {topic}
  backstory: >
    You're a seasoned researcher with a knack for finding relevant sources.
```

Each top-level key is a name referenced from `crew.py` (`self.agents_config["researcher"]`). The scaffold also defines a second agent, `reporting_analyst`, in the same file. The role, goal and backstory text may contain `{topic}`-style placeholders filled from `kickoff(inputs=...)`.

### `tasks.yaml`

```yaml
research_task:
  description: >
    Conduct thorough research about {topic} (current year: {current_year}).
  expected_output: >
    Ten bullet points of the most relevant findings.
  agent: researcher
```

### `crew.py`

```python
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class MyCrew:
    """Research + reporting crew."""

    @agent
    def researcher(self) -> Agent:
        return Agent(config=self.agents_config["researcher"], verbose=True)

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(config=self.agents_config["reporting_analyst"], verbose=True)

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])

    @task
    def reporting_task(self) -> Task:
        return Task(config=self.tasks_config["reporting_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, process=Process.sequential, verbose=True)
```

### `main.py`

```python
import sys

from my_crew.crew import MyCrew

INPUTS = {"topic": "AI agents"}

def run():
    MyCrew().crew().kickoff(inputs=INPUTS)

def train():
    MyCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=INPUTS)

def replay():
    MyCrew().crew().replay(task_id=sys.argv[1])

def test():
    MyCrew().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=INPUTS)
```

The generated `main.py` also has a `run_with_trigger()` function. `pyproject.toml` wires these to console scripts (`run_crew`, `train`, `replay`, `test`, `run_with_trigger`, plus one named after the project), so `crewai run`, `crewai train`, etc. work out of the box.

## Training mode

```bash
crewai train -n 5 -f trained_agents_data.pkl
```

Training is **human-in-the-loop**, not an automatic scorer. `Crew.train(n_iterations, filename, inputs)` (the second argument is required; the scaffolded `train()` reads `-n` and `-f` from the command line):

1. Copies the crew and turns on `human_input=True` for every task and `allow_delegation=False` for every agent.
2. Runs the copy `n` times with your `inputs`. After each task the console asks you for feedback on the output. The agent revises its answer, and the initial output, your feedback and the improved output are stored per agent in an intermediate `training_data.pkl`.
3. When the iterations finish, an LLM evaluator (`TaskEvaluator.evaluate_training_data`) condenses each agent's feedback into a list of suggestions, a 0 to 10 quality score and a summary, and saves them under the agent's `role` in the file you named.
4. If anything fails, both pickle files are cleared.

To use the result, point the crew at the file. At run time each agent appends `You MUST follow these instructions:` and its suggestions to its task prompt. The file is chosen in this order: `Crew(trained_agents_file="path.pkl")`, the `CREWAI_TRAINED_AGENTS_FILE` environment variable, then `trained_agents_data.pkl` in the working directory. `crewai run -f path.pkl`, `crewai test -f` and `crewai replay -f` set the environment variable for you. Suggestions are keyed by agent `role`, so renaming a role orphans its training data.

## Testing mode

```bash
crewai test -n 3 -m %%model=openai-small%%
```

Calls `Crew.test(n_iterations=3, eval_llm=...)` with the model you pass — runs the crew `n` times, scores every task with the eval LLM, and prints a consolidated table. Good for regression runs in CI.

## Replay

```bash
crewai log-tasks-outputs        # list the task ids from the last kickoff
crewai replay -t <task_id>
```

`replay` re-runs the crew **starting from the named task** and reuses the stored outputs of earlier tasks as context. The outputs come from the latest kickoff, saved by every `Crew.kickoff()` in `latest_kickoff_task_outputs.db` inside CrewAI's per-user data directory (`$CREWAI_STORAGE_DIR` if set). We replayed a two-task crew from its second task: one model call, with the first task's stored output in the prompt. The id is looked up in the stored outputs, so a freshly built crew with the same tasks can replay an id printed by an earlier process. Only the most recent kickoff is kept. `crewai reset-memories -k` clears it.

## Reset memories

```bash
crewai reset-memories -m        # unified Memory
crewai reset-memories -kn       # crew knowledge
crewai reset-memories -akn      # agent-level knowledge
crewai reset-memories -k        # cached kickoff outputs
crewai reset-memories -a        # all of the above
```

The legacy `--long`, `--short`, `--entities` flags are accepted (they do not appear in `--help`) and print a deprecation warning that says "all memory is now unified." Each flag resets only what the crew in the current project has configured: `-m` on a crew without memory enabled fails with "Memory memory system is not initialized".

## Checkpoints

```bash
crewai checkpoint                               # interactive TUI
crewai checkpoint list ./.checkpoints           # non-interactive list
crewai checkpoint info ./.checkpoints           # show latest (or pass a file)
crewai checkpoint resume [checkpoint_id]        # resume run
crewai checkpoint diff id1 id2                  # side-by-side compare
crewai checkpoint prune --keep 20 --older-than 30d --dry-run
```

`--location` on the group sets the default directory (`./.checkpoints`); SQLite files are auto-detected by `.db` extension.

## Flows

```bash
crewai create flow my_flow
crewai run                         # run the flow (`crewai flow kickoff` still works)
crewai flow plot                   # builds the HTML graph and opens it in your browser
crewai flow add-crew research      # embed a crew scaffold inside the flow project
```

## Deployments — CrewAI AMP

```bash
crewai login
crewai deploy create               # first-time
crewai deploy push                 # redeploy
crewai deploy list
crewai deploy status -u <uuid>
crewai deploy logs -u <uuid>
crewai deploy remove -u <uuid>
```

`crewai deploy validate` runs the same pre-flight checks as `create`/`push` without contacting the platform — handy in CI. In 1.15.28 it looks for `src/<package>/main.py` with a `Flow` subclass in flow projects, so a **declarative** flow project (`crewai create flow --declarative`, which has `flow.yaml` and no `main.py`) fails with `missing_flow_main` and `no_flow_subclass`. `crewai run` works for those projects; validation and deploy checks do not accept them yet.

## Tool Repository

```bash
crewai create tool my_tool
crewai tool install acme/my_tool
crewai tool publish --public
```

Requires `crewai login`. Tools are Python packages auto-wrapped as `BaseTool`s.

## Tracing

```bash
crewai traces enable
crewai traces status
crewai env view                    # view CREWAI_TRACING_ENABLED, etc.
```

Set `CREWAI_TRACING_ENABLED=true` in `.env` for an always-on setup.

## Patterns

### 1. Scripted CI check

```yaml
# .github/workflows/crew.yml
- run: pip install "crewai[tools]"
- run: crewai deploy validate
- run: crewai test -n 2 -m %%model=openai-small%%
```

`deploy validate` exits non-zero on blocking issues. `crewai test` calls the real model (it needs the API key as a CI secret and costs tokens) and prints a score table. It does not fail the job on a low score, and in 1.15.28 it also exits 0 when the run itself errors (the CLI catches the error and prints it), so keep offline unit tests with a stub LLM as the gate (see [Production](/crewai-guide/crewai_production_guide/#testing-without-a-model)).

### 2. Scheduled resume from checkpoint

```bash
# cron: every hour, pick up the most recent interrupted run
crewai checkpoint resume
```

`resume` without an ID defaults to the most recent checkpoint.

### 3. Dry-run prune

```bash
crewai checkpoint prune --keep 50 --older-than 14d --dry-run
```

Shows what would be removed; drop `--dry-run` to commit.

### 4. Provision + run a template

```bash
crewai template list
crewai create template marketing-campaign -o mkt-campaign
cd mkt-campaign && uv sync && crewai run
```

### 5. Ad-hoc chat against a deployed crew

```bash
crewai chat
```

Set `Crew(chat_llm=LLM(...))` for a predictable model; the CLI opens a REPL that routes inputs through the crew on demand.

## Gotchas

- **`crewai run` relies on `[project.scripts]`.** If you rename entry points in `pyproject.toml`, the CLI still expects the `run_crew` script. Edit both sides together.
- **Scaffolds pin `crewai[tools]>=<installed version>,<2.0.0`.** Projects created by an older CLI keep their older floor; bump it when you upgrade so the project and the CLI agree.
- **`reset-memories` without flags prints a help line and exits.** You must pick at least one type.
- **`replay -t` needs logged outputs.** `log-tasks-outputs` confirms what's available.
- **Legacy `--long`/`--short`/`--entities` are shims.** They all map to `--memory` now; don't rely on the distinction.
- **`crewai chat` model resolution.** It uses `Crew.chat_llm`; if that is unset it falls back to the `MODEL` environment variable, then CrewAI's default model (`gpt-4.1-mini`), not to the agents' own LLMs. If the model cannot be created it prints "Unable to find a Chat LLM" and exits.
- **`crewai uv` expects a `pyproject.toml`.** Outside a crew project it errors out with a clear message.

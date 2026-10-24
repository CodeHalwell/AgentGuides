---
title: "Quick Start"
description: "Five short, runnable first agents: one tool-using agent, a two-agent crew, a handoff, retrieval over your own files and a small graph."
---

Each path below is a complete script for one framework. They use the current release (checked %%versions-checked%%), and each was run against that release with a scripted stand-in for the model, so the code runs as written once a real model is attached. Pick the path closest to what you want to build; none depends on another.

Most paths call a hosted model, so you need an API key in your environment first (the key each path needs is noted in it). The one exception is [path 5](#path-5-a-workflow-with-state-langgraph), which makes no model calls.

Use a virtual environment, and pin the version as shown so that what you install matches what the guide describes.

## Path 1: One agent with a tool (SmolAgents)

SmolAgents agents write and run Python rather than emitting JSON tool calls. This is the smallest setup in the collection: one package, one decorated function, one model.

```bash
pip install "smolagents==%%version=smolagents%%"
export HF_TOKEN=hf_your_token
```

```python
from smolagents import CodeAgent, InferenceClientModel, tool


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: The first number.
        b: The second number.
    """
    return a + b


model = InferenceClientModel(model_id="Qwen/Qwen3.8-27B")  # served through Hugging Face providers
agent = CodeAgent(tools=[add_numbers], model=model)
print(agent.run("What is 15 + 27?"))
```

The docstring matters: `@tool` reads the type hints and the `Args:` section to build the tool description the model sees, and raises an error if a parameter is undocumented. `CodeAgent` runs the code the model writes in a restricted local interpreter; read the [production guide](/smolagents-guide/smolagents_production_guide/) before pointing it at anything that matters.

Next: the [SmolAgents overview](/smolagents-guide/), the [recipes](/smolagents-guide/smolagents_recipes/) and the [comprehensive guide](/smolagents-guide/smolagents_comprehensive_guide/).

## Path 2: A small crew (CrewAI)

CrewAI models work as agents with roles and tasks that run in order. A task passes its output to a later one through `context`.

```bash
pip install "crewai==%%version=crewai%%"
export OPENAI_API_KEY=sk-your-key
```

```python
from crewai import Agent, Crew, Task

analyst = Agent(
    role="Data Analyst",
    goal="Find the main drivers of quarterly sales",
    backstory="You are an experienced analyst who explains numbers plainly.",
    llm="openai/%%model=openai%%",
)
writer = Agent(
    role="Content Writer",
    goal="Write clear, short reports",
    backstory="You are a professional business writer.",
    llm="openai/%%model=openai%%",
)

analysis_task = Task(
    description="List the factors that most affect quarterly sales for a subscription business.",
    expected_output="A bulleted list of five factors, one line each.",
    agent=analyst,
)
report_task = Task(
    description="Turn the analyst's findings into a one-paragraph summary for managers.",
    expected_output="A single paragraph of no more than 120 words.",
    agent=writer,
    context=[analysis_task],  # the writer receives the analyst's output
)

crew = Crew(agents=[analyst, writer], tasks=[analysis_task, report_task])
result = crew.kickoff()
print(result)
```

`expected_output` is required on every `Task`. Without an `llm`, an agent falls back to CrewAI's default model, which also needs `OPENAI_API_KEY`.

Next: the [CrewAI overview](/crewai-guide/), the [recipes](/crewai-guide/crewai_recipes/) and the [comprehensive guide](/crewai-guide/crewai_comprehensive_guide/).

## Path 3: Handing off between agents (OpenAI Agents SDK)

In the OpenAI Agents SDK an agent can list other agents under `handoffs`. The model sees each one as a tool (here `transfer_to_technical`) and calls it to pass the conversation on.

```bash
pip install "openai-agents==%%version=openai-agents%%"
export OPENAI_API_KEY=sk-your-key
```

```python
from agents import Agent, Runner

technical_agent = Agent(
    name="Technical",
    instructions="You are a technical specialist. Diagnose the problem and give concrete steps.",
)
support_agent = Agent(
    name="Support",
    instructions=(
        "You are a customer support agent. Answer simple questions yourself and "
        "hand technical problems to the Technical agent."
    ),
    handoffs=[technical_agent],
)

result = Runner.run_sync(support_agent, "I'm having a technical issue: the app crashes on start.")
print(result.final_output)
print("Answered by:", result.last_agent.name)
```

`Runner.run_sync` is for ordinary scripts. Inside async code (a web handler, a notebook cell) use `await Runner.run(...)` instead; calling `await` at the top level of a plain script is a syntax error.

Next: the [OpenAI Agents SDK overview](/openai-agents-sdk-guides/), the [recipes](/openai-agents-sdk-guides/openai_agents_sdk_recipes/) and the [comprehensive guide](/openai-agents-sdk-guides/openai_agents_sdk_comprehensive_guide/).

## Path 4: Questions over your own files (LlamaIndex)

LlamaIndex loads documents, embeds them into an index and answers questions from the most relevant passages.

```bash
pip install "llama-index==%%version=llama-index%%"
export OPENAI_API_KEY=sk-your-key
mkdir data
echo "Paris is the capital of France." > data/france.txt
```

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("What is the capital of France?")
print(response)
```

By default LlamaIndex uses OpenAI for both the embeddings and the answer, hence the key. The index lives in memory, so it is rebuilt (and re-embedded) each time the script runs; the [comprehensive guide](/llamaindex-guide/python/llamaindex_comprehensive_guide/) covers persisting it and swapping the models.

Next: the [LlamaIndex overview](/llamaindex-guide/), the [recipes](/llamaindex-guide/python/llamaindex_recipes/) and the [comprehensive guide](/llamaindex-guide/python/llamaindex_comprehensive_guide/).

## Path 5: A workflow with state (LangGraph)

LangGraph describes a program as a graph: nodes are functions that read the shared state and return updates, and edges say what runs next. This example makes no model calls, so it runs as is and shows the control-flow model on its own.

```bash
pip install "langgraph==%%version=langgraph%%"
```

```python
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    current_step: str


def step_1(state: State) -> dict:
    return {"current_step": "completed_step_1"}


def step_2(state: State) -> dict:
    return {"current_step": "completed_step_2"}


builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", END)

graph = builder.compile()
print(graph.invoke({"current_step": "start"}))
# {'current_step': 'completed_step_2'}
```

Nodes return only the keys they change; LangGraph merges them into the state. To call a model, add a node that does so, or use the prebuilt agents described in the guide.

Next: the [LangGraph (Python) overview](/langgraph-guide/python/), the [recipes](/langgraph-guide/python/langgraph_recipes/) and the [comprehensive guide](/langgraph-guide/python/langgraph_comprehensive_guide/).

## Which path should I take?

| You want | Start with | Guide |
|----------|-----------|-------|
| A single agent that calls your functions, with the least setup | SmolAgents | [SmolAgents](/smolagents-guide/) |
| Several agents with defined roles working through tasks | CrewAI | [CrewAI](/crewai-guide/) |
| Agents that pass a conversation to one another, with guardrails and tracing | OpenAI Agents SDK | [OpenAI Agents SDK](/openai-agents-sdk-guides/) |
| Answers grounded in your documents | LlamaIndex | [LlamaIndex](/llamaindex-guide/) |
| Explicit control over branching, loops, state and resuming | LangGraph | [LangGraph](/langgraph-guide/python/) |

Not on the list? The [framework comparison](/frameworks/) covers the rest, including Agno, Strands Agents, PydanticAI, Haystack, AG2 and the cloud vendors' own SDKs.

## Common tasks

The snippets below are written for the frameworks shown and were run against the same releases as the paths above.

### Remember a conversation between runs (LangGraph)

Compile the graph with a checkpointer and pass a `thread_id` on each call. State is saved per thread, so a second call with the same id continues from the first. A reducer such as `operator.add` makes a list key accumulate instead of being overwritten.

```python
import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    history: Annotated[list[str], operator.add]


def reply(state: State) -> dict:
    return {"history": ["seen"]}


builder = StateGraph(State)
builder.add_node("reply", reply)
builder.add_edge(START, "reply")
builder.add_edge("reply", END)
graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "user-1"}}
graph.invoke({"history": ["hello"]}, config)
print(graph.invoke({"history": ["again"]}, config))
# {'history': ['hello', 'seen', 'again', 'seen']}
```

`InMemorySaver` is lost when the process exits. For storage that survives restarts see [checkpointers](/langgraph-guide/python/reference-checkpointers/) and the [memory chapter](/langgraph-guide/python/chapter-05-memory/).

### Run steps in parallel and merge the results (LangGraph)

Two edges leaving the same node start two branches in the same step. An edge whose source is a list waits for every listed node before running the target.

```python
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    out: Annotated[list[str], operator.add]


def make(name: str):
    return lambda state: {"out": [name]}


builder = StateGraph(State)
for name in ("agent1", "agent2", "join_node"):
    builder.add_node(name, make(name))
builder.add_edge(START, "agent1")
builder.add_edge(START, "agent2")
builder.add_edge(["agent1", "agent2"], "join_node")
builder.add_edge("join_node", END)

print(builder.compile().invoke({"out": []}))
# {'out': ['agent1', 'agent2', 'join_node']}
```

### Route on a condition (LangGraph)

`add_conditional_edges` takes a function that reads the state and returns the name of the next node.

```python
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    needs_help: bool
    route: str


def classify(state: State) -> dict:
    return {}


def route(state: State) -> str:
    return "escalate" if state["needs_help"] else "resolve"


def escalate(state: State) -> dict:
    return {"route": "escalated"}


def resolve(state: State) -> dict:
    return {"route": "resolved"}


builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("escalate", escalate)
builder.add_node("resolve", resolve)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route, ["escalate", "resolve"])
builder.add_edge("escalate", END)
builder.add_edge("resolve", END)

print(builder.compile().invoke({"needs_help": True, "route": ""}))
# {'needs_help': True, 'route': 'escalated'}
```

Listing the possible targets as the third argument is optional, but it lets LangGraph draw the graph correctly.

## Common mistakes

These are the errors people most often hit with the code above, each reproduced against the pinned release.

- **CrewAI: `depends_on` is not a `Task` argument.** Passing it raises a validation error. Use `context=[earlier_task]`.
- **CrewAI: `expected_output` is required.** A `Task` without it fails validation.
- **OpenAI Agents SDK: `Runner.run` is a coroutine.** Use `Runner.run_sync` in a script, or `await Runner.run` inside an `async` function.
- **LangGraph: a checkpointer needs a `thread_id`.** Invoking a graph compiled with `checkpointer=...` and no `config={"configurable": {"thread_id": ...}}` raises `ValueError: Checkpointer requires one or more of the following 'configurable' keys`.
- **LangGraph: parallel branches writing the same key need a reducer.** Without one (such as `Annotated[list, operator.add]`) LangGraph raises `InvalidUpdateError`.
- **A tool the agent never uses** usually has a vague name or docstring. The model chooses tools from their descriptions, so write the description as you would for a colleague.

## Where next

1. Work through the recipes for your framework to see realistic variations.
2. Read the comprehensive guide for the full feature set.
3. Before deploying, read the production guide: limits, timeouts, secrets, observability and testing.
4. Browse [all guides](/guides/), or compare options on the [framework comparison](/frameworks/) page.

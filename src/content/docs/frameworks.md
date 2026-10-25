---
title: "Framework Comparison"
description: "How the frameworks covered in these guides differ: languages, multi-agent models, state and human approval, integrations, and which one to pick for which job."
---

This page compares the frameworks the site covers, so you can choose one before you read its guide. Everything about a package below was checked against the release listed in the first table; where a claim is about a design choice rather than a feature, it is worded as one.

It does not rank frameworks by popularity. Star counts and download numbers change daily, say little about whether a library fits your problem, and cannot be checked from a page that is only rebuilt when the guides are. The two maintenance facts that do matter for choosing are in the table notes: Microsoft AutoGen is archived, and LlamaIndex.TS is deprecated upstream.

## The frameworks at a glance

| Framework | Languages | Best for | Current version |
|-----------|-----------|----------|-----------------|
| [SmolAgents](/smolagents-guide/) | Python | Small agents that write code, quick prototypes | %%version=smolagents%% |
| [Agno](/agno-guide/) | Python | Agents, teams and workflows served over HTTP with AgentOS | %%version=agno%% |
| [PydanticAI](/pydanticai-guide/) | Python | Validated, typed outputs and dependency injection | %%version=pydantic-ai%% |
| [OpenAI Agents SDK](/openai-agents-sdk-guides/) | Python, TypeScript | Handoffs, guardrails, sessions and tracing | Py %%version=openai-agents%% · TS %%version=@openai/agents%% |
| [CrewAI](/crewai-guide/) | Python | Role-based crews and event-driven Flows | %%version=crewai%% |
| [AG2](/ag2-guide/) | Python | Async agents, typed events, a multi-agent Network, MCP and A2A | %%version=ag2%% |
| [Microsoft AutoGen (archived)](/autogen-guide/) | Python, .NET | Existing AutoGen 0.4 to 0.7 code only | %%version=autogen-agentchat%% |
| [LangGraph](/langgraph-guide/python/) | Python, TypeScript | Stateful graphs with loops, checkpoints and interrupts | Py %%version=langgraph%% · TS %%version=@langchain/langgraph%% |
| [LlamaIndex](/llamaindex-guide/) | Python, TypeScript (deprecated) | Retrieval over your own data, event-driven workflows | Py %%version=llama-index%% · TS %%version=llamaindex%% |
| [Haystack](/haystack-guide/) | Python | Pipelines for retrieval and search, with agents | %%version=haystack-ai%% |
| [Strands Agents](/strands-agents-guide/) | Python, TypeScript | A model-driven agent loop, from AWS, on any provider | Py %%version=strands-agents%% · TS %%version=@strands-agents/sdk%% |
| [Amazon Bedrock Agents and AgentCore](/amazon-bedrock-agents-guide/) | Python, TypeScript | Managed agents and hosting on AWS | AgentCore %%version=bedrock-agentcore%% |
| [Microsoft Agent Framework](/microsoft-agent-framework-guide/) | Python, .NET, Go | Azure and Microsoft stacks; the successor to Semantic Kernel agents and AutoGen | Py %%version=agent-framework%% · .NET %%version=Microsoft.Agents.AI%% · Go %%version=github.com/microsoft/agent-framework-go%% |
| [Semantic Kernel](/semantic-kernel-guide/) | Python, .NET | Existing Semantic Kernel applications, plugin-based design | Py %%version=semantic-kernel%% · .NET %%version=Microsoft.SemanticKernel%% |
| [Google ADK](/google-adk-guide/) | Python, TypeScript, Go | Gemini and Google Cloud | Py %%version=google-adk%% · TS %%version=@google/adk%% · Go %%version=google.golang.org/adk%% |
| [Claude Agent SDK](/anthropic-claude-agent-sdk-guide/) | Python, TypeScript | Agents built on Claude Code's agent loop | Py %%version=claude-agent-sdk%% · TS %%version=@anthropic-ai/claude-agent-sdk%% |
| [Mistral Agents API](/mistral-agents-api-guide/) | Python, TypeScript | Hosted agents on Mistral's platform | Py %%version=mistralai%% · TS %%version=@mistralai/mistralai%% |

Versions checked against PyPI, npm, NuGet and the Go module proxy on %%versions-checked%%. The source of truth is [`src/data/versions.json`](https://github.com/CodeHalwell/AgentGuides/blob/main/src/data/versions.json).

Two notes on the table:

- **AutoGen and AG2 are different projects.** See [AG2 and Microsoft AutoGen](#ag2-and-microsoft-autogen) below.
- **LlamaIndex.TS is deprecated upstream.** The npm packages `@llamaindex/workflow-core` and its siblings carry a deprecation notice, and the final releases still install. The Python library is actively released. For new TypeScript projects the [LlamaIndex TypeScript guide](/llamaindex-guide/typescript/) recommends another framework.

## What each framework is for

### SmolAgents

SmolAgents (from Hugging Face) is small enough to read in an afternoon. Its default `CodeAgent` has the model write Python that calls your tools, rather than emitting JSON tool calls; a `ToolCallingAgent` is available when you want conventional tool calls. Multi-agent setups use `managed_agents`: an agent you list there is exposed to the parent as a callable tool.

- **Choose it for:** learning how agents work, quick prototypes, tasks that benefit from the model composing several tool calls in one piece of code.
- **Weigh against:** it has no checkpoint store, graph engine or approval flow, so long-running or audited workloads need that built around it. Because the model writes code, run it in a sandboxed executor for anything untrusted (see the [production guide](/smolagents-guide/smolagents_production_guide/)).

### Agno

Agno has agents, teams of agents and step-based workflows that share one run interface, and AgentOS, a FastAPI runtime that serves them. Sessions, memory and knowledge are backed by a database you choose, and there are built-in guardrails and an approval decorator for tools.

- **Choose it for:** a Python service that needs agents with persistent sessions and memory, a knowledge base and an HTTP API, without assembling those parts from separate libraries.
- **Weigh against:** Python only, and a larger surface than the minimal libraries; MCP support and OpenTelemetry export are optional extras.

### PydanticAI

PydanticAI is built around typed outputs: you give an agent an `output_type`, and the result is validated against it, with the model asked to retry on validation failure. Tools receive typed dependencies, and graphs (`pydantic_graph`) are available for explicit control flow. Durable execution wrappers exist for Temporal, DBOS and Prefect.

- **Choose it for:** services where the shape of the model's output matters, and teams that already use Pydantic and type checking.
- **Weigh against:** multi-agent work is done by calling one agent from another's tool; there is no handoff primitive.

### OpenAI Agents SDK

A small set of primitives: `Agent`, `Runner`, handoffs, guardrails, sessions and tracing. Agents hand a conversation to one another through `handoffs`, or are called as tools with `Agent.as_tool()`. Tools can require approval, and an interrupted run can be serialised and resumed. A TypeScript SDK mirrors the primitives.

- **Choose it for:** customer-service-style routing between specialists, and for teams who want guardrails and tracing without extra services.
- **Weigh against:** it is designed around OpenAI's API; other providers are supported through adapters (LiteLLM and any-llm extensions) with the usual caveat that features differ by provider. There is no graph engine.

### CrewAI

CrewAI offers two models. A **Crew** is a set of role-defined agents working through tasks, sequentially or under a manager (`Process.sequential`, `Process.hierarchical`). **Flows** are event-driven workflows built from `@start`, `@listen` and `@router` methods, which can contain crews. Tasks can validate output with guardrails, return Pydantic models, ask for human input, and crews can be checkpointed and resumed.

- **Choose it for:** pipelines that read naturally as a team (researcher, writer, reviewer) and need a declarative setup.
- **Weigh against:** the role-and-backstory abstraction is convenient but adds prompt text you do not fully control; for fine-grained control flow use Flows or a graph library.

### AG2 and Microsoft AutoGen

These are two separate projects with a shared ancestor.

- **AG2** is a community fork of AutoGen 0.2, started in November 2024 by AutoGen's original authors. **AG2 1.0** (PyPI release 27 July 2026) replaced the AutoGen-era API: `pip install ag2` now installs an async framework imported as `ag2`, with `Agent`, `@tool`, typed events and a multi-agent `Network`. The old `import autogen` API (`ConversableAgent`, `GroupChat`) continues as **AG2 Classic**, published on PyPI as `autogen` and in maintenance mode.
- **Microsoft AutoGen** is the 0.4 rewrite (`autogen-agentchat`, `autogen-core`, `autogen-ext`; imports `autogen_agentchat` and so on) that Microsoft develops separately. Its last release is 0.7.5 (30 September 2025), Microsoft moved it to maintenance mode when it launched Microsoft Agent Framework, and the repository is archived. The `pyautogen` name on PyPI is a proxy for Microsoft's `autogen-agentchat`, not for AG2.

Practical consequences: older AutoGen tutorials may describe either project, so check the import line (`import autogen` is the AG2 Classic or 0.2 API; `autogen_agentchat` is Microsoft's). For new work on Microsoft's stack use [Microsoft Agent Framework](/microsoft-agent-framework-guide/); for AG2 use the [AG2 guide](/ag2-guide/), and see [AG2 Classic](/ag2-guide/classic/) for existing `import autogen` code. The [AutoGen guide](/autogen-guide/) covers Microsoft's archived project and has a timeline of how the two diverged.

- **Choose AG2 for:** multi-agent designs where routing between agents should be explicit and inspectable (a hub with channels and a transition graph), typed events you can test against, and MCP or A2A interoperability.
- **Weigh against:** 1.x is not a drop-in upgrade from Classic, and it is Python only.

### LangGraph

LangGraph models a program as a graph of nodes over shared state. Edges can be conditional, loops are allowed, `Send` fans work out in parallel, and `Command` lets a node update state and choose the next node. With a checkpointer, every step is saved per thread, which enables resuming, time travel and `interrupt()` for human approval. A TypeScript version has the same model.

- **Choose it for:** workflows with loops, branching, long-running state or human review in the middle.
- **Weigh against:** it is the most explicit option, so simple agents take more code. MCP client support comes from a separate package (`langchain-mcp-adapters`), and tracing is through LangSmith or your own callbacks.

### LlamaIndex

LlamaIndex's core strength is retrieval: loaders, node parsers, indexes, retrievers, query engines and a large set of vector store integrations. Agents are built on event-driven **Workflows**; `AgentWorkflow` runs several agents with `can_handoff_to` rules between them.

- **Choose it for:** applications whose main job is answering from your own documents.
- **Weigh against:** many integration packages with their own versions, and, in TypeScript, deprecation (see above).

### Haystack

Haystack builds applications from components wired into **pipelines**, which are graphs that may contain loops and can be paused at a breakpoint and resumed from a snapshot. Retrieval components, document stores and rankers are in the core package. An `Agent` component handles tool-calling loops, and `AgentTool` lets one agent be used as a tool by another.

- **Choose it for:** search and question-answering systems where the pipeline structure should be explicit and serialisable.
- **Weigh against:** many components (including OpenTelemetry and Datadog tracing, web search and Hugging Face models) live in separate integration packages, so check what you need is installed.

### Strands Agents

Strands, from AWS, runs a model-driven loop: you give the model a prompt and tools, and it decides the steps. It is not tied to AWS: model providers include Bedrock, Anthropic, OpenAI, Gemini, Mistral, Ollama, LiteLLM and others. Multi-agent patterns are agents-as-tools, a `Swarm` and a `Graph`; there are session managers (file and S3), interrupts for human input, structured output, MCP and OpenTelemetry in the core package, and A2A support. Python and TypeScript SDKs exist.

- **Choose it for:** a small, readable agent loop that fits AWS deployment (see the Bedrock and AgentCore guide) but is not locked to it.
- **Weigh against:** the default model provider is Bedrock, so you need AWS credentials unless you configure another provider; retrieval tools come from the separate `strands-agents-tools` package.

### Cloud and vendor SDKs

- **Amazon Bedrock Agents and AgentCore:** managed agent hosting, memory, gateway and identity on AWS. Pair it with Strands or another framework for the agent code.
- **Microsoft Agent Framework:** Microsoft's agent SDK for Python, .NET and Go, combining Semantic Kernel's enterprise foundations with AutoGen's multi-agent patterns, with graph workflows, checkpointing, A2A and OpenTelemetry. The core is stable in Python and .NET; Go is in preview.
- **Semantic Kernel:** Microsoft's plugin-oriented SDK for Python and .NET. New agent work on Microsoft's stack should look at Agent Framework first; Semantic Kernel remains relevant for existing applications.
- **Google ADK:** Gemini-first agents with Python, TypeScript and Go SDKs.
- **Claude Agent SDK:** the Claude Code agent loop as a library: built-in tools, subagents, hooks, permissions and file checkpointing. It runs agents that use a computer (files, shell, web) more than it orchestrates workflows.
- **Mistral Agents API:** agents and conversations stored on Mistral's platform, with built-in tools, MCP connectors and handoffs. State lives on the server.

## Feature comparison

The tables cover the ten general-purpose Python frameworks. Cells describe what is in the package at the version listed above, using the names you will find in the code. "Extra" means an optional install extra of the same package; "separate package" means a different distribution you must also install. A dash means the package does not provide it, not that it cannot be built.

All ten stream model output and tool results as the run progresses.

### Multi-agent and control flow

| Framework | Multi-agent model | Explicit workflow or graph |
|-----------|-------------------|----------------------------|
| SmolAgents | `managed_agents` (sub-agents called as tools) | None; `planning_interval` adds periodic plans |
| OpenAI Agents SDK | `handoffs`; `Agent.as_tool()` | None |
| CrewAI | `Crew` with sequential or hierarchical process | Flows (`@start`, `@listen`, `@router`) |
| LangGraph | Subgraphs; `Command` and `Send` routing | Core feature (state graph, loops, fan-out) |
| AG2 | `Network` hub with channels, `Handoff` and `Finish`; sub-agents as tools | Transition graphs inside the Network |
| LlamaIndex | `AgentWorkflow` with `can_handoff_to` | Event-driven `Workflow` |
| Haystack | `AgentTool` (an agent used as a tool) | `Pipeline` (graph of components, may loop) |
| PydanticAI | An agent called from another agent's tool | `pydantic_graph` |
| Agno | `Team` | `Workflow` (step-based) |
| Strands Agents | Agents as tools; `Swarm`; `Graph` | `Graph` |

### State, oversight and output

| Framework | Persisted state | Human approval | Typed output |
|-----------|-----------------|----------------|--------------|
| SmolAgents | In-run step memory; no store | None built in | None (`final_answer_checks` can validate) |
| OpenAI Agents SDK | Sessions (SQLite and others); serialisable `RunState` | `needs_approval` on tools | `output_type` |
| CrewAI | Crew checkpoints and Flow persistence | Task `human_input`; Flow human feedback | Task `output_pydantic`, `output_json` |
| LangGraph | Checkpointers (in-memory in the core; SQLite and Postgres are separate packages) | `interrupt()` | State schemas; `response_format` on the prebuilt agent |
| AG2 | `Storage` for history (in-memory built in) | `hitl_hook`, `context.input` | `response_schema` |
| LlamaIndex | Serialisable workflow `Context`; memory classes | `InputRequiredEvent` and `HumanResponseEvent` | `output_cls` on agents |
| Haystack | Pipeline breakpoints and snapshots | `haystack.hooks.human_in_the_loop` | `response_format` on OpenAI chat generators |
| PydanticAI | Message history; Temporal, DBOS and Prefect wrappers for durable runs | `requires_approval`, `DeferredToolRequests` | `output_type` (central to the API) |
| Agno | Database-backed sessions and memory (SQLite, Postgres, MongoDB and others) | `@approval`, tool `requires_confirmation` | `output_schema` |
| Strands Agents | Session managers (file, S3) | Interrupts | `structured_output_model` |

### Integrations

| Framework | MCP client | Tracing | Retrieval and vector stores |
|-----------|-----------|---------|-----------------------------|
| SmolAgents | `MCPClient` (extra `mcp`) | OpenTelemetry through an extra (`telemetry`) | None |
| OpenAI Agents SDK | `agents.mcp` | Built-in tracing (sent to OpenAI by default; can be disabled or redirected) | Hosted `FileSearchTool` only |
| CrewAI | `crewai.mcp` | Event bus; optional `tracing` | Knowledge sources |
| LangGraph | Separate package (`langchain-mcp-adapters`) | LangSmith or custom callbacks | None (provided by LangChain) |
| AG2 | `ag2.mcp` (extra `mcp`) | Telemetry middleware (extra `tracing`) | None |
| LlamaIndex | Separate package (`llama-index-tools-mcp`) | Instrumentation module; integration packages | Core feature |
| Haystack | Separate package (`mcp-haystack`) | Tracer interface (`haystack.tracing`) and logging tracer; OpenTelemetry in integration packages | Core feature |
| PydanticAI | Included in the default `pydantic-ai` install | OpenTelemetry and Logfire (`Agent.instrument_all`) | Embeddings API; no vector stores |
| Agno | Extra `mcp` | `agno.tracing`; OpenTelemetry extra | Knowledge and `vectordb` |
| Strands Agents | In core (`strands.tools.mcp`) | OpenTelemetry in core | Separate package (`strands-agents-tools`) |

A few of these cells reflect packaging rather than capability. LangGraph and Haystack both work with MCP and OpenTelemetry through other packages, and every framework that lacks built-in retrieval can use any vector store library from inside a tool.

## Choosing by use case

| You are building | Look at first | Also consider |
|------------------|---------------|---------------|
| A first agent, to learn how they work | [SmolAgents](/smolagents-guide/) | [PydanticAI](/pydanticai-guide/), [Agno](/agno-guide/) |
| Structured extraction or an API with validated output | [PydanticAI](/pydanticai-guide/) | [OpenAI Agents SDK](/openai-agents-sdk-guides/) (`output_type`) |
| Customer service with routing to specialists | [OpenAI Agents SDK](/openai-agents-sdk-guides/) | [CrewAI](/crewai-guide/), [LangGraph](/langgraph-guide/python/) |
| A team of role-based agents | [CrewAI](/crewai-guide/) | [Agno](/agno-guide/) teams, [AG2](/ag2-guide/) |
| Loops, branching and human review in a long-running process | [LangGraph](/langgraph-guide/python/) | [Microsoft Agent Framework](/microsoft-agent-framework-guide/) workflows, [CrewAI](/crewai-guide/) Flows |
| Question answering over documents | [LlamaIndex](/llamaindex-guide/python/) | [Haystack](/haystack-guide/) |
| Search or retrieval pipelines you want to serialise | [Haystack](/haystack-guide/) | [LlamaIndex](/llamaindex-guide/python/) |
| Agents served as an HTTP service with sessions and memory | [Agno](/agno-guide/) | [OpenAI Agents SDK](/openai-agents-sdk-guides/) with FastAPI |
| Experiments with explicit multi-agent routing, MCP or A2A | [AG2](/ag2-guide/) | [Strands Agents](/strands-agents-guide/), [LangGraph](/langgraph-guide/python/) |
| AWS deployment | [Strands Agents](/strands-agents-guide/) with [AgentCore](/amazon-bedrock-agents-guide/) | [LangGraph](/langgraph-guide/python/) on AgentCore |
| Azure or .NET | [Microsoft Agent Framework](/microsoft-agent-framework-guide/) | [Semantic Kernel](/semantic-kernel-guide/) for existing code |
| Google Cloud or Gemini | [Google ADK](/google-adk-guide/) | |
| An agent that works with files, a shell and the web | [Claude Agent SDK](/anthropic-claude-agent-sdk-guide/) | |
| TypeScript | [LangGraph](/langgraph-guide/typescript/), [OpenAI Agents SDK](/openai-agents-sdk-typescript-guide/) | [Claude Agent SDK](/anthropic-claude-agent-sdk-typescript-guide/), [Strands Agents](/strands-agents-guide/), [Google ADK](/google-adk-guide/typescript/) |

### Decision tree

```
Start
  ├─ New to agents? → SmolAgents, or Agno / PydanticAI if you want typed or served
  │
  ├─ Need validated, typed output? → PydanticAI
  │
  ├─ Multi-agent?
  │  ├─ Role-based crew → CrewAI
  │  ├─ Handoffs between specialists → OpenAI Agents SDK
  │  ├─ Teams served over HTTP → Agno
  │  └─ Explicit routing, MCP and A2A → AG2 or Strands Agents
  │
  ├─ Loops, branching, resumable state? → LangGraph
  │
  ├─ Data and search?
  │  ├─ Retrieval over your documents → LlamaIndex
  │  └─ Serialisable pipelines → Haystack
  │
  ├─ Cloud platform?
  │  ├─ AWS → Strands Agents + AgentCore
  │  ├─ Azure / .NET → Microsoft Agent Framework
  │  └─ Google Cloud → Google ADK
  │
  └─ Tied to one model vendor?
     ├─ Claude → Claude Agent SDK
     ├─ Mistral → Mistral Agents API
     └─ OpenAI → OpenAI Agents SDK
```

## Moving between frameworks

Where an upstream or in-guide migration path exists:

- **OpenAI Swarm to the OpenAI Agents SDK:** see the [Swarm migration guide](/openai-agents-sdk-guides/openai_agents_sdk_swarm_migration_guide/).
- **AG2 Classic (`import autogen`) to AG2 1.x:** see the [AG2 Classic migration guide](/ag2-guide/classic/ag2_classic_migration_guide/).
- **Microsoft AutoGen or Semantic Kernel agents to Microsoft Agent Framework:** Microsoft publishes migration guides for both; they are linked from the [Agent Framework overview](/microsoft-agent-framework-guide/).
- **Claude Code SDK to Claude Agent SDK:** see the [Claude migration guide](/anthropic-claude-agent-sdk-guide/anthropic_claude_agent_sdk_migration_guide/).

Moves between other frameworks are rewrites. The parts that carry over are your tool functions, prompts and evaluation data, which is a good reason to keep them in plain functions and files rather than inside one framework's classes.

---

[Back to Home](/) | [All Guides](/guides/) | [Quick Start](/quick-start/)

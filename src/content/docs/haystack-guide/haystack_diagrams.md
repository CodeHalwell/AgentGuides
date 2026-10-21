---
title: "Haystack Architecture and Pattern Diagrams"
description: "Diagrams of how Haystack 3 works: pipelines, the Agent loop and its hooks, wrapping pipelines as components and tools, multi-agent composition and tracing spans."
framework: "haystack"
language: "python"
sidebar:
  label: "Diagrams"
  order: 9
---

> Reflects `haystack-ai` %%version=haystack-ai%% (checked %%versions-checked%%). Every box is a real Haystack class, socket or hook point; the linked sections have the code.

## Contents

1. [Components and pipelines](#components-and-pipelines)
2. [RAG: indexing and querying](#rag-indexing-and-querying)
3. [Routing and loops](#routing-and-loops)
4. [The Agent loop](#the-agent-loop)
5. [Hooks timeline](#hooks-timeline)
6. [Pipelines as components and tools](#pipelines-as-components-and-tools)
7. [Multi-agent composition](#multi-agent-composition)
8. [Tracing spans](#tracing-spans)

---

## Components and pipelines

A component declares typed inputs (the parameters of `run()`) and typed outputs (`@component.output_types`). `pipeline.connect("a.out", "b.in")` checks that the types match. `Pipeline.run()` returns the outputs of leaf components only, unless you pass `include_outputs_from`. See [Components and Pipelines](/haystack-guide/haystack_comprehensive_guide/#components-and-pipelines).

```
           pipeline.run({"stats": {"text": "..."}})
                          │
                          ▼
              ┌──────────────────────────┐
              │ stats: TextStats         │
              │  in:  text: str          │
              │  out: text: str          │
              │       word_count: int    │
              │       char_count: int ───┼──► (unconnected, not returned)
              └─────┬──────────┬─────────┘
          stats.text│          │stats.word_count
                    ▼          ▼
              ┌──────────────────────────┐
              │ formatter: Formatter     │
              │  in:  text, word_count   │
              │  out: formatted: str ────┼──► result["formatter"]["formatted"]
              └──────────────────────────┘
```

The same graph runs three ways: `run()` (sequential), `await run_async()` (independent branches concurrently, bounded by `concurrency_limit`) and `stream()` (an async iterator of `StreamingChunk`s, with `handle.result` at the end).

## RAG: indexing and querying

The components from the [knowledge-base recipe](/haystack-guide/haystack_recipes/#knowledge-base-question-answering-system). Indexing and querying are separate pipelines that share a document store.

```
INDEXING                                        QUERYING

list[Document]                                  question
     │                                              │
     ▼                                              ▼
DocumentCleaner                             InMemoryBM25Retriever ──documents──┐
     │ documents                                    │ documents                │
     ▼                                              ▼                          │
DocumentSplitter                            ChatPromptBuilder                  │
     │ documents                                    │ prompt                   │
     ▼                                              ▼                          │
DocumentWriter ──► InMemoryDocumentStore ◄── OpenAIChatGenerator              │
                   (or Elasticsearch,               │ replies                  │
                    Qdrant, ...)                    ▼                          │
                                              AnswerBuilder ◄──────────────────┘
                                                    │ answers: list[GeneratedAnswer]
                                                    ▼
```

For dense or hybrid retrieval, add `OpenAIDocumentEmbedder` before the writer, `OpenAITextEmbedder` before an embedding retriever, and join the two retrievers' results with `DocumentJoiner(join_mode="reciprocal_rank_fusion")`.

## Routing and loops

Only the branch that receives input runs. `BranchJoiner` merges branches (and closes loops); its output socket is `value`. See [Branching with ConditionalRouter](/haystack-guide/haystack_comprehensive_guide/#branching-with-conditionalrouter) and [Iterative refinement loop](/haystack-guide/haystack_comprehensive_guide/#iterative-refinement-loop).

```
Routing                                    Loop (Pipeline(max_runs_per_component=3))

        query                              ChatPromptBuilder
          │                                      │ prompt
          ▼                                      ▼
  ConditionalRouter                        BranchJoiner ◄─────────────┐
   │ billing    │ general                        │ value              │
   ▼            ▼                                ▼                    │
 Agent A      Agent B                     OpenAIChatGenerator         │
   │ last_message │                              │ replies            │ feedback
   └──────┬───────┘                              ▼                    │
          ▼                                LengthValidator ───────────┘
     BranchJoiner                                │ valid
          │ value                                ▼
          ▼
```

A loop that never satisfies its validator raises `PipelineMaxComponentRuns` once a component exceeds `max_runs_per_component`.

## The Agent loop

What `Agent.run()` does, with the hook points and state keys involved. See [Agent Concepts](/haystack-guide/haystack_comprehensive_guide/#agent-concepts-in-haystack).

```
agent.run(messages=[...], **prompt_variables, **state_values, hook_context={...})
  │
  ▼
┌────────────────────────────────────────────┐
│ render system_prompt / user_prompt         │
│ initialise State: messages, step_count,    │
│   token_usage, tool_call_counts, your keys │
│ before_run hooks                           │
└────────────────────┬───────────────────────┘
                     ▼
        ┌──────────────────────────┐
        │ before_llm hooks         │◄───────────────────────────────┐
        │ (TokenBudgetHook,        │                                │
        │  CompactionHook, ...)    │── stop_run set? ──► exit_reason │
        └────────────┬─────────────┘    = that value               │
                     ▼                                             │
        ┌──────────────────────────┐                               │
        │ chat_generator.run(      │                               │
        │   messages, tools)       │                               │
        └────────────┬─────────────┘                               │
                     │                                             │
           tool calls in reply?                                    │
            │ no               │ yes                               │
            ▼                  ▼                                   │
   exit_reason = "text"  ┌──────────────────────────────┐          │
   (or "length",         │ before_tool hooks            │          │
    "content_filter")    │   (ConfirmationHook)         │          │
            │            │ run tools concurrently       │          │
            │            │   (tool_concurrency_limit)   │          │
            │            │ outputs_to_state ──► State   │          │
            │            │ after_tool hooks             │          │
            │            │   (ToolResultOffloadHook)    │          │
            │            └──────────────┬───────────────┘          │
            │                           │                          │
            │            exit-condition tool called?               │
            │              │ yes                 │ no              │
            │              ▼                     └─ step_count < ──┘
            │       exit_reason = <tool name>       max_agent_steps
            │              │                     (else exit_reason =
            ▼              ▼                      "max_agent_steps")
        ┌──────────────────────────┐   continue_run = True
        │ on_exit hooks            │───────────────────────► back to before_llm
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │ after_run hooks          │
        │ return messages,         │
        │   last_message,          │
        │   step_count,            │
        │   token_usage,           │
        │   tool_call_counts,      │
        │   exit_reason, your keys │
        └──────────────────────────┘
```

## Hooks timeline

One run with two tool-calling steps and a final text reply. Built-in hooks are shown at the only point each accepts. See [Hooks](/haystack-guide/haystack_comprehensive_guide/#hooks) and [Context Management](/haystack-guide/haystack_comprehensive_guide/#context-management).

```
before_run
│
├─ before_llm ── TokenBudgetHook, CompactionHook ── model call 1 ── tool calls
│   ├─ before_tool ── ConfirmationHook (confirm / reject / modify)
│   │     tools run
│   └─ after_tool ── ToolResultOffloadHook
│
├─ before_llm ── model call 2 ── tool calls
│   ├─ before_tool
│   └─ after_tool
│
├─ before_llm ── model call 3 ── text reply
│
on_exit ── may set continue_run = True ──► another before_llm round
│
after_run
```

Every hook receives the live `State`. Hooks read request-scoped objects from `state.get("hook_context")` and end the run by setting `stop_run` (read before each model call).

## Pipelines as components and tools

`SuperComponent` gives a pipeline a component interface; `PipelineTool` gives it a tool interface for an agent. Both use `input_mapping` and `output_mapping`. See [Wrapping a pipeline as a component](/haystack-guide/haystack_comprehensive_guide/#wrapping-a-pipeline-as-a-component).

```
                         SuperComponent / PipelineTool
               ┌──────────────────────────────────────────────┐
  query ───────┼─► input_mapping {"query": ["retriever.query", │
               │                          "analyst.query"]}   │
               │      │                         │             │
               │      ▼                         ▼             │
               │  retriever ──documents──► analyst (Agent)    │
               │                                │last_message │
               │  output_mapping {"analyst.last_message":     │
               │                  "answer"} ◄───┘             │
               └──────────────────────┬───────────────────────┘
                                      ▼
                                   answer

  As a component: another pipeline connects to it like any component.
  As a tool:      the agent sees one tool (name, description, "query" parameter);
                  the result goes back to the model as text (outputs_to_string).
```

`ComponentTool` does the same for a single component, and `AgentTool` for a whole agent.

## Multi-agent composition

The two main shapes, from the [multi-agent guide](/haystack-guide/haystack_multi_agent_guide/).

```
Coordinator (order decided at run time)       Pipeline stages (order fixed)

 ┌──────────────────────────────┐             research (Agent, user_prompt "Topic: {{ topic }}")
 │ coordinator: Agent           │                  │ last_message
 │  tools = [AgentTool(...),    │                  ▼
 │           AgentTool(...)]    │             write (Agent, user_prompt "...{{ notes.text }}")
 └───┬──────────────────────┬───┘                  │ last_message
     │ tool call            │ tool call            ▼
     ▼                      ▼
 AgentTool "researcher"  AgentTool "writer"
   runs its own Agent      runs its own Agent
   loop, returns final     loop, returns final
   text (or copies keys    text
   via outputs_to_state)
```

Each sub-agent has its own loop, `max_agent_steps` and hooks. A coordinator's `token_usage` covers only its own model calls.

## Tracing spans

The spans Haystack emits once a tracer is enabled (`tracing.enable_tracing(...)` or a connector such as `LangfuseConnector`). Inputs and outputs are attached only with content tracing on. See [Observability](/haystack-guide/haystack_observability_python/#what-gets-traced).

```
haystack.pipeline.run            (tag haystack.pipeline.execution_mode = sync | async)
└─ haystack.component.run        (one per component run)
   └─ haystack.agent.run         (when the component is an Agent; also emitted for standalone runs)
      └─ haystack.agent.step     (one per loop iteration)
         ├─ haystack.agent.step.llm
         └─ haystack.agent.step.tool   (one per tool call, tag haystack.tool.name)
```

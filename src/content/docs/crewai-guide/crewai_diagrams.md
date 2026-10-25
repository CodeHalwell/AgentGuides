---
title: "CrewAI diagrams"
description: "Text diagrams of CrewAI's moving parts — crew components, the agent loop, sequential and hierarchical processes, delegation, flows, unified memory, events and hooks, and checkpoints."
framework: "crewai"
language: "python"
sidebar:
  label: "Diagrams"
  order: 60
---

> **Checked against `crewai` %%version=crewai%%** (checked %%versions-checked%%). These diagrams describe behaviour verified on the linked pages; they simplify, and the pages are the reference.

## Components of a crew

```text
                          Crew
   process · memory · knowledge_sources · max_rpm · checkpoint · tracing
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
      Agents              Tasks          manager_llm / manager_agent
                                         (hierarchical crews only)
         │                  │
         │                  ├─ description, expected_output   (required)
         │                  ├─ agent, context, async_execution
         │                  └─ output_pydantic / response_model, guardrail
         │
         ├─ role, goal, backstory   (required)
         ├─ llm ──────────────► LLM: native SDK (openai, anthropic, gemini, …) or LiteLLM
         ├─ tools, mcps, apps
         ├─ knowledge_sources ───► vector store (ChromaDB by default)
         └─ memory ──────────────► Memory (falls back to the crew's)
```

Reference: [Agent API](/crewai-guide/crewai_agent_api_python/), [Task API](/crewai-guide/crewai_task_api_python/), [LLM providers](/crewai-guide/crewai_llm_providers_python/).

## One task, inside the agent

```text
  task description + expected_output
  + context (earlier task outputs)
  + recalled memories (up to 5)
  + knowledge chunks (after an LLM query rewrite)
              │
              ▼
      ┌──────────────┐   tool call    ┌──────────────┐
      │     LLM      │ ─────────────► │     tool     │
      │              │ ◄───────────── │ (cached only │
      └──────────────┘  observation   │  if opted in)│
              │                       └──────────────┘
              │ final answer           loop ends at the final answer,
              ▼                        max_iter or max_execution_time
      ┌──────────────┐
      │  guardrail   │── fails ──► reason sent back, task retried
      └──────────────┘             (Task.guardrail; up to guardrail_max_retries)
              │ passes
              ▼
         TaskOutput (raw, pydantic, json_dict)
```

## Sequential process

```text
  kickoff(inputs={...})
        │
        ▼
   ┌─────────┐      ┌─────────┐      ┌─────────┐
   │ Task 1  │ ───► │ Task 2  │ ───► │ Task 3  │ ───► CrewOutput
   └─────────┘      └─────────┘      └─────────┘      (.raw = Task 3's output,
        │                │                ▲            .tasks_output = all three)
        └────────────────┴── outputs ─────┘
             without context=[...], each task sees ALL earlier outputs
```

Async tasks (`async_execution=True`) that follow each other run together; the next synchronous task waits for them:

```text
   ┌─────────┐
   │ Task A  │ async ─┐
   └─────────┘        │      ┌───────────┐
   ┌─────────┐        ├────► │ Summary   │  (sync: waits for A and B)
   │ Task B  │ async ─┘      └───────────┘
   └─────────┘
```

## Hierarchical process

```text
                 ┌──────────────────────────┐
  tasks ───────► │ Manager                  │  manager_llm (generic manager)
                 │ runs every task, using   │  or manager_agent (your own)
                 │ delegation tools         │  TaskOutput.agent = manager's role
                 └──────────────────────────┘
                    │ delegate        ▲ result
          ┌─────────┴────────┬────────┴─────────┐
          ▼                  ▼                  ▼
     ┌──────────┐      ┌──────────┐      ┌──────────┐
     │ Worker 1 │      │ Worker 2 │      │ Worker 3 │   agents=[...]
     └──────────┘      └──────────┘      └──────────┘   (manager not listed)
```

## Delegation between agents

```text
   Agent(allow_delegation=True)
        │
        │ gets two tools whose targets are the other agents in the crew
        ├──► "Delegate work to coworker"  ──► coworker runs a full sub-task
        │    (delegate_work_to_coworker)       sees only the task + context it is given
        └──► "Ask question to coworker"   ──► coworker answers one question
             (ask_question_to_coworker)
```

## A flow

```text
  kickoff(inputs={...}) ──► state (Pydantic model or dict, with an id)
        │
        ▼
  ┌──────────────┐  ┌──────────────┐
  │ @start()     │  │ @start()     │      several start methods are allowed
  │ fetch_news   │  │ fetch_filings│
  └──────┬───────┘  └──────┬───────┘
         └────────┬────────┘
                  ▼
        @listen(and_(fetch_news, fetch_filings))      or_(...) fires on the first
        ┌──────────────┐
        │ analyse      │
        └──────┬───────┘
               ▼
        @router(analyse) -> Literal["publish", "review"]
               │
       "publish"│"review"           labels must not equal a listener's own name
        ┌──────┴───────┐
        ▼              ▼
   @listen("publish")  @listen("review") + @human_feedback ──► pause / resume
        │
        ▼
   return value of the last method = kickoff() result

   @persist ──► state saved after each method (SQLite by default);
                kickoff(inputs={"id": ...}) restores it
```

Reference: [Flows](/crewai-guide/crewai_flows_guide/).

## Unified memory

```text
                      Memory(storage=..., embedder=..., llm=...)
                                    │
             ┌──────────────────────┼───────────────────────┐
             ▼                      ▼                       ▼
     mem.scope("/users/ada")   mem.slice(scopes=[...])   Crew(memory=mem)
     read and write under      read-only view across     recalled into each task
     one path                  several paths             prompt; search/save tools;
                                                         results saved afterwards
                                    │
                                    ▼
          remember()  ──► LLM infers whatever scope, categories, importance
                          you leave out; near-duplicates consolidated
          recall()    ──► depth="shallow": embed + vector search
                          depth="deep":    long queries (200+ chars) are split into
                                           sub-queries by the LLM; low-confidence
                                           results trigger one more LLM round
                          score = semantic · recency · importance weights
                                    │
                                    ▼
                    storage: LanceDB (default) · Qdrant Edge · custom backend
```

The separate short-term, long-term and entity memories of older releases no longer exist; scopes and categories replace them. Reference: [Memory](/crewai-guide/crewai_memory_python/).

## Events and hooks

```text
   crew / task / agent / LLM / tool / flow / memory
                    │
        ┌───────────┴─────────────────────────────┐
        │ hooks (in the call path, synchronous)   │  before_llm_call / before_tool_call:
        │ can block or rewrite                    │    return False to block
        └───────────┬─────────────────────────────┘  after_llm_call / after_tool_call:
                    │                                   return a replacement string
                    ▼
        crewai_event_bus.emit(event)
                    │
        ┌───────────┴─────────────────────────────┐
        │ listeners (observe only, thread pool)   │──► logs, metrics, OpenTelemetry spans,
        │ BaseEventListener / @bus.on(EventType)  │    CrewAI AMP traces
        └─────────────────────────────────────────┘
```

Reference: [Production](/crewai-guide/crewai_production_guide/#event-listeners).

## Checkpoints and resume

```text
   Crew(checkpoint=CheckpointConfig(on_events=["task_completed"]))

   Task 1 ✓ ──► .checkpoints/main/<time>_<id>_p-none.json
   Task 2 ✓ ──► .checkpoints/main/<time>_<id>_p-<parent>.json
   Task 3 ✗  (process dies)

   crewai checkpoint list ./.checkpoints
   crew.kickoff(from_checkpoint=CheckpointConfig(restore_from="<file>"))
        └──► restores the run and continues
```

Reference: [Production → Recovery with checkpoints](/crewai-guide/crewai_production_guide/#recovery-with-checkpoints).

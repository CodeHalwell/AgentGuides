---
title: "Start Here: Haystack Reading Paths"
description: "Which Haystack pages to read, in what order, depending on what you are building."
framework: haystack
---

This section covers Haystack 3 (`haystack-ai` %%version=haystack-ai%%, checked %%versions-checked%%). Each path below lists pages in reading order.

## I'm new to Haystack

1. [Comprehensive guide → Installation and Setup](/haystack-guide/haystack_comprehensive_guide/#installation-and-setup)
2. [Architecture overview](/haystack-guide/haystack_comprehensive_guide/#haystack-3-architecture-overview) and the [diagrams](/haystack-guide/haystack_diagrams/)
3. [Components and Pipelines](/haystack-guide/haystack_comprehensive_guide/#components-and-pipelines), including the minimal RAG pipeline
4. [Creating Agents with the Agent Class](/haystack-guide/haystack_comprehensive_guide/#creating-agents-with-the-agent-class)
5. One recipe from the [recipes page](/haystack-guide/haystack_recipes/)

## I'm upgrading from Haystack 2.x

1. [Migrating from Haystack 2.x](/haystack-guide/haystack_comprehensive_guide/#migrating-from-haystack-2x)
2. The upstream [MIGRATION.md](https://github.com/deepset-ai/haystack/blob/main/MIGRATION.md)
3. [Observability](/haystack-guide/haystack_observability_python/) — tracing is no longer enabled automatically

## I need knowledge-base Q&A (RAG)

1. [A Minimal RAG Pipeline](/haystack-guide/haystack_comprehensive_guide/#a-minimal-rag-pipeline)
2. [Recipe: knowledge-base Q&A](/haystack-guide/haystack_recipes/#knowledge-base-question-answering-system)
3. [LLMRanker](/haystack-guide/haystack_comprehensive_guide/#llmranker) for re-ranking

## I want multi-agent systems

1. [Multi-Agent Pipeline Design Patterns](/haystack-guide/haystack_comprehensive_guide/#multi-agent-pipeline-design-patterns)
2. [Multi-agent guide](/haystack-guide/haystack_multi_agent_guide/)
3. [Advanced agents](/haystack-guide/haystack_advanced_agents_python/) for hooks and shared state

## I need to deploy to production

1. [Production guide](/haystack-guide/haystack_production_guide/)
2. [Streaming server (FastAPI)](/haystack-guide/haystack_streaming_server_fastapi/)
3. [Observability](/haystack-guide/haystack_observability_python/)

## Page guide

| Page | Use it for |
|------|------------|
| [Comprehensive guide](/haystack-guide/haystack_comprehensive_guide/) | Core concepts, the Agent, tools, multi-agent basics, migration |
| [Advanced agents](/haystack-guide/haystack_advanced_agents_python/) | Hooks, state handlers, custom components, async |
| [Multi-agent guide](/haystack-guide/haystack_multi_agent_guide/) | Coordinator, pipeline and routing patterns |
| [Recipes](/haystack-guide/haystack_recipes/) | End-to-end examples to adapt |
| [Diagrams](/haystack-guide/haystack_diagrams/) | Architecture pictures |
| [Observability](/haystack-guide/haystack_observability_python/) | Tracing, logging, error recovery |
| [Production guide](/haystack-guide/haystack_production_guide/) | Serving, serialisation, containers, scaling |
| [Streaming server](/haystack-guide/haystack_streaming_server_fastapi/) | Server-sent events from an Agent |

Code examples assume `OPENAI_API_KEY` is set. To try them without an API key, swap `OpenAIChatGenerator` for `MockChatGenerator` from `haystack.components.generators.chat`, which returns canned or echoed replies.

---
title: "Mistral Agents API - May 2025 Launch and What Changed Since"
description: "What Mistral launched with the Agents API on 27 May 2025, and how the API and the Python SDK have moved on since."
framework: mistral-agents-api
---

This is a historical page. For current usage see the [comprehensive guide](/mistral-agents-api-guide/mistral_agents_api_comprehensive_guide/), which is verified against `mistralai` %%version=mistralai%% (checked %%versions-checked%%).

## The launch (27 May 2025)

Mistral released the Agents API as a beta alongside its chat completions API. The announcement covered:

- **Agents** — stored configurations of model, instructions and tools, created through `/v1/agents`.
- **Conversations** — server-side conversation state with append, restart (branching) and streaming, through `/v1/conversations`.
- **Built-in connectors** — code execution, image generation, web search (with a premium variant adding news-agency sources), and document libraries for retrieval.
- **Handoffs** — agents passing a conversation to other agents.
- **MCP** — use of Model Context Protocol tools.

Mistral reported that enabling web search raised SimpleQA accuracy from 23% to 75% for Mistral Large and from 22.08% to 82.32% for Mistral Medium. Those figures were for the models of the time (Mistral Medium 3, `mistral-medium-2505`, was then current).

"Persistent memory" in the launch material referred to Conversations storing history server-side; it was never a separate tool type.

## What changed since

| When | Change |
|---|---|
| 2025–2026 | New tool types and fields: `document_library` with `library_ids`, MCP `connector` tools with stored credentials and OAuth, `tool_configuration` (`include`, `exclude`, `requires_confirmation`), agent versions and version aliases, guardrails, `handoff_execution` (`server` / `client`). |
| March 2026 | Python SDK v2: all imports moved from `mistralai` to `mistralai.client`; Azure and GCP clients moved under `mistralai.azure` / `mistralai.gcp`; 42 request and response types renamed. |
| Mid 2026 | `mistral-medium-2505` and `mistral-medium-2508` deprecated, then retired on 31 August 2026; replaced by Mistral Medium 3.5 (`mistral-medium-3-5`, alias `mistral-medium-latest`). `mistral-moderation-2411` retired 30 June 2026 in favour of `mistral-moderation-2603`. |
| September 2026 | Python SDK v3.0.0: `httpx2` transport, `mcp` 2.2, service-account token support, built-in tools removed from chat and agents completions, connector and observability API changes. See [Changes in SDK v3](/mistral-agents-api-guide/mistral_agents_api_comprehensive_guide/#14-changes-in-sdk-v3). |

## Pages in this guide

- [Connectors and built-in tools](/mistral-agents-api-guide/mistral_agents_api_connectors_guide/)
- [Agent orchestration](/mistral-agents-api-guide/mistral_agents_api_orchestration_guide/)
- [MCP](/mistral-agents-api-guide/mistral_agents_api_mcp_guide/)
- [Recipes](/mistral-agents-api-guide/mistral_agents_api_recipes/)
- [Production](/mistral-agents-api-guide/mistral_agents_api_production_guide/)

# AgentGuides Update Report — April 18, 2026

**Prepared by:** Automated review process  
**Review date:** April 18, 2026  
**Previous review:** April 17, 2026  
**Scope:** All 18+ framework guides across Python, TypeScript, .NET, and Go

---

## Executive Summary

This report documents the changes made during the April 18, 2026 review of the AgentGuides repository. The primary driver was a same-day release of new versions of the **Anthropic Claude Agent SDK** (Python v0.1.63, TypeScript v0.2.113) with developer-quality-of-life improvements. All other framework guides were audited and confirmed current; the PydanticAI comprehensive guide received a documentation enhancement for the v1.84.0 features released the previous day.

---

## Changes Made

### 1. Anthropic Claude Agent SDK (Python) — Updated to v0.1.63

**File:** `Anthropic_Claude_Agent_SDK_Guide/README.md`  
**File:** `Anthropic_Claude_Agent_SDK_Guide/anthropic_claude_agent_sdk_comprehensive_guide.md`

**New features documented:**

#### `get_context_usage()` on `ClaudeSDKClient`
The new `get_context_usage()` method returns a per-category breakdown of context window consumption:
- `system_tokens` — tokens consumed by the system prompt
- `conversation_tokens` — tokens from conversation history
- `tool_tokens` — tokens attributed to tool results
- `total_tokens` / `context_window` / `utilisation_pct` — aggregate stats

This enables proactive context management — for example, triggering compaction or summary when utilisation exceeds a threshold.

#### `typing.Annotated` support in `@tool` and `create_sdk_mcp_server`
Tool function parameters can now carry per-parameter descriptions using `Annotated[type, "description"]`. The SDK automatically generates accurate JSON Schema from these annotations, improving LLM tool-call accuracy without requiring a separate schema dictionary.

```python
from typing import Annotated
from claude_agent_sdk import tool

@tool("search_codebase", "Search the project codebase for patterns")
async def search_codebase(
    pattern:   Annotated[str,  "The regex or literal string to search for"],
    directory: Annotated[str,  "Relative path to the directory to search in"] = ".",
    max_results: Annotated[int, "Maximum number of results to return (1–100)"] = 20
) -> dict:
    ...
```

#### `tool_use_id` and `agent_id` in `ToolPermissionContext`
Permission handler callbacks now receive two additional context fields:
- `tool_use_id` — the unique ID of the specific tool call (allows distinguishing parallel invocations)
- `agent_id` — the ID of the subagent making the permission request (useful in multi-agent setups)

This makes it possible to build fine-grained permission policies that behave differently based on which agent is requesting permission and which specific parallel invocation is being evaluated.

---

### 2. Anthropic Claude Agent SDK (TypeScript) — Updated to v0.2.113

**File:** `Anthropic_Claude_Agent_SDK_TypeScript_Guide/README.md`

Same features as the Python SDK, with TypeScript-idiomatic naming:
- `getContextUsage()` (camelCase) returning `ContextUsage` with `systemTokens`, `conversationTokens`, `toolTokens`, `totalTokens`, `contextWindow`, `utilisationPct`
- JSDoc `@param` annotations on `@tool` parameters for automatic JSON Schema description generation
- `toolUseId` and `agentId` in `ToolPermissionContext`

---

### 3. PydanticAI Comprehensive Guide — v1.84.0 Features Completed

**File:** `PydanticAI_Guide/pydantic_ai_comprehensive_guide.md`

The revision history entry for v1.84.0 was incomplete (only mentioned the Google FileSearchTool regex fix). The following new sections and code examples were added:

| Feature | Description |
|---------|-------------|
| `OllamaModel` subclass | Dedicated first-class class replacing `OpenAIModel` workaround; fixes structured output on Ollama Cloud |
| `XSearchTool` / `FileSearchTool` for xAI | Built-in search and file retrieval tools for Grok provider |
| `FastMCPToolset` metadata injection | `inject_metadata=True` attaches `call_id`, timestamp, and `agent_id` per tool call |
| Bedrock prompt cache TTL | `cache_ttl` parameter on `BedrockModel` for configurable response caching |
| Stateful `OpenAICompaction` | `mode='stateful'` compacts history while preserving state references |
| Claude Opus 4.7 support | `anthropic:claude-opus-4-7` recognised as a valid model string |

---

### 4. README.md — Stale Date Corrected

**File:** `README.md`

The "Last Updated" section incorrectly showed "November 21, 2025" despite the guides having been updated through April 2026. This has been corrected to "April 18, 2026" with per-framework version details. Revision history updated.

---

### 5. versions.json — Updated

**File:** `versions.json`

| Field | Previous | Updated |
|-------|----------|---------|
| `last_updated` | April 17, 2026 | April 18, 2026 |
| `anthropic-claude-agent-sdk-python` | 0.1.60 (April 16, 2026) | 0.1.63 (April 18, 2026) |
| `anthropic-claude-agent-sdk-typescript` | 0.2.110 (April 16, 2026) | 0.2.113 (April 18, 2026) |
| `anthropic-claude-sdk` notes | v0.1.60 features | Extended to include v0.1.63 features |

---

## Framework Audit Summary

All 18+ framework guides were audited against official release notes, changelogs, and PyPI/npm registries. The table below summarises the findings:

| Framework | Guide Version | Latest Release | Status |
|-----------|--------------|----------------|--------|
| OpenAI Agents SDK (Python) | 0.14.1 | 0.14.1 (Apr 15) | ✅ Current |
| OpenAI Agents SDK (TypeScript) | 0.8.3 | 0.8.3 (Apr 9) | ✅ Current |
| PydanticAI | 1.84.0 | 1.84.0 (Apr 17) | ✅ Current (enhanced) |
| CrewAI | 1.14.0 | 1.14.0 (Apr 7) | ✅ Current |
| AG2 (AutoGen) | 0.11.5 | 0.11.5 (Apr 5) | ✅ Current |
| LangGraph (Python) | 1.1.6 | 1.1.6 (Apr 10) | ✅ Current |
| LangGraph (TypeScript) | 1.2.8 | 1.2.8 (Apr 11) | ✅ Current |
| LlamaIndex (Python) | 0.14.20 | 0.14.20 (Apr 3) | ✅ Current |
| LlamaIndex (TypeScript) | @llamaindex/workflow 1.1.4 | 1.1.4 (Apr 15) | ✅ Current |
| Haystack | 2.27.0 | 2.27.0 (Apr 1) | ✅ Current |
| Google ADK (Python) | 1.30.0 | 1.30.0 (Apr 13) | ✅ Current |
| Google ADK (Go) | 1.0.0 GA | 1.0.0 GA (Apr 8) | ✅ Current |
| Google ADK (TypeScript) | 0.1.x | 0.1.x (Dec 2025) | ✅ Current |
| Amazon Bedrock Strands | 1.35.0 | 1.35.0 | ✅ Current |
| Microsoft Agent Framework | 1.0 GA | 1.0 GA (Apr 3) | ✅ Current |
| Semantic Kernel (Python) | 1.41.2 | 1.41.2 (Apr 8) | ✅ Current |
| Semantic Kernel (.NET) | 1.74.0 | 1.74.0 | ✅ Current |
| Mistral Agents API | 2.0.1 | 2.0.1 (Mar 12) | ✅ Current |
| Anthropic Claude SDK (Python) | **0.1.60 → 0.1.63** | 0.1.63 (Apr 18) | ✅ Updated |
| Anthropic Claude SDK (TypeScript) | **0.2.110 → 0.2.113** | 0.2.113 (Apr 18) | ✅ Updated |
| SmolAgents | 1.24.0 | 1.24.0 (Jan 16) | ✅ Current |
| AutoGen (Legacy) | — | Deprecated | ✅ No change needed |

---

## No Action Required

The following items were reviewed and found to require no changes:

- **Deprecated methods**: No deprecated patterns found in active guides. The AutoGen Legacy guide correctly directs users to AG2.
- **Breaking changes**: All breaking changes from recent releases (OpenAI SDK v2 requirement, LlamaIndex hard-removed legacy agents, Haystack Python 3.9 drop, PydanticAI `result_*` removal) are documented in their respective guides.
- **Code examples**: All guides have comprehensive code examples covering basic use through advanced patterns including multi-agent systems, tool use, memory, context management, checkpoints, observability, and streaming.

---

## Cross-Cutting Standards Verification

| Standard | Status |
|---------|--------|
| British English spelling | ✅ Consistent throughout |
| Revision history in all guides | ✅ Present and updated |
| Version numbers in guide headers | ✅ All current |
| Breaking changes documented | ✅ All documented |
| Code examples for new features | ✅ Added for v0.1.63 and v1.84.0 |
| Beginner-to-advanced learning paths | ✅ Present in all major guides |

---

## Recommended Follow-Up (Next Review)

1. **Google ADK Python** — The bi-weekly release cadence means a new version (approximately 1.32.x) is likely within 2 weeks. Monitor for new session management features and additional language model support.
2. **LangGraph** — Watch for v1.2.x with potential additional v2 API enhancements.
3. **Haystack** — v2.28.0 expected within weeks; monitor for additional component integrations.
4. **PydanticAI** — Rapid release cadence (2–3 releases per week); monitor for v1.85+.
5. **CrewAI** — Monitor for v1.15.0 and any changes to the Flows system.

---

*Report generated: April 18, 2026*

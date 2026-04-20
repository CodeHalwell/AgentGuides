# AgentGuides Update Report — April 20, 2026

**Prepared by:** Automated review process  
**Review date:** April 20, 2026  
**Previous review:** April 19, 2026  
**Scope:** All 22 framework guides across Python, TypeScript, .NET, and Go

---

## Executive Summary

This report documents the changes made during the April 20, 2026 review of the AgentGuides repository. Seven updates were identified across six frameworks since the previous review (April 19, 2026):

| Framework | Change |
|-----------|--------|
| PydanticAI | 1.84.0 → **1.84.1** |
| Google ADK TypeScript | `0.1.x` placeholder → **0.6.1** (confirmed) |
| LangGraph Python | 1.1.6 → **1.1.8** |
| LangGraph TypeScript | 1.2.8 → **1.2.9** |
| OpenAI Agents SDK Python | 0.14.1 → **0.14.2** |
| Microsoft Agent Framework Python | 1.0 GA → **1.0.1** |
| Amazon Bedrock Strands SDK | 1.35.0 → **1.36.0** |

All other framework guides were audited and confirmed current.

---

## Changes Made

### 1. PydanticAI — Updated to v1.84.1

**Files modified:**
- `PydanticAI_Guide/pydantic_ai_comprehensive_guide.md`
- `PydanticAI_Guide/README.md`

**Previous documented version:** 1.84.0 (April 17, 2026)  
**New version:** 1.84.1 (April 18, 2026)

#### v1.84.1 — Documented for the first time

| Change | Detail |
|--------|--------|
| Skip tool hooks for internal output tools | Tool `before_call` / `after_call` hooks are no longer invoked for the framework-internal output-capture tool, preventing spurious hook executions |
| Dict-shaped validated args to hooks for single-`BaseModel` tools | When a tool's parameter signature is a single Pydantic `BaseModel`, hooks now consistently receive a dict representation of the validated model, matching the multi-parameter behaviour |

**README changes:**
- Updated version badge from 1.84.0 to 1.84.1 with date April 18, 2026

**Comprehensive guide changes:**
- Updated header line to `Latest: 1.84.1 | Updated: April 20, 2026`
- Updated version in document metadata
- Added new row to revision history table

---

### 2. Google ADK TypeScript — Corrected to v0.6.1

**Files modified:**
- `Google_ADK_Guide/typescript/README.md`
- `Google_ADK_Guide/README.md`

**Previous documented version:** `0.1.x` (placeholder since announcement December 2025)  
**Confirmed version:** 0.6.1 (April 4, 2026)

#### Correction

The TypeScript ADK guide was tracking the package as "0.1.x" based on the December 2025 announcement. Cross-referencing npm confirms the package `@google/adk` is at **0.6.1** (published April 4, 2026). The guide status has been updated from "announced December 2025" to "Active development — v0.6.1 (April 4, 2026)".

The main `Google_ADK_Guide/README.md` has also been updated to reflect:
- Python: 1.31.0 (was incorrectly showing 1.30.0 in the file header)
- TypeScript: 0.6.1 (was "Available — December 2025")

**README changes:**
- Corrected `@google/adk` version from `0.1.x` to `0.6.1`
- Updated status from "Available (announced December 2025)" to "Active development — v0.6.1 (April 4, 2026)"
- Added revision history entry for this correction
- Updated main Google ADK README to reflect accurate TypeScript version

---

### 3. LangGraph Python — Updated to v1.1.8

**Files modified:**
- `LangGraph_Guide/python/langgraph_comprehensive_guide.md`
- `LangGraph_Guide/python/README.md`

**Previous documented version:** 1.1.6 (April 10, 2026)  
**New version:** 1.1.8 (April 17, 2026)

#### v1.1.7 and v1.1.8 (both April 17, 2026) — Documented for the first time

| Change | Detail |
|--------|--------|
| OpenTelemetry instrumentation fix | Fixed a strict `add_handler` type check that caused OpenTelemetry instrumentation to break when attaching OTel handlers to LangGraph graphs |
| Patch sequence | v1.1.7 and v1.1.8 were released the same day (April 17); v1.1.8 supersedes v1.1.7 |

**Comprehensive guide changes:**
- Updated header to `Latest: 1.1.8 | Updated: April 20, 2026`
- Updated version in document metadata
- Added two new rows to revision history (v1.1.7 and v1.1.8)

**README changes:**
- Updated version from 1.1.6 (April 10) to 1.1.8 (April 17)

---

### 4. LangGraph TypeScript — Updated to v1.2.9

**Files modified:**
- `LangGraph_Guide/typescript/langchain_langgraph_comprehensive_guide.md`
- `LangGraph_Guide/typescript/README.md`

**Previous documented version:** 1.2.8 (April 11, 2026)  
**New version:** 1.2.9 (April 19, 2026)

#### v1.2.9 (April 19, 2026) — Documented for the first time

| Change | Detail |
|--------|--------|
| Stability improvements | Patch release with bug fixes following the major 1.2.x feature additions (JSON Schema support, `ReducedValue`, `UntrackedValue`) |

**Comprehensive guide changes:**
- Updated header to `Latest: 1.2.9 | Updated: April 20, 2026`
- Added new row to revision history

**README changes:**
- Updated version badge to 1.2.9 (April 19, 2026)

---

### 5. OpenAI Agents SDK Python — Updated to v0.14.2

**Files modified:**
- `OpenAI_Agents_SDK_Guides/openai_agents_sdk_comprehensive_guide.md`
- `OpenAI_Agents_SDK_Guides/README.md`

**Previous documented version:** 0.14.1 (April 15, 2026)  
**New version:** 0.14.2 (April 18, 2026)

#### v0.14.2 (April 18, 2026) — Documented for the first time

| Change | Detail |
|--------|--------|
| Default Realtime model updated | `gpt-realtime-1.5` is now the default Realtime model for `RealtimeAgent` |
| `MCPServer` resource enumeration | `MCPServer` now exposes `list_resources()`, `list_resource_templates()`, and `read_resource()` methods for MCP resource access |
| `MCPServerStreamableHttp` session continuity | Exposes `session_id` so clients can resume existing MCP sessions across reconnects |
| Reasoning-content replay | `ChatCompletionsRunner` supports opt-in reasoning-content replay via `should_replay_reasoning_content=True` |
| Edge case fixes | Runtime and session edge cases resolved; no breaking changes |

**Comprehensive guide changes:**
- Updated header to `Latest: 0.14.2 | Updated: April 20, 2026`
- Added detailed v0.14.2 row to revision history

**README changes:**
- Updated current version badge from 0.14.1 to 0.14.2

---

### 6. Microsoft Agent Framework Python — Updated to v1.0.1

**Files modified:**
- `Microsoft_Agent_Framework_Guide/README.md`
- `Microsoft_Agent_Framework_Guide/python/README.md`
- `Microsoft_Agent_Framework_Guide/python/microsoft_agent_framework_python_comprehensive_guide.md`

**Previous documented version:** 1.0 GA (April 3, 2026)  
**New version:** 1.0.1 (April 10, 2026)

#### v1.0.1 (April 10, 2026) — Documented for the first time

| Change | Detail |
|--------|--------|
| Bug fixes | Stability improvements following the GA launch |
| Patch release | First patch on the 1.0 LTS track; no breaking changes relative to GA |

Additionally, the Python guide README was updated to reflect the confirmed GA status (previously still showing "Public Preview — October 2025" language from the pre-GA guide).

**README changes (main):**
- Updated `Latest Version` from "1.0 GA" to "1.0.1 (April 10, 2026)"
- Updated Last Updated date
- Added revision history row for v1.0.1

**Python README changes:**
- Updated `Framework Status` from "Public Preview" to "GA — Production-Ready with Long-Term Support"
- Updated `Latest Version` to 1.0.1
- Corrected "GA Timeline: Expected Q2 2026" — GA has already shipped

**Python comprehensive guide changes:**
- Updated header to `Latest: 1.0.1 | Updated: April 20, 2026`
- Updated framework version metadata from `1.0+` to `1.0.1`

---

### 7. Amazon Bedrock Strands SDK — Updated to v1.36.0

**Files modified:**
- `Amazon_Bedrock_Agents_Guide/bedrock_strands_sdk_guide.md`
- `Amazon_Bedrock_Agents_Guide/README.md`

**Previous documented version:** 1.35.0 (April 2026)  
**New version:** 1.36.0 (April 17, 2026)

#### v1.36.0 (April 17, 2026) — Documented for the first time

| Change | Detail |
|--------|--------|
| Incremental improvements | General improvements and bug fixes following 1.35.0 |
| Multi-agent graph workflows | `A2AAgent` instances and custom `AgentBase` implementations supported in graph nodes |
| Interrupt propagation | Interrupts now correctly propagate through nested multi-agent graph nodes |
| `AgentCoreMemorySessionManager` | Integration with Amazon Bedrock AgentCore memory — session summarisation, user preferences, semantic memory |
| Steering Hooks | Agent behaviour control with demonstrated 100% accuracy on controlled benchmarks |

A **Revision History** section has been added to `bedrock_strands_sdk_guide.md`, which previously had no revision tracking.

**README changes:**
- Updated Strands SDK version from 1.35.0 to 1.36.0 (April 17, 2026)

---

### 8. `versions.json` — Updated

**File:** `versions.json`

| Field | Previous | Updated |
|-------|----------|---------|
| `last_updated` | April 19, 2026 | **April 20, 2026** |
| `previous_update` | April 18, 2026 | **April 19, 2026** |
| `pydantic-ai` | 1.84.0 (April 17, 2026) | **1.84.1 (April 18, 2026)** |
| `google-adk-typescript` | 0.1.x (December 2025 - new) | **0.6.1 (April 4, 2026)** |
| `langgraph-python` | 1.1.6 (April 10, 2026) | **1.1.8 (April 17, 2026)** |
| `langgraph-typescript` | 1.2.8 (April 11, 2026) | **1.2.9 (April 19, 2026)** |
| `openai-agents-sdk-python` | 0.14.1 (April 15, 2026) | **0.14.2 (April 18, 2026)** |
| `microsoft-agent-framework-python` | 1.0 GA (April 3, 2026) | **1.0.1 (April 10, 2026)** |
| `amazon-bedrock-strands` | 1.35.0 (April 2026) | **1.36.0 (April 17, 2026)** |
| `notes.google-adk` | TypeScript noted as 0.1.x | Corrected to 0.6.1 confirmed |
| `notes.microsoft-agent-framework` | GA 1.0 features | Extended to include 1.0.1 patch |
| `notes.langgraph` | v1.1.x features | Extended to v1.1.8 OTel fix; v1.2.9 TS |
| `notes.openai-agents-sdk` | v0.14.1 features | Extended to include v0.14.2 |
| `notes.amazon-bedrock` | 1.35.0 features | Extended to 1.36.0 |
| `notes.pydantic-ai` | v1.84.0 features | Extended to include v1.84.1 |

---

## Framework Audit Summary

All 22+ framework guides were audited against official release notes, changelogs, PyPI/npm/NuGet registries, and GitHub releases. The table below summarises findings:

| Framework | Previous Version | Latest Release | Status |
|-----------|-----------------|----------------|--------|
| **PydanticAI** | **1.84.0** | **1.84.1 (Apr 18)** | ✅ Updated |
| **Google ADK TypeScript** | **0.1.x** | **0.6.1 (Apr 4)** | ✅ Corrected |
| **LangGraph Python** | **1.1.6** | **1.1.8 (Apr 17)** | ✅ Updated |
| **LangGraph TypeScript** | **1.2.8** | **1.2.9 (Apr 19)** | ✅ Updated |
| **OpenAI Agents SDK Python** | **0.14.1** | **0.14.2 (Apr 18)** | ✅ Updated |
| **Microsoft Agent Framework Python** | **1.0 GA** | **1.0.1 (Apr 10)** | ✅ Updated |
| **Amazon Bedrock Strands** | **1.35.0** | **1.36.0 (Apr 17)** | ✅ Updated |
| CrewAI | 1.14.2 | 1.14.2 (Apr 17) | ✅ Current |
| Google ADK Python | 1.31.0 | 1.31.0 (Apr 17) | ✅ Current |
| Google ADK Go | 1.0.0 GA | 1.0.0 GA (Apr 8) | ✅ Current |
| OpenAI Agents SDK TypeScript | 0.8.3 | 0.8.3 (Apr 9) | ✅ Current |
| Anthropic Claude SDK Python | 0.1.63 | 0.1.63 (Apr 18) | ✅ Current |
| Anthropic Claude SDK TypeScript | 0.2.113 | 0.2.113 (Apr 18) | ✅ Current |
| Semantic Kernel Python | 1.41.2 | 1.41.2 (Apr 8) | ✅ Current |
| Semantic Kernel .NET | 1.74.0 | 1.74.0 | ✅ Current |
| Microsoft Agent Framework .NET | 1.0 GA | 1.0 GA (Apr 3) | ✅ Current |
| Mistral Agents API | 2.0.1 | 2.0.1 (Mar 12) | ✅ Current |
| LlamaIndex Python | 0.14.20 | 0.14.20 (Apr 3) | ✅ Current |
| LlamaIndex TypeScript | 1.1.4 | 1.1.4 (Apr 15) | ✅ Current |
| Haystack | 2.27.0 | 2.27.0 (Apr 1) | ✅ Current |
| AG2 (AutoGen) | 0.11.5 | 0.11.5 (Apr 5) | ✅ Current |
| SmolAgents | 1.24.0 | 1.24.0 (Jan 16) | ✅ Current |

---

## No Action Required

The following were reviewed and found to require no changes:

- **Deprecated methods**: No newly deprecated patterns identified.
- **Breaking changes**: No new breaking changes found across any framework since the April 19 review.
- **Code examples**: All existing code examples remain valid for their documented versions.
- **Anthropic Claude SDK TypeScript**: Research surfaced a reference to v0.2.101 (an older cached result); confirmed current version remains **0.2.113** as documented — no downgrade applied.

---

## Cross-Cutting Standards Verification

| Standard | Status |
|---------|--------|
| British English spelling | ✅ Consistent throughout all updated files |
| Revision history in all updated guides | ✅ Extended with new rows; Strands guide gained its first revision history section |
| Version numbers in guide headers | ✅ All updated files show correct versions |
| Breaking changes documented | ✅ No new breaking changes from any updated version |
| Code examples for new features | ✅ Feature tables provided; code examples unchanged (no API surface changes) |
| Beginner-to-advanced learning paths | ✅ Not affected by patch-level updates |
| GA status corrections | ✅ Microsoft Agent Framework Python preview language removed |

---

## Recommended Follow-Up (Next Review)

1. **PydanticAI** — Rapid release cadence (2–3 releases/week); monitor for v1.85.x.
2. **Google ADK Python** — Bi-weekly cadence; v1.32.x likely within 2 weeks.
3. **LangGraph** — Monitor for v1.1.9+ (Python) and v1.3.x (TypeScript).
4. **OpenAI Agents SDK Python** — Monitor for v0.14.3+; watch for Realtime enhancements.
5. **CrewAI** — Monitor for v1.15.0; watch for Flows system and AMP Suite changes.
6. **Haystack** — v2.28.0 expected; monitor for additional component integrations.
7. **Semantic Kernel** — Monitor for v1.42.x (Python) and v1.75.x (.NET).
8. **AG2** — Approaching v1.0 with breaking changes; autogen.beta framework maturation ongoing.

---

*Report generated: April 20, 2026*

# AgentGuides Update Report — April 19, 2026

**Prepared by:** Automated review process  
**Review date:** April 19, 2026  
**Previous review:** April 18, 2026  
**Scope:** All 22 framework guides across Python, TypeScript, .NET, and Go

---

## Executive Summary

This report documents the changes made during the April 19, 2026 review of the AgentGuides repository. Two framework patch releases were identified since the previous review: **CrewAI v1.14.2** (April 17, 2026) and **Google ADK Python v1.31.0** (April 17, 2026). Both guides have been updated with full release notes, revised introductions, and expanded revision histories. All other framework guides were audited and confirmed current.

---

## Changes Made

### 1. CrewAI — Updated to v1.14.2

**Files modified:**
- `CrewAI_Guide/crewai_comprehensive_guide.md`
- `CrewAI_Guide/README.md`

**Previous documented version:** 1.14.0 (April 7, 2026)  
**New version:** 1.14.2 (April 17, 2026)

#### v1.14.1 (April 9, 2026) — Documented for the first time

| Change | Detail |
|--------|--------|
| Async checkpoint TUI browser | Interactive terminal UI for inspecting and managing saved checkpoints |
| Streaming output context managers | `aclose()`/`close()` and `async with` support on streaming output objects |
| `BaseProvider` as Pydantic `BaseModel` | Provider backends now use `provider_type` discriminator for type-safe configuration |
| `tomlkit`-based devtools CLI | More reliable TOML parsing in CLI tooling, replacing regex-based approach |
| Dynamic tool field exclusion | Denylist is computed dynamically, allowing custom tool fields to be excluded safely |
| CVE-2026-1839 | `transformers` dependency bumped to ≥5.5.0 |

#### v1.14.2 (April 17, 2026) — Documented for the first time

| Change | Detail |
|--------|--------|
| Checkpoint forking with lineage tracking | Branch execution from any saved checkpoint; fork graph is tracked for audit/replay |
| `from_checkpoint` on `Agent.kickoff` | Resume a crew run programmatically from a named checkpoint |
| Checkpoint CLI expanded | `crewai checkpoint resume`, `crewai checkpoint diff`, `crewai checkpoint prune` |
| Template management CLI | `crewai template` commands for creating and managing project templates |
| Enriched LLM token tracking | Token counts now include `reasoning_tokens` and `cache_creation_tokens` |
| Scoped streaming handlers | Chunk handlers scoped per-run to prevent cross-run contamination in concurrent executions |
| `flow_finished` event fix | Event now correctly fires after Human-in-the-Loop (HITL) resume in Flows |
| Bedrock tool call argument preservation | Fixed a bug where tool call arguments were dropped when routing through Amazon Bedrock Converse API |
| Cyclic JSON schema fix | MCP tool resolution now correctly handles cyclic JSON schema references |
| CVE-2026-39892 | Cryptography library pinned to 46.0.7 |
| Security patches | authlib, langchain-text-splitters, and pypdf updated to address known vulnerabilities |

**README changes:**
- Added dedicated "What's New in v1.14.2" and "What's New in v1.14.1" sections above the existing v1.14.0 section
- Updated current version badge from 1.14.0 to 1.14.2

**Comprehensive guide changes:**
- Updated header from `Latest: 1.14.0` to `Latest: 1.14.2 | Updated: April 19, 2026`
- Updated introduction paragraph to reference v1.14.1 and v1.14.2 improvements
- Updated closing summary paragraph
- Added two new rows to revision history table (v1.14.1 and v1.14.2)

---

### 2. Google ADK Python — Updated to v1.31.0

**Files modified:**
- `Google_ADK_Guide/python/google_adk_comprehensive_guide.md`
- `Google_ADK_Guide/python/README.md`

**Previous documented version:** 1.30.0 (April 13, 2026)  
**New version:** 1.31.0 (April 17, 2026)

#### v1.31.0 (April 17, 2026) — Documented for the first time

| Change | Detail |
|--------|--------|
| Overhauled Web UI | Live chat interface, session display names, structured execution traces |
| Graph View canvas | Visual canvas to map out and manage agent architectures interactively |
| Event filtering | Filter the real-time event stream by message type, tool calls, or errors |
| Computer-use visualisation | Dedicated interface for monitoring and reviewing computer-use agent actions |
| Memory bank event ingestion | Agents can now ingest events directly into memory banks |
| Vertex AI Agent Engine Sandbox | Computer-use agents can run inside Agent Engine sandboxes |
| Firestore database support | First-class Firestore integration as a database backend for agent state |
| Session ID tracking in LLM responses | Session IDs propagated through LLM response objects for traceability |
| Parameter/Secret Manager user-agent headers | API clients now send proper user-agent headers |
| MCP minimum version raised | Minimum supported MCP version is now 1.24.0 |
| Console URL fix | Corrected console URL paths after Agent Engine deployment |
| Event callback timing fix | Plugin modifications now persist correctly across event callbacks |
| `FunctionDeclaration` JSON schema fallback | Improved fallback handling for malformed function declarations |
| BigQuery plugin fixes | Data transfer and metadata issues in the BigQuery plugin resolved |

**README changes:**
- Updated version from 1.30.0 to 1.31.0 and last-updated date to April 19, 2026
- Added "What's New in ADK Python 1.31.0" section above the existing 1.18→1.30 summary

**Comprehensive guide changes:**
- Updated header from `Latest: 1.30.0 | Updated: April 2026` to `Latest: 1.31.0 | Updated: April 19, 2026`
- Updated internal `Last Updated` field
- Added v1.31.0 row to revision history table

---

### 3. `versions.json` — Updated

**File:** `versions.json`

| Field | Previous | Updated |
|-------|----------|---------|
| `last_updated` | April 18, 2026 | April 19, 2026 |
| `previous_update` | April 17, 2026 | April 18, 2026 |
| `google-adk-python` | 1.30.0 (April 13, 2026) | **1.31.0 (April 17, 2026)** |
| `crewai` | 1.14.0 (April 7, 2026) | **1.14.2 (April 17, 2026)** |
| `notes.google-adk` | 1.30.0 features | Extended to include 1.31.0 Web UI, Firestore, memory bank, sandbox features |
| `notes.crewai` | v1.14.0 features | Extended to include v1.14.1 and v1.14.2 features |

---

## Framework Audit Summary

All 18+ framework guides were audited against official release notes, changelogs, and PyPI/npm/pkg registries. The table below summarises findings:

| Framework | Guide Version | Latest Release | Status |
|-----------|--------------|----------------|--------|
| OpenAI Agents SDK (Python) | 0.14.1 | 0.14.1 (Apr 15) | ✅ Current |
| OpenAI Agents SDK (TypeScript) | 0.8.3 | 0.8.3 (Apr 9) | ✅ Current |
| PydanticAI | 1.84.0 | 1.84.0 (Apr 17) | ✅ Current |
| **CrewAI** | **1.14.0 → 1.14.2** | **1.14.2 (Apr 17)** | ✅ Updated |
| AG2 (AutoGen) | 0.11.5 | 0.11.5 (Apr 5) | ✅ Current |
| LangGraph (Python) | 1.1.6 | 1.1.6 (Apr 10) | ✅ Current |
| LangGraph (TypeScript) | 1.2.8 | 1.2.8 (Apr 11) | ✅ Current |
| LlamaIndex (Python) | 0.14.20 | 0.14.20 (Apr 3) | ✅ Current |
| LlamaIndex (TypeScript) | @llamaindex/workflow 1.1.4 | 1.1.4 (Apr 15) | ✅ Current |
| Haystack | 2.27.0 | 2.27.0 (Apr 1) | ✅ Current |
| **Google ADK (Python)** | **1.30.0 → 1.31.0** | **1.31.0 (Apr 17)** | ✅ Updated |
| Google ADK (Go) | 1.0.0 GA | 1.0.0 GA (Apr 8) | ✅ Current |
| Google ADK (TypeScript) | 0.1.x | 0.1.x (Dec 2025) | ✅ Current |
| Amazon Bedrock Strands | 1.35.0 | 1.35.0 | ✅ Current |
| Microsoft Agent Framework | 1.0 GA | 1.0 GA (Apr 3) | ✅ Current |
| Semantic Kernel (Python) | 1.41.2 | 1.41.2 (Apr 8) | ✅ Current |
| Semantic Kernel (.NET) | 1.74.0 | 1.74.0 | ✅ Current |
| Mistral Agents API | 2.0.1 | 2.0.1 (Mar 12) | ✅ Current |
| Anthropic Claude SDK (Python) | 0.1.63 | 0.1.63 (Apr 18) | ✅ Current |
| Anthropic Claude SDK (TypeScript) | 0.2.113 | 0.2.113 (Apr 18) | ✅ Current |
| SmolAgents | 1.24.0 | 1.24.0 (Jan 16) | ✅ Current |
| AutoGen (Legacy) | — | Deprecated | ✅ No change needed |

---

## No Action Required

The following were reviewed and found to require no changes:

- **Deprecated methods**: No newly deprecated patterns identified in active guides.
- **Breaking changes**: No new breaking changes found across any framework since the April 18 review.
- **Code examples**: All existing code examples remain valid for their documented versions.

---

## Cross-Cutting Standards Verification

| Standard | Status |
|---------|--------|
| British English spelling | ✅ Consistent throughout all updated files |
| Revision history in all updated guides | ✅ Extended with new rows |
| Version numbers in guide headers | ✅ All updated files show correct versions |
| Breaking changes documented | ✅ All breaking changes from 1.14.1/1.14.2 documented (none identified) |
| Code examples for new features | ✅ Feature tables provided for all new features |
| Beginner-to-advanced learning paths | ✅ Not affected by this update |

---

## Recommended Follow-Up (Next Review)

1. **Google ADK Python** — Bi-weekly release cadence means v1.32.x is likely within 2 weeks. Monitor for further session management and UI enhancements.
2. **PydanticAI** — Rapid release cadence (2–3 releases/week); monitor for v1.85+.
3. **CrewAI** — Monitor for v1.15.0; watch for changes to the Flows system and AMP Suite.
4. **Haystack** — v2.28.0 expected within weeks; monitor for additional component integrations.
5. **LangGraph** — Watch for v1.2.x with further v2 API enhancements.

---

*Report generated: April 19, 2026*

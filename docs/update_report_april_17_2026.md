# Agent Guides Update Report — April 17, 2026

**Summary**: Patch update covering three framework releases published on April 15–16, 2026, identified through cross-referencing official PyPI pages, GitHub release pages, and changelogs against the previous guide update (April 16, 2026).

All other frameworks verified as current: LangGraph (1.1.6), CrewAI (1.14.0), SmolAgents (1.24.0), AG2 (0.11.5), LlamaIndex (0.14.20), Haystack (2.27.0), Google ADK (1.30.0), Semantic Kernel Python (1.41.2) / .NET (1.74.0), Mistral (2.0.1), Amazon Bedrock Strands (1.35.0), Microsoft Agent Framework (1.0 GA), OpenAI Agents SDK TypeScript (0.8.3), Claude Agent SDK TypeScript (0.2.110).

---

## Changes Made

### 1. `versions.json`

| Field | Before | After |
|-------|--------|-------|
| `last_updated` | April 16, 2026 | April 17, 2026 |
| `previous_update` | November 21, 2025 | April 16, 2026 |
| `openai-agents-sdk-python` | `0.14.x (April 2026)` | `0.14.1 (April 15, 2026)` |
| `anthropic-claude-agent-sdk-python` | `0.1.59 (April 13, 2026)` | `0.1.60 (April 16, 2026)` |
| `pydantic-ai` | `1.83.0 (April 16, 2026)` | `1.84.0 (April 17, 2026)` |
| notes: `openai-agents-sdk` | Generic description | Clarified Harness API and v0.14.1 specifics |
| notes: `anthropic-claude-sdk` | References 0.1.59 | Updated to 0.1.60 |
| notes: `pydantic-ai` | References 1.83.0 | Updated; lists 1.84.0 new features |

---

### 2. OpenAI Agents SDK (Python) — `OpenAI_Agents_SDK_Guides/README.md`

**Version pinned**: `0.14.x` → `0.14.1` (April 15, 2026)

**What changed in 0.14.1**:
- Patch release on top of 0.14.0; no breaking changes
- Harness API for Sandbox Agents confirmed Python-only; TypeScript support listed as upcoming
- Minor stability fixes in the sandbox execution layer

**Revision history entry added**:
```
| April 17, 2026 | 0.14.1 | Pinned to v0.14.1 (April 15, 2026); Harness API for Sandbox Agents available
                            in Python only (TypeScript support pending); no breaking changes from 0.14.0 |
```

---

### 3. Anthropic Claude Agent SDK (Python) — `Anthropic_Claude_Agent_SDK_Guide/README.md`

**Version updated**: `0.1.59` → `0.1.60` (April 16, 2026)

**What changed in 0.1.60**:
- Patch release; no breaking changes from 0.1.59
- Stability and compatibility improvements
- Upgrade: `pip install --upgrade claude-agent-sdk`

**Revision history entry added**:
```
| April 17, 2026 | 0.1.60 | Patch release — stability and compatibility improvements; no breaking changes
                             from v0.1.59 |
```

---

### 4. PydanticAI — `PydanticAI_Guide/README.md`

**Version updated**: `1.83.0` → `1.84.0` (April 17, 2026)

**New features in 1.84.0**:

| Feature | Description |
|---------|-------------|
| `XSearchTool` / `FileSearch` for xAI | Built-in search and file retrieval tools for the xAI (Grok) provider |
| `FastMCPToolset` metadata injection | Per-tool-call metadata injection for richer tracing and auditing |
| Bedrock prompt cache TTL | Configurable cache time-to-live for AWS Bedrock provider responses |
| Claude Opus 4.7 support | `anthropic:claude-opus-4-7` now a recognised model string |
| `OllamaModel` subclass | Dedicated class replacing generic `OpenAIModel`; fixes structured output on Ollama Cloud |
| Stateful `OpenAICompaction` | Reduces token usage in long conversations by compacting history while preserving state |
| Google `FileSearchTool` fix | Resolves a regex parsing bug in Google Vertex AI file search responses |

**Code examples added** for `OllamaModel` and `FastMCPToolset` metadata injection.

**Revision history entry added**:
```
| April 17, 2026 | 1.84.0 | Updated to v1.84.0; XSearchTool/FileSearch for xAI; FastMCPToolset metadata
                             injection; Bedrock prompt cache TTL; Claude Opus 4.7; OllamaModel subclass;
                             stateful OpenAICompaction; Google FileSearchTool regex fix |
```

---

## Verification Summary

All 16 frameworks were checked against official release channels:

| Framework | Source Checked | Status |
|-----------|---------------|--------|
| OpenAI Agents SDK (Python) | GitHub releases, PyPI, TechCrunch coverage | ✅ Updated to 0.14.1 |
| OpenAI Agents SDK (TypeScript) | npm, GitHub releases | ✅ Current at 0.8.3 |
| PydanticAI | PyPI, GitHub releases | ✅ Updated to 1.84.0 |
| Claude Agent SDK (Python) | PyPI, GitHub releases, Anthropic platform | ✅ Updated to 0.1.60 |
| Claude Agent SDK (TypeScript) | npm, GitHub releases | ✅ Current at 0.2.110 |
| CrewAI | PyPI, GitHub releases | ✅ Current at 1.14.0 |
| LangGraph (Python/TypeScript) | GitHub releases, LangChain changelog | ✅ Current at 1.1.6 / 1.2.8 |
| SmolAgents | PyPI, Hugging Face | ✅ Current at 1.24.0 |
| AG2 / AutoGen | GitHub releases, PyPI | ✅ Current at 0.11.5 |
| LlamaIndex (Python) | PyPI, GitHub | ✅ Current at 0.14.20 |
| Haystack | PyPI, GitHub releases | ✅ Current at 2.27.0 |
| Google ADK (Python) | PyPI, Google Developers Blog | ✅ Current at 1.30.0 |
| Semantic Kernel (.NET / Python) | NuGet, PyPI, GitHub | ✅ Current at 1.74.0 / 1.41.2 |
| Mistral | PyPI, GitHub | ✅ Current at 2.0.1 |
| Amazon Bedrock Strands | AWS docs, GitHub | ✅ Current at 1.35.0 |
| Microsoft Agent Framework | GitHub, VS Magazine | ✅ Current at 1.0 GA |

---

## No Changes Required

The following guides were reviewed and confirmed up-to-date as of April 17, 2026:

- `CrewAI_Guide/` — v1.14.0, no new stable release
- `LangGraph_Guide/` — v1.1.6 (Python), v1.2.8 (TypeScript), no new stable release
- `SmolAgents_Guide/` — v1.24.0, no new stable release (January 2026)
- `AG2_Guide/` — v0.11.5, autogen.beta ongoing but no new stable tag
- `LlamaIndex_Guide/` — v0.14.20, no new stable release
- `Haystack_Guide/` — v2.27.0, no new stable release
- `Google_ADK_Guide/` — v1.30.0, no new stable release since April 13
- `Semantic_Kernel_Guide/` — v1.41.2 (Python) / v1.74.0 (.NET), no new release
- `Mistral_Agents_API_Guide/` — v2.0.1, no new stable release
- `Amazon_Bedrock_Agents_Guide/` — Strands v1.35.0, no new stable release
- `Microsoft_Agent_Framework_Guide/` — v1.0 GA, no new release
- `Anthropic_Claude_Agent_SDK_TypeScript_Guide/` — v0.2.110, no new release
- `OpenAI_Agents_SDK_TypeScript_Guide/` — v0.8.3, no new stable release

---

*Report generated: April 17, 2026*
*Previous report: [update_report_april_2026.md](update_report_april_2026.md)*

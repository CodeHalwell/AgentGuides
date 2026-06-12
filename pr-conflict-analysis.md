# Open PR Conflict Analysis — 2026-06-12

## Overview

4 open pull requests found in `CodeHalwell/AgentGuides`. All 4 are authored by **CodeHalwell**, all target `main`, and all share the same base commit (`3c391edb`). **No merge conflicts exist** — neither against `main` nor against each other.

---

## PR Inventory

| # | PR | Title | Author | State | Files Changed | +Adds / −Dels |
|---|-----|-------|--------|-------|---------------|---------------|
| 1 | [#180](https://github.com/CodeHalwell/AgentGuides/pull/180) | `docs(microsoft-agent-framework): Class Deep Dives Vol. 10 + upgrade to 1.8.1` | CodeHalwell | Open | 2 | +2016 / −3 |
| 2 | [#179](https://github.com/CodeHalwell/AgentGuides/pull/179) | `feat(pydantic-ai): Vol. 13 class deep dives + 4 new recipes (pydantic-ai 1.107.0)` | CodeHalwell | Open | 4 | +2736 / −5 |
| 3 | [#178](https://github.com/CodeHalwell/AgentGuides/pull/178) | `Google ADK: Class deep-dives Vol. 15 + 4 new recipes (source-verified, google-adk 2.2.0)` | CodeHalwell | Open | 2 | +2198 / −2 |
| 4 | [#177](https://github.com/CodeHalwell/AgentGuides/pull/177) | `LangGraph: Class deep-dives Vol. 13 + 4 new recipes (source-verified, langgraph 1.2.4)` | CodeHalwell | Open | 2 | +1863 / −0 |

---

## File Footprints

### PR #180 — Microsoft Agent Framework Vol. 10
| File | Status |
|------|--------|
| `src/content/docs/microsoft-agent-framework-guide/python/index.mdx` | modified |
| `src/content/docs/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v10.md` | added |

### PR #179 — PydanticAI Vol. 13
| File | Status |
|------|--------|
| `src/content/docs/frameworks.md` | modified |
| `src/content/docs/pydanticai-guide/index.mdx` | modified |
| `src/content/docs/pydanticai-guide/pydantic_ai_class_deep_dives_v13.md` | added |
| `src/content/docs/pydanticai-guide/pydantic_ai_recipes.md` | modified |

### PR #178 — Google ADK Vol. 15
| File | Status |
|------|--------|
| `src/content/docs/google-adk-guide/python/google_adk_class_deep_dives_v15.md` | added |
| `src/content/docs/google-adk-guide/python/google_adk_recipes.md` | modified |

### PR #177 — LangGraph Vol. 13
| File | Status |
|------|--------|
| `src/content/docs/langgraph-guide/python/langgraph_class_deep_dives_v13.md` | added |
| `src/content/docs/langgraph-guide/python/langgraph_recipes.md` | modified |

---

## Conflict Analysis

### vs. `main`

GitHub reports `mergeable_state: "clean"` for all 4 PRs. None has conflicts against the current `main` HEAD.

### Cross-PR Overlap

Each PR operates in a completely separate content subdirectory. There is **zero file overlap** across all 4 PRs:

| File | PR #177 | PR #178 | PR #179 | PR #180 |
|------|:-------:|:-------:|:-------:|:-------:|
| `microsoft-agent-framework-guide/…` | | | | ✓ |
| `pydanticai-guide/…` + `frameworks.md` | | | ✓ | |
| `google-adk-guide/…` | | ✓ | | |
| `langgraph-guide/…` | ✓ | | | |

All 4 PRs can be merged in any order without producing a conflict.

---

## Recommendations

1. **Merge order is flexible** — no ordering constraint exists due to the zero file overlap.
2. **Suggested merge sequence** (oldest first): #177 → #178 → #179 → #180.
3. **After each merge**, the remaining open PRs will automatically update their `mergeable_state` against the new `main`, but they will remain clean.
4. **No action required** on any branch before merging — no rebases or manual conflict resolution needed.

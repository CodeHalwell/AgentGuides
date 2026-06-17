# Open PR Conflict Analysis — 2026-06-17

**Repository:** CodeHalwell/AgentGuides  
**Analysed:** 4 open pull requests  
**Base branch:** `main` (SHA `0f74ab7bba64af7e6306aec6fe0a711fe0960d5b`)  
**Result: No merge conflicts detected.**

---

## Open Pull Requests

| # | Title | Author | Created | Files | +/- | Mergeable |
|---|-------|--------|---------|-------|-----|-----------|
| [#200](https://github.com/CodeHalwell/AgentGuides/pull/200) | docs(agent-framework/python): Class Deep Dives Vol. 15 — AG-UI, ChatKit, DevServer, GAIA, CopilotStudio, Azure Search, Cosmos, Durable, Functions | CodeHalwell | 2026-06-17 09:48 UTC | 2 | +1828 / -1 | ✅ Clean |
| [#199](https://github.com/CodeHalwell/AgentGuides/pull/199) | PydanticAI: Class Deep Dives Vol. 18 — new providers, Google ecosystem & graph persistence (1.107.0) | CodeHalwell | 2026-06-17 07:43 UTC | 2 | +1247 / 0 | ✅ Clean |
| [#198](https://github.com/CodeHalwell/AgentGuides/pull/198) | Google ADK: Class Deep Dives Vol. 20 — compaction internals, HITL utilities, workflow composition & backend detection (2.2.0) | CodeHalwell | 2026-06-17 07:11 UTC | 3 | +1591 / -2 | ✅ Clean |
| [#197](https://github.com/CodeHalwell/AgentGuides/pull/197) | LangGraph: Class Deep Dives Vol. 18 — channels, caching, functional patterns & debug streaming (1.2.5) | CodeHalwell | 2026-06-17 06:45 UTC | 2 | +1503 / 0 | ✅ Clean |

---

## File Inventory by PR

Each PR adds one new deep-dive document and modifies the `index.mdx` of its own guide section. There is **no file overlap** across any of the four PRs.

### PR #200 — Microsoft Agent Framework (Vol. 15)
| File | Status |
|------|--------|
| `src/content/docs/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v15.md` | Added |
| `src/content/docs/microsoft-agent-framework-guide/python/index.mdx` | Modified |

### PR #199 — PydanticAI (Vol. 18)
| File | Status |
|------|--------|
| `src/content/docs/pydanticai-guide/pydantic_ai_class_deep_dives_v18.md` | Added |
| `src/content/docs/pydanticai-guide/index.mdx` | Modified |

### PR #198 — Google ADK (Vol. 20)
| File | Status |
|------|--------|
| `src/content/docs/google-adk-guide/python/google_adk_class_deep_dives_v20.md` | Added |
| `src/content/docs/google-adk-guide/python/index.mdx` | Modified |

### PR #197 — LangGraph (Vol. 18)
| File | Status |
|------|--------|
| `src/content/docs/langgraph-guide/python/langgraph_class_deep_dives_v18.md` | Added |
| `src/content/docs/langgraph-guide/python/index.mdx` | Modified |

---

## Conflict Analysis

### Against base (`main`)
All four branches share the same base commit and GitHub reports `mergeable_state: clean` for each. No PR modifies any file that `main` has changed since the branches were cut.

### Cross-PR (PR vs PR)
Every PR operates in an isolated subdirectory:

| PR | Guide subdirectory |
|----|--------------------|
| #200 | `microsoft-agent-framework-guide/python/` |
| #199 | `pydanticai-guide/` |
| #198 | `google-adk-guide/python/` |
| #197 | `langgraph-guide/python/` |

There are **no shared files** between any pair of open PRs. Merging them in any order will not produce conflicts.

### Potential ordering consideration
All four PRs modify their respective `index.mdx` files. These files are in separate directories and will not conflict during merge. However, if a global index or navigation manifest exists outside these directories and is shared, it is not touched by any of these PRs, so no issue arises.

---

## Summary

**No action required on conflicts.** All 4 PRs are conflict-free against `main` and against each other. They can be merged in any order.

Recommended merge order (oldest first, to keep the commit timeline clean):

1. #197 — LangGraph Vol. 18
2. #198 — Google ADK Vol. 20
3. #199 — PydanticAI Vol. 18
4. #200 — Agent Framework Vol. 15

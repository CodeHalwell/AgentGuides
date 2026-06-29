# Open PR Conflict Analysis — AgentGuides
**Generated:** 2026-06-29  
**Analysed by:** Claude Code (automated routine)

---

## Open PRs at a glance

| # | Title | Author | Base branch | Mergeable state | Files changed |
|---|-------|--------|-------------|-----------------|---------------|
| [#249](https://github.com/CodeHalwell/AgentGuides/pull/249) | MS AF Class Deep Dives Vol. 27 (security, OTel, compaction) | CodeHalwell | `claude/relaxed-clarke-49v8qy` (PR #245's branch) | clean | 2 |
| [#248](https://github.com/CodeHalwell/AgentGuides/pull/248) | PydanticAI Class Deep Dives Vol. 29 (pydantic-ai 2.0.0) | CodeHalwell | `claude/loving-johnson-ettv80` (PR #244's branch) | clean | 2 |
| [#247](https://github.com/CodeHalwell/AgentGuides/pull/247) | Google ADK Vol. 30 + Vol. 31 + guide enhancements | CodeHalwell | `main` | clean | 7 |
| [#246](https://github.com/CodeHalwell/AgentGuides/pull/246) | LangGraph Class Deep Dives Vol. 29 + v1.2.6 guide update | CodeHalwell | `claude/gracious-tesla-88zmzs` (PR #242's branch) | clean | 2 |
| [#245](https://github.com/CodeHalwell/AgentGuides/pull/245) | MS AF Class Deep Dives Vol. 26 (durabletask) | CodeHalwell | `main` | clean | 2 |
| [#244](https://github.com/CodeHalwell/AgentGuides/pull/244) | PydanticAI Class Deep Dives Vol. 28 (pydantic-ai 2.0.0) | CodeHalwell | `main` | clean | 2 |
| [#243](https://github.com/CodeHalwell/AgentGuides/pull/243) | Google ADK Class Deep Dives Vol. 30 (google-adk 2.3.0) | CodeHalwell | `main` | clean | 3 |
| [#242](https://github.com/CodeHalwell/AgentGuides/pull/242) | LangGraph Class Deep Dives Vol. 28 (config internals, type vars) | CodeHalwell | `main` | clean | 1 |

All 8 PRs currently report `mergeable_state: clean` against their own base branches.  
**However**, two categories of conflict exist: a direct content duplication and a latent sequential-merge risk.

---

## PR structure — stacked chains and direct-to-main

```
main
├── PR #242  LangGraph Vol. 28
│   └── PR #246  LangGraph Vol. 29  (stacked on #242)
├── PR #243  Google ADK Vol. 30                      ← CONFLICTS with #247 (see §1)
├── PR #244  PydanticAI Vol. 28
│   └── PR #248  PydanticAI Vol. 29  (stacked on #244)
├── PR #245  MS Agent Framework Vol. 26
│   └── PR #249  MS Agent Framework Vol. 27  (stacked on #245)
└── PR #247  Google ADK Vol. 30 + Vol. 31 + enhancements  ← CONFLICTS with #243
```

---

## Conflict findings

### 1 · CRITICAL — Direct content duplication: PR #243 vs PR #247

**Both PRs target `main` and add/modify the same files:**

| File | PR #243 | PR #247 |
|------|---------|---------|
| `google-adk-guide/python/google_adk_class_deep_dives_v30.md` | Added (1 259 lines, Vol. 30, 10 classes) | Added (different version, same 10 classes) |
| `google-adk-guide/python/index.mdx` | Modified | Modified |
| `google-adk-guide/python/tools.md` | Modified | Modified |

**Nature of the conflict:**  
Both PRs independently document the same 10 Google ADK classes (GCPSkillRegistry, ApiRegistry, AgentRegistry, EnterpriseWebSearchTool, MultiTurnTaskSuccessV1Evaluator, session DB migration runner, Interactions API generator, _BasicLlmRequestProcessor, SandboxClient, RubricBasedFinalResponseQualityV1Evaluator). They are overlapping edits — not a deletion vs modification — because neither PR was branched from the other.

PR #247 is a strict superset: it contains everything in #243 (Vol. 30) **plus** Vol. 31 and five guide enhancements to `memory-and-artifacts.md`, `mcp-and-a2a.md`, `google_adk_comprehensive_guide.md`, and `tools.md`.

**Consequence if left unresolved:**  
Whichever PR merges second will fail with a conflict on `google_adk_class_deep_dives_v30.md` (adding a file that already exists with divergent content) and on `index.mdx` and `tools.md` (overlapping edits).

**Recommended resolution:**  
Close PR #243. Merge PR #247 instead — it is the complete, up-to-date version. If only Vol. 30 is wanted at this time, close PR #247 and merge PR #243, then rebase #247 on top.

---

### 2 · MEDIUM — Latent sequential-merge conflicts on `index.mdx`

The following PRs all modify `index.mdx` files, but they are in **separate directories** — so they do not currently conflict with each other against `main`. They would only conflict if they modified the same root-level index file, which they do not.

| PR | index.mdx path |
|----|---------------|
| #243 | `google-adk-guide/python/index.mdx` |
| #244 | `pydanticai-guide/index.mdx` |
| #245 | `microsoft-agent-framework-guide/python/index.mdx` |
| #247 | `google-adk-guide/python/index.mdx` |

The only true sequential-merge conflict here is again between **#243 and #247** (same path, covered under §1 above).

PRs #244 and #245 each modify their own framework's `index.mdx` exclusively — no cross-PR conflict.

---

### 3 · LOW — Stacked PR dependency risk (PRs #246, #248, #249)

These three PRs are intentionally stacked: they branch from a peer PR's head branch rather than `main`.

| Stacked PR | Depends on | Risk |
|------------|-----------|------|
| #246 (LangGraph Vol. 29) | #242 (LangGraph Vol. 28) | If #242 is squash-merged, #246's base history is rewritten — GitHub may show it as conflicted or "out of date" |
| #248 (PydanticAI Vol. 29) | #244 (PydanticAI Vol. 28) | Same risk on squash-merge of #244 |
| #249 (MS AF Vol. 27) | #245 (MS AF Vol. 26) | Same risk on squash-merge of #245 |

**Currently clean.** The risk only materialises if the parent PR is merged with squash (GitHub squash-merge creates a new commit not in the stacked PR's ancestry, leaving orphaned commits). To avoid this, merge the parent with a regular merge commit, then the stacked PR's base branch fast-forwards and remains clean.

---

## Files-changed matrix (PRs targeting `main`)

| File | #242 | #243 | #244 | #245 | #247 |
|------|------|------|------|------|------|
| `langgraph-guide/python/langgraph_class_deep_dives_v28.md` | ✚ added | | | | |
| `google-adk-guide/python/google_adk_class_deep_dives_v30.md` | | ✚ added | | | ✚ added ⚠️ |
| `google-adk-guide/python/google_adk_class_deep_dives_v31.md` | | | | | ✚ added |
| `google-adk-guide/python/google_adk_comprehensive_guide.md` | | | | | ✎ modified |
| `google-adk-guide/python/index.mdx` | | ✎ modified | | | ✎ modified ⚠️ |
| `google-adk-guide/python/mcp-and-a2a.md` | | | | | ✎ modified |
| `google-adk-guide/python/memory-and-artifacts.md` | | | | | ✎ modified |
| `google-adk-guide/python/tools.md` | | ✎ modified | | | ✎ modified ⚠️ |
| `pydanticai-guide/index.mdx` | | | ✎ modified | | |
| `pydanticai-guide/pydantic_ai_class_deep_dives_v28.md` | | | ✚ added | | |
| `microsoft-agent-framework-guide/python/index.mdx` | | | | ✎ modified | |
| `microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v26.md` | | | | ✚ added | |

⚠️ = same file touched by two PRs targeting `main` → merge conflict if both are merged

---

## Recommended merge order

1. **Close PR #243** (superseded by #247 which is a strict superset)
2. Merge PRs #242, #244, #245, #247 to `main` in any order (no overlapping files after #243 is closed)
3. After each parent merges, update the corresponding stacked PR:
   - After #242 merges → rebase/update #246
   - After #244 merges → rebase/update #248
   - After #245 merges → rebase/update #249
4. Merge #246, #248, #249 once rebased on the updated `main`

> **Note on merge strategy:** Use regular merge commits (not squash) when merging parent PRs if you want the stacked PRs to remain automatically clean. If squash is preferred, manually rebase the stacked PR onto `main` after the parent merges.

# Open PR Conflict Analysis — AgentGuides

> Generated: 2026-06-26

---

## Summary

**12 open pull requests** across 4 framework tracks. **No git-level merge conflicts** — every PR reports `mergeable_state: clean` against its base branch. However, several PRs across the LangGraph and Google ADK tracks have an ordering dependency that must be respected at merge time to avoid duplicate commits and content duplication in `index.mdx`.

---

## Open PRs by Track

### LangGraph (`langgraph==1.2.6`)

| PR | Title | Author | Head → Base | State |
|----|-------|--------|-------------|-------|
| [#226](https://github.com/CodeHalwell/AgentGuides/pull/226) | Class Deep-dives Vol. 24 | CodeHalwell | `gracious-tesla-viso96` → `main` | open |
| [#230](https://github.com/CodeHalwell/AgentGuides/pull/230) | Class Deep-dives Vol. 25 | CodeHalwell | `gracious-tesla-hpi70t` → `main` | open |
| [#234](https://github.com/CodeHalwell/AgentGuides/pull/234) | Class Deep-dives Vol. 26 | CodeHalwell | `gracious-tesla-rh031z` → `main` | open |

**Chain dependency:** All three target `main` but #230 was branched from #226, and #234 was branched from #230. Each PR cumulatively contains the prior volumes' commits. #234 (Vol. 26) contains all content from Vols. 24, 25, and 26.

### Google ADK (`google-adk==2.3.0`)

| PR | Title | Author | Head → Base | State |
|----|-------|--------|-------------|-------|
| [#227](https://github.com/CodeHalwell/AgentGuides/pull/227) | Class Deep-dives Vol. 26 | CodeHalwell | `gracious-clarke-a3jwn1` → `main` | open |
| [#231](https://github.com/CodeHalwell/AgentGuides/pull/231) | Class Deep-dives Vol. 27 | CodeHalwell | `gracious-clarke-e02zob` → `main` | open |
| [#235](https://github.com/CodeHalwell/AgentGuides/pull/235) | Class Deep-dives Vol. 28 | CodeHalwell | `gracious-clarke-6d6gzo` → `gracious-clarke-e02zob` | open |

**Chain dependency:** #231 was branched from #227 (Vol. 26 content is already inside Vol. 27's branch). #235 targets #231's head branch explicitly — it cannot be merged to `main` until #231 is merged first.

### PydanticAI (`pydantic-ai v2.0.0`)

| PR | Title | Author | Head → Base | State |
|----|-------|--------|-------------|-------|
| [#228](https://github.com/CodeHalwell/AgentGuides/pull/228) | Class Deep Dives Vol. 25 | CodeHalwell | `loving-johnson-qwk345` → `main` | open |
| [#232](https://github.com/CodeHalwell/AgentGuides/pull/232) | Class Deep Dives Vol. 26 | CodeHalwell | `loving-johnson-9wc6yq` → `loving-johnson-qwk345` | open |
| [#236](https://github.com/CodeHalwell/AgentGuides/pull/236) | Class Deep Dives Vol. 27 | CodeHalwell | `loving-johnson-2oa39z` → `loving-johnson-9wc6yq` | open |

**Chain dependency:** Explicitly chained — #232 targets #228's head, #236 targets #232's head. Must be merged in strict order: #228 → #232 → #236.

### Microsoft Agent Framework (`agent-framework==1.9.0`)

| PR | Title | Author | Head → Base | State |
|----|-------|--------|-------------|-------|
| [#229](https://github.com/CodeHalwell/AgentGuides/pull/229) | Class Deep Dives Vol. 22 | CodeHalwell | `relaxed-clarke-ueif8r` → `main` | open |
| [#233](https://github.com/CodeHalwell/AgentGuides/pull/233) | Class Deep Dives Vol. 23 | CodeHalwell | `relaxed-clarke-p0fr3o` → `relaxed-clarke-ueif8r` | open |
| [#237](https://github.com/CodeHalwell/AgentGuides/pull/237) | Class Deep Dives Vol. 24 | CodeHalwell | `relaxed-clarke-kfzqds` → `relaxed-clarke-p0fr3o` | open |

**Chain dependency:** Explicitly chained — #233 targets #229's head, #237 targets #233's head. Must be merged in strict order: #229 → #233 → #237.

---

## Conflict Status

| PR | mergeable_state | Conflicts? |
|----|----------------|------------|
| #226 | `clean` | None |
| #227 | `clean` | None |
| #228 | `clean` | None |
| #229 | `clean` | None |
| #230 | `clean` | None |
| #231 | `clean` | None |
| #232 | `clean` | None |
| #233 | `clean` | None |
| #234 | `clean` | None |
| #235 | `clean` | None |
| #236 | `clean` | None |
| #237 | `clean` | None |

**No conflicted PRs found.** All 12 pass GitHub's mergeable check against their respective base branches.

---

## Ordering Risk (not conflicts, but dependency violations)

Although no file-level conflicts exist today, the following pairs **will create problems** if merged out of order:

### LangGraph — #226, #230, #234 all target `main`

Because #230 (Vol. 25) was branched from #226 (Vol. 24) and contains all of #226's commits, and #234 (Vol. 26) was branched from #230 and contains all of both, **the cleanest path is one of**:

- **Option A (recommended):** Merge only **#234** — it includes Vols. 24, 25, and 26 in one shot. Close #226 and #230 as superseded.
- **Option B:** Merge in strict order (#226 → #230 → #234). After #226 merges into `main`, #230's base will auto-advance; after #230 merges, #234's base will auto-advance. No rebasing needed.

Merging #230 before #226 would duplicate #226's commits in `main`'s history and create redundant `index.mdx` edits.

### Google ADK — #227 and #231 both target `main`

#231 (Vol. 27) was branched from #227 (Vol. 26) and already contains all of #226's commits. Same situation as LangGraph:

- **Option A (recommended):** Merge only **#231** — it contains Vols. 26 and 27. Close #227 as superseded. Then merge #235 once #231 is in `main`.
- **Option B:** Merge in order #227 → #231. #231's base will auto-update after #227 lands. Then merge #235.

**#235 is blocked** on #231 regardless of which option is chosen — its base is `gracious-clarke-e02zob` (#231's head), not `main`.

### PydanticAI and Microsoft AF — already correctly chained

These chains use intermediate branches as PR bases. GitHub will block merging out of order automatically (the base branch won't exist in `main` until the predecessor is merged). No action needed beyond merging in order.

---

## Recommended Merge Order

```
# Microsoft Agent Framework
#229 → #233 → #237

# PydanticAI
#228 → #232 → #236

# Google ADK  (close #227 OR merge it first)
#231 (includes #227 content) → #235

# LangGraph  (close #226 and #230 OR merge in order)
#234 (includes #226 and #230 content)
```

---

## Files Touched (per track)

Each PR adds one new `.md` deep-dive file and updates the framework's `index.mdx`. Files do not overlap across frameworks, so there is no cross-framework conflict risk.

| Track | New file pattern | Index file |
|-------|-----------------|------------|
| LangGraph | `langgraph-guide/python/langgraph_class_deep_dives_v{N}.md` | `langgraph-guide/python/index.mdx` |
| Google ADK | `google-adk-guide/python/google_adk_class_deep_dives_v{N}.md` | `google-adk-guide/python/index.mdx` |
| PydanticAI | `pydanticai-guide/pydantic_ai_class_deep_dives_v{N}.md` | `pydanticai-guide/index.mdx` |
| Microsoft AF | `microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v{N}.md` | `microsoft-agent-framework-guide/python/index.mdx` |

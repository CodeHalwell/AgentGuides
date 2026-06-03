# PR Conflict Analysis — 2026-06-03

Generated: 2026-06-03  
Repository: [CodeHalwell/AgentGuides](https://github.com/CodeHalwell/AgentGuides)

---

## Open Pull Requests

| # | Title | Author | Base | Head | Files | State |
|---|---|---|---|---|---|---|
| [#151](https://github.com/CodeHalwell/AgentGuides/pull/151) | docs(microsoft-agent-framework): Class Deep Dives Vol. 6 — 10 source-verified class groups (agent-framework-core 1.7.0) | CodeHalwell | `claude/optimistic-hopper-KVIr5` | `claude/relaxed-clarke-i1fkL` | 2 (+1418 / -4) | Open |
| [#150](https://github.com/CodeHalwell/AgentGuides/pull/150) | Docs refresh — 2026-06-03 | CodeHalwell | `main` | `claude/optimistic-hopper-KVIr5` | 14 (+4665 / -29) | Open |
| [#149](https://github.com/CodeHalwell/AgentGuides/pull/149) | docs(pydantic-ai): Class deep dives Vol. 9 — 10 source-verified classes (pydantic-ai==1.105.0) | CodeHalwell | `main` | `claude/loving-johnson-yj8i2` | 2 (+1437 / -1) | Open |
| [#148](https://github.com/CodeHalwell/AgentGuides/pull/148) | docs(google-adk): Class deep dives Vol. 11 — 10 source-verified classes (google-adk==2.1.0) | CodeHalwell | `main` | `claude/gracious-clarke-KZ5Ie` | 2 (+1252 / -0) | Open |
| [#147](https://github.com/CodeHalwell/AgentGuides/pull/147) | docs(langgraph): Class deep-dives Vol. 8 — 10 source-verified types (langgraph==1.2.4) | CodeHalwell | `main` | `claude/gracious-tesla-3gI6A` | 2 (+1615 / -6) | Open |

---

## Merge Conflict Status

**GitHub `mergeable_state` for all 5 PRs: `clean`**

No PR has active git-level merge conflicts against its declared base branch at the time of this report.

---

## Structural / Logical Conflict Analysis

Although there are no hard git conflicts, there is a significant **dependency chain** that creates logical ordering constraints and a risk of duplicate content.

### Dependency chain

```
main
 └── #150  (Docs refresh — 2026-06-03)          [incorporates #147, #148, #149]
      └── #151  (MS Agent Framework Vol. 6)
```

### PR #150 already contains #147, #148, and #149

PR #150's description states:

> "Three open PRs (#147, #148, #149) were merged into this branch before any edits, incorporating LangGraph class deep-dives Vol. 8, Google ADK Vol. 11, and PydanticAI Vol. 9."

This means the commits from #147 (LangGraph Vol. 8), #148 (Google ADK Vol. 11), and #149 (PydanticAI Vol. 9) are **already present** in the `claude/optimistic-hopper-KVIr5` branch that backs #150.

#### Risk: merging #147, #148, or #149 after #150 would duplicate content

If #150 is merged to `main` first (which it should be, as the umbrella refresh), then attempting to merge #147, #148, or #149 afterwards would either:

- Re-apply identical file additions → likely a no-op merge or duplicate entries in `index.mdx` grids, `sidebar.order` assignments, and revision history rows.
- In the worst case, introduce stale version strings that #150 has already corrected (e.g., `frameworks.ts` version pins).

**Recommended action:** Close #147, #148, and #149 as superseded by #150, adding a closing comment that references #150.

### PR #151 depends on #150 merging first

PR #151 targets `claude/optimistic-hopper-KVIr5` (PR #150's branch) as its base, not `main`. This is intentional — it builds on top of the version-pin corrections in #150. Until #150 merges to `main`, #151 cannot be independently merged. After #150 merges, #151's base will need to be retargeted to `main` or merged via the normal cascade.

---

## Recommended Merge Order

1. **Close #147, #148, #149** — their content is superseded by #150. No merging needed.
2. **Merge #150 → `main`** — the umbrella docs refresh with all version corrections and the three deep-dive volumes already incorporated.
3. **Retarget #151 → `main`** (or merge it immediately after #150 lands, while the branch is still clean).
4. **Merge #151 → `main`** — adds Microsoft Agent Framework Vol. 6 on top of the refreshed base.

---

## Open Items Flagged in PRs

| Item | PR | Detail |
|---|---|---|
| `semantic-kernel` version mismatch | #150 | Guide declares `1.42.0`; package installs at `1.36.0`. Persists from prior reports. Requires manual investigation. |
| `sidebar.order: 25` conflict risk | #151 | Vol. 6 uses order 25; after #150 merges, azure-ai-agents steps are renumbered 25–28 — verify no collision. |

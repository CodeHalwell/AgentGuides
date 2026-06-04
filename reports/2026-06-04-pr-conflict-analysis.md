# PR Conflict Analysis — 2026-06-04

Generated: 2026-06-04  
Repository: `CodeHalwell/AgentGuides`  
Analyst: Claude Code

---

## Open Pull Requests

| # | Title | Author | Base Branch | Mergeable State | Files Changed | +/- |
|---|---|---|---|---|---|---|
| [#156](https://github.com/CodeHalwell/AgentGuides/pull/156) | Microsoft Agent Framework guides: Class Deep Dives Vol. 7 (1.7.0) | CodeHalwell | `claude/optimistic-hopper-uATD0` (PR #155) | ✅ clean | 2 | +1,436 |
| [#155](https://github.com/CodeHalwell/AgentGuides/pull/155) | Docs refresh — 2026-06-04 | CodeHalwell | `main` | ✅ clean | 21 | +4,539 / -39 |
| [#154](https://github.com/CodeHalwell/AgentGuides/pull/154) | PydanticAI guides: Class Deep Dives Vol. 10 (1.105.0) | CodeHalwell | `main` | ✅ clean | 2 | +1,646 / -1 |
| [#153](https://github.com/CodeHalwell/AgentGuides/pull/153) | Google ADK guides: class deep-dives Vol. 12 (2.1.0) | CodeHalwell | `main` | ✅ clean | 1 | +1,075 |
| [#152](https://github.com/CodeHalwell/AgentGuides/pull/152) | LangGraph guides: class deep-dives Vol. 9 + v1.2.4 reference updates | CodeHalwell | `main` | ✅ clean | 8 | +1,482 / -16 |

---

## Merge Conflict Analysis

**Result: No Git merge conflicts detected.** All five PRs report `mergeable_state: clean` against their respective base branches.

---

## Structural / Ordering Dependencies

While there are no raw Git conflicts today, there is a significant **logical dependency chain** that must be respected when merging.

### Dependency diagram

```
main
 ├── PR #152  (LangGraph Vol. 9)           ─┐
 ├── PR #153  (Google ADK Vol. 12)          ├─ already incorporated into #155's branch
 ├── PR #154  (PydanticAI Vol. 10)         ─┘
 │
 └── PR #155  (Docs refresh — daily run)
              └── PR #156  (MS Agent Framework Vol. 7)
                           (base = #155's branch, not main)
```

### Risk: PRs #152 / #153 / #154 vs PR #155

PR #155 explicitly states it incorporated #152, #153, and #154 **by merging those branches into its own branch** before running the version-bump sweep. This means #155's branch already contains the full content of all three PRs, possibly with additional edits on top.

If #152, #153, or #154 are merged into `main` **before** #155, the subsequent merge of #155 will encounter conflicts or duplicate content in the files those PRs touch:

| PR | Files at risk of duplication/conflict with #155 |
|---|---|
| #152 | `langgraph_class_deep_dives_v9.md`, `chapter-04-tools.md`, `reference-command-and-send.md`, `reference-functional-api.md`, `reference-state-graph.md`, `reference-remote-graph.md`, `chapter-01-setup-and-core-concepts.md`, `index.mdx` |
| #153 | `google_adk_class_deep_dives_v12.md` |
| #154 | `pydantic_ai_class_deep_dives_v10.md`, `index.mdx` |

### Risk: PR #156 depends on PR #155

PR #156's base branch is `claude/optimistic-hopper-uATD0` — the head branch of PR #155, not `main`. This means:

- **#156 cannot be merged to `main` until #155 is merged first.** Once #155 is merged, #156's base branch no longer exists as a separate ref; GitHub will typically redirect or require a base-branch update.
- If #155 is merged to `main`, #156 will need its base branch updated to `main` before it can be merged.

---

## Recommended Merge Order

To avoid conflicts and wasted effort, merge in this sequence:

1. **Close or supersede #152, #153, #154** — their content is already included in #155. If they are intended as standalone PRs independent of the daily refresh, review them carefully for any content that #155 may have altered. Otherwise, close them with a note that they were incorporated.

2. **Merge #155** (`Docs refresh — 2026-06-04`) into `main`.

3. **Update #156's base branch** from `claude/optimistic-hopper-uATD0` to `main`, then merge #156.

---

## Open Item Flagged in #155

PR #155 carries a carry-forward risk item that requires human attention:

> **`semantic-kernel 1.43.0`** — full symbol verification deferred. The `PyMeta3` wheel build fails in the CI environment. Version bump was applied on PyPI evidence alone. Verification should be completed in the next refresh run before this PR is merged to avoid shipping unverified docs.

---

## Summary

| Metric | Value |
|---|---|
| Open PRs | 5 |
| PRs with Git merge conflicts | 0 |
| PRs with structural ordering risk | 4 (#152, #153, #154, #156) |
| PRs requiring human decision before merge | 4 |
| Open carry-forward risk items | 1 (semantic-kernel 1.43.0 verification) |

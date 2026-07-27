# Open PR Conflict Review — 2026-07-27

Automated sweep of all open pull requests in `CodeHalwell/AgentGuides`, checking each for merge conflicts against its base branch.

## Summary

**5 open PRs, 0 with merge conflicts.** All report a `clean` mergeable state against their respective base branches.

## PR-by-PR status

| PR | Title | Author | Base ← Head | Status | Conflicts |
|----|-------|--------|-------------|--------|-----------|
| [#310](https://github.com/CodeHalwell/AgentGuides/pull/310) | Add Microsoft agent framework Python class deep dives Vol. 41 (agent-framework 1.12.1) | CodeHalwell | `main` ← `claude/intelligent-goldberg-ftc57q` | Open | None |
| [#309](https://github.com/CodeHalwell/AgentGuides/pull/309) | Add PydanticAI Class Deep Dives Vol. 39 (pydantic-ai==2.18.0) | CodeHalwell | `main` ← `claude/trusting-goodall-ct3w7x` | Open | None |
| [#308](https://github.com/CodeHalwell/AgentGuides/pull/308) | Add Google ADK class deep dives vol. 45 (google-adk==2.5.0) | CodeHalwell | `main` ← `claude/quirky-gauss-qzw27d` | Open | None |
| [#307](https://github.com/CodeHalwell/AgentGuides/pull/307) | Add LangGraph Class Deep-Dives Vol. 43 (langgraph==1.2.9) | CodeHalwell | `claude/adoring-hawking-ugveg2` ← `claude/loving-goodall-io2x7m` | Open | None (see note below) |
| [#306](https://github.com/CodeHalwell/AgentGuides/pull/306) | docs(index): add Vol. 36 to Jump-to-topic, Reference, What's Shipped, and revision history | CodeHalwell | `main` ← `claude/adoring-hawking-ugveg2` | Open | None |

## Notes / structural risk (not a conflict today)

- **PR #307 is stacked on PR #306**, not on `main` — its base branch is `claude/adoring-hawking-ugveg2` (PR #306's head branch), rather than `main` directly. GitHub reports it as cleanly mergeable *against that branch* right now, but this is a dependency chain, not an independent PR:
  - If #306 merges into `main` first, #307's base ref should be retargeted to `main` (or #307 rebased) before merging, otherwise its diff will include #306's commits.
  - If #306 is closed/changed significantly instead, #307 would need to be rebased onto `main` directly.
- **All 5 PRs touch `index.mdx`** (Jump-to-topic CardGrid, Reference CardGrid, Zero→Hero CardGrid, and/or What's Shipped/version-badge sections). Each is currently conflict-free in isolation against its *current* base, but merging them serially into `main` will very likely produce conflicts on `index.mdx` between whichever PRs merge 2nd through 5th, since they all edit the same CardGrid/list regions. Recommended merge order to minimize friction: merge #306 first (small, 4/-1 lines), retarget/rebase #307 onto `main`, then merge #307, #308, #309, #310 one at a time, resolving the small `index.mdx` list-insertion conflicts as they arise (each is an additive, non-overlapping content insertion, not a substantive logic conflict).

## Conclusion

No open PR currently has an unresolved merge conflict against its declared base branch. The one item worth the maintainer's attention is the #306/#307 branch stacking, plus the likelihood of cascading (but low-severity, purely additive) `index.mdx` conflicts as these PRs are merged one after another.

# Open PR Conflict Audit — 2026-07-01

## Executive Summary

All 16 open pull requests in `CodeHalwell/AgentGuides` (#242–#257) were checked against their respective base branches for merge conflicts. **Every PR reports a clean, mergeable state — no unresolved conflicts were found.** No action is required to unblock any PR on conflict grounds.

One non-conflict item worth the maintainer's attention: PR #256 and PR #244 both label themselves "PydanticAI Class Deep Dives Vol. 28," but #256 is a second, independently-branched vol. 28 (targeting `pydantic-ai==2.2.0`, based directly on `main`) while #244 is the original vol. 28 (targeting `pydantic-ai==2.0.0`, already extended by the stacked chain #244 → #248 → #252). This is a content/numbering duplication risk, not a git conflict, and won't be caught by GitHub's mergeability check. Similarly, PR #243 ("Google ADK class deep dives vol. 30") and PR #247 ("Google ADK vol. 30 + vol. 31 + guide") both target `main` and cover "vol. 30" for Google ADK, presenting another content duplication risk.

## PR Inventory & Status

| PR# | Title (short) | Author | Base branch | Head branch | Mergeable? | State |
|---|---|---|---|---|---|---|
| 242 | LangGraph Class Deep Dives Vol. 28 | CodeHalwell | main | claude/gracious-tesla-88zmzs | clean | open |
| 243 | Google ADK class deep dives vol. 30 | CodeHalwell | main | claude/gracious-clarke-2jmwuz | clean | open |
| 244 | PydanticAI class deep dives vol. 28 | CodeHalwell | main | claude/loving-johnson-ettv80 | clean | open |
| 245 | Microsoft Agent Framework vol. 26 | CodeHalwell | main | claude/relaxed-clarke-49v8qy | clean | open |
| 246 | LangGraph vol. 29 + v1.2.6 update | CodeHalwell | claude/gracious-tesla-88zmzs (#242) | claude/gracious-tesla-k4xwmt | clean | open |
| 247 | Google ADK vol. 30 + vol. 31 + guide | CodeHalwell | main | claude/gracious-clarke-2n3cmq | clean | open |
| 248 | PydanticAI vol. 29 | CodeHalwell | claude/loving-johnson-ettv80 (#244) | claude/loving-johnson-ouleg9 | clean | open |
| 249 | Microsoft Agent Framework vol. 27 | CodeHalwell | claude/relaxed-clarke-49v8qy (#245) | claude/relaxed-clarke-39y7ok | clean | open |
| 250 | LangGraph vol. 30 + bump v1.2.7 | CodeHalwell | claude/gracious-tesla-k4xwmt (#246) | claude/gracious-tesla-y76kke | clean | open |
| 251 | Google ADK vol. 32 | CodeHalwell | claude/gracious-clarke-2n3cmq (#247) | claude/gracious-clarke-33p2hh | clean | open |
| 252 | PydanticAI vol. 30 (2.1.0) | CodeHalwell | claude/loving-johnson-ouleg9 (#248) | claude/loving-johnson-p64bss | clean | open |
| 253 | Microsoft Agent Framework vol. 28 | CodeHalwell | claude/relaxed-clarke-39y7ok (#249) | claude/relaxed-clarke-oi8qte | clean | open |
| 254 | LangGraph vol. 31 | CodeHalwell | claude/gracious-tesla-y76kke (#250) | claude/gracious-tesla-s881b1 | clean | open |
| 255 | Google ADK vol. 33 | CodeHalwell | claude/gracious-clarke-33p2hh (#251) | claude/gracious-clarke-fanrv8 | clean | open |
| 256 | PydanticAI vol. 28 (v2.2.0) | CodeHalwell | main | claude/loving-johnson-fxoa1s | clean | open |
| 257 | Microsoft Agent Framework vol. 29 | CodeHalwell | claude/relaxed-clarke-oi8qte (#253) | claude/relaxed-clarke-dji33s | clean | open |

All 16 PRs: `state=open`, `draft=false`, `merged=false`, author=CodeHalwell.

## Stacked PR Chains

Several PRs intentionally target another open PR's branch instead of `main`, forming ordered chains so that volumes ship in sequence:

- **LangGraph**: #242 (main) → #246 → #250 → #254
- **Google ADK**: #247 (main) → #251 → #255 (separately, #243 is its own main-based PR — see duplication note above)
- **PydanticAI**: #244 (main) → #248 → #252 (separately, #256 is a second, independent main-based PR — see duplication note above)
- **Microsoft Agent Framework**: #245 (main) → #249 → #253 → #257

Chains are clean end-to-end. As long as each chain is merged in order (root first), no conflicts are expected to surface post-merge.

## Conflicted PRs

**None.** All 16 open PRs report `mergeable_state: clean` against their base branch — no textual merge conflicts (overlapping edits, deletion-vs-modification, etc.) were detected between any PR and its base.

## Recommendation

- Safe to merge any chain starting from its root PR, in order.
- Reconcile #256 against the #244 → #248 → #252 PydanticAI stack before merging, to avoid two "Vol. 28" documents / conflicting sidebar order and version numbers landing on `main`.
- Reconcile #243 against #247 to avoid duplicating Google ADK Vol. 30 content on `main`.

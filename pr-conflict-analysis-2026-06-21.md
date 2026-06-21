# PR Conflict Analysis Report — 2026-06-21

**Run date:** 2026-06-21  
**Scope:** All open pull requests in `codehalwell/AgentGuides`  
**Total open PRs:** 21 (#197 – #217)  
**Author:** All PRs authored by CodeHalwell

---

## 1. Summary

| Metric | Value |
|--------|-------|
| Open PRs | 21 |
| PRs with git-level merge conflicts (`mergeable_state: conflicting`) | **0** |
| PRs with structural / logical conflict risk | **2 groups** |
| PRs that are drafts | 0 |

All 21 PRs report `mergeable_state: clean` against their declared base branches. However, two structural conflict risks exist that will surface as real merge conflicts once certain PRs land:

1. **Google ADK "Vol. 20" duplication** — PR #215 is an independent re-implementation of Vol. 20 branching from `main`, while PR #198 (also Vol. 20) is the root of the existing 4-PR ADK stack (#198 → #202 → #206 → #210). Whichever lands second will conflict on `index.mdx` and the new `.md` file.
2. **Agent Framework consolidation PR (#212)** — PR #212 targets `main` directly with 39 commits that already incorporate the content of PRs #200, #204, and #208 (Vols. 15–17). Once any of those three chain PRs lands independently, #212 will conflict.

---

## 2. Full PR List

| PR # | Title | Base Branch | Head Branch | `mergeable_state` |
|------|-------|-------------|-------------|-------------------|
| #197 | LangGraph: Class Deep Dives Vol. 18 — channels, caching, functional patterns & debug streaming (1.2.5) | `main` | `claude/gracious-tesla-mnnjnr` | ✅ clean |
| #198 | Google ADK: Class Deep Dives Vol. 20 — compaction internals, HITL utilities, workflow composition & backend detection (2.2.0) | `main` | `claude/gracious-clarke-q852ly` | ✅ clean |
| #199 | PydanticAI: Class Deep Dives Vol. 18 — new providers, Google ecosystem & graph persistence (1.107.0) | `main` | `claude/loving-johnson-srxpx7` | ✅ clean |
| #200 | docs(agent-framework/python): Class Deep Dives Vol. 15 — AG-UI, ChatKit, DevServer, GAIA, CopilotStudio, Azure Search, Cosmos, Durable, Functions | `main` | `claude/relaxed-clarke-i4wg1o` | ✅ clean |
| #201 | LangGraph: Class Deep Dives Vol. 19 — streaming internals, error taxonomy, HITL protocol & execution model (1.2.5) | `claude/gracious-tesla-mnnjnr` | `claude/gracious-tesla-p3pdt4` | ✅ clean |
| #202 | Google ADK: class deep-dives Vol. 21 — conformance testing, simulation personas, rubric evaluation & eval infrastructure (2.2.0) | `claude/gracious-clarke-q852ly` | `claude/gracious-clarke-te88tv` | ✅ clean |
| #203 | Add PydanticAI Class Deep Dives Vol. 19 | `claude/loving-johnson-srxpx7` | `claude/loving-johnson-lc9pxr` | ✅ clean |
| #204 | docs(agent-framework/python): Class Deep Dives Vol. 16 — Foundry hosting, Bedrock providers, orchestration base classes (1.8.1) | `claude/relaxed-clarke-i4wg1o` | `claude/relaxed-clarke-2p3f0a` | ✅ clean |
| #205 | LangGraph: Class Deep Dives Vol. 20 — execution engine internals (1.2.6) | `claude/gracious-tesla-p3pdt4` | `claude/gracious-tesla-rdyn15` | ✅ clean |
| #206 | Google ADK: Class Deep Dives Vol. 22 — credential exchange, workflow errors, replay barrier, conformance testing & tool-connection analysis (2.3.0) | `claude/gracious-clarke-te88tv` | `claude/gracious-clarke-9sazix` | ✅ clean |
| #207 | PydanticAI: Class Deep Dives Vol. 20 — middleware, SSRF, toolset composition & model profiles (1.107.0) | `claude/loving-johnson-lc9pxr` | `claude/loving-johnson-2b7zfm` | ✅ clean |
| #208 | docs(agent-framework/python): Class Deep Dives Vol. 17 — agent-framework 1.9.0 | `claude/relaxed-clarke-2p3f0a` | `claude/relaxed-clarke-jyibkd` | ✅ clean |
| #209 | LangGraph: Class Deep Dives Vol. 21 — channels, serialization & graph protocols (1.2.6) | `claude/gracious-tesla-rdyn15` | `claude/gracious-tesla-0kgk5h` | ✅ clean |
| #210 | Add Google ADK class deep dives — vol. 23 (google-adk 2.3.0) | `claude/gracious-clarke-9sazix` | `claude/gracious-clarke-bnesa9` | ✅ clean |
| #211 | PydanticAI Class Deep Dives Vol. 21 — TestModel, FunctionModel, native tool 1.107.0 updates | `claude/loving-johnson-2b7zfm` | `claude/loving-johnson-naf22q` | ✅ clean |
| #212 | docs: Class deep dives Vol. 18 — agent-framework 1.9.0 (10 source-verified class groups) | `main` | `claude/relaxed-clarke-jilb76` | ✅ clean ⚠️ |
| #213 | docs: PR conflict analysis report — 2026-06-20 (all 16 PRs clean) | `main` | `claude/optimistic-einstein-8bs0o9` | ✅ clean |
| #214 | LangGraph: Class Deep Dives Vol. 22 — v3 streaming internals, custom transformers, encryption & embedding helpers (1.2.6) | `claude/gracious-tesla-0kgk5h` | `claude/gracious-tesla-d58kme` | ✅ clean |
| #215 | Google ADK: Class deep dives vol. 20 — 10 source-verified dives against 2.3.0 | `main` | `claude/gracious-clarke-s4q22x` | ✅ clean ⚠️ |
| #216 | PydanticAI: Class Deep Dives Vol. 22 — RunContext, UsageLimits, DeferredTools, ApprovalRequired, Concurrency & more (1.107.0) | `claude/loving-johnson-naf22q` | `claude/loving-johnson-l2og7q` | ✅ clean |
| #217 | docs(agent-framework/python): Class Deep Dives Vol. 19 — orchestration builders & harness providers (1.9.0) | `claude/relaxed-clarke-jilb76` | `claude/relaxed-clarke-gk40bq` | ✅ clean |

---

## 3. Stacked PR Chains

PRs are organised into parallel stacked chains, one per framework. Each PR in a chain uses the previous PR's head as its base — merging must happen in order (root first).

### 3a. LangGraph (5 PRs) — no conflict risk

```
main
 └── #197 Vol. 18 (1.2.5)  →  gracious-tesla-mnnjnr
      └── #201 Vol. 19 (1.2.5)  →  gracious-tesla-p3pdt4
           └── #205 Vol. 20 (1.2.6)  →  gracious-tesla-rdyn15
                └── #209 Vol. 21 (1.2.6)  →  gracious-tesla-0kgk5h
                     └── #214 Vol. 22 (1.2.6)  →  gracious-tesla-d58kme
```

Clean linear stack. Merge in order: #197 → #201 → #205 → #209 → #214.

---

### 3b. Google ADK (4 PRs + 1 rogue) — ⚠️ CONFLICT RISK

```
main
 ├── #198 Vol. 20 (2.2.0)  →  gracious-clarke-q852ly          ← Chain root
 │    └── #202 Vol. 21 (2.2.0)  →  gracious-clarke-te88tv
 │         └── #206 Vol. 22 (2.3.0)  →  gracious-clarke-9sazix
 │              └── #210 Vol. 23 (2.3.0)  →  gracious-clarke-bnesa9
 │
 └── #215 Vol. 20 (2.3.0)  →  gracious-clarke-s4q22x          ← ⚠️ DUPLICATE
```

**Risk:** PR #215 is a standalone Vol. 20 page branched directly from `main` because the author could not access the `gracious-clarke-*` stack branches at creation time (noted in PR body). Both #198 and #215 add `google_adk_class_deep_dives_v20.md` and modify `index.mdx` with a "Step 34" Zero→Hero card and sidebar `order: 89`.

**Nature of conflict:** If #198 lands first:
- `index.mdx` will have diverged (Step 34 card added by #198); #215 attempts to add the same card → **edit/edit conflict** on `index.mdx`
- `google_adk_class_deep_dives_v20.md` already exists on `main`; #215 adds a file of the same name → **add/add conflict** (two distinct implementations of "Vol. 20")

**Recommendation:** Review and choose one Vol. 20 to keep. The #198 chain (Vols. 20–23) is the established stack and should be preferred. PR #215 covers 2.3.0-specific content that may be worth merging into the stack as an amendment rather than a standalone PR. Close or rebase #215 after #198 lands.

---

### 3c. PydanticAI (5 PRs) — no conflict risk

```
main
 └── #199 Vol. 18 (1.107.0)  →  loving-johnson-srxpx7
      └── #203 Vol. 19 (1.107.0)  →  loving-johnson-lc9pxr
           └── #207 Vol. 20 (1.107.0)  →  loving-johnson-2b7zfm
                └── #211 Vol. 21 (1.107.0)  →  loving-johnson-naf22q
                     └── #216 Vol. 22 (1.107.0)  →  loving-johnson-l2og7q
```

Clean linear stack. Merge in order: #199 → #203 → #207 → #211 → #216.

---

### 3d. Microsoft Agent Framework (3 PRs + 1 consolidation) — ⚠️ CONFLICT RISK

```
main
 ├── #200 Vol. 15 (1.8.1)  →  relaxed-clarke-i4wg1o           ← Chain root
 │    └── #204 Vol. 16 (1.8.1)  →  relaxed-clarke-2p3f0a
 │         └── #208 Vol. 17 (1.9.0)  →  relaxed-clarke-jyibkd
 │
 └── #212 Vols. 15–18 consolidated (1.9.0)  →  relaxed-clarke-jilb76   ← ⚠️ CONSOLIDATION
          └── #217 Vol. 19 (1.9.0)  →  relaxed-clarke-gk40bq
```

**Risk:** PR #212 directly targets `main` and contains 39 commits that incorporate Vols. 15–18 of the agent-framework docs (it fast-forward-merged `claude/relaxed-clarke-jyibkd`, the head of #208). PR #217 is stacked on top of #212.

If the chain PRs (#200, #204, #208) are merged to `main` one by one first:
- Every merge to `main` changes the `main` SHA that #212's commits were based on
- `index.mdx` will have been modified by each chain PR; #212 will then attempt to apply its own version → **edit/edit conflict** on `index.mdx`
- The `.md` content files may also collide if the chain PRs introduce them with slightly different content than what #212 bundled

**Recommendation:** Do not merge the chain (#200 → #204 → #208) and #212 independently. Choose one path:
- **Option A:** Merge chain PRs (#200 → #204 → #208) to main, then close #212 and rebase #217 onto the updated `main` after the chain lands.
- **Option B:** Close chain PRs (#200, #204, #208) and merge #212 directly, treating it as the authoritative "catch-up" PR for Vols. 15–18. Then merge #217.

---

### 3e. Standalone docs PR

| PR # | Title | Notes |
|------|-------|-------|
| #213 | docs: PR conflict analysis report — 2026-06-20 (all 16 PRs clean) | Previous run of this analysis. Now superseded by this report. Can be merged or closed. |

---

## 4. Conflict Risk Summary

| Risk | PRs Involved | Files at Risk | Action |
|------|-------------|---------------|--------|
| Google ADK Vol. 20 duplication | #198 vs #215 | `google_adk_class_deep_dives_v20.md`, `index.mdx` | Merge #198 chain; close or cherry-pick #215 |
| Agent Framework consolidation collision | #200, #204, #208 vs #212 | `index.mdx`, `microsoft_agent_framework_python_class_deep_dives_v15–18.md` | Pick one path; don't merge both independently |

---

## 5. Recommended Merge Order

Merge the following groups independently and in parallel (they touch different file trees):

**Group 1 — LangGraph:** `#197 → #201 → #205 → #209 → #214`  
**Group 2 — Google ADK:** `#198 → #202 → #206 → #210` (then assess #215)  
**Group 3 — PydanticAI:** `#199 → #203 → #207 → #211 → #216`  
**Group 4 — Agent Framework:** Resolve #212 vs chain PRs first, then merge whichever wins → #217  
**Group 5 — Docs:** Merge or close #213 (previous analysis, superseded)

---

*Generated by Claude Code — automated PR conflict analysis run*

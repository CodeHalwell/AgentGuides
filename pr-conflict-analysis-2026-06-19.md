# PR Conflict Analysis — 2026-06-19

## Summary

**12 open pull requests · 0 merge conflicts · all `mergeable_state: clean`**

All open PRs are organised into four independent stacking chains, one per framework. Each chain is a series of sequential volumes branched off the previous volume's branch. No PR has conflicts with its base branch.

---

## Open PRs by Chain

### LangGraph

| PR | Title | Author | Base | Mergeable |
|----|-------|--------|------|-----------|
| [#197](https://github.com/CodeHalwell/AgentGuides/pull/197) | Class Deep Dives Vol. 18 — channels, caching, functional patterns & debug streaming (1.2.5) | CodeHalwell | `main` | ✅ clean |
| [#201](https://github.com/CodeHalwell/AgentGuides/pull/201) | Class Deep Dives Vol. 19 — streaming internals, error taxonomy, HITL protocol & execution model (1.2.5) | CodeHalwell | `claude/gracious-tesla-mnnjnr` (#197) | ✅ clean |
| [#205](https://github.com/CodeHalwell/AgentGuides/pull/205) | Class Deep Dives Vol. 20 — execution engine internals (1.2.6) | CodeHalwell | `claude/gracious-tesla-p3pdt4` (#201) | ✅ clean |

### Google ADK

| PR | Title | Author | Base | Mergeable |
|----|-------|--------|------|-----------|
| [#198](https://github.com/CodeHalwell/AgentGuides/pull/198) | Class Deep Dives Vol. 20 — compaction internals, HITL utilities, workflow composition & backend detection (2.2.0) | CodeHalwell | `main` | ✅ clean |
| [#202](https://github.com/CodeHalwell/AgentGuides/pull/202) | Class Deep Dives Vol. 21 — conformance testing, simulation personas, rubric evaluation & eval infrastructure (2.2.0) | CodeHalwell | `claude/gracious-clarke-q852ly` (#198) | ✅ clean |
| [#206](https://github.com/CodeHalwell/AgentGuides/pull/206) | Class Deep Dives Vol. 22 — credential exchange, workflow errors, replay barrier, conformance testing & tool-connection analysis (2.3.0) | CodeHalwell | `claude/gracious-clarke-te88tv` (#202) | ✅ clean |

### PydanticAI

| PR | Title | Author | Base | Mergeable |
|----|-------|--------|------|-----------|
| [#199](https://github.com/CodeHalwell/AgentGuides/pull/199) | Class Deep Dives Vol. 18 — new providers, Google ecosystem & graph persistence (1.107.0) | CodeHalwell | `main` | ✅ clean |
| [#203](https://github.com/CodeHalwell/AgentGuides/pull/203) | Class Deep Dives Vol. 19 | CodeHalwell | `claude/loving-johnson-srxpx7` (#199) | ✅ clean |
| [#207](https://github.com/CodeHalwell/AgentGuides/pull/207) | Class Deep Dives Vol. 20 — middleware, SSRF, toolset composition & model profiles (1.107.0) | CodeHalwell | `claude/loving-johnson-lc9pxr` (#203) | ✅ clean |

### Microsoft Agent Framework (Python)

| PR | Title | Author | Base | Mergeable |
|----|-------|--------|------|-----------|
| [#200](https://github.com/CodeHalwell/AgentGuides/pull/200) | Class Deep Dives Vol. 15 — AG-UI, ChatKit, DevServer, GAIA, CopilotStudio, Azure Search, Cosmos, Durable, Functions | CodeHalwell | `main` | ✅ clean |
| [#204](https://github.com/CodeHalwell/AgentGuides/pull/204) | Class Deep Dives Vol. 16 — Foundry hosting, Bedrock providers, orchestration base classes (1.8.1) | CodeHalwell | `claude/relaxed-clarke-i4wg1o` (#200) | ✅ clean |
| [#208](https://github.com/CodeHalwell/AgentGuides/pull/208) | Class Deep Dives Vol. 17 — agent-framework 1.9.0 | CodeHalwell | `claude/relaxed-clarke-2p3f0a` (#204) | ✅ clean |

---

## Chain Topology

Each framework follows the same stacking pattern — each successive volume branches off the previous volume's head branch rather than `main`. This means:

- Only the **root PR** in each chain (#197, #198, #199, #200) needs to be merged into `main` first.
- The subsequent PRs in a chain (#201→#205, #202→#206, #203→#207, #204→#208) can then be retargeted to `main` and merged in order.
- Because each chain works on distinct directories, no cross-framework conflicts are expected even after all root PRs land on `main` simultaneously.

---

## Conflict Analysis

**No conflicts found.** All 12 PRs report `mergeable_state: clean` against their respective base branches. GitHub has performed its merge simulation and confirmed each can be merged without conflict resolution.

The stacking chain structure is intentional: each PR carries its predecessor's changes in the branch history, so intermediate branches never diverge from one another.

---

## Recommended Merge Order

To land all 12 PRs cleanly, merge within each chain in order (root first), then optionally retarget and squash the chain PRs to `main`:

1. **LangGraph**: #197 → #201 → #205
2. **Google ADK**: #198 → #202 → #206
3. **PydanticAI**: #199 → #203 → #207
4. **Agent Framework**: #200 → #204 → #208

All four chains are independent and can be processed in parallel.

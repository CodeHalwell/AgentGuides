# Guide Consistency Review — 2026-09-08

A cross-guide sweep for two problems: framework guides stating several different
versions of the same package, and reading paths that don't work — tables of
contents pointing at sections that were never written, and code fences that swallow
the headings after them.

No packages were installed and no symbols were introspected during this pass. Where
a version had to be chosen, the canonical value is the highest version the guide
family already evidences (a version card, a `Latest:` stamp, or a source-verified
deep dive), not a fresh registry lookup. The next scheduled refresh should confirm
these against PyPI/npm/NuGet as usual.

---

## 1. Version numbers

### 1.1 What was wrong

Version claims lived in three places that had drifted apart: the `Version` card on
each guide's `index.mdx`, the `Latest:` stamp at the top of each page, and pinned
install commands in body text. Several guides stated three different versions of
the same package.

The worst cases:

| Guide | Stated versions before | Now |
|---|---|---|
| Microsoft Agent Framework | hub index `Core 1.4.0`; Python card `1.15.0`; pages 1.5.0 – 1.17.0; .NET card `1.0.1` vs pages `1.1.0` | Python `agent-framework` 1.17.0 · .NET `Microsoft.Agents.AI` 1.1.0 |
| OpenAI Agents SDK (Python) | comprehensive `0.17.4`; middleware `6.8.1`; streaming server `2.7.2` | `openai-agents` 0.17.4 everywhere |
| Claude Agent SDK (TypeScript) | comprehensive `0.3.165`; three pages `0.68.0` | `@anthropic-ai/claude-agent-sdk` 0.3.165 everywhere |
| Google ADK | hub index Py `2.7.1` / TS `0.6.1`; Python card `2.8.0` | Py 2.8.0 · Go 1.0.0 · TS 1.0.0 |
| Haystack | comprehensive `2.30.0`; three pages `2.27.0` | `haystack-ai` 2.30.0 everywhere |
| LangGraph (TypeScript) | comprehensive `1.3.5`; two pages `1.0.2`; hub index `1.3.0` | `@langchain/langgraph` 1.3.5 everywhere |
| Semantic Kernel | hub index Py `1.41.3`; Python card `1.43.0`; streaming page `1.41.2` | Py 1.43.0 · .NET 1.74.0 |
| Mistral | comprehensive `2.4.9`; two pages `2.0.1` | `mistralai` 2.4.9 everywhere |
| LlamaIndex (Python) | card `0.14.22`; two pages `0.14.20`; hub index `0.14.21` | `llama-index-core` 0.14.22 everywhere |
| Anthropic Claude Agent SDK (Python) | card `0.2.93`; comprehensive `0.2.91` | `claude-agent-sdk` 0.2.93 |
| PydanticAI | card `2.40.0`; comprehensive `2.33.0` | latest 2.40.0, verified-against 2.33.0 stated separately |

Two of the mismatches were not stale versions at all — the number belonged to a
*different package*. `Latest: 6.8.1` inside the OpenAI Agents SDK guide referred to
the base `openai` SDK, and `Latest: 0.68.0` inside the Claude Agent SDK guide
referred to `@anthropic-ai/sdk`. A reader had no way to tell.

### 1.2 What changed

- **Every `Latest:` stamp now names its package** — `Latest: haystack-ai 2.30.0`
  rather than a bare `Latest: 2.30.0`. This is what stops the same drift recurring:
  a number with a package name in front of it can be checked.
- **One version per guide family.** Every `Latest:` stamp, hub-index card and
  language card within a guide now states the same package and version.
- **Latest vs verified-against are now separate claims** where they differ. The
  Google ADK, Microsoft Agent Framework (Python) and PydanticAI comprehensive
  guides read `Latest: <pkg> <x> | Guide verified against: <y>`. Per-page
  "Verified against `pkg==x`" banners were left untouched — they are provenance,
  and rewriting them without re-running the verification would be a false claim.
- **Pages that document a different SDK now say so.** The four streaming/middleware
  pages that call the base `openai` or `@anthropic-ai/sdk` client carry their
  guide's framework version plus a one-line note naming the auxiliary package.
- **Stale install pins bumped**, including two that could not have worked:
  `claude-agent-sdk>=1.0.0` (no such release; latest is 0.2.93) and a production
  requirements block pinning `llama-index==0.14.6` alongside
  `llama-index-core==0.2.1`.

---

## 2. Reading paths

### 2.1 Broken tables of contents — 180 dead anchors

Anchor targets were checked against real headings across all 282 pages. 180 links
resolved to nothing. Most were in-page tables of contents that promised sections
the document never contained:

- **SmolAgents comprehensive** — ToC listed 20 sections; 10 existed.
- **Amazon Bedrock comprehensive** — ToC listed 18; 2 existed.
- **Haystack comprehensive** — ToC listed Parts I–XIV; Parts I–III existed.
- **OpenAI Agents SDK (TypeScript) comprehensive** — ToC listed 18; 5 existed.
- **Claude Agent SDK (TypeScript) comprehensive** — ToC listed 19; 5 existed.
- **LangGraph (TypeScript) comprehensive** — ToC listed 22; 14 existed under
  different names.

Each ToC was rewritten to list the sections that are actually there, in document
order, followed by an explicit pointer to the sibling pages that cover the rest.
Where a heading had simply been renamed (CrewAI recipes, the LangGraph Python
Zero → Hero deep links, the Microsoft Agent Framework Python index, `quick-start.md`),
the link was retargeted instead of removed.

### 2.2 Drafting artefacts left in published pages

Three pages carried text that was never meant to ship:

- `openai_agents_sdk_typescript_comprehensive_guide.md` — a section reading
  *"This is approximately 35% of the comprehensive guide … Would you like me to
  continue with the remaining sections?"*
- `haystack_production_guide.md` — *"Due to space constraints, I've covered the
  essential production topics."*
- `langchain_langgraph_comprehensive_guide.md` — a bare
  *"[Continuing with remaining sections...]"* marker.

All three replaced with an honest statement of what the page covers and where the
rest lives.

### 2.3 Broken code fences

Seven pages had fence errors that corrupted rendering, mostly nested triple-backtick
blocks inside prompt strings closing their parent block early:

| Page | Effect |
|---|---|
| `anthropic_claude_agent_sdk_recipes.md` | Recipe 10 and the "DevOps & Infrastructure" heading rendered as code |
| `autogen_recipes.md` | Recipe 3's heading rendered as code |
| `google_adk_recipes.md` | "Meeting Scheduler" heading rendered as code |
| `google_adk_comprehensive_guide.md` | "Context Caching Strategies" heading rendered as code |
| `google_adk_advanced_python.md` | last third of the page (security table, gotchas) rendered as code |
| `callbacks-and-plugins.md` | "Terminating an invocation from a callback" rendered as code |
| `bedrock_agents_comprehensive_guide.md` | "Code Interpretation" and "AWS CLI and SDK Setup" rendered as code |
| `claude_agent_sdk_typescript_middleware.md` | "Streaming" section rendered as code |
| `anthropic_claude_agent_sdk_production_guide.md` | whole Troubleshooting section wrapped in a ```markdown block |
| `bedrock_agents_observability_python.md` | stray trailing fence |

Fixed by widening the outer fence to four backticks where a nested block is
intentional, adding missing closing fences, and unwrapping the Anthropic
troubleshooting section so it renders as prose.

### 2.4 Duplicate page removed

`autogen-guide/python/autogen_comprehensive_guide_updated.md` was deleted. It was a
1,402-line near-duplicate of the linked 2,180-line `autogen_comprehensive_guide.md`,
was not referenced by the sidebar, any index, or `guides.md`, had its own broken
ToC, and gave the AutoGen guide two conflicting "comprehensive guides". Its content
is a subset of the page that remains; it is recoverable from git history.

### 2.5 Frontmatter descriptions

Ten pages had a `description` that was a truncated dump of their table of contents
("1. Core Fundamentals 2. Simple Agents 3. Multi-Agent Systems…"), which is what
search results and social previews show. Replaced with real one-line descriptions.

---

## 3. Verification

- `node scripts/check-links.mjs` — 281 files, 1,500 links, 0 broken.
- Anchor checker (headings parsed with CommonMark fence rules) — 0 broken, down
  from 180.
- Unclosed-fence checker — 0, down from 3.
- `npm run build` — 282 pages, 0 errors.

---

## 4. Not done

- **No upstream version verification.** Canonical versions were chosen from
  evidence already in the repo. If any of these have moved on PyPI/npm/NuGet, the
  next refresh routine will catch it — and now has a single value per guide to
  update rather than three.
- **Per-page "verified against" banners left as they are.** LangGraph Python
  reference pages, for example, span 1.2.1 – 1.2.11 depending on when each was
  checked. These are accurate statements about when verification happened; the
  headline version is what has been unified.
- **Auxiliary-SDK pages not rewritten.** Four streaming/middleware pages sit inside
  a framework guide but demonstrate the base vendor SDK rather than the agent
  framework. They are now labelled rather than rewritten — converting them to the
  agent-loop API is a content decision, not a consistency fix.

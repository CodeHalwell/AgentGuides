# AgentGuides

Astro + Starlight documentation site. Guides live in `src/content/docs/<framework>-guide/`, with `python/`, `typescript/`, `dotnet/` or `go/` subfolders where a framework has several languages. Sidebar is in `astro.config.mjs`; home-page framework cards are in `src/data/frameworks.ts`.

## Versions: one source of truth

- `src/data/versions.json` is the only place a current package version is written.
- In docs, write the current version as a placeholder: `pip install agent-framework==%%version=agent-framework%%`, `%%versions-checked%%` for the date. `src/plugins/remark-versions.mjs` fills these in at build time; an unknown key fails the build.
- Placeholders don't work in frontmatter, and don't belong in headings (anchors would change on every bump). Keep current version numbers out of titles and descriptions.
- Literal version numbers are only for history: "added in 1.17.0", changelog entries, migration notes, revision-history rows.
- To move to a new release: `node scripts/check-versions.mjs --latest --write`, then read the changelog and fix any examples a breaking change affects.
- `npm run check:versions` must pass before a PR (fails on hard-coded install pins and unknown placeholders).

## Content

- Improve existing pages. Material from studying a class or API goes into the guide section where a reader would look for it (comprehensive guide, topic page, recipes). Do not create "deep dive", "class reference", "Vol. N" or per-release pages.
- Every code example must work against the current release. Install the package and check imports and signatures (run with a mock/test model where possible). Never invent APIs; if a framework lacks a feature, say so.
- Use root-absolute internal links (`/langgraph-guide/python/...`); `src/plugins/remark-base-path.mjs` adds the `/AgentGuides` base. Check with `node scripts/check-links.mjs`.
- British English, plain and understated. No hype.
- In `.mdx` files, keep `{` and `<` inside code.

## Workflow

- Branch from `main`. Don't branch from or stack on other open PRs.
- Before a PR: `npm run check:versions`, `node scripts/check-links.mjs`, and `npm run build` if you changed config, components or MDX.

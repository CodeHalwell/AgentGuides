# Agent Guides

**Comprehensive, production-ready documentation for building AI agents across 18+ frameworks.**

🌐 **Live site:** https://codehalwell.github.io/AgentGuides/

This repository hosts a [Starlight](https://starlight.astro.build/) documentation site covering every major AI agent framework — OpenAI Agents SDK, LangGraph, CrewAI, LlamaIndex, Microsoft Agent Framework, Amazon Bedrock Agents, Google ADK, Anthropic Claude Agent SDK, and more — across Python, TypeScript, .NET and Go.

## What's inside

Each framework has a landing page plus deep-dive guides:

- **README / index** — install, key concepts, minimum-viable example
- **Comprehensive guide** — full feature reference
- **Production guide** — deployment, scaling, observability, security
- **Diagrams** — architecture and flow visualisations
- **Recipes** — copy-paste patterns for common tasks

Frameworks covered: OpenAI Agents SDK, CrewAI, AutoGen / AG2, LangGraph, LlamaIndex, Haystack, PydanticAI, SmolAgents, Amazon Bedrock Agents, Microsoft Agent Framework, Google ADK, Semantic Kernel, Anthropic Claude Agent SDK, Mistral Agents API.

## Tech stack

- **[Astro](https://astro.build/)** 5 — static site generation
- **[Starlight](https://starlight.astro.build/)** 0.37 — docs framework with search, sidebar, and TOC
- **[Tailwind CSS](https://tailwindcss.com/)** v4 via `@tailwindcss/vite`
- **[React](https://react.dev/)** 19 — interactive islands (framework filter, use-case picker)
- **[Pagefind](https://pagefind.app/)** — client-side search (bundled by Starlight)

The look follows the "Atoms to agents" brand guidelines and UI mockups: flat paper or ink ground, hairline rules, a single oxide accent, Newsreader / Hanken Grotesk / JetBrains Mono (self-hosted via `@fontsource`). Light and dark tokens live at the top of `src/styles/global.css`; code panels use `src/styles/code-theme.json`; the header mark and article header are `src/components/SiteTitle.astro` and `src/components/PageTitle.astro`.

## Local development

```bash
npm install
npm run dev       # dev server on http://localhost:4321
npm run build     # static build into ./dist
npm run preview   # serve the production build locally
```

## Contributing

All documentation lives in `src/content/docs/`. Each framework has its own folder (kebab-case), and multi-language frameworks have `python/`, `typescript/`, `dotnet/` or `go/` subfolders where applicable.

Frontmatter schema:

```yaml
---
title: "Page title"
description: "Short description used in search and OG tags"
framework: "openai-agents-sdk"   # optional
language: "python"               # optional: python | typescript | dotnet | go
---
```

## Keeping versions right

The current version of every framework is written in exactly one place: `src/data/versions.json`. Nothing else should hard-code it.

- **Model IDs** work the same way: `%%model=openai%%`, `%%model=anthropic-small%%` and so on, from `models` in `versions.json`.
- **Docs** write `%%version=<package>%%` (e.g. `pip install agent-framework==%%version=agent-framework%%`) and `%%versions-checked%%` for the date. `src/plugins/remark-versions.mjs` fills these in at build time, in prose, inline code, code blocks and links. An unknown package name fails the build. Placeholders don't work in frontmatter, and shouldn't go in headings (the anchor would change on every bump).
- **Framework cards** on the home page read `versions.json` directly via `src/data/frameworks.ts`.
- Literal numbers are only for history: "added in 1.17.0", changelog entries, migration notes.

When a framework releases:

```bash
npm run check:latest                              # which packages are behind PyPI / npm / NuGet / Go?
node scripts/check-versions.mjs --latest --write  # bump versions.json; every page follows on the next build
npm run check:versions                            # fails on hard-coded pins or unknown placeholders
```

Then read the changelog and fix any examples a breaking change affects. The `Check guide versions` workflow runs `check:versions` on pull requests and `check:latest` weekly.

## Deploy

Pushes to `main` trigger `.github/workflows/deploy.yml`, which builds with Astro and publishes to GitHub Pages via `actions/deploy-pages`. The site is served from `https://codehalwell.github.io/AgentGuides/`.

## License

MIT for code examples. Documentation is CC BY 4.0 unless noted otherwise.

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

## Deploy

Pushes to `main` trigger `.github/workflows/deploy.yml`, which builds with Astro and publishes to GitHub Pages via `actions/deploy-pages`. The site is served from `https://codehalwell.github.io/AgentGuides/`.

## License

MIT for code examples. Documentation is CC BY 4.0 unless noted otherwise.

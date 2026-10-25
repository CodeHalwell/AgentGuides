---
title: "About Agent Guides"
description: "What these guides cover, how they are kept current, and how to report a mistake or contribute."
---

Agent Guides is documentation for building AI agents with the main open-source and cloud frameworks, in Python, TypeScript, .NET and Go. Each framework has its own section with an overview, a comprehensive guide, recipes and a production guide; larger frameworks add topic pages for areas such as memory, workflows or deployment.

## Where to start

- [Quick Start](/quick-start/) has a short, runnable first agent for five frameworks.
- [Framework Comparison](/frameworks/) explains how the frameworks differ and which to pick for which job.
- [All Guides](/guides/) lists every page, grouped as in the sidebar, with the package version each guide targets.
- The [home page](/) has a use-case picker that suggests frameworks for what you want to build.

## How a guide is organised

The layout varies a little by framework, because the frameworks differ, but most guides follow this shape:

1. **Overview.** What the framework is for, the current version, how to install it, and a reading order (labelled Zero to Hero on most overviews).
2. **Comprehensive guide.** The main reference: concepts, the public API and worked examples.
3. **Topic pages.** Deeper treatment of one area, such as memory, streaming, workflows or the CLI.
4. **Recipes.** Self-contained examples for common tasks.
5. **Production guide.** Limits, timeouts, secrets, observability, testing and deployment.
6. **Diagrams.** Architecture and control-flow pictures.

Where a framework has several languages, each language has its own subsection.

## How the guides are kept accurate

- **One record of versions.** Current package versions and model IDs are written in a single data file, [`src/data/versions.json`](https://github.com/CodeHalwell/AgentGuides/blob/main/src/data/versions.json). Pages refer to it by placeholder and the site fills the values in when it is built, so a version bump is a one-line change. The date of the last check is shown on the pages that use it.
- **Examples are run.** Each example is meant to work against the release the guide names. Authors install the package, check imports and signatures against the installed source, and run the code with a mock or test model where one is possible. Examples that need a live service or an account are marked as such.
- **No invented APIs.** If a framework lacks a feature, the guide says so rather than describing something similar.
- **History is separate.** Literal version numbers appear only for history, such as when a feature was added, in migration notes and in revision-history tables. The [Updates](/updates/april-2026-summary/) section records past changes.

Frameworks change quickly. If a page and a release disagree, trust the release's own documentation and please report the page.

## Style

The guides use British English and aim to be plain and understated. They describe trade-offs as well as features, and avoid claims about popularity or learning time that cannot be checked.

## Reporting a mistake and contributing

The source is at [github.com/CodeHalwell/AgentGuides](https://github.com/CodeHalwell/AgentGuides). To report an error, open an issue naming the page and the release you are using. To contribute a fix or an improvement, follow the [contribution guidelines](https://github.com/CodeHalwell/AgentGuides/blob/main/README.md#contributing).

New material should improve an existing page rather than add a separate one: what you learn from studying one class or API belongs in the section of the guide where a reader would look for it.

---
layout: default
title: Haystack Observability & Error Recovery (Python)
description: Tracing pipelines, retries, fallbacks, and dead letters in Haystack.
---

# Haystack Observability & Error Recovery (Python)


Latest: 2.19.0
Upstream: https://github.com/deepset-ai/haystack/releases | https://haystack.deepset.ai/release-notes

## Pipelines
- Add retries around retrievers/generators; use fallbacks
- Emit traces per node; correlate with run ID

## Dead Letters
- On repeated failures, route to a handler that persists context

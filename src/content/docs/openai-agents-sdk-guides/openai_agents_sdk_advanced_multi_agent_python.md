---
title: "OpenAI Agents SDK Advanced Multi‑Agent (Python)"
description: "Advanced multi-agent coordination, handoffs, and guardrails in Python."
framework: openai-agents-sdk
---

# OpenAI Agents SDK Advanced Multi‑Agent (Python)

Last verified: 2025-11
Upstream: https://github.com/openai/openai-python | https://github.com/openai/openai-agents (if available)

## Patterns
- Role routing and capability routing
- Handoff to human or specialized agent
- Guardrails before/after steps

## Skeleton

```python
class Router:
    def pick(self, msg: str) -> str:
        if "search" in msg: return "researcher"
        return "writer"
```

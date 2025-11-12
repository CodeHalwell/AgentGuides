---
layout: default
title: AutoGen Observability & Recovery (Python)
description: Instrumenting group chats, retries, and HITL for AutoGen.
---

# AutoGen Observability & Recovery (Python)

Last verified: 2025-11
Upstream: https://github.com/microsoft/autogen (AG2 successor context) | https://pypi.org/project/autogen/

## Tracing
- Record per-message spans with agent roles and tool calls

## Termination & Recovery
- Define termination criteria; retry on transient failures; escalate to HITL where needed

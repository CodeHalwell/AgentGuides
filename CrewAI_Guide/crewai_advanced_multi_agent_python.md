---
layout: default
title: CrewAI Advanced Multi‑Agent Patterns (Python)
description: Complex team orchestration, error recovery, HITL, and performance with CrewAI.
---

# CrewAI Advanced Multi‑Agent Patterns (Python)

Last verified: 2025-11

## Goals
- Design hierarchical and peer teams with handoffs
- Add retries, timeouts, and compensation for tasks
- Track metrics and optimize performance

## Handoffs

```python
from crewai import Agent, Task, Crew

researcher = Agent(name="Researcher", role="research", backstory="Find sources")
writer = Agent(name="Writer", role="writer", backstory="Draft copy")

research = Task(description="Research topic and cite")
draft = Task(description="Draft concise summary")

crew = Crew(agents=[researcher, writer], tasks=[research, draft])
result = crew.kickoff()
```

Structure tasks so completion of one triggers the handoff to the next. Add guards to escalate to HITL on policy or confidence thresholds.

## Resilience
- Retries on transient tool failures
- Timeouts around external calls
- Compensating actions for side effects

## Observability
- Add logging at task start/finish; include correlation IDs
- Export traces to OpenTelemetry where possible


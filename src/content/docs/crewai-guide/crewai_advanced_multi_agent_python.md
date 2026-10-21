---
title: "CrewAI Advanced Multi‑Agent Patterns (Python)"
description: "Complex team orchestration, error recovery, HITL, and performance with CrewAI."
framework: "crewai"
language: "python"
---

> **Verified against `crewai` %%version=crewai%%** (checked %%versions-checked%%).

## Goals
- Design hierarchical and peer teams with handoffs
- Add retries, timeouts, and compensation for tasks
- Track metrics and optimize performance

## Handoffs

```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(role="Researcher", goal="Find and cite reliable sources", backstory="Methodical analyst.")
writer = Agent(role="Writer", goal="Draft concise copy from research", backstory="Plain-English editor.")

research = Task(
    description="Research {topic} and cite sources.",
    expected_output="Bullet list of findings with URLs.",
    agent=researcher,
)
draft = Task(
    description="Draft a concise summary from the research.",
    expected_output="Three-paragraph summary.",
    agent=writer,
    context=[research],  # explicit handoff
)

crew = Crew(agents=[researcher, writer], tasks=[research, draft], process=Process.sequential)
result = crew.kickoff(inputs={"topic": "agent evaluation"})
```

For a manager-led team, switch to `process=Process.hierarchical` and set `manager_llm=` (or `manager_agent=`); the manager then decides which agent takes each task.

Structure tasks so completion of one triggers the handoff to the next. Add guards to escalate to HITL on policy or confidence thresholds.

## Resilience
- Retries on transient failures: `Agent(max_retry_limit=...)`, and `guardrail` + `guardrail_max_retries` on tasks to re-run output that fails validation
- Timeouts around external calls: `Agent(max_execution_time=...)` caps wall-clock seconds per task
- Tool failures: `tool_failure_policy="raise"` on an agent, task or crew aborts instead of carrying on with a failed tool result
- Compensating actions for side effects: keep them in your own tool code, and use checkpoints (`Crew(checkpoint=...)`) to resume rather than re-run

## Observability
- Add logging at task start/finish; include correlation IDs
- Export traces to OpenTelemetry where possible


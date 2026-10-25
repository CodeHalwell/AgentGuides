---
title: "CrewAI recipes"
description: "Six worked CrewAI examples — a research-to-article crew, a report from CSV data, manager-led support triage, parallel code review, batch processing with kickoff_for_each, and a lead-qualification flow."
framework: "crewai"
language: "python"
sidebar:
  label: "Recipes"
  order: 45
---

> **Verified against `crewai` %%version=crewai%%** (checked %%versions-checked%%). Each recipe was run end to end against that release with a stub LLM, so the wiring is checked; the quality of the output depends on the model you use.

Each recipe is a complete script. They all set the model explicitly; swap in any provider from [LLM providers](/crewai-guide/crewai_llm_providers_python/).

## 1. Research, write, edit

Three agents in sequence, a typed final result and a Markdown file written from it.

```python
from pathlib import Path

from pydantic import BaseModel
from crewai import Agent, Crew, Process, Task, LLM

llm = LLM(model="openai/%%model=openai-small%%")


class Article(BaseModel):
    title: str
    summary: str
    body_markdown: str


researcher = Agent(
    role="Researcher",
    goal="Find accurate, current information about {topic}",
    backstory="Research librarian who always notes where a fact came from.",
    llm=llm,
)
writer = Agent(
    role="Writer",
    goal="Turn research into a clear article for {audience}",
    backstory="Technical journalist who avoids jargon.",
    llm=llm,
)
editor = Agent(
    role="Editor",
    goal="Make the article correct, concise and consistent",
    backstory="Sub-editor with a red pen and a style guide.",
    llm=llm,
)

research = Task(
    description="Research {topic}. List the key facts, open questions and sources.",
    expected_output="A bulleted list of facts, each with a source.",
    agent=researcher,
)
draft = Task(
    description="Write an article on {topic} for {audience} from the research.",
    expected_output="An article of about 600 words in Markdown.",
    agent=writer,
)
final = Task(
    description="Edit the draft. Fix errors, cut repetition and add a one-sentence summary.",
    expected_output="The edited article with a title and a one-sentence summary.",
    agent=editor,
    output_pydantic=Article,
)

crew = Crew(agents=[researcher, writer, editor], tasks=[research, draft, final], process=Process.sequential)
result = crew.kickoff(inputs={"topic": "heat pumps", "audience": "homeowners"})

article: Article = result.pydantic
print(article.title)
print(result.token_usage.total_tokens, "tokens")

out = Path("output") / "heat-pumps.md"
out.parent.mkdir(exist_ok=True)
out.write_text(f"# {article.title}\n\n_{article.summary}_\n\n{article.body_markdown}\n")
```

The Markdown file is written from the typed result on purpose. A task's `output_file` saves the task's *raw* output, so with `output_pydantic=Article` it would contain the JSON, not the article. Use `output_file` on a task whose raw output is the document you want (here, give the editor task no `output_pydantic` and add `output_file="output/{topic}.md"`), and use a separate structured task, or a field such as `body_markdown`, when you need both. `output_file` accepts the same `{placeholders}` as descriptions, and its folder is created if missing.

## 2. A report from CSV data

Load and aggregate the data in Python, pass the figures in as an input, and ask the crew for a typed report. Agents are better at interpreting numbers than at computing them.

```python
import csv
from collections import defaultdict
from pydantic import BaseModel
from crewai import Agent, Crew, Task, LLM

llm = LLM(model="openai/%%model=openai-small%%")


class SalesReport(BaseModel):
    headline: str
    findings: list[str]
    recommendations: list[str]


def summarise_csv(path: str) -> str:
    totals: dict[str, float] = defaultdict(float)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):          # columns: region, month, revenue
            totals[row["region"]] += float(row["revenue"])
    return "\n".join(f"{region}: {revenue:,.0f}" for region, revenue in sorted(totals.items()))


analyst = Agent(
    role="Sales Analyst",
    goal="Explain what the sales figures mean for the business",
    backstory="Commercial analyst who reports to the sales director every month.",
    llm=llm,
)
report = Task(
    description=(
        "These are revenue totals by region for {period}, computed from {data_path}:\n"
        "{figures}\n\n"
        "Identify the main patterns and recommend actions."
    ),
    expected_output="A headline, three to five findings and two or three recommendations.",
    agent=analyst,
    output_pydantic=SalesReport,
)

path = "sales.csv"
result = Crew(agents=[analyst], tasks=[report]).kickoff(
    inputs={"period": "Q3", "data_path": path, "figures": summarise_csv(path)}
)
sales_report: SalesReport = result.pydantic
print(sales_report.headline)
```

For larger datasets, give the agent a tool that runs specific queries rather than pasting rows into the prompt.

## 3. Manager-led support triage

A hierarchical crew: the manager reads the ticket and decides which specialists to involve.

```python
from crewai import Agent, Crew, Process, Task, LLM

llm = LLM(model="openai/%%model=openai-small%%")

billing = Agent(
    role="Billing Specialist",
    goal="Resolve invoice, refund and payment questions",
    backstory="Five years on the billing desk; knows the refund policy by heart.",
    llm=llm,
)
technical = Agent(
    role="Technical Support Engineer",
    goal="Diagnose and fix product problems",
    backstory="Support engineer who reproduces issues before answering.",
    llm=llm,
)
writer = Agent(
    role="Customer Communications",
    goal="Write clear, friendly replies to customers",
    backstory="Writes every customer-facing message the team sends.",
    llm=llm,
)

triage = Task(
    description="Work out what this ticket needs and get it resolved:\n{ticket}",
    expected_output="The resolution, with the facts it relies on.",
)
reply = Task(
    description="Write the reply to the customer based on the resolution.",
    expected_output="A reply email under 150 words.",
)

crew = Crew(
    agents=[billing, technical, writer],
    tasks=[triage, reply],
    process=Process.hierarchical,
    manager_llm=LLM(model="openai/%%model=openai%%"),   # required for hierarchical crews
)
ticket = "I was charged twice this month and now the dashboard won't load."
print(crew.kickoff(inputs={"ticket": ticket}).raw)
```

The tasks have no `agent`; the manager assigns and delegates them. Use a `manager_agent` instead of `manager_llm` when you want to write the manager's role and instructions yourself (see [Multi-agent patterns](/crewai-guide/crewai_advanced_multi_agent_python/#manager-led-crews)).

## 4. Parallel code review

Two reviewers look at the same diff at the same time; a lead merges their findings.

```python
from pydantic import BaseModel
from crewai import Agent, Crew, Task, LLM

llm = LLM(model="openai/%%model=openai-small%%")


class Review(BaseModel):
    verdict: str                  # "approve" or "request_changes"
    blocking: list[str]
    suggestions: list[str]


security = Agent(
    role="Security Reviewer",
    goal="Find security problems in code changes",
    backstory="Application security engineer who has seen every injection bug.",
    llm=llm,
)
maintainability = Agent(
    role="Maintainability Reviewer",
    goal="Find code that will be hard to change later",
    backstory="Staff engineer who owns the team's style guide.",
    llm=llm,
)
lead = Agent(
    role="Review Lead",
    goal="Give the author one clear, prioritised review",
    backstory="Tech lead who decides whether a change can merge.",
    llm=llm,
)

security_review = Task(
    description="Review this diff for security issues:\n{diff}",
    expected_output="A list of issues with line references, or 'none'.",
    agent=security,
    async_execution=True,
)
style_review = Task(
    description="Review this diff for maintainability issues:\n{diff}",
    expected_output="A list of issues with line references, or 'none'.",
    agent=maintainability,
    async_execution=True,
)
summary = Task(
    description="Merge both reviews into one. Separate blocking issues from suggestions.",
    expected_output="A verdict, blocking issues and suggestions.",
    agent=lead,
    context=[security_review, style_review],
    output_pydantic=Review,
)

diff = '''+ query = f"SELECT * FROM users WHERE name = '{name}'"
+ cursor.execute(query)'''
result = Crew(agents=[security, maintainability, lead], tasks=[security_review, style_review, summary]).kickoff(
    inputs={"diff": diff}
)
print(result.pydantic.verdict)
```

## 5. Batch processing

`kickoff_for_each` runs the same crew once per input and returns one `CrewOutput` per item. `akickoff_for_each` does the same concurrently inside an event loop.

```python
import asyncio
from typing import Literal
from pydantic import BaseModel
from crewai import Agent, Crew, Task, LLM

llm = LLM(model="openai/%%model=openai-small%%")


class TicketLabel(BaseModel):
    category: Literal["billing", "bug", "feature_request", "other"]
    urgency: Literal["low", "medium", "high"]


classifier = Agent(
    role="Ticket Classifier",
    goal="Label support tickets consistently",
    backstory="Support operations analyst who maintains the ticket taxonomy.",
    llm=llm,
)
label = Task(
    description="Classify this support ticket:\n{ticket}",
    expected_output="A category and an urgency.",
    agent=classifier,
    output_pydantic=TicketLabel,
)
crew = Crew(agents=[classifier], tasks=[label])

tickets = [
    "The invoice for March is wrong.",
    "Export to CSV crashes the app.",
    "Could you add dark mode?",
]

results = crew.kickoff_for_each(inputs=[{"ticket": t} for t in tickets])
for ticket, result in zip(tickets, results):
    print(result.pydantic.category, result.pydantic.urgency, "-", ticket)


async def label_all() -> list:
    return await crew.akickoff_for_each(inputs=[{"ticket": t} for t in tickets])

print(len(asyncio.run(label_all())))
```

Concurrency multiplies request rate: set `Crew(max_rpm=...)` to stay under your provider's limits.

## 6. Lead qualification flow

A flow scores a lead with one agent, routes on the score in plain Python, and only spends a second LLM call on leads worth contacting.

```python
from typing import Literal
from pydantic import BaseModel, Field
from crewai import Agent, LLM
from crewai.flow import Flow, start, listen, router

llm = LLM(model="openai/%%model=openai-small%%")


class LeadScore(BaseModel):
    score: int = Field(ge=0, le=100)
    reason: str


class LeadState(BaseModel):
    lead: str = ""
    score: int = 0
    reason: str = ""
    email: str = ""


class LeadFlow(Flow[LeadState]):
    @start()
    def score_lead(self):
        scorer = Agent(
            role="Sales Development Rep",
            goal="Judge how well a lead fits our ideal customer profile",
            backstory="SDR at a B2B analytics company selling to mid-sized retailers.",
            llm=llm,
        )
        out = scorer.kickoff(f"Score this lead from 0 to 100:\n{self.state.lead}", response_format=LeadScore)
        self.state.score, self.state.reason = out.pydantic.score, out.pydantic.reason

    @router(score_lead)
    def qualify(self) -> Literal["hot", "cold"]:
        return "hot" if self.state.score >= 70 else "cold"

    @listen("hot")
    def write_email(self):
        writer = Agent(
            role="Account Executive",
            goal="Write short, specific first-contact emails",
            backstory="AE who never sends a template unchanged.",
            llm=llm,
        )
        self.state.email = writer.kickoff(
            f"Write a first email to this lead. Why they fit: {self.state.reason}\n{self.state.lead}"
        ).raw
        return self.state.email

    @listen("cold")
    def archive(self):
        return f"Archived (score {self.state.score}): {self.state.reason}"


print(LeadFlow().kickoff(inputs={"lead": "Head of Data at a 40-store grocery chain, asked for pricing."}))
```

To have a person approve the email before it is sent, add `@human_feedback` to `write_email` ([Flows → Human feedback](/crewai-guide/crewai_flows_guide/#human-feedback--human_feedback)).

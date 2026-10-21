---
title: "smolagents: Practical Recipes & Code Patterns"
description: "Working smolagents recipes — data analysis, web research, reporting, content, code review, multi-agent pipelines, APIs, custom tools, self-correction and error handling."
framework: "smolagents"
language: "python"
---

> Verified against `smolagents` %%version=smolagents%% (checked %%versions-checked%%). Every recipe constructs and runs against the installed package; network-dependent tools are stubbed where noted.

All recipes assume:

```python
from smolagents import CodeAgent, InferenceClientModel, ToolCallingAgent, tool

model = InferenceClientModel()   # or LiteLLMModel(model_id="%%model=openai-small%%"), OpenAIModel(...), ...
```

Two rules catch most first-time errors: `@tool` functions need type hints **and** an `Args:` docstring entry for every parameter; and `Tool` subclasses must mark optional inputs with `"nullable": True`.

---

## Table of Contents

1. [Data Analysis & Processing](#data-analysis--processing)
2. [Web Research & Information Gathering](#web-research--information-gathering)
3. [Business Intelligence & Reporting](#business-intelligence--reporting)
4. [Content Generation](#content-generation)
5. [Code & Software Development](#code--software-development)
6. [Multi-Agent Workflows](#multi-agent-workflows)
7. [API Integration Patterns](#api-integration-patterns)
8. [Custom Tool Creation](#custom-tool-creation)
9. [Advanced Patterns](#advanced-patterns)
10. [Error Handling & Recovery](#error-handling--recovery)

---

## Data Analysis & Processing

### Recipe 1: CSV Data Analysis Agent

The simplest robust approach is to give a `CodeAgent` the DataFrame itself and let it use pandas, rather than wrapping every operation in a tool.

```python
import pandas as pd

sales = pd.DataFrame({
    "month": ["Jan", "Feb", "Mar", "Apr", "May"],
    "region": ["North", "South", "North", "South", "North"],
    "revenue": [100, 150, 200, 180, 220],
    "marketing": [10, 15, 20, 18, 25],
})

analyst = CodeAgent(
    tools=[],
    model=model,
    additional_authorized_imports=["pandas", "numpy"],
    instructions="You are a careful data analyst. Show the numbers you base conclusions on.",
    max_steps=8,
)

answer = analyst.run(
    "Using the DataFrame `df`, report total revenue by region and the correlation "
    "between marketing and revenue. Return a short paragraph.",
    additional_args={"df": sales},
)
print(answer)
```

`additional_args` places Python objects in the agent's namespace without pasting them into the prompt.

### Recipe 2: Time Series Forecasting Agent

Give the agent a well-defined tool for the part that must be exact, and let it handle the explanation.

```python
@tool
def linear_forecast(values: list, periods: int) -> list:
    """Fit a straight line to a series and extend it.

    Args:
        values: Historical numeric values, oldest first.
        periods: How many future periods to forecast.
    """
    n = len(values)
    xs = range(n)
    x_mean, y_mean = (n - 1) / 2, sum(values) / n
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / sum((x - x_mean) ** 2 for x in xs)
    intercept = y_mean - slope * x_mean
    return [round(intercept + slope * (n + i), 2) for i in range(periods)]


forecaster = CodeAgent(tools=[linear_forecast], model=model, max_steps=5)
print(forecaster.run("Monthly sign-ups were 120, 135, 150, 149, 170. Forecast the next three months and comment on the trend."))
```

---

## Web Research & Information Gathering

### Recipe 3: Multi-Source Research Agent

```python
from smolagents import VisitWebpageTool, WebSearchTool   # pip install "smolagents[toolkit]"

researcher = ToolCallingAgent(
    tools=[WebSearchTool(max_results=5), VisitWebpageTool(max_output_length=20_000)],
    model=model,
    max_steps=12,
    max_tool_threads=4,
    instructions="Use at least three independent sources. Cite each claim with its URL.",
)

print(researcher.run("What are the main technical obstacles to sodium-ion batteries in cars?"))
```

`WebSearchTool(engine="bing")` or `engine="exa"` (1.26.0+) switches the search backend.

### Recipe 4: News & Sentiment Analysis Agent

Combine a search tool with a small, deterministic scoring tool so the agent's judgement is anchored:

```python
POSITIVE = {"growth", "record", "beat", "surge", "upgrade"}
NEGATIVE = {"loss", "lawsuit", "recall", "miss", "downgrade"}


@tool
def headline_sentiment(headlines: list) -> dict:
    """Score headlines with a simple keyword lexicon.

    Args:
        headlines: List of headline strings.
    """
    scores = []
    for h in headlines:
        words = set(h.lower().split())
        scores.append(len(words & POSITIVE) - len(words & NEGATIVE))
    return {"mean_score": sum(scores) / max(len(scores), 1), "scores": scores}


news_agent = CodeAgent(tools=[WebSearchTool(), headline_sentiment], model=model, max_steps=6)
print(news_agent.run("Find five recent headlines about Rolls-Royce plc and summarise the sentiment."))
```

---

## Business Intelligence & Reporting

### Recipe 5: Financial Analysis Agent

```python
@tool
def get_financials(ticker: str) -> dict:
    """Return annual revenue and net income in millions for a company.

    Args:
        ticker: Stock ticker symbol, e.g. "ACME".
    """
    # Replace with your data source
    data = {"ACME": {"revenue": [820, 910, 1010], "net_income": [61, 70, 95], "years": [2023, 2024, 2025]}}
    if ticker.upper() not in data:
        raise ValueError(f"No data for {ticker}. Known tickers: {sorted(data)}")
    return data[ticker.upper()]


fin_agent = CodeAgent(
    tools=[get_financials],
    model=model,
    instructions="Compute growth rates and margins yourself in code; do not estimate them.",
)
print(fin_agent.run("Analyse ACME's revenue growth and net margin trend."))
```

### Recipe 6: Sales & CRM Agent

A `Tool` subclass with an enumerated input. Note that every input needs a `description`.

```python
from smolagents import Tool


class CRMQueryTool(Tool):
    name = "crm_query"
    description = "Query the CRM. Returns a list of records as dicts."
    inputs = {
        "query_type": {
            "type": "string",
            "description": "One of: 'open_deals', 'top_customers', 'churn_risk'.",
            "enum": ["open_deals", "top_customers", "churn_risk"],
        },
        "limit": {"type": "integer", "description": "Maximum records to return.", "nullable": True},
    }
    output_type = "array"

    DATA = {
        "open_deals": [{"customer": "Globex", "value": 42_000, "stage": "proposal"}],
        "top_customers": [{"customer": "Initech", "annual_value": 180_000}],
        "churn_risk": [{"customer": "Umbrella", "risk": 0.71}],
    }

    def forward(self, query_type: str, limit: int | None = None):
        rows = self.DATA[query_type]
        return rows[:limit] if limit else rows


crm_agent = ToolCallingAgent(tools=[CRMQueryTool()], model=model)
print(crm_agent.run("Which customers are at risk of churning, and what deals are open?"))
```

---

## Content Generation

### Recipe 7: Documentation Generator Agent

Pass the source in with `additional_args` and let the agent inspect it with the standard library:

```python
source = '''
def transfer(src: str, dst: str, amount: float, *, idempotency_key: str) -> dict:
    """Move money between accounts."""
    ...
'''

doc_agent = CodeAgent(
    tools=[],
    model=model,
    additional_authorized_imports=["ast", "inspect"],
    instructions="Write reference documentation in Markdown: signature, parameters, return value, errors, example.",
)
print(doc_agent.run("Document the functions in `code`.", additional_args={"code": source}))
```

### Recipe 8: Marketing Content Agent

Use a final-answer check to enforce constraints the model tends to ignore:

```python
def within_limits(final_answer, memory, agent=None) -> bool:
    posts = final_answer if isinstance(final_answer, list) else [final_answer]
    too_long = [p for p in posts if len(str(p)) > 280]
    if too_long:
        raise ValueError(f"{len(too_long)} post(s) exceed 280 characters; shorten them.")
    return True


social_agent = CodeAgent(
    tools=[],
    model=model,
    final_answer_checks=[within_limits],
    instructions="Return a Python list of strings, one per post.",
)
print(social_agent.run("Write three launch posts for a reusable coffee cup made from recycled glass."))
```

---

## Code & Software Development

### Recipe 9: Code Review Agent

```python
import ast


@tool
def check_syntax(code: str) -> str:
    """Check Python source for syntax errors.

    Args:
        code: Python source code.
    """
    try:
        ast.parse(code)
        return "No syntax errors."
    except SyntaxError as exc:
        return f"SyntaxError on line {exc.lineno}: {exc.msg}"


@tool
def find_risky_calls(code: str) -> list:
    """List calls that commonly cause security issues (eval, exec, pickle.loads, subprocess with shell=True).

    Args:
        code: Python source code.
    """
    risky = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name in {"eval", "exec", "pickle.loads"} or (
                name.startswith("subprocess.") and any(k.arg == "shell" for k in node.keywords)
            ):
                risky.append(f"line {node.lineno}: {name}")
    return risky


reviewer = ToolCallingAgent(tools=[check_syntax, find_risky_calls], model=model)
print(reviewer.run("Review this code:\n\nimport subprocess\ndef run(cmd):\n    return subprocess.run(cmd, shell=True)\n"))
```

### Recipe 10: Test Generation Agent

Let the agent write tests and *run* them in the interpreter before answering:

```python
function_src = "def slugify(s):\n    return '-'.join(s.lower().split())\n"

test_writer = CodeAgent(
    tools=[],
    model=model,
    instructions=(
        "Write pytest-style test functions for the code in `src`. Execute the code and your tests "
        "in the interpreter (call each test function) and fix failures before returning the test code as a string."
    ),
    max_steps=6,
)
print(test_writer.run("Write tests for slugify.", additional_args={"src": function_src}))
```

---

## Multi-Agent Workflows

### Recipe 11: Research → Analysis → Reporting Pipeline

A manager with managed agents decides when to call each specialist:

```python
from smolagents import VisitWebpageTool, WebSearchTool

web_agent = ToolCallingAgent(
    tools=[WebSearchTool(), VisitWebpageTool()],
    model=model,
    name="web_researcher",
    description="Researches a question on the web and returns findings with URLs.",
    max_steps=10,
)
analyst_agent = CodeAgent(
    tools=[],
    model=model,
    name="analyst",
    description="Analyses figures and text it is given; computes statistics in Python.",
    additional_authorized_imports=["statistics"],
)

manager = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[web_agent, analyst_agent],
    planning_interval=3,
    instructions="Delegate research and analysis, then write the final report yourself.",
)
print(manager.run("Report on the growth of UK public EV chargers over the last three years."))
```

A fixed sequence is plain Python:

```python
notes = web_agent.run("Collect figures on UK public EV charger numbers, 2023-2025, with sources.")
analysis = analyst_agent.run("Compute year-on-year growth from these notes.", additional_args={"notes": notes})
```

### Recipe 12: Parallel Processing with Agent Pool

Agents hold run state, so create one per worker:

```python
from concurrent.futures import ThreadPoolExecutor


def summarise(text: str) -> str:
    agent = CodeAgent(tools=[], model=model, max_steps=3, verbosity_level=0)
    return agent.run("Summarise `text` in two sentences.", additional_args={"text": text})


documents = ["First long document ...", "Second long document ...", "Third long document ..."]
with ThreadPoolExecutor(max_workers=3) as pool:
    summaries = list(pool.map(summarise, documents))
```

---

## API Integration Patterns

### Recipe 13: REST API Agent

Expose specific, safe operations rather than a generic HTTP tool:

```python
import requests

BASE_URL = "https://api.example.com"


@tool
def get_order(order_id: str) -> dict:
    """Fetch an order by ID from the orders API.

    Args:
        order_id: Order identifier such as "ORD-1001".
    """
    resp = requests.get(f"{BASE_URL}/orders/{order_id}", timeout=10)
    if resp.status_code == 404:
        raise ValueError(f"No order {order_id}")
    resp.raise_for_status()
    return resp.json()


api_agent = ToolCallingAgent(tools=[get_order], model=model)
```

A tool that can make arbitrary requests (any URL, any method) lets a prompt-injected agent reach internal services; avoid it.

### Recipe 14: Database Query Agent

```python
import sqlite3

from smolagents import Tool


class SQLiteTool(Tool):
    name = "sql"
    description = (
        "Run a read-only SQL SELECT on the shop database. Tables: "
        "orders(id INTEGER, customer TEXT, total REAL, created DATE). Returns rows as dicts."
    )
    inputs = {"query": {"type": "string", "description": "One SELECT statement."}}
    output_type = "array"

    def __init__(self, db_path: str = ":memory:"):
        super().__init__()
        self.db_path = db_path

    def setup(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(
            "CREATE TABLE IF NOT EXISTS orders(id INTEGER, customer TEXT, total REAL, created DATE);"
            "INSERT INTO orders VALUES (1,'Ada',40.0,'2026-09-01'),(2,'Bo',75.5,'2026-09-03');"
        )

    def forward(self, query: str):
        if not query.lstrip().lower().startswith("select"):
            raise ValueError("Only SELECT statements are allowed.")
        return [dict(r) for r in self.conn.execute(query).fetchall()]


db_agent = CodeAgent(tools=[SQLiteTool()], model=model)
print(db_agent.run("What is the average order value?"))
```

For real databases, connect with a read-only role as well as checking the statement.

---

## Custom Tool Creation

### Recipe 15: PDF Processing Tool

```python
from smolagents import Tool


class PDFTextTool(Tool):
    name = "pdf_text"
    description = "Extract text from a local PDF file. Returns at most `max_chars` characters."
    inputs = {
        "path": {"type": "string", "description": "Path to a PDF file."},
        "max_chars": {"type": "integer", "description": "Maximum characters to return.", "nullable": True},
    }
    output_type = "string"

    def forward(self, path: str, max_chars: int | None = 20_000):
        from pypdf import PdfReader          # pip install pypdf; import inside forward so the tool is portable
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text[: max_chars or 20_000]


pdf_agent = CodeAgent(tools=[PDFTextTool()], model=model)
# pdf_agent.run("Summarise the key obligations in contract.pdf")
```

### Recipe 16: Email Automation Tool

Side-effecting tools should require explicit confirmation. Here the tool only drafts; sending is a separate human step.

```python
from smolagents import Tool


class DraftEmailTool(Tool):
    name = "draft_email"
    description = "Save an email draft for human review. Does not send anything. Returns the draft ID."
    inputs = {
        "to": {"type": "string", "description": "Recipient email address."},
        "subject": {"type": "string", "description": "Subject line."},
        "body": {"type": "string", "description": "Plain-text body."},
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.drafts: list[dict] = []

    def forward(self, to: str, subject: str, body: str):
        if "@" not in to:
            raise ValueError("Recipient must be an email address.")
        self.drafts.append({"to": to, "subject": subject, "body": body})
        return f"draft-{len(self.drafts)}"


drafts = DraftEmailTool()
email_agent = ToolCallingAgent(tools=[drafts], model=model)
email_agent.run("Draft a polite reminder to ada@example.com that invoice 1042 is overdue.")
print(drafts.drafts)
```

---

## Advanced Patterns

### Recipe 17: Self-Correcting Agent

Built-in self-correction comes from two features: errors are shown to the model on the next step, and `final_answer_checks` reject answers that fail validation. No subclassing is needed.

```python
import re


def looks_like_price(final_answer, memory, agent=None) -> bool:
    if not re.search(r"\d", str(final_answer)) or "EUR" not in str(final_answer).upper():
        raise ValueError("The answer must state a numeric amount in EUR.")
    return True


checked_agent = CodeAgent(
    tools=[WebSearchTool()],
    model=model,
    final_answer_checks=[looks_like_price],
    max_steps=8,
)
print(checked_agent.run("What is the current price of one Bitcoin in EUR?"))
```

For a second opinion, add a reviewer as a managed agent and instruct the manager to have its answer reviewed before finishing.

### Recipe 18: Few-Shot Learning Agent

Put examples in `instructions` once instead of overriding `run()`:

```python
examples = """
Examples of the expected answer format:
- Task: Convert 100 USD to EUR -> Answer: "92.00 EUR"
- Task: Convert 50 GBP to USD -> Answer: "63.50 USD"
"""

fx_agent = CodeAgent(tools=[], model=model, instructions=examples.strip())
print(fx_agent.run("Convert 200 JPY to USD at a rate of 0.0067."))
```

To change the system prompt more deeply, edit `agent.prompt_templates["system_prompt"]` (a Jinja template) after construction, or pass `prompt_templates=` when creating the agent.

---

## Error Handling & Recovery

### Recipe 19: Robust Error Handler

```python
import time

from smolagents import AgentError, AgentGenerationError


def run_robustly(agent, task: str, attempts: int = 3, backoff: float = 2.0):
    """Retry provider failures; report step exhaustion instead of hiding it."""
    for attempt in range(1, attempts + 1):
        try:
            result = agent.run(task, return_full_result=True)
        except AgentGenerationError as exc:          # model/API call failed
            if attempt == attempts:
                raise
            time.sleep(backoff ** attempt)
            continue
        if result.state == "success":
            return result.output
        print(f"Attempt {attempt}: hit max_steps after {len(result.steps)} steps")
    return None


robust_agent = CodeAgent(tools=[], model=model, max_steps=6)
print(run_robustly(robust_agent, "Compute the 30th triangular number."))
```

### Recipe 20: Logging & Debugging Agent

Use step callbacks and memory instead of subclassing (agents already use `self.logger` internally, so do not overwrite it):

```python
import json
import logging

from smolagents import ActionStep

logging.basicConfig(filename="agent_debug.log", level=logging.INFO)
log = logging.getLogger("agent.debug")


def log_step(step: ActionStep, agent=None):
    log.info(json.dumps({
        "step": step.step_number,
        "code": step.code_action,
        "observations": (step.observations or "")[:500],
        "error": str(step.error) if step.error else None,
        "tokens": step.token_usage.total_tokens if step.token_usage else None,
    }))


debug_agent = CodeAgent(tools=[], model=model, step_callbacks={ActionStep: log_step}, verbosity_level=2)
debug_agent.run("What is 2 ** 20?")

debug_agent.replay(detailed=True)                 # re-print the run with full prompts
for step in debug_agent.memory.get_full_steps():  # raw dicts for your own analysis
    print(step.keys())
```

For traces across many runs, use OpenTelemetry or Weave as described in the [production guide](/smolagents-guide/smolagents_production_guide/#monitoring--observability).

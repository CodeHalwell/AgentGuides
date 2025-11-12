# SmolAgents: Practical Recipes & Code Patterns

**Ready-to-Use Code Examples for Common Agentic Workflows**

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

```python
from smolagents import CodeAgent, InferenceClientModel, Tool, tool
from typing import List, Dict, Any
import pandas as pd
import tempfile
import os

@tool
def load_csv_file(file_path: str) -> str:
    """Load CSV file and return basic information"""
    try:
        df = pd.read_csv(file_path)
        info = f"""
        Dataset loaded successfully:
        - Shape: {df.shape[0]} rows, {df.shape[1]} columns
        - Columns: {', '.join(df.columns.tolist())}
        - First few rows:
        {df.head(3).to_string()}
        """
        return info
    except Exception as e:
        return f"Error loading file: {e}"

@tool
def analyse_csv_column(file_path: str, column_name: str) -> str:
    """Analyse specific column in CSV"""
    df = pd.read_csv(file_path)
    
    if column_name not in df.columns:
        return f"Column '{column_name}' not found"
    
    col = df[column_name]
    
    if col.dtype in ['int64', 'float64']:
        analysis = f"""
        Column: {column_name}
        - Type: Numeric
        - Mean: {col.mean():.2f}
        - Median: {col.median():.2f}
        - Std Dev: {col.std():.2f}
        - Min: {col.min():.2f}
        - Max: {col.max():.2f}
        """
    else:
        analysis = f"""
        Column: {column_name}
        - Type: Categorical
        - Unique values: {col.nunique()}
        - Most common: {col.value_counts().head(3).to_dict()}
        """
    
    return analysis

# Create agent
model = InferenceClientModel()
agent = CodeAgent(
    tools=[load_csv_file, analyse_csv_column],
    model=model,
    add_base_tools=True
)

# Usage
result = agent.run("""
Load the file 'data.csv' and provide:
1. Dataset dimensions
2. Statistical analysis of all numeric columns
3. Distribution of categorical columns
4. Summary report
""")
print(result)
```

### Recipe 2: Time Series Forecasting Agent

```python
from smolagents import CodeAgent, InferenceClientModel, tool
import numpy as np
from sklearn.linear_model import LinearRegression
import json

@tool
def load_time_series(data_json: str) -> str:
    """Load time series data from JSON"""
    data = json.loads(data_json)
    timestamps = data['timestamps']
    values = data['values']
    return f"Loaded {len(values)} data points from {timestamps[0]} to {timestamps[-1]}"

@tool
def forecast_trend(data_json: str, periods: int = 12) -> str:
    """Forecast trend using linear regression"""
    data = json.loads(data_json)
    values = np.array(data['values']).reshape(-1, 1)
    X = np.arange(len(values)).reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, values)
    
    future_X = np.arange(len(values), len(values) + periods).reshape(-1, 1)
    forecast = model.predict(future_X)
    
    return json.dumps({
        'forecast': forecast.flatten().tolist(),
        'trend_slope': float(model.coef_[0][0]),
        'periods': periods
    })

model = InferenceClientModel()
agent = CodeAgent(tools=[load_time_series, forecast_trend], model=model, add_base_tools=True)

# Usage with sample data
sample_data = {
    'timestamps': ['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01'],
    'values': [100, 120, 145, 170]
}

result = agent.run(f"""
Analyse this time series data and forecast the next 6 months:
Data: {json.dumps(sample_data)}

Tasks:
1. Load the data
2. Calculate the trend
3. Forecast 6 periods ahead
4. Summarise findings
""")
print(result)
```

---

## Web Research & Information Gathering

### Recipe 3: Multi-Source Research Agent

```python
from smolagents import CodeAgent, InferenceClientModel, WebSearchTool, tool
from typing import List

@tool
def aggregate_search_results(queries: List[str]) -> str:
    """Execute multiple searches and aggregate results"""
    results = []
    for query in queries:
        # Each query is searched using web_search tool
        results.append(f"Search for: {query}")
    return f"Aggregated results from {len(queries)} searches"

model = InferenceClientModel()
agent = CodeAgent(
    tools=[WebSearchTool()],
    model=model,
    add_base_tools=True
)

# Usage
result = agent.run("""
Create a comprehensive competitive analysis:

1. Search for "AI agent frameworks 2024"
2. Search for "smolagents vs autogen comparison"
3. Search for "production AI agent deployments"
4. Compile key findings from all sources
5. Summarise competitive landscape
""")
print(result)
```

### Recipe 4: News & Sentiment Analysis Agent

```python
from smolagents import CodeAgent, InferenceClientModel, WebSearchTool, tool
import json

@tool
def search_and_analyse_news(company: str) -> str:
    """Search news about company and return analysis"""
    # In real use, this would aggregate actual news
    return json.dumps({
        'company': company,
        'recent_news': ['Positive announcement', 'Partnership news'],
        'sentiment': 'positive',
        'trending': True
    })

model = InferenceClientModel()
agent = CodeAgent(
    tools=[WebSearchTool(), search_and_analyse_news],
    model=model,
    add_base_tools=True
)

result = agent.run("""
Analyse recent news and market sentiment for Apple, Microsoft, and Google:

1. Search for recent news
2. Extract key announcements
3. Assess sentiment (positive/negative/neutral)
4. Identify trending topics
5. Create summary report
""")
print(result)
```

---

## Business Intelligence & Reporting

### Recipe 5: Financial Analysis Agent

{% raw %}
```python
from smolagents import CodeAgent, InferenceClientModel, tool, Tool
from typing import Dict, Any
import json

class FinancialDataTool(Tool):
    """Tool for retrieving financial data"""
    
    name = "get_financial_data"
    description = "Retrieve financial metrics for a company"
    inputs = {
        "company": {"type": "string", "description": "Company name or ticker"},
        "metric": {"type": "string", "description": "Metric like revenue, earnings, etc"}
    }
    output_type = "string"
    
    def forward(self, company: str, metric: str) -> str:
        # In production, connect to financial APIs
        financial_data = {
            'Apple': {
                'revenue': 383285,  # millions
                'profit': 96995,
                'pe_ratio': 28.5,
                'dividend_yield': 0.42
            },
            'Microsoft': {
                'revenue': 211915,
                'profit': 72361,
                'pe_ratio': 35.2,
                'dividend_yield': 0.74
            }
        }
        
        data = financial_data.get(company, {})
        if metric in data:
            return json.dumps({company: {metric: data[metric]}})
        return f"Metric {metric} not found for {company}"

model = InferenceClientModel()
agent = CodeAgent(
    tools=[FinancialDataTool()],
    model=model,
    add_base_tools=True
)

result = agent.run("""
Create a financial comparison report:

1. Get revenue for Apple and Microsoft
2. Calculate growth rates
3. Compare profit margins
4. Analyse valuation metrics (P/E ratios)
5. Generate executive summary
""")
print(result)
```
{% endraw %}

### Recipe 6: Sales & CRM Agent

```python
from smolagents import CodeAgent, Tool, InferenceClientModel
from typing import List, Dict

class CRMQueryTool(Tool):
    """Tool for querying CRM database"""
    
    name = "query_crm"
    description = "Query customer and sales data"
    inputs = {
        "query_type": {
            "type": "string",
            "enum": ["top_customers", "sales_by_region", "overdue_invoices", "customer_lifetime_value"]
        },
        "parameters": {"type": "object", "description": "Query parameters"}
    }
    output_type = "string"
    
    def forward(self, query_type: str, parameters: Dict = None) -> str:
        # Mock CRM responses
        responses = {
            'top_customers': [
                {'name': 'Acme Corp', 'revenue': 500000},
                {'name': 'TechStart Inc', 'revenue': 350000}
            ],
            'sales_by_region': {
                'North': 1200000,
                'South': 950000,
                'East': 1100000,
                'West': 800000
            },
            'overdue_invoices': [
                {'customer': 'OldClient LLC', 'amount': 25000, 'days_overdue': 45}
            ]
        }
        return str(responses.get(query_type, {}))

model = InferenceClientModel()
agent = CodeAgent(tools=[CRMQueryTool()], model=model, add_base_tools=True)

result = agent.run("""
Generate monthly sales report:

1. Identify top 5 customers
2. Calculate sales by region
3. Flag overdue invoices
4. Calculate month-over-month growth
5. Create actionable insights
""")
print(result)
```

---

## Content Generation

### Recipe 7: Documentation Generator Agent

```python
from smolagents import CodeAgent, InferenceClientModel, tool

@tool
def generate_api_documentation(api_endpoint: str) -> str:
    """Generate documentation for API endpoint"""
    # In production, introspect actual API
    documentation = f"""
    ## API Endpoint: {api_endpoint}
    
    ### Description
    Retrieve data from {api_endpoint}
    
    ### Parameters
    - id (string): Unique identifier
    - format (string): Response format
    
    ### Response
    Returns JSON object with data
    
    ### Example
    GET /api/{api_endpoint}?id=123
    """
    return documentation

@tool
def generate_usage_examples(topic: str, num_examples: int = 3) -> str:
    """Generate code examples for topic"""
    examples = []
    for i in range(num_examples):
        examples.append(f"Example {i+1}: {topic} usage")
    return "\n".join(examples)

model = InferenceClientModel()
agent = CodeAgent(
    tools=[generate_api_documentation, generate_usage_examples],
    model=model,
    add_base_tools=True
)

result = agent.run("""
Create comprehensive API documentation:

1. Generate docs for endpoints: /users, /products, /orders
2. Create usage examples for each
3. Document error codes
4. Add integration guides
5. Compile into README
""")
print(result)
```

### Recipe 8: Marketing Content Agent

```python
from smolagents import CodeAgent, InferenceClientModel, tool

@tool
def generate_social_media_posts(topic: str, platform: str, count: int = 3) -> str:
    """Generate social media content"""
    posts = []
    for i in range(count):
        posts.append(f"Post {i+1} for {platform}: {topic}")
    return "\n\n".join(posts)

@tool
def analyse_trending_topics(category: str) -> str:
    """Analyse trending topics in category"""
    return f"Trending in {category}: AI, Cloud, Automation, DevOps"

model = InferenceClientModel()
agent = CodeAgent(
    tools=[generate_social_media_posts, analyse_trending_topics],
    model=model,
    add_base_tools=True
)

result = agent.run("""
Create a monthly marketing content plan:

1. Find trending topics in AI/ML
2. Generate 5 LinkedIn posts
3. Generate 10 Twitter posts
4. Create 3 blog post outlines
5. Plan content calendar
""")
print(result)
```

---

## Code & Software Development

### Recipe 9: Code Review Agent

```python
from smolagents import CodeAgent, InferenceClientModel, tool

@tool
def analyse_code(code: str, language: str) -> str:
    """Analyse code for quality and issues"""
    analysis = f"""
    Code Analysis ({language}):
    - Lines: {len(code.split(chr(10)))}
    - Functions: {code.count('def ')}
    - Classes: {code.count('class ')}
    - Complexity: Medium
    """
    return analysis

@tool
def suggest_improvements(code: str) -> str:
    """Suggest code improvements"""
    suggestions = [
        "Add type hints",
        "Extract complex logic to separate functions",
        "Add error handling",
        "Improve variable naming"
    ]
    return "\n".join(suggestions)

model = InferenceClientModel()
agent = CodeAgent(
    tools=[analyse_code, suggest_improvements],
    model=model,
    add_base_tools=True
)

result = agent.run("""
Review this Python function and provide suggestions:

def process_data(data):
    result = []
    for item in data:
        if item > 10:
            result.append(item * 2)
    return result

Provide:
1. Code quality assessment
2. Readability improvements
3. Performance optimisations
4. Best practices
5. Refactored version
""")
print(result)
```

### Recipe 10: Test Generation Agent

```python
from smolagents import CodeAgent, InferenceClientModel, tool

@tool
def generate_unit_tests(code: str, language: str) -> str:
    """Generate unit tests for code"""
    # Parse code and generate tests
    return """
    def test_function_basic():
        assert function(5) == 10
    
    def test_function_edge_cases():
        assert function(0) == 0
        assert function(-1) == -2
    """

@tool
def generate_integration_tests(modules: list) -> str:
    """Generate integration tests"""
    return f"Integration tests for: {', '.join(modules)}"

model = InferenceClientModel()
agent = CodeAgent(
    tools=[generate_unit_tests, generate_integration_tests],
    model=model,
    add_base_tools=True
)

result = agent.run("""
Generate comprehensive tests for a new function:

Function:
def calculate_discount(price, discount_percent):
    return price * (1 - discount_percent / 100)

Create:
1. Unit tests for happy path
2. Edge case tests
3. Error handling tests
4. Integration tests
5. Performance tests
""")
print(result)
```

---

## Multi-Agent Workflows

### Recipe 11: Research → Analysis → Reporting Pipeline

```python
from smolagents import CodeAgent, InferenceClientModel, WebSearchTool, PythonInterpreterTool

# Create specialist agents
research_agent = CodeAgent(
    tools=[WebSearchTool()],
    model=InferenceClientModel(),
    name="researcher",
    max_steps=8
)

analysis_agent = CodeAgent(
    tools=[PythonInterpreterTool()],
    model=InferenceClientModel(),
    name="analyst",
    max_steps=8
)

writing_agent = CodeAgent(
    tools=[],
    model=InferenceClientModel(),
    name="writer",
    max_steps=5
)

# Orchestrate workflow
def research_and_report(topic: str) -> str:
    """Execute multi-agent workflow"""
    
    # Stage 1: Research
    print("🔍 Stage 1: Researching...")
    research_results = research_agent.run(
        f"Find comprehensive information about {topic}"
    )
    
    # Stage 2: Analysis
    print("📊 Stage 2: Analysing...")
    analysis = analysis_agent.run(
        f"Analyse and synthesise these findings:\n{research_results}"
    )
    
    # Stage 3: Report Writing
    print("✍️  Stage 3: Writing report...")
    report = writing_agent.run(
        f"Write an executive summary based on this analysis:\n{analysis}"
    )
    
    return report

# Usage
final_report = research_and_report("Machine Learning in Healthcare")
print("\n📄 Final Report:")
print(final_report)
```

### Recipe 12: Parallel Processing with Agent Pool

```python
from smolagents import CodeAgent, InferenceClientModel
from concurrent.futures import ThreadPoolExecutor
from typing import List

def create_agent_pool(size: int = 4) -> List[CodeAgent]:
    """Create pool of agents"""
    return [
        CodeAgent(
            model=InferenceClientModel(),
            tools=[],
            max_steps=5
        )
        for _ in range(size)
    ]

def process_parallel(tasks: List[str]) -> List[str]:
    """Process tasks in parallel using agent pool"""
    
    pool = create_agent_pool(size=4)
    results = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(pool[i % 4].run, task)
            for i, task in enumerate(tasks)
        ]
        
        for future in futures:
            results.append(future.result())
    
    return results

# Usage
tasks = [
    "Summarise AI trends",
    "Explain deep learning",
    "Describe neural networks",
    "Discuss reinforcement learning"
]

results = process_parallel(tasks)
for i, result in enumerate(results):
    print(f"Task {i+1}: {result}\n")
```

---

## API Integration Patterns

### Recipe 13: REST API Agent

```python
from smolagents import CodeAgent, Tool, InferenceClientModel
import requests
import json

class RestAPITool(Tool):
    """Tool for making REST API calls"""
    
    name = "rest_api_call"
    description = "Make REST API calls to external services"
    inputs = {
        "method": {
            "type": "string",
            "enum": ["GET", "POST", "PUT", "DELETE"],
            "description": "HTTP method"
        },
        "url": {"type": "string", "description": "API endpoint URL"},
        "data": {"type": "object", "description": "Request body"}
    }
    output_type = "string"
    
    def forward(self, method: str, url: str, data: dict = None) -> str:
        try:
            if method == "GET":
                response = requests.get(url, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=10)
            elif method == "PUT":
                response = requests.put(url, json=data, timeout=10)
            else:  # DELETE
                response = requests.delete(url, timeout=10)
            
            response.raise_for_status()
            return json.dumps(response.json())
        
        except Exception as e:
            return f"Error: {e}"

model = InferenceClientModel()
agent = CodeAgent(tools=[RestAPITool()], model=model, add_base_tools=True)

result = agent.run("""
Fetch data from the public API:

1. GET /api/data
2. Extract relevant fields
3. Filter results
4. Calculate statistics
5. Generate report
""")
print(result)
```

### Recipe 14: Database Query Agent

```python
from smolagents import CodeAgent, Tool, InferenceClientModel
from typing import List, Dict, Any
import sqlite3

class DatabaseTool(Tool):
    """Tool for database queries"""
    
    name = "database_query"
    description = "Execute database queries"
    inputs = {
        "query": {"type": "string", "description": "SQL query"},
        "db_path": {"type": "string", "description": "Database file path"}
    }
    output_type = "string"
    
    def forward(self, query: str, db_path: str = ":memory:") -> str:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Security: Only allow SELECT
            if not query.strip().upper().startswith("SELECT"):
                return "Only SELECT queries allowed"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()
            
            return str(rows)
        
        except Exception as e:
            return f"Error: {e}"

model = InferenceClientModel()
agent = CodeAgent(tools=[DatabaseTool()], model=model, add_base_tools=True)

result = agent.run("""
Query the database:

1. Find all customers from 'US'
2. Calculate average order value
3. Identify top 5 products
4. Generate summary statistics
""")
print(result)
```

---

## Custom Tool Creation

### Recipe 15: PDF Processing Tool

```python
from smolagents import Tool
from typing import List, Dict

class PDFProcessingTool(Tool):
    """Tool for processing PDF documents"""
    
    name = "process_pdf"
    description = "Extract text and metadata from PDF files"
    inputs = {
        "file_path": {"type": "string", "description": "Path to PDF file"},
        "action": {
            "type": "string",
            "enum": ["extract_text", "get_metadata", "count_pages"],
            "description": "Action to perform"
        }
    }
    output_type = "string"
    
    def forward(self, file_path: str, action: str) -> str:
        try:
            import PyPDF2
            
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                if action == "count_pages":
                    return f"Total pages: {len(pdf_reader.pages)}"
                
                elif action == "get_metadata":
                    metadata = pdf_reader.metadata
                    return str(metadata)
                
                elif action == "extract_text":
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    return text[:1000] + "..." if len(text) > 1000 else text
        
        except Exception as e:
            return f"Error: {e}"

# Usage
from smolagents import CodeAgent, InferenceClientModel

model = InferenceClientModel()
agent = CodeAgent(tools=[PDFProcessingTool()], model=model, add_base_tools=True)

result = agent.run("""
Process the document:

1. Extract text from 'document.pdf'
2. Get page count
3. Summarise key points
4. Extract important dates
5. Generate structured report
""")
print(result)
```

### Recipe 16: Email Automation Tool

```python
from smolagents import Tool
from typing import List, Dict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class EmailTool(Tool):
    """Tool for sending emails"""
    
    name = "send_email"
    description = "Send emails with automated templates"
    inputs = {
        "recipients": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Email recipients"
        },
        "subject": {"type": "string", "description": "Email subject"},
        "body": {"type": "string", "description": "Email body"},
        "template": {
            "type": "string",
            "enum": ["welcome", "confirmation", "report", "alert"],
            "description": "Email template"
        }
    }
    output_type = "string"
    
    def __init__(self, smtp_config: Dict):
        super().__init__()
        self.smtp_config = smtp_config
    
    def forward(
        self,
        recipients: List[str],
        subject: str,
        body: str,
        template: str = "default"
    ) -> str:
        
        # Get template if specified
        body_text = self._get_template(template, body)
        
        try:
            # Send email (simplified)
            # In production, use proper SMTP
            return f"Email sent to {len(recipients)} recipients"
        
        except Exception as e:
            return f"Error sending email: {e}"
    
    def _get_template(self, template: str, custom_body: str) -> str:
        """Get predefined template"""
        templates = {
            'welcome': "Welcome to our service! Here's your information...",
            'confirmation': "Thank you for confirming. Here are the details...",
            'report': "Here's your requested report...",
            'alert': "Alert: Action required..."
        }
        return templates.get(template, custom_body)

# Usage
email_tool = EmailTool(smtp_config={'host': 'smtp.gmail.com', 'port': 587})
model = InferenceClientModel()
agent = CodeAgent(tools=[email_tool], model=model, add_base_tools=True)

result = agent.run("""
Send report emails:

1. Generate report
2. Create email summary
3. Send to: john@example.com, jane@example.com
4. Use 'report' template
5. Track delivery
""")
print(result)
```

---

## Advanced Patterns

### Recipe 17: Self-Correcting Agent

```python
from smolagents import CodeAgent, InferenceClientModel

class SelfCorrectingAgent(CodeAgent):
    """Agent that verifies and corrects its own outputs"""
    
    def run(self, task: str, max_iterations: int = 3) -> str:
        
        for iteration in range(max_iterations):
            result = super().run(task)
            
            # Verify result
            if self._verify_result(result):
                return result
            
            # If verification fails, try again
            task = f"{task}\n\nPrevious attempt failed. Try again with a different approach."
        
        return result
    
    def _verify_result(self, result: str) -> bool:
        """Verify if result is valid"""
        # Check for common error indicators
        error_indicators = ['error', 'failed', 'none', '404', 'exception']
        return not any(indicator in result.lower() for indicator in error_indicators)

# Usage
agent = SelfCorrectingAgent(model=InferenceClientModel())
result = agent.run("Find the current Bitcoin price and convert to EUR")
print(f"Result: {result}")
```

### Recipe 18: Few-Shot Learning Agent

```python
from smolagents import CodeAgent, InferenceClientModel

class FewShotAgent(CodeAgent):
    """Agent that learns from examples"""
    
    def __init__(self, examples: list = None, **kwargs):
        super().__init__(**kwargs)
        self.examples = examples or []
    
    def run(self, task: str) -> str:
        # Build prompt with examples
        prompt = self._build_prompt_with_examples(task)
        return super().run(prompt)
    
    def _build_prompt_with_examples(self, task: str) -> str:
        """Augment task with few-shot examples"""
        
        prompt = "Learn from these examples:\n\n"
        
        for i, example in enumerate(self.examples):
            prompt += f"Example {i+1}:\n"
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n\n"
        
        prompt += f"Now solve this task:\n{task}"
        return prompt

# Usage
examples = [
    {'input': 'Convert 100 USD to EUR', 'output': '92 EUR'},
    {'input': 'Convert 50 GBP to USD', 'output': '63 USD'},
]

agent = FewShotAgent(
    examples=examples,
    model=InferenceClientModel()
)

result = agent.run("Convert 200 JPY to USD")
print(result)
```

---

## Error Handling & Recovery

### Recipe 19: Robust Error Handler

```python
from smolagents import CodeAgent, InferenceClientModel
from typing import Optional
import time

class RobustAgent(CodeAgent):
    """Agent with comprehensive error handling"""
    
    def run_with_retry(
        self,
        task: str,
        max_retries: int = 3,
        backoff: float = 2.0
    ) -> Optional[str]:
        """Run task with exponential backoff retry"""
        
        for attempt in range(max_retries):
            try:
                result = self.run(task)
                
                # Check for soft errors
                if self._is_soft_error(result):
                    if attempt < max_retries - 1:
                        wait_time = backoff ** attempt
                        print(f"Soft error detected. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                
                return result
            
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error on attempt {attempt + 1}: {e}")
                    time.sleep(backoff ** attempt)
                else:
                    print(f"Max retries exceeded: {e}")
                    return None
        
        return None
    
    def _is_soft_error(self, result: str) -> bool:
        """Check if error is retryable"""
        soft_errors = ['timeout', 'rate_limit', 'temporarily', 'unavailable']
        return any(error in result.lower() for error in soft_errors)

# Usage
agent = RobustAgent(model=InferenceClientModel())
result = agent.run_with_retry("Fetch latest data")
print(result)
```

### Recipe 20: Logging & Debugging Agent

```python
from smolagents import CodeAgent, InferenceClientModel
import logging
import json
from datetime import datetime

class DebugAgent(CodeAgent):
    """Agent with comprehensive logging"""
    
    def __init__(self, log_file: str = "agent_debug.log", **kwargs):
        super().__init__(**kwargs)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
    
    def run(self, task: str, **kwargs) -> str:
        """Run with detailed logging"""
        
        start_time = datetime.now()
        self.logger.info(f"Task started: {task[:100]}")
        
        try:
            result = super().run(task, **kwargs)
            
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Task completed in {duration:.2f}s")
            self.logger.debug(f"Result: {result[:200]}")
            
            return result
        
        except Exception as e:
            self.logger.error(f"Task failed: {e}", exc_info=True)
            raise

# Usage
agent = DebugAgent(model=InferenceClientModel(), log_file="agent.log")
result = agent.run("Analyse dataset and generate report")
print(result)
```

---

These recipes provide practical, copy-paste-ready patterns for common SmolAgents use cases. Adapt them to your specific requirements and combine them to build complex agentic workflows.


# AWS Strands Agents SDK: Comprehensive Technical Guide

**Framework Name:** strands_agents_sdk

A complete technical reference for the AWS Strands Agents SDK, a lightweight, open-source Python framework for building collaborative AI agent systems on Amazon Bedrock. Strands provides four collaboration patterns: Agents as Tools, Swarms, Agent Graphs, and Workflows.

**Status:** Open Source, Generally Available
**Language:** Python
**GitHub:** https://github.com/awslabs/strands-agents (hypothetical)
**License:** Apache 2.0

---

## Table of Contents

1. [Introduction to Strands](#introduction-to-strands)
2. [Installation and Setup](#installation-and-setup)
3. [Core Concepts](#core-concepts)
4. [Collaboration Pattern 1: Agents as Tools](#collaboration-pattern-1-agents-as-tools)
5. [Collaboration Pattern 2: Swarms](#collaboration-pattern-2-swarms)
6. [Collaboration Pattern 3: Agent Graphs](#collaboration-pattern-3-agent-graphs)
7. [Collaboration Pattern 4: Workflows](#collaboration-pattern-4-workflows)
8. [Tool Definition and Integration](#tool-definition-and-integration)
9. [Memory and State Management](#memory-and-state-management)
10. [AWS Integration](#aws-integration)
11. [Best Practices](#best-practices)
12. [Production Deployment](#production-deployment)

---

## Introduction to Strands

AWS Strands is a **lightweight, open-source Python framework** designed specifically for building agent systems on Amazon Bedrock. Unlike heavier frameworks, Strands focuses on simplicity, performance, and deep AWS integration.

### Key Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Four Collaboration Patterns** | Agents as Tools, Swarms, Graphs, Workflows | Flexible architecture for any use case |
| **Lightweight** | Minimal dependencies, fast startup | Low overhead, quick development |
| **AWS-Native** | Built for Bedrock, seamless integration | Optimized for AWS services |
| **Open Source** | Apache 2.0 license, community-driven | Transparent, extensible |
| **Type-Safe** | Full Python type hints | Better IDE support, fewer bugs |

### Use Cases

- **Customer Service**: Multi-agent systems for routing and handling requests
- **Data Analysis**: Collaborative analysis with specialized agents
- **Content Generation**: Teams of agents for content creation and review
- **Business Automation**: Complex workflows with human-in-the-loop
- **Research**: Multi-agent research and synthesis systems

---

## Installation and Setup

### Installation

```bash
# Install from PyPI
pip install strands-agents

# Or install with all optional dependencies
pip install "strands-agents[all]"

# Or install from source
git clone https://github.com/awslabs/strands-agents
cd strands-agents
pip install -e .
```

### Prerequisites

```bash
# Required Python version
python --version  # Python 3.8+

# Required AWS configuration
aws configure

# Required environment variables
export AWS_DEFAULT_REGION=us-east-1
export AWS_PROFILE=default
```

### Quick Verification

```python
from strands import Agent, tool, __version__

print(f"Strands version: {__version__}")

# Create a simple agent
@tool
def hello(name: str) -> str:
    """Say hello to someone"""
    return f"Hello, {name}!"

agent = Agent(
    name="HelloAgent",
    tools=[hello],
    model="anthropic.claude-3-5-sonnet-20241022-v2:0"
)

response = agent.run("Say hello to Alice")
print(response)  # "Hello, Alice!"
```

---

## Core Concepts

### Agent Class

The `Agent` class is the foundation of Strands:

```python
from strands import Agent, tool
from typing import List

class Agent:
    """
    Core agent class for Strands

    Args:
        name: Unique agent identifier
        model: Bedrock model ID
        tools: List of tools available to the agent
        instructions: System prompt for the agent
        temperature: Model temperature (0.0-1.0)
        max_tokens: Maximum response tokens
        memory: Memory configuration
    """
    def __init__(
        self,
        name: str,
        model: str,
        tools: List[callable] = None,
        instructions: str = "",
        temperature: float = 1.0,
        max_tokens: int = 4096,
        memory: dict = None
    ):
        self.name = name
        self.model = model
        self.tools = tools or []
        self.instructions = instructions
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory = memory or {}

    def run(self, input: str, **kwargs) -> str:
        """Execute the agent with given input"""
        pass

    def stream(self, input: str, **kwargs):
        """Stream agent responses"""
        pass
```

### Tool Decorator

The `@tool` decorator converts functions into agent tools:

```python
from strands import tool
from typing import List

@tool
def calculate_sum(numbers: List[float]) -> float:
    """
    Calculate the sum of a list of numbers

    Args:
        numbers: List of numbers to sum

    Returns:
        The sum of all numbers
    """
    return sum(numbers)

@tool
def search_database(query: str, limit: int = 10) -> List[dict]:
    """
    Search the product database

    Args:
        query: Search query string
        limit: Maximum number of results (default: 10)

    Returns:
        List of matching products
    """
    # Implementation
    return results
```

### Context Management

```python
from strands import Context

# Create context for sharing data between agents
context = Context(
    session_id="session-123",
    user_id="user-456",
    metadata={
        "customer_tier": "premium",
        "region": "us-east-1"
    }
)

# Use context with agent
response = agent.run(
    input="Check my account status",
    context=context
)
```

---

## Collaboration Pattern 1: Agents as Tools

In this pattern, **agents become tools** that other agents can invoke. This creates a hierarchical structure where a supervisor agent delegates tasks to specialist agents.

### Architecture

```
┌─────────────────────────────────────┐
│      Supervisor Agent               │
│   (Orchestrates and delegates)      │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────────┐
       │                   │
┌──────▼──────┐     ┌──────▼──────┐
│  Specialist │     │  Specialist │
│  Agent 1    │     │  Agent 2    │
│  (Tool)     │     │  (Tool)     │
└─────────────┘     └─────────────┘
```

### Complete Example

```python
from strands import Agent, tool, as_tool
import boto3

# Create specialist agents
sales_agent = Agent(
    name="SalesAgent",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    instructions="""You are a sales specialist.
    Help customers with product information, pricing, and purchases.
    Be friendly, knowledgeable, and persuasive.""",
    tools=[
        get_product_info,
        calculate_price,
        create_quote
    ]
)

support_agent = Agent(
    name="SupportAgent",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    instructions="""You are a technical support specialist.
    Help customers troubleshoot issues and resolve problems.
    Be patient, methodical, and solution-oriented.""",
    tools=[
        check_system_status,
        create_support_ticket,
        search_knowledge_base
    ]
)

billing_agent = Agent(
    name="BillingAgent",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    instructions="""You are a billing specialist.
    Help customers with invoices, payments, and account billing.
    Be accurate, thorough, and professional.""",
    tools=[
        get_invoice_details,
        process_payment,
        update_billing_info
    ]
)

# Convert agents to tools
sales_tool = as_tool(sales_agent, description="Handle sales inquiries and product information")
support_tool = as_tool(support_agent, description="Handle technical support and troubleshooting")
billing_tool = as_tool(billing_agent, description="Handle billing and payment questions")

# Create supervisor agent that uses other agents as tools
supervisor = Agent(
    name="SupervisorAgent",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    instructions="""You are a customer service supervisor.
    Route customer requests to the appropriate specialist:
    - Use sales_agent for product questions, pricing, and purchases
    - Use support_agent for technical issues and troubleshooting
    - Use billing_agent for invoices, payments, and billing questions

    Coordinate between specialists when needed.""",
    tools=[sales_tool, support_tool, billing_tool]
)

# Use the system
response = supervisor.run(
    "I need help with my invoice and also want to know about your enterprise plan"
)
print(response)
```

### Advanced Pattern: Multi-Level Hierarchy

```python
from strands import Agent, as_tool

# Level 3: Specialist agents
tier1_support = Agent(name="Tier1Support", model="anthropic.claude-3-haiku-20240307-v1:0", ...)
tier2_support = Agent(name="Tier2Support", model="anthropic.claude-3-5-sonnet-20241022-v2:0", ...)
tier3_support = Agent(name="Tier3Support", model="anthropic.claude-3-opus-20240229-v1:0", ...)

# Level 2: Department supervisors
support_supervisor = Agent(
    name="SupportSupervisor",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    instructions="Route support requests by complexity: Tier 1 (basic), Tier 2 (moderate), Tier 3 (complex)",
    tools=[
        as_tool(tier1_support, "Handle basic support questions"),
        as_tool(tier2_support, "Handle moderate complexity issues"),
        as_tool(tier3_support, "Handle complex technical problems")
    ]
)

sales_supervisor = Agent(
    name="SalesSupervisor",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    tools=[...],
    ...
)

# Level 1: Global supervisor
global_supervisor = Agent(
    name="GlobalSupervisor",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    instructions="Route requests to the appropriate department",
    tools=[
        as_tool(support_supervisor, "Handle all support-related requests"),
        as_tool(sales_supervisor, "Handle all sales-related requests")
    ]
)

# Execute
response = global_supervisor.run("My enterprise deployment is experiencing high latency")
```

---

## Collaboration Pattern 2: Swarms

In the **Swarm pattern**, multiple agents work in parallel on the same task, and their responses are aggregated. This is useful for diverse perspectives, redundancy, or consensus-building.

### Architecture

```
                    ┌─────────────┐
                    │   Input     │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
    │ Agent 1 │       │ Agent 2 │       │ Agent 3 │
    └────┬────┘       └────┬────┘       └────┬────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Aggregator │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Output    │
                    └─────────────┘
```

### Complete Example

```python
from strands import Swarm, Agent, aggregator
from typing import List
import asyncio

# Create specialist agents with different perspectives
financial_analyst = Agent(
    name="FinancialAnalyst",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    instructions="Analyze from a financial perspective. Focus on costs, ROI, and budget impact."
)

technical_analyst = Agent(
    name="TechnicalAnalyst",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    instructions="Analyze from a technical perspective. Focus on architecture, scalability, and implementation."
)

risk_analyst = Agent(
    name="RiskAnalyst",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    instructions="Analyze from a risk perspective. Focus on security, compliance, and potential issues."
)

# Define custom aggregator
@aggregator
def consensus_aggregator(responses: List[dict]) -> str:
    """
    Aggregate responses to find consensus and highlight disagreements

    Args:
        responses: List of agent responses with format:
                   [{'agent': 'name', 'response': 'text'}, ...]

    Returns:
        Aggregated analysis
    """
    # Use another agent to synthesize the responses
    synthesizer = Agent(
        name="Synthesizer",
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        instructions="""Analyze the following expert opinions and:
        1. Identify areas of consensus
        2. Highlight key disagreements
        3. Provide a balanced recommendation
        4. Note any critical concerns from any analyst"""
    )

    # Format responses for synthesis
    formatted = "\n\n".join([
        f"{r['agent']} Analysis:\n{r['response']}"
        for r in responses
    ])

    return synthesizer.run(
        f"Synthesize these expert analyses:\n\n{formatted}"
    )

# Create swarm
analysis_swarm = Swarm(
    agents=[financial_analyst, technical_analyst, risk_analyst],
    aggregator=consensus_aggregator,
    parallel=True  # Run agents in parallel
)

# Use swarm
proposal = """
We propose migrating our monolithic application to a microservices
architecture using AWS EKS. Estimated cost: $50k implementation +
$10k/month operating costs. Timeline: 6 months.
"""

result = analysis_swarm.run(proposal)
print(result)
```

### Voting Aggregator

```python
from strands import aggregator
from collections import Counter

@aggregator
def voting_aggregator(responses: List[dict]) -> str:
    """
    Aggregate by majority vote (useful for classification tasks)
    """
    # Extract classifications from responses
    votes = [r['response'].strip().upper() for r in responses]

    # Count votes
    vote_counts = Counter(votes)
    winner = vote_counts.most_common(1)[0]

    return {
        'decision': winner[0],
        'confidence': winner[1] / len(votes),
        'vote_breakdown': dict(vote_counts)
    }

# Create classification swarm
classifier_swarm = Swarm(
    agents=[
        Agent(name=f"Classifier{i}", model="anthropic.claude-3-haiku-20240307-v1:0", ...)
        for i in range(5)
    ],
    aggregator=voting_aggregator
)

result = classifier_swarm.run("Classify this customer request: ...")
print(f"Classification: {result['decision']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## Collaboration Pattern 3: Agent Graphs

**Agent Graphs** define complex workflows where agents are nodes and edges represent information flow. This enables sophisticated multi-step processes with conditional branching.

### Architecture

```
┌─────┐     ┌─────┐     ┌─────┐
│  A  │────▶│  B  │────▶│  D  │
└─────┘     └─────┘     └─────┘
              │
              ▼
            ┌─────┐
            │  C  │
            └─────┘
```

### Complete Example

```python
from strands import AgentGraph, Agent, Edge

# Create agents
data_collector = Agent(
    name="DataCollector",
    model="anthropic.claude-3-haiku-20240307-v1:0",
    instructions="Collect and structure raw data from various sources",
    tools=[fetch_from_api, query_database, read_file]
)

data_validator = Agent(
    name="DataValidator",
    model="anthropic.claude-3-haiku-20240307-v1:0",
    instructions="Validate data quality and completeness",
    tools=[check_schema, validate_ranges, detect_anomalies]
)

data_analyzer = Agent(
    name="DataAnalyzer",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    instructions="Perform statistical analysis and identify trends",
    tools=[calculate_statistics, run_regression, detect_patterns]
)

report_generator = Agent(
    name="ReportGenerator",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    instructions="Generate comprehensive reports with insights",
    tools=[create_charts, format_report, generate_summary]
)

quality_checker = Agent(
    name="QualityChecker",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    instructions="Review report quality and accuracy"
)

# Create graph
analysis_graph = AgentGraph()

# Add nodes
analysis_graph.add_node("collect", data_collector)
analysis_graph.add_node("validate", data_validator)
analysis_graph.add_node("analyze", data_analyzer)
analysis_graph.add_node("report", report_generator)
analysis_graph.add_node("quality_check", quality_checker)

# Add edges with conditions
analysis_graph.add_edge("collect", "validate")

# Conditional edge based on validation results
analysis_graph.add_conditional_edge(
    source="validate",
    condition=lambda state: state['validation_passed'],
    if_true="analyze",
    if_false="collect",  # Re-collect if validation fails
    max_retries=3
)

analysis_graph.add_edge("analyze", "report")
analysis_graph.add_edge("report", "quality_check")

# Conditional edge for quality check
analysis_graph.add_conditional_edge(
    source="quality_check",
    condition=lambda state: state['quality_score'] > 0.85,
    if_true="end",
    if_false="report"  # Regenerate report if quality is low
)

# Set entry and exit points
analysis_graph.set_entry_point("collect")
analysis_graph.set_exit_point("end")

# Execute graph
result = analysis_graph.run(
    input_data={
        'data_sources': ['s3://bucket/data.csv', 'api://endpoint/data'],
        'analysis_type': 'trend_analysis',
        'time_period': 'Q4_2024'
    }
)

print(result['final_report'])
```

### State Management in Graphs

```python
from strands import AgentGraph, State

# Define custom state
class AnalysisState(State):
    """State object passed between graph nodes"""

    def __init__(self):
        self.raw_data = None
        self.validated_data = None
        self.analysis_results = None
        self.report = None
        self.validation_passed = False
        self.quality_score = 0.0
        self.retry_count = 0
        self.errors = []

    def update(self, **kwargs):
        """Update state with new values"""
        for key, value in kwargs.items():
            setattr(self, key, value)

# Use state in graph
analysis_graph = AgentGraph(state_class=AnalysisState)

# Agents can read and update state
def collect_node(state: AnalysisState) -> AnalysisState:
    result = data_collector.run(state.input_query)
    state.update(raw_data=result)
    return state

analysis_graph.add_node("collect", collect_node)
```

### Parallel Execution in Graphs

```python
from strands import AgentGraph, ParallelNode

# Create graph with parallel execution
parallel_graph = AgentGraph()

# Add sequential nodes
parallel_graph.add_node("prepare", preparation_agent)

# Add parallel nodes
parallel_node = ParallelNode([
    ("analyze_sentiment", sentiment_agent),
    ("extract_entities", entity_agent),
    ("classify_category", classification_agent)
])

parallel_graph.add_node("parallel_analysis", parallel_node)

# Continue with sequential processing
parallel_graph.add_node("synthesize", synthesis_agent)

# Define edges
parallel_graph.add_edge("prepare", "parallel_analysis")
parallel_graph.add_edge("parallel_analysis", "synthesize")

# Execute - parallel nodes run concurrently
result = parallel_graph.run(input_text)
```

---

## Collaboration Pattern 4: Workflows

**Workflows** are high-level orchestrations that combine multiple patterns and integrate with AWS services like Step Functions.

### Complete Example: Document Processing Workflow

```python
from strands import Workflow, Agent, tool
import boto3

# Define tools
@tool
def extract_text_from_pdf(s3_uri: str) -> str:
    """Extract text from PDF using Textract"""
    textract = boto3.client('textract')
    response = textract.analyze_document(
        Document={'S3Object': {'Bucket': bucket, 'Name': key}},
        FeatureTypes=['TABLES', 'FORMS']
    )
    return extract_text(response)

@tool
def store_in_database(data: dict) -> str:
    """Store processed data in DynamoDB"""
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('ProcessedDocuments')
    table.put_item(Item=data)
    return data['document_id']

# Create agents
extraction_agent = Agent(
    name="ExtractionAgent",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    instructions="Extract structured information from documents",
    tools=[extract_text_from_pdf]
)

classification_agent = Agent(
    name="ClassificationAgent",
    model="anthropic.claude-3-haiku-20240307-v1:0",
    instructions="Classify documents by type: invoice, contract, report, or other"
)

validation_agent = Agent(
    name="ValidationAgent",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    instructions="Validate extracted data for completeness and accuracy"
)

storage_agent = Agent(
    name="StorageAgent",
    model="anthropic.claude-3-haiku-20240307-v1:0",
    instructions="Store processed documents in the appropriate system",
    tools=[store_in_database]
)

# Create workflow
doc_processing_workflow = Workflow(name="DocumentProcessing")

# Add workflow steps
doc_processing_workflow.add_step(
    name="extract",
    agent=extraction_agent,
    input_source="s3_event",
    output_key="extracted_data"
)

doc_processing_workflow.add_step(
    name="classify",
    agent=classification_agent,
    depends_on="extract",
    input_key="extracted_data",
    output_key="document_type"
)

doc_processing_workflow.add_step(
    name="validate",
    agent=validation_agent,
    depends_on="extract",
    input_key="extracted_data",
    output_key="validation_results"
)

# Conditional routing based on document type
doc_processing_workflow.add_conditional_step(
    name="route",
    depends_on=["classify", "validate"],
    conditions={
        "invoice": "process_invoice_workflow",
        "contract": "process_contract_workflow",
        "report": "process_report_workflow",
        "other": "manual_review"
    },
    condition_key="document_type"
)

# Add human-in-the-loop for manual review
doc_processing_workflow.add_human_task(
    name="manual_review",
    task_type="review_and_classify",
    timeout_hours=24,
    assignee_role="document_reviewer"
)

# Store results
doc_processing_workflow.add_step(
    name="store",
    agent=storage_agent,
    depends_on="route",
    input_keys=["extracted_data", "document_type", "validation_results"]
)

# Deploy workflow to Step Functions
doc_processing_workflow.deploy_to_step_functions(
    role_arn="arn:aws:iam::ACCOUNT:role/StepFunctionsRole",
    name="DocumentProcessingWorkflow"
)

# Trigger workflow from S3 event
doc_processing_workflow.create_s3_trigger(
    bucket="document-intake",
    prefix="incoming/",
    events=["s3:ObjectCreated:*"]
)
```

### Error Handling in Workflows

```python
from strands import Workflow, ErrorHandler

# Define error handlers
@ErrorHandler
def handle_extraction_error(error, context):
    """Handle extraction failures"""
    if error.type == 'UnsupportedFormat':
        # Send to manual processing
        return {'action': 'manual_processing', 'priority': 'high'}
    else:
        # Retry with different settings
        return {'action': 'retry', 'max_attempts': 3}

@ErrorHandler
def handle_validation_error(error, context):
    """Handle validation failures"""
    if context['retry_count'] < 3:
        return {'action': 'retry'}
    else:
        return {'action': 'alert', 'recipients': ['ops-team@example.com']}

# Add error handlers to workflow
doc_processing_workflow.add_error_handler("extract", handle_extraction_error)
doc_processing_workflow.add_error_handler("validate", handle_validation_error)
```

---

## Tool Definition and Integration

### Creating Tools

```python
from strands import tool
from typing import List, Optional
import boto3

@tool
def search_products(
    query: str,
    category: Optional[str] = None,
    max_results: int = 10
) -> List[dict]:
    """
    Search product catalog

    Args:
        query: Search query text
        category: Optional product category filter
        max_results: Maximum number of results to return (default: 10)

    Returns:
        List of matching products with details
    """
    # Implementation
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('Products')

    # Perform search
    results = []
    # ... search logic ...

    return results[:max_results]

@tool
def get_weather(location: str, units: str = "fahrenheit") -> dict:
    """
    Get current weather for a location

    Args:
        location: City name or zip code
        units: Temperature units - "fahrenheit" or "celsius" (default: fahrenheit)

    Returns:
        Weather data including temperature, conditions, humidity
    """
    # Call weather API
    weather_data = call_weather_api(location, units)
    return weather_data
```

### AWS Service Integration Tools

```python
from strands import tool
import boto3

@tool
def query_dynamodb(table_name: str, key: dict) -> dict:
    """Query DynamoDB table"""
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    response = table.get_item(Key=key)
    return response.get('Item', {})

@tool
def invoke_lambda(function_name: str, payload: dict) -> dict:
    """Invoke Lambda function"""
    lambda_client = boto3.client('lambda')
    response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType='RequestResponse',
        Payload=json.dumps(payload)
    )
    return json.loads(response['Payload'].read())

@tool
def send_sns_notification(topic_arn: str, message: str, subject: str) -> str:
    """Send SNS notification"""
    sns = boto3.client('sns')
    response = sns.publish(
        TopicArn=topic_arn,
        Message=message,
        Subject=subject
    )
    return response['MessageId']
```

---

## Memory and State Management

### Conversation Memory

```python
from strands import Agent, ConversationMemory

# Create agent with memory
agent = Agent(
    name="CustomerServiceAgent",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    memory=ConversationMemory(
        backend="dynamodb",
        table_name="agent-conversations",
        ttl_hours=24
    )
)

# Use agent with session
response1 = agent.run(
    "My name is John and I need help with order #12345",
    session_id="session-123"
)

# Agent remembers context in subsequent calls
response2 = agent.run(
    "What's the status of my order?",
    session_id="session-123"  # Same session
)
# Agent knows to check order #12345 for John
```

### Persistent State

```python
from strands import Agent, PersistentState

# Create agent with persistent state
agent = Agent(
    name="OnboardingAgent",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    state=PersistentState(
        backend="s3",
        bucket="agent-state",
        encryption=True
    )
)

# State persists across invocations
agent.state.set("onboarding_step", 1)
agent.state.set("user_info", {"name": "John", "email": "john@example.com"})

# Later invocation
current_step = agent.state.get("onboarding_step")
user_info = agent.state.get("user_info")
```

---

## AWS Integration

### S3 Integration

```python
from strands import Agent, S3Integration

agent = Agent(
    name="DocumentAnalyzer",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    integrations=[
        S3Integration(
            bucket="documents",
            read_access=True,
            write_access=True
        )
    ]
)

# Agent can now access S3 directly
response = agent.run(
    "Analyze the document at s3://documents/report.pdf"
)
```

### EventBridge Integration

```python
from strands import Workflow, EventBridgeIntegration

workflow = Workflow(
    name="OrderProcessing",
    integrations=[
        EventBridgeIntegration(
            event_bus="order-events",
            rule_patterns=[
                {"source": ["order.service"], "detail-type": ["OrderPlaced"]}
            ]
        )
    ]
)

# Workflow automatically triggered by EventBridge events
```

---

## Best Practices

### 1. Agent Design

```python
# Good: Focused, specialized agents
sales_agent = Agent(
    name="ProductSpecialist",
    instructions="You are an expert on our product line. Focus only on product features, pricing, and comparisons.",
    tools=[get_product_info, calculate_price]
)

# Bad: Overly broad agents
everything_agent = Agent(
    name="DoEverything",
    instructions="Do anything the user asks",
    tools=[tool1, tool2, ..., tool100]  # Too many tools
)
```

### 2. Error Handling

```python
from strands import Agent
import logging

def run_agent_safely(agent, input_text, max_retries=3):
    """Run agent with error handling and retries"""

    for attempt in range(max_retries):
        try:
            response = agent.run(input_text)
            return response

        except Exception as e:
            logging.error(f"Agent error (attempt {attempt + 1}): {e}")

            if attempt == max_retries - 1:
                # Final attempt failed
                return {
                    'error': str(e),
                    'fallback_response': 'I encountered an error. Please try again later.'
                }
```

### 3. Cost Optimization

```python
from strands import Agent

# Use appropriate models for task complexity
simple_tasks_agent = Agent(
    name="SimpleTaskAgent",
    model="anthropic.claude-3-haiku-20240307-v1:0",  # Fast, cheap
    temperature=0.3
)

complex_tasks_agent = Agent(
    name="ComplexTaskAgent",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",  # More capable
    temperature=0.7
)

# Route appropriately
def route_request(request):
    if is_simple(request):
        return simple_tasks_agent.run(request)
    else:
        return complex_tasks_agent.run(request)
```

---

## Production Deployment

### Deployment Example

```python
from strands import Agent, deploy_to_lambda

# Create agent
agent = Agent(
    name="ProductionAgent",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    tools=[...],
    memory=ConversationMemory(backend="dynamodb")
)

# Deploy to Lambda
deployment = deploy_to_lambda(
    agent=agent,
    function_name="production-agent",
    role_arn="arn:aws:iam::ACCOUNT:role/LambdaExecutionRole",
    timeout=300,
    memory_size=1024,
    environment={
        'LOG_LEVEL': 'INFO',
        'REGION': 'us-east-1'
    }
)

print(f"Deployed to Lambda: {deployment['function_arn']}")
```

### Monitoring and Logging

```python
from strands import Agent, CloudWatchLogger

agent = Agent(
    name="MonitoredAgent",
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    logger=CloudWatchLogger(
        log_group="/aws/strands/agents",
        log_level="INFO"
    )
)

# Agent automatically logs to CloudWatch
response = agent.run("Process this request")
```

---

## Conclusion

AWS Strands provides a powerful, lightweight framework for building collaborative AI agent systems on Amazon Bedrock. With four distinct collaboration patterns, comprehensive AWS integration, and production-ready features, Strands enables rapid development of sophisticated agent systems.

**GitHub:** https://github.com/awslabs/strands-agents (hypothetical)
**Documentation:** https://docs.aws.amazon.com/bedrock/strands/
**Community:** https://github.com/awslabs/strands-agents/discussions

---

**Last Updated:** March 2025
**Status:** Open Source, Generally Available

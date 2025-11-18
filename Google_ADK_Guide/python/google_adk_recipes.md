# Google ADK - Practical Recipes and Examples

**Version:** 1.0  
**Focus:** Real-world implementations and use cases

---

## Table of Contents

1. [Basic Chat Assistant](#basic-chat-assistant)
2. [Web Research Agent](#web-research-agent)
3. [Data Analysis Agent](#data-analysis-agent)
4. [Customer Support System](#customer-support-system)
5. [Content Generation Pipeline](#content-generation-pipeline)
6. [Code Review Agent](#code-review-agent)
7. [Meeting Scheduler](#meeting-scheduler)
8. [Document Processor](#document-processor)
9. [Sales Lead Qualifier](#sales-lead-qualifier)
10. [System Health Monitor](#system-health-monitor)

---

## Basic Chat Assistant

A simple but effective chat assistant for general conversation.

```python
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio

# Create the agent
chat_assistant = Agent(
    name="chat_assistant",
    model="gemini-2.5-flash",
    description="A friendly conversation partner",
    instruction="""You are a helpful, friendly chat assistant. 
    Your job is to have engaging conversations and help with general questions.
    Be warm, personable, and genuinely interested in the user.""",
    temperature=0.7,
    max_total_tokens=2048
)

# Set up runner with session management
session_service = InMemorySessionService()
runner = Runner(
    app_name="chat_app",
    agent=chat_assistant,
    session_service=session_service
)

# Run the chat interface
async def chat_loop():
    """Interactive chat loop."""
    user_id = "user_123"
    
    print("Chat Assistant: Hello! How can I help you today?")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Chat Assistant: Goodbye! It was nice talking to you!")
                break
            
            # Create message
            message = types.Content(
                role='user',
                parts=[types.Part(text=user_input)]
            )
            
            # Get response
            response_text = ""
            async for event in runner.run_async(
                user_id=user_id,
                session_id="main_session",
                new_message=message
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            response_text += part.text
            
            print(f"Chat Assistant: {response_text}\n")
        
        except KeyboardInterrupt:
            print("\nChat Assistant: Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

# Run
# asyncio.run(chat_loop())
```

---

## Web Research Agent

An agent that conducts comprehensive web research on topics.

```python
from google.adk.agents import SequentialAgent, LlmAgent
from google.adk.tools import google_search, url_context
from google.genai import types
import asyncio

# Step 1: Research agent
researcher = LlmAgent(
    name="researcher",
    model="gemini-2.5-pro",
    description="Searches for comprehensive information",
    instruction="""You are a research specialist. Your role is to:
    1. Search for relevant information on the topic
    2. Explore multiple sources and perspectives
    3. Gather comprehensive data
    4. Compile all findings into a structured format""",
    tools=[google_search, url_context]
)

# Step 2: Summariser agent
summariser = LlmAgent(
    name="summariser",
    model="gemini-2.5-flash",
    description="Summarises research findings",
    instruction="""You are an excellent summariser. Your role is to:
    1. Analyse the research provided
    2. Extract key points and insights
    3. Remove redundancy
    4. Create a concise but comprehensive summary"""
)

# Step 3: Analyst agent
analyst = LlmAgent(
    name="analyst",
    model="gemini-2.5-pro",
    description="Provides analysis and insights",
    instruction="""You are a critical analyst. Your role is to:
    1. Analyse the summary for insights
    2. Identify patterns and trends
    3. Highlight important implications
    4. Suggest further areas of research if needed"""
)

# Create workflow
research_pipeline = SequentialAgent(
    name="research_pipeline",
    description="Comprehensive research workflow",
    instruction="Execute research, summarisation, and analysis in sequence",
    sub_agents=[researcher, summariser, analyst]
)

async def conduct_research(topic: str) -> dict:
    """Conduct research on a topic."""
    
    from google.adk import Runner
    from google.adk.sessions import InMemorySessionService
    
    runner = Runner(
        app_name="research_app",
        agent=research_pipeline,
        session_service=InMemorySessionService()
    )
    
    message = types.Content(
        role='user',
        parts=[types.Part(text=f"Research the following topic: {topic}")]
    )
    
    results = {
        "research": "",
        "summary": "",
        "analysis": ""
    }
    
    async for event in runner.run_async(
        user_id="researcher_1",
        session_id="session_1",
        new_message=message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    results["research"] += part.text
    
    return results

# Example usage
# result = asyncio.run(conduct_research("Latest developments in quantum computing"))
```

---

## Data Analysis Agent

An agent that analyses data using BigQuery.

```python
from google.adk import Agent
from google.adk.tools.bigquery_toolset import BigQueryToolset
from pydantic import BaseModel, Field
from typing import List
import json

class DataAnalysisRequest(BaseModel):
    """Request for data analysis."""
    dataset: str = Field(..., description="Dataset name")
    table: str = Field(..., description="Table name")
    question: str = Field(..., description="Analysis question")
    metrics: List[str] = Field(default_factory=list, description="Metrics to calculate")

# Initialize BigQuery tools
bq_tools = BigQueryToolset(project_id="my-gcp-project")

# Create data analyst agent
data_analyst = Agent(
    name="data_analyst",
    model="gemini-2.5-pro",
    description="Analyses data and provides insights",
    instruction="""You are a professional data analyst with expertise in SQL and data insights.
    
    Your responsibilities:
    1. Understand the analysis question thoroughly
    2. Write appropriate SQL queries
    3. Execute queries on the provided dataset
    4. Analyse results for patterns and insights
    5. Present findings in a clear, actionable format
    
    Always:
    - Write efficient SQL queries
    - Include relevant aggregations
    - Consider business context
    - Provide recommendations based on findings""",
    tools=bq_tools.get_tools(),
    temperature=0.3,  # Lower temp for consistency
    max_total_tokens=4096
)

async def analyse_data(request: DataAnalysisRequest) -> dict:
    """Perform data analysis."""
    
    from google.adk import Runner
    from google.adk.sessions import InMemorySessionService
    
    runner = Runner(
        app_name="analytics_app",
        agent=data_analyst,
        session_service=InMemorySessionService()
    )
    
    query_message = f"""Analyse data from {request.dataset}.{request.table}:
    
    Question: {request.question}
    
    Provide:
    1. SQL query you'll execute
    2. Results and key findings
    3. Insights and recommendations
    4. Any caveats or limitations"""
    
    message = types.Content(
        role='user',
        parts=[types.Part(text=query_message)]
    )
    
    response = ""
    async for event in runner.run_async(
        user_id="analyst_1",
        session_id="session_1",
        new_message=message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    response += part.text
    
    return {
        "question": request.question,
        "analysis": response
    }

# Example usage
# result = asyncio.run(analyse_data(DataAnalysisRequest(
#     dataset="analytics",
#     table="sales",
#     question="What are the top 10 products by revenue?"
# )))
```

---

## Customer Support System

A multi-agent customer support system.

```python
from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.genai import types
import asyncio

# Support tier agents
tier1_support = LlmAgent(
    name="tier1_support",
    model="gemini-2.5-flash",
    description="First-line support agent",
    instruction="""You are a first-line support agent. Your role is to:
    1. Listen to the customer's issue
    2. Ask clarifying questions if needed
    3. Provide solutions to common problems
    4. Be empathetic and professional
    5. Escalate complex issues to Tier 2
    
    Common issues you can handle:
    - Password resets
    - Account access issues
    - Basic troubleshooting
    - General questions""",
    tools=[google_search]
)

tier2_support = LlmAgent(
    name="tier2_support",
    model="gemini-2.5-pro",
    description="Technical support specialist",
    instruction="""You are a technical support specialist. Your role is to:
    1. Handle complex technical issues
    2. Provide advanced troubleshooting
    3. Work with system logs and diagnostics
    4. Offer workarounds and solutions
    5. Document solutions for knowledge base""",
    tools=[google_search]
)

manager = LlmAgent(
    name="support_manager",
    model="gemini-2.5-pro",
    description="Routes and manages support requests",
    instruction="""You are a support manager. Your responsibilities:
    1. Assess incoming support requests
    2. Route to appropriate tier
    3. Monitor resolution quality
    4. Escalate if needed
    5. Gather customer feedback""",
    sub_agents=[tier1_support, tier2_support]
)

async def handle_support_request(customer_issue: str, customer_id: str) -> dict:
    """Handle a customer support request."""
    
    from google.adk import Runner
    from google.adk.sessions import FirestoreSessionService
    
    session_service = FirestoreSessionService(project_id="my-gcp-project")
    
    runner = Runner(
        app_name="support_app",
        agent=manager,
        session_service=session_service
    )
    
    message = types.Content(
        role='user',
        parts=[types.Part(text=customer_issue)]
    )
    
    response = ""
    async for event in runner.run_async(
        user_id=customer_id,
        session_id=f"ticket_{customer_id}",
        new_message=message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    response += part.text
    
    return {
        "customer_id": customer_id,
        "issue": customer_issue,
        "resolution": response
    }

# Example usage
# result = asyncio.run(handle_support_request(
#     customer_issue="I can't log into my account after password change",
#     customer_id="cust_12345"
# ))
```

---

## Content Generation Pipeline

A sophisticated content generation system.

```python
from google.adk.agents import SequentialAgent, LlmAgent
from google.adk.tools import google_search
from pydantic import BaseModel, Field
from typing import List

class ContentRequest(BaseModel):
    """Content generation request."""
    topic: str = Field(..., description="Content topic")
    content_type: str = Field(..., description="Type: blog, article, guide, etc")
    target_audience: str = Field(..., description="Target audience")
    tone: str = Field(..., description="Tone: formal, casual, technical, etc")
    length: str = Field(..., description="Length: short, medium, long")

# Research phase
researcher = LlmAgent(
    name="content_researcher",
    model="gemini-2.5-pro",
    description="Researches content topic",
    instruction="""You are a content researcher. Your role is to:
    1. Research the topic thoroughly
    2. Find credible sources
    3. Gather key information
    4. Identify main points and sub-topics
    5. Note important statistics and examples""",
    tools=[google_search]
)

# Outline phase
outliner = LlmAgent(
    name="content_outliner",
    model="gemini-2.5-flash",
    description="Creates content outline",
    instruction="""You are an expert at creating content outlines. Your role is to:
    1. Create a logical structure
    2. Organise information hierarchically
    3. Ensure good flow
    4. Plan for audience engagement
    5. Include CTAs where appropriate"""
)

# Writing phase
writer = LlmAgent(
    name="content_writer",
    model="gemini-2.5-pro",
    description="Writes main content",
    instruction="""You are a professional content writer. Your role is to:
    1. Write engaging, well-structured content
    2. Follow the outline provided
    3. Use appropriate tone and style
    4. Include examples and stories
    5. Optimise for readability"""
)

# Editing phase
editor = LlmAgent(
    name="content_editor",
    model="gemini-2.5-pro",
    description="Edits and polishes content",
    instruction="""You are a professional editor. Your role is to:
    1. Proofread for grammar and spelling
    2. Improve clarity and flow
    3. Check for consistency
    4. Optimise for SEO (keywords, headings)
    5. Ensure brand voice consistency"""
)

# SEO optimiser
seo_optimizer = LlmAgent(
    name="seo_optimizer",
    model="gemini-2.5-flash",
    description="Optimises for search engines",
    instruction="""You are an SEO specialist. Your role is to:
    1. Optimise headline for keywords
    2. Improve meta description
    3. Suggest internal links
    4. Optimise content structure
    5. Ensure readability for search engines"""
)

# Create pipeline
content_pipeline = SequentialAgent(
    name="content_generation_pipeline",
    description="Complete content generation workflow",
    instruction="Execute research, outline, writing, editing, and SEO in sequence",
    sub_agents=[researcher, outliner, writer, editor, seo_optimizer]
)

async def generate_content(request: ContentRequest) -> dict:
    """Generate content through the pipeline."""
    
    from google.adk import Runner
    from google.adk.sessions import InMemorySessionService
    
    runner = Runner(
        app_name="content_gen_app",
        agent=content_pipeline,
        session_service=InMemorySessionService()
    )
    
    query = f"""Generate {request.content_type} content with:
    Topic: {request.topic}
    Target Audience: {request.target_audience}
    Tone: {request.tone}
    Length: {request.length}
    
    Please go through research, outline, writing, editing, and SEO optimisation phases."""
    
    message = types.Content(
        role='user',
        parts=[types.Part(text=query)]
    )
    
    response = ""
    async for event in runner.run_async(
        user_id="content_creator",
        session_id="session_1",
        new_message=message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    response += part.text
    
    return {
        "topic": request.topic,
        "content_type": request.content_type,
        "content": response
    }
```

---

## Code Review Agent

An intelligent code review system.

```python
from google.adk.agents import ParallelAgent, LlmAgent
from pydantic import BaseModel, Field

class CodeReviewRequest(BaseModel):
    """Code review request."""
    code: str = Field(..., description="Code to review")
    language: str = Field(..., description="Programming language")
    focus_areas: List[str] = Field(default_factory=list, description="Focus areas for review")

# Quality reviewer
quality_reviewer = LlmAgent(
    name="quality_reviewer",
    model="gemini-2.5-pro",
    description="Checks code quality",
    instruction="""Review this code for:
    1. Readability and clarity
    2. Variable and function naming
    3. Code organisation
    4. Comments and documentation
    5. Complexity and maintainability"""
)

# Security reviewer
security_reviewer = LlmAgent(
    name="security_reviewer",
    model="gemini-2.5-pro",
    description="Checks for security issues",
    instruction="""Review this code for security:
    1. Input validation
    2. Authentication/authorisation
    3. Encryption and data protection
    4. SQL injection risks
    5. Common vulnerabilities"""
)

# Performance reviewer
performance_reviewer = LlmAgent(
    name="performance_reviewer",
    model="gemini-2.5-pro",
    description="Checks performance",
    instruction="""Review this code for performance:
    1. Algorithmic efficiency
    2. Time complexity
    3. Space complexity
    4. Database queries
    5. Resource usage"""
)

# Create parallel review
code_reviewer = ParallelAgent(
    name="comprehensive_code_reviewer",
    description="Reviews code from multiple perspectives",
    instruction="Run quality, security, and performance reviews in parallel",
    sub_agents=[quality_reviewer, security_reviewer, performance_reviewer]
)

async def review_code(request: CodeReviewRequest) -> dict:
    """Review code comprehensively."""
    
    from google.adk import Runner
    from google.adk.sessions import InMemorySessionService
    
    runner = Runner(
        app_name="code_review_app",
        agent=code_reviewer,
        session_service=InMemorySessionService()
    )
    
    focus = f"\nFocus areas: {', '.join(request.focus_areas)}" if request.focus_areas else ""
    
    query = f"""Review this {request.language} code:{focus}

```{request.language}
{request.code}
```

Provide detailed, constructive feedback."""
    
    message = types.Content(
        role='user',
        parts=[types.Part(text=query)]
    )
    
    response = ""
    async for event in runner.run_async(
        user_id="reviewer",
        session_id="session_1",
        new_message=message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    response += part.text
    
    return {
        "language": request.language,
        "review": response
    }
```

---

## Meeting Scheduler

An agent that schedules meetings.

```python
from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class MeetingRequest(BaseModel):
    """Meeting scheduling request."""
    attendees: List[str] = Field(..., description="Email addresses of attendees")
    topic: str = Field(..., description="Meeting topic")
    duration: int = Field(..., description="Duration in minutes")
    preferred_date: str = Field(..., description="Preferred date")
    timezone: str = Field(default="UTC", description="Timezone")

meeting_scheduler = LlmAgent(
    name="meeting_scheduler",
    model="gemini-2.5-flash",
    instruction="""You are a meeting scheduling assistant. Your role is to:
    1. Parse meeting requirements
    2. Suggest optimal meeting times
    3. Consider time zones
    4. Find available slots
    5. Confirm meeting details"""
)

async def schedule_meeting(request: MeetingRequest) -> dict:
    """Schedule a meeting."""
    
    from google.adk import Runner
    from google.adk.sessions import InMemorySessionService
    
    runner = Runner(
        app_name="scheduler_app",
        agent=meeting_scheduler,
        session_service=InMemorySessionService()
    )
    
    query = f"""Schedule a meeting with:
    Attendees: {', '.join(request.attendees)}
    Topic: {request.topic}
    Duration: {request.duration} minutes
    Preferred Date: {request.preferred_date}
    Timezone: {request.timezone}
    
    Suggest the best meeting time considering all attendees."""
    
    message = types.Content(
        role='user',
        parts=[types.Part(text=query)]
    )
    
    response = ""
    async for event in runner.run_async(
        user_id="scheduler",
        session_id="session_1",
        new_message=message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    response += part.text
    
    return {
        "topic": request.topic,
        "attendees": request.attendees,
        "scheduled_time": response
    }
```

---

## Document Processor

An agent that processes and extracts information from documents.

```python
from google.adk.agents import SequentialAgent, LlmAgent
from pydantic import BaseModel, Field

class DocumentProcessRequest(BaseModel):
    """Document processing request."""
    document_content: str = Field(..., description="Document content")
    document_type: str = Field(..., description="Type: contract, invoice, report, etc")
    extraction_fields: list = Field(..., description="Fields to extract")

# Parser agent
parser = LlmAgent(
    name="document_parser",
    model="gemini-2.5-pro",
    description="Parses document structure",
    instruction="""Parse the document structure:
    1. Identify sections and hierarchy
    2. Locate key information
    3. Note important dates
    4. Extract metadata"""
)

# Extractor agent
extractor = LlmAgent(
    name="information_extractor",
    model="gemini-2.5-pro",
    description="Extracts key information",
    instruction="""Extract the required information:
    1. Find and extract specified fields
    2. Validate extracted data
    3. Format consistently
    4. Note any missing information"""
)

# Validator agent
validator = LlmAgent(
    name="data_validator",
    model="gemini-2.5-flash",
    description="Validates extracted data",
    instruction="""Validate the extracted data:
    1. Check for completeness
    2. Verify data formats
    3. Identify any inconsistencies
    4. Provide quality assessment"""
)

document_processor = SequentialAgent(
    name="document_processor",
    description="Complete document processing workflow",
    instruction="Parse, extract, and validate document data",
    sub_agents=[parser, extractor, validator]
)
```

---

## Sales Lead Qualifier

An agent that qualifies sales leads.

```python
from google.adk.agents import LlmAgent

lead_qualifier = LlmAgent(
    name="lead_qualifier",
    model="gemini-2.5-pro",
    instruction="""You are an expert sales lead qualifier. Evaluate leads based on:
    1. Company size and industry fit
    2. Budget availability indicators
    3. Timeline and urgency
    4. Decision-maker involvement
    5. Competitive landscape
    6. Past engagement signals
    
    Provide a qualification score (1-10) and next steps."""
)

async def qualify_lead(lead_info: str) -> dict:
    """Qualify a sales lead."""
    
    from google.adk import Runner
    from google.adk.sessions import InMemorySessionService
    
    runner = Runner(
        app_name="lead_qualification",
        agent=lead_qualifier,
        session_service=InMemorySessionService()
    )
    
    query = f"""Qualify this sales lead:

{lead_info}

Provide:
1. Qualification score (1-10)
2. Lead quality assessment
3. Recommended next steps
4. Any red flags or concerns"""
    
    message = types.Content(
        role='user',
        parts=[types.Part(text=query)]
    )
    
    response = ""
    async for event in runner.run_async(
        user_id="sales_team",
        session_id="session_1",
        new_message=message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    response += part.text
    
    return {
        "assessment": response
    }
```

---

## System Health Monitor

An agent that monitors system health.

```python
from google.adk.agents import ParallelAgent, LlmAgent
from typing import Dict

performance_monitor = LlmAgent(
    name="performance_monitor",
    model="gemini-2.5-flash",
    instruction="""Monitor performance metrics:
    1. CPU usage
    2. Memory utilisation
    3. Disk space
    4. Network I/O
    Identify any concerning trends."""
)

error_monitor = LlmAgent(
    name="error_monitor",
    model="gemini-2.5-flash",
    instruction="""Monitor error logs:
    1. Error frequency and types
    2. Error severity
    3. Affected components
    4. Impact assessment"""
)

security_monitor = LlmAgent(
    name="security_monitor",
    model="gemini-2.5-pro",
    instruction="""Monitor security metrics:
    1. Authentication failures
    2. Suspicious activities
    3. Permission violations
    4. Security patch status"""
)

health_monitor = ParallelAgent(
    name="system_health_monitor",
    description="Comprehensive system monitoring",
    instruction="Monitor performance, errors, and security in parallel",
    sub_agents=[performance_monitor, error_monitor, security_monitor]
)

async def monitor_system_health(metrics: Dict) -> dict:
    """Monitor system health."""
    
    from google.adk import Runner
    from google.adk.sessions import InMemorySessionService
    import json
    
    runner = Runner(
        app_name="monitoring_app",
        agent=health_monitor,
        session_service=InMemorySessionService()
    )
    
    query = f"""Analyse system health based on these metrics:

{json.dumps(metrics, indent=2)}

Provide:
1. Health status (Healthy/Warning/Critical)
2. Key metrics summary
3. Any concerning trends
4. Recommended actions"""
    
    message = types.Content(
        role='user',
        parts=[types.Part(text=query)]
    )
    
    response = ""
    async for event in runner.run_async(
        user_id="ops_team",
        session_id="session_1",
        new_message=message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    response += part.text
    
    return {
        "health_assessment": response
    }
```

---

## Best Practices for Recipes

### 1. Error Handling
Always wrap agent calls in try-except blocks:

```python
try:
    result = await agent.execute(query)
except TimeoutError:
    return {"error": "Agent execution timed out"}
except Exception as e:
    return {"error": f"Unexpected error: {str(e)}"}
```

### 2. Logging
Log all agent interactions for debugging:

```python
import logging

logger = logging.getLogger(__name__)

logger.info(f"Agent {agent.name} called with query: {query}")
logger.info(f"Agent response: {response}")
```

### 3. Input Validation
Always validate input before passing to agents:

```python
if not query or len(query) > 10000:
    return {"error": "Invalid query"}
```

### 4. Session Management
Use appropriate session services:

```python
# Development
InMemorySessionService()

# Production
FirestoreSessionService(project_id="prod-project")
```

### 5. Cost Monitoring
Track token usage for cost control:

```python
# Use Flash model for simple tasks
model = "gemini-2.5-flash"  # Cheaper

# Use Pro for complex reasoning
model = "gemini-2.5-pro"  # More expensive
```

---

*These recipes provide practical starting points for common ADK use cases. Adapt and extend them for your specific requirements.*

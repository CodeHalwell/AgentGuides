---
title: "Google ADK - Practical Recipes and Examples"
description: "Version: 2.2.0 Focus: Real-world implementations and use cases"
framework: google-adk
language: python
---

# Google ADK - Practical Recipes and Examples

**Version:** Verified against google-adk==2.2.0  
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
11. [SSE Streaming Chat](#sse-streaming-chat)
12. [Resilient Tool Pipeline](#resilient-tool-pipeline)

---

## Basic Chat Assistant

A simple but effective chat assistant for general conversation.

```python
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio

# Create the agent
# NOTE: temperature and max_output_tokens go in generate_content_config, not on the agent directly
chat_assistant = LlmAgent(
    name="chat_assistant",
    model="gemini-2.5-flash",
    description="A friendly conversation partner",
    instruction="""You are a helpful, friendly chat assistant. 
    Your job is to have engaging conversations and help with general questions.
    Be warm, personable, and genuinely interested in the user.""",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=2048,
    ),
)

# Set up runner with session management
session_service = InMemorySessionService()
runner = Runner(
    app_name="chat_app",
    agent=chat_assistant,
    session_service=session_service,
)

async def chat_loop():
    """Interactive chat loop."""
    user_id = "user_123"
    session_id = "main_session"

    # Sessions must be created before runner.run_async is called
    await session_service.create_session(
        app_name="chat_app",
        user_id=user_id,
        session_id=session_id,
    )

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
                role="user",
                parts=[types.Part(text=user_input)],
            )

            # Get response
            response_text = ""
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=message,
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    response_text = "".join(p.text or "" for p in event.content.parts)

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

> **Note:** `SequentialAgent` and `ParallelAgent` are deprecated in google-adk==2.0.0.
> For new projects, prefer the `Workflow` API (see [Resilient Tool Pipeline](#resilient-tool-pipeline)).
> These agents remain functional but the `instruction` field belongs only on `LlmAgent`, not on
> `SequentialAgent`/`ParallelAgent`.

```python
from google.adk.agents import SequentialAgent, LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
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
    tools=[google_search, url_context],
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
    4. Create a concise but comprehensive summary""",
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
    4. Suggest further areas of research if needed""",
)

# Create workflow
# NOTE: SequentialAgent does NOT accept an instruction= argument
research_pipeline = SequentialAgent(
    name="research_pipeline",
    description="Comprehensive research workflow",
    sub_agents=[researcher, summariser, analyst],
)

async def conduct_research(topic: str) -> dict:
    """Conduct research on a topic."""
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="research_app",
        agent=research_pipeline,
        session_service=session_service,
    )

    # Create the session before calling run_async
    await session_service.create_session(
        app_name="research_app",
        user_id="researcher_1",
        session_id="session_1",
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text=f"Research the following topic: {topic}")],
    )

    result_text = ""
    async for event in runner.run_async(
        user_id="researcher_1",
        session_id="session_1",
        new_message=message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            result_text = "".join(p.text or "" for p in event.content.parts)

    return {
        "topic": topic,
        "research": result_text,
    }

# Example usage
# result = asyncio.run(conduct_research("Latest developments in quantum computing"))
```

---

## Data Analysis Agent

An agent that analyses data using BigQuery.

```python
from google.adk.agents import LlmAgent
from google.adk.integrations.bigquery import BigQueryToolset
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
import asyncio

class DataAnalysisRequest(BaseModel):
    """Request for data analysis."""
    dataset: str = Field(..., description="Dataset name")
    table: str = Field(..., description="Table name")
    question: str = Field(..., description="Analysis question")
    metrics: List[str] = Field(default_factory=list, description="Metrics to calculate")

# BigQueryToolset uses Application Default Credentials automatically.
# Do NOT pass project_id here; configure it via gcloud auth or GOOGLE_CLOUD_PROJECT env var.
bq_tools = BigQueryToolset()

# Create data analyst agent
# NOTE: Pass the toolset directly in the tools list — do NOT call bq_tools.get_tools()
data_analyst = LlmAgent(
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
    tools=[bq_tools],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.3,  # Lower temperature for consistency
        max_output_tokens=4096,
    ),
)

async def analyse_data(request: DataAnalysisRequest) -> dict:
    """Perform data analysis."""
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="analytics_app",
        agent=data_analyst,
        session_service=session_service,
    )

    # Create the session before calling run_async
    await session_service.create_session(
        app_name="analytics_app",
        user_id="analyst_1",
        session_id="session_1",
    )

    query_message = f"""Analyse data from {request.dataset}.{request.table}:
    
    Question: {request.question}
    
    Provide:
    1. SQL query you'll execute
    2. Results and key findings
    3. Insights and recommendations
    4. Any caveats or limitations"""

    message = types.Content(
        role="user",
        parts=[types.Part(text=query_message)],
    )

    response = ""
    async for event in runner.run_async(
        user_id="analyst_1",
        session_id="session_1",
        new_message=message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response = "".join(p.text or "" for p in event.content.parts)

    return {
        "question": request.question,
        "analysis": response,
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
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
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
    tools=[google_search],
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
    tools=[google_search],
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
    sub_agents=[tier1_support, tier2_support],
)

async def handle_support_request(customer_issue: str, customer_id: str) -> dict:
    """Handle a customer support request."""
    # DatabaseSessionService persists sessions to a SQLite database.
    # For production on GCP, replace with VertexAiSessionService(project=..., location=...).
    # FirestoreSessionService does NOT exist in google-adk==2.0.0.
    session_service = DatabaseSessionService(db_url="sqlite+aiosqlite:///./adk_support.db")

    runner = Runner(
        app_name="support_app",
        agent=manager,
        session_service=session_service,
    )

    session_id = f"ticket_{customer_id}"

    # Create the session before calling run_async
    await session_service.create_session(
        app_name="support_app",
        user_id=customer_id,
        session_id=session_id,
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text=customer_issue)],
    )

    response = ""
    async for event in runner.run_async(
        user_id=customer_id,
        session_id=session_id,
        new_message=message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response = "".join(p.text or "" for p in event.content.parts)

    return {
        "customer_id": customer_id,
        "issue": customer_issue,
        "resolution": response,
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

> **Note:** `SequentialAgent` is deprecated in google-adk==2.0.0. The `instruction` field is only
> valid on `LlmAgent`. For new projects, prefer the `Workflow` API.

```python
from google.adk.agents import SequentialAgent, LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
from pydantic import BaseModel, Field
import asyncio

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
    tools=[google_search],
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
    5. Include CTAs where appropriate""",
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
    5. Optimise for readability""",
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
    5. Ensure brand voice consistency""",
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
    5. Ensure readability for search engines""",
)

# NOTE: SequentialAgent does NOT accept an instruction= argument
content_pipeline = SequentialAgent(
    name="content_generation_pipeline",
    description="Complete content generation workflow",
    sub_agents=[researcher, outliner, writer, editor, seo_optimizer],
)

async def generate_content(request: ContentRequest) -> dict:
    """Generate content through the pipeline."""
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="content_gen_app",
        agent=content_pipeline,
        session_service=session_service,
    )

    # Create the session before calling run_async
    await session_service.create_session(
        app_name="content_gen_app",
        user_id="content_creator",
        session_id="session_1",
    )

    query = f"""Generate {request.content_type} content with:
    Topic: {request.topic}
    Target Audience: {request.target_audience}
    Tone: {request.tone}
    Length: {request.length}
    
    Please go through research, outline, writing, editing, and SEO optimisation phases."""

    message = types.Content(
        role="user",
        parts=[types.Part(text=query)],
    )

    response = ""
    async for event in runner.run_async(
        user_id="content_creator",
        session_id="session_1",
        new_message=message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response = "".join(p.text or "" for p in event.content.parts)

    return {
        "topic": request.topic,
        "content_type": request.content_type,
        "content": response,
    }
```

---

## Code Review Agent

An intelligent code review system.

> **Note:** `ParallelAgent` is deprecated in google-adk==2.0.0. The `instruction` field is only
> valid on `LlmAgent`. For new projects, prefer the `Workflow` API.

```python
from google.adk.agents import ParallelAgent, LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
import asyncio

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
    5. Complexity and maintainability""",
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
    5. Common vulnerabilities""",
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
    5. Resource usage""",
)

# NOTE: ParallelAgent does NOT accept an instruction= argument
code_reviewer = ParallelAgent(
    name="comprehensive_code_reviewer",
    description="Reviews code from multiple perspectives",
    sub_agents=[quality_reviewer, security_reviewer, performance_reviewer],
)

async def review_code(request: CodeReviewRequest) -> dict:
    """Review code comprehensively."""
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="code_review_app",
        agent=code_reviewer,
        session_service=session_service,
    )

    # Create the session before calling run_async
    await session_service.create_session(
        app_name="code_review_app",
        user_id="reviewer",
        session_id="session_1",
    )

    focus = f"\nFocus areas: {', '.join(request.focus_areas)}" if request.focus_areas else ""

    query = f"""Review this {request.language} code:{focus}

```{request.language}
{request.code}
```

Provide detailed, constructive feedback."""

    message = types.Content(
        role="user",
        parts=[types.Part(text=query)],
    )

    response = ""
    async for event in runner.run_async(
        user_id="reviewer",
        session_id="session_1",
        new_message=message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response = "".join(p.text or "" for p in event.content.parts)

    return {
        "language": request.language,
        "review": response,
    }
```

---

## Meeting Scheduler

An agent that schedules meetings.

```python
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
import asyncio

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
    5. Confirm meeting details""",
)

async def schedule_meeting(request: MeetingRequest) -> dict:
    """Schedule a meeting."""
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="scheduler_app",
        agent=meeting_scheduler,
        session_service=session_service,
    )

    # Create the session before calling run_async
    await session_service.create_session(
        app_name="scheduler_app",
        user_id="scheduler",
        session_id="session_1",
    )

    query = f"""Schedule a meeting with:
    Attendees: {', '.join(request.attendees)}
    Topic: {request.topic}
    Duration: {request.duration} minutes
    Preferred Date: {request.preferred_date}
    Timezone: {request.timezone}
    
    Suggest the best meeting time considering all attendees."""

    message = types.Content(
        role="user",
        parts=[types.Part(text=query)],
    )

    response = ""
    async for event in runner.run_async(
        user_id="scheduler",
        session_id="session_1",
        new_message=message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response = "".join(p.text or "" for p in event.content.parts)

    return {
        "topic": request.topic,
        "attendees": request.attendees,
        "scheduled_time": response,
    }
```

---

## Document Processor

An agent that processes and extracts information from documents.

> **Note:** `SequentialAgent` is deprecated in google-adk==2.0.0. The `instruction` field is only
> valid on `LlmAgent`. For new projects, prefer the `Workflow` API.

```python
from google.adk.agents import SequentialAgent, LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
import asyncio

class DocumentProcessRequest(BaseModel):
    """Document processing request."""
    document_content: str = Field(..., description="Document content")
    document_type: str = Field(..., description="Type: contract, invoice, report, etc")
    extraction_fields: List[str] = Field(..., description="Fields to extract")

# Parser agent
parser = LlmAgent(
    name="document_parser",
    model="gemini-2.5-pro",
    description="Parses document structure",
    instruction="""Parse the document structure:
    1. Identify sections and hierarchy
    2. Locate key information
    3. Note important dates
    4. Extract metadata""",
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
    4. Note any missing information""",
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
    4. Provide quality assessment""",
)

# NOTE: SequentialAgent does NOT accept an instruction= argument
document_processor = SequentialAgent(
    name="document_processor",
    description="Complete document processing workflow",
    sub_agents=[parser, extractor, validator],
)

async def process_document(request: DocumentProcessRequest) -> dict:
    """Process a document and extract structured information."""
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="doc_processor_app",
        agent=document_processor,
        session_service=session_service,
    )

    # Create the session before calling run_async
    await session_service.create_session(
        app_name="doc_processor_app",
        user_id="doc_user",
        session_id="session_1",
    )

    query = f"""Process this {request.document_type}:

{request.document_content}

Extract the following fields: {', '.join(request.extraction_fields)}

Parse the structure, extract the fields, then validate the results."""

    message = types.Content(
        role="user",
        parts=[types.Part(text=query)],
    )

    response = ""
    async for event in runner.run_async(
        user_id="doc_user",
        session_id="session_1",
        new_message=message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response = "".join(p.text or "" for p in event.content.parts)

    return {
        "document_type": request.document_type,
        "extracted_data": response,
    }

# Example usage
# result = asyncio.run(process_document(DocumentProcessRequest(
#     document_content="Invoice #1234 ...",
#     document_type="invoice",
#     extraction_fields=["invoice_number", "total_amount", "due_date"]
# )))
```

---

## Sales Lead Qualifier

An agent that qualifies sales leads.

```python
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio

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
    
    Provide a qualification score (1-10) and next steps.""",
)

async def qualify_lead(lead_info: str) -> dict:
    """Qualify a sales lead."""
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="lead_qualification",
        agent=lead_qualifier,
        session_service=session_service,
    )

    # Create the session before calling run_async
    await session_service.create_session(
        app_name="lead_qualification",
        user_id="sales_team",
        session_id="session_1",
    )

    query = f"""Qualify this sales lead:

{lead_info}

Provide:
1. Qualification score (1-10)
2. Lead quality assessment
3. Recommended next steps
4. Any red flags or concerns"""

    message = types.Content(
        role="user",
        parts=[types.Part(text=query)],
    )

    response = ""
    async for event in runner.run_async(
        user_id="sales_team",
        session_id="session_1",
        new_message=message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response = "".join(p.text or "" for p in event.content.parts)

    return {
        "assessment": response,
    }
```

---

## System Health Monitor

An agent that monitors system health.

> **Note:** `ParallelAgent` is deprecated in google-adk==2.0.0. The `instruction` field is only
> valid on `LlmAgent`. For new projects, prefer the `Workflow` API.

```python
from google.adk.agents import ParallelAgent, LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from typing import Dict
import asyncio
import json

performance_monitor = LlmAgent(
    name="performance_monitor",
    model="gemini-2.5-flash",
    instruction="""Monitor performance metrics:
    1. CPU usage
    2. Memory utilisation
    3. Disk space
    4. Network I/O
    Identify any concerning trends.""",
)

error_monitor = LlmAgent(
    name="error_monitor",
    model="gemini-2.5-flash",
    instruction="""Monitor error logs:
    1. Error frequency and types
    2. Error severity
    3. Affected components
    4. Impact assessment""",
)

security_monitor = LlmAgent(
    name="security_monitor",
    model="gemini-2.5-pro",
    instruction="""Monitor security metrics:
    1. Authentication failures
    2. Suspicious activities
    3. Permission violations
    4. Security patch status""",
)

# NOTE: ParallelAgent does NOT accept an instruction= argument
health_monitor = ParallelAgent(
    name="system_health_monitor",
    description="Comprehensive system monitoring",
    sub_agents=[performance_monitor, error_monitor, security_monitor],
)

async def monitor_system_health(metrics: Dict) -> dict:
    """Monitor system health."""
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="monitoring_app",
        agent=health_monitor,
        session_service=session_service,
    )

    # Create the session before calling run_async
    await session_service.create_session(
        app_name="monitoring_app",
        user_id="ops_team",
        session_id="session_1",
    )

    query = f"""Analyse system health based on these metrics:

{json.dumps(metrics, indent=2)}

Provide:
1. Health status (Healthy/Warning/Critical)
2. Key metrics summary
3. Any concerning trends
4. Recommended actions"""

    message = types.Content(
        role="user",
        parts=[types.Part(text=query)],
    )

    response = ""
    async for event in runner.run_async(
        user_id="ops_team",
        session_id="session_1",
        new_message=message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response = "".join(p.text or "" for p in event.content.parts)

    return {
        "health_assessment": response,
    }
```

---

## SSE Streaming Chat

Real-time token streaming using `RunConfig` with `StreamingMode.SSE`. Partial events carry
incremental text as the model generates it; the final event signals completion. This approach
works with any `LlmAgent` and any runner that supports `run_async`.

```python
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio

# Agent configured for streaming (temperature via generate_content_config)
streaming_agent = LlmAgent(
    name="streaming_assistant",
    model="gemini-2.5-pro",
    description="A chat assistant with SSE streaming enabled",
    instruction="""You are a helpful assistant. Respond clearly and in detail.""",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=4096,
    ),
)

async def stream_chat(user_query: str) -> str:
    """Run a single-turn streamed chat and return the complete response text."""
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="streaming_app",
        agent=streaming_agent,
        session_service=session_service,
    )

    # Create the session before calling run_async
    await session_service.create_session(
        app_name="streaming_app",
        user_id="stream_user",
        session_id="stream_session",
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text=user_query)],
    )

    # RunConfig controls streaming behaviour and safety caps
    cfg = RunConfig(streaming_mode=StreamingMode.SSE, max_llm_calls=50)

    full_response = ""

    async for event in runner.run_async(
        user_id="stream_user",
        session_id="stream_session",
        new_message=message,
        run_config=cfg,
    ):
        if event.partial:
            # Partial events carry incremental text — print immediately for real-time effect
            if event.content and event.content.parts:
                has_text = any(p.text for p in event.content.parts)
                has_function_call = any(p.function_call for p in event.content.parts)
                # Only stream text parts; skip function-call parts to avoid noise
                if has_text and not has_function_call:
                    chunk = "".join(p.text or "" for p in event.content.parts)
                    print(chunk, end="", flush=True)
                    full_response += chunk
        elif event.is_final_response():
            # The final event consolidates the complete response.
            # Because we already printed via partials, just mark completion.
            print()  # newline after streaming completes

    return full_response

async def multi_turn_stream():
    """Interactive multi-turn streaming chat."""
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="streaming_app_multi",
        agent=streaming_agent,
        session_service=session_service,
    )

    await session_service.create_session(
        app_name="streaming_app_multi",
        user_id="user",
        session_id="chat",
    )

    cfg = RunConfig(streaming_mode=StreamingMode.SSE, max_llm_calls=100)

    print("Streaming Chat (type 'exit' to quit)")
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input or user_input.lower() in ("exit", "quit"):
            break

        message = types.Content(
            role="user",
            parts=[types.Part(text=user_input)],
        )

        print("Assistant: ", end="", flush=True)
        async for event in runner.run_async(
            user_id="user",
            session_id="chat",
            new_message=message,
            run_config=cfg,
        ):
            if event.partial and event.content and event.content.parts:
                has_text = any(p.text for p in event.content.parts)
                has_fc = any(p.function_call for p in event.content.parts)
                if has_text and not has_fc:
                    print("".join(p.text or "" for p in event.content.parts), end="", flush=True)
            elif event.is_final_response():
                print()  # newline after each turn

# Example usage
# asyncio.run(stream_chat("Explain the difference between TCP and UDP in detail."))
# asyncio.run(multi_turn_stream())
```

---

## Resilient Tool Pipeline

A data-fetching pipeline built with the `Workflow` API that uses `RetryConfig` for
exponential-backoff retries on transient network errors. This pattern replaces the deprecated
`SequentialAgent`/`ParallelAgent` approach and gives fine-grained control over error handling,
timeouts, and retry logic per node.

```python
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.workflow import Workflow, node, START, RetryConfig
from google.genai import types
import asyncio
import httpx

# ---------------------------------------------------------------------------
# Workflow nodes
# ---------------------------------------------------------------------------

@node(
    retry_config=RetryConfig(
        max_attempts=4,
        initial_delay=1.0,
        backoff_factor=2.0,
        # Retry only on transient network / timeout errors
        exceptions=["httpx.TimeoutException", "httpx.ConnectError"],
    ),
    timeout=30.0,
)
async def fetch_data(node_input: dict) -> dict:
    """Fetch JSON data from a URL with automatic retry on transient errors."""
    url = node_input.get("url") if isinstance(node_input, dict) else str(node_input)
    if not url:
        return {"error": "No URL provided", "data": {}}
    async with httpx.AsyncClient(timeout=25.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return {"url": url, "data": response.json(), "status": response.status_code}


@node(
    retry_config=RetryConfig(
        max_attempts=3,
        initial_delay=0.5,
        backoff_factor=2.0,
        exceptions=["httpx.TimeoutException"],
    ),
    timeout=20.0,
)
async def enrich_data(raw: dict) -> dict:
    """Enrich the fetched data with additional metadata."""
    data = raw.get("data", {})
    # Guard: jsonplaceholder-style endpoints return a list, not a mapping
    base = data if isinstance(data, dict) else {"items": data}
    enriched = {
        **base,
        "source_url": raw.get("url"),
        "record_count": len(data) if isinstance(data, list) else 1,
        "enriched": True,
    }
    return enriched


@node(timeout=10.0)
async def summarise(enriched: dict) -> str:
    """Format the enriched data as a human-readable summary."""
    count = enriched.get("record_count", "unknown")
    source = enriched.get("source_url", "unknown source")
    return f"Fetched {count} record(s) from {source}. Enrichment applied: {enriched.get('enriched')}."


# ---------------------------------------------------------------------------
# Wire nodes into a Workflow (replaces deprecated SequentialAgent)
# ---------------------------------------------------------------------------

pipeline = Workflow(
    name="resilient_data_pipeline",
    edges=[(START, fetch_data, enrich_data, summarise)],
)

# ---------------------------------------------------------------------------
# LlmAgent that drives the workflow and interprets results
# ---------------------------------------------------------------------------

analyst_agent = LlmAgent(
    name="pipeline_analyst",
    model="gemini-2.5-flash",
    description="Runs the data pipeline and explains results",
    instruction="""You are a data pipeline analyst. Use the provided workflow tools to
    fetch, enrich, and summarise data. Explain the results clearly to the user.""",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=2048,
    ),
    # Attach the compiled workflow as a tool the LLM can invoke
    tools=[pipeline],
)

async def run_pipeline(data_url: str) -> str:
    """Run the resilient data pipeline and return the analyst's summary."""
    # InMemoryRunner auto-creates a session — no explicit session_service needed
    runner = InMemoryRunner(agent=analyst_agent)

    events = await runner.run_debug(
        f"Fetch and analyse data from this URL: {data_url}"
    )

    final_text = ""
    for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(p.text or "" for p in event.content.parts)

    return final_text

# Example usage
# result = asyncio.run(run_pipeline("https://jsonplaceholder.typicode.com/posts"))
# print(result)
```

---

## Best Practices for Recipes

### 1. Error Handling
Always wrap agent calls in try-except blocks:

```python
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

async def safe_run(runner: Runner, user_id: str, session_id: str, query: str) -> str:
    message = types.Content(role="user", parts=[types.Part(text=query)])
    try:
        response = ""
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                response = "".join(p.text or "" for p in event.content.parts)
        return response
    except TimeoutError:
        return "Error: Agent execution timed out"
    except Exception as e:
        return f"Error: Unexpected error: {str(e)}"
```

### 2. Logging
Log all agent interactions for debugging:

```python
import logging

logger = logging.getLogger(__name__)

logger.info(f"Agent called with query: {query[:200]}")
logger.info(f"Agent response length: {len(response)} chars")
```

### 3. Input Validation
Always validate input before passing to agents:

```python
def validate_query(query: str) -> str:
    if not query or not query.strip():
        raise ValueError("Query must not be empty")
    if len(query) > 10_000:
        raise ValueError("Query exceeds maximum length of 10,000 characters")
    return query.strip()
```

### 4. Session Management
Use the appropriate session service for your environment:

```python
from google.adk.sessions import InMemorySessionService, DatabaseSessionService

# Development / testing — sessions lost on restart
session_service = InMemorySessionService()

# Staging / production — sessions persisted to a database
# Requires: pip install aiosqlite
session_service = DatabaseSessionService(db_url="sqlite+aiosqlite:///./adk.db")

# GCP production — sessions managed by Vertex AI
# from google.adk.sessions import VertexAiSessionService
# session_service = VertexAiSessionService(project="my-project", location="us-central1")

# NOTE: FirestoreSessionService does NOT exist in google-adk==2.0.0
```

### 5. Cost Monitoring
Track token usage for cost control:

```python
from google.genai import types

# Use Flash model for simple tasks (cheaper)
flash_config = types.GenerateContentConfig(temperature=0.5, max_output_tokens=1024)
simple_agent = LlmAgent(name="simple", model="gemini-2.5-flash",
                        generate_content_config=flash_config)

# Use Pro for complex reasoning (more capable, higher cost)
pro_config = types.GenerateContentConfig(temperature=0.3, max_output_tokens=8192)
complex_agent = LlmAgent(name="complex", model="gemini-2.5-pro",
                         generate_content_config=pro_config)
```

### 6. Temperature and Generation Config
`temperature`, `max_output_tokens`, and other generation parameters belong in
`generate_content_config`, not on the agent constructor directly:

```python
from google.genai import types

# CORRECT
agent = LlmAgent(
    name="my_agent",
    model="gemini-2.5-pro",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=2048,
        top_p=0.95,
    ),
)

# WRONG — these kwargs do not exist on LlmAgent/Agent
# agent = LlmAgent(name="my_agent", model="gemini-2.5-pro", temperature=0.7)
```

---

## OpenAPI Agent

An agent that talks to an external REST API described by an OpenAPI spec, with
API-key authentication and per-request tenant headers.

```python
import asyncio
import yaml
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from google.adk.tools.openapi_tool.auth.auth_helpers import token_to_scheme_credential
from google.adk.agents.readonly_context import ReadonlyContext

# ── Example: Petstore-style spec ──────────────────────────────────────────────
PETSTORE_SPEC = """
openapi: "3.0.0"
info:
  title: Petstore
  version: "1.0"
paths:
  /pets:
    get:
      operationId: listPets
      summary: List all pets
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
      responses:
        "200":
          description: A list of pets
  /pets/{petId}:
    get:
      operationId: showPetById
      summary: Get a pet by ID
      parameters:
        - name: petId
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          description: A single pet
"""

# ── Auth: API key in header ────────────────────────────────────────────────────
scheme, cred = token_to_scheme_credential(
    token_type="apikey",
    location="header",
    name="X-API-Key",
    credential_value="my-secret-api-key",
)

# ── Dynamic per-request headers (e.g. correlation ID) ─────────────────────────
def add_correlation_header(ctx: ReadonlyContext) -> dict[str, str]:
    return {"X-Correlation-ID": ctx.invocation_id[:16]}

toolset = OpenAPIToolset(
    spec_str=PETSTORE_SPEC,
    spec_str_type="yaml",
    auth_scheme=scheme,
    auth_credential=cred,
    header_provider=add_correlation_header,
    tool_name_prefix="petstore_",
    # ssl_verify="/etc/ssl/corp-ca.pem"  # uncomment for corporate TLS proxies
)

agent = LlmAgent(
    name="petstore_agent",
    model="gemini-2.5-flash",
    instruction="Help the user look up and manage pets in the store.",
    tools=[toolset],
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="petstore")
    events = await runner.run_debug("List all the pets available.", user_id="u1", session_id="s1")
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

---

## Observability with Cloud Trace

Wire up Google Cloud Trace before your first runner invocation so every agent,
model, and tool call is captured as a span.

```python
import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.telemetry.google_cloud import get_gcp_exporters
from google.adk.telemetry.setup import maybe_set_otel_providers
from google.genai import types

# Prerequisite: pip install opentelemetry-exporter-gcp-trace

agent = LlmAgent(
    name="traced_agent",
    model="gemini-2.5-flash",
    instruction="Answer questions and use tools as needed.",
)

app_container = App(name="traced_app", root_agent=agent)
runner: Runner | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runner
    # Wire Cloud Trace BEFORE creating the runner
    maybe_set_otel_providers([get_gcp_exporters(enable_cloud_tracing=True)])

    runner = Runner(
        app=app_container,
        session_service=DatabaseSessionService(db_url=os.environ["SESSION_DB_URL"]),
    )
    yield
    await runner.close()


web_app = FastAPI(lifespan=lifespan)


@web_app.post("/chat/{session_id}")
async def chat(session_id: str, body: dict):
    user_id = body.get("user_id", "anon")
    text = body.get("message", "")

    svc = runner.session_service
    if not await svc.get_session(app_name="traced_app", user_id=user_id, session_id=session_id):
        await svc.create_session(app_name="traced_app", user_id=user_id, session_id=session_id)

    reply = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=text)]),
    ):
        if event.is_final_response() and event.content:
            reply = "".join(p.text or "" for p in event.content.parts or [])
    return {"reply": reply}
```

Every `run_async` call emits an `adk.invocation` root span with child spans for each agent, model call, and tool. Spans are visible in the Google Cloud Trace console under your project.

---

## Multi-Model Agent System

Each agent uses a different model selected for cost and capability:

```python
import asyncio
import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search

# Requires: pip install google-adk[extensions]
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
os.environ["OPENAI_API_KEY"] = "sk-..."

# Fast Gemini Flash for routing
triage = LlmAgent(
    name="triage",
    model="gemini-2.5-flash",
    instruction=(
        "Classify the user's question as exactly one of: "
        "'research', 'code', or 'billing'. Reply with only the label."
    ),
)

# Claude for deep research (high-quality reasoning)
researcher = LlmAgent(
    name="researcher",
    model=LiteLlm(model="anthropic/claude-opus-4-5"),
    description="Answers research questions with citations.",
    instruction="Research the question thoroughly and cite at least two sources.",
    tools=[google_search],
)

# GPT-4o for code (strong at coding tasks)
coder = LlmAgent(
    name="coder",
    model=LiteLlm(model="openai/gpt-4o"),
    description="Writes and explains code.",
    instruction="Provide a working Python example and explain each step.",
)

# Cheap GPT-4o-mini for billing lookups (simple task, cost-sensitive)
billing = LlmAgent(
    name="billing",
    model=LiteLlm(model="openai/gpt-4o-mini"),
    description="Handles billing and account enquiries.",
    instruction="Answer billing questions concisely.",
)

root = LlmAgent(
    name="router",
    model="gemini-2.5-flash",
    instruction=(
        "Route the user to the right specialist. "
        "For research questions use `researcher`. "
        "For code questions use `coder`. "
        "For billing questions use `billing`."
    ),
    sub_agents=[researcher, coder, billing],
)


async def main():
    runner = InMemoryRunner(agent=root, app_name="multi_model")
    await runner.session_service.create_session(
        app_name="multi_model", user_id="u1", session_id="s1"
    )

    questions = [
        "How does RAFT consensus work?",
        "Write a Python function to check if a string is a palindrome.",
        "What's included in the Pro plan?",
    ]

    for q in questions:
        events = await runner.run_debug(q, user_id="u1", session_id="s1")
        print(f"Q: {q}")
        print(f"A: {events[-1].content.parts[0].text}\n")


asyncio.run(main())
```

---

## Recipe 13 — Self-Healing API Agent with `ReflectAndRetryToolPlugin`

An agent that automatically retries failed tool calls using structured reflection guidance, including detecting soft failures returned as error dicts.

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.plugins.reflect_retry_tool_plugin import (
    ReflectAndRetryToolPlugin, TrackingScope,
)
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from typing import Any, Optional

# Simulates a flaky third-party payments API
_attempt_count = {}

def charge_card(amount: float, currency: str, card_token: str) -> dict:
    """Charge a payment card."""
    key = card_token
    _attempt_count[key] = _attempt_count.get(key, 0) + 1
    if _attempt_count[key] <= 2:
        return {"status": "error", "error_code": 429, "message": "Rate limit exceeded"}
    return {"status": "success", "charge_id": f"ch_{card_token[:8]}", "amount": amount}

class PaymentsRetryPlugin(ReflectAndRetryToolPlugin):
    """Detects rate-limit responses (status=error) and triggers retry."""

    async def extract_error_from_result(
        self, *, tool: BaseTool, tool_args: dict, tool_context: ToolContext, result: Any
    ) -> Optional[dict]:
        if isinstance(result, dict) and result.get("status") == "error":
            return result
        return None

agent = LlmAgent(
    name="payments_agent",
    model="gemini-2.5-flash",
    instruction=(
        "Process payment requests. Use charge_card to charge the card. "
        "If a charge fails, retry with the same arguments."
    ),
    tools=[charge_card],
)

async def main():
    runner = InMemoryRunner(
        agent=agent,
        app_name="payments",
        plugins=[
            PaymentsRetryPlugin(
                max_retries=4,
                throw_exception_if_retry_exceeded=False,
                tracking_scope=TrackingScope.INVOCATION,
            )
        ],
    )
    await runner.session_service.create_session(
        app_name="payments", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "Charge $49.99 USD to card token 'tok_abc123'.",
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

---

## Recipe 14 — Few-Shot Agent with Dynamic Example Provider

An agent that injects domain-specific few-shot examples based on the user's query, improving accuracy without increasing the base system prompt size.

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.example_tool import ExampleTool
from google.adk.examples.base_example_provider import BaseExampleProvider
from google.adk.examples.example import Example
from google.genai import types

class ISODateProvider(BaseExampleProvider):
    """Shows date-format examples only when the user asks about dates."""

    _DATE_EXAMPLES = [
        Example(
            input=types.Content(
                role="user",
                parts=[types.Part.from_text("Convert '25th December 2024' to ISO format.")],
            ),
            output=[types.Content(
                role="model",
                parts=[types.Part.from_text("2024-12-25")],
            )],
        ),
        Example(
            input=types.Content(
                role="user",
                parts=[types.Part.from_text("What is ISO format for March 3rd, 2025?")],
            ),
            output=[types.Content(
                role="model",
                parts=[types.Part.from_text("2025-03-03")],
            )],
        ),
    ]

    _CURRENCY_EXAMPLES = [
        Example(
            input=types.Content(
                role="user",
                parts=[types.Part.from_text("What is the ISO currency code for British pounds?")],
            ),
            output=[types.Content(
                role="model",
                parts=[types.Part.from_text("GBP")],
            )],
        ),
    ]

    def get_examples(self, query: str) -> list[Example]:
        q = query.lower()
        if any(kw in q for kw in ["date", "iso", "convert", "format", "december", "january"]):
            return self._DATE_EXAMPLES
        if any(kw in q for kw in ["currency", "code", "pound", "dollar", "euro"]):
            return self._CURRENCY_EXAMPLES
        return []

agent = LlmAgent(
    name="standards_agent",
    model="gemini-2.5-flash",
    instruction="Answer questions about ISO standards precisely.",
    tools=[ExampleTool(ISODateProvider())],
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="iso")
    await runner.session_service.create_session(
        app_name="iso", user_id="u1", session_id="s1"
    )
    for q in [
        "Convert 'January 1st, 2026' to ISO date format.",
        "ISO currency code for Japanese yen?",
        "What's the speed of light?",
    ]:
        events = await runner.run_debug(q, user_id="u1", session_id="s1")
        print(f"Q: {q}")
        print(f"A: {events[-1].content.parts[0].text}\n")

asyncio.run(main())
```

---

## Recipe 15 — Tool Environment Simulation for CI Testing

A complete CI-safe test harness that mocks all external tool calls without any real network access, supporting both deterministic injection and LLM-generated mocks.

```python
import asyncio
import json
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.environment_simulation.environment_simulation_config import (
    EnvironmentSimulationConfig, ToolSimulationConfig, InjectionConfig,
    InjectedError, MockStrategy,
)
from google.adk.tools.environment_simulation.environment_simulation_engine import (
    EnvironmentSimulationEngine,
)
from google.adk.tools.environment_simulation.environment_simulation_plugin import (
    EnvironmentSimulationPlugin,
)

# Real tool definitions (would call external APIs in production)
# Return minimal stubs so the agent works even without the simulation layer.
def get_exchange_rate(base: str, target: str) -> dict:
    """Fetch current exchange rate."""
    return {"base": base, "target": target, "rate": 1.0, "source": "stub"}

def get_market_data(symbol: str) -> dict:
    """Fetch market data for a stock symbol."""
    return {"symbol": symbol, "price": 0.0, "change_pct": 0.0, "source": "stub"}

def send_trade_notification(email: str, message: str) -> dict:
    """Send a trade notification email."""
    return {"status": "sent", "email": email, "source": "stub"}

agent = LlmAgent(
    name="trading_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a trading assistant. Check exchange rates and market data, "
        "then send notifications when trades are executed."
    ),
    tools=[get_exchange_rate, get_market_data, send_trade_notification],
)

# CI simulation config — no real API calls
sim_config = EnvironmentSimulationConfig(
    tool_simulation_configs=[
        # Deterministic FX rate injections
        ToolSimulationConfig(
            tool_name="get_exchange_rate",
            injection_configs=[
                InjectionConfig(
                    match_args={"base": "USD", "target": "GBP"},
                    injected_response={"base": "USD", "target": "GBP", "rate": 0.79},
                ),
                InjectionConfig(
                    match_args={"base": "USD", "target": "EUR"},
                    injected_response={"base": "USD", "target": "EUR", "rate": 0.92},
                ),
            ],
            # Fallback for unmatched currency pairs
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
        ),
        # 10% rate-limit errors for chaos testing
        ToolSimulationConfig(
            tool_name="get_market_data",
            injection_configs=[
                InjectionConfig(
                    injection_probability=0.1,
                    random_seed=99,
                    injected_error=InjectedError(
                        injected_http_error_code=429,
                        error_message="Rate limit: retry after 1s",
                    ),
                ),
            ],
            mock_strategy_type=MockStrategy.MOCK_STRATEGY_TOOL_SPEC,
        ),
        # Email always succeeds in tests
        ToolSimulationConfig(
            tool_name="send_trade_notification",
            injection_configs=[
                InjectionConfig(
                    injection_probability=1.0,
                    injected_response={"status": "sent", "message_id": "msg_test_001"},
                )
            ],
        ),
    ],
    simulation_model="gemini-2.0-flash",
    environment_data=json.dumps({
        "stocks": [
            {"symbol": "GOOG", "price": 185.50, "change_pct": 1.2},
            {"symbol": "AAPL", "price": 212.00, "change_pct": -0.5},
        ]
    }),
)

async def main():
    runner = InMemoryRunner(
        agent=agent,
        app_name="trading",
        plugins=[EnvironmentSimulationPlugin(
            simulator_engine=EnvironmentSimulationEngine(sim_config)
        )],
    )
    await runner.session_service.create_session(
        app_name="trading", user_id="u1", session_id="s1"
    )
    events = await runner.run_debug(
        "What's the USD to GBP rate? Also check GOOG market data and "
        "send a notification to trader@example.com.",
        user_id="u1", session_id="s1",
    )
    print(events[-1].content.parts[0].text)

asyncio.run(main())
```

---

## Recipe 16 — Code Analysis Agent with `VertexAiCodeExecutor`

A stateful data analysis agent that executes Python across multiple turns, retaining variables between turns for interactive data exploration.

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.code_executors.vertex_ai_code_executor import VertexAiCodeExecutor

executor = VertexAiCodeExecutor(
    stateful=True,           # variables persist across turns
    optimize_data_file=True, # CSV attachments auto-extracted
    error_retry_attempts=2,
    timeout_seconds=30,
)

agent = LlmAgent(
    name="data_explorer",
    model="gemini-2.5-flash",
    instruction=(
        "You are an interactive data analyst. "
        "Execute Python code to answer user questions about data. "
        "Show your work — write code, execute it, and explain the output. "
        "Variables defined in earlier turns are still available."
    ),
    code_executor=executor,
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="explorer")
    await runner.session_service.create_session(
        app_name="explorer", user_id="analyst1", session_id="session1"
    )

    conversation = [
        "Load this dataset: data = {'month': ['Jan','Feb','Mar','Apr','May'], "
        "'sales': [12000, 15000, 11000, 18000, 22000]}. Convert to a pandas DataFrame.",
        "Calculate the month-over-month growth rate for each month.",
        "Which month had the highest growth rate? Plot sales as a line chart.",
    ]

    for turn in conversation:
        print(f"\n>>> {turn}")
        events = await runner.run_debug(
            turn, user_id="analyst1", session_id="session1"
        )
        if events and events[-1].content:
            print(events[-1].content.parts[0].text)

asyncio.run(main())
```

---

*These recipes provide practical starting points for common ADK use cases. Adapt and extend them for your specific requirements.*

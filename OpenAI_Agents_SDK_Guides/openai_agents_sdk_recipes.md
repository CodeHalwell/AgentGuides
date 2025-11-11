# OpenAI Agents SDK: Practical Recipes and Use Cases

Production-ready implementations for common use cases and patterns with the OpenAI Agents SDK. Copy and adapt these recipes for your specific applications.

## Table of Contents

1. [Customer Service Agents](#customer-service-agents)
2. [Research and Knowledge Retrieval](#research-and-knowledge-retrieval)
3. [Financial Analysis](#financial-analysis)
4. [Code Generation and Review](#code-generation-and-review)
5. [Multi-Language Translation](#multi-language-translation)
6. [Content Moderation](#content-moderation)
7. [Personal Assistant](#personal-assistant)
8. [Team Collaboration](#team-collaboration)
9. [Data Analysis](#data-analysis)
10. [Enterprise Document Processing](#enterprise-document-processing)

---

## Customer Service Agents

### Airline Customer Support System

Complete airline support implementation:

```python
# airline_support.py
from agents import Agent, Runner, SQLiteSession, WebSearchTool, function_tool
from pydantic import BaseModel
import asyncio
from datetime import datetime

class FlightInfo(BaseModel):
    flight_number: str
    status: str
    departure: str
    arrival: str

@function_tool
def lookup_flight(flight_number: str) -> FlightInfo:
    """Look up flight information."""
    return FlightInfo(
        flight_number=flight_number,
        status="On time",
        departure="12:00 PM",
        arrival="3:30 PM"
    )

@function_tool
def process_refund(booking_reference: str, amount: float) -> dict:
    """Process ticket refund."""
    return {
        "status": "approved",
        "amount": amount,
        "reference": f"REF-{booking_reference}",
        "processed_date": datetime.now().isoformat()
    }

@function_tool
def rebook_flight(booking_reference: str, new_flight: str) -> dict:
    """Rebook passenger to different flight."""
    return {
        "status": "success",
        "booking_ref": booking_reference,
        "new_flight": new_flight,
        "confirmation": f"CONF-{new_flight}"
    }

# Specialised agents
refund_agent = Agent(
    name="Refund Specialist",
    handoff_description="Handles refunds and cancellations",
    instructions="Process refund requests efficiently. Verify booking and calculate refund amounts.",
    tools=[lookup_flight, process_refund]
)

rebooking_agent = Agent(
    name="Rebooking Specialist",
    handoff_description="Handles flight changes and rebooking",
    instructions="Help passengers rebook to alternative flights. Offer best alternatives.",
    tools=[lookup_flight, rebook_flight]
)

general_agent = Agent(
    name="Information Agent",
    handoff_description="Provides flight and booking information",
    instructions="Answer questions about flights and bookings. Escalate complex issues.",
    tools=[lookup_flight, WebSearchTool()]
)

# Triage agent
triage_agent = Agent(
    name="Airline Support Triage",
    instructions="""You are an airline customer support agent. 
    Understand the customer's need and route appropriately:
    - Refund/cancellation requests -> Refund Specialist
    - Rebooking/flight changes -> Rebooking Specialist
    - Information/status queries -> Information Agent
    Be empathetic and professional.""",
    handoffs=[refund_agent, rebooking_agent, general_agent]
)

async def handle_support_request(customer_id: str, query: str) -> str:
    """Process customer support request."""
    session = SQLiteSession(customer_id, "airline_support.db")
    
    result = await Runner.run(
        triage_agent,
        query,
        session=session,
        max_turns=5
    )
    
    return result.final_output

async def main():
    # Example support interactions
    customer_id = "CUST-123456"
    
    responses = await asyncio.gather(
        handle_support_request(customer_id, "I need to cancel my flight and get a refund"),
        handle_support_request(customer_id, "Can I change my flight to an earlier time?"),
        handle_support_request(customer_id, "What's the status of flight AA101?")
    )
    
    for response in responses:
        print(f"Response: {response}\n")

asyncio.run(main())
```

### E-commerce Support System

```python
# ecommerce_support.py
from agents import Agent, Runner, SQLiteSession, function_tool
from pydantic import BaseModel
import asyncio

class Order(BaseModel):
    order_id: str
    status: str
    items: list[str]
    total: float

@function_tool
def lookup_order(order_id: str) -> Order:
    """Look up order information."""
    return Order(
        order_id=order_id,
        status="Shipped",
        items=["Item 1", "Item 2"],
        total=99.99
    )

@function_tool
def initiate_return(order_id: str, item_name: str) -> dict:
    """Start return process for item."""
    return {
        "status": "approved",
        "return_label": f"RETURN-{order_id}",
        "instructions": "Use provided label to ship item back"
    }

@function_tool
def apply_discount(order_id: str, reason: str) -> dict:
    """Apply discount or credit to customer account."""
    return {
        "status": "applied",
        "credit_amount": 25.00,
        "order_id": order_id,
        "reason": reason
    }

returns_agent = Agent(
    name="Returns Specialist",
    handoff_description="Handles product returns and exchanges",
    instructions="Process return requests. Generate return labels and track returns.",
    tools=[lookup_order, initiate_return]
)

billing_agent = Agent(
    name="Billing Specialist",
    handoff_description="Handles billing issues and refunds",
    instructions="Resolve billing problems and apply credits when appropriate.",
    tools=[lookup_order, apply_discount]
)

general_agent = Agent(
    name="General Support",
    handoff_description="General product and order information",
    instructions="Answer general questions and provide order information.",
    tools=[lookup_order]
)

ecommerce_triage = Agent(
    name="E-commerce Support",
    instructions="""Route customer requests:
    - Returns/exchanges -> Returns Specialist
    - Billing/discount issues -> Billing Specialist
    - Order information -> General Support
    Maintain professional and helpful tone.""",
    handoffs=[returns_agent, billing_agent, general_agent]
)

async def process_support_query(customer_id: str, query: str) -> str:
    session = SQLiteSession(customer_id, "ecommerce.db")
    result = await Runner.run(ecommerce_triage, query, session=session)
    return result.final_output
```

---

## Research and Knowledge Retrieval

### Research Assistant with Web Search

```python
# research_assistant.py
from agents import Agent, Runner, SQLiteSession, WebSearchTool, FileSearchTool
from pydantic import BaseModel
import asyncio

class ResearchFindings(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    recommendations: list[str]

researcher = Agent(
    name="Research Assistant",
    instructions="""Conduct thorough research on topics.
    - Use web search for current information
    - Consult document database for internal resources
    - Synthesise findings into clear summaries
    - Provide source citations
    - Offer actionable recommendations""",
    tools=[
        WebSearchTool(),
        FileSearchTool(max_num_results=5)
    ],
    output_type=ResearchFindings
)

async def research_topic(user_id: str, topic: str) -> ResearchFindings:
    """Conduct research on specified topic."""
    session = SQLiteSession(user_id, "research.db")
    
    result = await Runner.run(
        researcher,
        f"Research the following topic: {topic}",
        session=session
    )
    
    return result.final_output_as(ResearchFindings)

async def main():
    findings = await research_topic(
        "user_1",
        "Latest developments in renewable energy"
    )
    
    print(f"Topic: {findings.topic}")
    print(f"Summary: {findings.summary[:200]}")
    print(f"Sources: {findings.sources}")
    print(f"Recommendations: {findings.recommendations}")

asyncio.run(main())
```

### Knowledge Base Assistant

```python
# knowledge_base.py
from agents import Agent, Runner, FileSearchTool, function_tool
import asyncio

@function_tool
def get_documentation_section(section: str) -> str:
    """Get documentation for section."""
    docs = {
        "installation": "Installation instructions...",
        "configuration": "Configuration guide...",
        "troubleshooting": "Troubleshooting tips..."
    }
    return docs.get(section, "Section not found")

knowledge_agent = Agent(
    name="Knowledge Assistant",
    instructions="""Help users find information in knowledge base.
    - Search documentation
    - Retrieve relevant sections
    - Provide step-by-step guidance
    - Escalate complex issues to specialists""",
    tools=[
        FileSearchTool(vector_store_ids=["vs_knowledge_base"]),
        get_documentation_section
    ]
)

async def query_knowledge_base(query: str) -> str:
    result = await Runner.run(knowledge_agent, query)
    return result.final_output
```

---

## Financial Analysis

### Stock Analysis System

```python
# stock_analyzer.py
from agents import Agent, Runner, WebSearchTool, function_tool
from pydantic import BaseModel
from datetime import datetime
import asyncio

class StockAnalysis(BaseModel):
    ticker: str
    current_price: float
    analysis: str
    recommendation: str
    risks: list[str]

@function_tool
def get_stock_data(ticker: str) -> dict:
    """Fetch current stock data."""
    return {
        "ticker": ticker,
        "price": 150.25,
        "change": 2.5,
        "market_cap": "2.5T",
        "pe_ratio": 25.3
    }

@function_tool
def fetch_financial_statements(ticker: str) -> dict:
    """Retrieve financial statements."""
    return {
        "revenue": "365B",
        "net_income": "99B",
        "cash_flow": "110B",
        "debt": "120B"
    }

collector_agent = Agent(
    name="Data Collector",
    instructions="Collect financial data and market information",
    tools=[get_stock_data, fetch_financial_statements, WebSearchTool()]
)

analyser_agent = Agent(
    name="Financial Analyser",
    instructions="""Analyse collected financial data.
    - Evaluate financial metrics
    - Assess market position
    - Identify risks and opportunities
    - Provide structured analysis""",
    tools=[get_stock_data]
)

recommendation_agent = Agent(
    name="Investment Advisor",
    instructions="Generate investment recommendations based on analysis",
    output_type=StockAnalysis
)

portfolio_manager = Agent(
    name="Portfolio Manager",
    instructions="""Coordinate stock analysis workflow:
    1. Use Data Collector to gather information
    2. Use Financial Analyser for detailed analysis
    3. Use Investment Advisor for recommendations
    Synthesise into actionable investment decisions.""",
    handoffs=[collector_agent, analyser_agent, recommendation_agent]
)

async def analyse_stock(ticker: str) -> StockAnalysis:
    result = await Runner.run(
        portfolio_manager,
        f"Analyse stock {ticker} and provide recommendation"
    )
    
    return result.final_output_as(StockAnalysis)

async def main():
    analysis = await analyse_stock("AAPL")
    print(f"Ticker: {analysis.ticker}")
    print(f"Current Price: ${analysis.current_price}")
    print(f"Recommendation: {analysis.recommendation}")
    print(f"Risks: {', '.join(analysis.risks)}")

asyncio.run(main())
```

---

## Code Generation and Review

### Code Generation Assistant

```python
# code_generator.py
from agents import Agent, Runner, function_tool
from pydantic import BaseModel
import asyncio

class CodeImplementation(BaseModel):
    language: str
    code: str
    explanation: str
    tests: str

@function_tool
def validate_code_syntax(code: str, language: str) -> dict:
    """Check code syntax validity."""
    return {
        "valid": True,
        "errors": [],
        "warnings": []
    }

@function_tool
def generate_unit_tests(code: str, language: str) -> str:
    """Generate unit tests for code."""
    return "test_code = ..."

developer_agent = Agent(
    name="Code Developer",
    instructions="Write clean, efficient, well-documented code",
    tools=[validate_code_syntax]
)

reviewer_agent = Agent(
    name="Code Reviewer",
    instructions="Review code quality and suggest improvements"
)

tester_agent = Agent(
    name="Test Generator",
    instructions="Generate comprehensive unit tests",
    tools=[generate_unit_tests]
)

code_manager = Agent(
    name="Code Manager",
    instructions="""Manage code generation workflow:
    1. Use Code Developer to write implementation
    2. Use Code Reviewer for quality review
    3. Use Test Generator for test creation
    Produce complete, production-ready code.""",
    handoffs=[developer_agent, reviewer_agent, tester_agent],
    output_type=CodeImplementation
)

async def generate_code(
    description: str,
    language: str = "Python"
) -> CodeImplementation:
    result = await Runner.run(
        code_manager,
        f"Generate {language} code for: {description}"
    )
    
    return result.final_output_as(CodeImplementation)

async def main():
    code = await generate_code(
        "Sort a list using merge sort algorithm",
        "Python"
    )
    
    print(f"Language: {code.language}")
    print(f"Code:\n{code.code}")
    print(f"Tests:\n{code.tests}")

asyncio.run(main())
```

### Code Review System

```python
# code_reviewer.py
from agents import Agent, Runner, function_tool
from pydantic import BaseModel
import asyncio

class CodeReview(BaseModel):
    issues: list[str]
    suggestions: list[str]
    security_concerns: list[str]
    performance_tips: list[str]
    overall_score: int

@function_tool
def check_code_patterns(code: str) -> dict:
    """Check for common patterns and anti-patterns."""
    return {
        "patterns_found": ["Factory", "Observer"],
        "anti_patterns": []
    }

@function_tool
def scan_security_issues(code: str) -> list[str]:
    """Scan for security vulnerabilities."""
    return ["SQL injection risk in query", "No input validation"]

@function_tool
def analyse_performance(code: str) -> dict:
    """Analyse performance characteristics."""
    return {
        "time_complexity": "O(n²)",
        "space_complexity": "O(1)",
        "optimisation_opportunities": ["Loop unrolling", "Caching"]
    }

reviewer = Agent(
    name="Code Reviewer",
    instructions="""Review code thoroughly.
    - Identify bugs and issues
    - Find security vulnerabilities
    - Check performance
    - Suggest improvements
    - Rate overall quality""",
    tools=[
        check_code_patterns,
        scan_security_issues,
        analyse_performance
    ],
    output_type=CodeReview
)

async def review_code(code_snippet: str) -> CodeReview:
    result = await Runner.run(
        reviewer,
        f"Review this code:\n{code_snippet}"
    )
    
    return result.final_output_as(CodeReview)
```

---

## Multi-Language Translation

### Translation Service

```python
# translation_service.py
from agents import Agent, Runner, SQLiteSession
from pydantic import BaseModel
import asyncio

class TranslationResult(BaseModel):
    original: str
    translated: str
    source_language: str
    target_language: str
    confidence: float

translators = {
    "English": Agent(
        name="English Translator",
        instructions="Translate text into English maintaining meaning and style"
    ),
    "Spanish": Agent(
        name="Spanish Translator",
        instructions="Translate text into Spanish maintaining meaning and style"
    ),
    "French": Agent(
        name="French Translator",
        instructions="Translate text into French maintaining meaning and style"
    ),
    "German": Agent(
        name="German Translator",
        instructions="Translate text into German maintaining meaning and style"
    ),
    "Chinese": Agent(
        name="Chinese Translator",
        instructions="Translate text into Chinese maintaining meaning and style"
    ),
    "Japanese": Agent(
        name="Japanese Translator",
        instructions="Translate text into Japanese maintaining meaning and style"
    ),
}

async def translate_text(
    text: str,
    source_language: str,
    target_language: str
) -> TranslationResult:
    """Translate text to target language."""
    
    if target_language not in translators:
        raise ValueError(f"Unsupported language: {target_language}")
    
    translator = translators[target_language]
    
    result = await Runner.run(
        translator,
        f"Translate from {source_language}: {text}"
    )
    
    return TranslationResult(
        original=text,
        translated=result.final_output,
        source_language=source_language,
        target_language=target_language,
        confidence=0.95
    )

async def main():
    # Translate to multiple languages
    text = "Hello, how are you today?"
    
    translations = await asyncio.gather(
        translate_text(text, "English", "Spanish"),
        translate_text(text, "English", "French"),
        translate_text(text, "English", "German"),
        translate_text(text, "English", "Japanese")
    )
    
    for trans in translations:
        print(f"{trans.target_language}: {trans.translated}")

asyncio.run(main())
```

---

## Content Moderation

### Content Safety System

```python
# content_moderation.py
from agents import Agent, Runner, input_guardrail, output_guardrail
from agents import GuardrailFunctionOutput, RunContextWrapper
from pydantic import BaseModel
import asyncio
import re

@input_guardrail
async def check_harmful_content(ctx: RunContextWrapper, agent: Agent, input_data):
    """Check for harmful or inappropriate input."""
    text = input_data if isinstance(input_data, str) else str(input_data)
    
    harmful_patterns = [
        r"hate speech",
        r"harassment",
        r"violence",
        r"discrimination"
    ]
    
    for pattern in harmful_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return GuardrailFunctionOutput(
                output_info={"threat": "harmful content"},
                tripwire_triggered=True
            )
    
    return GuardrailFunctionOutput(
        output_info={"threat": "none"},
        tripwire_triggered=False
    )

@output_guardrail
async def check_response_safety(ctx: RunContextWrapper, agent: Agent, output):
    """Verify response is safe and appropriate."""
    text = str(output)
    
    if any(word in text.lower() for word in ["violence", "harm", "illegal"]):
        return GuardrailFunctionOutput(
            output_info={"safe": False},
            tripwire_triggered=True
        )
    
    return GuardrailFunctionOutput(
        output_info={"safe": True},
        tripwire_triggered=False
    )

safe_agent = Agent(
    name="Safe Assistant",
    instructions="Respond helpfully and safely to all queries",
    input_guardrails=[check_harmful_content],
    output_guardrails=[check_response_safety]
)

async def moderate_interaction(user_input: str) -> str:
    try:
        result = await Runner.run(safe_agent, user_input)
        return result.final_output
    except Exception as e:
        return f"Content blocked: {str(e)}"
```

---

## Personal Assistant

### Daily Assistant

```python
# personal_assistant.py
from agents import Agent, Runner, SQLiteSession, function_tool
from pydantic import BaseModel
from datetime import datetime
import asyncio

class Task(BaseModel):
    title: str
    description: str
    priority: str
    due_date: str

@function_tool
def get_calendar_events() -> list[dict]:
    """Get today's calendar events."""
    return [
        {"time": "10:00 AM", "event": "Team meeting"},
        {"time": "2:00 PM", "event": "Project review"}
    ]

@function_tool
def add_to_todo(task_title: str, priority: str = "medium") -> dict:
    """Add item to to-do list."""
    return {
        "status": "added",
        "task": task_title,
        "priority": priority
    }

@function_tool
def get_weather() -> dict:
    """Get weather forecast."""
    return {
        "temperature": 72,
        "condition": "Sunny",
        "forecast": "Clear skies all day"
    }

assistant = Agent(
    name="Personal Assistant",
    instructions="""Help manage daily tasks and information.
    - Check calendar and schedule
    - Manage to-do lists
    - Provide weather and news
    - Set reminders and notifications
    Be proactive and helpful.""",
    tools=[
        get_calendar_events,
        add_to_todo,
        get_weather
    ]
)

async def assist_user(user_id: str, request: str) -> str:
    session = SQLiteSession(user_id, "assistant.db")
    result = await Runner.run(assistant, request, session=session)
    return result.final_output

async def main():
    user_id = "user_1"
    
    responses = await asyncio.gather(
        assist_user(user_id, "What's on my calendar today?"),
        assist_user(user_id, "Add 'Review reports' to my to-do"),
        assist_user(user_id, "What's the weather like?")
    )
    
    for response in responses:
        print(f"Assistant: {response}\n")

asyncio.run(main())
```

---

## Team Collaboration

### Meeting Coordinator

```python
# meeting_coordinator.py
from agents import Agent, Runner, function_tool
from pydantic import BaseModel
import asyncio

class MeetingSchedule(BaseModel):
    title: str
    participants: list[str]
    time: str
    duration: int
    agenda: list[str]

@function_tool
def check_availability(participant: str, time_slot: str) -> bool:
    """Check participant availability."""
    return True  # Simulate availability check

@function_tool
def send_meeting_invite(
    participants: list[str],
    time: str,
    meeting_title: str
) -> dict:
    """Send meeting invitation."""
    return {
        "status": "sent",
        "participants": participants,
        "meeting_link": "https://meet.example.com/abc123"
    }

@function_tool
def book_meeting_room(
    time: str,
    duration: int,
    capacity: int
) -> dict:
    """Book conference room."""
    return {
        "status": "booked",
        "room": "Conference Room A",
        "time": time,
        "duration": duration
    }

coordinator = Agent(
    name="Meeting Coordinator",
    instructions="""Coordinate meetings efficiently.
    - Check participant availability
    - Find best meeting time
    - Book appropriate meeting rooms
    - Send invitations
    - Create agendas""",
    tools=[
        check_availability,
        send_meeting_invite,
        book_meeting_room
    ],
    output_type=MeetingSchedule
)

async def schedule_meeting(
    title: str,
    participants: list[str],
    duration: int = 60
) -> MeetingSchedule:
    result = await Runner.run(
        coordinator,
        f"Schedule meeting '{title}' with {', '.join(participants)}"
    )
    
    return result.final_output_as(MeetingSchedule)
```

---

## Data Analysis

### Analytics Pipeline

```python
# data_analyzer.py
from agents import Agent, Runner, function_tool
from pydantic import BaseModel
import asyncio
import json

class AnalysisReport(BaseModel):
    dataset_name: str
    row_count: int
    key_insights: list[str]
    visualisations: list[str]
    recommendations: list[str]

@function_tool
def load_dataset(dataset_name: str) -> dict:
    """Load dataset for analysis."""
    return {
        "rows": 10000,
        "columns": ["id", "date", "value", "category"],
        "sample": [[1, "2024-01-01", 100, "A"]]
    }

@function_tool
def calculate_statistics(data: list) -> dict:
    """Calculate statistical measures."""
    return {
        "mean": 150.5,
        "median": 145,
        "std_dev": 25.3,
        "min": 50,
        "max": 300
    }

@function_tool
def detect_trends(data: list) -> list[str]:
    """Detect trends in data."""
    return [
        "Upward trend over time",
        "Seasonal pattern detected",
        "Anomaly on 2024-03-15"
    ]

analyser = Agent(
    name="Data Analyst",
    instructions="""Analyse datasets comprehensively.
    - Load and explore data
    - Calculate statistics
    - Detect trends and patterns
    - Identify anomalies
    - Generate visualisations
    - Provide actionable insights""",
    tools=[
        load_dataset,
        calculate_statistics,
        detect_trends
    ],
    output_type=AnalysisReport
)

async def analyse_dataset(dataset_name: str) -> AnalysisReport:
    result = await Runner.run(
        analyser,
        f"Analyse dataset: {dataset_name}"
    )
    
    return result.final_output_as(AnalysisReport)

async def main():
    report = await analyse_dataset("sales_2024")
    
    print(f"Dataset: {report.dataset_name}")
    print(f"Rows: {report.row_count}")
    print(f"Insights: {report.key_insights}")
    print(f"Recommendations: {report.recommendations}")

asyncio.run(main())
```

---

## Enterprise Document Processing

### Contract Analysis System

```python
# contract_analyzer.py
from agents import Agent, Runner, FileSearchTool, function_tool
from pydantic import BaseModel
import asyncio

class ContractAnalysis(BaseModel):
    document_name: str
    summary: str
    key_terms: dict
    risks: list[str]
    recommendations: list[str]
    compliance_status: str

@function_tool
def extract_clauses(document_id: str) -> dict:
    """Extract key clauses from contract."""
    return {
        "payment_terms": "Net 30",
        "termination": "90 days notice",
        "liability": "Limited to contract value",
        "confidentiality": "3 years post-termination"
    }

@function_tool
def check_legal_compliance(
    document_id: str,
    jurisdiction: str
) -> dict:
    """Check legal compliance."""
    return {
        "compliant": True,
        "issues": [],
        "jurisdiction": jurisdiction
    }

contract_analyser = Agent(
    name="Contract Analyst",
    instructions="""Analyse contracts thoroughly.
    - Extract key terms and conditions
    - Identify risks and liabilities
    - Check legal compliance
    - Highlight unusual provisions
    - Provide recommendations""",
    tools=[
        extract_clauses,
        check_legal_compliance,
        FileSearchTool(vector_store_ids=["vs_legal_docs"])
    ],
    output_type=ContractAnalysis
)

async def analyse_contract(document_id: str) -> ContractAnalysis:
    result = await Runner.run(
        contract_analyser,
        f"Analyse contract: {document_id}"
    )
    
    return result.final_output_as(ContractAnalysis)

async def main():
    analysis = await analyse_contract("contract_2024_001")
    
    print(f"Document: {analysis.document_name}")
    print(f"Summary: {analysis.summary}")
    print(f"Key Terms: {analysis.key_terms}")
    print(f"Risks: {analysis.risks}")
    print(f"Compliance: {analysis.compliance_status}")

asyncio.run(main())
```

These recipes provide production-ready implementations you can adapt for your specific use cases with the OpenAI Agents SDK.

